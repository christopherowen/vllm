# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED's bounded replay on the TileLang kernels: however a prompt is chunked, the
decoder stack's rows from the replay floor on come out bit-identical.

A stack of sliding-window decoder layers runs each prefill step through the model
state's CED staging (the requests and row-map kernels), the decoder's token metadata,
the window metadata and the sparse MLA. A layer writes its input as the KV records of
the step's kept rows into a cache that starts as NaN, and the next layer's input is
its input plus its output, so a row's output depends on every row it saw on the way
up and an unwritten read shows as NaN.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
pytest.importorskip("tile_kernels")

from tests.kernels.attention.test_deepseek_v41_tilelang import (  # noqa: E402
    _pack_swa,
    _paged,
)
from tests.kernels.deepseek_v41_contracts import rope_table  # noqa: E402
from vllm.models.deepseek_v4_1.ced import CED_WINDOW, CEDState  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import ced as ced_kernels  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (  # noqa: E402
    HEAD_DIM,
    SparseMlaV41Plan,
)

DEVICE = "cuda"
PAGE = 64
HEADS = 16
LAYERS = 3
RECORD = 528
RECOVERY_ROWS = LAYERS * (CED_WINDOW - 1) + 1


@dataclass
class DecoderTrace:
    outputs: dict[int, torch.Tensor]
    # Actual layer inputs and stored bytes, in logical order [end-128, end).
    kv_inputs: tuple[torch.Tensor, ...]
    kv_records: tuple[torch.Tensor, ...]

    @property
    def deepest_aux(self):
        # V4.1's auxiliary states are layer inputs, not layer outputs.
        return self.kv_inputs[-1]


def aligned_chunks(threshold, start, end):
    """The scheduler's aligned prefill chunks ``(start, rows)`` from ``start``."""
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=threshold)
    )
    chunks = []
    while start < end:
        rows = Scheduler._aligned_prefill_chunk(
            scheduler, start, min(end - start, threshold)
        )
        chunks.append((start, rows))
        start += rows
    return chunks


def model_state_stager(prefix, *, original_prompt_end=None, prefill_end=None):
    """Stages each chunk as serving does: the model state's ``_stage_ced``."""
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState

    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state = CEDState(4, 8192, (128, 256, 384, 512, 8192), DEVICE)
    state._ced_prompt_logprobs, state._ced_prefix_start = set(), {"prompt": prefix}
    state._ced_recovery_prompt_end = {}
    state._ced_streaming_prefix_start = None
    if original_prompt_end is not None:
        # Exercise the real request admission hook: a resumed request keeps the
        # original prompt length while its prefill ids include accepted outputs.
        from vllm.v1.core.sched.output import NewRequestData

        state.rope_state = state.prompt_embeds_state = None
        state.steering = state.steered_requests = None
        state.disk_engram_models, state._prefill_tokens = (), {}
        state.add_request(
            0,
            NewRequestData(
                req_id="prompt",
                prompt_token_ids=list(range(original_prompt_end)),
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=(),
                num_computed_tokens=prefix,
                lora_request=None,
                prefill_token_ids=list(range(prefill_end)),
            ),
        )

    def stage(start, rows, prompt_end, starts, seq):
        state._stage_ced(
            SimpleNamespace(
                num_reqs=1,
                req_ids=["prompt"],
                query_start_loc=starts,
                seq_lens=seq,
                num_scheduled_tokens=np.array([rows]),
                num_computed_prefill_tokens_np=np.array(
                    [min(start, prompt_end if prefill_end is None else prefill_end)]
                ),
                prefill_len_np=np.array(
                    [prompt_end if prefill_end is None else prefill_end]
                ),
            )
        )
        return state.ced_state

    return stage


def finite_cone_stager(prefix, original_prompt_end, prefill_end, horizon):
    """Test-only recovery policy, using the real CED staging kernels.

    Preserve the original prompt floor when it is later than the finite cone.
    This deliberately does not change production policy or fake prompt lengths
    passed to the real request-admission hook.
    """
    floor = max(0, original_prompt_end - CED_WINDOW, prefill_end - horizon)
    assert prefix <= floor
    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), DEVICE)

    def stage(start, rows, prompt_end, starts, seq):
        assert prompt_end == prefill_end
        # As with original-floor recovery, a chunk before the floor keeps one
        # discarded row; all rows at or after the floor are materialized.
        state.stage(
            starts, seq, [rows], [False], [floor], [max(start + rows - floor, 1)]
        )
        return state

    return stage


def decoder_stack(
    prompt_end,
    chunks,
    stage,
    seed=5,
    encoder=None,
    *,
    capture_state=False,
    query_mode="self",
):
    """The stack's last-layer rows for the kept positions of every step, in
    ``{position: row}``, the last step's kept row winning.

    With ``capture_state``, also expose the last 128 KV inputs and actual packed
    cache bytes of every layer. These are the live state needed by subsequent
    target/draft steps; the last 128 *final outputs* have a larger receptive field.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    # The encoder's output: one fixed row a position, whatever the chunking.
    if encoder is None:
        encoder = torch.randn((prompt_end, HEAD_DIM), generator=gen, device=DEVICE)
    assert encoder.shape == (prompt_end, HEAD_DIM)
    scales = torch.linspace(0.5, 1.5, HEADS, device=DEVICE)
    sinks = torch.linspace(-1.0, 1.0, HEADS, device=DEVICE)
    pages = prompt_end // PAGE + 2
    # Logical page j is physical page j + 1 (page 0 is the null page).
    table = (torch.arange(pages, dtype=torch.int32, device=DEVICE) + 1)[None]
    records = [
        torch.full(((pages + 1) * PAGE, RECORD), 0xFF, dtype=torch.uint8, device=DEVICE)
        for _ in range(LAYERS)
    ]
    tail_start = prompt_end - CED_WINDOW
    inputs = (
        [
            torch.full((CED_WINDOW, HEAD_DIM), float("nan"), device=DEVICE)
            for _ in range(LAYERS)
        ]
        if capture_state
        else None
    )
    assert query_mode in ("self", "edge_marker")
    rope = rope_table(prompt_end + 1)
    plan = SparseMlaV41Plan(
        HEADS, CED_WINDOW, 0, 128, swa_page_size=PAGE,
        max_rows=512, decode_max_rows=256, split_max_rows=512,
    )  # fmt: skip
    outputs = {}
    for start, rows in chunks:
        starts = torch.tensor([0, rows], dtype=torch.int32, device=DEVICE)
        seq = torch.tensor([start + rows], dtype=torch.int32, device=DEVICE)
        state = stage(start, rows, prompt_end, starts, seq)
        step = torch.arange(start, start + rows, device=DEVICE)
        indices = state.get_indices()
        if indices is None:
            positions = step
        else:
            positions = ced_kernels.token_metadata(
                indices,
                step,
                torch.zeros(rows, dtype=torch.int32, device=DEVICE),
                step + PAGE,
                (step + 1).int(),
            )[0]
        n = positions.numel()
        swa = torch.empty((n, CED_WINDOW), dtype=torch.int32, device=DEVICE)
        lengths = torch.empty(n, dtype=torch.int32, device=DEVICE)
        top = torch.empty_like(lengths)
        tl_metadata.chunk(
            positions, torch.zeros(n, dtype=torch.int32, device=DEVICE), table, swa,
            lengths, top, (positions + 1).int(), state.starts, state.request_positions,
            0, n, page=PAGE, window=CED_WINDOW, draft=False,
            replay_start=state.replay_start,
        )  # fmt: skip
        x = encoder[positions]
        for layer in range(LAYERS):
            if inputs is not None:
                tail = positions >= tail_start
                inputs[layer][positions[tail] - tail_start] = x[tail]
            records[layer][positions + PAGE] = _pack_swa(x)
            if query_mode == "edge_marker":
                # A marker in non-RoPE coordinate 0 dominates the softmax. It
                # survives each outermost 127-position hop without the tiny edge
                # weight of uniform attention disappearing in KV quantization.
                q = torch.zeros(
                    (n, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE
                )
                q[:, :, 0] = 4096
            else:
                q = (x[:, None, :] * scales[None, :, None]).bfloat16()
            out = torch.empty((n, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
            plan.run(
                q, _paged(records[layer], PAGE), swa, lengths, sinks, positions, rope,
                out,
            )  # fmt: skip
            # Heads added in a fixed order: elementwise, the same for any row count.
            for head in range(HEADS):
                x = x + out[:, head].float() / HEADS
        for position, row in zip(positions.tolist(), x):
            outputs[position] = row
    if inputs is not None:
        return DecoderTrace(
            outputs,
            tuple(inputs),
            tuple(
                record[tail_start + PAGE : prompt_end + PAGE].clone()
                for record in records
            ),
        )
    return outputs


def replayed(outputs, floor, prompt_end):
    return torch.stack([outputs[p] for p in range(floor, prompt_end)])


@pytest.mark.parametrize("final_rows", [1, 64, 128, 129])
def test_chunkings_give_the_same_decoder_rows(final_rows):
    """A 1,024-row prompt start plus ``final_rows``: one chunk, 512-row chunks,
    384-row chunks and, after a prefix hit at 448, 512-row chunks again. Every
    chunking's rows from the floor (the prompt end less the window) on are the single
    chunk's, bit for bit, and finite."""
    prompt_end = 1024 + final_rows
    floor = prompt_end - CED_WINDOW
    whole = decoder_stack(prompt_end, [(0, prompt_end)], model_state_stager(0))
    reference = replayed(whole, floor, prompt_end)
    assert torch.isfinite(reference).all()
    for threshold, prefix in ((512, 0), (384, 0), (512, 448)):
        chunks = aligned_chunks(threshold, prefix, prompt_end)
        rows = replayed(
            decoder_stack(prompt_end, chunks, model_state_stager(prefix)),
            floor,
            prompt_end,
        )
        assert torch.equal(rows, reference), (threshold, prefix, chunks)


@pytest.mark.parametrize("generated_rows", [129, 257])
@pytest.mark.parametrize("resumed_chunk", [64, 512])
def test_preempted_decode_rebuilds_original_decoder_history(
    generated_rows, resumed_chunk
):
    """Losing private windows must not change an in-flight continuation.

    The first run completes one prompt and then computes accepted generated rows
    one at a time. The second run has lost every private decoder page; its worker
    input contains the same original prompt plus the accepted output suffix, as
    MRV2 sends after preemption. All encoder rows are identical and untouched.
    Comparing the final row traverses three real TileLang SWA layers, so dropping
    the early generated rows cannot hide behind one layer's 128-token horizon.
    """
    original_end = 1024
    total = original_end + generated_rows
    # Values are deterministic dyadic fractions; no random sampler or model
    # weights participate. Both executions receive the exact same encoder rows.
    positions = torch.arange(total, device=DEVICE, dtype=torch.float32)[:, None]
    columns = torch.arange(HEAD_DIM, device=DEVICE, dtype=torch.float32)[None, :]
    encoder = ((positions % 17) - 8) / 512 + ((columns % 11) - 5) / 1024
    uninterrupted = decoder_stack(
        total,
        [(0, original_end), *((p, 1) for p in range(original_end, total))],
        model_state_stager(
            0, original_prompt_end=original_end, prefill_end=original_end
        ),
        encoder=encoder,
    )
    expected = uninterrupted[total - 1]
    assert torch.isfinite(expected).all()

    # Last legal 256-token encoder hit before the ORIGINAL prompt replay floor.
    # This deliberately provides enough encoder history to reconstruct the exact
    # continuation; changing only the cache-hit cap cannot make this test pass.
    hit = (original_end - CED_WINDOW) // 256 * 256
    resumed = decoder_stack(
        total,
        aligned_chunks(resumed_chunk, hit, total),
        model_state_stager(hit, original_prompt_end=original_end, prefill_end=total),
        encoder=encoder,
    )
    actual = resumed[total - 1]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.fixture(scope="module")
def finite_cone_reference():
    """Reuse uninterrupted runs across chunk sizes and negative horizons."""
    cases = {}

    def reference(generated_rows, query_mode):
        key = generated_rows, query_mode
        if key not in cases:
            original_end = 1024
            total = original_end + generated_rows
            if query_mode == "edge_marker":
                encoder = torch.zeros((total, HEAD_DIM), device=DEVICE)
                # The first row in the sufficient cone, omitted by a horizon
                # one row shorter. Three extreme 127-position hops reach L-1.
                encoder[total - RECOVERY_ROWS, 0] = 0.5
            else:
                positions = torch.arange(total, device=DEVICE)[:, None]
                columns = torch.arange(HEAD_DIM, device=DEVICE)[None, :]
                encoder = ((positions % 17) - 8) / 512 + ((columns % 11) - 5) / 1024
            uninterrupted = decoder_stack(
                total,
                [(0, original_end), *((p, 1) for p in range(original_end, total))],
                model_state_stager(
                    0, original_prompt_end=original_end, prefill_end=original_end
                ),
                encoder=encoder,
                capture_state=True,
                query_mode=query_mode,
            )
            cases[key] = original_end, total, encoder, uninterrupted
        return cases[key]

    return reference


def assert_same_final_state(actual, expected, total):
    for trace in (actual, expected):
        assert torch.isfinite(trace.outputs[total - 1]).all()
        assert all(torch.isfinite(x).all() for x in trace.kv_inputs)
        assert torch.isfinite(trace.deepest_aux).all()
    torch.testing.assert_close(
        actual.outputs[total - 1], expected.outputs[total - 1], atol=0, rtol=0
    )
    for actual_input, expected_input, actual_record, expected_record in zip(
        actual.kv_inputs, expected.kv_inputs, actual.kv_records, expected.kv_records
    ):
        torch.testing.assert_close(actual_input, expected_input, atol=0, rtol=0)
        assert torch.equal(actual_record, expected_record)
    torch.testing.assert_close(actual.deepest_aux, expected.deepest_aux, atol=0, rtol=0)


@pytest.mark.parametrize("generated_rows", [257, 513, 1025])
@pytest.mark.parametrize("resumed_chunk", [64, 512])
@pytest.mark.parametrize("query_mode", ["self", "edge_marker"])
def test_finite_cone_reconstructs_final_live_state(
    generated_rows, resumed_chunk, query_mode, finite_cone_reference
):
    """382 rows cover the final output and all live KV/aux inputs for three layers.

    An output at t depends on at most 3*127 earlier positions. The deepest KV
    input depends on 2*127, and the drafter needs 128 such context rows. Both
    bounds are 382. This does NOT claim all 128 final-layer outputs are exact;
    that stronger, unnecessary requirement would need 3*127+128 = 509 rows.
    """
    assert RECOVERY_ROWS == 382
    original_end, total, encoder, expected = finite_cone_reference(
        generated_rows, query_mode
    )
    original_hit = (original_end - CED_WINDOW) // 256 * 256
    full_recovery = decoder_stack(
        total,
        aligned_chunks(resumed_chunk, original_hit, total),
        model_state_stager(
            original_hit, original_prompt_end=original_end, prefill_end=total
        ),
        encoder=encoder,
        capture_state=True,
        query_mode=query_mode,
    )
    assert_same_final_state(full_recovery, expected, total)

    floor = max(original_end - CED_WINDOW, total - RECOVERY_ROWS)
    hit = floor // 256 * 256
    bounded = decoder_stack(
        total,
        aligned_chunks(resumed_chunk, hit, total),
        finite_cone_stager(hit, original_end, total, RECOVERY_ROWS),
        encoder=encoder,
        capture_state=True,
        query_mode=query_mode,
    )
    assert_same_final_state(bounded, full_recovery, total)


@pytest.mark.parametrize("generated_rows", [257, 513, 1025])
@pytest.mark.parametrize("resumed_chunk", [64, 512])
@pytest.mark.parametrize("horizon", [381, 128])
def test_shorter_recovery_loses_outermost_dependency(
    generated_rows, resumed_chunk, horizon, finite_cone_reference
):
    """The inclusive endpoint matters: 381 rows can lose observable live state."""
    original_end, total, encoder, expected = finite_cone_reference(
        generated_rows, "edge_marker"
    )
    floor = max(original_end - CED_WINDOW, total - horizon)
    hit = floor // 256 * 256
    actual = decoder_stack(
        total,
        aligned_chunks(resumed_chunk, hit, total),
        finite_cone_stager(hit, original_end, total, horizon),
        encoder=encoder,
        capture_state=True,
        query_mode="edge_marker",
    )
    assert torch.isfinite(expected.outputs[total - 1]).all()
    assert torch.isfinite(actual.outputs[total - 1]).all()
    assert all(torch.isfinite(x).all() for x in actual.kv_inputs)
    # The original run carries a strong marker, while truncated recovery loses
    # it. Require a finite signal, not merely a mismatch caused by unwritten NaNs.
    assert expected.outputs[total - 1][0] > 0.25
    assert actual.outputs[total - 1][0] == 0
    assert not torch.equal(actual.kv_inputs[-1], expected.kv_inputs[-1])
    assert not torch.equal(actual.kv_records[-1], expected.kv_records[-1])
    assert not torch.equal(actual.deepest_aux, expected.deepest_aux)
