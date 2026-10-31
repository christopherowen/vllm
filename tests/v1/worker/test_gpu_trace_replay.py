# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Model Runner V2 inference trace-replay.

Trace-replay forces the sampler to emit a predetermined sequence of decode
token IDs. The replay step for a request is derived from GPU state as
``total_len - prompt_len`` (output tokens produced so far), and the trace token
overwrites the sampled token in place before logprobs are computed.
"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for trace-replay tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.trace_replay import (
    TraceReplayState,
    apply_trace_tokens,
)
from vllm.v1.worker.gpu.states import RequestState

DEVICE = "cuda"
TEST_MAX_MODEL_LEN = 2048


def _i32(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32, device=DEVICE)


def _i64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device=DEVICE)


def _trace_state(max_num_reqs: int) -> TraceReplayState:
    """Build a state whose req_states exposes the buffers apply_trace reads.

    total_len/prompt_len are written by the caller to position each request at
    the desired replay step.
    """
    req_states = cast(
        RequestState,
        SimpleNamespace(
            max_num_reqs=max_num_reqs,
            max_model_len=TEST_MAX_MODEL_LEN,
            device=torch.device(DEVICE),
            total_len=SimpleNamespace(
                gpu=torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
            ),
            prompt_len=SimpleNamespace(
                gpu=torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
            ),
        ),
    )
    return TraceReplayState(req_states)


def _set_lens(state: TraceReplayState, total_len, prompt_len) -> None:
    """Position each request at a replay step via total_len - prompt_len."""
    state.req_states.total_len.gpu[: len(total_len)] = _i32(total_len)
    state.req_states.prompt_len.gpu[: len(prompt_len)] = _i32(prompt_len)


# ------------------------------ Kernel ------------------------------------


def test_replay_overwrites_sampled_at_each_step():
    """The trace token for the current step replaces the sampled token."""
    trace = [[100, 101, 102], [200, 201, 202]]
    trace_len = _i32([3, 3])
    trace_token_ids = torch.zeros(
        2, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    for i, t in enumerate(trace):
        trace_token_ids[i, : len(t)] = _i32(t)
    prompt_len = _i32([5, 8])
    idx_mapping = _i32([0, 1])

    for step in range(3):
        sampled = _i64([-7, -7])  # sentinel that must be overwritten
        total_len = _i32([5 + step, 8 + step])
        apply_trace_tokens(
            sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
        )
        assert sampled.tolist() == [trace[0][step], trace[1][step]]


def test_past_end_of_trace_leaves_sampled_untouched():
    """Once step >= trace_len, the sampler's own token is kept."""
    trace_token_ids = torch.zeros(
        1, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    trace_token_ids[0, :2] = _i32([100, 101])
    trace_len = _i32([2])
    prompt_len = _i32([4])
    idx_mapping = _i32([0])

    sampled = _i64([999])
    total_len = _i32([4 + 2])  # step == 2 == trace_len -> out of range
    apply_trace_tokens(
        sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
    )
    assert sampled.tolist() == [999]


def test_non_trace_request_untouched():
    """trace_len == 0 means the request never uses replay."""
    trace_token_ids = torch.zeros(
        1, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    trace_len = _i32([0])
    prompt_len = _i32([3])
    idx_mapping = _i32([0])

    sampled = _i64([42])
    total_len = _i32([3])
    apply_trace_tokens(
        sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
    )
    assert sampled.tolist() == [42]


def test_idx_mapping_indirection_and_negative_skip():
    """batch_idx -> req_state_idx indirection, and negative entries are skipped."""
    # req_state 0: no trace; req_state 1: trace [500, 501].
    trace_token_ids = torch.zeros(
        2, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    trace_token_ids[1, :2] = _i32([500, 501])
    trace_len = _i32([0, 2])
    prompt_len = _i32([10, 6])

    # batch: [maps to state 1, masked (-1), maps to state 0].
    idx_mapping = _i32([1, -1, 0])
    sampled = _i64([1, 2, 3])
    # Indexed by req_state_idx: state 0 at total_len=10, state 1 at step 1.
    total_len = _i32([10, 6 + 1])
    apply_trace_tokens(
        sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
    )
    # batch 0 -> state 1 step 1 -> 501; batch 1 masked; batch 2 -> state 0 no trace.
    assert sampled.tolist() == [501, 2, 3]


# --------------------------- TraceReplayState -----------------------------


def test_state_end_to_end():
    """add_request -> apply_staged_writes -> apply_trace overwrites correctly."""
    state = _trace_state(4)
    state.add_request(0, SamplingParams(trace_decode_token_ids=[11, 22, 33]))
    state.add_request(1, SamplingParams())  # no trace
    state.apply_staged_writes()

    idx_mapping = _i32([0, 1])
    sampled = _i64([-1, -1])
    _set_lens(state, [7 + 1, 7, 0, 0], [7, 7, 0, 0])  # req 0 at step 1
    state.apply_trace(sampled, idx_mapping)
    assert sampled.tolist() == [22, -1]


def test_state_leaves_non_trace_batch_unchanged():
    """Requests with trace_len == 0 remain unchanged after a trace was seen."""
    state = _trace_state(4)
    state.add_request(0, SamplingParams(trace_decode_token_ids=[11, 22]))
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()

    # Only the non-trace request (state 1) is in this batch.
    idx_mapping = _i32([1])
    sampled = _i64([555])
    _set_lens(state, [0, 7, 0, 0], [7, 7, 0, 0])
    state.apply_trace(sampled, idx_mapping)
    assert sampled.tolist() == [555]


def test_slot_reuse_clears_trace():
    """Reusing a slot for a non-trace request must not replay stale tokens."""
    state = _trace_state(2)
    state.add_request(0, SamplingParams(trace_decode_token_ids=[11, 22]))
    state.apply_staged_writes()
    # Slot 0 reused by a request without a trace.
    state.add_request(0, SamplingParams())
    state.apply_staged_writes()

    idx_mapping = _i32([0])
    sampled = _i64([888])
    _set_lens(state, [3, 0], [3, 0])
    state.apply_trace(sampled, idx_mapping)
    assert sampled.tolist() == [888]


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_tilelang_trace_replay_and_compact_sampling_mask_match_triton(idx_dtype):
    """TileLang trace_replay and compact_sampling_mask match their Triton references."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling as tls
    from vllm.v1.worker.gpu.sample.output import (
        SamplingMaskTensors,
        _compact_sampling_mask_kernel,
    )
    from vllm.v1.worker.gpu.sample.trace_replay import _trace_replay_kernel

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    trace_token_ids = torch.randint(
        0, 50000, (8, 128), generator=gen, dtype=torch.int32, device=DEVICE
    )
    trace_len = torch.tensor([0, 4, 10, 2, 0, 8, 1, 5], dtype=torch.int32, device=DEVICE)
    prompt_len = torch.tensor([5, 3, 12, 8, 4, 2, 9, 1], dtype=torch.int32, device=DEVICE)
    total_len = torch.tensor([5, 5, 12, 10, 7, 9, 9, 4], dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.tensor([1, -1, 2, 3, 5, 6, 7, 0], dtype=idx_dtype, device=DEVICE)
    expected = torch.arange(100, 108, dtype=torch.int64, device=DEVICE)
    actual = expected.clone()
    _trace_replay_kernel[(idx_mapping.shape[0],)](
        expected,
        idx_mapping,
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_len,
        total_len,
        prompt_len,
    )
    apply_trace_tokens(
        actual, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
    )
    assert torch.equal(actual, expected)
    assert tls.trace_replay_calls(idx_dtype) == [
        (tls.trace_replay_kernel, tls.trace_replay_key(idx_dtype), {})
    ]

    vocab_size = 4099
    max_num_kept = 64
    logits = torch.full((4, vocab_size), -float("inf"), dtype=torch.float32, device=DEVICE)
    logits[0, [0, 7, 511, 512, 2048, 4098]] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=DEVICE
    )
    logits[1, :150] = torch.linspace(-1.0, 1.0, 150, device=DEVICE)
    logits[2, [3, 9, 15]] = 0.5
    num_sampled = torch.tensor([1, 1, 0, 1], dtype=torch.int32, device=DEVICE)

    ref_ids = torch.zeros((4, max_num_kept), dtype=torch.int32, device=DEVICE)
    ref_mask = torch.empty((4, (vocab_size + 7) // 8), dtype=torch.uint8, device=DEVICE)
    ref_counts = torch.empty(4, dtype=torch.int32, device=DEVICE)
    _compact_sampling_mask_kernel[(4,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        num_sampled,
        ref_ids,
        ref_ids.stride(0),
        ref_mask,
        ref_mask.stride(0),
        ref_counts,
        vocab_size,
        max_num_kept,
        BLOCK_SIZE=8192,
    )
    masks = SamplingMaskTensors.from_logits(logits, num_sampled, max_num_kept)
    assert torch.equal(masks.counts, ref_counts)
    assert torch.equal(masks.packed_mask, ref_mask)
    assert torch.equal(masks.token_ids[0, :6], ref_ids[0, :6])
    assert torch.equal(masks.token_ids[1], ref_ids[1])
    assert masks.tolists().to_nested_list() == SamplingMaskTensors(
        ref_ids, ref_mask, ref_counts, vocab_size
    ).tolists().to_nested_list()
