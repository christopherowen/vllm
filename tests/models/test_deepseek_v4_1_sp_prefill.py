# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill sequence parallelism: row layout, threshold and CPU collectives.

The collectives are simulated in one process: every rank's contribution is
computed first, then each rank's collective result is assembled from them.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1 import sp_prefill

WORLD_SIZES = (2, 3, 4)
# T mod 3 = 0, 1, 2 around the 4096-row prefill chunk, plus small edge cases.
TOKEN_COUNTS = (1, 2, 4, 7, 4095, 4096, 4097, 4098)


@pytest.mark.parametrize(
    ("raw", "rows"), [(None, 0), ("", 0), (" ", 0), ("0", 0), ("2048", 2048)]
)
def test_min_rows_from_env(monkeypatch, raw, rows):
    if raw is None:
        monkeypatch.delenv(sp_prefill.MIN_ROWS_ENV, raising=False)
    else:
        monkeypatch.setenv(sp_prefill.MIN_ROWS_ENV, raw)
    assert sp_prefill.min_rows_from_env() == rows


def test_min_rows_from_env_rejects_negative(monkeypatch):
    monkeypatch.setenv(sp_prefill.MIN_ROWS_ENV, "-1")
    with pytest.raises(ValueError):
        sp_prefill.min_rows_from_env()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_row_layout_partitions_real_rows(world_size, num_tokens):
    layouts = [
        sp_prefill.SPRows(num_tokens, world_size, rank) for rank in range(world_size)
    ]
    padded = layouts[0].padded_rows
    local = layouts[0].local_rows
    assert padded % world_size == 0
    assert 0 <= padded - num_tokens < world_size
    assert local * world_size == padded
    covered = []
    for rank, layout in enumerate(layouts):
        assert (layout.padded_rows, layout.local_rows) == (padded, local)
        assert layout.start == rank * local
        assert 0 <= layout.real_rows <= local
        covered.extend(range(layout.start, layout.start + layout.real_rows))
        if layout.real_rows < local:
            # Only trailing ranks own padding, and nothing real follows it.
            assert all(later.real_rows == 0 for later in layouts[rank + 1 :])
    assert covered == list(range(num_tokens))
    assert sum(local - layout.real_rows for layout in layouts) == padded - num_tokens


@pytest.mark.parametrize("num_tokens", (4096, 4097, 4098))
def test_row_layout_for_the_prefill_chunk_at_tp3(num_tokens):
    last = sp_prefill.SPRows(num_tokens, 3, 2)
    assert last.padded_rows == 4098
    assert last.local_rows == 1366
    assert last.start == 2732
    assert last.real_rows == num_tokens - 2732


def test_plan_rows_threshold():
    assert sp_prefill.plan_rows(4096, 0, 3, 0) is None
    assert sp_prefill.plan_rows(4096, 2048, 1, 0) is None
    assert sp_prefill.plan_rows(2047, 2048, 3, 0) is None
    assert sp_prefill.plan_rows(2048, 2048, 3, 1) == sp_prefill.SPRows(2048, 3, 1)
    with pytest.raises(ValueError):
        sp_prefill.plan_rows(4096, 2048, 3, 3)


def _config(
    *,
    tp=3,
    pp=1,
    capture=48,
    seqs=8,
    spec_tokens=5,
    boundary=False,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=capture),
        scheduler_config=SimpleNamespace(max_num_seqs=seqs),
        speculative_config=(
            None
            if spec_tokens is None
            else SimpleNamespace(num_speculative_tokens=spec_tokens)
        ),
        use_request_boundary_checkpoints=boundary,
    )


def test_resolve_min_rows(monkeypatch):
    monkeypatch.delenv(sp_prefill.MIN_ROWS_ENV, raising=False)
    # Off needs no configuration at all.
    assert sp_prefill.resolve_min_rows(None) == 0
    monkeypatch.setenv(sp_prefill.MIN_ROWS_ENV, "1024")
    assert sp_prefill.resolve_min_rows(_config()) == 1024
    # Raised above every decode forward: graphs and max_num_seqs * (1 + k).
    monkeypatch.setenv(sp_prefill.MIN_ROWS_ENV, "16")
    assert sp_prefill.resolve_min_rows(_config()) == 49
    assert sp_prefill.resolve_min_rows(_config(capture=512)) == 513
    assert sp_prefill.resolve_min_rows(_config(capture=None, spec_tokens=None)) == 16
    for disabled in (
        _config(tp=1),
        _config(pp=2),
        _config(boundary=True),
    ):
        assert sp_prefill.resolve_min_rows(disabled) == 0
    assert (
        sp_prefill.resolve_min_rows(_config(), static_sequence_parallel=True) == 0
    )


def _simulate_reduce_scatter(monkeypatch, layouts, contributions):
    """Install a reduce-scatter that returns each rank's slice of the sum."""
    total = sum(contributions)
    calls = iter(range(len(layouts)))

    def reduce_scatter(padded, dim):
        rank = next(calls)
        assert dim == 0 and padded is contributions[rank]
        return total.chunk(len(layouts))[rank].clone()

    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )


@pytest.mark.parametrize("num_tokens", (9, 10, 11))
def test_reduce_scatter_then_gather_round_trips(monkeypatch, num_tokens):
    world = 3
    full = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(num_tokens, 2)
    layouts = [sp_prefill.SPRows(num_tokens, world, rank) for rank in range(world)]
    # Partial sums whose exact total is `full`, as TP row-parallel outputs.
    contributions = []
    for layout, weight in zip(layouts, (1.0, 2.0, -2.0)):
        padded = layout.empty_padded((2,), torch.float32, torch.device("cpu"))
        assert padded.shape == (layout.padded_rows, 2)
        assert not padded[num_tokens:].any()
        padded[:num_tokens] = full * weight
        contributions.append(padded)
    _simulate_reduce_scatter(monkeypatch, layouts, contributions)
    local = [
        layout.reduce_scatter(padded)
        for layout, padded in zip(layouts, contributions)
    ]
    for layout, rows in zip(layouts, local):
        assert rows.shape == (layout.local_rows, 2)
        real = layout.real_rows
        torch.testing.assert_close(
            rows[:real], full[layout.start : layout.start + real], rtol=0, atol=0
        )
        assert not rows[real:].any()

    def all_gather(x, dim):
        assert dim == 0
        return torch.cat(local, dim=0)

    monkeypatch.setattr(sp_prefill, "tensor_model_parallel_all_gather", all_gather)
    for layout, rows in zip(layouts, local):
        torch.testing.assert_close(layout.gather(rows), full, rtol=0, atol=0)
    with pytest.raises(ValueError):
        layouts[0].gather(full)
    with pytest.raises(ValueError):
        layouts[0].reduce_scatter(torch.zeros(layouts[0].padded_rows + 1, 2))


def test_embedding_reduce_scatter_has_one_owner_per_token(monkeypatch):
    from vllm.model_executor.layers import vocab_parallel_embedding as embedding

    vocab, width, world = 12, 3, 3
    shard = vocab // world
    table = torch.arange(vocab * width, dtype=torch.float32).reshape(vocab, width)
    table += 1
    ids = torch.tensor([0, 11, 5, 4, 7, 3, 9])
    padded_rows = 9
    buffers = []

    def reduce_scatter(padded, dim):
        assert dim == 0
        buffers.append(padded.clone())
        return padded

    monkeypatch.setattr(
        embedding, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )
    for rank in range(world):
        rows = table[rank * shard : (rank + 1) * shard]
        layer = SimpleNamespace(
            tp_size=world,
            parallel_group=None,
            use_fused_embedding=False,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=rank * shard,
                org_vocab_end_index=(rank + 1) * shard,
                num_org_vocab_padding=0,
                added_vocab_start_index=vocab,
                added_vocab_end_index=vocab,
            ),
            quant_method=SimpleNamespace(
                embedding=lambda _, local_ids, rows=rows: rows[local_ids].clone()
            ),
        )
        embedding.VocabParallelEmbedding.forward_reduce_scatter(
            layer, ids, padded_rows
        )
    expected = torch.zeros(padded_rows, width)
    expected[: ids.numel()] = table[ids]
    torch.testing.assert_close(sum(buffers), expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        embedding.VocabParallelEmbedding.forward_reduce_scatter(layer, ids, 8)
