# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_markov_shard_pair,
    sample_greedy_markov,
    scratch_shape,
    select_first_argmax,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _sharded(base, bias, ranks, padded):
    """Reduce per-shard pairs as TP ranks would, padding each shard like a
    vocabulary-parallel head (the padded columns are never read)."""
    rows, vocab = base.shape
    per_rank = -(-vocab // ranks)
    values = torch.empty(scratch_shape(rows, padded), dtype=torch.float32, device="cuda")
    indices = torch.empty(scratch_shape(rows, padded), dtype=torch.int32, device="cuda")
    pairs = []
    for rank in range(ranks):
        start = rank * per_rank
        stop = min(vocab, start + per_rank)
        shard_base = torch.full((rows, padded), 7.0, dtype=base.dtype, device="cuda")
        shard_bias = torch.full((rows, padded), 7.0, dtype=bias.dtype, device="cuda")
        shard_base[:, : stop - start] = base[:, start:stop]
        shard_bias[:, : stop - start] = bias[:, start:stop]
        pair = torch.empty(rows, 2, dtype=torch.float32, device="cuda")
        greedy_markov_shard_pair(
            shard_base[:, : stop - start], shard_bias[:, : stop - start], start,
            pair, values, indices,
        )
        pairs.append(pair)
    out = torch.empty(rows, 5, dtype=torch.int64, device="cuda")[:, 2]
    select_first_argmax(torch.cat(pairs, dim=-1), out)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ranks", [1, 2, 3])
def test_shard_pairs_reproduce_the_full_argmax(dtype, ranks):
    torch.manual_seed(0)
    rows, vocab = 8, 129280
    base = torch.randn(rows, vocab, device="cuda").to(dtype)
    for r in range(rows):
        base[r, (r * 3) * 2048 + 7] = 25.0
        base[r, (r * 3 + 1) * 2048 + 13] = 22.0
    bias = torch.randn(rows, vocab, device="cuda").to(dtype)
    # Ties across shard boundaries resolve to the lowest index; NaN wins.
    base[1, [100, 50000, 120000]] = 30
    bias[1, [100, 50000, 120000]] = 0
    base[2, [90000, 20000]] = 40
    bias[2, [90000, 20000]] = 0
    base[3, 70000] = float("nan")
    expected = torch.empty(rows, dtype=torch.int64, device="cuda")
    values = torch.empty(scratch_shape(rows, vocab), dtype=torch.float32, device="cuda")
    indices = torch.empty(scratch_shape(rows, vocab), dtype=torch.int32, device="cuda")
    sample_greedy_markov(base, bias, expected, values, indices)
    assert torch.equal(expected, (base + bias).argmax(dim=-1))
    got = _sharded(base, bias, ranks, padded=-(-vocab // ranks) + 64)
    assert got.tolist() == expected.tolist()
    assert expected[1].item() == 100 and expected[2].item() == 20000 and expected[3].item() == 70000


def _sharded_top2(base, bias, ranks, padded, embed=None, branch_cfg=None):
    rows, vocab = base.shape
    per_rank = -(-vocab // ranks)
    values = torch.empty(
        scratch_shape(rows, padded, top2=True), dtype=torch.float32, device="cuda"
    )
    indices = torch.empty(
        scratch_shape(rows, padded, top2=True), dtype=torch.int32, device="cuda"
    )
    pairs = []
    for rank in range(ranks):
        start = rank * per_rank
        stop = min(vocab, start + per_rank)
        shard_base = torch.full((rows, padded), 7.0, dtype=base.dtype, device="cuda")
        shard_bias = torch.full((rows, padded), 7.0, dtype=bias.dtype, device="cuda")
        shard_base[:, : stop - start] = base[:, start:stop]
        shard_bias[:, : stop - start] = bias[:, start:stop]
        pair = torch.empty(rows, 4, dtype=torch.float32, device="cuda")
        greedy_markov_shard_pair(
            shard_base[:, : stop - start],
            shard_bias[:, : stop - start],
            start,
            pair,
            values,
            indices,
            top2=True,
        )
        pairs.append(pair)
    out = torch.empty(rows, 5, dtype=torch.int64, device="cuda")[:, 2]
    runner_up = torch.empty(rows, 3, dtype=torch.int64, device="cuda")[:, 1]
    margin = torch.empty(rows, 2, dtype=torch.float32, device="cuda")[:, 0]
    weight, branch_margin, branch_weight = (
        branch_cfg if branch_cfg is not None else (None, 0.0, 1.0)
    )
    select_first_argmax(
        torch.cat(pairs, dim=-1),
        out,
        pair_width=4,
        embed=embed,
        top2_branch=(weight, runner_up, margin, branch_margin, branch_weight),
    )
    return out, runner_up, margin


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ranks", [1, 2, 3, 4])
def test_shard_pairs_top2_reproduce_full_top1_and_runner_up(dtype, ranks):
    torch.manual_seed(7)
    rows, vocab = 8, 129280
    base = torch.randn(rows, vocab, device="cuda").to(dtype)
    for r in range(rows):
        base[r, (r * 3) * 2048 + 7] = 25.0
        base[r, (r * 3 + 1) * 2048 + 13] = 22.0
    bias = torch.randn(rows, vocab, device="cuda").to(dtype)
    # Row 1: three-way tie across shards -> top1=100, runner-up=50000
    base[1, [100, 50000, 120000]] = 30
    bias[1, [100, 50000, 120000]] = 0
    # Row 2: distinct top1=20000 (40.0) and runner-up=90000 (39.5) across shards
    base[2, 20000] = 40.0
    bias[2, 20000] = 0.0
    base[2, 90000] = 39.5
    bias[2, 90000] = 0.0
    # Row 3: top1 and runner-up in the same shard (even within the same 2048-col part)
    base[3, 4100] = 50.0
    bias[3, 4100] = 0.0
    base[3, 6200] = 48.0
    bias[3, 6200] = 0.0

    summed = (base.float() + bias.float()).to(dtype).float()
    expected_top1 = summed.argmax(dim=-1)
    masked = summed.clone()
    masked[torch.arange(rows, device="cuda"), expected_top1] = -float("inf")
    expected_top2 = masked.argmax(dim=-1)
    expected_margin = (
        summed[torch.arange(rows, device="cuda"), expected_top1]
        - summed[torch.arange(rows, device="cuda"), expected_top2]
    )

    got_top1, got_top2, got_margin = _sharded_top2(
        base, bias, ranks, padded=-(-vocab // ranks) + 64
    )
    assert torch.equal(got_top1, expected_top1)
    assert torch.equal(got_top2, expected_top2)
    assert torch.allclose(got_margin, expected_margin, atol=0.0, rtol=0.0)
    assert got_top1[1].item() == 100 and got_top2[1].item() == 50000
    assert got_top1[2].item() == 20000 and got_top2[2].item() == 90000
    assert got_top1[3].item() == 4100 and got_top2[3].item() == 6200


@pytest.mark.parametrize("ranks", [1, 3, 4])
def test_top2_branch_markov_confidence_promotion(ranks):
    torch.manual_seed(11)
    rows, vocab, embed_width = 4, 129280, 256
    base = torch.full((rows, vocab), -10.0, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((rows, vocab), dtype=torch.bfloat16, device="cuda")

    # Row 0: narrow margin (0.125 <= 0.25), runner-up has strong Markov confidence -> promote
    base[0, 1000] = 10.0
    base[0, 95000] = 9.875
    # Row 1: narrow margin (0.125 <= 0.25), but top1 has stronger Markov confidence -> keep top1
    base[1, 2000] = 10.0
    base[1, 96000] = 9.875
    # Row 2: wide margin (1.0 > 0.25), even with strong runner-up confidence -> keep top1
    base[2, 3000] = 10.0
    base[2, 97000] = 9.0
    # Row 3: exact tie (margin == 0.0); with branch_margin=0.0 must stay bit-identical to argmax
    base[3, 4000] = 10.0
    base[3, 98000] = 10.0

    table = torch.zeros((vocab, embed_width), dtype=torch.bfloat16, device="cuda")
    table[95000] = 1.0
    table[1000] = -1.0
    table[2000] = 1.0
    table[96000] = -1.0
    table[97000] = 1.0
    table[3000] = -1.0
    table[98000] = 1.0
    table[4000] = -1.0
    markov_conf_weight = torch.full(
        (embed_width,), 0.01, dtype=torch.float32, device="cuda"
    )

    step_out0 = torch.empty((rows, embed_width), dtype=torch.bfloat16, device="cuda")
    conf_out0 = torch.empty((rows, embed_width), dtype=torch.bfloat16, device="cuda")
    got_zero_margin, got_zero_runner, _ = _sharded_top2(
        base,
        bias,
        ranks,
        padded=-(-vocab // ranks) + 64,
        embed=(table, step_out0, conf_out0),
        branch_cfg=(markov_conf_weight, 0.0, 1.0),
    )
    assert got_zero_margin.tolist() == [1000, 2000, 3000, 4000]
    assert got_zero_runner.tolist() == [95000, 96000, 97000, 98000]
    assert torch.equal(step_out0, table[got_zero_margin])
    assert torch.equal(conf_out0, table[got_zero_margin])

    step_out = torch.empty((rows, embed_width), dtype=torch.bfloat16, device="cuda")
    conf_out = torch.empty((rows, embed_width), dtype=torch.bfloat16, device="cuda")
    got_branched, got_runner, got_margin = _sharded_top2(
        base,
        bias,
        ranks,
        padded=-(-vocab // ranks) + 64,
        embed=(table, step_out, conf_out),
        branch_cfg=(markov_conf_weight, 0.25, 1.0),
    )
    assert got_branched.tolist() == [95000, 2000, 3000, 98000]
    assert got_runner.tolist() == [1000, 96000, 97000, 4000]
    assert torch.equal(step_out, table[got_branched])
    assert torch.equal(conf_out, table[got_branched])
    assert got_margin.tolist() == [0.125, 0.125, 1.0, 0.0]
