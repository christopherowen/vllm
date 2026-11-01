# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM


def _draft(weight: torch.Tensor, scale: float = 1.0) -> DSparkDeepseekV4ForCausalLM:
    draft = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    head = SimpleNamespace(markov_w2=SimpleNamespace(weight=weight))
    object.__setattr__(draft, "model", SimpleNamespace(markov_head=head))
    object.__setattr__(draft, "logits_processor", SimpleNamespace(scale=scale))
    return draft


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_gathered_markov_bias_matches_dense_bias_on_candidates(scale):
    torch.manual_seed(0)
    vocab, rank, reqs, k = 512, 16, 3, 64
    weight = torch.randn(vocab, rank, dtype=torch.float32)
    embed = torch.randn(reqs, rank, dtype=torch.float32)
    base = torch.randn(reqs, vocab, dtype=torch.float32)
    values, index = base.topk(k, dim=-1)
    logits = torch.full_like(base, float("-inf"))

    out = _draft(weight, scale).apply_markov_bias_gathered(embed, logits, values, index)

    dense = base + scale * (embed @ weight.T)
    torch.testing.assert_close(out.gather(1, index), dense.gather(1, index))
    mask = torch.ones_like(out, dtype=torch.bool).scatter_(1, index, False)
    assert torch.isneginf(out[mask]).all()


def test_gathered_markov_bias_with_full_vocab_keeps_greedy_argmax():
    torch.manual_seed(1)
    vocab, rank, reqs = 256, 8, 4
    weight = torch.randn(vocab, rank)
    embed = torch.randn(reqs, rank)
    base = torch.randn(reqs, vocab)
    values, index = base.topk(vocab, dim=-1)
    out = _draft(weight).apply_markov_bias_gathered(
        embed, torch.full_like(base, float("-inf")), values, index
    )
    assert torch.equal(out.argmax(-1), (base + embed @ weight.T).argmax(-1))


def test_gathered_markov_bias_with_packed_bf16_w2():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("TileLang V4.1 kernels target SM120/SM121")
    pytest.importorskip("tilelang")
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight

    torch.manual_seed(2)
    vocab, rank, reqs, k = 512, 256, 3, 64
    weight = (torch.randn(vocab, rank, device="cuda") * 0.05).bfloat16()
    weight[5, 17] = 300.0
    embed = torch.randn(reqs, rank, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(reqs, vocab, device="cuda", dtype=torch.bfloat16)
    values, index = base.topk(k, dim=-1)

    unpacked_out = _draft(weight).apply_markov_bias_gathered(
        embed, torch.full_like(base, float("-inf")), values, index
    )

    packed = pack_bf16_weight(weight.clone())
    shard = VocabParallelEmbedding._get_indices(vocab, vocab, vocab, vocab, 0, 1)
    w2 = SimpleNamespace(
        weight=packed.values,
        weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries,
        weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=rank,
        packed_bf16_exponent_base=packed.exponent_base,
        shard_indices=shard,
    )
    draft = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    object.__setattr__(draft, "model", SimpleNamespace(markov_head=SimpleNamespace(markov_w2=w2)))
    object.__setattr__(draft, "logits_processor", SimpleNamespace(scale=1.0))

    packed_out = draft.apply_markov_bias_gathered(
        embed, torch.full_like(base, float("-inf")), values, index
    )
    assert torch.equal(packed_out.view(torch.int16), unpacked_out.view(torch.int16))
