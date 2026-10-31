# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    DeadRows,
    combine_sampled_and_draft_tokens,
    get_num_sampled_and_rejected,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _combine(dead_rows):
    dev = "cuda"
    # Two decode requests (slots 3 and 1) with 5 drafts each, one prefill.
    idx_mapping = torch.tensor([3, 1, 0], dtype=torch.int32, device=dev)
    query_start_loc = torch.tensor([0, 6, 12, 20], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([106, 206, 8], dtype=torch.int32, device=dev)
    prefill_len = torch.tensor([50, 100, 0, 100], dtype=torch.int32, device=dev)
    last_sampled = torch.tensor([10, 11, 12, 13], dtype=torch.int32, device=dev)
    draft_tokens = torch.arange(20, dtype=torch.int32, device=dev).view(4, 5) + 100
    cu_num_logits = torch.tensor([0, 6, 12, 13], dtype=torch.int32, device=dev)
    input_ids = torch.zeros(24, dtype=torch.int32, device=dev)
    combine_sampled_and_draft_tokens(
        input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens,
        prefill_len, draft_tokens, cu_num_logits, 13, 1, dead_rows=dead_rows,
    )
    return input_ids


def test_dead_rows_mark_rows_past_the_survival_cut():
    dev = "cuda"
    confidence = torch.ones(4, 5, dtype=torch.float32, device=dev)
    # Slot 3: survival 0.9, 0.72, 0.36, ... -> cut 2 at tau 0.5.
    confidence[3] = torch.tensor([0.9, 0.8, 0.5, 0.9, 0.9])
    # Slot 1: first draft already below tau -> cut 0.
    confidence[1] = torch.tensor([0.4, 0.9, 0.9, 0.9, 0.9])
    is_padding = torch.zeros(24, dtype=torch.bool, device=dev)
    cut = torch.empty(3, dtype=torch.int32, device=dev)
    ids_dead = _combine(DeadRows(confidence, is_padding, cut, math.log(0.5)))
    ids_plain = _combine(None)
    assert torch.equal(ids_dead, ids_plain)
    assert cut.tolist() == [2, 0, 9]  # the prefill request is never clamped
    expected = torch.zeros(24, dtype=torch.bool)
    expected[3:6] = True    # request 0: rows of drafts 3-5
    expected[7:12] = True   # request 1: rows of drafts 1-5
    assert torch.equal(is_padding.cpu(), expected)


def test_dead_row_cut_clamps_committed_tokens():
    dev = "cuda"
    num_sampled = torch.tensor([6, 1, 4], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([106, 206, 306], dtype=torch.int32, device=dev)
    cu_num_logits = torch.tensor([0, 6, 12, 18], dtype=torch.int32, device=dev)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
    prefill_len = torch.tensor([10, 10, 10], dtype=torch.int32, device=dev)
    cut = torch.tensor([2, 0, 6], dtype=torch.int32, device=dev)
    sampled, rejected = get_num_sampled_and_rejected(
        num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len,
        dead_row_cut=cut,
    )
    assert sampled.tolist() == [3, 1, 4]
    assert rejected.tolist() == [3, 5, 2]


def test_input_buffers_reuse_and_contiguous_logits_rows(monkeypatch):
    from dataclasses import replace

    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.input_batch import (
        InputBatch,
        InputBuffers,
        expand_idx_mapping,
    )

    dev = torch.device("cuda")
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=32, device=dev)
    idx_mapping = torch.tensor([3, 1], dtype=torch.int64, device=dev)
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([20, 30], dtype=torch.int32, device=dev)
    prefill_len = torch.tensor([5, 5, 5, 5], dtype=torch.int32, device=dev)
    last_sampled = torch.tensor([10, 11, 12, 13], dtype=torch.int64, device=dev)
    draft_tokens = torch.arange(12, dtype=torch.int64, device=dev).view(4, 3) + 100

    logits_indices = combine_sampled_and_draft_tokens(
        buffers.input_ids,
        idx_mapping,
        last_sampled,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        query_start_loc,
        8,
        1,
        out=buffers.logits_indices,
    )
    assert logits_indices.data_ptr() == buffers.logits_indices.data_ptr()
    assert logits_indices.tolist() == list(range(8))

    exp_idx, exp_pos = expand_idx_mapping(
        idx_mapping,
        8,
        query_start_loc,
        4,
        out=(buffers.expanded_idx_mapping, buffers.expanded_local_pos),
    )
    assert exp_idx.data_ptr() == buffers.expanded_idx_mapping.data_ptr()
    assert exp_pos.data_ptr() == buffers.expanded_local_pos.data_ptr()
    assert exp_idx.tolist() == [3, 3, 3, 3, 1, 1, 1, 1]
    assert exp_pos.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]

    batch = InputBatch.make_dummy(2, 8, buffers)
    batch.logits_indices = logits_indices
    batch.cu_num_logits = query_start_loc
    batch._logits_rows_contiguous = True
    batch._deferred_expand_max_len = 4

    def _forbid_logits_rows(*args, **kwargs):
        raise AssertionError("tilelang_inputs.logits_rows should not be called")

    monkeypatch.setattr(tilelang_inputs, "logits_rows", _forbid_logits_rows)
    ids_view, pos_view = batch.logits_rows()
    assert ids_view.data_ptr() == batch.input_ids.data_ptr()
    assert pos_view.data_ptr() == batch.positions.data_ptr()
    assert ids_view.shape == (8,)
    assert pos_view.shape == (8,)

    # dataclasses.replace must not inherit _logits_rows_contiguous or _logits_rows.
    replaced = replace(batch, num_reqs=1)
    assert not replaced._logits_rows_contiguous
    assert replaced._logits_rows is None

    # Lazy expansion populates expanded_idx_mapping/expanded_local_pos into buffers.
    buffers.expanded_idx_mapping.zero_()
    buffers.expanded_local_pos.zero_()
    batch.idx_mapping = idx_mapping
    batch.ensure_expanded_idx_mapping(buffers)
    assert batch._deferred_expand_max_len is None
    assert batch.expanded_idx_mapping.data_ptr() == buffers.expanded_idx_mapping.data_ptr()
    assert batch.expanded_idx_mapping.tolist() == [3, 3, 3, 3, 1, 1, 1, 1]
    assert batch.expanded_local_pos.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_sample_contiguous_fastpath_skips_gather_and_lazy_expands_on_fallback(
    monkeypatch,
):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    dev = torch.device("cuda")
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=16, device=dev)
    batch = InputBatch.make_dummy(2, 6, buffers)
    batch.idx_mapping = torch.tensor([2, 0], dtype=torch.int64, device=dev)
    batch.cu_num_logits = torch.tensor([0, 3, 6], dtype=torch.int32, device=dev)
    batch.logits_indices = torch.arange(6, dtype=torch.int64, device=dev)
    batch.num_draft_tokens = 4
    batch._logits_rows_contiguous = True
    batch._deferred_expand_max_len = 3

    hidden_states = torch.randn(8, 32, dtype=torch.bfloat16, device=dev)

    def _forbid_gather(*args, **kwargs):
        raise AssertionError("gather_hidden_rows should not be called")

    monkeypatch.setattr(tilelang_inputs, "gather_hidden_rows", _forbid_gather)

    captured_hidden = {}

    def _fake_verify_greedily(sample_hidden, input_batch, sink_flags):
        captured_hidden["tensor"] = sample_hidden
        return SamplerOutput(
            sampled_token_ids=torch.zeros((2, 3), dtype=torch.int64, device=dev),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.ones(2, dtype=torch.int32, device=dev),
            num_rejected=torch.zeros(2, dtype=torch.int32, device=dev),
        )

    runner = SimpleNamespace(
        sink=None,
        batch_sharder=None,
        input_buffers=buffers,
        _verifies_greedily=lambda ib, go: go is None,
        _verify_greedily=_fake_verify_greedily,
        model=SimpleNamespace(
            compute_logits=lambda hs: torch.zeros((hs.shape[0], 16), device=dev)
        ),
        sampler=lambda logits, ib: SamplerOutput(
            sampled_token_ids=torch.zeros((2, 1), dtype=torch.int64, device=dev),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.ones(2, dtype=torch.int32, device=dev),
            num_rejected=torch.zeros(2, dtype=torch.int32, device=dev),
        ),
        rejection_sampler=None,
    )

    # 1. Greedy verification path: slices hidden_states[:6] without gather or expand.
    GPUModelRunner.sample(runner, hidden_states, batch, grammar_output=None)
    assert captured_hidden["tensor"].data_ptr() == hidden_states.data_ptr()
    assert captured_hidden["tensor"].shape == (6, 32)
    assert batch._deferred_expand_max_len == 3

    # 2. Non-greedy fallback path: lazily runs ensure_expanded_idx_mapping before sampling.
    runner._verifies_greedily = lambda ib, go: False
    GPUModelRunner.sample(runner, hidden_states, batch, grammar_output=None)
    assert batch._deferred_expand_max_len is None
    assert batch.expanded_idx_mapping.tolist() == [2, 2, 2, 0, 0, 0]
    assert batch.expanded_local_pos.tolist() == [0, 1, 2, 0, 1, 2]
