# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _run_prepare(
    *,
    target_positions: list[int],
    block_table_values: list[int],
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
):
    device = torch.device("cuda")
    max_num_reqs = 4
    max_num_tokens = 16
    num_speculative_steps = 3

    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device),
        positions=torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device),
        query_start_loc=torch.full(
            (max_num_reqs + 1,), -1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        positions=torch.tensor(target_positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([2], dtype=torch.int32, device=device),
    )
    query_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    sample_indices = torch.full(
        (max_num_reqs * num_speculative_steps,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    sample_pos = torch.full_like(sample_indices, -1)
    sample_idx_mapping = torch.full(
        sample_indices.shape, -1, dtype=torch.int32, device=device
    )
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    input_temperature = torch.tensor(
        [0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device
    )
    input_seeds = torch.tensor([0, 0, 17, 0], dtype=torch.int64, device=device)
    last_sampled = torch.tensor([0, 0, 99, 0], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.zeros_like(last_sampled)
    block_table = torch.tensor([block_table_values], dtype=torch.int32, device=device)

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        4,
        cp_rank,
        cp_size,
        cp_interleave,
        123,
        num_speculative_steps,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        128,
        sample_from_anchor=True,
    )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mapping.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mapping.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
        temperature=temperature.cpu(),
        seeds=seeds.cpu(),
    )


def test_prepare_dflash_inputs_excludes_rejected_context_suffix():
    # Positions 10/11 use physical block 7. Rejected positions 12/13 would use
    # block 8, but must be PAD context rather than contaminating draft KV.
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [30, 31, PAD_SLOT_ID, PAD_SLOT_ID]

    # The replacement query starts immediately after the two valid rows and
    # advances from the last accepted position (11).
    assert out.input_buffers.input_ids[:3].cpu().tolist() == [99, 123, 123]
    assert out.input_buffers.positions[:3].cpu().tolist() == [12, 13, 14]
    assert out.query_slot_mapping[:3].tolist() == [32, 33, 34]
    assert out.sample_indices[:3].tolist() == [0, 1, 2]
    assert out.sample_pos[:3].tolist() == [13, 14, 15]
    assert out.sample_idx_mapping[:3].tolist() == [2, 2, 2]
    assert out.temperature[2].item() == 1.0
    assert out.seeds[2].item() == 17


def test_prepare_dflash_inputs_excludes_rejected_context_suffix_with_dcp():
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 7, 8, 9],
        cp_rank=1,
        cp_size=2,
        cp_interleave=2,
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [28, 29, PAD_SLOT_ID, PAD_SLOT_ID]
    assert out.query_slot_mapping[:3].tolist() == [PAD_SLOT_ID, PAD_SLOT_ID, 30]


def test_prepare_dflash_inputs_never_writes_the_null_block():
    # The valid context uses logical block 0 and the replacement query uses
    # logical block 1. Both map to the null block and must remain unwritable.
    out = _run_prepare(
        target_positions=[2, 3, 4, 5],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_slot_mapping[:4].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert out.query_slot_mapping[:3].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]


def test_dflash_full_cg_attn_metadata_fastpath_detection():
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    FlashAttentionMetadataBuilder = type("FlashAttentionMetadataBuilder", (), {})

    def _make_builder(**kwargs):
        defaults = dict(
            aot_schedule=False,
            dcp_world_size=1,
            rswa_window=None,
            mm_prefix_query_ranges_np=None,
        )
        defaults.update(kwargs)
        b = FlashAttentionMetadataBuilder()
        for k, v in defaults.items():
            setattr(b, k, v)
        return SimpleNamespace(get_metadata_builder=lambda: b)

    spec = DFlashSpeculator.__new__(DFlashSpeculator)
    spec.draft_cp_size = 1
    spec.block_tables = SimpleNamespace(cp_size=1)
    spec.model_state = SimpleNamespace()
    spec.attn_groups = [[_make_builder()], [_make_builder()], [_make_builder()]]
    assert spec._compute_full_cg_needs_attn_metadata_build() is False

    class _CustomState:
        def prepare_draft_attn_metadata(self, **kwargs):
            return {}

    spec.model_state = _CustomState()
    assert spec._compute_full_cg_needs_attn_metadata_build() is True
    spec.model_state = SimpleNamespace()

    spec.attn_groups = [[_make_builder(aot_schedule=True)]]
    assert spec._compute_full_cg_needs_attn_metadata_build() is True

    spec.attn_groups = [[_make_builder(rswa_window=(128, 0))]]
    assert spec._compute_full_cg_needs_attn_metadata_build() is True

    spec.attn_groups = [[_make_builder()]]
    spec.draft_cp_size = 2
    assert spec._compute_full_cg_needs_attn_metadata_build() is True


def test_dflash_build_uniform_attn_metadata_forwards_max_query_len(monkeypatch):
    import numpy as np
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.spec_decode import speculator as speculator_mod
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    captured = {}

    def _fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        speculator_mod, "build_attn_metadata", _fake_build_attn_metadata
    )

    spec = DFlashSpeculator.__new__(DFlashSpeculator)
    spec.arange_np = np.arange(5, dtype=np.int32)
    spec._query_start_loc_cpu = torch.zeros(5, dtype=torch.int32, device="cpu")
    spec._draft_seq_lens_cpu_upper_bound = torch.zeros(
        4, dtype=torch.int32, device="cpu"
    )
    spec.draft_attn_layer_names = ["layer.0"]
    spec.draft_cp_size = 1
    spec.block_tables = SimpleNamespace(
        cp_size=1,
        input_block_tables=[torch.zeros((4, 8), dtype=torch.int32)],
        slot_mappings=torch.zeros((1, 20), dtype=torch.int64),
    )
    spec.input_buffers = SimpleNamespace(
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
    )
    spec.max_model_len = 1024
    spec.draft_max_seq_len = 64
    spec.kv_cache_config = SimpleNamespace()
    spec.attn_groups = []
    spec.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    spec.idx_mapping = torch.arange(4, dtype=torch.int32)
    spec.model_state = SimpleNamespace(prepare_draft_attn_metadata=lambda **kw: {})

    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=20,
        num_reqs=4,
    )
    seq_lens_ub = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    out = spec._build_uniform_attn_metadata(
        batch_desc=batch_desc,
        num_reqs=2,
        num_query_per_req=5,
        seq_lens_cpu_upper_bound=seq_lens_ub,
        step=5,
        causal=True,
    )
    assert out == {"ok": True}
    assert captured["max_query_len"] == 5
    assert captured["num_reqs"] == 4
    assert captured["num_tokens"] == 20
    assert captured["query_start_loc_cpu"].tolist() == [0, 5, 10, 10, 10]
    assert captured["seq_lens_cpu_upper_bound"].tolist() == [15, 25, 0, 0]
