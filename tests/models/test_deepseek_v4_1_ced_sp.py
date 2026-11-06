# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED boundary selection and real model orchestration with CPU collectives.

Integer GPU mapping and graph replay are tested in the TileLang CED suite.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.models.test_deepseek_v4_1_ced_model import _gather, _Layer, _scatter
from tests.models.test_deepseek_v4_1_sp_prefill import _World
from vllm.models.deepseek_v4_1 import ced, sp_prefill
from vllm.models.deepseek_v4_1.nvidia import model as native


def _index_reference(indices, local, merged, start, local_rows, total_rows):
    valid = (indices >= 0) & (indices < total_rows)
    own = valid & (indices >= start) & (indices < start + local_rows)
    local.copy_(torch.where(own, indices - start, -1))
    merged.copy_(
        torch.where(
            valid, indices // local_rows * len(indices) + torch.arange(len(indices)), -1
        )
    )


def _copy_rows(source, indices, output, *, scatter):
    assert not scatter
    valid = (indices >= 0) & (indices < source.shape[0])
    output.zero_()
    output[valid] = source[indices[valid]]


def _cpu_collectives(monkeypatch, world):
    monkeypatch.setattr(ced.ced_kernels, "sp_indices_kernel", lambda: _index_reference)
    monkeypatch.setattr(ced.ced_kernels, "copy_rows", _copy_rows)
    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_all_gather", world.all_gather
    )
    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_reduce_scatter", world.reduce_scatter
    )
    monkeypatch.setattr(sp_prefill, "model_parallel_is_initialized", lambda: False)


@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("rows", [1, 2, 31, 32, 33])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_selected_sp_carries_preserve_bits(monkeypatch, world_size, rows, dtype):
    world = _World(world_size)
    _cpu_collectives(monkeypatch, world)
    layout = sp_prefill.SPRows(rows, world_size, 0)
    # Include signed zero and nonfinite payloads: an arithmetic reduction cannot
    # stand in for selection. The final SP rank can contain only padding.
    bits = torch.tensor([0, -32768, 0x7FC1, 0x7F80, -128, 0x3F80], dtype=torch.int16)
    full = bits.repeat((layout.padded_rows * 4 + 5) // 6)[: layout.padded_rows * 4]
    full = full.view(torch.bfloat16).reshape(layout.padded_rows, 4).to(dtype)
    indices = torch.tensor([rows - 1, 0, -1, rows, rows + 11, rows - 1])
    expected = full.new_zeros((len(indices), 4))
    valid = (indices >= 0) & (indices < rows)
    expected[valid] = full[indices[valid]]

    def run(rank):
        sp = sp_prefill.SPRows(rows, world_size, rank)
        local = full[sp.start : sp.start + sp.local_rows]
        return ced.gather_sp_rows(sp, indices, local, local[:, :2])

    integer_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    for actual, narrow in world.run(run):
        assert torch.equal(actual.view(integer_dtype), expected.view(integer_dtype))
        assert torch.equal(
            narrow.view(integer_dtype), expected[:, :2].contiguous().view(integer_dtype)
        )
    empty = ced.gather_sp_rows(layout, indices[:0], full[: layout.local_rows])
    assert empty[0].shape == (0, 4)
    with pytest.raises(ValueError, match="expects"):
        ced.gather_sp_rows(layout, indices, full[:0])


@pytest.mark.parametrize("world_size", [3, 4])
@pytest.mark.parametrize(
    "rows, selection", [(31, "sparse"), (32, "sparse"), (33, "dense"), (1025, "replay")]
)
def test_sp_ced_boundary_matches_full_carry(monkeypatch, world_size, rows, selection):
    world = _World(world_size)
    _cpu_collectives(monkeypatch, world)
    monkeypatch.setattr(
        native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        native, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(native, "get_tensor_model_parallel_rank", lambda: world.rank)
    monkeypatch.setattr(native.l2_prefetch, "ENABLED", False)
    monkeypatch.setattr(native, "gather_rows", _gather)
    monkeypatch.setattr(native, "scatter_rows", _scatter)
    positions = torch.arange(100, 100 + rows)
    ids = torch.arange(rows)
    inputs = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) / 8
    if selection == "dense":
        indices = torch.arange(rows - 1)
    elif selection == "replay":
        indices = torch.arange(rows - 128, rows)
    else:
        indices = torch.tensor([rows // 3 - 1, rows // 3, rows - 1, -1])
    layout = sp_prefill.SPRows(rows, world_size, 0)
    compact_exchange = len(indices) * world_size < rows

    def partial(result, padded_rows):
        padded = result.new_zeros((padded_rows, *result.shape[1:]))
        if world.rank == 0:
            padded[: result.shape[0]] = result
        return world.reduce_scatter(padded, 0)

    def run(min_rows):
        shared, row_work, pre_rows = {}, [], []
        layers = nn.ModuleList(_Layer(i, shared, row_work) for i in range(4))
        for index, layer in enumerate(layers):
            attention = layer.attn.forward
            experts = layer.ffn.forward
            pre = layer.mhc.pre

            def attn(positions, hidden, scaling, original=attention):
                sp = sp_prefill.current()
                if sp is not None:
                    hidden = sp.gather(hidden)
                result = original(positions, hidden, scaling)
                return result if sp is None else partial(result, sp.padded_rows)

            def ffn(hidden, ids, *, reduce_scatter_rows=None, original=experts):
                result = original(hidden, ids)
                return (
                    result
                    if reduce_scatter_rows is None
                    else partial(result, reduce_scatter_rows)
                )

            def track_pre(residual, *args, original=pre, index=index, **kwargs):
                pre_rows.append((index, residual.shape[0]))
                return original(residual, *args, **kwargs)

            layer.attn.forward, layer.ffn.forward, layer.mhc.pre = attn, ffn, track_pre
        target = SimpleNamespace(
            embed_input_ids=lambda _: inputs,
            prefill_sp_min_rows=min_rows,
            prefill_sp_unreduced=False,
            disk_engram=False,
            engram_hash=None,
            start_layer=0,
            end_layer=4,
            ced_decoder_start=2,
            layers=layers,
            aux_hidden_state_layers={2, 3, 4},
            _sink_columns=None,
            _graph_out=lambda index, rows: None,
            glp=None,
            norm=lambda hidden: hidden,
        )
        output, aux = native.DeepseekV4Model.forward(
            target, ids, positions, None, inputs_embeds=inputs, ced_indices=indices
        )
        return output, aux, shared, pre_rows

    expected, expected_aux, context, _ = run(0)
    for output, aux, shared, pre_rows in world.run(lambda rank: run(4)):
        assert torch.equal(output, expected)
        for actual, want in zip(aux, expected_aux, strict=True):
            assert torch.equal(actual, want)
        assert shared["global_rows"] == rows
        assert torch.equal(shared["global"], context["global"])
        # First pre call at the boundary is normalization; post_pre also calls
        # the mock pre, now with compact rows, after attention.
        assert next(count for index, count in pre_rows if index == 2) == (
            layout.local_rows if compact_exchange else rows
        )


def test_sp_index_mapping_is_declared_before_cache_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        ced.prepare, "declare", lambda declaration, **kw: calls.extend(declaration())
    )
    state = ced.CEDState(8, 512, (128, 256, 512), "cpu")
    state.warmup(5120, 4, pipeline=True, sink_columns=True)
    assert (ced.ced_kernels.sp_indices_kernel, (), {}) in calls
