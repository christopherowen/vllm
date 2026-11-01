# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared expert's clamped SwiGLU against DeepSeek's reference arithmetic."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    INF,
    NAN,
    assert_finite_equal,
    assert_same_nonfinite,
)
from vllm.models.deepseek_v4_1.tilelang.activation import (  # noqa: E402
    SiluAndMulClampFP32,
)


def _reference(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """``Expert.forward`` of DeepSeek's inference/model.py, between w1/w3 and w2."""
    n = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :n].float(), gate_up[..., n:].float()
    up = torch.clamp(up, min=-limit, max=limit)
    gate = torch.clamp(gate, max=limit)
    return (F.silu(gate) * up).to(gate_up.dtype)


@pytest.mark.parametrize("rows", [1, 6, 96, 2024])
@pytest.mark.parametrize("n", [512, 576])
def test_matches_the_reference(rows, n):
    limit = 10.0
    generator = torch.Generator(device="cuda").manual_seed(rows + n)
    gate_up = (
        torch.randn(rows, 2 * n, generator=generator, device="cuda") * 6
    ).bfloat16()
    out = SiluAndMulClampFP32(limit)(gate_up)
    assert out.shape == (rows, n) and out.dtype == torch.bfloat16
    assert torch.equal(out, _reference(gate_up, limit))


def test_rows_independent_of_batch():
    limit, n = 10.0, 512
    gate_up = (torch.randn(130, 2 * n, device="cuda") * 6).bfloat16()
    act = SiluAndMulClampFP32(limit)
    full = act(gate_up)
    for rows in (1, 7, 64, 129):
        assert torch.equal(act(gate_up[:rows]), full[:rows]), rows


def test_nonfinite_like_the_reference():
    """``torch.clamp`` passes a NaN through and saturates the infinities; a clamp
    built on ``fminf``/``fmaxf`` would return the limit for a NaN and hide it."""
    limit, n = 10.0, 512
    generator = torch.Generator(device="cuda").manual_seed(5)
    gate_up = (torch.randn(8, 2 * n, generator=generator, device="cuda") * 6).bfloat16()
    for row, col, value in (
        (0, 3, NAN),  # gate
        (1, n + 5, NAN),  # up
        (2, 7, INF),  # gate: silu(limit)
        (3, 9, -INF),  # gate: silu(-inf) is NaN
        (4, n + 11, INF),  # up saturates at the limit
        (5, n + 13, -INF),
    ):
        gate_up[row, col] = value
    out = SiluAndMulClampFP32(limit)(gate_up)
    ref = _reference(gate_up, limit)
    assert_same_nonfinite(out, ref)
    assert_finite_equal(out, ref)


@pytest.mark.parametrize("rows", [1, 6, 33, 96, 2024])
@pytest.mark.parametrize("n", [512, 576, 768])
def test_swiglu_clamp_cast_matches_separate_cast(rows, n):
    """``SiluAndMulClampFP32.forward_mxfp8`` (``swiglu_clamp_cast_kernel``) writes
    the exact E4M3 values and UE8M0 scale bytes of ``swiglu_clamp_kernel`` followed
    by ``cast_mxfp8``."""
    from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8

    limit = 10.0
    generator = torch.Generator(device="cuda").manual_seed(100 + rows + n)
    gate_up = (
        torch.randn(rows, 2 * n, generator=generator, device="cuda") * 6
    ).bfloat16()
    act = SiluAndMulClampFP32(limit, n, mxfp8_output=True)
    got = act.forward_mxfp8(gate_up)
    want = cast_mxfp8(_reference(gate_up, limit))
    assert got.values.shape == want.values.shape
    assert got.scales.shape == want.scales.shape
    assert torch.equal(
        got.values[:rows].view(torch.uint8), want.values[:rows].view(torch.uint8)
    )
    assert torch.equal(got.scales[:rows, : n // 32], want.scales[:rows, : n // 32])


def test_swiglu_clamp_cast_feeds_block32_down_proj():
    """Fused ``swiglu_clamp_cast_kernel`` + ``provide_mxfp8`` into ``down_proj``
    (5120 x 576) produces the exact bits of ``swiglu_clamp_kernel`` followed by
    ``down_proj``'s own cast."""
    from types import SimpleNamespace
    from torch import nn
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm.models.deepseek_v4_1.tilelang.gemm import pack_scale_words
    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangFP8LinearMethod,
        provide_mxfp8,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    limit, hidden, shared_n = 10.0, 5120, 576
    gen = torch.Generator(device="cuda").manual_seed(77)
    weight = (torch.randn((hidden, shared_n), generator=gen, device="cuda") * 2).to(
        torch.float8_e4m3fn
    )
    exps = torch.randint(
        120, 125, (hidden // 32, shared_n // 32), generator=gen, device="cuda"
    ).to(torch.uint8)
    qcfg = SimpleNamespace(weight_block_size=[32, 32])

    with set_current_vllm_config(VllmConfig()):
        plain_down = nn.Module()
        plain_down.weight = nn.Parameter(weight.clone(), requires_grad=False)
        plain_down.weight_scale_inv = nn.Parameter(exps.clone(), requires_grad=False)
        plain_method = TileLangFP8LinearMethod(qcfg)
        plain_method.process_weights_after_loading(plain_down)
        plain_method.declare_layer_kernels(plain_down)

        fused_down = nn.Module()
        fused_down.weight = nn.Parameter(weight.clone(), requires_grad=False)
        fused_down.weight_scale_inv = nn.Parameter(exps.clone(), requires_grad=False)
        fused_down.tilelang_mxfp8_input = True
        fused_method = TileLangFP8LinearMethod(qcfg)
        fused_method.process_weights_after_loading(fused_down)
        fused_method.declare_layer_kernels(fused_down)
        assert fused_method.get_workspace_size(fused_down, 64) == 0
        assert torch.equal(plain_down.tilelang_weight_sf, pack_scale_words(exps))

        plain_act = SiluAndMulClampFP32(limit, shared_n)
        fused_act = SiluAndMulClampFP32(limit, shared_n, mxfp8_output=True)
        gate_up = (torch.randn(200, 2 * shared_n, generator=gen, device="cuda") * 6).bfloat16()
        scratch = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
        with use_preallocated_workspace(scratch):
            for rows in (1, 6, 16, 17, 32, 64, 96, 200):
                gu = gate_up[:rows]
                expected = plain_method.apply(plain_down, plain_act(gu))
                act_q = fused_act.forward_mxfp8(gu)
                with provide_mxfp8(act_q):
                    got = fused_method.apply(fused_down, act_q.rows)
                assert torch.equal(got, expected), f"rows={rows}"
