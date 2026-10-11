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
