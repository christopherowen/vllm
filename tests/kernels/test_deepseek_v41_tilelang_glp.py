# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering's TileLang projection against FP64 and the FP32 torch projection
it replaces (``h - (alpha (h @ d)) d`` in FP32, then BF16)."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    INF,
    NAN,
    assert_same_nonfinite,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import glp, prepare  # noqa: E402

STREAMS, HIDDEN = 4, 5120


def _inputs(tokens: int, seed: int = 0, layers: int = 3):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    streams = torch.randn(
        tokens, STREAMS, HIDDEN, generator=generator, device="cuda"
    ).bfloat16()
    directions = torch.randn(layers, HIDDEN, generator=generator, device="cuda")
    directions /= directions.double().norm(dim=-1, keepdim=True).float()
    return streams, directions


def _project(streams, direction, alpha: float) -> torch.Tensor:
    out = torch.empty_like(streams)
    glp.glp_project(
        streams, direction, torch.tensor([alpha], device="cuda"), out
    )
    return out


def _fp64(streams, direction, alpha: float) -> torch.Tensor:
    h, d = streams.double(), direction.double()
    return h - (alpha * (h @ d)).unsqueeze(-1) * d


def _fp32_torch(streams, direction, alpha: float) -> torch.Tensor:
    """The projection as GLP steering first ran it, in eager torch."""
    h = streams.float()
    return (h - (alpha * (h @ direction)).unsqueeze(-1) * direction).bfloat16()


def _ulp(x: torch.Tensor) -> torch.Tensor:
    """BF16's spacing at ``|x|`` (eight significant bits)."""
    _, exponent = torch.frexp(x.abs().clamp_min(2.0**-126))
    return torch.ldexp(torch.ones_like(x), exponent - 8)


@pytest.mark.parametrize("tokens", [1, 7, 96, 384, 2048])
@pytest.mark.parametrize("alpha", [0.75, 1.0, 2.0])
def test_within_one_rounding_of_fp64(tokens, alpha):
    streams, directions = _inputs(tokens, seed=tokens)
    out = _project(streams, directions[1], alpha)
    exact = _fp64(streams, directions[1], alpha)
    assert out.dtype == torch.bfloat16 and out.shape == streams.shape
    # One rounding to BF16, plus the FP32 dot product's own error where an output
    # cancels to nearly zero (about 1e-8 here; 2^-24 bounds it).
    assert ((out.double() - exact).abs() <= _ulp(exact) + 2.0**-24).all()
    # At least as often correctly rounded as the FP32 torch projection.
    rounded = exact.bfloat16()
    ours = (out != rounded).double().mean().item()
    torch_path = (_fp32_torch(streams, directions[1], alpha) != rounded)
    assert ours <= torch_path.double().mean().item() + 1e-4, (ours, torch_path)


def test_the_direction_is_removed():
    """At alpha 1 what remains along d is only the outputs' BF16 rounding: no more than
    the correctly rounded projection leaves, and a small part of what was there."""
    streams, directions = _inputs(256)
    d = directions[2].double()
    before = (streams.double() @ d).abs()
    after = (_project(streams, directions[2], 1.0).double() @ d).abs()
    rounded = (_fp64(streams, directions[2], 1.0).bfloat16().double() @ d).abs()
    assert after.max().item() <= 1.25 * rounded.max().item() + 1e-4
    assert after.mean().item() <= 0.05 * before.mean().item()


def test_alpha_zero_and_a_zero_direction_change_nothing():
    streams, directions = _inputs(96)
    assert torch.equal(_project(streams, directions[0], 0.0), streams)
    zero = torch.zeros(HIDDEN, device="cuda")
    assert torch.equal(_project(streams, zero, 0.75), streams)


def test_rows_independent_of_batch():
    streams, directions = _inputs(130)
    full = _project(streams, directions[1], 0.75)
    for tokens in (1, 7, 64, 129):
        assert torch.equal(
            _project(streams[:tokens], directions[1], 0.75), full[:tokens]
        ), tokens


def test_repeatable():
    streams, directions = _inputs(384)
    first = _project(streams, directions[1], 0.75)
    for _ in range(3):
        assert torch.equal(_project(streams, directions[1], 0.75), first)


def test_a_row_of_the_layer_stack_is_read_in_place():
    """The model passes ``directions[layer]``, a view into the stack."""
    streams, directions = _inputs(16, layers=5)
    for layer in range(5):
        assert torch.equal(
            _project(streams, directions[layer], 0.75),
            _project(streams, directions[layer].clone(), 0.75),
        )


def test_nonfinite_like_the_torch_projection():
    """A NaN or an infinity in a stream reaches its whole row, as through
    ``h @ d``; other rows keep their bits."""
    streams, directions = _inputs(8)
    clean = _project(streams, directions[1], 0.75)
    for token, stream, col, value in ((0, 1, 3, NAN), (2, 0, 9, INF), (5, 3, 17, -INF)):
        streams[token, stream, col] = value
    out = _project(streams, directions[1], 0.75)
    assert_same_nonfinite(out, _fp32_torch(streams, directions[1], 0.75))
    poisoned = {(0, 1), (2, 0), (5, 3)}
    for token in range(8):
        for stream in range(STREAMS):
            if (token, stream) not in poisoned:
                assert torch.equal(out[token, stream], clean[token, stream])


def test_no_tokens_launch_nothing():
    streams, directions = _inputs(0)
    out = torch.empty_like(streams)
    glp.glp_project(streams, directions[0], torch.tensor([0.75], device="cuda"), out)


def test_declared_as_the_forward_calls_it():
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        glp.declare(STREAMS, HIDDEN)
    calls = [
        (function, args, kwargs)
        for function, args, kwargs in prepare.declared_calls(registry)
    ]
    assert calls == [(glp.glp_project_kernel, (STREAMS, HIDDEN), {})]
