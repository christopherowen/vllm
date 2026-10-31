# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang RMSNorm for DeepSeek-V4.1, through TileKernels' ``norm_forward``."""

import functools

import torch
from torch import nn

from vllm.models.deepseek_v4_1.tilelang import pdl, prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import SF_BLOCK
from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, empty_mxfp8


def _rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float
) -> None:
    pdl.norm_forward(x, weight, eps, out=out)


@functools.cache
def _warm(hidden: int, eps: float, dtype: torch.dtype) -> None:
    """TileKernels' RMSNorm of one zero row of ``hidden`` (``dtype`` input and weight),
    so its kernel (keyed by the width, eps and dtype, not the rows) is compiled
    before the cache freezes."""
    x = torch.zeros((1, hidden), dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")
    _rmsnorm(x, weight, torch.empty_like(x), eps)


@functools.cache
def _warm_with_cast(hidden: int, eps: float, dtype: torch.dtype) -> None:
    """TileKernels' fused RMSNorm + MXFP8 per-token cast of one zero row."""
    x = torch.zeros((1, hidden), dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")
    out_bf16 = torch.empty_like(x)
    cast = empty_mxfp8(out_bf16)
    pdl.norm_forward_and_per_token_cast(
        x,
        weight,
        eps,
        "e4m3",
        SF_BLOCK,
        round_sf=True,
        use_packed_ue8m0=True,
        with_out_bf16=True,
        out=(cast.values[:1], cast.scales[:1].view(torch.int32)),
        out_bf16=out_bf16,
    )


class TileLangRMSNorm(nn.Module):
    """``y = x * rsqrt(mean(x^2) + eps) * weight`` in FP32, in the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = eps
        self.with_cast = False

    def declare_kernels(self) -> None:
        """TileKernels' norm of this width; its input has the weight's dtype."""
        hidden, eps, dtype = (
            self.weight.shape[0],
            self.variance_epsilon,
            self.weight.dtype,
        )
        prepare.declare(warm=lambda: _warm(hidden, eps, dtype))
        if self.with_cast:
            prepare.declare(warm=lambda: _warm_with_cast(hidden, eps, dtype))

    def forward_with_cast(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> MXFP8Rows:
        """The normalized rows (written into ``out`` when given) beside their
        MXFP8 cast, in one TileKernels ``norm_forward_and_per_token_cast`` launch."""
        if x.ndim != 2 or (out is not None and out.ndim != 2):
            raise ValueError("forward_with_cast requires 2D input and output")
        if out is None:
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        cast = empty_mxfp8(out)
        m = x.shape[0]
        if m:
            pdl.norm_forward_and_per_token_cast(
                x,
                self.weight,
                self.variance_epsilon,
                "e4m3",
                SF_BLOCK,
                round_sf=True,
                use_packed_ue8m0=True,
                with_out_bf16=True,
                out=(cast.values[:m], cast.scales[:m].view(torch.int32)),
                out_bf16=out,
            )
        return cast

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """The normalized rows, into ``out`` (contiguous, ``x``'s rows) when given."""
        # TileKernels reads each row by its stride (a contiguous last dimension, rows
        # 16 elements apart): the q and kv halves of the fused projection's output are
        # normalized where they lie, with no copy.
        if x.ndim == 2 and (out is None or out.ndim == 2):
            if out is None:
                out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            pdl.norm_forward(x, self.weight, self.variance_epsilon, out=out)
            return out
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if out is None:
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        pdl.norm_forward(x, self.weight, self.variance_epsilon, out=out.view(x.shape))
        return out.view(shape)
