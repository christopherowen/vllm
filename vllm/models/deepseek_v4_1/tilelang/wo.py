# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang output projection of the DeepSeek-V4.1 attention, as DeepSeek's
reference computes it::

    o[..., -64:] = inverse_rope(o[..., -64:])          (BF16, in place)
    a = wo_a(o.view(T, groups, -1))                    (grouped, block-32 FP8, BF16 out)
    out = wo_b(a.flatten(1))                           (block-32 FP8, BF16 out)

WO-A keeps the BF16 activations, as the reference does (``convert.py`` dequantizes
``wo_a`` to BF16 and the projection is a BF16 einsum): one launch of the grouped
W8A16 GEMM expands each FP8 weight tile exactly to BF16 and accumulates in FP32.
WO-B is the ordinary block-32 projection with MXFP8 activations (DeepSeek's E4M3 and
one UE8M0 scale per 32), as the reference's ``linear`` quantizes them. The
checkpoint tensors are read in place.
"""

import torch
from torch import nn

from vllm.v1.worker.workspace import current_preallocated_workspace

from .. import l2_prefetch
from .gemm import fp8w_bf16_gemm
from .linear import _block32_linear, _prepare_block32
from .rope import rope_

# WO-A row tiles: every tile runs the same K loop, so the bits do not depend on it.
_WO_A_TILES = {
    16: dict(block_N=32, num_stages=2),
    32: dict(block_N=32, num_stages=2),
    64: dict(block_N=64, num_stages=2),
    128: dict(block_N=64, num_stages=3),
}


class _Weight:
    """A block-32 FP8 weight and its scales; weak-referenceable for the custom op."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.weight_scale_inv = layer.weight_scale_inv.detach()
        _prepare_block32(self, groups)
        # 16-stream steps (up to 96 rows) stay on decode tiles, two of 64 rows.
        self.tilelang_decode_rows = 128


class _GroupedBF16:
    """WO-A for the W8A16 GEMM: the FP8 weight, its UE8M0 exponents (also what the L2
    prefetch reads beside it, as ``tilelang_weight_sf``) and one kernel per row tile."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.tilelang_weight_sf = layer.weight_scale_inv.detach().view(torch.uint8)
        n, k = self.weight.shape
        self.kernels = {
            rows: fp8w_bf16_gemm(n // groups, k, groups, block_M=rows, **config)
            for rows, config in _WO_A_TILES.items()
        }

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> None:
        rows = x.shape[0]
        tile = next((t for t in _WO_A_TILES if rows <= t), max(_WO_A_TILES))
        self.kernels[tile](x, self.weight, self.tilelang_weight_sf, out)


class TileLangWOProjection:
    def __init__(self, wo_a: nn.Module, wo_b: nn.Module, *, groups: int):
        self.a = _GroupedBF16(wo_a, groups)
        self.b = _Weight(wo_b, 1)
        self.hidden = wo_b.weight.shape[0]

    def l2_segments(self) -> list:
        """The bytes both GEMMs read, WO-A first: each weight and its scale words."""
        return l2_prefetch.linear_segments(
            "wo_a", self.a
        ) + l2_prefetch.linear_segments("wo_b", self.b)

    def __call__(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Project ``o`` ([T, heads, head_dim], contiguous BF16; its RoPE columns
        are overwritten) into ``out`` ([T, hidden] BF16)."""
        rows = o.shape[0]
        rope_(o, positions, cos_sin_cache, 1, True)
        scratch = (
            None if torch.compiler.is_compiling() else current_preallocated_workspace()
        )
        a = torch.empty(
            (rows, self.a.weight.shape[0]), dtype=torch.bfloat16, device=o.device
        )
        if rows:
            self.a(o.view(rows, -1), a)
        _block32_linear(a, out, self.b.tilelang_key, scratch)
        return out


__all__ = ["TileLangWOProjection"]
