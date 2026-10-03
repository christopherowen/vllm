# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang GEMMs and RMSNorm against torch references."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
tile_kernels = pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    DECODE_ROWS,
    bf16_gemm,
    bf16_gemm_partials,
    bf16_shards,
    default_config,
    mxfp8_gemm,
    pack_scale_words,
    splitk_reduce,
)
from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm  # noqa: E402

DEVICE = "cuda"
ROWS = (1, 7, 64, 65, 200)


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _block32_weight(n, k, seed):
    """E4M3 ``[N, K]`` with one UE8M0 scale per 32x32 block, and its FP32 values."""
    gen = _gen(seed)
    weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(
        torch.float8_e4m3fn
    )
    exps = torch.randint(120, 125, (n // 32, k // 32), generator=gen, device=DEVICE).to(
        torch.uint8
    )
    scale = (
        torch.exp2(exps.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    )
    return weight, exps, weight.float() * scale


def _mxfp8(x):
    xq, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
    )
    sf = sf.contiguous().view(torch.uint8)
    # Rows of packed exponents are padded to whole words.
    exps = sf[:, : x.shape[1] // 32]
    values = xq.float() * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)
    return xq, sf, values


# K blocks of 128 and, for TP4's 576-wide shared expert rows, 192 and 64.
@pytest.mark.parametrize(
    "n, k", [(256, 512), (1536, 1024), (224, 384), (320, 576), (1152, 576)]
)
def test_mxfp8_gemm(n, k):
    weight, exps, weight_values = _block32_weight(n, k, 1)
    weight_sf = pack_scale_words(exps, rows=n)
    x = torch.randn((max(ROWS), k), generator=_gen(2), device=DEVICE).bfloat16()
    xq, sf, x_values = _mxfp8(x)
    ref = x_values @ weight_values.T
    words = sf.contiguous().view(torch.uint32)
    decode = mxfp8_gemm(n, k, **default_config(DECODE_ROWS, k), padded_rows=True)
    large = mxfp8_gemm(n, k, **default_config(DECODE_ROWS + 1, k))
    outs = {}
    for rows in ROWS:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=DEVICE)
        if rows <= DECODE_ROWS:
            # Decode rows read one whole 64-row tile; only live rows are stored.
            decode(xq[:DECODE_ROWS], weight, words[:DECODE_ROWS], weight_sf, out)
        else:
            large(xq[:rows], weight, words[:rows], weight_sf, out)
        torch.testing.assert_close(
            out.float(), ref[:rows], rtol=1e-2, atol=1e-2 * ref.abs().max().item()
        )
        outs[rows] = out
    # Rows agree bit for bit across batch sizes, padding and tile configurations.
    for rows, out in outs.items():
        assert torch.equal(out, outs[max(ROWS)][:rows]), (
            f"rows={rows} differs from the full batch"
        )


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
# The router (384 x 5120) splits K eight ways; (100, 200) cannot split.
@pytest.mark.parametrize("n, k", [(32, 4096), (384, 512), (384, 5120), (100, 200)])
def test_bf16_gemm(n, k, out_dtype):
    gen = _gen(3)
    weight = torch.randn((n, k), generator=gen, device=DEVICE).bfloat16()
    x = torch.randn((max(ROWS), k), generator=gen, device=DEVICE).bfloat16()
    tl_dtype = {torch.bfloat16: "bfloat16", torch.float32: "float32"}[out_dtype]
    shards = bf16_shards(n, k)
    gemm = bf16_gemm(n, k, out_dtype=tl_dtype, shards=shards)
    ref = x.float() @ weight.float().T
    full = torch.empty((max(ROWS), n), dtype=out_dtype, device=DEVICE)
    gemm(x, weight, full)
    torch.testing.assert_close(
        full.float(), ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item()
    )
    if shards > 1:
        partials = bf16_gemm_partials(n, k, shards)
        reduce = splitk_reduce(n, shards, out_dtype=tl_dtype)
    for rows in ROWS:
        out = torch.empty((rows, n), dtype=out_dtype, device=DEVICE)
        if shards > 1 and rows <= DECODE_ROWS:
            # Decode rows: split-K partials, reduced in shard order.
            p = torch.empty((shards, rows, n), dtype=torch.float32, device=DEVICE)
            partials(x[:rows], weight, p)
            reduce(p, out)
        else:
            gemm(x[:rows], weight, out)
        assert torch.equal(out, full[:rows]), f"rows={rows} differs from the full batch"


@pytest.mark.parametrize("hidden", [128, 512, 4096])
def test_rmsnorm(hidden, rows=33, eps=1e-6):
    gen = _gen(4)
    norm = TileLangRMSNorm(hidden, eps).to(DEVICE)
    norm.weight.data.copy_(torch.rand((hidden,), generator=gen, device=DEVICE) + 0.5)
    x = torch.randn((rows, hidden), generator=gen, device=DEVICE).bfloat16()
    out = norm(x)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * norm.weight.float()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
