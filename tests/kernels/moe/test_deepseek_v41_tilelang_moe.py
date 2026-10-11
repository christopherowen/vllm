# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang routed experts against an FP32 torch reference."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
tile_kernels = pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.moe import (  # noqa: E402
    MoeExpertsV41Plan,
    workspace_bytes,
)

DEVICE = "cuda"
E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
LIMIT = 10.0


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _experts(num_experts, n, k, seed):
    """Packed E2M1 ``[E, N, K / 2]``, UE8M0 ``[E, N, K / 32]`` and their FP32 values."""
    gen = _gen(seed)
    codes = torch.randint(0, 16, (num_experts, n, k), generator=gen, device=DEVICE)
    exps = torch.randint(
        118, 122, (num_experts, n, k // 32), generator=gen, device=DEVICE
    )
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    lut = torch.tensor(E2M1, device=DEVICE)
    values = lut[codes] * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)
    return packed.contiguous(), exps.to(torch.uint8).contiguous(), values


def _routes(rows, num_experts, topk, seed, dtype):
    gen = _gen(seed)
    ids = torch.stack(
        [
            torch.randperm(num_experts, generator=gen, device=DEVICE)[:topk]
            for _ in range(rows)
        ]
    ).to(dtype)
    weights = torch.rand((rows, topk), generator=gen, device=DEVICE) + 0.1
    return ids.contiguous(), weights.float().contiguous()


def _reference(x, ids, weights, w13, w2, intermediate):
    """FP32 experts over the kernels' MXFP8 tokens; the down input stays unquantized."""
    xq, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
    )
    # Rows of packed exponents are padded to whole words.
    exps = sf.contiguous().view(torch.uint8)[:, : x.shape[1] // 32]
    xd = xq.float() * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)
    out = torch.zeros(x.shape, dtype=torch.float32, device=DEVICE)
    for slot in range(ids.shape[1]):
        e = ids[:, slot].long()
        h = torch.einsum("tk,tnk->tn", xd, w13[e]).bfloat16().float()
        gate = h[:, :intermediate].clamp(max=LIMIT)
        up = h[:, intermediate:].clamp(-LIMIT, LIMIT)
        act = torch.nn.functional.silu(gate) * up * weights[:, slot : slot + 1]
        out += torch.einsum("tn,tkn->tk", act, w2[e])
    return out


# Down projection K blocks of 128, 192 (TP4's 576-wide rows) and 64.
@pytest.mark.parametrize("hidden, intermediate", [(512, 256), (384, 576), (384, 320)])
@pytest.mark.parametrize("rows", [1, 12, 300])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_moe_experts(rows, idx_dtype, hidden, intermediate, num_experts=32, topk=6):
    w13_packed, w13_sf, w13 = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, w2 = _experts(num_experts, hidden, intermediate, 2)
    plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT
    )
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    ids, route_weights = _routes(rows, num_experts, topk, 4, idx_dtype)
    out = torch.empty_like(x)
    workspace = torch.empty(
        workspace_bytes(plan.workspace_specs(rows, idx_dtype)),
        dtype=torch.uint8,
        device=DEVICE,
    )
    plan.run(x, ids, route_weights, weights, out, workspace)
    ref = _reference(x, ids, route_weights, w13, w2, intermediate)
    scale = ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2 * scale)

    # A token's output does not depend on the rest of the batch.
    for r in {0, rows - 1}:
        solo = torch.empty_like(x[r : r + 1])
        plan.run(x[r : r + 1], ids[r : r + 1], route_weights[r : r + 1], weights, solo)
        assert torch.equal(solo[0], out[r]), f"row {r} changed with batch composition"


@pytest.mark.parametrize("rows", [1, 12, 300])
def test_moe_experts_fp32_output(
    rows, hidden=384, intermediate=576, num_experts=32, topk=6
):
    """An FP32 combine keeps the slot-order FP32 sum the BF16 combine rounds: the
    same sum, unrounded, and as independent of the batch."""
    w13_packed, w13_sf, _ = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, _ = _experts(num_experts, hidden, intermediate, 2)
    plans = {
        dtype: MoeExpertsV41Plan(
            num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT, out_dtype=dtype
        )
        for dtype in (torch.bfloat16, torch.float32)
    }
    weights = plans[torch.bfloat16].prepare_weights(
        w13_packed, w13_sf, w2_packed, w2_sf
    )
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
    outs = {}
    for dtype, plan in plans.items():
        assert plan.out_dtype == dtype
        outs[dtype] = torch.empty((rows, hidden), dtype=dtype, device=DEVICE)
        plan.run(x, ids, route_weights, weights, outs[dtype])
    assert torch.equal(outs[torch.float32].bfloat16(), outs[torch.bfloat16])
    assert not torch.equal(outs[torch.float32], outs[torch.bfloat16].float())
    for r in {0, rows - 1}:
        solo = torch.empty((1, hidden), dtype=torch.float32, device=DEVICE)
        plans[torch.float32].run(
            x[r : r + 1], ids[r : r + 1], route_weights[r : r + 1], weights, solo
        )
        assert torch.equal(solo[0], outs[torch.float32][r])


@pytest.mark.parametrize("rows", [1, 12, 300])
def test_moe_experts_fold_shared(
    rows, hidden=384, intermediate=576, num_experts=32, topk=6
):
    """The FP32 plan's combine with the shared expert's output folded in (BF16 out)
    gives the runner's bits: the FP32 partial plus the shared output in FP32, rounded
    once, as the all-reduce path and the reduce-scatter buffer compute it. A row's
    result does not depend on which of them a step takes."""
    w13_packed, w13_sf, _ = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, _ = _experts(num_experts, hidden, intermediate, 2)
    plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT,
        out_dtype=torch.float32,
    )  # fmt: skip
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    shared = (torch.randn((rows, hidden), generator=_gen(5), device=DEVICE)).bfloat16()
    ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
    wide = torch.empty((rows, hidden), dtype=torch.float32, device=DEVICE)
    plan.run(x, ids, route_weights, weights, wide)
    folded = torch.empty((rows, hidden), dtype=torch.bfloat16, device=DEVICE)
    plan.run(x, ids, route_weights, weights, folded, shared=shared)
    assert torch.equal(folded, (shared + wide).to(torch.bfloat16))  # all-reduce path
    buffer = torch.empty_like(folded)
    torch.add(shared, wide, out=buffer)  # reduce-scatter buffer
    assert torch.equal(folded, buffer)
    with pytest.raises(AssertionError):
        plan.run(x, ids, route_weights, weights, folded)  # BF16 out needs the shared


@pytest.mark.parametrize("rows", [1, 12, 96])
def test_moe_experts_fold_deferred_shared(
    rows, hidden=384, intermediate=576, num_experts=32, topk=6
):
    """A callable shared addend, as the runner passes for a shared expert on the aux
    stream: the plan calls it once, just before its combine, and the folded output
    has the bits of the shared rows passed directly."""
    w13_packed, w13_sf, _ = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, _ = _experts(num_experts, hidden, intermediate, 2)
    plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT,
        out_dtype=torch.float32,
    )  # fmt: skip
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    source = torch.randn((rows, hidden), generator=_gen(5), device=DEVICE)
    ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
    direct = torch.empty((rows, hidden), dtype=torch.bfloat16, device=DEVICE)
    plan.run(x, ids, route_weights, weights, direct, shared=source.bfloat16())

    side, ready = torch.cuda.Stream(), torch.cuda.Event()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        shared = source.bfloat16()
        ready.record(side)
    calls = []

    def resolve():
        calls.append(1)
        ready.wait(torch.cuda.current_stream())
        return shared

    deferred = torch.empty_like(direct)
    plan.run(x, ids, route_weights, weights, deferred, shared=resolve)
    assert calls == [1]
    assert torch.equal(deferred, direct)
