# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang attention kernels against torch references.

Caches use vLLM's page layout, including allocator padding past each page's
records, so the kernels are checked against the strides they see in serving.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

tilelang = pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang.indexer import (  # noqa: E402
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    IndexerTopK,
    IndexerV41Plan,
    page_bytes,
    quantize_mxfp4,
    write_index_keys,
)
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (  # noqa: E402
    HEAD_DIM,
    IDX_GROUP,
    SWA_GROUP,
    SparseMlaV41Plan,
)

DEVICE = "cuda"
SF_BLOCK = 32
PAD = 256  # allocator padding past each page's records
LAYERS = 3  # BLHNC: each block holds every layer's page, so a layer's rows stride
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _page_buffer(pages, page_bytes, fill=0xAB):
    """One layer's view of a BLHNC buffer: padded pages, rows strided over layers.

    The other layers' pages and the padding hold ``fill``.
    """
    buffer = torch.full(
        (pages, LAYERS, page_bytes + PAD), fill, dtype=torch.uint8, device=DEVICE
    )
    view = buffer[:, 1]
    assert view.stride(0) == LAYERS * (page_bytes + PAD)
    return view


def _paged(records, page_size):
    """Lay ``[slots, record_bytes]`` records out as one layer's cache pages."""
    slots, record_bytes = records.shape
    pages = slots // page_size
    pool = _page_buffer(pages, page_size * record_bytes)
    pool[:, : page_size * record_bytes] = records.view(pages, -1)
    return pool


def _e2m1_codes(normalized):
    lut = torch.tensor(E2M1, device=normalized.device)
    codes = (normalized.abs().clamp(max=6.0).unsqueeze(-1) - lut).abs().argmin(-1)
    return codes.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)


def _e2m1_values(codes):
    lut = torch.tensor(E2M1, device=codes.device)
    return torch.where((codes & 8) != 0, -lut[codes & 7], lut[codes & 7])


# --------------------------------------------------------------------------
# Indexer
# --------------------------------------------------------------------------


def _e2m1_rne(x):
    """``cvt.rn.satfinite.e2m1``: nearest code, ties to even, saturating at 6."""
    m = x.abs()
    code = sum(
        (m > t if strict else m >= t).to(torch.uint8)
        for t, strict in (
            (0.25, True),
            (0.75, False),
            (1.25, True),
            (1.75, False),
            (2.5, True),
            (3.5, False),
            (5.0, True),
        )
    )
    return code | (torch.signbit(x).to(torch.uint8) << 3)


def _quantize_mxfp4(x):
    """B12X's MXFP4 quantizer: UE8M0 ``pow2_ceil(amax * fp32(1/6))``, RNE values."""
    groups = x.float().reshape(*x.shape[:-1], INDEX_DIM // SF_BLOCK, SF_BLOCK)
    amax = groups.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    bits = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    sf = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = torch.exp2(127.0 - sf.float())
    codes = _e2m1_rne((groups * scale.unsqueeze(-1)).reshape(x.shape))
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), sf.to(torch.uint8).contiguous()


def _decode_mxfp4(packed, scales):
    u = packed.long()
    codes = torch.stack([u & 15, u >> 4], -1).reshape(*packed.shape[:-1], INDEX_DIM)
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(SF_BLOCK, -1)
    return _e2m1_values(codes) * scale


def _index_inputs(rows, context, page_size, seed=0):
    """Exactly representable MXFP4 operands, so scores compare bit for bit."""
    gen = _gen(seed)
    logical_pages = (context + page_size - 1) // page_size
    pool_pages = logical_pages * rows + 3
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=DEVICE)

    def operand(shape):
        values = lut[
            torch.randint(0, 16, (*shape, INDEX_DIM), generator=gen, device=DEVICE)
        ]
        exps = torch.randint(
            -3, 4, (*shape, INDEX_DIM // SF_BLOCK), generator=gen, device=DEVICE
        )
        return values * torch.exp2(exps.float()).repeat_interleave(SF_BLOCK, -1)

    q_packed, q_scales = _quantize_mxfp4(operand((rows, INDEX_HEADS)))
    k_packed, k_scales = _quantize_mxfp4(operand((pool_pages * page_size,)))
    pool = _page_buffer(pool_pages, page_bytes(page_size))
    pool[:, : page_size * 64].copy_(k_packed.view(pool_pages, -1))
    pool[:, page_size * 64 : page_size * 68].copy_(k_scales.view(pool_pages, -1))
    weights = (
        torch.randn((rows, INDEX_HEADS), generator=gen, device=DEVICE) / 64
    ).bfloat16()
    page_table = torch.stack(
        [
            torch.randperm(pool_pages, generator=gen, device=DEVICE)[:logical_pages]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    lengths = torch.randint(
        context // 2,
        context + 1,
        (rows,),
        generator=gen,
        device=DEVICE,
        dtype=torch.int32,
    )
    active = torch.tensor([context], dtype=torch.int32, device=DEVICE)
    return dict(
        q_packed=q_packed,
        q_scales=q_scales,
        weights=weights,
        pool=pool,
        page_table=page_table,
        lengths=lengths,
        active=active,
        q_values=_decode_mxfp4(q_packed, q_scales),
        k_values=_decode_mxfp4(k_packed, k_scales),
    )


def _mxfp4_inputs(n, seed):
    gen = _gen(seed)
    x = torch.randn((n, INDEX_DIM), generator=gen, device=DEVICE) * 3
    # Exact midpoints after scaling exercise the ties-to-even rule.
    x[:4] = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device=DEVICE
    ).repeat(16)[:INDEX_DIM]
    x[4] = -x[0]
    x[5] = 0
    x[6, :32] = -0.0
    return x.bfloat16()


def test_mxfp4_quantize(n=3000):
    x = _mxfp4_inputs(n, 11)
    q = torch.empty((n, 64), dtype=torch.uint8, device=DEVICE)
    scales = torch.empty((n, 4), dtype=torch.uint8, device=DEVICE)
    quantize_mxfp4(x, q, scales)
    ref_q, ref_scales = _quantize_mxfp4(x)
    assert torch.equal(scales, ref_scales)
    assert torch.equal(q, ref_q)


def test_write_index_keys(page_size=64, pages=5):
    n = pages * page_size // 2
    x = _mxfp4_inputs(n, 12)
    gen = _gen(13)
    slots = torch.randperm(pages * page_size, generator=gen, device=DEVICE)[:n]
    slots[:3] = -1  # padding rows write nothing
    pool = _page_buffer(pages, page_bytes(page_size))
    expected = pool.clone()
    write_index_keys(x, pool, slots, page_size)
    ref_q, ref_scales = _quantize_mxfp4(x)
    keep = slots >= 0
    page, token = slots[keep] // page_size, slots[keep] % page_size
    data = expected[:, : page_size * 64].reshape(pages, page_size, 64)
    scale_rows = expected[:, page_size * 64 : page_size * 68].reshape(
        pages, page_size, 4
    )
    data[page, token] = ref_q[keep]
    scale_rows[page, token] = ref_scales[keep]
    expected[:, : page_size * 64] = data.reshape(pages, -1)
    expected[:, page_size * 64 : page_size * 68] = scale_rows.reshape(pages, -1)
    assert torch.equal(pool, expected)
    # Neither the other layers' pages nor the padding were written.
    assert bool((pool._base[:, [0, 2]] == 0xAB).all())


def _ref_scores(inp, page_size, extent, positions_of):
    """Staged BF16 score: BF16 dot, ReLU, BF16 weighted product, ordered head sum."""
    rows = inp["q_values"].shape[0]
    scores = []
    for r in range(rows):
        positions = positions_of(r)
        safe = positions.clamp_min(0).long()
        slots = (
            inp["page_table"][r].long()[safe // page_size] * page_size
            + safe % page_size
        )
        dots = (inp["k_values"][slots].double() @ inp["q_values"][r].double().T).float()
        contrib = dots.bfloat16().float().clamp_min(0) * inp["weights"][r].float()
        contrib = contrib.bfloat16().float()
        total = torch.zeros(len(positions), device=DEVICE)
        for h in range(INDEX_HEADS):
            total = total + contrib[:, h]
        total = torch.where(positions >= 0, total, float("-inf")).bfloat16()
        scores.append(total[: extent[r]])
    return scores


def _ref_topk(scores, extent, topk, group=1, gather=None):
    """Larger scores first, then lower positions; ascending, ``-1`` padded."""
    rows = scores.shape[0]
    out = torch.full((rows, topk * group), -1, dtype=torch.int32, device=DEVICE)
    counts = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        n = int(extent[r])
        values = scores[r, :n].float()
        if group > 1:
            blocks = (n + group - 1) // group
            padded = torch.full((blocks * group,), float("-inf"), device=DEVICE)
            padded[:n] = values
            values = padded.view(blocks, group).amax(1)
            if blocks:
                values[(n - 1) // group] = float("inf")
        order = torch.sort(values, descending=True, stable=True).indices
        chosen = order[values[order] > float("-inf")][:topk].sort().values
        if group > 1:
            pos = (
                chosen[:, None] * group + torch.arange(group, device=DEVICE)
            ).flatten()
            chosen = torch.where(pos < n, pos, -1)
            counts[r] = int((chosen >= 0).sum())
        elif gather is not None:
            chosen = gather[r, chosen]
        out[r, : chosen.numel()] = chosen.to(torch.int32)
    return out, counts


def _dense_extent(inp, width):
    return inp["lengths"].clamp(0, int(inp["active"][0])).clamp(max=width).tolist()


@pytest.mark.parametrize("context", [4096, 9000])
def test_indexer_dense(context, page_size=64, rows=4):
    inp = _index_inputs(rows, context, page_size)
    width = (context + page_size - 1) // page_size * page_size
    plan = IndexerV41Plan(width, page_size)
    out, scores = plan.run(
        *(
            inp[k]
            for k in (
                "q_packed",
                "q_scales",
                "weights",
                "pool",
                "page_table",
                "lengths",
                "active",
            )
        )
    )
    extent = _dense_extent(inp, width)
    ref = _ref_scores(
        inp, page_size, extent, lambda r: torch.arange(extent[r], device=DEVICE)
    )
    for r in range(rows):
        assert torch.equal(scores[r, : extent[r]], ref[r]), f"row {r} scores differ"
    expected, _ = _ref_topk(scores, extent, INDEX_TOPK)
    assert torch.equal(out, expected)


def test_indexer_candidates(page_size=64, rows=3, context=8192):
    inp = _index_inputs(rows, context, page_size)
    width = CANDIDATE_BLOCKS * CANDIDATE_BLOCK
    gen = _gen(7)
    candidate_lengths = torch.randint(
        width // 8,
        context // 2,
        (rows,),
        generator=gen,
        device=DEVICE,
        dtype=torch.int32,
    )
    candidates = torch.full((rows, width), -1, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        n = int(candidate_lengths[r])
        chosen = torch.randperm(int(inp["lengths"][r]), generator=gen, device=DEVICE)[
            :n
        ]
        candidates[r, :n] = chosen.sort().values.to(torch.int32)
    candidates[0, 5] = -1  # masked candidates score -inf
    plan = IndexerV41Plan(width, page_size, candidates=True)
    args = (
        inp[k]
        for k in (
            "q_packed",
            "q_scales",
            "weights",
            "pool",
            "page_table",
            "lengths",
            "active",
        )
    )
    out, scores = plan.run(*args, candidates, candidate_lengths)
    extent = candidate_lengths.clamp(0, width).tolist()
    ref = _ref_scores(inp, page_size, extent, lambda r: candidates[r, : extent[r]])
    for r in range(rows):
        assert torch.equal(scores[r, : extent[r]], ref[r]), f"row {r} scores differ"
    expected, _ = _ref_topk(scores, extent, INDEX_TOPK, gather=candidates)
    assert torch.equal(out, expected)


def test_indexer_publishes_candidates(page_size=64, rows=2, context=20000):
    inp = _index_inputs(rows, context, page_size)
    width = (context + page_size - 1) // page_size * page_size
    plan = IndexerV41Plan(width, page_size, candidate_blocks=True)
    candidate_out = torch.full(
        (rows, CANDIDATE_BLOCKS * CANDIDATE_BLOCK), -7, dtype=torch.int32, device=DEVICE
    )
    counts = torch.full((rows,), -7, dtype=torch.int32, device=DEVICE)
    args = (
        inp[k]
        for k in (
            "q_packed",
            "q_scales",
            "weights",
            "pool",
            "page_table",
            "lengths",
            "active",
        )
    )
    out, scores = plan.run(
        *args, candidate_out=candidate_out, candidate_out_lengths=counts
    )
    extent = _dense_extent(inp, width)
    expected, _ = _ref_topk(scores, extent, INDEX_TOPK)
    assert torch.equal(out, expected)
    blocks, block_counts = _ref_topk(scores, extent, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
    assert torch.equal(candidate_out, blocks)
    assert torch.equal(counts, block_counts)


def _tie_scores(rows, width, seed=3):
    # Few distinct values force large ties at every selection threshold.
    gen = _gen(seed)
    return (
        torch.randint(0, 6, (rows, width), generator=gen, device=DEVICE).float() * 0.25
    ).bfloat16()


@pytest.mark.parametrize(
    "width, topk, group",
    [
        (4000, INDEX_TOPK, 1),  # one pass
        (20000, INDEX_TOPK, 1),  # segments, one merge
        (300000, INDEX_TOPK, 1),  # segments, two merge levels
        (
            4000 * CANDIDATE_BLOCK + 8,
            CANDIDATE_BLOCKS,
            CANDIDATE_BLOCK,
        ),  # blocks, one merge
        (200000, CANDIDATE_BLOCKS, CANDIDATE_BLOCK),  # blocks, two merge levels
    ],
)
def test_topk_routes(width, topk, group, rows=4):
    scores = _tie_scores(rows, width)
    scores[0, 100:] = float("-inf")  # fewer selectable entries than topk
    lengths = torch.tensor(
        [width, width - 17, 1000, 0], dtype=torch.int32, device=DEVICE
    )
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    op = IndexerTopK(width, topk, group)
    indices = torch.full((rows, topk * group), -7, dtype=torch.int32, device=DEVICE)
    values = torch.empty((rows, topk), dtype=torch.bfloat16, device=DEVICE)
    counts = (
        torch.full((rows,), -7, dtype=torch.int32, device=DEVICE) if group > 1 else None
    )
    op.run(scores, lengths, active, indices, values, counts=counts)
    expected, expected_counts = _ref_topk(scores, lengths.tolist(), topk, group)
    assert torch.equal(indices, expected), (
        f"{op.segments} segments, {len(op.levels)} merges"
    )
    if counts is not None:
        assert torch.equal(counts, expected_counts)


# --------------------------------------------------------------------------
# Sparse MLA
# --------------------------------------------------------------------------


def _pack_swa(kv):
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // SWA_GROUP, SWA_GROUP)
    exponent = torch.ceil(
        torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)
    ).clamp(-126, 127)
    payload = (
        (groups / torch.exp2(exponent).unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    scales = (exponent + 127).to(torch.uint8)
    return torch.cat(
        [payload.view(torch.uint8).reshape(n, HEAD_DIM), scales], -1
    ).contiguous()


def _pack_idx(kv):
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // IDX_GROUP, IDX_GROUP)
    scales = (
        (groups.abs().amax(-1).clamp_min(6 * 2**-9) / 6.0)
        .clamp(max=448)
        .to(torch.float8_e4m3fn)
    )
    codes = _e2m1_codes(groups / scales.float().unsqueeze(-1)).reshape(n, HEAD_DIM)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat([payload, scales.view(torch.uint8)], -1).contiguous()


def _decode_swa(records):
    payload = records[:, :HEAD_DIM].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.exp2(records[:, HEAD_DIM:].float() - 127.0)
    return (
        payload.reshape(-1, HEAD_DIM // SWA_GROUP, SWA_GROUP) * scales.unsqueeze(-1)
    ).reshape(-1, HEAD_DIM)


def _decode_idx(records):
    packed = records[:, : HEAD_DIM // 2].long()
    codes = torch.stack([packed & 15, packed >> 4], -1).reshape(-1, HEAD_DIM)
    scales = records[:, HEAD_DIM // 2 :].contiguous().view(torch.float8_e4m3fn).float()
    values = _e2m1_values(codes).reshape(-1, HEAD_DIM // IDX_GROUP, IDX_GROUP)
    return (values * scales.unsqueeze(-1)).reshape(-1, HEAD_DIM)


def _ref_mla(q, swa, swa_idx, swa_len, sinks, idx=None, idx_slots=None, idx_len=None):
    rows, heads, _ = q.shape
    out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    for r in range(rows):
        s = swa_idx[r, : int(swa_len[r])]
        kv = [_decode_swa(swa[s[s >= 0].long()])]
        if idx is not None:
            i = idx_slots[r, : int(idx_len[r])]
            kv.append(_decode_idx(idx[i[i >= 0].long()]))
        kv = torch.cat(kv)
        scores = q[r].float() @ kv.T * HEAD_DIM**-0.5
        top = torch.maximum(scores.max(-1).values if kv.shape[0] else sinks, sinks)
        weights = torch.exp(scores - top[:, None])
        out[r] = weights @ kv / (weights.sum(-1) + torch.exp(sinks - top))[:, None]
    return out.bfloat16()


@pytest.mark.parametrize("idx_width", [0, 512])
# Heads per rank: TP4 (16), TP3 (24 of 72 padded), TP1 (64).
@pytest.mark.parametrize("heads", [16, 24, 64])
def test_sparse_mla(
    idx_width, heads, rows=6, swa_width=128, swa_page=128, idx_page=128
):
    gen = _gen(1)
    swa_pages, idx_pages, idx_context = 24, 40, 4096
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    sinks = torch.randn((heads,), generator=gen, device=DEVICE)
    swa = _pack_swa(
        torch.randn((swa_pages * swa_page, HEAD_DIM), generator=gen, device=DEVICE)
    )

    def lists(width, extent, min_len):
        lengths = torch.randint(
            min_len, width + 1, (rows,), generator=gen, device=DEVICE, dtype=torch.int32
        )
        chosen = torch.stack(
            [
                torch.randperm(extent, generator=gen, device=DEVICE)[:width]
                for _ in range(rows)
            ]
        )
        keep = torch.arange(width, device=DEVICE) < lengths[:, None]
        return torch.where(keep, chosen, -1).to(torch.int32).contiguous(), lengths

    swa_idx, swa_len = lists(swa_width, swa_pages * swa_page, 1)
    plan = SparseMlaV41Plan(
        heads, swa_width, idx_width, idx_page, swa_page_size=swa_page,
        max_rows=rows, decode_max_rows=rows, split_max_rows=rows,
    )  # fmt: skip
    indexed, ref_indexed = {}, {}
    if idx_width:
        idx = _pack_idx(
            torch.randn((idx_pages * idx_page, HEAD_DIM), generator=gen, device=DEVICE)
        )
        idx_logical, idx_len = lists(idx_width, idx_context, 0)
        table = torch.stack(
            [
                torch.randperm(idx_pages, generator=gen, device=DEVICE)[
                    : idx_context // idx_page
                ]
                for _ in range(rows)
            ]
        ).to(torch.int32)
        pages = table.long().gather(1, idx_logical.long().clamp_min(0) // idx_page)
        idx_slots = torch.where(
            idx_logical >= 0, pages * idx_page + idx_logical.long() % idx_page, -1
        )
        indexed = dict(
            idx_cache=_paged(idx, idx_page),
            idx_indices=idx_logical,
            idx_lengths=idx_len,
            idx_page_table=table,
        )
        ref_indexed = dict(idx=idx, idx_slots=idx_slots, idx_len=idx_len)
    swa_pool = _paged(swa, swa_page)
    outs = {}
    for kernel in plan.kernels:
        out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        plan.run(q, swa_pool, swa_idx, swa_len, sinks, out, kernel=kernel, **indexed)
        outs[kernel] = out
    ref = _ref_mla(q, swa, swa_idx, swa_len, sinks, **ref_indexed)
    torch.testing.assert_close(
        outs["extend"].float(), ref.float(), rtol=2e-2, atol=2e-2
    )
    for kernel, out in outs.items():
        assert torch.equal(out, outs["extend"]), f"{kernel} differs from extend"


# Heads per rank: TP4 (16), TP3 (24 of 72 padded).
@pytest.mark.parametrize("heads", [16, 24])
@pytest.mark.parametrize("keys", [1, 2, 3, 8])
def test_sparse_mla_routes_short_rows(heads, keys, rows=1024, swa_width=128, page=128):
    """A prompt's first rows attend to a few keys. On the SWA-only layer (one segment)
    extend's fold starts from the constant state; every route must round alike (FOLD)."""
    gen = _gen(keys)
    swa_pages = 24
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    sinks = torch.randn((heads,), generator=gen, device=DEVICE)
    swa = _pack_swa(
        torch.randn((swa_pages * page, HEAD_DIM), generator=gen, device=DEVICE)
    )
    swa_idx = torch.full((rows, swa_width), -1, dtype=torch.int32, device=DEVICE)
    swa_idx[:, :keys] = torch.randint(
        0, swa_pages * page, (rows, keys), generator=gen, device=DEVICE, dtype=torch.int32
    )
    swa_len = torch.full((rows,), keys, dtype=torch.int32, device=DEVICE)
    plan = SparseMlaV41Plan(
        heads, swa_width, 0, page, swa_page_size=page,
        max_rows=rows, decode_max_rows=rows, split_max_rows=rows,
    )  # fmt: skip
    pool = _paged(swa, page)
    outs = {}
    for kernel in plan.kernels:
        out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        plan.run(q, pool, swa_idx, swa_len, sinks, out, kernel=kernel)
        outs[kernel] = out
    for kernel, out in outs.items():
        differ = int((out.view(torch.int16) != outs["extend"].view(torch.int16)).sum())
        assert differ == 0, f"{kernel} differs from extend in {differ} elements"
