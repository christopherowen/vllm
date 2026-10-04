# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 routing through TileKernels' TileLang top-k gate."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("tile_kernels")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm.models.deepseek_v4_1.tilelang import router as tilelang_router  # noqa: E402

LO = 129264  # the target's image sentinel token
SCALE = 1.5  # V4.1's routed scaling factor


def _reference(logits, bias, topk, image=None, bias_vl=None):
    """DeepSeek's V4.1 reference gate (inference/model.py), in fp32."""
    scores = F.softplus(logits).sqrt()
    biased = scores + (
        torch.where(image[:, None], bias_vl, bias) if image is not None else bias
    )
    # A stable descending sort puts the lower expert id first on exact ties,
    # which torch.topk leaves unspecified.
    ids = biased.sort(dim=-1, descending=True, stable=True).indices[:, :topk]
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * SCALE
    return weights, ids


def _router(experts, topk, bias, bias_vl=None, lo=0):
    return tilelang_router.TileKernelsRouter(
        top_k=topk,
        global_num_experts=experts,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
        bias_vl=bias_vl,
        image_sentinel_lo=lo,
        image_sentinel_count=1,
    )


@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
@pytest.mark.parametrize("rows", [1, 6, 96, 8192])
@pytest.mark.parametrize("scale", [0.1, 4.0, 40.0])
def test_routes_like_deepseeks_reference(experts, topk, rows, scale):
    generator = torch.Generator(device="cuda").manual_seed(rows * experts)
    logits = torch.randn(rows, experts, device="cuda", generator=generator) * scale
    bias = torch.randn(experts, device="cuda", generator=generator) * 0.1
    weights, ids = _router(experts, topk, bias)._compute_routing(None, logits, None)
    expected_weights, expected_ids = _reference(logits, bias, topk)
    assert ids.dtype == torch.int64 and weights.dtype == torch.float32
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    # Only the order of the six-term sum may differ from torch's.
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=0)


def test_ties_go_to_the_lowest_expert():
    logits = torch.zeros(4, 384, device="cuda")
    bias = torch.zeros(384, device="cuda")
    weights, ids = _router(384, 6, bias)._compute_routing(None, logits, None)
    assert ids.tolist() == [list(range(6))] * 4
    torch.testing.assert_close(weights, torch.full_like(weights, SCALE / 6))


def test_image_tokens_route_with_the_vision_bias():
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(64, 384, device="cuda", generator=generator)
    bias = torch.randn(384, device="cuda", generator=generator)
    bias_vl = torch.randn(384, device="cuda", generator=generator)
    input_ids = torch.randint(0, 1000, (64,), device="cuda", generator=generator)
    input_ids[::3] = LO
    router = _router(
        384, 6, bias, torch.nn.Parameter(bias_vl, requires_grad=False), lo=LO
    )
    weights, ids = router._compute_routing(None, logits, None, input_ids=input_ids)
    expected_weights, expected_ids = _reference(
        logits, bias, 6, input_ids == LO, bias_vl
    )
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=0)
    with pytest.raises(ValueError, match="requires input_ids"):
        router._compute_routing(None, logits, None)


def test_image_token_mask_is_computed_once_per_step(monkeypatch):
    context = type("Context", (), {})()
    monkeypatch.setattr(tilelang_router, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(tilelang_router, "get_forward_context", lambda: context)
    input_ids = torch.tensor([1, LO, LO + 1, LO], device="cuda")
    first = tilelang_router._image_token_mask(input_ids, LO, 1)
    assert first.tolist() == [False, True, False, True]
    assert tilelang_router._image_token_mask(input_ids, LO, 1) is first
    # The next step's ids are a new tensor.
    other = tilelang_router._image_token_mask(input_ids.clone(), LO, 1)
    assert other is not first and torch.equal(other, first)


def test_refuses_what_v41_routing_does_not_have():
    bias = torch.zeros(384, device="cuda")
    table = torch.zeros(1000, 6, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="no hash-routed layers"):
        tilelang_router.TileKernelsRouter(
            top_k=6,
            global_num_experts=384,
            e_score_correction_bias=bias,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            hash_indices_table=table,
        )
    logits = torch.randn(2, 384, device="cuda")
    with pytest.raises(ValueError, match="int64 expert ids"):
        _router(384, 6, bias)._compute_routing(None, logits, torch.int32)


def test_matches_the_triton_router_it_replaces():
    """Same experts as vLLM's dsv4_topk; weights within its approximate math."""
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    generator = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn(512, 384, device="cuda", generator=generator) * 3
    bias = torch.randn(384, device="cuda", generator=generator) * 0.1
    weights, ids = _router(384, 6, bias)._compute_routing(None, logits, None)
    triton_weights, triton_ids = dsv4_topk(logits, bias, torch.int64, SCALE)
    torch.testing.assert_close(ids, triton_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, triton_weights, rtol=1e-5, atol=0)
