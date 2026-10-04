# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 expert routing with DeepSeek's TileLang top-k gate.

The TileLang kernel family routes both the target's MoE layers (6 of 384
experts) and the drafter's (3 of 128) with TileKernels'
``moe_topk_gate_forward``, in place of vLLM's Triton ``dsv4_topk`` and its
CUDA ``topk_hash_softplus_sqrt`` op. It implements DeepSeek's V4.1 reference
routing:

- scores ``sqrt(softplus(logits))``, with softplus the identity above 20;
- the top-k experts by score plus the correction bias, or the vision bias for
  image tokens, the lowest expert id first on ties;
- weights the chosen scores without bias, divided by their sum plus 1e-20 and
  multiplied by the routed scaling factor.

One warp routes one token. The kernel refuses NaN or infinite logits instead
of routing on them. V4.1 has no hash-routed layers.

``TileLangGateRouter`` also owns the gate projection. At decode the gate runs
as split-K FP32 partials, and ``route_partials`` adds them in shard order,
which reproduces ``splitk_reduce``'s logits bit for bit. It then routes them
with TileKernels' arithmetic in the same launch, so the logits never reach
memory and each MoE layer runs one launch fewer. At prefill the gate runs its
own GEMM and TileKernels routes the logits.
"""

import tilelang
import tilelang.language as T
import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)


def _image_token_mask(input_ids: torch.Tensor, lo: int, count: int) -> torch.Tensor:
    """Image sentinel tokens of this step, computed once and shared by every layer."""
    context = get_forward_context() if is_forward_context_available() else None
    cached = getattr(context, "_ds41_image_token_mask", None)
    if cached is not None and cached[0] is input_ids:
        return cached[1]
    mask = (input_ids >= lo) & (input_ids < lo + count)
    if context is not None:
        context._ds41_image_token_mask = (input_ids, mask)
    return mask


class TileKernelsRouter(FusedTopKBiasRouter):
    """V4.1 routing through TileKernels' TileLang top-k gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._hash_indices_table is not None:
            raise ValueError("DeepSeek V4.1 routing has no hash-routed layers")
        if self.scoring_func != "sqrtsoftplus" or not self.renormalize:
            raise ValueError("DeepSeek V4.1 routing is normalized sqrtsoftplus")
        if self.e_score_correction_bias is None:
            raise ValueError("DeepSeek V4.1 routing needs the correction bias")
        if self.num_fused_shared_experts or self.eplb_state is not None:
            raise ValueError(
                "TileKernels routing covers routed experts without EPLB only"
            )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tile_kernels.moe import moe_topk_gate_forward

        image_bias = image_token_mask = None
        if self.bias_vl is not None and self.image_sentinel_lo > 0:
            if input_ids is None:
                raise ValueError("DeepSeek V4.1 vision routing requires input_ids")
            image_bias = self.bias_vl.data
            image_token_mask = _image_token_mask(
                input_ids, self.image_sentinel_lo, self.image_sentinel_count
            )
        topk_ids, topk_weights = moe_topk_gate_forward(
            router_logits,
            self.top_k,
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=self.routed_scaling_factor,
            ep_rank=0,
            bias=self.e_score_correction_bias.data,
            image_bias=image_bias,
            image_token_mask=image_token_mask,
        )
        # TileKernels writes int64 expert ids; the TileLang dispatch reads them.
        if indices_type not in (None, torch.int64):
            raise ValueError(
                f"TileKernels routing writes int64 expert ids, not {indices_type}"
            )
        return topk_weights, topk_ids


_WARP = 32
_VECTOR = 4


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def route_partials(
    num_topk: int, num_experts: int, shards: int, image_mask: bool, warps: int = 2
):
    """Sum a gate's split-K partials in shard order and route each token.

    The routing is TileKernels' ``moe_topk_gate_forward`` for sqrtsoftplus
    with a correction bias: the same scores, selection order and weights. Its
    inputs are the partials ``P[shards, rows, experts]``, added from shard 0
    as ``splitk_reduce`` adds them. Each round's warp-wide winner takes two
    integer reductions instead of TileKernels' ten shuffles. Non-finite
    logits are refused once per lane, not once per logit.
    """
    assert num_topk <= _WARP and num_experts % _VECTOR == 0
    per_lane = tilelang.cdiv(tilelang.cdiv(num_experts, _WARP), _VECTOR) * _VECTOR
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        P: T.Tensor((shards, rows, num_experts), T.float32),
        bias: T.Tensor((num_experts,), T.float32),
        image_bias: T.Tensor((num_experts,), T.float32),
        image_token_mask: T.Tensor((rows,), T.bool),
        topk_idx: T.Tensor((rows, num_topk), T.int64),
        topk_weights: T.Tensor((rows, num_topk), T.float32),
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(T.ceildiv(rows, warps), threads=warps * _WARP) as pid:
            thread = T.get_thread_binding()
            token = thread // _WARP
            row = token + pid * warps
            lane = thread % _WARP
            if row >= rows:
                T.thread_return()

            unbiased = T.alloc_shared((warps, num_experts), T.float32)
            bias_local = T.alloc_local((per_lane,), T.float32)
            scores = T.alloc_local((per_lane,), T.float32)
            ids = T.alloc_local((per_lane,), T.int32)
            finite = T.alloc_var(T.bool, init=True)
            best_score = T.alloc_var(T.float32)
            best_idx = T.alloc_var(T.int32, init=-1)
            previous = T.alloc_var(T.int32, init=-1)
            chosen_idx = T.alloc_var(T.int64)
            chosen_score = T.alloc_var(T.float32)
            total = T.alloc_var(T.float32)
            is_image = T.alloc_var(T.bool)
            is_image = image_token_mask[row] if image_mask else False

            T.fill(ids, -1)
            T.fill(scores, -T.infinity(T.float32))
            T.fill(bias_local, 0.0)
            for i in T.unroll(0, per_lane // _VECTOR):
                start = i * _VECTOR * _WARP + lane * _VECTOR
                if start < num_experts:
                    for j in T.vectorized(_VECTOR):
                        scores[i * _VECTOR + j] = P[0, row, start + j]
                        bias_local[i * _VECTOR + j] = (
                            image_bias[start + j] if is_image else bias[start + j]
                        )
                    for s in T.serial(1, shards):
                        for j in T.vectorized(_VECTOR):
                            scores[i * _VECTOR + j] += P[s, row, start + j]
                    for j in T.unroll(_VECTOR):
                        if not T.isfinite(scores[i * _VECTOR + j]):
                            finite = False
            T.device_assert(
                finite, msg="route_partials: gate logits contain nan or inf"
            )

            for i in T.unroll(0, T.ceildiv(num_experts, _VECTOR * _WARP)):
                start = i * _VECTOR * _WARP + lane * _VECTOR
                if start < num_experts:
                    for j in T.unroll(_VECTOR):
                        x = scores[i * _VECTOR + j]
                        scores[i * _VECTOR + j] = T.sqrt(
                            T.Select(x > 20.0, x, T.log1p(T.exp(x)))
                        )
                    for j in T.vectorized(_VECTOR):
                        unbiased[token, start + j] = scores[i * _VECTOR + j]
                        scores[i * _VECTOR + j] += bias_local[i * _VECTOR + j]
                        ids[i * _VECTOR + j] = start + j
            T.sync_warp()

            for k in T.unroll(num_topk):
                best_score = -T.infinity(T.float32)
                best_idx = -1
                for i in T.unroll(0, per_lane):
                    if k != 0 and previous == ids[i]:
                        scores[i] = -T.infinity(T.float32)
                    elif scores[i] > best_score:
                        best_score = scores[i]
                        best_idx = ids[i]
                # The winner across the warp in two hardware reductions: the
                # largest score as an integer that orders like the float, then
                # the lowest expert id among the lanes holding it.
                bits = T.reinterpret(best_score, T.int32)
                key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
                top = T.warp_reduce_max(key)
                best_idx = T.warp_reduce_min(
                    T.if_then_else(key == top, best_idx, 0x7FFFFFFF)
                )
                previous = best_idx
                if lane == k:
                    chosen_idx = best_idx

            chosen_score = 0.0
            if lane < num_topk:
                chosen_score = unbiased[token, chosen_idx]
            total = chosen_score
            for i in T.unroll(0, 5):
                total += T.shfl_xor(total, 1 << (4 - i))
            total += 1e-20
            if lane < num_topk:
                topk_idx[row, lane] = chosen_idx
                topk_weights[row, lane] = chosen_score / total * routed_scaling_factor

    return main


class TileLangGateRouter(TileKernelsRouter):
    """TileKernelsRouter that also runs the gate, fusing its decode reduction.

    The fused-MoE runner hands it the hidden states in place of router
    logits (the MoE layer gives it the gate instead of the runner).
    """

    routes_hidden_states = True

    def __init__(self, *args, gate: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate
        self._kernels: dict[bool, object] = {}
        # Text-only layers pass no image tokens; one all-false mask serves them.
        self._no_image = None

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .linear import _partial_specs, _scratch, partials_kernel

        gate = self.gate
        x = router_logits.reshape(-1, gate.weight.shape[1])
        specs = _partial_specs(gate, x.shape[0])
        if not specs:
            logits, _ = gate(x)
            return super()._compute_routing(
                hidden_states, logits, indices_type, input_ids=input_ids
            )
        if indices_type not in (None, torch.int64):
            raise ValueError(
                f"TileLang routing writes int64 expert ids, not {indices_type}"
            )
        (partials,) = _scratch(specs, None)
        partials_kernel(gate, x.shape[0])(x, gate.weight, partials)
        image = self.bias_vl is not None and self.image_sentinel_lo > 0
        if image and input_ids is None:
            raise ValueError("DeepSeek V4.1 vision routing requires input_ids")
        if image not in self._kernels:
            self._kernels[image] = route_partials(
                self.top_k, gate.weight.shape[0], gate.tilelang_shards, image
            )
        rows = x.shape[0]
        topk_ids = torch.empty(rows, self.top_k, dtype=torch.int64, device=x.device)
        topk_weights = torch.empty(
            rows, self.top_k, dtype=torch.float32, device=x.device
        )
        bias = self.e_score_correction_bias.data
        if image:
            image_bias = self.bias_vl.data
            mask = _image_token_mask(
                input_ids, self.image_sentinel_lo, self.image_sentinel_count
            )
        else:
            if self._no_image is None or self._no_image.device != x.device:
                from .linear import DECODE_ROWS

                self._no_image = torch.zeros(
                    DECODE_ROWS, dtype=torch.bool, device=x.device
                )
            image_bias, mask = bias, self._no_image[:rows]
        self._kernels[image](
            partials,
            bias,
            image_bias,
            mask,
            topk_ids,
            topk_weights,
            self.routed_scaling_factor,
        )
        return topk_weights, topk_ids
