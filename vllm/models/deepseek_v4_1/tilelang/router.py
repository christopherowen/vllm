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
"""

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
