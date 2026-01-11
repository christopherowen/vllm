# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 quantization utilities for Marlin kernel.

This module provides MXFP4 quantization for use with the Marlin FP4 kernel.
Uses FlashInfer's fp4_quantize with linear (non-swizzled) scale layout
to match Marlin's expected format.

MXFP4 format:
- 4-bit E2M1 values packed as uint8 (2 values per byte)
- E8M0 block scales (1 byte per 32 elements)
- Block size: 32 (OCP MX format)
- Linear scale layout (required for Marlin compatibility)
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def mxfp4_e2m1_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to MXFP4 format for Marlin kernel.
    
    Uses fp4_quantize with linear (non-swizzled) scale layout for
    compatibility with Marlin's prepare_fp4_layer_for_marlin().
    
    Note: FlashInfer's mxfp4_quantize uses swizzled layout which is
    incompatible with Marlin's scale processing.
    
    Args:
        x: Input tensor of shape [N, K] with dtype fp16/bf16.
        
    Returns:
        Tuple of:
            - Quantized tensor of shape [N, K/2] with dtype uint8 (packed FP4)
            - Scale factors tensor of shape [N, K/32] with dtype uint8 (E8M0)
    """
    try:
        from flashinfer import fp4_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP4 quantization. Please install it with "
            "`pip install flashinfer`"
        ) from err

    # Calculate global scale factor (same as mxfp4_quantize)
    # 448 * 6 = 2688 is the max representable value in E2M1 (6) * E8M0 max (448)
    global_scale = (448 * 6) / x.float().abs().nan_to_num().max()
    
    # Use fp4_quantize with:
    # - sf_vec_size=32: MXFP4 block size
    # - sf_use_ue8m0=True: E8M0 scale format
    # - is_sf_swizzled_layout=False: Linear layout for Marlin compatibility
    x_q, x_scales = fp4_quantize(
        x.cuda(),
        global_scale.cuda(),
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,  # Critical: Marlin expects linear layout
    )
    
    # Ensure scales have proper 2D shape [N, K/32]
    if x_scales.ndim == 1:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


