# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

logger = init_logger(__name__)


def _swizzle_mxfp4(quant_tensor, scale, num_warps):
    """weight swizzle for mxfp4 moe, used for OAI mxfp4 kernel"""
    assert has_triton_kernels()
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.tensor_details.layout import StridedLayout

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}

    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability(90)
        and not is_torch_equal_or_newer("2.8.1")
    ):
        logger.warning_once(
            "Mxfp4 on hopper is running on torch < 2.8.1, "
            "this cause swizling to be disabled, which may "
            "cause performance degradation. Please upgrade to torch nightly"
        )
        value_layout = StridedLayout
        scale_layout = StridedLayout
    elif current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        value_layout = StridedLayout
        if on_gfx950():
            from triton_kernels.tensor_details.layout import GFX950MXScaleLayout

            scale_layout = GFX950MXScaleLayout
        else:
            scale_layout = StridedLayout
    else:
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
            mx_axis=1
        )
        scale_layout, scale_layout_opts = (
            layout.make_default_matmul_mxfp4_w_scale_layout(
                mx_axis=1, num_warps=num_warps
            )
        )
    if current_platform.is_cuda():
        if current_platform.is_device_capability(90):
            constraints = {
                "split_k": 1,
            }
            opt_flags.update_opt_flags_constraints(constraints)
        elif current_platform.is_blackwell_class():
            constraints = {
                "is_persistent": True,
                "epilogue_subtile": 1,
            }
            opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return quant_tensor, InFlexData(), scale


def _can_support_mxfp4(
    use_grouped_topk: bool = False,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    expert_map: torch.Tensor | None = None,
    custom_routing_function: Callable | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    scoring_func: str = "softmax",
    activation: str = "swigluoai",
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
):
    return not (
        use_grouped_topk
        or topk_group
        or num_expert_group
        or custom_routing_function
        or e_score_correction_bias
        or apply_router_weight_on_input
        or scoring_func != "softmax"
        or activation != "swigluoai"
        or expert_load_view
        or logical_to_physical_map
        or logical_replica_count
    )


def get_padding_alignment():
    return (
        256
        if triton.runtime.driver.active.get_current_target().arch in ("gfx950",)
        else 128
    )


def _dequant_mxfp4(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def _dequant_mxfp4_fake(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2), dtype=float_dtype, device=x.device
    )


def _quant_dequant_mxfp4(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.qdq_mxfp4(x, scale_calculation_mode)


def _quant_dequant_mxfp4_fake(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    return torch.empty_like(x)


# Protect these operations into a torch custom op to avoid errors as
# torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
# Explanation: Dynamo does not know how to trace the builtin
# `kernel_ext.PyCapsule.dq_uint8_mxfp4_to_half.` This function is either a
# Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python
# extension (perhaps created with pybind).
# TODO: Make sure there is no way to avoid having these functions
# marked as skipped by dynamo.
try:
    direct_register_custom_op(
        op_name="dequant_mxfp4",
        op_func=_dequant_mxfp4,
        fake_impl=_dequant_mxfp4_fake,
    )
    dequant_mxfp4 = torch.ops.vllm.dequant_mxfp4
except AttributeError as error:
    raise error

try:
    direct_register_custom_op(
        op_name="quant_dequant_mxfp4",
        op_func=_quant_dequant_mxfp4,
        fake_impl=_quant_dequant_mxfp4_fake,
    )
    quant_dequant_mxfp4 = torch.ops.vllm.quant_dequant_mxfp4
except AttributeError as error:
    raise error


# -----------------------------------------------------------------------------
# FlashInfer-based MXFP4 quantization utilities
# -----------------------------------------------------------------------------
# These functions provide real MXFP4 quantization using FlashInfer,
# as opposed to the AMD quark-based functions above.
#
# MXFP4 format (OCP MX specification):
# - 4-bit E2M1 values packed as uint8 (2 values per byte)
# - E8M0 block scales (1 byte per 32 elements)
# - Block size: 32


def mxfp4_e2m1_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to MXFP4 format using FlashInfer.

    Uses non-swizzled scale layout for compatibility with mxfp4_dequantize_host.

    Args:
        x: Input tensor of shape [M, K] with dtype fp16/bf16.

    Returns:
        Tuple of:
            - Quantized tensor of shape [M, K/2] with dtype uint8 (packed FP4)
            - Scale factors tensor of shape [M, K/32] with dtype uint8 (E8M0)
    """
    try:
        from flashinfer import fp4_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP4 quantization. Please install it with "
            "`pip install flashinfer`"
        ) from err

    # Calculate global scale like mxfp4_quantize does
    a_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
    
    # Use fp4_quantize with:
    # - sf_vec_size=32 (MXFP4 block size)
    # - sf_use_ue8m0=True (E8M0 scale format)
    # - is_sf_swizzled_layout=False (linear layout for dequantize_host compatibility)
    x_q, x_scales = fp4_quantize(
        x.cuda(),
        a_global_sf.cuda(),
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,
    )
    
    # Ensure scales have proper 2D shape [M, K/32]
    if x_scales.ndim == 1:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def mxfp4_e2m1_dequantize(
    x_q: torch.Tensor,
    x_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a tensor from MXFP4 format using FlashInfer.

    Uses mxfp4_dequantize_host which handles the scale format correctly
    for non-swizzled linear layout scales.

    Args:
        x_q: Quantized tensor of shape [M, K/2] with dtype uint8.
        x_scales: Scale factors tensor of shape [M, K/32] with dtype uint8.
        out_dtype: Output dtype (bf16 or fp16).

    Returns:
        Dequantized tensor of shape [M, K] with dtype out_dtype.
    """
    try:
        from flashinfer import mxfp4_dequantize_host
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP4 dequantization. Please install it with "
            "`pip install flashinfer`"
        ) from err

    # mxfp4_dequantize_host handles the scale format correctly
    # Note: scale tensor should be in linear layout [M, K/32]
    orig_device = x_q.device
    result = mxfp4_dequantize_host(x_q.cpu(), x_scales.cpu(), group_size=32)
    return result.to(out_dtype).to(orig_device)
