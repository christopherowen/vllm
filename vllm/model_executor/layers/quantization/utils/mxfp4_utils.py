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
        elif current_platform.is_device_capability_family(100):
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


def mxfp4_e2m1_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to MXFP4 format for Marlin kernel.
    
    Uses fp4_quantize with linear (non-swizzled) scale layout for
    compatibility with Marlin's prepare_fp4_layer_for_marlin().
    
    Note: FlashInfer's mxfp4_quantize uses swizzled layout which is
    incompatible with Marlin's scale processing.
    
    Args:
        x: Input tensor of shape [N, K] with dtype fp16/bf16.
            Must already be on the target CUDA device.
        
    Returns:
        Tuple of:
            - Quantized tensor of shape [N, K/2] with dtype uint8 (packed FP4)
            - Scale factors tensor of shape [N, K/32] with dtype uint8 (E8M0)
            
    Note:
        Output tensors are on the same device as input tensor x.
    """
    try:
        from flashinfer import fp4_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP4 quantization. Please install it with "
            "`pip install flashinfer`"
        ) from err

    # Preserve input device for multi-GPU correctness
    device = x.device
    
    # Calculate global scale factor (same as mxfp4_quantize)
    # 448 * 6 = 2688 is the max representable value in E2M1 (6) * E8M0 max (448)
    #
    # Optimized path: compute max in native dtype (BF16/FP16) to avoid full FP32 copy
    # Only convert the scalar result to FP32 for the division.
    # For typical lm_head weights (vocab_size × hidden_dim), this saves ~2x memory
    # and reduces startup latency.
    max_val = x.abs().max()  # Native dtype reduction, no copy
    
    # Handle pathological cases (NaN/Inf in weights - shouldn't happen for valid models)
    if not torch.isfinite(max_val):
        # Fallback: use nan_to_num to handle NaN/Inf (creates FP32 copy, but rare)
        max_val = x.float().abs().nan_to_num().max()
    
    # Convert scalar to FP32 for division (single value, not full tensor)
    max_val_f32 = max_val.float()
    
    # Use epsilon large enough that (448*6)/eps stays within float32 range
    # float32 max ≈ 3.4e38, so eps should be > 2688/3.4e38 ≈ 8e-36
    # Use 1e-30 as a safe practical floor (any weight this small is effectively zero)
    eps = 1e-30
    max_val_clamped = torch.clamp(max_val_f32, min=eps)
    global_scale = (448 * 6) / max_val_clamped
    
    # Ensure global_scale is a proper tensor on the correct device
    if not isinstance(global_scale, torch.Tensor):
        global_scale = torch.tensor(global_scale, dtype=torch.float32, device=device)
    else:
        global_scale = global_scale.to(device=device, dtype=torch.float32)
    
    # Use fp4_quantize with:
    # - sf_vec_size=32: MXFP4 block size
    # - sf_use_ue8m0=True: E8M0 scale format
    # - is_sf_swizzled_layout=False: Linear layout for Marlin compatibility
    x_q, x_scales = fp4_quantize(
        x,  # Already on correct device
        global_scale,  # Already on correct device
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,  # Critical: Marlin expects linear layout
    )
    
    # Ensure scales have proper 2D shape [N, K/32]
    if x_scales.ndim == 1:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales
