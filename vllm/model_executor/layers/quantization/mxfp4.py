# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.attention.layer import Attention
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    mxfp4_mxfp8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    BatchedMarlinExperts,
    MarlinExperts,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    OAITritonExperts,
    UnfusedOAITritonExperts,
)
from vllm.model_executor.layers.fused_moe.trtllm_moe import TrtLlmGenExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    _can_support_mxfp4,
    _swizzle_mxfp4,
    get_padding_alignment,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_sm12x_cutlass_moe
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import is_torch_equal_or_newer

logger = init_logger(__name__)


# Enum for MXFP4 backend selection
#
# Naming convention: ENGINE_ARCH_QUANT
#   ENGINE: Kernel technology (CUTLASS, TRTLLM, MARLIN, TRITON)
#   ARCH: GPU architecture (BLACKWELL=SM10x+SM12x, SM100, SM90, or omitted for universal)
#   QUANT: Quantization format (FP4FP8=FP4 weights×FP8 acts, FP4BF16=FP4 weights×BF16 acts)
#
class Mxfp4Backend(Enum):
    NONE = 0

    # Universal backends (all architectures)
    MARLIN = 5                        # Dequant to BF16, works everywhere
    TRITON = 6                        # OpenAI Triton, SM90-SM100

    # CUTLASS backends (FlashInfer dispatch)
    CUTLASS_BLACKWELL_FP4FP8 = 2      # SM10x + SM12x: FP8×FP4 native MMA
    CUTLASS_SM90_FP4BF16 = 4          # SM90: BF16×FP4 (opt-in)

    # TRT-LLM backends (SM100 only, FlashInfer dispatch)
    TRTLLM_SM100_FP4FP8 = 1           # FP8×FP4
    TRTLLM_SM100_FP4BF16 = 3          # BF16×FP4


def get_mxfp4_backend_with_lora() -> Mxfp4Backend:
    """
    Select LoRA-compatible MXFP4 backend.
    
    Not all MXFP4 backends support LoRA. Currently supported:
    - MARLIN: Full LoRA support
    - TRITON: Full LoRA support (SM90-SM100)
    
    Not yet supported:
    - CUTLASS_BLACKWELL_FP4FP8: LoRA not implemented
    - TRTLLM_*: LoRA not implemented
    """
    if not current_platform.is_cuda():
        return Mxfp4Backend.NONE

    # LoRA-compatible backends
    LORA_COMPATIBLE = {"marlin", "triton"}
    
    # Check if user explicitly requested a backend via VLLM_MXFP4_BACKEND
    explicit_backend = envs.VLLM_MXFP4_BACKEND
    if explicit_backend and explicit_backend != "auto":
        if explicit_backend in LORA_COMPATIBLE:
            logger.info_once(
                f"[MXFP4+LoRA] Using explicit backend: {explicit_backend}"
            )
            return MXFP4_BACKEND_MAP[explicit_backend]
        else:
            logger.warning_once(
                f"[MXFP4+LoRA] VLLM_MXFP4_BACKEND={explicit_backend} does not support LoRA. "
                f"Falling back to Marlin. LoRA-compatible backends: {', '.join(LORA_COMPATIBLE)}"
            )
            return Mxfp4Backend.MARLIN

    # Auto-select from LoRA-compatible backends
    triton_kernels_supported = (
        has_triton_kernels()
        and is_torch_equal_or_newer("2.8.0")
        # NOTE: triton_kernels are only confirmed to work on SM90 and SM100
        # SM110 fails with this error: https://github.com/vllm-project/vllm/issues/29317
        # SM120 needs this fix: https://github.com/triton-lang/triton/pull/8498
        and (9, 0) <= current_platform.get_device_capability() < (11, 0)
    )

    logger.info_once("[MXFP4+LoRA] Auto-selected: Marlin")
    return Mxfp4Backend.MARLIN


def _check_legacy_mxfp4_flags() -> Mxfp4Backend | None:
    """Check legacy MXFP4 env vars and return backend if set, with deprecation warnings."""
    import os as _os

    # Check legacy kernel override (only if explicitly set)
    kernel = _os.environ.get("VLLM_MXFP4_MOE_KERNEL", None)
    if kernel is not None and kernel.lower() != "auto":
        kernel = kernel.lower()

        logger.warning_once(
            f"[MXFP4] VLLM_MXFP4_MOE_KERNEL is deprecated. "
            f"Use VLLM_MXFP4_BACKEND instead. "
            f"Example: VLLM_MXFP4_BACKEND=CUTLASS"
        )
        if kernel in ("cutlass", "gemm"):
            return Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8
        elif kernel == "marlin":
            return Mxfp4Backend.MARLIN
        elif kernel == "triton":
            return Mxfp4Backend.TRITON

    # Check legacy boolean flags
    if _os.environ.get("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS") is not None and _os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS", "0") != "0":
        logger.warning_once(
            "[MXFP4] VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS is deprecated. "
            "Use VLLM_MXFP4_BACKEND=CUTLASS instead."
        )
        return Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8

    if _os.environ.get("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8") is not None and _os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "0") != "0":
        logger.warning_once(
            "[MXFP4] VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 is deprecated. "
            "Use VLLM_MXFP4_BACKEND=TRTLLM_MXFP8 instead."
        )
        return Mxfp4Backend.TRTLLM_SM100_FP4FP8

    if _os.environ.get("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16") is not None and _os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16", "0") != "0":
        logger.warning_once(
            "[MXFP4] VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 is deprecated. "
            "Use VLLM_MXFP4_BACKEND=TRTLLM instead."
        )
        return Mxfp4Backend.TRTLLM_SM100_FP4BF16

    if _os.environ.get("VLLM_MXFP4_USE_MARLIN") is not None and _os.getenv("VLLM_MXFP4_USE_MARLIN", "0") != "0":
        logger.warning_once(
            "[MXFP4] VLLM_MXFP4_USE_MARLIN is deprecated. "
            "Use VLLM_MXFP4_BACKEND=MARLIN instead."
        )
        return Mxfp4Backend.MARLIN

    return None


def _auto_select_mxfp4_backend() -> Mxfp4Backend:
    """Auto-select MXFP4 backend based on hardware capabilities."""
    if not current_platform.is_cuda():
        if current_platform.is_xpu():
            logger.info_once("[MXFP4] Auto-selected: Marlin (XPU)")
            return Mxfp4Backend.MARLIN
        elif current_platform.is_rocm() and has_triton_kernels():
            logger.info_once("[MXFP4] Auto-selected: Triton (ROCm)")
            return Mxfp4Backend.TRITON
        return Mxfp4Backend.NONE

    # SM12x (GB10, Thor): use CUTLASS FP8×FP4 for native block-scaled MMA
    # This uses a stronger predicate that verifies:
    # 1. FlashInfer is installed with CUTLASS MoE support
    # 2. SM12x-specific kernels are available
    # 3. Current GPU is SM12x (major=12)
    capability = current_platform.get_device_capability()
    if has_flashinfer_sm12x_cutlass_moe():
        logger.info_once(
            "[MXFP4] Auto-selected: cutlass (FlashInfer CUTLASS FP8×FP4 for SM12x)"
        )
        return Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8

    # SM100 (B100/B200): default to TRT-LLM BF16×FP4 (upstream behavior)
    # Can opt-in to CUTLASS FP8×FP4 for potentially better performance
    is_sm100 = capability is not None and capability.major == 10
    if is_sm100 and has_flashinfer():
        logger.info_once(
            "[MXFP4] SM100 auto-selected: trtllm (BF16×FP4). "
            "For CUTLASS FP8×FP4, set VLLM_MXFP4_BACKEND=CUTLASS"
        )
        return Mxfp4Backend.TRTLLM_SM100_FP4BF16

    # Hopper (SM90): prefer Triton or Marlin
    if current_platform.is_device_capability(90):
        if has_flashinfer():
            logger.warning_once(
                "[MXFP4] SM90 auto-selected: Marlin. "
                "For FlashInfer BF16, use VLLM_MXFP4_BACKEND=TRTLLM"
            )

    # Fallback: Triton or Marlin
    triton_kernels_supported = (
        has_triton_kernels()
        and is_torch_equal_or_newer("2.8.0")
        and (9, 0) <= current_platform.get_device_capability() < (11, 0)
    )
    if triton_kernels_supported:
        logger.info_once("[MXFP4] Auto-selected: Triton")
        return Mxfp4Backend.TRITON
    else:
        logger.info_once("[MXFP4] Auto-selected: Marlin (universal fallback)")
        return Mxfp4Backend.MARLIN


# Backend name to enum mapping
MXFP4_BACKEND_MAP: dict[str, Mxfp4Backend] = {
    # Primary names (UPPERCASE for consistency with VLLM_ATTENTION_BACKEND)
    "MARLIN": Mxfp4Backend.MARLIN,
    "CUTLASS": Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8,
    "TRITON": Mxfp4Backend.TRITON,
    "TRTLLM": Mxfp4Backend.TRTLLM_SM100_FP4BF16,
    "TRTLLM_MXFP8": Mxfp4Backend.TRTLLM_SM100_FP4FP8,
    # Lowercase aliases for backwards compatibility
    "marlin": Mxfp4Backend.MARLIN,
    "cutlass": Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8,
    "triton": Mxfp4Backend.TRITON,
    "trtllm": Mxfp4Backend.TRTLLM_SM100_FP4BF16,
    "trtllm-mxfp8": Mxfp4Backend.TRTLLM_SM100_FP4FP8,
    # Legacy aliases
    "gemm": Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8,
}


def get_mxfp4_backend(with_lora_support: bool) -> Mxfp4Backend:
    """Select the MXFP4 MoE backend.

    Configuration priority:
    1. CLI argument --mxfp4-backend (highest)
    2. VLLM_MXFP4_BACKEND env var
    3. Legacy env vars (with deprecation warnings)
    4. Hardware-based auto-selection

    Backend options:
    - marlin: Marlin dequant→BF16 (works on all GPUs)
    - cutlass: FlashInfer CUTLASS FP8×FP4 (SM12x/SM100 native)
    - triton: OpenAI Triton (SM90-SM100)
    - trtllm: TRT-LLM BF16×FP4 (SM100 only)
    - trtllm-mxfp8: TRT-LLM FP8×FP4 (SM100 only)

    For SM12x (GB10):
        vllm serve ... --mxfp4-backend cutlass
    """
    # Step 1: Check CLI argument (highest priority)
    try:
        vllm_config = get_current_vllm_config()
        cli_backend = vllm_config.model_config.mxfp4_backend
        if cli_backend and cli_backend != "auto":
            if cli_backend not in MXFP4_BACKEND_MAP:
                valid = ", ".join(sorted(set(MXFP4_BACKEND_MAP.keys()) - {"gemm"}))
                logger.warning_once(
                    f"[MXFP4] Unknown backend '{cli_backend}' from --mxfp4-backend. "
                    f"Valid options: auto, {valid}. Using auto."
                )
            else:
                backend = MXFP4_BACKEND_MAP[cli_backend]
                logger.info_once(
                    f"[MXFP4] Using backend: {cli_backend} (--mxfp4-backend)"
                )
                return backend
    except Exception:
        # Config not available yet, fall through to env var
        pass

    # Step 2: Check unified backend env var
    backend_name = envs.VLLM_MXFP4_BACKEND
    if backend_name != "auto":
        if backend_name not in MXFP4_BACKEND_MAP:
            valid = ", ".join(sorted(set(MXFP4_BACKEND_MAP.keys()) - {"gemm"}))
            logger.warning_once(
                f"[MXFP4] Unknown backend '{backend_name}'. "
                f"Valid options: auto, {valid}. Using auto."
            )
        else:
            backend = MXFP4_BACKEND_MAP[backend_name]
            logger.info_once(
                f"[MXFP4] Using backend: {backend_name} "
                f"(VLLM_MXFP4_BACKEND={backend_name})"
            )
            return backend

    # Step 2: Check legacy env vars (with deprecation warnings)
    legacy_backend = _check_legacy_mxfp4_flags()
    if legacy_backend is not None:
        return legacy_backend

    # Step 3: LoRA support requires specific backends
    if with_lora_support:
        return get_mxfp4_backend_with_lora()

    # Step 4: Auto-select based on hardware
    return _auto_select_mxfp4_backend()


class Mxfp4Config(QuantizationConfig):
    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            # TODO: Add support for MXFP4 Linear Method.
            # MXFP4 LinearMethod is available in AMD-Quark, refer to that implementation
            # if you are interested in enabling MXFP4 here.
            logger.debug_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
                scope="local",
            )
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            if current_platform.is_xpu():
                return IpexMxfp4MoEMethod(layer.moe_config)
            else:
                quant_method = Mxfp4MoEMethod(layer.moe_config)
                quant_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return quant_method
        elif isinstance(layer, Attention):
            # TODO: Add support for MXFP4 Attention.
            logger.debug_once(
                "MXFP4 attention layer is not implemented. "
                "Skipping quantization for this layer.",
                scope="local",
            )
        return None


class Mxfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.mxfp4_backend = get_mxfp4_backend(moe.is_lora_enabled)

        self.marlin_input_dtype = None
        self.max_capture_size = (
            get_current_vllm_config().compilation_config.max_cudagraph_capture_size
        )

        assert self.mxfp4_backend != Mxfp4Backend.NONE, (
            f"get_mxfp4_backend(with_lora_support={moe.is_lora_enabled}) found"
            "no compatible MXFP4 MoE backend (FlashInfer/Marlin/Triton)."
            "Please check your environment and try again."
        )
        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # FIXME (zyongye): ship after torch and safetensors support mxfp4
        # is_torch_mxfp4_available = (
        #     hasattr(torch, "float4_e2m1fn_x2") and
        #     hasattr(torch, "float8_e8m0fnu"))
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = intermediate_size_per_partition
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            # The moe marlin kernel requires that for each linear
            # n % 256 == 0 and k % 128 == 0.
            # In gate_up_proj:
            #    n = 2 * intermediate_size_per_partition_after_pad
            #    k = hidden_size
            # In down_proj
            #    n = hidden_size
            #    k = intermediate_size_per_partition_after_pad
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 128
            )
            if current_platform.is_xpu():
                hidden_size = round_up(hidden_size, 128)
            else:
                hidden_size = round_up(hidden_size, 256)

            layer.params_dtype = params_dtype
            layer.num_experts = num_experts
            layer.hidden_size = hidden_size
            layer.intermediate_size_per_partition = (
                intermediate_size_per_partition_after_pad
            )
        elif (
            self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4FP8
            or self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4BF16
        ):
            # pad the intermediate size to be a multiple of 2 * mxfp4_block
            # for to hold non-uniform sharded tensor as well as swizzling
            # other padding to increase performance
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 256
            )
            hidden_size = round_up(hidden_size, 256)
        elif (
            self.mxfp4_backend == Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8
            or self.mxfp4_backend == Mxfp4Backend.CUTLASS_SM90_FP4BF16
        ):
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 128
            )
            hidden_size = round_up(hidden_size, 128)
        elif current_platform.is_rocm():
            pad_align = get_padding_alignment()
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, pad_align
            )
            hidden_size = round_up(hidden_size, pad_align)
        else:
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 64
            )

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            prepare_moe_fp4_layer_for_marlin(layer, input_dtype=self.marlin_input_dtype)
        elif (
            self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4FP8
            or self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4BF16
        ):
            from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
            from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

            layer.gemm1_alpha = Parameter(
                torch.tensor([1.702] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )
            layer.gemm1_beta = Parameter(
                torch.tensor([1.0] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )
            layer.gemm1_clamp_limit = Parameter(
                torch.tensor([7.0] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )
            sf_block_size = 32  # mxfp4 block size

            assert (
                layer.w13_weight.dim() == 3
                and layer.w13_weight.shape[0] == self.num_experts
                and layer.w13_weight.shape[1] == self.intermediate_size * 2
                and layer.w13_weight.shape[2] == self.hidden_size // 2
            )
            assert (
                layer.w13_weight_scale.dim() == 3
                and layer.w13_weight_scale.shape[0] == self.num_experts
                and layer.w13_weight_scale.shape[1] == self.intermediate_size * 2
                and layer.w13_weight_scale.shape[2] == self.hidden_size // sf_block_size
            )
            assert (
                layer.w2_weight.dim() == 3
                and layer.w2_weight.shape[0] == self.num_experts
                and layer.w2_weight.shape[1] == self.hidden_size
                and layer.w2_weight.shape[2] == self.intermediate_size // 2
            )
            assert (
                layer.w2_weight_scale.dim() == 3
                and layer.w2_weight_scale.shape[1] == self.hidden_size
                and layer.w2_weight_scale.shape[2]
                == self.intermediate_size // sf_block_size
            )
            assert (
                layer.w13_bias.dim() == 2
                and layer.w13_bias.shape[0] == self.num_experts
                and layer.w13_bias.shape[1] == self.intermediate_size * 2
            )
            assert (
                layer.w2_bias.dim() == 2
                and layer.w2_bias.shape[0] == self.num_experts
                and layer.w2_bias.shape[1] == self.hidden_size
            )

            w13_weight_scale = layer.w13_weight_scale.data
            w2_weight_scale = layer.w2_weight_scale.data
            w13_weight = layer.w13_weight.data
            w2_weight = layer.w2_weight.data
            w13_bias = layer.w13_bias.data.to(torch.float32)
            w2_bias = layer.w2_bias.data.to(torch.float32)

            # Swap w1 and w3 as the definition of
            # swiglu is different in the trtllm-gen
            def swap_every_two_rows(x, axis=-1):
                shape = x.shape
                if axis < 0:
                    axis = len(shape) + axis

                # Create a new shape with pairs swapped along specified axis
                new_shape = list(shape)
                new_shape[axis] = shape[axis] // 2
                new_shape.insert(axis + 1, 2)

                # Reshape to expose pairs, swap them, and reshape back
                x = x.reshape(*new_shape)
                x = x.flip(axis + 1)
                new_shape = list(shape)
                return x.reshape(*new_shape)

            w13_weight_scale = swap_every_two_rows(w13_weight_scale, -2)
            w13_weight = swap_every_two_rows(w13_weight, -2)
            w13_bias = swap_every_two_rows(w13_bias, -1)

            # Do not interleave as the checkpoint is already interleaved

            # Shuffle weights and scaling factors for transposed mma output
            gemm1_weights_mxfp4_shuffled = []
            gemm1_scales_mxfp4_shuffled = []
            gemm2_weights_mxfp4_shuffled = []
            gemm2_scales_mxfp4_shuffled = []
            gemm1_bias_shuffled = []
            gemm2_bias_shuffled = []
            epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
            for i in range(self.num_experts):
                # w13 weight shuffling
                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    w13_weight[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                gemm1_weights_mxfp4_shuffled.append(
                    w13_weight[i]
                    .view(torch.uint8)[permute_indices.to(w13_weight.device)]
                    .contiguous()
                )
                # w13 scale shuffling
                permute_sf_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    w13_weight_scale[i].view(torch.uint8),
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                gemm1_scales_mxfp4_shuffled.append(
                    nvfp4_block_scale_interleave(
                        w13_weight_scale[i]
                        .view(torch.uint8)[
                            permute_sf_indices.to(w13_weight_scale.device)
                        ]
                        .contiguous()
                    )
                )
                # w13 bias shuffling
                permute_bias_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    w13_bias[i].clone().reshape(-1, 1),
                    epilogue_tile_m,
                )
                gemm1_bias_shuffled.append(
                    w13_bias[i]
                    .clone()
                    .reshape(-1, 1)[permute_bias_indices.to(w13_bias.device)]
                    .contiguous()
                )
                # w2 weight shuffling
                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    w2_weight[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                gemm2_weights_mxfp4_shuffled.append(
                    w2_weight[i]
                    .view(torch.uint8)[permute_indices.to(w2_weight.device)]
                    .contiguous()
                )
                # w2 scale shuffling
                permute_sf_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    w2_weight_scale[i].view(torch.uint8),
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                gemm2_scales_mxfp4_shuffled.append(
                    nvfp4_block_scale_interleave(
                        w2_weight_scale[i]
                        .view(torch.uint8)[
                            permute_sf_indices.to(w2_weight_scale.device)
                        ]
                        .contiguous()
                    )
                )
                # w2 bias shuffling
                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    w2_bias[i].clone().reshape(-1, 1),
                    epilogue_tile_m,
                )
                gemm2_bias_shuffled.append(
                    w2_bias[i]
                    .clone()
                    .reshape(-1, 1)[permute_indices.to(w2_bias.device)]
                    .contiguous()
                )

            w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
            w13_weight_scale = (
                torch.stack(gemm1_scales_mxfp4_shuffled)
                .reshape(
                    self.num_experts,
                    2 * self.intermediate_size,
                    self.hidden_size // sf_block_size,
                )
                .view(torch.float8_e4m3fn)
            )

            w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
            w2_weight_scale = (
                torch.stack(gemm2_scales_mxfp4_shuffled)
                .reshape(
                    self.num_experts,
                    self.hidden_size,
                    self.intermediate_size // sf_block_size,
                )
                .view(torch.float8_e4m3fn)
            )

            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = Parameter(w13_weight_scale, requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = Parameter(w2_weight_scale, requires_grad=False)
            layer.w13_bias = Parameter(
                torch.stack(gemm1_bias_shuffled).reshape(self.num_experts, -1),
                requires_grad=False,
            )
            layer.w2_bias = Parameter(
                torch.stack(gemm2_bias_shuffled).reshape(self.num_experts, -1),
                requires_grad=False,
            )
        elif (
            self.mxfp4_backend == Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8
            or self.mxfp4_backend == Mxfp4Backend.CUTLASS_SM90_FP4BF16
        ):
            layer.gemm1_alpha = Parameter(
                torch.tensor([1.702] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )
            layer.gemm1_beta = Parameter(
                torch.tensor([1.0] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )
            layer.gemm1_clamp_limit = Parameter(
                torch.tensor([7.0] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )

            sf_block_size = 32  # mxfp4 block size

            # Common shape assertions
            assert (
                layer.w13_weight.dim() == 3
                and layer.w13_weight.shape[0] == self.num_experts
                and layer.w13_weight.shape[1] == self.intermediate_size * 2
                and layer.w13_weight.shape[2] == self.hidden_size // 2
            )
            assert (
                layer.w13_weight_scale.dim() == 3
                and layer.w13_weight_scale.shape[0] == self.num_experts
                and layer.w13_weight_scale.shape[1] == self.intermediate_size * 2
                and layer.w13_weight_scale.shape[2] == self.hidden_size // sf_block_size
            )
            assert (
                layer.w2_weight.dim() == 3
                and layer.w2_weight.shape[0] == self.num_experts
                and layer.w2_weight.shape[1] == self.hidden_size
                and layer.w2_weight.shape[2] == self.intermediate_size // 2
            )
            assert (
                layer.w2_weight_scale.dim() == 3
                and layer.w2_weight_scale.shape[1] == self.hidden_size
                and layer.w2_weight_scale.shape[2]
                == self.intermediate_size // sf_block_size
            )
            assert (
                layer.w13_bias.dim() == 2
                and layer.w13_bias.shape[0] == self.num_experts
                and layer.w13_bias.shape[1] == self.intermediate_size * 2
            )
            assert (
                layer.w2_bias.dim() == 2
                and layer.w2_bias.shape[0] == self.num_experts
                and layer.w2_bias.shape[1] == self.hidden_size
            )

            # De-interleave and swap for w13 weight, bias, and scales
            w13_w = layer.w13_weight.data
            gate_w, up_w = w13_w[:, ::2, :], w13_w[:, 1::2, :]
            deinterleaved_w13_w = torch.cat([gate_w, up_w], dim=1)
            w1_w, w3_w = torch.chunk(deinterleaved_w13_w, 2, dim=1)
            w13_weight_swapped = torch.cat([w3_w, w1_w], dim=1)

            w13_b = layer.w13_bias.data.to(torch.float32)
            gate_b, up_b = w13_b[:, ::2], w13_b[:, 1::2]
            deinterleaved_w13_b = torch.cat([gate_b, up_b], dim=1)
            b1, b3 = torch.chunk(deinterleaved_w13_b, 2, dim=-1)
            w13_bias_swapped = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

            w13_s = layer.w13_weight_scale.data
            gate_s, up_s = w13_s[:, ::2, :], w13_s[:, 1::2, :]
            deinterleaved_w13_s = torch.cat([gate_s, up_s], dim=1)
            s1, s3 = torch.chunk(deinterleaved_w13_s, 2, dim=1)
            w13_scale_swapped = torch.cat([s3, s1], dim=1)

            if self.mxfp4_backend == Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8:
                from flashinfer import block_scale_interleave

                orig_shape = w13_scale_swapped.shape
                w13_scale_interleaved = block_scale_interleave(
                    w13_scale_swapped.view(torch.uint8)
                ).reshape(orig_shape)

                w2_s = layer.w2_weight_scale.data
                orig_shape = w2_s.shape
                w2_scale_interleaved = block_scale_interleave(
                    w2_s.view(torch.uint8)
                ).reshape(orig_shape)

                layer.w13_weight = Parameter(w13_weight_swapped, requires_grad=False)
                layer.w13_weight_scale = Parameter(
                    w13_scale_interleaved, requires_grad=False
                )
                layer.w13_bias = Parameter(w13_bias_swapped, requires_grad=False)
                layer.w2_weight_scale = Parameter(
                    w2_scale_interleaved, requires_grad=False
                )
            elif self.mxfp4_backend == Mxfp4Backend.CUTLASS_SM90_FP4BF16:

                def _interleave_mxfp4_cutlass_sm90(w):
                    w_shape = w.shape
                    w_interleaved = w.reshape(
                        w_shape[0], w_shape[1], (w_shape[2] // 4), 4
                    )
                    w_interleaved = w_interleaved.permute(0, 2, 1, 3)
                    w_interleaved = w_interleaved.reshape(
                        w_shape[0], w_shape[2] // 4, w_shape[1] * 4
                    )
                    return w_interleaved

                w31_scales = w13_scale_swapped.to(torch.uint8).view(torch.uint8)
                w31_scales_interleaved = _interleave_mxfp4_cutlass_sm90(w31_scales)

                w2_weight_scale = layer.w2_weight_scale.data
                w2_scales = w2_weight_scale.to(torch.uint8).view(torch.uint8)
                w2_scales_interleaved = _interleave_mxfp4_cutlass_sm90(w2_scales)

                layer.w13_weight = torch.nn.Parameter(
                    torch.cat([w3_w, w1_w], dim=1), requires_grad=False
                )
                layer.w13_bias = torch.nn.Parameter(
                    w13_bias_swapped, requires_grad=False
                )
                layer.w13_weight_scale = torch.nn.Parameter(
                    w31_scales_interleaved, requires_grad=False
                )
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_scales_interleaved, requires_grad=False
                )
        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

            w13_bias = layer.w13_bias.to(torch.float32)
            w2_bias = layer.w2_bias.to(torch.float32)

            layer.w13_bias = Parameter(w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(w2_bias, requires_grad=False)

            # Ideally we'd use FusedMoEModularKernel.prepare_finalize object
            # (stored in self.fused_experts) to determine if the MoE has a
            # batched activation format. As self.fused_experts is not
            # initialized at this point, we resort to checking the MoE config
            # directly.
            is_batched_moe = self.moe.use_pplx_kernels or self.moe.use_deepep_ll_kernels
            if is_batched_moe:
                num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
            else:
                num_warps = 8

            w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
                layer.w13_weight, layer.w13_weight_scale, num_warps
            )
            w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
                layer.w2_weight, layer.w2_weight_scale, num_warps
            )

            self.w13_precision_config = PrecisionConfig(
                weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
            )
            self.w2_precision_config = PrecisionConfig(
                weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
            )

            self.w13_weight = w13_weight
            self.w2_weight = w2_weight
            del layer.w13_weight
            del layer.w2_weight
            layer.w13_weight = w13_weight
            layer.w2_weight = w2_weight
        else:
            raise ValueError(
                f"Unsupported mxfp4_backend: {self.mxfp4_backend}: "
                f"should be one of: {list(Mxfp4Backend)}."
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            return mxfp4_w4a16_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
            return mxfp4_w4a16_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        elif self.mxfp4_backend in [
            Mxfp4Backend.TRTLLM_SM100_FP4FP8,
            Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8,
        ]:
            return mxfp4_mxfp8_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        elif self.mxfp4_backend in [Mxfp4Backend.TRTLLM_SM100_FP4BF16]:
            return mxfp4_w4a16_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale
            return ocp_mx_moe_quant_config(
                quant_dtype="mxfp4",
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            if self.mxfp4_backend == Mxfp4Backend.MARLIN:
                max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
                assert max_num_tokens_per_rank is not None
                assert self.moe_quant_config is not None
                return BatchedMarlinExperts(
                    max_num_tokens=max_num_tokens_per_rank,
                    num_dispatchers=prepare_finalize.num_dispatchers(),
                    quant_config=self.moe_quant_config,
                )
            else:
                raise NotImplementedError(
                    f"Incompatible Mxfp4 backend ({self.mxfp4_backend}) for "
                    "EP batched experts format"
                )
        else:
            assert self.moe_quant_config is not None
            if (
                self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4FP8
                or self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4BF16
            ):
                # B200 code-path
                kwargs = {
                    "gemm1_alpha": layer.gemm1_alpha,
                    "gemm1_beta": layer.gemm1_beta,
                    "gemm1_clamp_limit": layer.gemm1_clamp_limit,
                    # TODO(bnell): part of quant_config
                    "max_capture_size": self.max_capture_size,
                }
                return TrtLlmGenExperts(self.moe, self.moe_quant_config, **kwargs)
            elif self.mxfp4_backend == Mxfp4Backend.MARLIN:
                return MarlinExperts(self.moe_quant_config)
            elif self.mxfp4_backend == Mxfp4Backend.TRITON:
                if self.moe.is_lora_enabled:
                    return UnfusedOAITritonExperts(self.moe_quant_config)
                return OAITritonExperts(self.moe_quant_config)
            else:
                raise NotImplementedError(
                    f"Incompatible Mxfp4 backend ({self.mxfp4_backend}) for EP"
                )

    @property
    def allow_inplace(self) -> bool:
        return True

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if layer.enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            topk_weights, topk_ids = layer.select_experts(
                hidden_states=x,
                router_logits=router_logits,
            )

            return fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_bias,
                layer.w2_bias,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                global_scale1=None,
                global_scale2=None,
                quant_type_id=scalar_types.float4_e2m1f.id,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                activation=layer.activation,
                expert_map=layer.expert_map,
                input_dtype=self.marlin_input_dtype,
            )

        assert _can_support_mxfp4(
            layer.use_grouped_topk,
            layer.topk_group,
            layer.num_expert_group,
            layer.expert_map,
            layer.custom_routing_function,
            layer.e_score_correction_bias,
            layer.apply_router_weight_on_input,
            layer.scoring_func,
            layer.activation,
            layer.expert_load_view,
            layer.logical_to_physical_map,
            layer.logical_replica_count,
        ), "MXFP4 are not supported with this configuration."

        if (
            self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4FP8
            or self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4BF16
        ):
            from flashinfer import trtllm_fp4_block_scale_moe

            if self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4BF16:
                assert x.dtype == torch.bfloat16
                x_quant = x
                x_scale = None
            elif self.mxfp4_backend == Mxfp4Backend.TRTLLM_SM100_FP4FP8:
                from flashinfer import mxfp8_quantize

                x_quant, x_scale = mxfp8_quantize(x, False)  # to mxfp8
                x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)

            trtllm_gen_output = trtllm_fp4_block_scale_moe(
                router_logits.to(torch.bfloat16),
                None,  # routing_bias
                x_quant,
                x_scale,
                layer.w13_weight,  # uint8 (e2m1 x 2)
                layer.w13_weight_scale,  # uint8 (e4m3 x 2)
                layer.w13_bias,  # fp32 per expert per channel
                layer.gemm1_alpha,  # fp32 per expert
                layer.gemm1_beta,  # fp32 per expert
                layer.gemm1_clamp_limit,  # fp32 per expert
                layer.w2_weight,  # uint8 (e2m1 x 2)
                layer.w2_weight_scale,  # ue8m0
                layer.w2_bias,  # fp32 per expert per channel
                None,  # output1_scale_scalar
                None,  # output1_scale_gate_scalar
                None,  # output2_scale_scalar
                layer.global_num_experts,
                layer.top_k,
                None,  # n_group
                None,  # topk_group
                self.intermediate_size,  # padded to multiple of 256
                layer.ep_rank * layer.local_num_experts,  # local_expert_offset
                self.num_experts,  # local num experts
                None,
                None,
                1 if layer.renormalize else 0,  # routing_method_type, renormalize
                True,  # do finalize
                # FlashInfer uses tune_max_num_tokens to set up tuning/profile buckets.
                # It must be >= the actual number of tokens passed at runtime, otherwise
                # CUTLASS/runner initialization may fail internally for larger shapes.
                tune_max_num_tokens=max(self.max_capture_size, int(x.shape[0]), 1),
            )[0]
            return trtllm_gen_output
        elif (
            self.mxfp4_backend == Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8
            or self.mxfp4_backend == Mxfp4Backend.CUTLASS_SM90_FP4BF16
        ):
            from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

            topk_weights, topk_ids = layer.select_experts(
                hidden_states=x,
                router_logits=router_logits,
            )

            # Optional warmup fix: vLLM startup profile runs can route nearly all tokens to a single
            # expert, creating many empty experts (M=0) that some grouped GEMM implementations
            # cannot initialize. When enabled, force a balanced routing pattern ONLY for dummy runs.
            try:
                from vllm.forward_context import get_forward_context

                _ctx = get_forward_context()
                _is_dummy = bool(_ctx.additional_kwargs.get("vllm_dummy_run", False))
            except Exception:
                _is_dummy = False

            # Optional debug: summarize routing distribution (helps diagnose empty experts / M=0 groups
            # during startup profile runs).
            # NOTE: this will incur GPU work and a small device->host copy.
            import os as _os
            # Backend-specific preparation
            if self.mxfp4_backend == Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8:
                from flashinfer import mxfp8_quantize

                import os as _os
                x_quant, x_scale = mxfp8_quantize(x, True, 32)

                fake_input_scale = torch.ones(self.num_experts, device=x.device)
                quant_scales = [
                    layer.w13_weight_scale.contiguous().view(torch.int32),
                    fake_input_scale,
                    layer.w2_weight_scale.contiguous().view(torch.int32),
                    fake_input_scale,
                ]

                fi_input = x_quant
                extra_kwargs = dict(
                    use_mxfp8_act_scaling=True,
                    input_sf=x_scale,
                    fc1_expert_weights=layer.w13_weight.contiguous().view(torch.long),
                    fc2_expert_weights=layer.w2_weight.contiguous().view(torch.long),
                )
            elif self.mxfp4_backend == Mxfp4Backend.CUTLASS_SM90_FP4BF16:
                assert x.dtype == torch.bfloat16

                quant_scales = [
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                ]

                fi_input = x
                extra_kwargs = dict(
                    use_w4_group_scaling=True,
                    fc1_expert_weights=layer.w13_weight,
                    fc2_expert_weights=layer.w2_weight,
                )

            output = torch.empty_like(x, dtype=torch.bfloat16)
            _ = flashinfer_cutlass_fused_moe(
                input=fi_input,
                token_selected_experts=topk_ids.to(torch.int).contiguous(),
                token_final_scales=topk_weights,
                output_dtype=torch.bfloat16,
                output=output,
                quant_scales=quant_scales,
                fc1_expert_biases=layer.w13_bias,
                fc2_expert_biases=layer.w2_bias,
                swiglu_alpha=layer.gemm1_alpha,
                swiglu_beta=layer.gemm1_beta,
                swiglu_limit=layer.gemm1_clamp_limit,
                tp_size=self.moe.tp_size,
                tp_rank=self.moe.tp_rank,
                ep_size=self.moe.ep_size,
                ep_rank=self.moe.ep_rank,
                # FlashInfer uses tune_max_num_tokens to set up tuning/profile buckets.
                # It must be >= the actual number of tokens passed at runtime, otherwise
                # CUTLASS/runner initialization may fail internally for larger shapes.
                tune_max_num_tokens=max(self.max_capture_size, int(fi_input.shape[0]), 1),
                **extra_kwargs,
            )

            return output
        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
                triton_kernel_moe_forward,
            )

            return triton_kernel_moe_forward(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                gating_output=router_logits,
                topk=layer.top_k,
                renormalize=layer.renormalize,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.mxfp4_backend}")


class IpexMxfp4MoEMethod(Mxfp4MoEMethod):
    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self.moe_config = moe_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        self.original_hidden_size = hidden_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import intel_extension_for_pytorch as ipex

        layer.w13_weight.data = layer.w13_weight.data.view(torch.int32)
        layer.w2_weight.data = layer.w2_weight.data.view(torch.int32)
        ep_rank_start = self.moe_config.ep_rank * self.moe_config.num_local_experts
        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,
            layer.w2_weight,
            w1_scale_inv=layer.w13_weight_scale,
            w2_scale_inv=layer.w2_weight_scale,
            w13_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            is_mxfp4=True,
            experts_start_id=ep_rank_start,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert layer.activation == "swigluoai", (
            "Only swiglu_oai activation is supported for IPEX MXFP4 MoE"
        )
        hidden_size_pad = round_up(self.original_hidden_size, 128)
        x_pad = torch.nn.functional.pad(x, (0, hidden_size_pad - x.size(-1)))
        hidden_states = layer.ipex_fusion(
            x_pad,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            activation="swiglu_oai",
        )
        hidden_states = hidden_states[..., : self.original_hidden_size].contiguous()
        return hidden_states
