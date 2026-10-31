# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.worker.gpu.sample import tilelang_auxiliary


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    return tilelang_auxiliary.count_nans(logits)


_TRITON_REFS = frozenset(("_num_nans_kernel",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
