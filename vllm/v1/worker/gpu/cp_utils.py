# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


def maybe_prepare_dcp_local_seq_lens(
    dcp_local_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
    *,
    num_reqs_padded: int | None = None,
) -> torch.Tensor | None:
    """Populate caller-owned storage and return its padded view, or None without DCP."""
    if dcp_size == 1:
        return None

    max_num_reqs = dcp_local_seq_lens.shape[0]
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv(max_num_reqs, BLOCK_SIZE)
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )
    return dcp_local_seq_lens[:num_reqs_padded]


@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    # Distribute KV cache among different ranks, in a round-robin manner.
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)

    remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
    remainder = tl.minimum(remainder, cp_interleave)
    local_seq_lens = rounds * cp_interleave + remainder

    # For [num_reqs, max_num_reqs), pad with 0
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)


_TRITON_REFS = frozenset(("cp_local_slot",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
