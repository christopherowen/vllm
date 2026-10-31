# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheSpec
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter,
    StagedWriteTensor,
    UvaBackedTensor,
)


def slot_mapping_mode(layer_spec: KVCacheSpec) -> tuple[bool, bool]:
    """Return whether the worker computes slot mappings for a group with this
    layer spec, and whether the group is a per-request ring."""
    return layer_spec.uses_slot_mapping, isinstance(layer_spec, CircularBufferSpec)


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int],
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
        slot_mapping_enabled: list[bool] | None = None,
        group_cp_sizes: list[int] | None = None,
        slot_mapping_circular: list[bool] | None = None,
    ):
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave
        if group_cp_sizes is None:
            group_cp_sizes = [cp_size] * len(block_sizes)
        assert len(group_cp_sizes) == len(block_sizes)
        assert all(group_cp_size in (1, cp_size) for group_cp_size in group_cp_sizes)
        self.group_cp_sizes_list = list(group_cp_sizes)

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        if slot_mapping_enabled is None:
            slot_mapping_enabled = [True] * self.num_kv_cache_groups
        assert len(slot_mapping_enabled) == self.num_kv_cache_groups
        self._slot_mapping_enabled = slot_mapping_enabled
        # A circular group keeps one block per request as a ring: every position maps
        # into that block at position modulo the block size.
        if slot_mapping_circular is None:
            slot_mapping_circular = [False] * self.num_kv_cache_groups
        assert len(slot_mapping_circular) == self.num_kv_cache_groups
        self._slot_mapping_circular = slot_mapping_circular

        self.blocks_per_kv_block = [
            bs // kbs for bs, kbs in zip(block_sizes, kernel_block_sizes)
        ]

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            max_num_blocks = max_num_blocks_per_group[i] * self.blocks_per_kv_block[i]
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)

        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )
        self.fused_writer: FusedStagedWriter | None = None
        if self.num_kv_cache_groups > 1:
            # Only the multi-group path uses the fused writer.
            self.fused_writer = FusedStagedWriter(
                self.device, self.num_kv_cache_groups * self.max_num_reqs
            )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        # The TileLang kernels read pointer tables as int64 (same bits).
        self._block_table_ptrs_i64 = self.block_table_ptrs.view(torch.int64)
        from vllm.v1.worker.gpu import tilelang_inputs

        # The gather's blocks: each group's 1024-column chunks of a row.
        self._gather_chunks = tilelang_inputs.gather_chunks(
            [b.gpu.stride(0) for b in self.block_tables], self.device
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.kernel_block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.slot_mapping_enabled = torch.tensor(
            self._slot_mapping_enabled, dtype=torch.bool, device=self.device
        )
        self.slot_mapping_circular = torch.tensor(
            self._slot_mapping_circular, dtype=torch.bool, device=self.device
        )
        self.group_cp_sizes = torch.tensor(
            self.group_cp_sizes_list, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)
        self._input_block_table_ptrs_i64 = self.input_block_table_ptrs.view(torch.int64)

    def get_group_cp_parameters(self, group_id: int) -> tuple[int, int, int]:
        group_cp_size = self.group_cp_sizes_list[group_id]
        if group_cp_size == 1:
            return 0, 1, 1
        return self.cp_rank, group_cp_size, self.cp_interleave

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            bpk = self.blocks_per_kv_block[i]
            if bpk > 1:
                block_ids = [b * bpk + k for b in block_ids for k in range(bpk)]
            end = start + len(block_ids)
            row_capacity = self.block_tables[i].gpu.shape[1]
            if end > row_capacity:
                raise RuntimeError(
                    f"Block table write for request {req_index}, group {i} exceeds "
                    f"row capacity ({end} > {row_capacity})"
                )
            self.block_tables[i].stage_write(req_index, start, block_ids)
            self.num_blocks.np[i, req_index] = end

    def apply_staged_writes(self) -> None:
        if self.num_kv_cache_groups == 0:
            return
        if self.num_kv_cache_groups == 1:
            # Single group: write directly, skipping the per-write group lookup.
            self.block_tables[0].apply_write()
        elif self.num_kv_cache_groups > 1:
            # Multiple groups: apply all block tables with one fused kernel.
            assert self.fused_writer is not None
            self.fused_writer.apply(
                self.block_tables, self._block_table_ptrs_i64, self.block_table_strides
            )
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
            dst_ptrs = self._input_block_table_ptrs_i64
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups
            dst_ptrs = out_ptrs.view(torch.int64)
        from vllm.v1.worker.gpu import tilelang_inputs

        # Every row up to num_reqs_padded: the padded rows are zeroed.
        tilelang_inputs.gather_block_tables(
            idx_mapping,
            self._block_table_ptrs_i64,
            dst_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self._gather_chunks,
            num_reqs_padded,
        )
        return tuple(bt[:num_reqs_padded] for bt in out)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        #
        # Zero the rows so dummy runs write mamba state to the reserved null
        # block rather than through the previous real step's (stale) block
        # ids, which may point at blocks since freed and reallocated.
        return tuple(
            block_table[:num_reqs].zero_() for block_table in self.input_block_tables
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_kv_cache_groups == 0:
            return (self.slot_mappings if out is None else out)[:, :num_tokens_padded]
        slot_mappings = self.slot_mappings if out is None else out
        from vllm.v1.worker.gpu import tilelang_inputs

        # Pads from the step's token count to the end of each row.
        tilelang_inputs.compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            self._block_table_ptrs_i64,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.block_sizes_tensor,
            self.kernel_block_sizes_tensor,
            self.slot_mapping_enabled,
            self.slot_mapping_circular,
            self.group_cp_sizes,
            slot_mappings,
            self.cp_rank,
            self.cp_size,
            self.cp_interleave,
        )
        return slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


def declare_block_table_kernels(*, cp_size: int, cp_interleave: int) -> None:
    """Declare the gather of a step's block tables and its slot mappings for
    ``BlockTables`` with at least one KV cache group (per-group geometry is tensor
    data, not part of the kernels' keys)."""
    from vllm.v1.worker.gpu import tilelang_inputs

    tilelang_inputs.declare(
        lambda: tilelang_inputs.block_table_calls(cp_size, cp_interleave)
    )


_TRITON_REFS = frozenset(
    ("_gather_block_tables_kernel", "_compute_slot_mappings_kernel")
)


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
