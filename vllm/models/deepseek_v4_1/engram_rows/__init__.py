# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 Engram rows read from the checkpoint as steps ask (see ``rows``)."""

from vllm.models.deepseek_v4_1.engram_rows.rows import (
    PACKED_PAGE_BYTES,
    PACKED_ROW_BYTES,
    PACKED_ROWS_PER_PAGE,
    EngramDiskRows,
    MappedHostAllocation,
    column_compaction_map,
    pack_engram_columns,
    packed_table_bytes,
)

__all__ = [
    "EngramDiskRows",
    "MappedHostAllocation",
    "PACKED_PAGE_BYTES",
    "PACKED_ROWS_PER_PAGE",
    "PACKED_ROW_BYTES",
    "column_compaction_map",
    "pack_engram_columns",
    "packed_table_bytes",
]
