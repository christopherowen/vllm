# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention-sink detection and rejection: configuration and host-side events.

A generated token can become an attention sink: from a middle layer on, one
channel of its residual (DeepSeek V4.1 Flash: column 2455, the BOS sink channel)
runs away by orders of magnitude and the row's next-token distribution turns
into document-start noise. The forward pass is the model's own, so the remedy is
a decoding policy. The detector reads that column of the pre-norm residual at
every sampled row (vllm.v1.worker.gpu.sample.sink).

With rejection on, the last token a step samples is not committed when it is
sampled: it is the request's provisional token, and the scheduler schedules it
as draft zero of the request's next step, ahead of the drafter's tokens. It then
goes through the speculative-decoding machinery like any draft: its KV slot is
rolled back if it is rejected, the prefix cache and the grammar only see it once
it is committed, and the frontend only receives it then. The verifier accepts
draft zero unless its own row (the row whose input it is) is sunk; a sunk draft
zero is rejected and replaced by the next alternate of the draw that produced
it. A sunk drafter token (row j >= 1) cuts the verification at that draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import vllm.envs as envs

# Fields of a request's per-step sink event, as the worker encodes them.
NUM_SINK_EVENT_FIELDS = 6


class SinkOutcome(IntEnum):
    """What happened at a request's row 0 (the row whose input is the step's
    first token: draft zero of a decoding request, the last prompt token of a
    request finishing its prefill)."""

    NONE = 0
    # Row 0 was sunk and nothing was replaced: the detector runs without
    # rejection, or the row's input is a prompt token.
    FLAGGED = 1
    # Draft zero was sunk and replaced by its next alternate.
    REJECTED = 2
    # Draft zero was sunk past its alternates and committed as it is.
    FALLTHROUGH = 3


class SinkRow(IntEnum):
    """Where a flagged row sits; the counters record flags per kind."""

    # Row 0 of a decoding request: its input is draft zero, the token the
    # request's previous step sampled.
    DRAFT_ZERO = 0
    # A row whose input is a drafter token (row j >= 1).
    DRAFTER = 1
    # Row 0 of a request finishing its prefill: its input is a prompt token.
    PROMPT = 2

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class SinkConfig:
    """The detector's column and threshold, and the rejection policy's switch."""

    # Column of the pre-norm residual watched at every sampled row.
    column: int
    # |residual[column]| above which the row is sunk.
    threshold: float
    # Hold the last sampled token as draft zero and reject it when sunk (the
    # fix); off counts only and leaves decoding as it is.
    rejection: bool
    # Alternates kept per provisional token: the rejections a position may take.
    max_retries: int

    @classmethod
    def from_env(cls) -> SinkConfig | None:
        column = envs.VLLM_SINK_DETECTOR_COLUMN
        if column < 0:
            if envs.VLLM_SINK_REJECTION:
                raise ValueError(
                    "VLLM_SINK_REJECTION needs the detector column "
                    "(VLLM_SINK_DETECTOR_COLUMN)."
                )
            return None
        threshold = envs.VLLM_SINK_DETECTOR_THRESHOLD
        if threshold <= 0:
            raise ValueError("VLLM_SINK_DETECTOR_THRESHOLD must be positive.")
        max_retries = envs.VLLM_SINK_REJECTION_MAX_RETRIES
        if envs.VLLM_SINK_REJECTION and max_retries < 1:
            raise ValueError("VLLM_SINK_REJECTION_MAX_RETRIES must be at least 1.")
        return cls(
            column=column,
            threshold=threshold,
            rejection=bool(envs.VLLM_SINK_REJECTION),
            max_retries=max_retries,
        )

    @property
    def num_alternates(self) -> int:
        """Alternates kept per provisional token (0: none needed)."""
        return self.max_retries if self.rejection else 0


@dataclass(frozen=True)
class SinkEvent:
    """One request's sink event of one step (host form of the worker's
    ``[num_reqs, NUM_SINK_EVENT_FIELDS]`` record)."""

    outcome: SinkOutcome
    # The kind of row 0: DRAFT_ZERO for a decoding request, PROMPT otherwise.
    row0: SinkRow
    # Flagged drafter rows among the rows whose inputs were committed or cut.
    num_drafter_flags: int
    # Rejection on: the drafter row at which the verification was cut, or -1.
    cut_row: int
    # Draft zero rejected: the logprob and rank of its replacement in the
    # distribution that produced the replaced token (None without logprobs).
    recovered_logprob: float | None = None
    recovered_rank: int | None = None
    # The largest |residual[column]| among the flagged rows, and its row in
    # the request's step (0: the step's first input; -1: none).
    value: int = 0
    value_row: int = -1

    @property
    def flagged_row0(self) -> bool:
        return self.outcome != SinkOutcome.NONE

    @property
    def num_flagged_rows(self) -> int:
        return int(self.flagged_row0) + self.num_drafter_flags


def events_to_dict(
    events: list[list[int]],
    req_ids: list[str],
    recovered_logprobs: list[float] | None = None,
    recovered_ranks: list[int] | None = None,
) -> dict[str, SinkEvent] | None:
    """The host form: request id -> event, for the requests with one."""
    result: dict[str, SinkEvent] = {}
    for i, (req_id, fields) in enumerate(zip(req_ids, events)):
        outcome, row0, num_drafter_flags, cut_row, value, value_row = fields
        if outcome == SinkOutcome.NONE and num_drafter_flags == 0:
            continue
        rejected = outcome == SinkOutcome.REJECTED
        result[req_id] = SinkEvent(
            outcome=SinkOutcome(outcome),
            row0=SinkRow(row0),
            num_drafter_flags=int(num_drafter_flags),
            cut_row=int(cut_row),
            value=int(value),
            value_row=int(value_row),
            recovered_logprob=(
                float(recovered_logprobs[i])
                if rejected and recovered_logprobs is not None
                else None
            ),
            recovered_rank=(
                int(recovered_ranks[i])
                if rejected and recovered_ranks is not None
                else None
            ),
        )
    return result or None
