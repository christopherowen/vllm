# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def _has_windowed_repeating_pattern(
    token_ids: Sequence[int],
    min_pattern_size: int,
    max_pattern_size: int,
    min_count: int,
    start_idx: int = 0,
    window_size: int = 160,
    min_density: float = 0.6,
) -> bool:
    """Check if the tail of token_ids ends with a phrase that repeats at least
    ``min_count`` times within the last ``window_size`` tokens (allowing short
    interspersed variations such as ``3xA + B + 2xA``) and covers at least
    ``min_density`` of the span from its first matched occurrence to the end.
    """
    seq = getattr(token_ids, "_x", token_ids)
    n = len(seq)
    avail = n - start_idx
    eff_min = max(min_pattern_size, 5)
    if avail < eff_min * min_count or eff_min > max_pattern_size:
        return False

    win_start = max(start_idx, n - window_size)
    last_tok = seq[n - 1]
    checked_patterns = 0

    for i in range(n - 2, win_start - 1, -1):
        if seq[i] != last_tok:
            continue
        gap = n - 1 - i
        if gap < eff_min:
            continue
        lcs = 1
        while (
            lcs <= max_pattern_size
            and i - lcs >= start_idx
            and seq[i - lcs] == seq[n - 1 - lcs]
        ):
            lcs += 1
        if lcs < eff_min:
            continue

        phrase = seq[n - lcs : n]
        cand_lens = [lcs]
        for b in range(1, lcs - eff_min + 1):
            if phrase[:b] == phrase[lcs - b :]:
                cand_lens.append(lcs - b)

        for cand_len in cand_lens:
            if cand_len * min_count > avail:
                continue
            cand = seq[n - cand_len : n]
            count = 1
            pos = n - cand_len
            while pos - cand_len >= win_start:
                nxt = pos - 1
                while nxt >= win_start and seq[nxt] != last_tok:
                    nxt -= 1
                if nxt - cand_len + 1 < win_start:
                    break
                if seq[nxt - cand_len + 1 : nxt + 1] == cand:
                    count += 1
                    pos = nxt - cand_len + 1
                    if (
                        count >= min_count
                        and count * cand_len >= (n - pos) * min_density
                    ):
                        return True
                else:
                    pos = nxt

        checked_patterns += 1
        if checked_patterns >= 2:
            break

    return False


def has_thinking_loop(
    token_ids: Sequence[int],
    start_idx: int = 0,
) -> bool:
    """Detect degenerate repetition loops inside a ``<think>`` block.

    Catches:
    1. Short contiguous token loops (pattern length 1..3 repeated >= 8 times
       with total repeated span >= 16 tokens).
    2. Phrase/sentence contiguous loops (pattern length 4..64 repeated >= 4
       times contiguously).
    3. Interspersed/windowed phrase loops (pattern length 5..64 repeated >= 5
       times within a 160-token window covering >= 60% of the span).
    """
    seq = getattr(token_ids, "_x", token_ids)
    avail = len(seq) - start_idx
    if avail < 16:
        return False

    for pattern_len in range(1, 4):
        min_count = max(8, (16 + pattern_len - 1) // pattern_len)
        if pattern_len * min_count <= avail and _has_repeating_pattern(
            seq, pattern_len, min_count
        ):
            return True

    max_len = min(64, avail // 4)
    for pattern_len in range(4, max_len + 1):
        if _has_repeating_pattern(seq, pattern_len, 4):
            return True

    return _has_windowed_repeating_pattern(
        seq,
        min_pattern_size=5,
        max_pattern_size=64,
        min_count=5,
        start_idx=start_idx,
        window_size=160,
        min_density=0.6,
    )


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
    start_idx: int = 0,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
        start_idx: Minimum starting index in token_ids to inspect.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    seq = getattr(token_ids, "_x", token_ids)
    avail = len(seq) - start_idx
    if avail <= 0:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > avail:
            break

        if _has_repeating_pattern(seq, pattern_len, min_count):
            return True

    if max_pattern_size >= 5 and min_count >= 3:
        return _has_windowed_repeating_pattern(
            seq,
            min_pattern_size=max(min_pattern_size, 5),
            max_pattern_size=max_pattern_size,
            min_count=max(min_count, 5),
            start_idx=start_idx,
        )

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def stops_on_next_token(request: Request, token_id: int, max_model_len: int) -> bool:
    """Whether committing ``token_id`` next stops the request by EOS, a stop
    token or a length cap: check_stop's checks that no later token changes.
    Sink rejection commits such a token as soon as it is sampled."""
    sampling_params = request.sampling_params
    assert sampling_params is not None
    return (
        token_id == sampling_params.eos_token_id
        or token_id in (sampling_params.stop_token_ids or ())
        or request.num_tokens + 1 >= max_model_len
        or request.num_output_tokens + 1 >= request.max_tokens
    )


def check_stop(
    request: Request,
    max_model_len: int,
    thinking_loop_break: bool = False,
) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # Note(arpera):
    # Order of checks is important for min_tokens
    # If you decide to change the order, first look at these PRs please:
    # 47489, 49521, and 51299
    # These PRs show some reasoning about the order of checks
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    if request.thinking_disabled:
        return False

    repetition_detection = sampling_params.repetition_detection
    if request.thinking_output_start >= 0:
        if request.force_thinking_end:
            return False
        start_idx = request.thinking_output_start
        loop_detected = False
        if thinking_loop_break and request.thinking_loop_breaks == 0:
            loop_detected = has_thinking_loop(
                request._output_token_ids, start_idx=start_idx
            )
        if not loop_detected and repetition_detection is not None:
            loop_detected = check_sequence_repetition(
                request._output_token_ids,
                repetition_detection,
                start_idx=start_idx,
            )
        if loop_detected:
            if thinking_loop_break and request.thinking_loop_breaks == 0:
                request.force_thinking_end = True
                request.thinking_loop_breaks += 1
                request.copy_draft_alloc = 0
                return False
            request.status = RequestStatus.FINISHED_REPETITION
            request.stop_reason = "repetition_detected"
            return True
        return False

    start_idx = request.thinking_end_output_idx
    if repetition_detection is not None and check_sequence_repetition(
        request._output_token_ids,
        repetition_detection,
        start_idx=start_idx,
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    if request.thinking_loop_breaks > 0 and has_thinking_loop(
        request._output_token_ids, start_idx=start_idx
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
