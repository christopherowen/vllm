# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which prefill rows the host hashes early in a step with verifying requests (CPU)."""

import numpy as np

from vllm.models.deepseek_v4_1.engram_rows.host_hash import UNKNOWN, placed_by_host


def _request(first, rows, prefill_positions):
    tokens = np.arange(100, dtype=np.int64)
    return (first, rows, tokens, np.asarray(prefill_positions, dtype=np.int64))


def _kept(requests):
    return [list(positions) for *_, positions in requests]


def test_no_drafts_keeps_every_prefill_row():
    requests = [_request(0, 1, []), _request(1, 4, [10, 11, 12, 13])]
    assert placed_by_host(requests, None) is requests
    assert _kept(placed_by_host(requests, np.array([0, 0]))) == [[], [10, 11, 12, 13]]


def test_a_chunk_after_the_verifying_requests_keeps_its_rows():
    requests = [
        _request(0, 6, []),
        _request(6, 6, []),
        _request(12, 4, [20, 21, 22, 23]),
    ]
    assert _kept(placed_by_host(requests, np.array([5, 5, 0]))) == [
        [],
        [],
        [20, 21, 22, 23],
    ]


def test_a_prefilling_request_with_drafts_is_left_to_the_gpu():
    """#677: a prompt with one token left, padded to the verification width with
    placeholder drafts, leads the batch; a later chunk still starts where the host
    puts it."""
    requests = [
        _request(0, 6, [99]),
        _request(6, 6, []),
        _request(12, 4, [20, 21, 22, 23]),
    ]
    assert _kept(placed_by_host(requests, np.array([5, 5, 0]))) == [
        [],
        [],
        [20, 21, 22, 23],
    ]


def test_a_verifying_request_after_one_without_drafts_leaves_the_step_to_the_gpu():
    requests = [_request(0, 4, [20, 21, 22, 23]), _request(4, 6, [])]
    assert _kept(placed_by_host(requests, np.array([0, 5]))) == [[], []]


def test_dropped_rows_stay_unknown_in_the_step_ids():
    from vllm.models.deepseek_v4_1.engram_rows import host_hash

    host = host_hash.HostEngramHash.__new__(host_hash.HostEngramHash)
    host.layers = 1
    host.hash_positions = lambda tokens, positions: np.zeros(
        (len(positions), 1, host_hash.HEADS), dtype=np.int64
    )
    requests = placed_by_host(
        [_request(0, 6, [99]), _request(6, 2, [20, 21])], np.array([5, 0])
    )
    ids = host.step_ids(10, requests)
    assert (ids[:6] == UNKNOWN).all()
    assert (ids[6:8] == 0).all()
    assert (ids[8:] == -1).all()
