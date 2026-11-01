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


def test_compact_int32_token_map_matches_int64_reference():
    from vllm.models.deepseek_v4_1.engram_rows import host_hash

    rng = np.random.default_rng(7)
    vocab = 129_280
    token_map = rng.integers(0, 96_000, size=vocab, dtype=np.int64)
    multipliers = rng.integers(1, 10_000_000, size=(2, 4), dtype=np.int64) | 1
    primes = rng.integers(100_003, 500_009, size=(2, host_hash.HEADS), dtype=np.int64)
    offsets = np.arange(host_hash.HEADS, dtype=np.int64) * 500_000
    offsets = np.broadcast_to(offsets, (2, host_hash.HEADS)).copy()

    host = host_hash.HostEngramHash(token_map, multipliers, primes, offsets)
    assert host.token_map.dtype == np.int32
    assert host.token_map.nbytes == vocab * 4

    tokens = rng.integers(-2, vocab + 10, size=512, dtype=np.int64)
    tokens[17] = host_hash.IMAGE_SENTINEL
    positions = np.arange(512, dtype=np.int64)

    compressed = host.compress(tokens)
    assert compressed.dtype == np.int64
    got = host.hash_positions(tokens, positions)

    # Compare against an explicit int64 token_map copy.
    host.token_map = token_map.astype(np.int64)
    host.pad = host.token_map[host_hash.PAD_TOKEN]
    expected_compressed = host.compress(tokens)
    expected = host.hash_positions(tokens, positions)

    np.testing.assert_array_equal(compressed, expected_compressed)
    np.testing.assert_array_equal(got, expected)


def test_max_l2_cpus_selects_big_l2_cores(tmp_path, monkeypatch):
    from vllm.models.deepseek_v4_1.engram_rows import host_hash

    for cpu in range(20):
        size_dir = tmp_path / f"cpu{cpu}" / "cache" / "index2"
        size_dir.mkdir(parents=True)
        is_x925 = (5 <= cpu <= 9) or (15 <= cpu <= 19)
        (size_dir / "size").write_text("2048K\n" if is_x925 else "512K\n", encoding="utf-8")

    expected = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
    assert host_hash.max_l2_cpus(tmp_path, allowed=set(range(20))) == expected
    assert host_hash.max_l2_cpus(tmp_path, allowed={0, 1, 5, 15}) == [5, 15]

    monkeypatch.setenv("SPARK_DS41F_ENGRAM_CPU_AFFINITY", "0")
    assert host_hash.pin_to_max_l2_cpus(sysfs_root=tmp_path) == []
def test_step_ids_layer_slices_are_contiguous_and_column_mask_matches_full_hash():
    from vllm.models.deepseek_v4_1.engram_rows.host_hash import HostEngramHash

    rng = np.random.default_rng(7)
    token_map = rng.integers(0, 1000, size=256, dtype=np.int64)
    multipliers = rng.integers(1, 1 << 30, size=(2, 4), dtype=np.int64)
    primes = rng.integers(1000, 10000, size=(2, 24), dtype=np.int64)
    offsets = rng.integers(0, 100000, size=(2, 24), dtype=np.int64)
    host = HostEngramHash(token_map, multipliers, primes, offsets)

    tokens = rng.integers(0, 256, size=32, dtype=np.int64)
    reqs = [(0, 16, tokens, np.arange(16, dtype=np.int64))]
    full = host.step_ids(20, reqs)
    assert full.shape == (20, 2, 24)
    for layer in range(2):
        assert full[:, layer].flags.c_contiguous

    for tp_size in (3, 4):
        for tp_rank in range(tp_size):
            owned = [c for c in range(24) if c % tp_size == tp_rank]
            unowned = [c for c in range(24) if c % tp_size != tp_rank]
            mask = sum(1 << c for c in owned)
            masked = host.step_ids(20, reqs, columns=mask)
            for layer in range(2):
                assert masked[:, layer].flags.c_contiguous
            assert np.array_equal(masked[:, :, owned], full[:, :, owned])
            assert (masked[:16, :, unowned] == -1).all()
            assert (masked[16:] == -1).all()
