# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram n-gram hash on the host, for the tokens whose ids the host already has.

A step's Engram lookups split by when their row IDs can exist. A prompt token's id
is on the host when the runner prepares the step, a step before the GPU runs it, so
its rows can be read while the previous step computes; a token the GPU samples (a
decode token or a draft) is known only to the GPU when the step starts. This module
hashes the first kind exactly as ``tilelang/engram_hash.py`` does on the GPU (the
same int64 arithmetic: wrapping multiplies, XOR, TileLang's floor remainder), so the
reader can be handed those rows early and the GPU's ids for them are never needed.

``UNKNOWN`` marks a lookup whose id only the GPU will know; ``-1`` is a known "no
row" (a padding token past the step's live tokens), as the GPU writes it.
"""

import os
from pathlib import Path

import numpy as np

HEADS = 24
LAGS = 4
PAD_TOKEN = 2
IMAGE_SENTINEL = 129264
UNKNOWN = np.iinfo(np.int64).min


def _parse_cache_kib(raw: str) -> int:
    text = raw.strip().upper()
    if not text:
        return 0
    if text.endswith("K"):
        return int(text[:-1])
    if text.endswith("M"):
        return int(text[:-1]) * 1024
    if text.endswith("G"):
        return int(text[:-1]) * 1024 * 1024
    return max(1, int(text) // 1024)


def max_l2_cpus(
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
    allowed: set[int] | None = None,
) -> list[int]:
    """Return the online CPUs with the largest private L2 cache (`cache/index2/size`).

    On GB10's 20-core Grace CPU, `cpu0..4` and `cpu10..14` are Cortex-A725 cores with a
    512 KiB L2, while `cpu5..9` and `cpu15..19` are Cortex-X925 cores with a 2048 KiB L2.
    """
    root = Path(sysfs_root)
    if allowed is None and hasattr(os, "sched_getaffinity"):
        try:
            allowed = set(os.sched_getaffinity(0))
        except OSError:
            allowed = None
    best_kib = 0
    best_cpus: list[int] = []
    if not root.is_dir():
        return sorted(allowed) if allowed else []
    for entry in root.iterdir():
        name = entry.name
        if not (name.startswith("cpu") and name[3:].isdigit()):
            continue
        cpu = int(name[3:])
        if allowed is not None and cpu not in allowed:
            continue
        size_path = entry / "cache" / "index2" / "size"
        try:
            kib = _parse_cache_kib(size_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if kib > best_kib:
            best_kib = kib
            best_cpus = [cpu]
        elif kib == best_kib and kib > 0:
            best_cpus.append(cpu)
    best_cpus.sort()
    if best_cpus:
        return best_cpus
    return sorted(allowed) if allowed else []


def pin_to_max_l2_cpus(
    pid: int = 0,
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
) -> list[int]:
    """Pin thread/process `pid` to the allowed CPUs with the largest L2 cache."""
    if os.environ.get("SPARK_DS41F_ENGRAM_CPU_AFFINITY") == "0":
        return []
    cpus = max_l2_cpus(sysfs_root=sysfs_root)
    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(pid, set(cpus))
        except OSError:
            return []
    return cpus


class HostEngramHash:
    """The hash geometry and tokenizer map of a model's Engram layers, on the host."""

    def __init__(self, token_map, multipliers, primes, offsets):
        token_map_arr = np.asarray(token_map)
        if (
            token_map_arr.size == 0
            or (
                int(token_map_arr.min()) >= np.iinfo(np.int32).min
                and int(token_map_arr.max()) <= np.iinfo(np.int32).max
            )
        ):
            self.token_map = np.asarray(token_map_arr, dtype=np.int32)
        else:
            self.token_map = np.asarray(token_map_arr, dtype=np.int64)
        self.multipliers = np.asarray(multipliers, dtype=np.int64)  # [layers, 4]
        self.primes = np.asarray(primes, dtype=np.int64)  # [layers, 24]
        self.offsets = np.asarray(offsets, dtype=np.int64)  # [layers, 24]
        self.layers = self.multipliers.shape[0]
        if self.multipliers.shape != (self.layers, LAGS) or self.primes.shape != (
            self.layers,
            HEADS,
        ):
            raise ValueError(
                "Engram hash geometry must be [layers, 4] multipliers and "
                "[layers, 24] primes"
            )
        self.pad = self.token_map[PAD_TOKEN]
        self.vocab = self.token_map.shape[0]
        if os.environ.get("SPARK_DS41F_ENGRAM_HOST_HASH_AFFINITY", "1") != "0":
            pin_to_max_l2_cpus(0)

    def _compress_compact(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.int64)
        valid = (raw >= 0) & (raw < self.vocab) & (raw != IMAGE_SENTINEL)
        missing = self.token_map.dtype.type(-1)
        return np.where(valid, self.token_map[np.where(valid, raw, 0)], missing)

    def compress(self, raw: np.ndarray) -> np.ndarray:
        """Compressed ids of raw token ids; -1 for a missing token (negative), an
        image token or one outside the tokenizer map."""
        return self._compress_compact(raw).astype(np.int64, copy=False)

    def hash_positions(self, tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Row ids ``[len(positions), layers, 24]`` of the tokens at ``positions``
        of one request whose token ids up to the last position are ``tokens`` (prompt
        order; the GPU's query tokens and committed history are the same ids)."""
        positions = np.asarray(positions, dtype=np.int64)
        lags = positions[:, None] - np.arange(LAGS, dtype=np.int64)[None, :]  # [T, 4]
        raw = np.where(lags >= 0, tokens[np.maximum(lags, 0)], -1)
        sources = self._compress_compact(raw)
        # From the first missing source on, every lag takes the pad id.
        blocked = np.logical_or.accumulate(sources == -1, axis=1)
        values = np.where(blocked, self.pad, sources)
        # value * multiplier in wrapping int64, as the GPU computes it.
        products = (
            values[:, None, :].astype(np.uint64)
            * self.multipliers[None, :, :].astype(np.uint64)
        ).view(np.int64)  # [T, layers, 4]
        mixed2 = products[..., 0] ^ products[..., 1]
        mixed3 = mixed2 ^ products[..., 2]
        mixed4 = mixed3 ^ products[..., 3]
        # Heads 0-7 hash 2-grams, 8-15 3-grams, 16-23 4-grams.
        mixed = np.concatenate(
            [np.repeat(m[..., None], 8, axis=-1) for m in (mixed2, mixed3, mixed4)],
            axis=-1,
        )  # [T, layers, 24]
        # The floor remainder (the sign of the divisor), as TileLang's int64 %. The
        # model's bounded multipliers keep ``mixed`` non-negative, where it equals
        # C's truncating one.
        return np.remainder(mixed, self.primes[None]) + self.offsets[None]

    def step_ids(self, num_tokens: int, requests) -> np.ndarray:
        """A step's row ids ``[num_tokens, layers, 24]``: ``requests`` yields, per
        request in the step's token order, ``(first_row, rows, tokens, positions)``:
        the request's first row and row count in the step, its token ids, and the
        positions of its leading rows whose ids the host has (the rest of its rows are
        ``UNKNOWN``). Rows past the last request's rows are padding, -1 as the GPU
        writes them."""
        out = np.full((num_tokens, self.layers, HEADS), UNKNOWN, dtype=np.int64)
        live = 0
        for first_row, rows, tokens, positions in requests:
            live = max(live, first_row + rows)
            if len(positions):
                out[first_row : first_row + len(positions)] = self.hash_positions(
                    tokens, positions
                )
        out[live:] = -1
        return out


def placed_by_host(requests: list, drafts: np.ndarray | None) -> list:
    """``requests`` (``(first_row, rows, tokens, positions)`` in batch order) with the
    host's positions dropped from every request with drafts (``drafts``, per request).

    Adaptive verification gives verifying requests row counts the GPU chooses, but
    keeps them at the front of the batch with the host's total, so a request after
    them starts at the same row on the host and the GPU, and one among them does
    not. A request still prefilling can carry drafts (the scheduler pads a prompt
    with one token left to the verification width with placeholder drafts), so its
    rows stay ``UNKNOWN`` for the GPU to hash. Verifying requests after one without
    drafts would leave no row placed by the host, so the GPU hashes the whole step."""
    if drafts is None:
        return requests
    verifying = np.flatnonzero(drafts)
    front = len(verifying)
    if front and verifying[-1] >= front:
        front = len(requests)
    empty = np.empty(0, dtype=np.int64)
    return [
        (first, rows, tokens, empty if i < front else positions)
        for i, (first, rows, tokens, positions) in enumerate(requests)
    ]


__all__ = [
    "HEADS",
    "UNKNOWN",
    "HostEngramHash",
    "max_l2_cpus",
    "pin_to_max_l2_cpus",
    "placed_by_host",
]
