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

import numpy as np

HEADS = 24
LAGS = 4
PAD_TOKEN = 2
IMAGE_SENTINEL = 129264
UNKNOWN = np.iinfo(np.int64).min


class HostEngramHash:
    """The hash geometry and tokenizer map of a model's Engram layers, on the host."""

    def __init__(self, token_map, multipliers, primes, offsets):
        self.token_map = np.asarray(token_map, dtype=np.int64)
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

    def compress(self, raw: np.ndarray) -> np.ndarray:
        """Compressed ids of raw token ids; -1 for a missing token (negative), an
        image token or one outside the tokenizer map."""
        raw = np.asarray(raw, dtype=np.int64)
        valid = (raw >= 0) & (raw < self.vocab) & (raw != IMAGE_SENTINEL)
        return np.where(valid, self.token_map[np.where(valid, raw, 0)], -1)

    def hash_positions(self, tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Row ids ``[len(positions), layers, 24]`` of the tokens at ``positions``
        of one request whose token ids up to the last position are ``tokens`` (prompt
        order; the GPU's query tokens and committed history are the same ids)."""
        positions = np.asarray(positions, dtype=np.int64)
        lags = positions[:, None] - np.arange(LAGS, dtype=np.int64)[None, :]  # [T, 4]
        raw = np.where(lags >= 0, tokens[np.maximum(lags, 0)], -1)
        sources = self.compress(raw)
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


__all__ = ["HEADS", "UNKNOWN", "HostEngramHash"]
