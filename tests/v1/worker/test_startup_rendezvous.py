# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The startup rendezvous waits on liveness, not on a deadline."""

import threading

import pytest

from vllm.v1.worker import startup_rendezvous
from vllm.v1.worker.startup_rendezvous import StartupHeartbeat, rendezvous


class _Store:
    """The part of torch's TCPStore the rendezvous uses."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value.encode()

    def check(self, keys: list[str]) -> bool:
        with self._lock:
            return all(key in self._data for key in keys)

    def get(self, key: str) -> bytes:
        with self._lock:
            return self._data[key]


def test_every_rank_arrived_returns_at_once():
    store = _Store()
    store.set("vllm/startup/arrived/profiling/1", "1")
    waited = rendezvous(store, 0, 2, "profiling", 60, print, poll_s=0.01)
    assert waited < 1


class _SlowRankStore(_Store):
    """Rank 1's heartbeat changes at every read; it arrives after ``polls`` checks."""

    def __init__(self, polls: int) -> None:
        super().__init__()
        self.polls = polls
        self.beats = 0

    def check(self, keys: list[str]) -> bool:
        if keys == ["vllm/startup/arrived/profiling/1"]:
            self.polls -= 1
            if self.polls <= 0:
                self.set(keys[0], "1")
        return super().check(keys)

    def get(self, key: str) -> bytes:
        if key == "vllm/startup/heartbeat/1":
            self.beats += 1
            return f"{self.beats}|preparing kernels|prepared {self.beats}/900".encode()
        return super().get(key)


def test_a_slow_rank_is_awaited_while_its_heartbeat_moves():
    """However long a rank prepares, it is awaited while its heartbeat changes:
    here for far longer than the heartbeat timeout (100 polls of 10 ms vs 1 ms)."""
    store = _SlowRankStore(polls=100)
    store.set("vllm/startup/heartbeat/1", "0|preparing kernels")
    waited = rendezvous(store, 0, 2, "profiling", 0.001, print, poll_s=0.01)
    assert waited > 0.5 and store.beats >= 99


def test_a_rank_whose_heartbeat_stops_fails_by_name():
    store = _Store()
    store.set("vllm/startup/heartbeat/1", "7|preparing kernels|prepared 3/9 kernels")
    with pytest.raises(RuntimeError) as raised:
        rendezvous(store, 0, 2, "profiling", 0.05, print, poll_s=0.01)
    message = str(raised.value)
    assert "rank 1 stopped responding before profiling" in message
    assert "preparing kernels, prepared 3/9 kernels" in message


def test_progress_reports_name_the_missing_rank(monkeypatch):
    monkeypatch.setattr(startup_rendezvous, "REPORT_INTERVAL_S", 0.02)
    monkeypatch.setattr(startup_rendezvous, "HEARTBEAT_INTERVAL_S", 0.01)
    store = _Store()
    heartbeat = StartupHeartbeat(store, 1, lambda: "prepared 3/9 kernels")
    heartbeat.phase = "preparing kernels"
    heartbeat.start()
    reports: list[str] = []
    threading.Timer(
        0.2, lambda: store.set("vllm/startup/arrived/warm-up/1", "1")
    ).start()
    rendezvous(store, 0, 2, "warm-up", 5, reports.append, poll_s=0.01)
    heartbeat.stop()
    assert any("rank 1 (preparing kernels, prepared 3/9 kernels)" in r for r in reports)
