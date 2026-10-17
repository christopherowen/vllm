# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel preparation for DeepSeek V4.1's TileLang kernel family.

Every TileLang kernel the deployment can launch is declared as a call, exactly as its
call site makes it, by the module that owns it, while the model is built (``declare``).
After loading, the worker compiles every declared call before any distributed work, in
one thread pool (``prepare_declared``, through ``tilelang.prepare``), and then freezes
TileLang's kernel cache. From then on a call that was not declared raises
``KernelNotPreparedError`` on every rank alike, whatever each rank's disk cache holds,
instead of compiling inside a distributed forward, where the ranks that already have the
kernel would wait in a collective for the one that is compiling.

A module declares a function and a callable that lists its calls. The callable runs when
the worker prepares, after the weights are loaded, so a call may depend on what loading
produced. The call site makes its call through the same key function, so the declared
call and the launched call cannot drift apart.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry, VllmJitKernel

# (positional arguments, keyword arguments), exactly as the call site passes them.
Call = tuple[tuple[Any, ...], dict[str, Any]]


def call(*args: Any, **kwargs: Any) -> Call:
    """One declared call."""
    return args, kwargs


class _CallKey:
    """A hashable call (TileLang argument values need not be hashable)."""

    __slots__ = ("args", "kwargs", "_repr")

    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self._repr = repr((self.args, sorted(self.kwargs.items())))

    def __hash__(self) -> int:
        return hash(self._repr)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CallKey) and self._repr == other._repr

    def __repr__(self) -> str:
        return self._repr


class TileLangCalls(VllmJitKernel[_CallKey]):
    """One ``@tilelang.jit`` function's declared calls in the runner's JIT registry."""

    def __init__(self, function: Any) -> None:
        super().__init__()
        self.function = function
        self.name = getattr(function, "__name__", None) or repr(function)

    def get_warmup_keys(self, calls: Callable[[], Iterable[Call]]) -> list[_CallKey]:
        return [_CallKey(args, kwargs) for args, kwargs in calls()]

    def compile(self, compile_key: _CallKey) -> None:
        self.compile_many([compile_key])

    def compile_many(self, compile_keys: Iterable[_CallKey]) -> None:
        import tilelang

        tilelang.prepare(
            [(self.function, key.args, key.kwargs) for key in compile_keys],
            progress=PROGRESS.report,
        )


_adapters: dict[int, TileLangCalls] = {}
_adapters_lock = threading.Lock()


def declare(function: Any, calls: Callable[[], Iterable[Call]]) -> None:
    """Declare the calls ``calls()`` lists for the ``@tilelang.jit`` ``function``.

    ``calls`` runs when the worker prepares kernels, after loading. Must be called
    while the model runner's JIT registry collects (model construction and loading).
    """
    with _adapters_lock:
        adapter = _adapters.get(id(function))
        if adapter is None:
            adapter = _adapters[id(function)] = TileLangCalls(function)
    JitWarmupRegistry.register(adapter, calls)


class _Progress:
    """Kernels prepared so far, for the startup rendezvous's progress reports."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.done = 0
        self.total = 0

    def start(self, total: int) -> None:
        with self._lock:
            self.done, self.total = 0, total

    def report(self, done: int, total: int) -> None:
        with self._lock:
            # tilelang.prepare counts the calls it compiles; keep the larger figures.
            self.done = max(self.done, done)
            self.total = max(self.total, total)

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.done, self.total


PROGRESS = _Progress()


def declared_calls(registry: JitWarmupRegistry) -> list[tuple[Any, tuple, dict]]:
    """Every TileLang call declared in ``registry``, as ``(function, args, kwargs)``."""
    calls: dict[tuple[int, _CallKey], tuple[Any, tuple, dict]] = {}
    for kernel, registrations in registry.registrations():
        if not isinstance(kernel, TileLangCalls):
            continue
        for args, kwargs in registrations:
            for key in kernel.get_warmup_keys(*args, **kwargs):
                calls.setdefault(
                    (id(kernel), key), (kernel.function, key.args, key.kwargs)
                )
    return list(calls.values())


def prepare_declared(registry: JitWarmupRegistry) -> int:
    """Compile every TileLang call declared in ``registry`` in one thread pool.

    Returns the number of distinct calls; kernels already compiled are not counted
    again by the progress reports.
    """
    import tilelang

    calls = declared_calls(registry)
    PROGRESS.start(len(calls))
    tilelang.prepare(calls, progress=PROGRESS.report)
    return len(calls)


def freeze(reason: str) -> None:
    """Refuse every TileLang kernel that is not prepared, from now on."""
    from tilelang.cache import freeze as freeze_cache

    freeze_cache(reason)
