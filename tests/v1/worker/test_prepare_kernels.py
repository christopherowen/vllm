# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the GPU worker registers a step's JIT kernels while preparing.

A step's Triton kernels specialize on the KV cache groups' block tables, whose
packing reads the KV cache layout the engine core sets before memory profiling.
The worker asks the runner for them in ``_prepare_kernels`` (inside memory
profiling, so after the layout is set), while the registry collects, and only
then compiles the registry.
"""

import contextlib
from types import SimpleNamespace

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.v1.worker.gpu_worker import Worker


class _Registry:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    @contextlib.contextmanager
    def activate(self):
        self._trace.append("collect")
        yield
        self._trace.append("collected")

    def warmup(self) -> None:
        self._trace.append("compile")


def _worker(monkeypatch, runner) -> Worker:
    worker = Worker.__new__(Worker)  # no GPU, no init: only preparation is used
    worker.model_runner = runner
    worker.vllm_config = None
    worker._startup_heartbeat = None
    monkeypatch.setattr(
        gpu_worker_module, "set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        gpu_worker_module.envs, "VLLM_DS41_KERNEL_BACKEND", "b12x", raising=False
    )
    return worker


def test_step_kernels_register_while_collecting_before_the_registry_compiles(
    monkeypatch,
):
    trace: list[str] = []
    runner = SimpleNamespace(
        jit_warmup_registry=_Registry(trace),
        declare_step_kernels=lambda: trace.append("step kernels"),
    )
    _worker(monkeypatch, runner)._prepare_kernels()
    assert trace == ["collect", "step kernels", "collected", "compile"]


def test_a_runner_without_step_kernels_still_compiles_its_registry(monkeypatch):
    trace: list[str] = []
    runner = SimpleNamespace(jit_warmup_registry=_Registry(trace))
    _worker(monkeypatch, runner)._prepare_kernels()
    assert trace == ["compile"]
