# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight transport preserves routing and ordinary Torch copy semantics."""

import inspect

import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.weight_transfer import (
    copy_weight,
    weight_transfer,
)


def test_writer_receives_final_parameter_slice_and_scope_restores_on_error():
    parameter = torch.zeros(4, 2)
    source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    calls = []

    def writer(destination, loaded):
        calls.append(destination.data_ptr())
        destination.copy_(loaded + 1)
        return True

    with (
        pytest.raises(RuntimeError, match="abort loading"),
        weight_transfer(writer),
    ):
        default_weight_loader(parameter[2:], source)
        raise RuntimeError("abort loading")
    torch.testing.assert_close(parameter[2:], source + 1)
    assert calls == [parameter[2:].data_ptr()]
    default_weight_loader(parameter[:2], source)
    torch.testing.assert_close(parameter[:2], source)
    assert len(calls) == 1


def test_declined_transfer_preserves_torch_cast_and_broadcast():
    destination = torch.empty((2, 3), dtype=torch.float64)
    with weight_transfer(lambda destination, source: False):
        copy_weight(destination, torch.tensor([1, 2, 3]))
    torch.testing.assert_close(
        destination, torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float64)
    )


def test_layerwise_meta_probe_counts_elements_without_io():
    from vllm.model_executor.model_loader.reload.meta import get_numel_loaded

    parameter = torch.empty((2, 3), device="meta")
    source = torch.empty_like(parameter)
    arguments = inspect.signature(default_weight_loader).bind(parameter, source)

    def unexpected_io(destination, source):
        pytest.fail("metadata-only load probes must not submit I/O")

    with weight_transfer(unexpected_io):
        count, _ = get_numel_loaded(default_weight_loader, arguments)
    assert count == 6


def test_composed_loader_waits_for_queued_reads_before_transforming():
    from vllm.model_executor.model_loader.weight_utils import composed_weight_loader

    class DeferredWriter:
        def __call__(self, destination, source):
            self.pending = (destination, source)
            return True

        def flush(self):
            destination, source = self.pending
            destination.copy_(source)

    parameter = torch.zeros(4)
    loader = composed_weight_loader(default_weight_loader, lambda value: value * 2)
    with weight_transfer(DeferredWriter()):
        loader(parameter, torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(parameter, torch.tensor([0.0, 2.0, 4.0, 6.0]))


def test_explicit_completion_preserves_rank_local_flush_boundaries():
    from vllm.model_executor.weight_transfer import (
        finish_weight_transfers,
        flush_weight_transfers,
    )

    calls = []

    class Writer:
        def __call__(self, destination, source):
            return True

        def flush(self):
            calls.append("flush")

        def finish(self):
            calls.append("finish")

    with weight_transfer(Writer()):
        flush_weight_transfers()
        finish_weight_transfers()
    assert calls == ["flush", "finish"]
    finish_weight_transfers()
    assert calls == ["flush", "finish"]


_MAPPED_COPY = """
import mmap, os, sys
import torch
from vllm.model_executor.weight_transfer import copy_weight

path, offset, nbytes = sys.argv[1], 1504, 1 << 20
with open(path, "wb") as file:
    file.write(os.urandom(nbytes + 2 * 65536))
mapping = mmap.mmap(os.open(path, os.O_RDONLY), 0, access=mmap.ACCESS_COPY)
# A view into file pages this process has not touched, off a page boundary, as a
# safetensors tensor is.
source = torch.frombuffer(mapping, dtype=torch.uint8)[offset : offset + nbytes]
destination = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
pinned = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
copy_weight(destination, source)
torch.cuda.synchronize()
# Nothing stays pinned: Torch's host allocator would keep it for the process's life.
assert torch.cuda.host_memory_stats().get("allocated_bytes.current", 0) == pinned
with open(path, "rb") as file:
    expected = file.read()[offset : offset + nbytes]
assert destination.cpu().numpy().tobytes() == expected
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_from_an_untouched_file_mapping_completes(tmp_path):
    """A checkpoint tensor that views memory-mapped file pages the process has not
    touched, starting off a page boundary, reaches the device, and leaves no pinned
    memory behind. (A pageable copy of it never completes with driver R580 on a 64 KiB-
    page GB10 kernel.) The copy runs in a child process so a hang fails the test
    instead of the suite."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _MAPPED_COPY, str(tmp_path / "weights.bin")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
