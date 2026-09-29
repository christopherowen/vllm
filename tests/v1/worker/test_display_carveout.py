# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the display carve-out weight placement (display_carveout.py)."""

import ctypes
import os
import pathlib

import pytest
import torch

from vllm.v1.worker import display_carveout

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Table(torch.nn.Module):
    def __init__(self, rows: int, width: int, device: str):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, width, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )


def _model(device: str) -> torch.nn.Module:
    model = torch.nn.Module()
    model.embed_tokens = _Table(8192, 4096, device)  # 64 MiB
    model.lm_head = _Table(8192, 4096, device)
    model.small = _Table(16, 64, device)
    model.drafter = torch.nn.Module()
    model.drafter.lm_head = model.lm_head  # aliased, like the DSpark drafter
    return model


@pytest.fixture
def tables(monkeypatch):
    monkeypatch.setattr(display_carveout, "VocabParallelEmbedding", _Table)


def test_struct_layouts_match_cuda_h():
    assert ctypes.sizeof(display_carveout._ExternalMemoryHandleDesc) == 104
    assert ctypes.sizeof(display_carveout._ExternalMemoryBufferDesc) == 88
    assert display_carveout._ExternalMemoryHandleDesc.size.offset == 24


@needs_cuda
def test_selects_large_tables_once(tables):
    names = [name for name, _ in display_carveout.streamed_weights(_model("cuda"))]
    assert names == ["embed_tokens.weight", "lm_head.weight"]


@needs_cuda
def test_missing_drm_node_refuses(tables, monkeypatch):
    monkeypatch.setattr(display_carveout, "_DRM_NODE", "/nonexistent/card0")
    with pytest.raises(RuntimeError, match="display carve-out cannot hold"):
        display_carveout.relocate_streamed_weights(_model("cuda"), torch.device("cuda"))


@needs_cuda
@pytest.mark.skipif(
    not os.access("/dev/dri/card0", os.R_OK | os.W_OK), reason="no DRM node"
)
def test_relocation_preserves_values(tables):
    model = _model("cuda")
    before = {
        name: weight.detach().clone() for name, weight in model.named_parameters()
    }
    head_ptr = model.lm_head.weight.data_ptr()
    display_carveout.relocate_streamed_weights(model, torch.device("cuda"))
    assert model.lm_head.weight.data_ptr() != head_ptr
    assert model.drafter.lm_head.weight.data_ptr() == model.lm_head.weight.data_ptr()
    for name, weight in model.named_parameters():
        assert torch.equal(weight, before[name]), name
    # No DRM file stays open, so the worker never keeps DRM master (which would
    # stop the kernel's framebuffer console from drawing).
    fds = pathlib.Path("/proc/self/fd")
    assert not [fd for fd in fds.iterdir() if os.path.realpath(fd) == "/dev/dri/card0"]
