# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The multi-node RoCE policy has no fallback: every collective runs on the
selected backends (one-shot runtime, PyNCCL) or raises. Nothing reaches the
PyTorch NCCL process group, which the policy never creates (about 3.4 GB of
unified memory per rank on GB10). CPU only."""

import inspect
import re

import pytest
import torch
import torch.distributed

from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

PROCESS_GROUP_COLLECTIVES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "gather",
    "broadcast",
    "send",
    "recv",
    "all_to_all_single",
    "batch_isend_irecv",
    "barrier",
)


class _PyNccl:
    disabled = False

    def __init__(self):
        self.calls = []

    def broadcast(self, tensor, src):
        self.calls.append(("broadcast", src))

    def batch_isend_irecv(self, p2p_ops):
        self.calls.append(("batch_isend_irecv", len(p2p_ops)))


class _OneShot:
    disabled = False


def _policy_communicator(pynccl=None):
    comm = object.__new__(cuda_communicator.CudaCommunicator)
    comm._roce_policy = True
    comm.b12x_ar_comm = _OneShot()
    comm.pynccl_comm = pynccl if pynccl is not None else _PyNccl()
    comm.world_size = 4
    comm.rank_in_group = 0
    comm.device_group = object()
    return comm


@pytest.fixture(autouse=True)
def no_process_group(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the RoCE policy reached the PyTorch process group")

    for name in PROCESS_GROUP_COLLECTIVES:
        monkeypatch.setattr(torch.distributed, name, refuse, raising=False)


def test_in_place_all_reduce_is_not_part_of_the_policy():
    with pytest.raises(RuntimeError, match="no in-place all-reduce"):
        _policy_communicator().all_reduce_in_place(torch.zeros(4))


def test_gather_is_not_part_of_the_policy():
    with pytest.raises(RuntimeError, match="no gather"):
        _policy_communicator().gather(torch.zeros(4))


def test_broadcast_runs_on_pynccl():
    pynccl = _PyNccl()
    tensor = torch.zeros(4)
    assert _policy_communicator(pynccl).broadcast(tensor, src=0) is tensor
    assert pynccl.calls == [("broadcast", 0)]


def test_batch_isend_irecv_runs_on_pynccl():
    pynccl = _PyNccl()
    _policy_communicator(pynccl).batch_isend_irecv([object(), object()])
    assert pynccl.calls == [("batch_isend_irecv", 2)]


@pytest.mark.parametrize("call", ["broadcast", "batch_isend_irecv"])
def test_lost_pynccl_stops_instead_of_switching_backend(call):
    pynccl = _PyNccl()
    pynccl.disabled = True
    comm = _policy_communicator(pynccl)
    args = (torch.zeros(4),) if call == "broadcast" else ([object()],)
    with pytest.raises(RuntimeError, match="lost its PyNCCL communicator"):
        getattr(comm, call)(*args)


def test_no_inherited_collective_uses_the_process_group():
    """A base-class collective that calls torch.distributed must be overridden,
    so a new upstream collective cannot bypass the policy unnoticed."""
    collective = re.compile(
        r"\b(?:torch\.distributed|dist)\.(?:"
        + "|".join(PROCESS_GROUP_COLLECTIVES)
        + r")\("
    )
    inherited = []
    for name, method in inspect.getmembers(DeviceCommunicatorBase, inspect.isfunction):
        if not collective.search(inspect.getsource(method)):
            continue
        if getattr(cuda_communicator.CudaCommunicator, name) is method:
            inherited.append(name)
    assert inherited == []
