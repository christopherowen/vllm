# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Embedding and output-head weights in the GB10 display carve-out.

DGX Spark firmware reserves about 2 GiB at the top of physical memory for
display scanout surfaces (the DISPLAY_FRM carve-out). Linux and ordinary CUDA
allocations never use it, so on a headless node it sits idle; a text console
takes only its framebuffer (8 MiB at 1920x1080). nvidia-drm serves DRM dumb
buffers from that carve-out, and a dumb buffer exported as a dma-buf and
imported with ``cuImportExternalMemory`` is readable by kernels.

Measured on GB10, the imported mapping streams at full speed (235 GB/s
sequential reads; a 6-row GEMV over a 43K-row head runs 10% faster than from
ordinary memory) but gives no reuse: random 576-byte reads run at 7 GB/s
against 221 GB/s, and large-M GEMMs that re-read their weights run 2.3x
slower. So only data read once per step belongs there. The embedding table
(row lookups) and the output head (only ever multiplied by decode rows and
prefill's sampled positions) qualify, while the KV cache, which attention
gathers and re-reads, does not: holding it there cost 8% at eight streams.

With ``SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`` the worker moves those weights into
the carve-out right after loading, before any kernel plan or graph captures
them, and returns their ordinary memory to the allocator for the KV cache. It
refuses to start if the carve-out cannot hold them.

Nothing writes the weights after the copy, and nothing else can reach them:
the DRM file and its buffer handle close right after the export, the dma-buf
stays private to the worker, the display engine only scans out its own
surfaces, and the scanout heap never hands out a range in use. A driver fault
or GPU reset could still leave the mapping on memory that something else
reuses, and the result would be wrong embeddings or logits with no error.

For debugging such a fault, ``SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS`` above 0
turns on an integrity check; production leaves it unset, which adds no work
at all. The worker then checksums every moved weight before and after the
copy and, from the end of graph capture, re-checks it from a background
thread on a side stream at that interval (about 20 ms per check for the
842.5 MiB of a DS4.1 rank). A mismatch is logged and kills the worker, which
stops the engine; nothing runs in the serving loop.
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import os
import signal
import struct
import threading
import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = init_logger(__name__)

_REQUIRED = os.environ.get("SPARK3_DISPLAY_CARVEOUT_WEIGHTS", "0") == "1"
_CHECK_SECONDS = float(os.environ.get("SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS", "0"))
# 16 MiB of 32-bit words per partial checksum (a 32 MiB int64 transient).
_CHECK_CHUNK_WORDS = 1 << 22
_DRM_NODE = "/dev/dri/card0"

_DRM_IOCTL_MODE_CREATE_DUMB = 0xC02064B2
_DRM_IOCTL_PRIME_HANDLE_TO_FD = 0xC00C642D
_DUMB_WIDTH = 4096
_DUMB_BPP = 32
_ROW_BYTES = _DUMB_WIDTH * _DUMB_BPP // 8
_GRANULE = 16 << 20
_ALIGN = 4096
# Vocabulary-sized tables only; small embeddings stay where they are.
_MIN_WEIGHT_BYTES = 64 << 20
_CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD = 9


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _HandleUnion(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("win32", _Win32Handle),
        ("nvSciBufObject", ctypes.c_void_p),
    ]


class _ExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _Carveout:
    """One dumb buffer imported into CUDA; alive for the whole process.

    The DRM file is held only while creating and exporting the buffer: the
    first opener of a card with no display server becomes DRM master, and the
    kernel's framebuffer console stops drawing while a userspace master exists.
    The exported dma-buf keeps the buffer alive after the file closes.
    """

    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        drm_fd = os.open(_DRM_NODE, os.O_RDWR | os.O_CLOEXEC)
        try:
            request = bytearray(
                struct.pack(
                    "IIIIIIQ", nbytes // _ROW_BYTES, _DUMB_WIDTH, _DUMB_BPP, 0, 0, 0, 0
                )
            )
            fcntl.ioctl(drm_fd, _DRM_IOCTL_MODE_CREATE_DUMB, request)
            _, _, _, _, handle, _, size = struct.unpack("IIIIIIQ", request)
            if size != nbytes:
                raise RuntimeError(
                    f"display carve-out returned {size} bytes, asked for {nbytes}"
                )
            prime = bytearray(struct.pack("IIi", handle, os.O_CLOEXEC | os.O_RDWR, -1))
            fcntl.ioctl(drm_fd, _DRM_IOCTL_PRIME_HANDLE_TO_FD, prime)
            self.dmabuf_fd = struct.unpack("IIi", prime)[2]
        finally:
            os.close(drm_fd)

        cuda = ctypes.CDLL("libcuda.so.1")
        desc = _ExternalMemoryHandleDesc()
        desc.type = _CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD
        desc.handle.fd = self.dmabuf_fd
        desc.size = nbytes
        self.external = ctypes.c_void_p()
        _check(
            cuda,
            cuda.cuImportExternalMemory(
                ctypes.byref(self.external), ctypes.byref(desc)
            ),
            "cuImportExternalMemory",
        )
        buffer = _ExternalMemoryBufferDesc()
        buffer.size = nbytes
        self.pointer = ctypes.c_ulonglong()
        _check(
            cuda,
            cuda.cuExternalMemoryGetMappedBuffer(
                ctypes.byref(self.pointer), self.external, ctypes.byref(buffer)
            ),
            "cuExternalMemoryGetMappedBuffer",
        )

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.nbytes,),
            "typestr": "|i1",
            "data": (self.pointer.value, False),
            "version": 3,
            "strides": None,
        }


def _check(cuda, status: int, what: str) -> None:
    if status:
        text = ctypes.c_char_p()
        cuda.cuGetErrorString(status, ctypes.byref(text))
        detail = text.value.decode() if text.value else status
        raise RuntimeError(f"{what} failed: {detail}")


_carveouts: list[_Carveout] = []


def required() -> bool:
    return _REQUIRED


def _checksum(tensor: torch.Tensor) -> torch.Tensor:
    """The sum of the tensor's 32-bit words in 64 bits: any flipped bit changes it.

    Summed in chunks: a 64-bit sum casts its input first, which over a whole
    421 MiB table would allocate 842 MiB, and the check stream's allocator
    pool would keep it.
    """
    raw = tensor.detach().reshape(-1).view(torch.uint8)
    words = raw.view(torch.int32) if raw.numel() % 4 == 0 else raw
    parts = words.split(_CHECK_CHUNK_WORDS)
    return torch.stack([part.sum(dtype=torch.int64) for part in parts]).sum()


def _kill_worker(reason: str) -> None:
    """Stop the engine from a background thread: its executor sees the worker die."""
    logger.error("Stopping the worker: %s", reason)
    os.kill(os.getpid(), signal.SIGKILL)


class _IntegrityCheck:
    """Periodic checksums of the weights held in the carve-out (debug mode)."""

    def __init__(
        self,
        weights: list[tuple[str, torch.Tensor]],
        expected: list[int],
        device: torch.device,
        interval: float,
        on_failure=_kill_worker,
    ):
        self.weights = weights
        self.expected = expected
        self.device = device
        self.interval = interval
        self.on_failure = on_failure
        self.failure: str | None = None
        self.passed = 0
        self._started = False
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def sums(self) -> list[int]:
        stream = (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else contextlib.nullcontext()
        )
        with stream:
            return torch.stack(
                [_checksum(weight) for _, weight in self.weights]
            ).tolist()

    def check(self) -> None:
        changed = [
            f"{name} (checksum {want:#x}, now {got:#x})"
            for (name, _), want, got in zip(self.weights, self.expected, self.sums())
            if want != got
        ]
        if changed:
            self.failure = "display carve-out weights changed: " + "; ".join(changed)
            logger.error("%s", self.failure)
            return
        self.passed += 1
        if self.passed == 1 or self.passed % 12 == 0:
            logger.info(
                "Display carve-out integrity check %d passed (every %g s)",
                self.passed,
                self.interval,
            )

    def start(self) -> None:
        if self._started or self.interval <= 0:
            return
        self._started = True
        threading.Thread(
            target=self._run, name="display-carveout-check", daemon=True
        ).start()

    def _run(self) -> None:
        if self._stream is not None:
            torch.cuda.set_device(self.device)
        while self.failure is None:
            time.sleep(self.interval)
            try:
                self.check()
            except Exception as error:
                # A check that cannot run must not pass silently either.
                self.failure = f"display carve-out integrity check failed: {error!r}"
                logger.exception("Display carve-out integrity check failed")
        self.on_failure(self.failure)


_integrity: _IntegrityCheck | None = None


def start_checks() -> None:
    """Start the debug-mode periodic check, if on; called once after graph capture.

    Starting after the last capture keeps the check's side-stream work away
    from graph captures.
    """
    if _integrity is not None:
        _integrity.start()


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def streamed_weights(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Vocabulary-sized embedding and output-head weights, deduplicated.

    ``ParallelLMHead`` is a ``VocabParallelEmbedding``; a drafter that aliases
    the target's modules shares their parameters, so each tensor appears once.
    """
    found: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, VocabParallelEmbedding):
            continue
        weight = getattr(module, "weight", None)
        if (
            isinstance(weight, torch.nn.Parameter)
            and weight.is_cuda
            and weight.is_contiguous()
            and _nbytes(weight) >= _MIN_WEIGHT_BYTES
        ):
            found.setdefault(weight.data_ptr(), (f"{name}.weight", weight))
    return list(found.values())


def relocate_streamed_weights(model: torch.nn.Module, device: torch.device) -> None:
    """Move the embedding and output-head weights into the display carve-out."""
    assert ctypes.sizeof(_ExternalMemoryHandleDesc) == 104
    assert ctypes.sizeof(_ExternalMemoryBufferDesc) == 88
    weights = streamed_weights(model)
    if not weights:
        raise RuntimeError(
            "SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1 but the model has no "
            "vocabulary-sized embedding or output-head weights"
        )
    spans = [-(-_nbytes(weight) // _ALIGN) * _ALIGN for _, weight in weights]
    rounded = -(-sum(spans) // _GRANULE) * _GRANULE
    torch.cuda.synchronize(device)
    try:
        carveout = _Carveout(rounded)
    except OSError as error:
        raise RuntimeError(
            f"SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1 but the display carve-out cannot "
            f"hold {rounded >> 20} MiB of weights ({_DRM_NODE}: {error})"
        ) from error
    _carveouts.append(carveout)
    backing = torch.as_tensor(carveout, device=device)
    assert backing.data_ptr() == carveout.pointer.value
    free_before = torch.cuda.mem_get_info(device)[0]
    checking = _CHECK_SECONDS > 0
    if checking:
        expected = [int(_checksum(weight.data)) for _, weight in weights]
    held: list[tuple[str, torch.Tensor]] = []
    offset = 0
    for (name, weight), span in zip(weights, spans):
        moved = backing[offset : offset + _nbytes(weight)].view(weight.dtype)
        moved = moved.view(weight.shape)
        moved.copy_(weight.data)
        weight.data = moved
        held.append((name, moved))
        offset += span
    torch.cuda.synchronize(device)
    if checking:
        global _integrity
        _integrity = _IntegrityCheck(held, expected, device, _CHECK_SECONDS)
        _integrity.check()
        if _integrity.failure is not None:
            raise RuntimeError(
                f"copy into the display carve-out failed: {_integrity.failure}"
            )
        _integrity.passed = 0
    torch.cuda.empty_cache()
    freed = torch.cuda.mem_get_info(device)[0] - free_before
    logger.info(
        "Display carve-out holds %.1f MiB of streamed weights (%s); "
        "ordinary memory freed: %.1f MiB; debug integrity check %s",
        sum(_nbytes(weight) for _, weight in weights) / (1 << 20),
        ", ".join(name for name, _ in weights),
        freed / (1 << 20),
        f"every {_CHECK_SECONDS:g} s" if checking else "off",
    )
