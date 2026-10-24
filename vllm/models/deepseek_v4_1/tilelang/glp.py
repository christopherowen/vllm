# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering's projection, ``h <- h - alpha (h . d) d``, in one launch.

``streams`` ([T, hc_mult, hidden], BF16) are a layer's folded hyper-connection
streams, ``direction`` ([hidden], FP32) the layer's unit row (zero for a layer the
file does not steer) and ``alpha`` ([1], FP32) a buffer, so a file's contents never
change the traced graph. One block projects one stream of one token: each of its
threads loads eight values of the stream once (640 threads cover the model's 5,120),
widens them to FP32, and the block reduces ``h . d`` in a fixed order (each thread's
values in sequence, then the warp's lanes, then the warps) before every thread writes
``bf16(h - (alpha (h . d)) d)``. With the row spread that wide a thread holds sixteen
values, so more rows are in flight at once: at 8K and 16K-token prefill chunks it is
2.0-2.5% faster than the compiled torch projection, where 128 threads a row (five loads
a thread) were 1.2-1.4% slower (experiments/2026-10-23-next-release, glp-variants). No row depends on another, so a token's result
does not depend on what else the step carries. Alpha zero and a zero direction
return the stream unchanged; a non-finite value in a stream reaches its whole row,
as it does through ``h @ d``.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
from functools import cache

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import prepare


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def glp_project_kernel(hc_mult: int, hidden: int, threads: int = 640, vec: int = 8):
    assert hidden % (threads * vec) == 0 and threads % 32 == 0
    chunks = hidden // (threads * vec)
    warps = threads // 32
    tokens = T.dynamic("tokens")

    @T.prim_func
    def glp_project(
        streams: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
        direction: T.Tensor((hidden,), T.float32),
        alpha: T.Tensor((1,), T.float32),
        out: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
    ):
        with T.Kernel(hc_mult, tokens, threads=threads) as (h, t):
            tx = T.get_thread_binding()
            xs = T.alloc_local((chunks * vec,), T.float32)
            ds = T.alloc_local((chunks * vec,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((warps,), T.float32)
            scale = T.alloc_shared((1,), T.float32)
            acc[0] = T.float32(0)
            for c in T.serial(chunks):
                for i in T.vectorized(vec):
                    xs[c * vec + i] = streams[t, h, (c * threads + tx) * vec + i]
                for i in T.vectorized(vec):
                    ds[c * vec + i] = direction[(c * threads + tx) * vec + i]
                for i in T.serial(vec):
                    acc[0] += xs[c * vec + i] * ds[c * vec + i]
            acc[0] = T.warp_reduce_sum(acc[0])
            if tx % 32 == 0:
                partial[tx // 32] = acc[0]
            T.sync_threads()
            if tx == 0:
                acc[0] = T.float32(0)
                for warp in T.serial(warps):
                    acc[0] += partial[warp]
                scale[0] = alpha[0] * acc[0]
            T.sync_threads()
            for c in T.serial(chunks):
                for i in T.vectorized(vec):
                    out[t, h, (c * threads + tx) * vec + i] = T.cast(
                        xs[c * vec + i] - scale[0] * ds[c * vec + i], T.bfloat16
                    )

    return glp_project


@cache
def _kernel(hc_mult: int, hidden: int):
    return glp_project_kernel(hc_mult, hidden)


@torch.library.custom_op("vllm::dsv41_tilelang_glp_project", mutates_args=("out",))
def glp_project(
    streams: torch.Tensor,
    direction: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Project ``direction`` off ``streams`` ([T, hc_mult, hidden], BF16) into
    ``out``, scaled by ``alpha`` ([1], FP32)."""
    tokens, hc_mult, hidden = streams.shape
    if tokens == 0:
        return
    _kernel(hc_mult, hidden)(streams, direction, alpha, out)


@glp_project.register_fake
def _glp_project_fake(streams, direction, alpha, out):
    return None


def declare(hc_mult: int, hidden: int) -> None:
    """Declare the projection of ``hc_mult`` streams of ``hidden``, and keep it."""
    prepare.declare(
        lambda: [prepare.call(glp_project_kernel, hc_mult, hidden)],
        bind=lambda: _kernel(hc_mult, hidden),
    )


__all__ = ["declare", "glp_project", "glp_project_kernel"]
