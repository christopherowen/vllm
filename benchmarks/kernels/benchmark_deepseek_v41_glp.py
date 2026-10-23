# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of GLP steering's TileLang projection against the torch
projection it replaces, compiled as the model's forward compiles it (Inductor: a
cuBLAS GEMV for ``h @ d`` and a fused elementwise kernel), at serving shapes.

Each shape is measured on eight fresh inputs; on each, CUDA graphs of 100 calls of
both run back to back, alternating which goes first, and the case's 95% interval
comes from the eight per-input differences. The gate (``--gate``) is that the
TileLang projection is not slower at any serving shape: a decode step (16 requests
of 1 + 5 drafted tokens) and a long-prefill chunk.
"""

import argparse
import json
import math
import statistics

import torch

from vllm.models.deepseek_v4_1.tilelang import glp

STREAMS, HIDDEN = 4, 5120
SHAPES = {"decode c1": 6, "decode c16": 96, "prefill 8K": 8192, "prefill 16K": 16384}
GATED = ("decode c16", "prefill 16K")


def _torch_projection(h, direction, alpha):
    x = h.float()
    return (x - (alpha * (x @ direction)).unsqueeze(-1) * direction).to(h.dtype)


def graph(function):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(100):
            function()
    return capture


def elapsed(capture) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10  # ms / 100 calls to microseconds


def measure(tokens: int, compiled, samples: int = 8) -> dict:
    ours_us, torch_us, differences = [], [], []
    for sample in range(samples):
        generator = torch.Generator(device="cuda").manual_seed(sample)
        h = torch.randn(
            tokens, STREAMS, HIDDEN, generator=generator, device="cuda"
        ).bfloat16()
        d = torch.randn(HIDDEN, generator=generator, device="cuda")
        d /= d.norm()
        alpha = torch.tensor([0.75], device="cuda")
        out = torch.empty_like(h)
        ours = graph(lambda: glp.glp_project(h, d, alpha, out))
        theirs = graph(lambda: compiled(h, d, alpha))
        pair = (ours, theirs) if sample % 2 == 0 else (theirs, ours)
        times = {id(capture): elapsed(capture) for capture in pair}
        ours_us.append(times[id(ours)])
        torch_us.append(times[id(theirs)])
        differences.append(times[id(ours)] - times[id(theirs)])
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    return {
        "tokens": tokens,
        "tilelang_us": statistics.median(ours_us),
        "torch_us": statistics.median(torch_us),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    compiled = torch.compile(_torch_projection, dynamic=False)
    results = {name: measure(tokens, compiled) for name, tokens in SHAPES.items()}
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [name for name in GATED if results[name]["interval_us"][0] > 0]
        if slower:
            raise SystemExit(f"TileLang projection slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
