# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 TileLang kernels are declared into the runner's JIT registry."""

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.models.deepseek_v4_1.tilelang import prepare


def _gemm(n, k, block_M=16):  # stands in for a @tilelang.jit function
    raise AssertionError("declaring must not compile")


def _rope(heads, dim):
    raise AssertionError("declaring must not compile")


def test_declared_calls_are_collected_once_each():
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        prepare.declare(_gemm, lambda: [prepare.call(1024, 512, block_M=16)])
        # A second owner declaring the same call (another layer of the same shape).
        prepare.declare(
            _gemm,
            lambda: [
                prepare.call(1024, 512, block_M=16),
                prepare.call(1024, 512, block_M=32),
            ],
        )
        prepare.declare(_rope, lambda: [prepare.call(8, 512)])
    calls = prepare.declared_calls(registry)
    assert sorted(
        (fn.__name__, args, tuple(sorted(kw.items()))) for fn, args, kw in calls
    ) == [
        ("_gemm", (1024, 512), (("block_M", 16),)),
        ("_gemm", (1024, 512), (("block_M", 32),)),
        ("_rope", (8, 512), ()),
    ]


def test_declaring_outside_a_collecting_registry_is_a_no_op():
    prepare.declare(_rope, lambda: [prepare.call(8, 512)])  # no active registry


def test_calls_are_listed_when_preparing_not_when_declaring():
    registry = JitWarmupRegistry(vllm_config=None)
    listed = []

    def calls():
        listed.append(True)
        return [prepare.call(4, 128)]

    with registry.activate():
        prepare.declare(_rope, calls)
    assert not listed
    prepare.declared_calls(registry)
    assert listed
