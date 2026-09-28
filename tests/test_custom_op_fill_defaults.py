# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the one-read custom-op fill_defaults override in env_override.py."""

import torch
import torch._library.utils as library_utils

from vllm.env_override import _fill_defaults_one_read

SCHEMA = torch._C.parse_schema(
    "f(Tensor a, int b=3, *, float c=1.5, str d='x', int[] e=[1, 2]) -> ()"
)


def test_installed_by_env_override():
    assert library_utils.fill_defaults is _fill_defaults_one_read


def test_positional_defaults_and_kwarg_only_defaults():
    a = torch.zeros(1)
    args, kwargs = _fill_defaults_one_read(SCHEMA, (a,), {})
    assert args == (a, 3)
    assert kwargs == {"c": 1.5, "d": "x", "e": [1, 2]}


def test_given_values_win():
    a = torch.zeros(1)
    args, kwargs = _fill_defaults_one_read(SCHEMA, (a, 7), {"c": 2.5, "e": [4]})
    assert args == (a, 7)
    assert kwargs == {"c": 2.5, "d": "x", "e": [4]}


def test_mutating_custom_op_still_bumps_versions():
    @torch.library.custom_op("vllm_test::fill_defaults_bump", mutates_args=("out",))
    def bump(x: torch.Tensor, out: torch.Tensor, scale: float = 2.0) -> None:
        out.copy_(x * scale)

    x, out = torch.ones(2), torch.zeros(2)
    before = out._version
    with torch.no_grad():
        torch.ops.vllm_test.fill_defaults_bump(x, out)
    assert out._version > before
    assert torch.equal(out, torch.full((2,), 2.0))
