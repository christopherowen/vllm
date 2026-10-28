# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering's TileLang launch: a marked row's fold, as mHC's post kernel folds
it, projected within one rounding of FP64 and handed on so that the next fold
gives it back; an unmarked row untouched; and the step's row marks."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from tests.kernels.deepseek_v41_contracts import NAN  # noqa: E402
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import glp, glp_rows, prepare  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.collectives import (  # noqa: E402
    RankOrderParts,
)
from vllm.models.deepseek_v4_1.tilelang.mhc import _mhc_post  # noqa: E402

STREAMS, HIDDEN = 4, 5120


def _inputs(tokens: int, world: int = 1, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    parts = (randn(world, tokens, HIDDEN) / world).bfloat16()
    residual = randn(tokens, STREAMS, HIDDEN).bfloat16()
    post = torch.rand(tokens, STREAMS, generator=generator, device="cuda") * 2
    comb = torch.rand(tokens, STREAMS, STREAMS, generator=generator, device="cuda")
    comb /= comb.sum(dim=1, keepdim=True)
    direction = randn(HIDDEN)
    direction /= direction.double().norm().float()
    return parts, residual, post, comb, direction


def _write(parts: torch.Tensor) -> torch.Tensor:
    return parts[0] if parts.shape[0] == 1 else RankOrderParts(parts).sum()


def _fold(parts, residual, post, comb) -> torch.Tensor:
    """mHC's post kernel: the fold the next layer's post/pre computes."""
    out = torch.empty_like(residual)
    _mhc_post(_write(parts), residual, post, comb, out)
    return out


def _steer(parts, residual, post, comb, rows, direction, alpha: float):
    """The launch on copies: the steered residual, post and comb."""
    residual, post, comb = residual.clone(), post.clone(), comb.clone()
    glp.glp_steer(
        parts,
        residual,
        post,
        comb,
        rows,
        direction,
        torch.tensor([alpha], device="cuda"),
    )
    return residual, post, comb


def _fp64(streams, direction, alpha: float) -> torch.Tensor:
    h, d = streams.double(), direction.double()
    return h - (alpha * (h @ d)).unsqueeze(-1) * d


def _ulp(x: torch.Tensor) -> torch.Tensor:
    """BF16's spacing at ``|x|`` (eight significant bits)."""
    _, exponent = torch.frexp(x.abs().clamp_min(2.0**-126))
    return torch.ldexp(torch.ones_like(x), exponent - 8)


def _marks(tokens: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (tokens,), generator=generator, dtype=torch.int32).cuda()


@pytest.mark.parametrize("tokens", [1, 7, 96, 384, 2048])
@pytest.mark.parametrize("alpha", [0.75, 1.0, 2.0])
def test_a_marked_row_is_its_fold_projected_within_one_rounding(tokens, alpha):
    parts, residual, post, comb, direction = _inputs(tokens, seed=tokens)
    rows = torch.ones(tokens, dtype=torch.int32, device="cuda")
    steered, _, _ = _steer(parts, residual, post, comb, rows, direction, alpha)
    exact = _fp64(_fold(parts, residual, post, comb), direction, alpha)
    # One rounding to BF16, plus the FP32 dot product's own error where an output
    # cancels to nearly zero (2^-24 bounds it).
    assert ((steered.double() - exact).abs() <= _ulp(exact) + 2.0**-24).all()


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_the_next_fold_gives_a_marked_row_back(world):
    parts, residual, post, comb, direction = _inputs(130, world=world, seed=world)
    rows = _marks(130, seed=world)
    steered, post2, comb2 = _steer(parts, residual, post, comb, rows, direction, 0.75)
    marked = rows.bool()
    eye = torch.eye(STREAMS, device="cuda").expand(int(marked.sum()), -1, -1)
    assert torch.equal(comb2[marked], eye)
    assert torch.equal(post2[marked], torch.zeros_like(post2[marked]))
    # Equal values: a negative zero may come back positive.
    assert torch.equal(_fold(parts, steered, post2, comb2)[marked], steered[marked])


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def test_an_unmarked_row_is_untouched():
    parts, residual, post, comb, direction = _inputs(130)
    rows = _marks(130, seed=3)
    steered, post2, comb2 = _steer(parts, residual, post, comb, rows, direction, 0.75)
    kept = ~rows.bool()
    for before, after in ((residual, steered), (post, post2), (comb, comb2)):
        assert torch.equal(_bits(after)[kept], _bits(before)[kept])
    unmarked = torch.zeros(130, dtype=torch.int32, device="cuda")
    none, _, _ = _steer(parts, residual, post, comb, unmarked, direction, 0.75)
    assert torch.equal(_bits(none), _bits(residual))


def test_a_row_is_independent_of_the_batch_and_the_other_marks():
    parts, residual, post, comb, direction = _inputs(130)
    every = torch.ones(130, dtype=torch.int32, device="cuda")
    full, _, _ = _steer(parts, residual, post, comb, every, direction, 0.75)
    for tokens in (1, 7, 64, 129):
        part, _, _ = _steer(
            parts[:, :tokens],
            residual[:tokens],
            post[:tokens],
            comb[:tokens],
            every[:tokens],
            direction,
            0.75,
        )
        assert torch.equal(part, full[:tokens]), tokens
    some = _marks(130, seed=5)
    mixed, _, _ = _steer(parts, residual, post, comb, some, direction, 0.75)
    assert torch.equal(mixed[some.bool()], full[some.bool()])


def test_repeatable():
    parts, residual, post, comb, direction = _inputs(384)
    rows = _marks(384)
    first = _steer(parts, residual, post, comb, rows, direction, 0.75)
    for _ in range(3):
        again = _steer(parts, residual, post, comb, rows, direction, 0.75)
        assert all(torch.equal(a, b) for a, b in zip(again, first))


def test_alpha_zero_leaves_the_fold():
    parts, residual, post, comb, direction = _inputs(96)
    rows = torch.ones(96, dtype=torch.int32, device="cuda")
    steered, _, _ = _steer(parts, residual, post, comb, rows, direction, 0.0)
    assert torch.equal(steered, _fold(parts, residual, post, comb))


def test_a_nonfinite_value_stays_in_its_row():
    parts, residual, post, comb, direction = _inputs(8)
    rows = torch.ones(8, dtype=torch.int32, device="cuda")
    clean, _, _ = _steer(parts, residual, post, comb, rows, direction, 0.75)
    residual[2, 1, 17] = NAN
    out, _, _ = _steer(parts, residual, post, comb, rows, direction, 0.75)
    assert torch.isnan(out[2]).any()
    others = [t for t in range(8) if t != 2]
    assert torch.equal(out[others], clean[others])


def test_no_tokens_launch_nothing():
    parts, residual, post, comb, direction = _inputs(0)
    rows = torch.zeros(0, dtype=torch.int32, device="cuda")
    _steer(parts, residual, post, comb, rows, direction, 0.75)


def test_declared_as_the_forward_calls_it():
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        glp.declare(STREAMS, HIDDEN, (1, 4))
    calls = list(prepare.declared_calls(registry))
    assert calls == [
        (glp.glp_steer_kernel, (STREAMS, HIDDEN, 1), {}),
        (glp.glp_steer_kernel, (STREAMS, HIDDEN, 4), {}),
    ]


# The step's row marks.


def _host_marks(starts: list[int], idx: list[int], marks: list[int], rows: int):
    out = [0] * rows
    for r, state in enumerate(idx):
        for t in range(starts[r], starts[r + 1]):
            out[t] = marks[state]
    return out


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_rows_take_their_requests_marks_and_padding_none(idx_dtype):
    # Three requests in states 4, 0 and 2 (marked 1, 0, 1), then graph padding.
    starts, idx, marks = [0, 3, 4, 10], [4, 0, 2], [0, 0, 1, 0, 1, 0]
    rows = torch.full((16,), 7, dtype=torch.int32, device="cuda")
    glp_rows.mark_rows(
        rows,
        torch.tensor(idx, dtype=idx_dtype, device="cuda"),
        torch.tensor(starts, dtype=torch.int32, device="cuda"),
        torch.tensor(marks, dtype=torch.int32, device="cuda"),
    )
    assert rows.tolist() == _host_marks(starts, idx, marks, 16)


def test_no_requests_mark_no_row():
    rows = torch.ones(8, dtype=torch.int32, device="cuda")
    glp_rows.mark_rows(
        rows,
        torch.zeros(0, dtype=torch.int64, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.ones(4, dtype=torch.int32, device="cuda"),
    )
    assert rows.tolist() == [0] * 8


def test_query_bounds_must_cover_the_requests():
    with pytest.raises(ValueError, match="query bounds"):
        glp_rows.mark_rows(
            torch.zeros(4, dtype=torch.int32, device="cuda"),
            torch.zeros(2, dtype=torch.int64, device="cuda"),
            torch.zeros(2, dtype=torch.int32, device="cuda"),
            torch.zeros(4, dtype=torch.int32, device="cuda"),
        )
