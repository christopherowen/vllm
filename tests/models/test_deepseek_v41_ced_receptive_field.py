# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical evidence for a finite sliding-window dependency cone.

This is a scalar mathematical model, NOT a native-kernel or checkpoint test. It
uses NumPy float64 softmax attention, no quantization, and no persistent global
attention. Its inputs stand for exact encoder outputs; global attention is safe
to omit from this dependency argument only when all its stored values are exact
and independent of the truncated decoder state. Native coverage lives in
tests/kernels/attention/test_deepseek_v41_ced_replay.py.
"""

from dataclasses import dataclass

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_test

WINDOW = 128
LAYERS = 3
RECOVERY_ROWS = LAYERS * (WINDOW - 1) + 1


@dataclass
class NumericalState:
    # Row j is the input to layer j; the final row is the stack's output.
    values: np.ndarray

    @property
    def final_output(self):
        return self.values[-1, -1]

    @property
    def live_inputs(self):
        return self.values[:-1, -WINDOW:]

    @property
    def deepest_aux(self):
        return self.values[-2, -WINDOW:]


def chunks(start, end, size):
    """Independent chunking; this numerical oracle does not import the scheduler."""
    return [(p, min(size, end - p)) for p in range(start, end, size)]


def numerical_stack(encoder, scheduled, floor, *, query_mode):
    """Evaluate the causal graph, preserving each layer's input between chunks."""
    values = np.full((LAYERS + 1, len(encoder)), np.nan, dtype=np.float64)
    for start, rows in scheduled:
        stop = start + rows
        begin = max(start, floor)
        # Earlier chunks have no influence on any retained row. Production CED
        # may compute a discarded dummy row there; this oracle needs none.
        if begin >= stop:
            continue
        values[0, begin:stop] = encoder[begin:stop]
        for layer in range(LAYERS):
            source = values[layer]
            for position in range(begin, stop):
                history = source[max(floor, position + 1 - WINDOW) : position + 1]
                assert np.isfinite(history).all()
                query = 4096.0 if query_mode == "edge_marker" else source[position]
                scores = query * history / np.sqrt(512.0)
                # A zero-logit attention sink carries no value. Stable softmax
                # keeps the marker case finite without clipping its influence.
                maximum = max(0.0, scores.max())
                weights = np.exp(scores - maximum)
                attended = np.dot(weights, history) / (
                    weights.sum() + np.exp(-maximum)
                )
                values[layer + 1, position] = source[position] + attended
    return NumericalState(values)


@pytest.fixture(scope="module")
def uninterrupted_reference():
    cases = {}

    def reference(generated, query_mode):
        key = generated, query_mode
        if key not in cases:
            prompt = 1024
            total = prompt + generated
            if query_mode == "edge_marker":
                encoder = np.zeros(total, dtype=np.float64)
                encoder[total - RECOVERY_ROWS] = 0.5
            else:
                encoder = ((np.arange(total) % 17) - 8) / 512.0
            expected = numerical_stack(
                encoder,
                [(0, prompt), *chunks(prompt, total, 1)],
                prompt - WINDOW,
                query_mode=query_mode,
            )
            cases[key] = prompt, encoder, expected
        return cases[key]

    return reference


def assert_same_live_state(actual, expected):
    assert np.isfinite(actual.final_output)
    assert np.isfinite(expected.final_output)
    assert np.isfinite(actual.live_inputs).all()
    assert np.isfinite(expected.live_inputs).all()
    np.testing.assert_array_equal(actual.final_output, expected.final_output)
    np.testing.assert_array_equal(actual.live_inputs, expected.live_inputs)
    np.testing.assert_array_equal(actual.deepest_aux, expected.deepest_aux)


@pytest.mark.parametrize("generated", [257, 513, 1025])
@pytest.mark.parametrize("chunk_size", [64, 512])
@pytest.mark.parametrize("query_mode", ["self", "edge_marker"])
def test_numerical_cone_restores_final_output_and_live_inputs(
    generated, chunk_size, query_mode, uninterrupted_reference
):
    """D*127+1 final rows also cover 128 inputs at depth D-1, here 382."""
    prompt, encoder, expected = uninterrupted_reference(generated, query_mode)
    total = len(encoder)
    original_floor = prompt - WINDOW
    full = numerical_stack(
        encoder, chunks(0, total, chunk_size), original_floor, query_mode=query_mode
    )
    assert_same_live_state(full, expected)
    floor = max(original_floor, total - RECOVERY_ROWS)
    bounded = numerical_stack(
        encoder, chunks(0, total, chunk_size), floor, query_mode=query_mode
    )
    assert_same_live_state(bounded, full)


@pytest.mark.parametrize("generated", [257, 513, 1025])
@pytest.mark.parametrize("chunk_size", [64, 512])
@pytest.mark.parametrize("horizon", [381, 128])
def test_numerical_shorter_cone_drops_visible_outermost_dependency(
    generated, chunk_size, horizon, uninterrupted_reference
):
    prompt, encoder, expected = uninterrupted_reference(generated, "edge_marker")
    total = len(encoder)
    floor = max(prompt - WINDOW, total - horizon)
    actual = numerical_stack(
        encoder, chunks(0, total, chunk_size), floor, query_mode="edge_marker"
    )
    assert np.isfinite(actual.live_inputs).all()
    assert np.isfinite(expected.live_inputs).all()
    assert np.isfinite(actual.final_output) and np.isfinite(expected.final_output)
    assert expected.final_output > 0.25
    assert actual.final_output == 0
    assert not np.array_equal(actual.live_inputs[-1], expected.live_inputs[-1])
    assert not np.array_equal(actual.deepest_aux, expected.deepest_aux)


@pytest.mark.parametrize("generated", [0, 1, 129])
def test_numerical_short_recovery_preserves_original_prompt_floor(
    generated, uninterrupted_reference
):
    prompt, encoder, expected = uninterrupted_reference(generated, "self")
    floor = max(prompt - WINDOW, len(encoder) - RECOVERY_ROWS)
    assert floor == prompt - WINDOW
    bounded = numerical_stack(
        encoder, chunks(0, len(encoder), 64), floor, query_mode="self"
    )
    assert_same_live_state(bounded, expected)
