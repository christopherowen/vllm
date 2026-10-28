# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering: the fail-closed reader, the projection, and the off state."""

import math
import os
import struct

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.models.deepseek_v4_1 import glp

NUM_LAYERS = 8
HIDDEN = 16
STREAMS = 4
LAYERS = (1, 2, 5)
METADATA = {
    "general.base_model.0.name": "test-model",
    "glp.mode": "project",
    "glp.spec_version": 1,
    "glp.rank": 1,
    "glp.hook_point": glp.HOOK_POINT,
    "glp.alpha_default": 0.75,
    "glp.layer_ids_zero_based": "1,2,5",
}
STEER_ENV = ("WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_MODEL")
STEERED = "model-weightless"
MAX_TOKENS = 64


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(value) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return struct.pack("<II", 4, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    return struct.pack("<I", 8) + _string(value)


def write_gguf(path, metadata, tensors, *, version=3, kind=0, trailing=b""):
    """A GGUF file as llama.cpp's writer lays it out, aligned to 32 bytes."""
    header = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _string(key) + _value(value)
    blobs, offset = [], 0
    for name, values in tensors.items():
        values = np.asarray(values, dtype="<f4")
        dims = values.shape[::-1]
        header += _string(name) + struct.pack(f"<I{len(dims)}Q", len(dims), *dims)
        header += struct.pack("<IQ", kind, offset)
        blob = values.tobytes()
        blob += b"\0" * (-len(blob) % 32)
        blobs.append(blob)
        offset += len(blob)
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + b"".join(blobs) + trailing)
    return str(path)


def unit(seed: int, width: int = HIDDEN) -> np.ndarray:
    vector = np.random.default_rng(seed).standard_normal(width)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def glp_file(tmp_path, metadata=None, tensors=None, **kwargs):
    merged = dict(METADATA)
    for key, value in (metadata or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    if tensors is None:
        tensors = {f"direction.{layer}": unit(layer) for layer in LAYERS}
    return write_gguf(tmp_path / "vector.gguf", merged, tensors, **kwargs)


@pytest.fixture(autouse=True)
def _no_steering_env(monkeypatch):
    for name in STEER_ENV:
        monkeypatch.delenv(name, raising=False)


# The container reader.


def test_reader_returns_the_metadata_and_tensors(tmp_path):
    path = glp_file(tmp_path)
    metadata, tensors = glp.read_gguf(path)
    assert metadata["glp.mode"] == "project"
    assert metadata["glp.spec_version"] == 1
    assert metadata["glp.alpha_default"] == pytest.approx(0.75)
    assert sorted(tensors) == [f"direction.{layer}" for layer in LAYERS]
    for layer in LAYERS:
        np.testing.assert_array_equal(tensors[f"direction.{layer}"], unit(layer))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"version": 2}, "GGUF version 2"),
        ({"kind": 1}, "not F32"),
        ({"trailing": b"\x7f" * 64}, "unexpected bytes"),
        ({"tensors": {"direction.1": np.zeros((2, HIDDEN))}}, "not 1-D"),
    ],
)
def test_reader_refuses_what_it_cannot_read(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        glp.read_gguf(glp_file(tmp_path, **kwargs))


def test_reader_refuses_a_file_that_is_not_gguf(tmp_path):
    path = tmp_path / "vector.gguf"
    path.write_bytes(b"PK\x03\x04" + b"\0" * 60)
    with pytest.raises(ValueError, match="not a GGUF file"):
        glp.read_gguf(str(path))


def test_reader_refuses_a_truncated_file(tmp_path):
    glp_file(tmp_path)
    path = tmp_path / "vector.gguf"
    path.write_bytes(path.read_bytes()[:-40])
    with pytest.raises(ValueError, match="outside the file|truncated"):
        glp.read_gguf(str(path))


# The GLP contract.


def test_directions_land_on_their_own_layer_numbers(tmp_path):
    directions, alpha = glp.read_glp(glp_file(tmp_path), NUM_LAYERS, HIDDEN)
    assert sorted(directions) == list(LAYERS)
    assert alpha == pytest.approx(0.75)
    for layer in LAYERS:
        np.testing.assert_allclose(directions[layer], unit(layer), atol=1e-7)
        assert np.linalg.norm(directions[layer]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "metadata, tensors, message",
    [
        ({"glp.mode": None}, None, "absent mode means 'add'"),
        ({"glp.mode": "add"}, None, "never adds"),
        ({"glp.spec_version": None}, None, "spec_version"),
        ({"glp.spec_version": 2}, None, "spec_version"),
        ({"glp.rank": 2}, None, "rank"),
        ({"glp.hook_point": None}, None, "hook_point"),
        ({"glp.hook_point": "ffn_out_pre_residual"}, None, "hook_point"),
        ({"glp.alpha_default": math.nan}, None, "alpha_default"),
        ({"glp.alpha_default": math.inf}, None, "alpha_default"),
        ({"glp.layer_ids_zero_based": "1,2,6"}, None, "disagrees"),
        ({}, {"direction.0": unit(0)}, "direction.0"),
        ({}, {f"direction.{NUM_LAYERS}": unit(9)}, "outside layers"),
        ({}, {"direction.1": unit(1, HIDDEN + 1)}, "width"),
        ({}, {"direction.1": np.full(HIDDEN, np.nan)}, "not finite"),
        ({}, {"direction.1": 2 * unit(1)}, "norm"),
        ({}, {"bias.1": unit(1)}, "not a direction"),
        ({}, {}, "no direction tensors"),
    ],
)
def test_reader_refuses_files_it_cannot_apply_exactly(
    tmp_path, metadata, tensors, message
):
    if tensors is not None:
        metadata = {"glp.layer_ids_zero_based": None, **metadata}
    path = glp_file(tmp_path, metadata=metadata, tensors=tensors)
    with pytest.raises(ValueError, match=message):
        glp.read_glp(path, NUM_LAYERS, HIDDEN)


def test_a_transferred_vector_is_read(tmp_path):
    path = glp_file(tmp_path, metadata={"glp.derived_at": "ffn_out_pre_residual"})
    directions, _ = glp.read_glp(path, NUM_LAYERS, HIDDEN)
    assert sorted(directions) == list(LAYERS)


# Steering on and off.


def steer(monkeypatch, path) -> None:
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", path)
    monkeypatch.setenv("WEIGHTLESS_STEER_MODEL", STEERED)


def test_unset_means_no_steering_module():
    assert glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS) is None


def test_steering_stacks_the_file_directions(tmp_path, monkeypatch):
    steer(monkeypatch, glp_file(tmp_path))
    steering = glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS)
    assert steering.directions.shape == (NUM_LAYERS, HIDDEN)
    assert steering.directions.dtype == torch.float32
    for layer in range(NUM_LAYERS):
        expected = unit(layer) if layer in LAYERS else np.zeros(HIDDEN)
        np.testing.assert_allclose(steering.directions[layer], expected, atol=1e-7)
    assert steering.alpha.item() == pytest.approx(0.75)
    assert steering.layers == frozenset(LAYERS)
    # Buffers, never weights: the checkpoint loader must not see them.
    assert dict(steering.named_parameters()) == {}
    assert steering.state_dict() == {}


@pytest.mark.parametrize("name", STEER_ENV)
def test_one_variable_without_the_other_is_refused(tmp_path, monkeypatch, name):
    steer(monkeypatch, glp_file(tmp_path))
    monkeypatch.delenv(name)
    with pytest.raises(ValueError, match="needs both"):
        glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS)


@pytest.mark.parametrize("name", ["WEIGHTLESS_STEER_ALPHA", "WEIGHTLESS_STEER_LAYERS"])
def test_the_file_is_applied_as_it_is(tmp_path, monkeypatch, name):
    steer(monkeypatch, glp_file(tmp_path))
    monkeypatch.setenv(name, "2")
    steering = glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS)
    assert steering.alpha.item() == pytest.approx(0.75)
    assert steering.layers == frozenset(LAYERS)


def test_a_missing_file_is_refused(tmp_path, monkeypatch):
    steer(monkeypatch, str(tmp_path / "missing.gguf"))
    with pytest.raises(FileNotFoundError):
        glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS)


def test_a_file_without_alpha_is_refused(tmp_path, monkeypatch):
    steer(monkeypatch, glp_file(tmp_path, metadata={"glp.alpha_default": None}))
    with pytest.raises(ValueError, match="no glp.alpha_default"):
        glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS)


def test_the_steering_variables_are_part_of_the_compile_key(monkeypatch):
    unsteered = envs.compile_factors()
    for name in STEER_ENV:
        assert name in unsteered
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", "/vectors/a.gguf")
    assert envs.compile_factors() != unsteered


def test_rows_start_unmarked(tmp_path, monkeypatch):
    steer(monkeypatch, glp_file(tmp_path))
    steering = glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS)
    assert steering.rows.dtype == torch.int32
    assert steering.rows.shape == (MAX_TOKENS,)
    assert bool((steering.rows == 0).all())


def test_a_step_reads_a_view_of_the_marks(tmp_path, monkeypatch):
    steer(monkeypatch, glp_file(tmp_path))
    steering = glp.glp_steering(NUM_LAYERS, HIDDEN, STREAMS, MAX_TOKENS)
    rows = steering.step_rows(5)
    assert rows.shape == (5,)
    assert rows.data_ptr() == steering.rows.data_ptr()
    with pytest.raises(ValueError, match="at most 64 rows"):
        steering.step_rows(MAX_TOKENS + 1)


# A layer's step: one TileLang launch (tilelang/glp.py) for a covered layer. The
# kernel's results are tests/kernels/test_deepseek_v41_tilelang_glp.py's.

WIDE = 5120


def steering(layers=LAYERS) -> glp.GLPSteering:
    directions = torch.zeros(NUM_LAYERS, WIDE)
    for layer in layers:
        directions[layer] = torch.from_numpy(unit(layer, WIDE))
    return glp.GLPSteering(directions, 0.75, STREAMS, list(layers), MAX_TOKENS)


def _layer_outputs(tokens: int = 3):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(tokens, WIDE, generator=generator).bfloat16()
    residual = torch.randn(tokens, STREAMS, WIDE, generator=generator).bfloat16()
    post_mix = torch.rand(tokens, STREAMS, generator=generator)
    res_mix = torch.rand(tokens, STREAMS, STREAMS, generator=generator)
    return x, residual, post_mix, res_mix


@pytest.fixture
def launches(monkeypatch):
    calls = []
    monkeypatch.setattr(glp.tilelang_glp, "glp_steer", lambda *args: calls.append(args))
    return calls


def test_a_layer_the_file_does_not_cover_launches_nothing(launches):
    x, residual, post_mix, res_mix = _layer_outputs()
    steering().steer(
        3, x, residual, post_mix, res_mix, torch.ones(3, dtype=torch.int32)
    )
    assert launches == []


def test_a_covered_layer_steers_its_write_in_place(launches):
    module = steering()
    x, residual, post_mix, res_mix = _layer_outputs()
    rows = torch.tensor([0, 1, 0], dtype=torch.int32)
    module.steer(5, x, residual, post_mix, res_mix, rows)
    ((parts, *rest),) = launches
    assert parts.shape == (1, 3, WIDE) and torch.equal(parts[0], x)
    assert rest[0] is residual and rest[1] is post_mix and rest[2] is res_mix
    assert rest[3] is rows
    assert torch.equal(rest[4], module.directions[5])
    assert rest[5] is module.alpha


def test_the_ranks_unreduced_parts_are_handed_on_as_they_are(launches):
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts

    x, residual, post_mix, res_mix = _layer_outputs()
    packed = torch.stack([x, x, x])
    steering().steer(
        1,
        RankOrderParts(packed),
        residual,
        post_mix,
        res_mix,
        torch.ones(3, dtype=torch.int32),
    )
    assert launches[0][0] is packed


@pytest.mark.parametrize(
    "change",
    [
        lambda o: (o[0], o[1].float(), o[2], o[3], o[4]),
        lambda o: (o[0], o[1][:, :2], o[2], o[3], o[4]),
        lambda o: (o[0][:2], o[1], o[2], o[3], o[4]),
        lambda o: (o[0], o[1], o[2], o[3], o[4][:2]),
    ],
)
def test_only_the_models_streams_and_their_marks_are_steered(launches, change):
    outputs = (*_layer_outputs(), torch.ones(3, dtype=torch.int32))
    with pytest.raises(ValueError, match="BF16 streams"):
        steering().steer(1, *change(outputs))
    assert launches == []


@pytest.mark.parametrize("unreduced, worlds", [(False, (1,)), (True, (1, 4))])
def test_the_steering_is_declared_for_the_writes_it_takes(
    monkeypatch, unreduced, worlds
):
    import vllm.distributed

    declared = []
    monkeypatch.setattr(
        glp.tilelang_glp, "declare", lambda *shape: declared.append(shape)
    )
    monkeypatch.setattr(glp.sp_prefill, "unreduced_enabled", lambda: unreduced)
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 4
    )
    steering().declare_kernels()
    assert declared == [(STREAMS, WIDE, worlds)]


# A real GLP file, when one is named (it is not redistributed with the tests).

REAL_FILE = os.environ.get("GLP_TEST_FILE")


@pytest.mark.skipif(not REAL_FILE, reason="GLP_TEST_FILE names no file")
def test_a_published_ds41_file_holds_only_unit_directions():
    metadata, tensors = glp.read_gguf(REAL_FILE)
    # controlvector.*: llama.cpp's control-vector header, which the reader ignores.
    assert all(
        key.startswith(("general.", "glp.", "controlvector.")) for key in metadata
    )
    assert all(name.startswith("direction.") for name in tensors)
    directions, alpha = glp.read_glp(REAL_FILE, 40, 5120)
    assert sorted(directions) == list(range(1, 40))
    assert alpha == pytest.approx(0.75)
    for direction in directions.values():
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-5)
