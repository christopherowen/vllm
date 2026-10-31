# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering: a weightless control vector projected off the residual stream.

A GLP file (GGUF Layer Projection; github.com/msuiche/weightless, spec
docs/spec/GLP.md) holds one unit direction ``direction.N`` per layer N. Steering
applies

    h <- h - alpha * (h . d) d

at ``residual_stream_post_layer``: layer N's write folded into the
hyper-connection streams, before layer N + 1 reads Engram. Every stream is
projected, so their mean, the reduction the directions are derived on, is
projected too. The layer's write alone (``ffn_out_pre_residual``) is a different
site, and a file for it is refused.

Off unless WEIGHTLESS_STEER_PATH names a file and WEIGHTLESS_STEER_MODEL the
served model name it steers; one without the other stops the boot. Off, the
model has no steering module and its forward is unchanged. The reader is
fail-closed: a file this runtime cannot apply exactly stops the boot rather than
serving unsteered or steering at another site. The file's own layers and
``glp.alpha_default`` are applied as they are.

On, steering is a property of rows. Each covered layer runs one TileLang launch
(``tilelang/glp.py``) that folds and projects the rows ``GLPSteering.rows`` marks,
in place, and leaves every other row untouched. Only the rows of requests made to
the steered model name are marked (``vllm.weightless``): one engine serves the
model and its steered variant side by side, and a request to the other names
gets the same bits as with steering off.
"""

import argparse
import math
import re
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.deepseek_v4_1 import sp_prefill
from vllm.models.deepseek_v4_1.tilelang import glp as tilelang_glp
from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts

logger = init_logger(__name__)

HOOK_POINT = "residual_stream_post_layer"
SPEC_VERSION = 1
# A direction's norm may differ from 1 by float32 export rounding, no more.
UNIT_NORM_TOLERANCE = 1e-3

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGUF_F32 = 0
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_DIRECTION = re.compile(r"direction\.(\d+)")


class _Cursor:
    def __init__(self, path: str, data: bytes):
        self.path = path
        self.data = data
        self.offset = 0

    def bytes(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise ValueError(f"{self.path}: truncated GGUF at byte {self.offset}")
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        return raw

    def take(self, fmt: str):
        values = struct.unpack(fmt, self.bytes(struct.calcsize(fmt)))
        return values[0] if len(values) == 1 else values

    def string(self) -> str:
        try:
            return self.bytes(self.take("<Q")).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{self.path}: GGUF string is not UTF-8") from None

    def value(self, kind: int):
        if kind == _GGUF_STRING:
            return self.string()
        if kind == _GGUF_ARRAY:
            item, count = self.take("<IQ")
            if item == _GGUF_ARRAY:
                raise ValueError(f"{self.path}: nested GGUF arrays are not supported")
            return [self.value(item) for _ in range(count)]
        if kind not in _GGUF_SCALARS:
            raise ValueError(f"{self.path}: unknown GGUF value type {kind}")
        return self.take(_GGUF_SCALARS[kind])


def read_gguf(path: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """A GGUF v3 file's metadata and tensors, which must be 1-D float32.

    Anything else, and any byte past the last tensor's alignment, is refused.
    """
    data = Path(path).read_bytes()
    cursor = _Cursor(path, data)
    if cursor.bytes(4) != _GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file")
    version = cursor.take("<I")
    if version != _GGUF_VERSION:
        raise ValueError(f"{path}: GGUF version {version}, expected {_GGUF_VERSION}")
    tensor_count, key_count = cursor.take("<QQ")

    metadata: dict[str, object] = {}
    for _ in range(key_count):
        key = cursor.string()
        if key in metadata:
            raise ValueError(f"{path}: duplicate GGUF key {key!r}")
        metadata[key] = cursor.value(cursor.take("<I"))

    infos = []
    for _ in range(tensor_count):
        name = cursor.string()
        dims = cursor.take(f"<{cursor.take('<I')}Q")
        dims = (dims,) if isinstance(dims, int) else dims
        kind, offset = cursor.take("<IQ")
        if kind != _GGUF_F32:
            raise ValueError(f"{path}: tensor {name} is GGML type {kind}, not F32")
        if len(dims) != 1:
            raise ValueError(f"{path}: tensor {name} has shape {dims}, not 1-D")
        infos.append((name, dims[0], offset))

    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0:
        raise ValueError(f"{path}: general.alignment is {alignment!r}")
    base = -(-cursor.offset // alignment) * alignment
    tensors: dict[str, np.ndarray] = {}
    end = base
    for name, count, offset in infos:
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        start = base + offset
        if start % alignment or start + 4 * count > len(data):
            raise ValueError(f"{path}: tensor {name} lies outside the file")
        tensors[name] = np.frombuffer(data, "<f4", count, start).copy()
        end = max(end, start + 4 * count)
    if len(data) - end >= alignment:
        raise ValueError(
            f"{path}: {len(data) - end} unexpected bytes after the last tensor"
        )
    return metadata, tensors


def _layer_list(text: str, what: str) -> list[int]:
    try:
        return sorted(int(item) for item in text.split(",") if item.strip())
    except ValueError:
        raise ValueError(f"{what} is not a comma list of layer ids: {text!r}") from None


def read_glp(
    path: str, num_layers: int, hidden_size: int
) -> tuple[dict[int, np.ndarray], float | None]:
    """The file's unit directions by layer, and its ``glp.alpha_default``.

    Refuses whatever this reader cannot apply exactly (spec, Reader
    conformance): another mode, spec version, rank or hook point; a direction
    for layer 0, outside the model, of another width, not finite or not unit
    norm; a tensor that is not a direction; a layer list that disagrees with the
    tensor names.
    """
    metadata, tensors = read_gguf(path)
    mode = metadata.get("glp.mode")
    if mode != "project":
        raise ValueError(
            f"{path}: glp.mode is {mode!r}; this runtime projects ('project') "
            "and never adds a control vector (an absent mode means 'add')"
        )
    version = metadata.get("glp.spec_version")
    if type(version) is not int or version != SPEC_VERSION:
        raise ValueError(
            f"{path}: glp.spec_version is {version!r}; this reader implements "
            f"version {SPEC_VERSION} (rank 1, no alpha scaling keys)"
        )
    rank = metadata.get("glp.rank", 1)
    if type(rank) is not int or rank != 1:
        raise ValueError(f"{path}: glp.rank is {rank!r}; this reader applies rank 1")
    hook = metadata.get("glp.hook_point")
    if hook != HOOK_POINT:
        raise ValueError(
            f"{path}: glp.hook_point is {hook!r}; this runtime applies at "
            f"{HOOK_POINT} only"
        )
    derived_at = metadata.get("glp.derived_at", hook)
    if derived_at != hook:
        logger.warning(
            "%s: transferred vector, derived at %s and applied at %s",
            path,
            derived_at,
            hook,
        )
    alpha = metadata.get("glp.alpha_default")
    if alpha is not None and not (isinstance(alpha, float) and math.isfinite(alpha)):
        raise ValueError(f"{path}: glp.alpha_default is {alpha!r}")

    directions: dict[int, np.ndarray] = {}
    for name, values in tensors.items():
        match = _DIRECTION.fullmatch(name)
        if match is None:
            raise ValueError(f"{path}: tensor {name} is not a direction.<layer>")
        layer = int(match.group(1))
        if not 1 <= layer < num_layers:
            raise ValueError(
                f"{path}: {name} is outside layers 1..{num_layers - 1} "
                "(direction.0 cannot be expressed)"
            )
        if values.size != hidden_size:
            raise ValueError(
                f"{path}: {name} has width {values.size}, not {hidden_size}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} is not finite")
        norm = float(np.linalg.norm(values.astype(np.float64)))
        if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            raise ValueError(f"{path}: {name} has norm {norm:.6f}, not 1")
        directions[layer] = (values.astype(np.float64) / norm).astype(np.float32)
    if not directions:
        raise ValueError(f"{path}: no direction tensors")
    declared = metadata.get("glp.layer_ids_zero_based")
    if declared is not None and _layer_list(str(declared), path) != sorted(directions):
        raise ValueError(
            f"{path}: glp.layer_ids_zero_based {declared!r} disagrees with the "
            f"direction tensors {sorted(directions)}; re-export the file"
        )
    logger.info(
        "GLP file %s: %d directions, layers %s, alpha_default %s, base %s@%s",
        path,
        len(directions),
        sorted(directions),
        alpha,
        metadata.get("general.base_model.0.name"),
        metadata.get("general.base_model.0.version"),
    )
    return directions, alpha


class GLPSteering(nn.Module):
    """Steers the marked rows of a covered layer's outputs, in place."""

    def __init__(
        self,
        directions: torch.Tensor,
        alpha: float,
        hc_mult: int,
        layers: list[int],
        max_tokens: int,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.layers = frozenset(layers)
        # One float32 row per model layer; unsteered layers are zero rows.
        self.register_buffer("directions", directions, persistent=False)
        # A tensor, not a Python float, which would be a graph constant.
        self.register_buffer(
            "alpha", torch.tensor([alpha], dtype=torch.float32), persistent=False
        )
        # The step's row marks (int32, 0 or 1): the model state marks the rows of
        # requests made to the steered model name, in place before every step, so
        # a captured graph reads the replayed step's.
        self.register_buffer(
            "rows", torch.zeros(max_tokens, dtype=torch.int32), persistent=False
        )
        # True when the current step schedules at least one steered request (or
        # during CUDA-graph capture); eager steps with no steered request skip the
        # per-layer glp_steer launches.
        self.step_active: bool = True

    def declare_kernels(self) -> None:
        """The steering of the model's streams, for a layer's write and, when
        sequence-parallel steps hand it on unreduced, for the ranks' parts."""
        worlds = [1]
        if sp_prefill.unreduced_enabled():
            from vllm.distributed import get_tensor_model_parallel_world_size

            world = get_tensor_model_parallel_world_size()
            if world > 1:
                worlds.append(world)
        tilelang_glp.declare(self.hc_mult, self.directions.shape[1], tuple(worlds))

    def step_rows(self, num_tokens: int) -> torch.Tensor:
        """The marks of a step's ``num_tokens`` rows, a view of ``rows``."""
        if num_tokens > self.rows.shape[0]:
            raise ValueError(
                f"GLP steering marks at most {self.rows.shape[0]} rows, "
                f"not {num_tokens}"
            )
        return self.rows[:num_tokens]

    def steer(
        self,
        layer: int,
        hidden_states,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        """Steer layer ``layer``'s outputs in the rows ``rows`` marks.

        A marked row's write is folded into its streams and projected, and the
        result left as its ``residual``, with ``res_mix`` the identity and
        ``post_mix`` zero, so the next layer's fold gives it back. Other rows, and
        every row of a layer the file does not cover, are untouched.
        """
        if layer not in self.layers:
            return
        parts = (
            hidden_states.packed
            if isinstance(hidden_states, RankOrderParts)
            else hidden_states.unsqueeze(0)
        )
        tokens = residual.shape[0]
        if (
            residual.dtype != torch.bfloat16
            or residual.shape[1:] != (self.hc_mult, self.directions.shape[1])
            or parts.shape[1:] != (tokens, self.directions.shape[1])
            or rows.shape != (tokens,)
        ):
            raise ValueError(
                f"GLP steering takes [T, {self.hc_mult}, {self.directions.shape[1]}] "
                f"BF16 streams with a write and a mark per row, not streams "
                f"{list(residual.shape)} {residual.dtype}, write {list(parts.shape)} "
                f"and marks {list(rows.shape)}"
            )
        if not self.step_active and not (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            return
        tilelang_glp.glp_steer(
            parts,
            residual,
            post_mix,
            res_mix,
            rows,
            self.directions[layer],
            self.alpha,
        )


def glp_steering(
    num_layers: int, hidden_size: int, hc_mult: int, max_tokens: int
) -> GLPSteering | None:
    """The steering WEIGHTLESS_STEER_PATH and WEIGHTLESS_STEER_MODEL ask for, or
    None when neither is set."""
    path, model = envs.WEIGHTLESS_STEER_PATH, envs.WEIGHTLESS_STEER_MODEL
    if path is None and model is None:
        return None
    if path is None or model is None:
        raise ValueError(
            "GLP steering needs both WEIGHTLESS_STEER_PATH (the GLP file) and "
            "WEIGHTLESS_STEER_MODEL (the served model name it steers), not "
            f"WEIGHTLESS_STEER_PATH={path!r} and WEIGHTLESS_STEER_MODEL={model!r}"
        )
    directions, alpha = read_glp(path, num_layers, hidden_size)
    if alpha is None:
        raise ValueError(f"{path}: no glp.alpha_default")
    layers = sorted(directions)
    stack = torch.zeros((num_layers, hidden_size), dtype=torch.float32)
    for layer in layers:
        stack[layer] = torch.from_numpy(directions[layer])
    logger.info(
        "GLP steering on: alpha %s, layers %s, requests to %r", alpha, layers, model
    )
    return GLPSteering(stack, alpha, hc_mult, layers, max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a GLP file against this runtime's reader without serving."
    )
    parser.add_argument("path")
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=5120)
    args = parser.parse_args()
    metadata, _ = read_gguf(args.path)
    directions, alpha = read_glp(args.path, args.num_layers, args.hidden_size)
    for key in sorted(metadata):
        print(f"{key} = {metadata[key]!r}")
    print(f"directions: {len(directions)}, layers {sorted(directions)}, alpha {alpha}")


if __name__ == "__main__":
    main()
