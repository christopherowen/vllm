# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests to the steered model name (WEIGHTLESS_STEER_MODEL).

One engine serves a model and, under a second served model name, the model
steered by a GLP file (``vllm/models/deepseek_v4_1/glp.py``). The OpenAI server
marks a request to the steered name in its sampling parameters' ``extra_args``;
the scheduler keys the request's cached prefix blocks by the mark, since steered
rows write different KV; the worker marks the request's rows for the steering
kernels. Unmarked requests are served as with steering off.
"""

from typing import Any

import vllm.envs as envs

EXTRA_ARGS_KEY = "weightless_steer"
# A block hash extra key no other kind of key can equal (cache salts and LoRA
# names are strings).
BLOCK_HASH_KEY = ("weightless_steer",)


def steered_model() -> str | None:
    """The served model name whose requests are steered, or None."""
    return envs.WEIGHTLESS_STEER_MODEL


def is_steered_model(model_name: str | None) -> bool:
    name = steered_model()
    return name is not None and model_name == name


def mark(extra_args: dict[str, Any] | None, steered: bool) -> dict[str, Any] | None:
    """``extra_args`` with the request's mark set, or cleared: only the server
    marks a request, never its caller."""
    extra = dict(extra_args or {})
    if steered:
        extra[EXTRA_ARGS_KEY] = True
    else:
        extra.pop(EXTRA_ARGS_KEY, None)
    return extra or None


def is_marked(sampling_params: Any) -> bool:
    """Whether ``sampling_params`` (or None) belong to a steered request."""
    extra = getattr(sampling_params, "extra_args", None)
    return bool(extra and extra.get(EXTRA_ARGS_KEY))
