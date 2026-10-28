# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests to the steered model name: the server's mark, the response's model
name, the servers that refuse the name, and the prefix cache's key."""

import asyncio
from types import SimpleNamespace

import pytest

from vllm import weightless
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys

STOCK, STEERED = "model", "model-weightless"


@pytest.fixture(autouse=True)
def steered_name(monkeypatch):
    monkeypatch.setenv("WEIGHTLESS_STEER_MODEL", STEERED)


class _Models:
    lora_requests: dict = {}

    def is_base_model(self, name):
        return name in (STOCK, STEERED)

    def model_name(self, lora_request=None):
        return STOCK if lora_request is None else lora_request.lora_name


def _server(serves_steered_model: bool) -> BaseServing:
    server = BaseServing(_Models(), model_config=None)
    server.serves_steered_model = serves_steered_model
    return server


def _request(model, **fields):
    return SimpleNamespace(model=model, **fields)


def test_marks_set_and_clear():
    assert weightless.mark(None, True) == {weightless.EXTRA_ARGS_KEY: True}
    assert weightless.mark({"a": 1, weightless.EXTRA_ARGS_KEY: True}, False) == {"a": 1}
    assert weightless.mark({weightless.EXTRA_ARGS_KEY: True}, False) is None
    assert weightless.is_marked(SamplingParams(extra_args={"weightless_steer": True}))
    assert not weightless.is_marked(SamplingParams())
    assert not weightless.is_marked(None)


def test_no_steered_name_means_no_model_is_steered(monkeypatch):
    monkeypatch.delenv("WEIGHTLESS_STEER_MODEL")
    assert not weightless.is_steered_model(STEERED)
    assert not weightless.is_steered_model(None)
    params = SamplingParams(extra_args={"a": 1})
    _server(True)._mark_steering(_request(STEERED), params)
    assert params.extra_args == {"a": 1}


def test_a_request_to_the_steered_name_is_marked():
    params = SamplingParams(extra_args={"a": 1})
    _server(True)._mark_steering(_request(STEERED), params)
    assert params.extra_args == {"a": 1, weightless.EXTRA_ARGS_KEY: True}


def test_a_callers_mark_on_the_model_is_cleared():
    params = SamplingParams(extra_args={weightless.EXTRA_ARGS_KEY: True})
    _server(True)._mark_steering(_request(STOCK), params)
    assert not weightless.is_marked(params)


def test_beam_search_parameters_are_never_marked():
    params = BeamSearchParams(beam_width=2, max_tokens=4)
    _server(True)._mark_steering(_request(STEERED), params)
    assert not hasattr(params, "extra_args") or not params.extra_args


def test_a_response_reports_the_name_it_was_asked_for():
    server = _server(True)
    assert server._served_model_name(_request(STEERED), None) == STEERED
    assert server._served_model_name(_request(STOCK), None) == STOCK
    assert server._served_model_name(_request(None), None) == STOCK
    lora = SimpleNamespace(lora_name="adapter")
    assert server._served_model_name(_request(STEERED), lora) == "adapter"


def test_servers_that_cannot_mark_refuse_the_steered_name():
    refused = asyncio.run(_server(False)._check_model(_request(STEERED)))
    assert refused is not None and refused.error.param == "model"
    assert asyncio.run(_server(False)._check_model(_request(STOCK))) is None
    assert asyncio.run(_server(True)._check_model(_request(STEERED))) is None


def test_beam_search_refuses_the_steered_name():
    server = _server(True)
    beam = _request(STEERED, use_beam_search=True)
    assert asyncio.run(server._check_model(beam)) is not None
    stock = _request(STOCK, use_beam_search=True)
    assert asyncio.run(server._check_model(stock)) is None


def _cached_request(params, cache_salt=None):
    return SimpleNamespace(
        sampling_params=params,
        cache_salt=cache_salt,
        mm_features=[],
        lora_request=None,
        prompt_embeds=None,
    )


def test_a_steered_prefix_is_cached_apart():
    steered = _cached_request(SamplingParams(extra_args={"weightless_steer": True}))
    stock = _cached_request(SamplingParams())
    assert generate_block_hash_extra_keys(stock, 0, 16, 0) == (None, 0)
    keys, _ = generate_block_hash_extra_keys(steered, 0, 16, 0)
    assert keys == (weightless.BLOCK_HASH_KEY,)
    # The first block keys the chain, as a cache salt does.
    assert generate_block_hash_extra_keys(steered, 16, 32, 0) == (None, 0)


def test_no_cache_salt_takes_the_steered_key():
    salted = _cached_request(SamplingParams(), cache_salt="weightless_steer")
    steered = _cached_request(SamplingParams(extra_args={"weightless_steer": True}))
    assert (
        generate_block_hash_extra_keys(salted, 0, 16, 0)[0]
        != generate_block_hash_extra_keys(steered, 0, 16, 0)[0]
    )
