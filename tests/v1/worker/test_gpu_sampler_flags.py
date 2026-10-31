# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import tilelang_sampling
from vllm.v1.worker.gpu.sample.gumbel import warmup_processed_gumbel
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, id="penalty"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), True, id="bad-words"),
        pytest.param(SamplingParams(temperature=0.7), True, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, prompt_len=1, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, 1, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, 1, SamplingParams())

    assert not sampler.needs_logits_processing[3]


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(2, 1, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])


def test_seeded_filtered_sampler_uses_precompiled_target_kernel():
    """Exercise real UVA sampling states and the InputBatch int64 mapping ABI: the
    seeded target sampling of processed logits launches the Gumbel kernel the warmup
    compiled, and no other."""
    sampler = _make_sampler()
    warmup_processed_gumbel(VOCAB_SIZE, DEVICE)

    def gumbel_kernels():
        factory = id(tilelang_sampling.gumbel_sample_kernel)
        return {key for key in tilelang_sampling._KERNELS if key[0] == factory}

    compiled = gumbel_kernels()
    for req_idx in (0, 3):
        sampler.add_request(
            req_idx, 1, SamplingParams(temperature=1, top_p=0.95, seed=43)
        )
        sampler.apply_staged_writes()
        mapping_np = np.array([req_idx], dtype=np.intp)
        mapping = torch.from_numpy(mapping_np).to(DEVICE)
        logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.bfloat16, device=DEVICE)
        position = torch.ones(1, dtype=torch.int64, device=DEVICE)
        input_ids = torch.ones(1, dtype=torch.int32, device=DEVICE)
        local_position = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        sampled, processed = sampler.sample(
            logits,
            mapping,
            mapping,
            mapping_np,
            position,
            input_ids,
            local_position,
        )
        assert processed.dtype == torch.float32
        assert sampled.shape == (1,)
    torch.accelerator.synchronize()
    assert gumbel_kernels() == compiled, (
        "seeded filtered sampling compiled a Gumbel kernel after target warmup"
    )


def test_temperature_only_sampling_skips_fp32_copy_and_matches_fp32_path(monkeypatch):
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0, seed=11))
    sampler.add_request(1, 1, SamplingParams(temperature=1.0, seed=22))
    sampler.add_request(2, 1, SamplingParams(temperature=0.6, seed=33))
    sampler.apply_staged_writes()

    assert sampler.needs_logits_processing[2]
    assert not sampler.needs_non_temperature_logits_processing[2]

    mapping_np = np.array([0, 1, 2], dtype=np.intp)
    mapping = torch.from_numpy(mapping_np).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(1234)
    logits = torch.randn((3, VOCAB_SIZE), generator=gen, device=DEVICE).to(
        torch.bfloat16
    )
    position = torch.tensor([5, 6, 7], dtype=torch.int64, device=DEVICE)
    input_ids = torch.ones(3, dtype=torch.int32, device=DEVICE)
    local_position = torch.zeros(3, dtype=torch.int32, device=DEVICE)

    ref_processed = sampler.apply_sampling_params(
        logits,
        mapping,
        mapping,
        mapping_np,
        position,
        input_ids,
        local_position,
        skip_top_k_top_p=True,
    )
    assert ref_processed.dtype == torch.float32
    ref_sampled = gumbel_sample(
        ref_processed,
        mapping,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        position,
        apply_temperature=False,
        is_drafting=False,
    )

    def forbid_temp(*args, **kwargs):
        raise AssertionError("standalone apply_temperature should be skipped")

    monkeypatch.setattr(sampler.sampling_states, "apply_temperature", forbid_temp)
    sampled, processed = sampler.sample(
        logits,
        mapping,
        mapping,
        mapping_np,
        position,
        input_ids,
        local_position,
    )
    torch.accelerator.synchronize()

    assert processed.data_ptr() == logits.data_ptr()
    assert processed.dtype == torch.bfloat16
    assert torch.equal(sampled, ref_sampled)


def test_temperature_only_block_verification_skips_fp32_copy_and_matches(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0, seed=101))
    sampler.add_request(1, 1, SamplingParams(temperature=0.6, seed=202))
    sampler.add_request(2, 1, SamplingParams(temperature=1.3, seed=303))
    sampler.apply_staged_writes()

    rs = RejectionSampler.__new__(RejectionSampler)
    rs.sampler = sampler
    rs.watermark_key = None
    rs.sink = None
    rs.num_speculative_steps = 3
    rs.num_draft_steps = 3
    rs.enable_adaptive_verification = False
    rs.use_block_verification = True
    rs.synthetic_conditional_rates = None

    idx_mapping_np = np.array([0, 1, 2], dtype=np.intp)
    idx_mapping = torch.from_numpy(idx_mapping_np).to(DEVICE)
    cu_num_logits = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = idx_mapping.repeat_interleave(4)
    expanded_local_pos = torch.tensor([0, 1, 2, 3] * 3, dtype=torch.int32, device=DEVICE)
    pos = torch.arange(12, dtype=torch.int64, device=DEVICE)

    gen = torch.Generator(device=DEVICE).manual_seed(5678)
    logits = (
        torch.randn((12, VOCAB_SIZE), generator=gen, device=DEVICE) * 3.0
    ).to(torch.bfloat16)
    draft_sampled = torch.randint(
        0, VOCAB_SIZE, (12,), generator=gen, dtype=torch.int32, device=DEVICE
    )
    for r in range(11):
        logits[r, draft_sampled[r + 1]] += 4.0

    ref_processed = sampler.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        draft_sampled,
        expanded_local_pos,
    )
    assert ref_processed.dtype == torch.float32
    ref_sampled, ref_num_sampled = rejection_sample(
        ref_processed,
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        rs.num_speculative_steps,
        None,
        use_fp64=False,
        use_block_verification=True,
    )

    def forbid_temp(*args, **kwargs):
        raise AssertionError("standalone apply_temperature should be skipped")

    monkeypatch.setattr(sampler.sampling_states, "apply_temperature", forbid_temp)
    processed, sampled, num_sampled = rs._verify(
        logits,
        None,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        idx_mapping_np,
        expanded_idx_mapping,
        expanded_local_pos,
    )
    torch.accelerator.synchronize()

    assert processed.data_ptr() == logits.data_ptr()
    assert processed.dtype == torch.bfloat16
    assert torch.equal(num_sampled, ref_num_sampled)
    for req in range(3):
        n = int(num_sampled[req].item())
        assert torch.equal(sampled[req, :n], ref_sampled[req, :n])
