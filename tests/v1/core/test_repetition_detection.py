# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.utils import (
    check_sequence_repetition,
    check_stop,
    has_thinking_loop,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

# ============================================================================
# UNIT TESTS - check_sequence_repetition function
# ============================================================================


class TestCheckSequenceRepetition:
    """Unit tests for the check_sequence_repetition function"""

    def test_simple_repetition_detected(self):
        """Test detection of simple repetitive patterns"""
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_repetition_below_min_count(self):
        """Test that pattern below min_count is not detected"""
        token_ids = [1, 2, 3, 1, 2, 3]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_two_token_pattern(self):
        """Test detection of 2-token patterns"""
        token_ids = [1, 2, 1, 2, 1, 2, 1, 2]
        params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=4,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_no_repetition_varied_sequence(self):
        """Test that non-repetitive sequences are not flagged"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        params = RepetitionDetectionParams(
            max_pattern_size=5,
            min_pattern_size=2,
            min_count=2,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_partial_repetition_not_detected(self):
        """Test that incomplete repetitions are not detected"""
        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 4]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert not check_sequence_repetition(token_ids, params)

    def test_empty_token_list(self):
        """Test with empty token list"""
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=2,
        )
        assert not check_sequence_repetition([], params)

    def test_detection_disabled_max_size_zero(self):
        """Test that zero max_pattern_size disables detection"""
        token_ids = [1, 2, 1, 2, 1, 2]
        params = RepetitionDetectionParams()
        assert not check_sequence_repetition(token_ids, params)

    def test_invalid_min_count(self):
        """Test that min_count < 2 returns False"""
        token_ids = [1, 2, 1, 2]
        params = RepetitionDetectionParams()
        assert not check_sequence_repetition(token_ids, params)

    def test_repetition_at_end_of_sequence(self):
        """Test detection when repetition occurs at the end"""
        token_ids = [1, 2, 3, 4, 5, 6, 5, 6, 5, 6]
        params = RepetitionDetectionParams(
            max_pattern_size=3,
            min_pattern_size=2,
            min_count=3,
        )
        assert check_sequence_repetition(token_ids, params)

    def test_large_pattern_many_repetitions(self):
        """Test large pattern repeated many times"""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8] * 5
        params = RepetitionDetectionParams(
            max_pattern_size=10,
            min_pattern_size=2,
            min_count=3,
        )
        assert check_sequence_repetition(token_ids, params)


# ============================================================================
# INTEGRATION TESTS - check_stop with repetition detection
# ============================================================================


class TestRepetitionDetectionIntegration:
    """Integration tests for repetition detection in check_stop"""

    def test_basic_repetition_stops_generation(self):
        """Test that repetition is detected and stops generation"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"

    def test_detection_disabled_no_stop(self):
        """Test that disabled detection doesn't stop generation"""
        params = SamplingParams(
            max_tokens=100,
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_repetition_respects_min_tokens(self):
        """Test that repetition detection respects min_tokens"""
        params = SamplingParams(
            min_tokens=10,
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_no_repetition_continues_generation(self):
        """Test that non-repetitive tokens don't stop generation"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 30, 40, 50, 60])
        assert not check_stop(request, max_model_len=1024)

    def test_pattern_at_size_boundary(self):
        """Test detection at exact pattern size boundary"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=3,
                min_pattern_size=3,
                min_count=2,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 30, 10, 20, 30])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION

    def test_multiple_pattern_sizes_checked(self):
        """Test that function checks pattern sizes in range"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_REPETITION

    def test_eos_takes_precedence_over_repetition(self):
        """Test that EOS token stops before repetition check"""
        params = SamplingParams(
            max_tokens=100,
            stop_token_ids=[999],
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 999])
        assert check_stop(request, max_model_len=1024)
        assert request.status == RequestStatus.FINISHED_STOPPED

    def test_min_pattern_size_filters_small_patterns(self):
        """Test that min_pattern_size filters out smaller patterns"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=3,
                min_count=3,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_high_repetition_threshold(self):
        """Test that high min_count requires many repetitions"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=5,
                min_pattern_size=2,
                min_count=5,
            ),
        )
        request = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=params,
            pooling_params=None,
        )
        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
        assert not check_stop(request, max_model_len=1024)

    def test_sampling_params_dict_coercion(self):
        """Test that dict repetition_detection is coerced to RepetitionDetectionParams"""
        params = SamplingParams(
            max_tokens=100,
            repetition_detection={
                "max_pattern_size": 30,
                "min_pattern_size": 5,
                "min_count": 3,
            },
        )
        assert isinstance(params.repetition_detection, RepetitionDetectionParams)
        assert params.repetition_detection.max_pattern_size == 30
        assert params.repetition_detection.min_pattern_size == 5
        assert params.repetition_detection.min_count == 3

    def test_has_thinking_loop_contiguous_and_interspersed(self):
        """Test detection of contiguous and interspersed reasoning loops"""
        let_me_do_it = [101, 102, 103, 104, 13, 198]  # "Let me do it.\n"
        read_lines = [201, 202, 203, 204, 205, 206, 207, 13, 198]  # "I'll read lines 25-60.\n"
        check_it = [301, 302, 303, 13, 198]

        # Contiguous 4x repeat of 6-token phrase
        assert has_thinking_loop(let_me_do_it * 4)

        # Interspersed 3x A + B + 2x A (shared ".\n" suffix between B and A)
        interspersed = let_me_do_it * 3 + read_lines + let_me_do_it * 2
        assert has_thinking_loop(interspersed)

        # Double interjection: 2x A + B + 2x A + C + 1x A
        double_inter = (
            let_me_do_it * 2
            + read_lines
            + let_me_do_it * 2
            + check_it
            + let_me_do_it
        )
        assert has_thinking_loop(double_inter)

        # Normal varied reasoning trace should not trigger
        varied = list(range(500, 700))
        assert not has_thinking_loop(varied)

    def test_thinking_loop_break_forces_end_then_recovers(self):
        """Test that a loop inside <think> sets force_thinking_end without stopping,
        clears force_thinking_end when </think> is emitted, and only stops with
        FINISHED_REPETITION if a loop occurs after </think>."""
        start_ids = (128805,)
        end_ids = (128806,)
        let_me_do_it = [101, 102, 103, 104, 13, 198]
        read_lines = [201, 202, 203, 204, 205, 206, 207, 13, 198]

        request = Request(
            request_id="think-loop-test",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=512),
            pooling_params=None,
        )

        # Stream the interspersed loop inside <think>
        loop_tokens = let_me_do_it * 3 + read_lines + let_me_do_it * 2
        stopped = False
        for tok in loop_tokens:
            prev_out = len(request._output_token_ids)
            request.append_output_token_ids(tok)
            request.update_reasoning_markers(
                start_ids, end_ids, prev_num_output=prev_out
            )
            stopped = check_stop(
                request, max_model_len=2048, thinking_loop_break=True
            )
            if request.force_thinking_end:
                break

        assert not stopped
        assert request.force_thinking_end
        assert request.thinking_loop_breaks == 1
        assert request.thinking_output_start == 0
        assert not request.is_finished()

        # Worker forces </think> (128806)
        prev_out = len(request._output_token_ids)
        request.append_output_token_ids(128806)
        request.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
        assert not request.force_thinking_end
        assert request.thinking_output_start == -1
        assert request.thinking_end_output_idx == len(request._output_token_ids)
        assert not check_stop(request, max_model_len=2048, thinking_loop_break=True)

        # Normal tool-call tokens after </think> must not false-trigger on pre-</think> tokens
        for tok in [401, 402, 403, 404, 405]:
            prev_out = len(request._output_token_ids)
            request.append_output_token_ids(tok)
            request.update_reasoning_markers(
                start_ids, end_ids, prev_num_output=prev_out
            )
            assert not check_stop(
                request, max_model_len=2048, thinking_loop_break=True
            )

        # If the model still loops after </think>, safety net finishes with FINISHED_REPETITION
        for tok in let_me_do_it * 4:
            prev_out = len(request._output_token_ids)
            request.append_output_token_ids(tok)
            request.update_reasoning_markers(
                start_ids, end_ids, prev_num_output=prev_out
            )
            if check_stop(request, max_model_len=2048, thinking_loop_break=True):
                stopped = True
                break

        assert stopped
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"

    def test_thinking_off_skips_repetition_guard(self):
        """Test that when thinking is disabled (via prompt ending in </think>,
        thinking_token_budget=0, reasoning_ended=True, or reasoning_parser_kwargs),
        the repetition guard does not run."""
        start_ids = (128805,)
        end_ids = (128806,)
        let_me_do_it = [101, 102, 103, 104, 13, 198]
        default_rep = RepetitionDetectionParams(
            max_pattern_size=64, min_pattern_size=5, min_count=3
        )

        # 1. Prompt ends with </think> (DeepSeek-V4.1 thinking=False prompt)
        req_prompt_off = Request(
            request_id="think-off-prompt",
            prompt_token_ids=[1, 2, 128804, 128806],
            sampling_params=SamplingParams(
                max_tokens=512, repetition_detection=default_rep
            ),
            pooling_params=None,
        )
        assert not req_prompt_off.thinking_disabled
        for tok in let_me_do_it * 5:
            prev_out = len(req_prompt_off._output_token_ids)
            req_prompt_off.append_output_token_ids(tok)
            req_prompt_off.update_reasoning_markers(
                start_ids, end_ids, prev_num_output=prev_out
            )
            assert req_prompt_off.thinking_disabled
            assert not check_stop(
                req_prompt_off, max_model_len=2048, thinking_loop_break=True
            )
            assert not req_prompt_off.force_thinking_end

        # 2. Explicit thinking-off flags at Request initialization
        for init_kwargs in (
            {"reasoning_ended": True},
            {
                "sampling_params": SamplingParams(
                    max_tokens=512,
                    thinking_token_budget=0,
                    repetition_detection=default_rep,
                )
            },
            {
                "reasoning_parser_kwargs": {
                    "chat_template_kwargs": {"thinking": False}
                }
            },
            {
                "reasoning_parser_kwargs": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            },
            {"reasoning_parser_kwargs": {"reasoning_effort": "none"}},
        ):
            params = init_kwargs.get(
                "sampling_params",
                SamplingParams(max_tokens=512, repetition_detection=default_rep),
            )
            req = Request(
                request_id="think-off-init",
                prompt_token_ids=[1, 2, 128805, 198],
                sampling_params=params,
                pooling_params=None,
                reasoning_ended=init_kwargs.get("reasoning_ended"),
                reasoning_parser_kwargs=init_kwargs.get("reasoning_parser_kwargs"),
            )
            assert req.thinking_disabled
            req.append_output_token_ids(let_me_do_it * 5)
            req.update_reasoning_markers(start_ids, end_ids, prev_num_output=0)
            assert not check_stop(req, max_model_len=2048, thinking_loop_break=True)

    def test_protocol_to_sampling_params_skips_default_repetition_when_thinking_off(
        self,
    ):
        """Test that ChatCompletionRequest and ResponsesRequest do not apply
        server-default repetition_detection when thinking is disabled."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        default_sampling_params = {
            "repetition_detection": {
                "max_pattern_size": 64,
                "min_pattern_size": 5,
                "min_count": 3,
            }
        }
        msgs = [{"role": "user", "content": "hi"}]

        # Thinking enabled by default -> inherits default repetition_detection
        req_on = ChatCompletionRequest(messages=msgs)
        sp_on = req_on.to_sampling_params(128, default_sampling_params)
        assert sp_on.repetition_detection is not None

        # Thinking disabled via each supported API switch -> skips default
        for kwargs in (
            {"thinking": {"type": "disabled"}},
            {"chat_template_kwargs": {"thinking": False}},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"reasoning_effort": "none"},
            {"thinking_token_budget": 0},
        ):
            req_off = ChatCompletionRequest(messages=msgs, **kwargs)
            sp_off = req_off.to_sampling_params(128, default_sampling_params)
            assert sp_off.repetition_detection is None

        # ResponsesRequest with reasoning.effort="none" or enable_thinking=False
        resp_on = ResponsesRequest(input="hi")
        assert (
            resp_on.to_sampling_params(
                128, default_sampling_params
            ).repetition_detection
            is not None
        )
        for r_kwargs in (
            {"reasoning": {"effort": "none"}},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"chat_template_kwargs": {"thinking": False}},
        ):
            resp_off = ResponsesRequest(input="hi", **r_kwargs)
            assert (
                resp_off.to_sampling_params(
                    128, default_sampling_params
                ).repetition_detection
                is None
            )
