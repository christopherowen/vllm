# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.utils import (
    check_sequence_repetition,
    check_stop,
    detect_thinking_loop,
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
        assert detect_thinking_loop(let_me_do_it * 4) == ("contiguous_phrase", 6, 4)

        # Interspersed 3x A + B + 2x A (shared ".\n" suffix between B and A)
        interspersed = let_me_do_it * 3 + read_lines + let_me_do_it * 2
        assert has_thinking_loop(interspersed)
        assert detect_thinking_loop(interspersed) == ("windowed_phrase", 6, 5)

        # Double interjection: 2x A + B + 2x A + C + 1x A
        double_inter = (
            let_me_do_it * 2
            + read_lines
            + let_me_do_it * 2
            + check_it
            + let_me_do_it
        )
        assert has_thinking_loop(double_inter)
        assert detect_thinking_loop(double_inter) == ("windowed_phrase", 6, 5)

        # Normal varied reasoning trace should not trigger
        varied = list(range(500, 700))
        assert not has_thinking_loop(varied)
        assert detect_thinking_loop(varied) is None

    def test_windowed_parallel_enumeration_does_not_false_positive(self):
        """Parallel list enumerations (e.g. Rank 0..4 with shared 7-token suffix)
        have zero contiguous A + A pairs and must not false-positive."""
        shared_suffix = [10, 11, 12, 13, 14, 15, 198]  # " shard is aligned to 16 bytes.\n"
        parallel_enum = []
        for rank in range(5):
            parallel_enum.extend([500, rank] + shared_suffix)
        assert not has_thinking_loop(parallel_enum)
        assert detect_thinking_loop(parallel_enum) is None

    def test_short_pattern_min_span_24_tokens(self):
        """Short 2-token and 4-token patterns require a >= 24-token repeated span
        so 8-element array initializers or short table borders do not misfire."""
        # 8x 2-token (", 0" * 8 = 16 tokens) and 4x 4-token (16 tokens) must not trigger
        assert not has_thinking_loop([44, 48] * 8)
        assert not has_thinking_loop([91, 92, 93, 94] * 4)

        # 12x 2-token (24 tokens) and 6x 4-token (24 tokens) must trigger
        assert detect_thinking_loop([44, 48] * 12) == ("contiguous_short", 2, 12)
        assert detect_thinking_loop([91, 92, 93, 94] * 6) == ("contiguous_phrase", 4, 6)

    def test_windowed_multi_clause_loop_beyond_two_internal_suffix_matches(self):
        """When internal clauses of a repeating paragraph share a >= 5-token ending,
        _find_windowed_repeating_pattern scans up to 4 candidate periods to find the
        full repeating phrase."""
        shared_clause_end = [901, 902, 903, 13, 198]
        clause_1 = [801, 802] + shared_clause_end
        clause_2 = [811, 812] + shared_clause_end
        clause_3 = [821, 822] + shared_clause_end
        paragraph = clause_1 + clause_2 + clause_3
        interjection = [701, 702, 703, 13, 198]
        trace = paragraph * 3 + interjection + paragraph * 2
        info = detect_thinking_loop(trace)
        assert info is not None
        assert info[0] == "windowed_phrase"

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
            stopped = check_stop(request, max_model_len=2048)
            if request.force_thinking_end:
                break

        assert not stopped
        assert request.force_thinking_end
        assert request.thinking_loop_breaks == 1
        assert request.last_loop_info == ("windowed_phrase", 6, 5)
        assert request.thinking_output_start == 0
        assert not request.is_finished()

        # Worker forces </think> (128806)
        prev_out = len(request._output_token_ids)
        request.append_output_token_ids(128806)
        request.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
        assert not request.force_thinking_end
        assert request.thinking_output_start == -1
        assert request.thinking_end_output_idx == len(request._output_token_ids)
        assert not check_stop(request, max_model_len=2048)

        # Normal tool-call tokens after </think> must not false-trigger on pre-</think> tokens
        for tok in [401, 402, 403, 404, 405]:
            prev_out = len(request._output_token_ids)
            request.append_output_token_ids(tok)
            request.update_reasoning_markers(
                start_ids, end_ids, prev_num_output=prev_out
            )
            assert not check_stop(request, max_model_len=2048)

        # If the model still loops after </think>, safety net finishes with FINISHED_REPETITION
        for tok in let_me_do_it * 4:
            prev_out = len(request._output_token_ids)
            request.append_output_token_ids(tok)
            request.update_reasoning_markers(
                start_ids, end_ids, prev_num_output=prev_out
            )
            if check_stop(request, max_model_len=2048):
                stopped = True
                break

        assert stopped
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"
        assert request.last_loop_info == ("contiguous_phrase", 6, 4)

    def test_default_loop_detection_post_think_and_thinking_off(self):
        """Loop detection is active by default after natural </think> (even when
        thinking_loop_breaks == 0 and repetition_detection is None) and when
        thinking is disabled (finishing with FINISHED_REPETITION without setting
        force_thinking_end)."""
        start_ids = (128805,)
        end_ids = (128806,)
        let_me_do_it = [101, 102, 103, 104, 13, 198]

        # 1. Natural </think> without an in-<think> loop break, followed by post-</think> loop
        req_post = Request(
            request_id="natural-post-think-loop",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=512),
            pooling_params=None,
        )
        for tok in [301, 302, 303, 128806]:
            prev_out = len(req_post._output_token_ids)
            req_post.append_output_token_ids(tok)
            req_post.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
            assert not check_stop(req_post, max_model_len=2048)
        assert req_post.thinking_loop_breaks == 0
        assert req_post.thinking_output_start == -1

        stopped = False
        for tok in let_me_do_it * 4:
            prev_out = len(req_post._output_token_ids)
            req_post.append_output_token_ids(tok)
            req_post.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
            if check_stop(req_post, max_model_len=2048):
                stopped = True
                break
        assert stopped
        assert not req_post.force_thinking_end
        assert req_post.status == RequestStatus.FINISHED_REPETITION
        assert req_post.stop_reason == "repetition_detected"

        # 2. Thinking disabled -> skips marker tracking & force_thinking_end, but still catches loops by default
        for init_kwargs in (
            {"prompt_token_ids": [1, 2, 128804, 128806]},
            {"reasoning_ended": True},
            {"sampling_params": SamplingParams(max_tokens=512, thinking_token_budget=0)},
            {"reasoning_parser_kwargs": {"chat_template_kwargs": {"thinking": False}}},
            {"reasoning_parser_kwargs": {"chat_template_kwargs": {"enable_thinking": False}}},
            {"reasoning_parser_kwargs": {"reasoning_effort": "none"}},
        ):
            req = Request(
                request_id="think-off-default-loop",
                prompt_token_ids=init_kwargs.get("prompt_token_ids", [1, 2, 128805, 198]),
                sampling_params=init_kwargs.get("sampling_params", SamplingParams(max_tokens=512)),
                pooling_params=None,
                reasoning_ended=init_kwargs.get("reasoning_ended"),
                reasoning_parser_kwargs=init_kwargs.get("reasoning_parser_kwargs"),
            )
            stopped = False
            for tok in let_me_do_it * 4:
                prev_out = len(req._output_token_ids)
                req.append_output_token_ids(tok)
                req.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
                assert req.thinking_disabled
                assert not req.force_thinking_end
                if check_stop(req, max_model_len=2048):
                    stopped = True
                    break
            assert stopped
            assert not req.force_thinking_end
            assert req.status == RequestStatus.FINISHED_REPETITION

        # 3. Explicit opt-out via RepetitionDetectionParams(max_pattern_size=0) disables detection
        req_opt_out = Request(
            request_id="explicit-opt-out",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(
                max_tokens=512,
                repetition_detection=RepetitionDetectionParams(max_pattern_size=0),
            ),
            pooling_params=None,
        )
        req_opt_out.append_output_token_ids(let_me_do_it * 5)
        assert not check_stop(req_opt_out, max_model_len=2048)

    def test_leading_newline_before_think_does_not_disable_tracking(self):
        """When <think> is emitted in the output after 1..3 leading newline/whitespace
        tokens, update_reasoning_markers must not prematurely latch thinking_disabled."""
        start_ids = (128805,)
        end_ids = (128806,)
        let_me_do_it = [101, 102, 103, 104, 13, 198]

        req = Request(
            request_id="leading-newline-think",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=512),
            pooling_params=None,
        )
        for tok in [198, 128805, 198]:
            prev_out = len(req._output_token_ids)
            req.append_output_token_ids(tok)
            req.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)

        assert not req.thinking_disabled
        assert req.thinking_output_start == 2

        for tok in let_me_do_it * 4:
            prev_out = len(req._output_token_ids)
            req.append_output_token_ids(tok)
            req.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
            check_stop(req, max_model_len=2048)
            if req.force_thinking_end:
                break

        assert req.force_thinking_end
        assert req.thinking_loop_breaks == 1

    def test_protocol_to_sampling_params_inherits_default_repetition_detection(self):
        """Test that ChatCompletionRequest and ResponsesRequest inherit
        server-default repetition_detection by default."""
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

        for kwargs in (
            {},
            {"thinking": {"type": "disabled"}},
            {"chat_template_kwargs": {"thinking": False}},
            {"reasoning_effort": "none"},
        ):
            req = ChatCompletionRequest(messages=msgs, **kwargs)
            sp = req.to_sampling_params(128, default_sampling_params)
            assert sp.repetition_detection is not None

        for r_kwargs in (
            {},
            {"reasoning": {"effort": "none"}},
            {"chat_template_kwargs": {"thinking": False}},
        ):
            resp = ResponsesRequest(input="hi", **r_kwargs)
            assert (
                resp.to_sampling_params(128, default_sampling_params).repetition_detection
                is not None
            )

    def test_thinking_enabled_non_looping_state_invariance(self):
        """Non-looping thinking-enabled requests keep force_thinking_end=False,
        thinking_loop_breaks=0, last_loop_info=None, and copy_draft_alloc unchanged."""
        start_ids = (128805,)
        end_ids = (128806,)
        req = Request(
            request_id="thinking-bit-identical",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=512),
            pooling_params=None,
        )
        req.copy_draft_alloc = 6
        stream = list(range(600, 680)) + [128806] + list(range(700, 740))
        for tok in stream:
            prev_out = len(req._output_token_ids)
            req.append_output_token_ids(tok)
            req.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
            assert not check_stop(req, max_model_len=2048)
            assert not req.force_thinking_end
            assert req.thinking_loop_breaks == 0
            assert req.last_loop_info is None
            assert req.copy_draft_alloc == 6
