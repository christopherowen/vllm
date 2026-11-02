# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.utils import (
    FORCED_REASONING_END_GRACE,
    OUTPUT_LOOP_PERSIST,
    OUTPUT_SOUP_PERSIST,
    _find_dominant_token_window,
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
            sampling_params=SamplingParams(max_tokens=4096),
            pooling_params=None,
        )
        for tok in [301, 302, 303, 128806]:
            prev_out = len(req_post._output_token_ids)
            req_post.append_output_token_ids(tok)
            req_post.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
            assert not check_stop(req_post, max_model_len=8192)
        assert req_post.thinking_loop_breaks == 0
        assert req_post.thinking_output_start == -1

        stopped = False
        for tok in let_me_do_it * 400:
            prev_out = len(req_post._output_token_ids)
            req_post.append_output_token_ids(tok)
            req_post.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
            if check_stop(req_post, max_model_len=8192):
                stopped = True
                break
        assert stopped
        # A healthy answer stops only once the loop has persisted.
        assert (
            req_post.num_output_tokens - req_post.output_loop_since
            == OUTPUT_LOOP_PERSIST
        )
        assert not req_post.force_thinking_end
        assert req_post.status == RequestStatus.FINISHED_REPETITION
        assert req_post.stop_reason == "repetition_detected"

        # 2. Thinking disabled -> skips marker tracking & force_thinking_end, but still catches loops by default
        for init_kwargs in (
            {"prompt_token_ids": [1, 2, 128804, 128806]},
            {"reasoning_ended": True},
            {"sampling_params": SamplingParams(max_tokens=4096, thinking_token_budget=0)},
            {"reasoning_parser_kwargs": {"chat_template_kwargs": {"thinking": False}}},
            {"reasoning_parser_kwargs": {"chat_template_kwargs": {"enable_thinking": False}}},
            {"reasoning_parser_kwargs": {"reasoning_effort": "none"}},
        ):
            req = Request(
                request_id="think-off-default-loop",
                prompt_token_ids=init_kwargs.get("prompt_token_ids", [1, 2, 128805, 198]),
                sampling_params=init_kwargs.get("sampling_params", SamplingParams(max_tokens=4096)),
                pooling_params=None,
                reasoning_ended=init_kwargs.get("reasoning_ended"),
                reasoning_parser_kwargs=init_kwargs.get("reasoning_parser_kwargs"),
            )
            stopped = False
            for tok in let_me_do_it * 400:
                prev_out = len(req._output_token_ids)
                req.append_output_token_ids(tok)
                req.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
                assert req.thinking_disabled
                assert not req.force_thinking_end
                if check_stop(req, max_model_len=8192):
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

    @staticmethod
    def _feed(req, tokens, start_ids=(128805,), end_ids=(128806,)):
        """Append tokens one at a time as the scheduler does; return the
        number fed when check_stop stopped the request, else None."""
        for i, tok in enumerate(tokens, 1):
            prev_out = len(req._output_token_ids)
            req.append_output_token_ids(tok)
            req.update_reasoning_markers(start_ids, end_ids, prev_num_output=prev_out)
            if check_stop(req, max_model_len=1 << 20):
                return i
        return None

    def test_token_soup_detected(self):
        """One token repeated with short interjections (the production 'Vila'
        collapse): every run is under 24 tokens and the period is irregular, so
        only the token-soup rule sees it."""
        vila, estud, compar, evelyn, laud = 125221, 96830, 6289, 88227, 100308
        incident = [vila] * 11 + [estud, compar] + [vila] * 9 + [estud, compar]
        incident += [vila] * 9
        assert detect_thinking_loop(incident) == ("token_soup", 3, 29)

        soup = []
        for run in (12, 7, 15, 9, 11, 13):
            soup += [vila] * run + [evelyn, laud]
        # Longer soups may also meet the windowed rule; either catches them.
        assert detect_thinking_loop(soup) is not None
        # Ending on an interjection: still caught.
        assert detect_thinking_loop(soup + [estud]) is not None
        # Interjections of new cone tokens every few tokens: only soup.
        cone = [evelyn, laud, estud, compar, 92036, 68908]
        mixed = []
        for i, run in enumerate((5, 3, 6, 4, 5, 3, 6, 4, 5, 6)):
            mixed += [vila] * run + [cone[i % len(cone)]]
        assert detect_thinking_loop(mixed)[0] == "token_soup"

    def test_token_soup_ignores_alternations_and_prose(self):
        """No single token dominates bit strings, zero matrices or prose."""
        import random

        rng = random.Random(0)
        bits = [rng.choice((15, 16)) for _ in range(64)]
        assert _find_dominant_token_window(bits) is None

        # [[0, 0, 0], [0, 0, 0], ...]: '0' is under 60% of the tokens.
        matrix = []
        for _ in range(8):
            matrix += [58, 15, 11, 15, 11, 15, 59, 11]
        assert _find_dominant_token_window(matrix) is None

        prose = [rng.randrange(1000, 60000) for _ in range(200)]
        assert detect_thinking_loop(prose) is None

    def test_reasoning_markers_follow_the_parser(self):
        """A start marker inside reasoning is reasoning text, an end marker in
        the answer is answer text, and a start marker in the answer opens
        reasoning again, as DeepSeekV4Parser decides."""
        req = Request(
            request_id="markers",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=4096),
            pooling_params=None,
        )
        assert self._feed(req, [501, 502, 128805, 503]) is None
        assert req.thinking_output_start == 0

        # A loop breaks reasoning; a repeated start marker before the forced
        # end must not cancel it.
        assert self._feed(req, [9] * 24) is None
        assert req.force_thinking_end and req.thinking_loop_breaks == 1
        assert self._feed(req, [128805]) is None
        assert req.force_thinking_end
        assert req.thinking_output_start == 0

        assert self._feed(req, [128806]) is None
        end_idx = req.thinking_end_output_idx
        assert end_idx == req.num_output_tokens
        assert req.thinking_output_start == -1 and not req.force_thinking_end

        # A stray end marker in the answer does not move the answer's start.
        assert self._feed(req, [601, 128806, 602]) is None
        assert req.thinking_end_output_idx == end_idx

        # A start marker in the answer opens reasoning again.
        assert self._feed(req, [128805, 701]) is None
        assert req.thinking_output_start == req.num_output_tokens - 1

    def test_forced_reasoning_end_that_never_arrives_stops(self):
        """Once a reasoning loop forces the end, the loop may run on through
        the steps in flight, but not past the grace."""
        req = Request(
            request_id="forced-end-lost",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=4096),
            pooling_params=None,
        )
        assert self._feed(req, [9] * 24) is None
        assert req.force_thinking_end
        forced_at = req.num_output_tokens
        assert req.force_thinking_end_at == forced_at

        fed = self._feed(req, [9] * (FORCED_REASONING_END_GRACE + 8))
        assert fed == FORCED_REASONING_END_GRACE + 1
        assert req.status == RequestStatus.FINISHED_REPETITION
        assert req.num_output_tokens - forced_at == FORCED_REASONING_END_GRACE + 1

    def test_healthy_answer_repetition_that_ends_is_not_stopped(self):
        """A zero-filled array or a table rule in a healthy answer repeats for
        a while and moves on: detections that stop before the persistence
        span cost nothing, and a later loop is timed from its own start."""
        req = Request(
            request_id="zeros",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=8192),
            pooling_params=None,
        )
        assert self._feed(req, [301, 302, 128806]) is None
        zeros = [44, 48] * 100  # ", 0" x 100: detected from the 24th token
        assert self._feed(req, zeros) is None
        assert req.output_loop_since >= 0
        tail = list(range(2000, 2080))
        assert self._feed(req, tail) is None
        assert req.output_loop_since == -1

        loop_start = req.num_output_tokens
        fed = self._feed(req, [101, 102, 103, 104, 13, 198] * 400)
        assert fed is not None
        assert req.status == RequestStatus.FINISHED_REPETITION
        assert req.output_loop_since > loop_start
        assert req.num_output_tokens - req.output_loop_since == OUTPUT_LOOP_PERSIST

    def test_answer_after_a_reasoning_loop_stops_at_its_first_loop(self):
        """After a forced end the context is degenerate: no persistence."""
        req = Request(
            request_id="degraded",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=4096),
            pooling_params=None,
        )
        assert self._feed(req, [9] * 24 + [128806]) is None
        assert req.thinking_loop_breaks == 1
        vila, laud = 125221, 100308
        fed = self._feed(req, ([vila] * 10 + [laud]) * 4)
        assert fed is not None and fed <= 33
        assert req.status == RequestStatus.FINISHED_REPETITION

    @staticmethod
    def _answer_request(request_id):
        """A thinking request whose reasoning closed naturally."""
        req = Request(
            request_id=request_id,
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=1 << 16),
            pooling_params=None,
        )
        assert TestRepetitionDetectionIntegration._feed(req, [301, 302, 128806]) is None
        assert req.thinking_loop_breaks == 0 and req.thinking_output_start == -1
        return req

    @staticmethod
    def _varied(n, seed):
        import random

        rng = random.Random(seed)
        return [rng.randrange(1000, 60000) for _ in range(n)]

    @staticmethod
    def _soup(n, seed):
        """The 2026-11-01 production collapse (#787): ' Vila' in runs of 3 to
        12 with interjections of one to three other tokens."""
        import random

        vila, cone = 125221, [96830, 6289, 88227, 100308, 92036, 68908, 3221]
        rng = random.Random(seed)
        soup = []
        while len(soup) < n:
            if rng.random() < 0.62:
                soup += [vila] * rng.randint(3, 12)
            else:
                soup += [rng.choice(cone) for _ in range(rng.randint(1, 3))]
        return soup[:n]

    def test_known_trigger_single_token_attractor_in_answer(self):
        """The production collapse in the visible answer after normal output
        stops soon after it starts, long before its 736 tokens of ' Vila'."""
        req = self._answer_request("vila-soup")
        assert self._feed(req, self._varied(1500, 1)) is None
        fed = self._feed(req, self._soup(2330, 6))
        assert fed is not None
        assert req.status == RequestStatus.FINISHED_REPETITION
        assert OUTPUT_SOUP_PERSIST < fed <= 512

    def test_periodic_loop_in_healthy_answer_stops_late(self):
        """An exactly periodic loop (a run of one token, a repeated phrase)
        stops only once it has persisted OUTPUT_LOOP_PERSIST tokens: before
        that it may still be legitimate output."""
        for name, loop in {"run": [125221], "phrase": [101, 102, 103, 104, 13, 198]}.items():
            req = self._answer_request(name)
            assert self._feed(req, self._varied(300, 7)) is None
            fed = self._feed(req, loop * (3000 // len(loop)))
            assert fed is not None, name
            assert req.num_output_tokens - req.output_loop_since == OUTPUT_LOOP_PERSIST
            assert req.status == RequestStatus.FINISHED_REPETITION

    def test_known_trigger_separator_rule_in_varied_answer_is_kept(self):
        """Legitimate messages in the same sessions carried 199-token runs of
        one token (separator rules) at a distinct-token ratio of 0.82-0.84.
        Runs up to 300 inside varied output are never stopped."""
        for run in (199, 300):
            req = self._answer_request(f"separator-{run}")
            stream = self._varied(400, 2) + [4521] * run + self._varied(400, 3)
            assert self._feed(req, stream) is None, run
            assert not req.is_finished()

    def test_known_trigger_requested_repetition_is_kept(self):
        """Requested repetition at its real token counts: 'hello' 200 times, a
        512-zero C array (three tokens a zero: 1,548 tokens), a 16x16 matrix
        of ones (792 tokens), a 200-row table and an answer that is one token
        100 times: legitimate, bounded, never stopped. The 2026-11-02 soak
        stopped a 128-zero array at its 292nd of 396 tokens."""
        hello, comma, space, zero, one = 23714, 11, 223, 15, 16
        open_row, close_row = [262, 58], [1592, 198]
        cases = {
            "hello-200": [hello] * 200,
            "zeros-512": [zero, comma, space] * 512,
            "ones-16x16": (open_row + [one, comma, space] * 16 + close_row) * 16,
            "table-200": [t for i in range(200) for t in (124, 2000 + i, 124, zero, 124, 198)],
            "one-token-100": [hello] * 100,
        }
        for name, stream in cases.items():
            req = self._answer_request(name)
            assert self._feed(req, stream + self._varied(40, 4)) is None, name

    def test_known_trigger_reasoning_stall_breaks_at_once(self):
        """The 2026-11-01 replay (#787): 'Let me do it.' repeated inside
        reasoning at effort max. Reasoning breaks at the first loop."""
        let_me_do_it = [101, 102, 103, 104, 13, 198]
        req = Request(
            request_id="stall",
            prompt_token_ids=[1, 2, 128805, 198],
            sampling_params=SamplingParams(max_tokens=4096),
            pooling_params=None,
        )
        assert self._feed(req, self._varied(120, 5)) is None
        assert self._feed(req, let_me_do_it * 4) is None
        assert req.force_thinking_end and req.thinking_loop_breaks == 1
        assert req.last_loop_info == ("contiguous_phrase", 6, 4)
