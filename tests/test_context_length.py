"""Tests for the guard that stops a meeting before it overruns the context window."""

import pytest
from openai.types.chat import ChatCompletionMessageParam

from virtual_lab.agent import Agent
from virtual_lab.constants import CONTEXT_WARNING_THRESHOLD
from virtual_lab.utils import (
    ContextLengthExceededError,
    check_context_length,
    count_message_tokens,
    get_max_input_tokens,
)

from conftest import TEST_MODEL, FakeClient


def messages_of_size(num_tokens: int) -> list[ChatCompletionMessageParam]:
    """Builds a message list of roughly the requested token count."""
    return [{"role": "user", "content": "word " * num_tokens}]


class TestMaxInputTokens:
    def test_known_model(self) -> None:
        assert get_max_input_tokens(TEST_MODEL) == 128_000

    def test_matches_by_prefix(self) -> None:
        assert get_max_input_tokens("gpt-4o-2024-08-06-ft-personal-abc") == 128_000

    def test_unknown_model_returns_none(self) -> None:
        assert get_max_input_tokens("some-unreleased-model") is None


class TestCountMessageTokens:
    def test_counts_content_and_per_message_overhead(self) -> None:
        one = count_message_tokens([{"role": "user", "content": "hello"}])
        two = count_message_tokens(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello"}]
        )

        assert one > 0
        assert two > 2 * 0 + one

    def test_handles_tool_calls_without_content(self) -> None:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "pubmed_search", "arguments": '{"query": "spike"}'},
                    }
                ],
            }
        ]

        assert count_message_tokens(messages) > 0


class TestCheckContextLength:
    def test_oversized_request_raises(self) -> None:
        with pytest.raises(ContextLengthExceededError, match="exceeds"):
            check_context_length(messages=messages_of_size(150_000), model=TEST_MODEL)

    def test_error_names_the_model_and_the_limit(self) -> None:
        with pytest.raises(ContextLengthExceededError) as error:
            check_context_length(messages=messages_of_size(150_000), model=TEST_MODEL)

        assert TEST_MODEL in str(error.value)
        assert "128,000" in str(error.value)

    def test_request_near_the_limit_is_flagged(self) -> None:
        _, near_limit = check_context_length(messages=messages_of_size(110_000), model=TEST_MODEL)

        assert near_limit

    def test_small_request_is_not_flagged(self) -> None:
        estimated, near_limit = check_context_length(messages=messages_of_size(10), model=TEST_MODEL)

        assert not near_limit
        assert estimated < CONTEXT_WARNING_THRESHOLD * 128_000

    def test_unknown_model_is_never_blocked(self) -> None:
        # Guessing a limit for an unrecognised model would block valid requests
        estimated, near_limit = check_context_length(
            messages=messages_of_size(500_000), model="some-unreleased-model"
        )

        assert estimated > 128_000
        assert not near_limit


class TestGuardInsideMeeting:
    def test_oversized_meeting_fails_before_any_api_call(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        from virtual_lab.run_meeting import run_meeting

        with pytest.raises(ContextLengthExceededError):
            run_meeting(
                meeting_type="individual",
                agenda="word " * 150_000,
                save_dir=tmp_path,
                team_member=team_member,
                num_rounds=0,
            )

        assert fake_client.completions.calls == []

    def test_near_limit_warning_is_printed_once(
        self,
        fake_client: FakeClient,
        team_member: Agent,
        tmp_path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from virtual_lab.run_meeting import run_meeting

        run_meeting(
            meeting_type="individual",
            agenda="word " * 110_000,
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=1,
        )

        assert capsys.readouterr().out.count("may not fit for many more rounds") == 1
