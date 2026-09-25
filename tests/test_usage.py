"""Tests for token accounting and cost reporting."""

import pytest

from virtual_lab.constants import DEFAULT_ENCODING, MODEL_TO_INPUT_PRICE_PER_TOKEN
from virtual_lab.utils import (
    MeetingUsage,
    count_discussion_tokens,
    count_tokens,
    compute_token_cost,
)

from conftest import TEST_MODEL, make_usage


def test_default_encoding_matches_the_default_model() -> None:
    # gpt-4o and later use o200k_base; cl100k_base undercounts their tokens
    assert DEFAULT_ENCODING == "o200k_base"
    assert count_tokens("hello world") == count_tokens("hello world", "o200k_base")


class TestDiscussionTokenCounts:
    """Tool output is sent to the model, so it is input, and it is never model output."""

    @staticmethod
    def discussion() -> list[dict[str, str]]:
        return [
            {"agent": "User", "message": "Here is the agenda."},
            {"agent": "Immunologist", "message": "I will search the literature."},
            {"agent": "Tool", "message": "A very long article " * 500},
            {"agent": "Immunologist", "message": "Based on that article, here is my answer."},
        ]

    def test_tool_output_is_not_counted_as_model_output(self) -> None:
        counts = count_discussion_tokens(self.discussion())
        agent_output = count_tokens("I will search the literature.") + count_tokens(
            "Based on that article, here is my answer."
        )

        assert counts["output"] == agent_output

    def test_tool_output_is_reported_separately(self) -> None:
        counts = count_discussion_tokens(self.discussion())

        assert counts["tool"] == count_tokens("A very long article " * 500)

    def test_tool_output_is_counted_once_within_input(self) -> None:
        # The tool turn reaches the model as part of the prefix of the following turn, so it
        # already appears in the input count and must not be added again.
        counts = count_discussion_tokens(self.discussion())

        assert counts["input"] > counts["tool"]
        assert counts["input"] < 2 * counts["tool"]

    def test_tool_key_present_when_no_tools_were_used(self) -> None:
        counts = count_discussion_tokens([{"agent": "Immunologist", "message": "Hello."}])

        assert counts["tool"] == 0


class TestMeetingUsage:
    """MeetingUsage records what the API reports rather than re-tokenizing the transcript."""

    def test_accumulates_across_calls(self) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage(prompt_tokens=100, completion_tokens=20))
        usage.add(TEST_MODEL, make_usage(prompt_tokens=300, completion_tokens=40))

        assert usage.input_tokens == 400
        assert usage.output_tokens == 60
        assert usage.num_calls == 2
        assert usage.max_input_tokens == 300

    def test_tracks_each_model_separately(self) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage(prompt_tokens=100, completion_tokens=10))
        usage.add("o3-mini-2025-01-31", make_usage(prompt_tokens=50, completion_tokens=5))

        assert set(usage.per_model) == {TEST_MODEL, "o3-mini-2025-01-31"}
        assert usage.per_model[TEST_MODEL].input_tokens == 100
        assert usage.per_model["o3-mini-2025-01-31"].input_tokens == 50

    def test_prices_each_model_at_its_own_rate(self) -> None:
        cheap, expensive = "gpt-4o-mini-2024-07-18", TEST_MODEL
        usage = MeetingUsage()
        usage.add(cheap, make_usage(prompt_tokens=1000, completion_tokens=0))
        usage.add(expensive, make_usage(prompt_tokens=1000, completion_tokens=0))

        expected = 1000 * (
            MODEL_TO_INPUT_PRICE_PER_TOKEN[cheap] + MODEL_TO_INPUT_PRICE_PER_TOKEN[expensive]
        )

        assert usage.compute_cost() == pytest.approx(expected)
        # A single blended rate would have produced the same number for both models
        assert compute_token_cost(cheap, 1000, 0) != compute_token_cost(expensive, 1000, 0)

    def test_cached_input_is_reported_but_not_double_counted(self) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage(prompt_tokens=1000, cached_tokens=400))

        assert usage.input_tokens == 1000
        assert usage.cached_input_tokens == 400

    def test_reasoning_tokens_are_a_subset_of_output(self) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage(completion_tokens=100, reasoning_tokens=70))

        assert usage.output_tokens == 100
        assert usage.reasoning_tokens == 70

    def test_missing_usage_is_ignored(self) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, None)

        assert usage.num_calls == 0
        assert usage.compute_cost() == 0.0

    def test_unknown_model_cost_is_an_error(self) -> None:
        usage = MeetingUsage()
        usage.add("some-unreleased-model", make_usage())

        assert usage.num_calls == 1

        with pytest.raises(ValueError, match="some-unreleased-model"):
            usage.compute_cost()

    def test_unknown_model_does_not_break_the_summary(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # An unpriced model must still report its token counts rather than crashing the meeting
        usage = MeetingUsage()
        usage.add("some-unreleased-model", make_usage(prompt_tokens=123))
        usage.print_summary(elapsed_time=1.0)

        output = capsys.readouterr().out

        assert "123" in output
        assert "Warning" in output

    def test_model_prefixes_are_matched(self) -> None:
        usage = MeetingUsage()
        usage.add(
            "gpt-4o-2024-08-06-ft-personal-abc123",
            make_usage(prompt_tokens=1000, completion_tokens=0),
        )

        assert usage.compute_cost() == pytest.approx(
            1000 * MODEL_TO_INPUT_PRICE_PER_TOKEN[TEST_MODEL]
        )

    def test_summary_reports_every_model(self, capsys: pytest.CaptureFixture) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage())
        usage.add("o3-mini-2025-01-31", make_usage())
        usage.print_summary(elapsed_time=1.0)

        output = capsys.readouterr().out

        assert TEST_MODEL in output
        assert "o3-mini-2025-01-31" in output
