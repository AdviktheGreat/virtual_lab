"""Tests for typed meeting outputs."""

import json

import pytest
from pydantic import BaseModel, Field

from virtual_lab.agent import Agent
from virtual_lab.constants import METADATA_DIR_NAME, OUTPUT_DIR_NAME
from virtual_lab.run_meeting import run_meeting
from virtual_lab.structured import StructuredOutputError, request_structured_output, save_output

from conftest import TEST_MODEL, FakeClient, make_usage, parsed_response, text_response


class Decision(BaseModel):
    """A stand-in schema for a meeting's conclusions.

    No field has a default, because strict structured output mode makes every field required and
    would ignore one.
    """

    recommendation: str = Field(description="What the meeting decided to do.")
    confidence: float = Field(description="How confident the team is, from 0 to 1.")
    open_questions: list[str] = Field(description="Anything the meeting could not settle.")


DECISION = Decision(
    recommendation="Use ESM log-likelihoods to rank mutations.",
    confidence=0.8,
    open_questions=["Which variants to prioritise?"],
)


def run_with_schema(team_member: Agent, save_dir, **kwargs):
    return run_meeting(
        meeting_type="individual",
        agenda="Design a nanobody.",
        save_dir=save_dir,
        team_member=team_member,
        num_rounds=kwargs.pop("num_rounds", 0),
        output_schema=Decision,
        **kwargs,
    )


def test_defaults_in_a_schema_are_ignored_by_strict_mode() -> None:
    # Recorded because it is surprising: a default does not make a field optional, the API
    # promotes it to required and the default never applies. Schemas should not rely on them.
    from openai.lib._pydantic import to_strict_json_schema

    class WithDefault(BaseModel):
        needed: str
        defaulted: list[str] = Field(default_factory=list)

    schema = to_strict_json_schema(WithDefault)

    assert set(schema["required"]) == {"needed", "defaulted"}


class TestRequestStructuredOutput:
    def test_returns_the_validated_instance(self, fake_client: FakeClient) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]

        result, response = request_structured_output(
            client=fake_client,
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "Decide."}],
            schema=Decision,
            temperature=0.2,
        )

        assert result == DECISION
        assert response.usage is not None

    def test_schema_is_sent_as_the_response_format(self, fake_client: FakeClient) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]

        request_structured_output(
            client=fake_client,
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "Decide."}],
            schema=Decision,
            temperature=0.2,
        )

        assert fake_client.completions.parse_calls[0]["response_format"] is Decision

    def test_a_refusal_is_reported_clearly(self, fake_client: FakeClient) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=None, refusal="I cannot comply with that.")
        ]

        with pytest.raises(StructuredOutputError, match="refused"):
            request_structured_output(
                client=fake_client,
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Decide."}],
                schema=Decision,
                temperature=0.2,
            )

    def test_an_unparseable_answer_is_reported_clearly(self, fake_client: FakeClient) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=None, content="not json at all")
        ]

        with pytest.raises(StructuredOutputError, match="not json at all"):
            request_structured_output(
                client=fake_client,
                model=TEST_MODEL,
                messages=[{"role": "user", "content": "Decide."}],
                schema=Decision,
                temperature=0.2,
            )


class TestSaveOutput:
    def test_writes_into_the_outputs_subdirectory(self, tmp_path) -> None:
        path = save_output(save_dir=tmp_path, save_name="discussion", output=DECISION)

        assert path == tmp_path / OUTPUT_DIR_NAME / "discussion.json"
        assert json.loads(path.read_text())["confidence"] == 0.8

    def test_output_is_not_matched_by_transcript_globs(self, tmp_path) -> None:
        save_output(save_dir=tmp_path, save_name="discussion_1", output=DECISION)

        assert list(tmp_path.glob("discussion_*.json")) == []


class TestMeetingWithSchema:
    def test_meeting_returns_the_validated_instance(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]
        result = run_with_schema(team_member, tmp_path)

        assert isinstance(result, Decision)
        assert result == DECISION

    def test_output_is_saved(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]
        run_with_schema(team_member, tmp_path)

        saved = json.loads((tmp_path / OUTPUT_DIR_NAME / "discussion.json").read_text())

        assert saved == DECISION.model_dump(mode="json")

    def test_extraction_is_a_separate_pass_after_the_meeting(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The structured answer should be drawn from the finished meeting, so the parse call
        # comes last and sees every prior turn
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]

        run_with_schema(team_member, tmp_path, num_rounds=1)

        assert len(fake_client.completions.parse_calls) == 1
        assert fake_client.completions.calls[-1] is fake_client.completions.parse_calls[0]

        messages = fake_client.completions.parse_calls[0]["messages"]

        assert sum(1 for message in messages if message["role"] == "assistant") == 3

    def test_the_closing_agent_does_the_extraction(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]

        run_meeting(
            meeting_type="team",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=(team_member,),
            num_rounds=1,
            output_schema=Decision,
        )

        system_prompt = fake_client.completions.parse_calls[0]["messages"][0]

        assert system_prompt["role"] == "system"
        assert team_lead.title in system_prompt["content"]

    def test_structured_output_appears_in_the_transcript(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]
        run_with_schema(team_member, tmp_path)

        discussion = json.loads((tmp_path / "discussion.json").read_text())

        assert json.loads(discussion[-1]["message"]) == DECISION.model_dump(mode="json")
        assert discussion[-1]["agent"] == team_member.title

    def test_extraction_is_recorded_in_the_provenance(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=DECISION, usage=make_usage(prompt_tokens=900, completion_tokens=40))
        ]
        run_with_schema(team_member, tmp_path)

        record = json.loads((tmp_path / METADATA_DIR_NAME / "discussion.json").read_text())
        extraction = next(turn for turn in record["turns"] if turn["kind"] == "structured_output")

        assert extraction["model"] == team_member.model
        assert extraction["name"] == team_member.name
        assert extraction["input_tokens"] == 900
        assert extraction["output_tokens"] == 40
        assert extraction["num_api_calls"] == 1

    def test_extraction_cost_is_included_in_the_total(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [
            text_response("Answer.", usage=make_usage(prompt_tokens=100, completion_tokens=10))
        ]
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=DECISION, usage=make_usage(prompt_tokens=200, completion_tokens=20))
        ]
        run_with_schema(team_member, tmp_path)

        record = json.loads((tmp_path / METADATA_DIR_NAME / "discussion.json").read_text())

        assert record["usage"]["num_calls"] == 2
        assert record["usage"]["input_tokens"] == 300
        assert record["usage"]["output_tokens"] == 30

    def test_turns_still_align_with_the_transcript(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=DECISION)]

        run_with_schema(team_member, tmp_path, num_rounds=1)

        discussion = json.loads((tmp_path / "discussion.json").read_text())
        record = json.loads((tmp_path / METADATA_DIR_NAME / "discussion.json").read_text())

        assert [turn["speaker"] for turn in record["turns"]] == [
            entry["agent"] for entry in discussion
        ]

    def test_schema_and_return_summary_are_mutually_exclusive(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="not both"):
            run_with_schema(team_member, tmp_path, return_summary=True)

    def test_a_refusal_leaves_a_partial_transcript(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The meeting itself succeeded; only the extraction failed, and that work is worth keeping
        fake_client.completions.parsed_responses = [
            parsed_response(parsed=None, refusal="I cannot comply.")
        ]

        with pytest.raises(StructuredOutputError):
            run_with_schema(team_member, tmp_path)

        assert (tmp_path / "partial" / "discussion.json").exists()
        assert not (tmp_path / OUTPUT_DIR_NAME).exists()

    def test_no_schema_means_no_parse_call_and_no_output_file(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
        )

        assert fake_client.completions.parse_calls == []
        assert not (tmp_path / OUTPUT_DIR_NAME).exists()
