"""Tests for the provenance record written alongside each transcript."""

import json

import pytest

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.constants import METADATA_DIR_NAME, PARTIAL_MEETING_DIR_NAME
from virtual_lab.prompts import SCIENTIFIC_CRITIC
from virtual_lab.run_meeting import run_meeting
from virtual_lab.tools import PUBMED_TOOL, Tool

from conftest import TEST_MODEL, FakeClient, make_usage, text_response, tool_call_response


def read_record(save_dir, save_name: str = "discussion") -> dict:
    return json.loads((save_dir / METADATA_DIR_NAME / f"{save_name}.json").read_text())


def run_simple_meeting(team_member: Agent, save_dir, **kwargs) -> dict:
    run_meeting(
        meeting_type="individual",
        agenda="Design a nanobody.",
        save_dir=save_dir,
        team_member=team_member,
        num_rounds=kwargs.pop("num_rounds", 0),
        **kwargs,
    )

    return read_record(save_dir)


class TestRecordIsWritten:
    def test_record_accompanies_the_transcript(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_simple_meeting(team_member, tmp_path)

        assert (tmp_path / "discussion.json").exists()
        assert (tmp_path / METADATA_DIR_NAME / "discussion.json").exists()

    def test_transcript_format_is_unchanged(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # 1,554 existing transcripts and several readers depend on a flat list of turns
        run_simple_meeting(team_member, tmp_path)
        discussion = json.loads((tmp_path / "discussion.json").read_text())

        assert isinstance(discussion, list)
        assert set(discussion[0]) == {"agent", "message"}

    def test_record_is_not_matched_by_transcript_globs(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # A sibling "discussion_1.meta.json" would match "discussion_*.json" and be loaded as
        # if it were a transcript, so the record lives in a subdirectory instead
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            save_name="discussion_1",
            team_member=team_member,
            num_rounds=0,
        )

        assert [path.name for path in tmp_path.glob("discussion_*.json")] == ["discussion_1.json"]
        assert list(tmp_path.glob("*.json")) == [tmp_path / "discussion_1.json"]

    def test_record_is_valid_json_with_no_stray_objects(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # Everything recorded must survive a JSON round trip, including usage and turns
        record = run_simple_meeting(team_member, tmp_path)

        assert json.loads(json.dumps(record)) == record


class TestRunLevelFields:
    def test_versions_are_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path)

        assert record["virtual_lab_version"] == __version__
        assert record["openai_version"]
        assert record["python_version"]

    def test_configuration_is_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path, temperature=0.7, max_retries=3)

        assert record["meeting_type"] == "individual"
        assert record["save_name"] == "discussion"
        assert record["temperature"] == 0.7
        assert record["max_retries"] == 3
        assert record["num_rounds"] == 0

    def test_team_personas_are_recorded_in_full(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # Enough to reconstruct the agents, not just name them
        record = run_simple_meeting(team_member, tmp_path)
        member = next(agent for agent in record["team"] if agent["title"] == team_member.title)

        assert member == {
            "title": team_member.title,
            "name": team_member.name,
            "model": team_member.model,
            "expertise": team_member.expertise,
            "goal": team_member.goal,
            "role": team_member.role,
        }

    def test_critic_is_identified(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path)

        assert record["critic"]["title"] == SCIENTIFIC_CRITIC.title
        assert record["critic"]["model"] == team_member.model

    def test_team_meeting_has_no_critic(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="team",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=(team_member,),
            num_rounds=0,
        )

        assert read_record(tmp_path)["critic"] is None

    def test_tools_are_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path, pubmed_search=True)

        assert record["tools"] == [PUBMED_TOOL.name]

    def test_timing_is_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path)

        assert record["started_at"] < record["ended_at"]
        assert record["elapsed_seconds"] >= 0

    def test_usage_matches_what_the_api_reported(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [
            text_response("Answer.", usage=make_usage(prompt_tokens=700, completion_tokens=90))
        ]
        record = run_simple_meeting(team_member, tmp_path)

        assert record["usage"]["input_tokens"] == 700
        assert record["usage"]["output_tokens"] == 90
        assert record["usage"]["num_calls"] == 1
        assert record["usage"]["per_model"][TEST_MODEL]["input_tokens"] == 700
        assert record["usage"]["cost"] > 0

    def test_unpriced_model_records_null_cost(
        self, fake_client: FakeClient, tmp_path
    ) -> None:
        agent = Agent("Immunologist", "a", "b", "c", "some-unreleased-model")
        record = run_simple_meeting(agent, tmp_path)

        assert record["usage"]["cost"] is None
        assert record["usage"]["num_calls"] == 1


class TestTurnLevelFields:
    def test_turns_align_with_the_transcript(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=1,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())
        turns = read_record(tmp_path)["turns"]

        assert len(turns) == len(discussion)
        assert [turn["speaker"] for turn in turns] == [entry["agent"] for entry in discussion]
        assert [turn["index"] for turn in turns] == list(range(len(turns)))

    def test_each_response_records_its_model_and_author(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path)
        responses = [turn for turn in record["turns"] if turn["kind"] == "response"]

        assert responses
        for turn in responses:
            assert turn["model"] == team_member.model
            assert turn["name"] == team_member.name

    def test_prompts_have_no_model(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path)
        prompts = [turn for turn in record["turns"] if turn["kind"] == "prompt"]

        assert prompts
        for turn in prompts:
            assert turn["model"] is None
            assert turn["name"] is None

    def test_per_turn_usage_sums_to_the_meeting_total(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=1,
        )

        record = read_record(tmp_path)

        assert sum(turn["input_tokens"] for turn in record["turns"]) == record["usage"][
            "input_tokens"
        ]
        assert sum(turn["num_api_calls"] for turn in record["turns"]) == record["usage"][
            "num_calls"
        ]

    def test_tool_use_is_attributed_to_the_turn_that_caused_it(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # A local tool rather than PUBMED_TOOL, which is frozen and would hit the network
        lookup = Tool(
            name="uniprot_lookup",
            description="Look up a protein.",
            parameters={"type": "object", "properties": {"accession": {"type": "string"}}},
            function=lambda accession: f"UniProt {accession}",
        )
        fake_client.completions.responses = [
            tool_call_response("uniprot_lookup", {"accession": "P0DTC2"}),
            text_response("Answer citing the record."),
        ]

        record = run_simple_meeting(team_member, tmp_path, tools=(lookup,))

        response = next(turn for turn in record["turns"] if turn["kind"] == "response")

        assert response["tool_calls"] == ["uniprot_lookup"]
        # One call requested the tool, the second produced the answer
        assert response["num_api_calls"] == 2

        tool_turn = next(turn for turn in record["turns"] if turn["kind"] == "tool_output")

        assert tool_turn["speaker"] == "Tool"
        assert tool_turn["num_api_calls"] == 0


class TestFailedMeetings:
    @staticmethod
    def run_failing_meeting(fake_client: FakeClient, team_member: Agent, save_dir) -> dict:
        fake_client.completions.responses = [
            text_response("First answer."),
            RuntimeError("API is down"),
        ]

        with pytest.raises(RuntimeError):
            run_meeting(
                meeting_type="individual",
                agenda="Design a nanobody.",
                save_dir=save_dir,
                team_member=team_member,
                num_rounds=1,
            )

        return json.loads(
            (save_dir / PARTIAL_MEETING_DIR_NAME / METADATA_DIR_NAME / "discussion.json").read_text()
        )

    def test_failure_is_recorded_next_to_the_partial_transcript(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = self.run_failing_meeting(fake_client, team_member, tmp_path)

        assert record["status"] == "failed"
        assert record["error"] == {"type": "RuntimeError", "message": "API is down"}

    def test_completed_meeting_is_marked_completed(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = run_simple_meeting(team_member, tmp_path)

        assert record["status"] == "completed"
        assert record["error"] is None

    def test_usage_up_to_the_failure_is_kept(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        record = self.run_failing_meeting(fake_client, team_member, tmp_path)

        assert record["usage"]["num_calls"] == 1
        assert record["turns"]
