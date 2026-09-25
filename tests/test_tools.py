"""Tests for the tool registry and the tool-calling loop."""

import json

import openai
import pytest

from virtual_lab.agent import Agent
from virtual_lab.constants import MAX_TOOL_ITERATIONS
from virtual_lab.run_meeting import run_meeting
from virtual_lab.tools import PUBMED_TOOL, Tool, run_tool_calls

from conftest import FakeClient, text_response, tool_call_response


@pytest.fixture
def lookup_calls() -> list[str]:
    return []


@pytest.fixture
def lookup_tool(lookup_calls: list[str]) -> Tool:
    def lookup(accession: str) -> str:
        lookup_calls.append(accession)
        return f"UniProt {accession}: spike glycoprotein"

    return Tool(
        name="uniprot_lookup",
        description="Look up a protein by accession.",
        parameters={
            "type": "object",
            "properties": {"accession": {"type": "string"}},
            "required": ["accession"],
        },
        function=lookup,
    )


@pytest.fixture
def failing_tool() -> Tool:
    def fail() -> str:
        raise ConnectionError("PubMed unreachable")

    return Tool(
        name="broken_tool",
        description="Always fails.",
        parameters={"type": "object", "properties": {}},
        function=fail,
    )


def tool_turns(discussion: list[dict[str, str]]) -> list[str]:
    return [turn["message"] for turn in discussion if turn["agent"] == "Tool"]


class TestToolDefinition:
    def test_definition_is_valid_api_json(self, lookup_tool: Tool) -> None:
        definition = lookup_tool.definition

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "uniprot_lookup"
        assert definition["function"]["parameters"] == lookup_tool.parameters
        json.dumps(definition)

    def test_pubmed_tool_is_an_ordinary_tool(self) -> None:
        assert isinstance(PUBMED_TOOL, Tool)
        assert PUBMED_TOOL.name == "pubmed_search"


class TestRunToolCalls:
    def test_runs_the_matching_tool(self, lookup_tool: Tool, lookup_calls: list[str]) -> None:
        call = tool_call_response("uniprot_lookup", {"accession": "P0DTC2"}).choices[0].message
        outputs, messages = run_tool_calls(call.tool_calls, (lookup_tool,))

        assert lookup_calls == ["P0DTC2"]
        assert outputs == ["UniProt P0DTC2: spike glycoprotein"]
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"

    def test_tool_failure_is_reported_to_the_model(self, failing_tool: Tool) -> None:
        call = tool_call_response("broken_tool").choices[0].message
        outputs, _ = run_tool_calls(call.tool_calls, (failing_tool,))

        assert "ConnectionError" in outputs[0]
        assert "PubMed unreachable" in outputs[0]

    def test_unknown_tool_lists_what_is_available(self, lookup_tool: Tool) -> None:
        call = tool_call_response("does_not_exist").choices[0].message
        outputs, _ = run_tool_calls(call.tool_calls, (lookup_tool,))

        assert "unknown tool" in outputs[0]
        assert "uniprot_lookup" in outputs[0]

    def test_bad_arguments_are_reported_to_the_model(self, lookup_tool: Tool) -> None:
        call = tool_call_response("uniprot_lookup", {"wrong_argument": "x"}).choices[0].message
        outputs, _ = run_tool_calls(call.tool_calls, (lookup_tool,))

        assert "Error running tool" in outputs[0]


class TestToolLoopInMeeting:
    def test_tools_are_offered_again_after_a_result(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, lookup_calls, tmp_path
    ) -> None:
        # The second search is only possible if the tools are re-offered with the first result
        fake_client.completions.responses = [
            tool_call_response("uniprot_lookup", {"accession": "P0DTC2"}, "call_1"),
            tool_call_response("uniprot_lookup", {"accession": "P59594"}, "call_2"),
            text_response("Both proteins considered."),
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Compare two spike proteins.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        assert lookup_calls == ["P0DTC2", "P59594"]
        assert all(call["tools"] is not openai.NOT_GIVEN for call in fake_client.completions.calls)

    def test_tool_output_is_recorded_in_the_transcript(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, tmp_path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("uniprot_lookup", {"accession": "P0DTC2"}),
            text_response("Answer."),
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Look it up.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())

        assert tool_turns(discussion) == ["UniProt P0DTC2: spike glycoprotein"]

    def test_a_failing_tool_does_not_end_the_meeting(
        self, fake_client: FakeClient, team_member: Agent, failing_tool: Tool, tmp_path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("broken_tool"),
            text_response("Proceeding without the tool."),
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Try the tool.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(failing_tool,),
            num_rounds=0,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())

        assert "ConnectionError" in tool_turns(discussion)[0]
        assert discussion[-1]["message"] == "Proceeding without the tool."

    def test_runaway_tool_use_is_capped(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, lookup_calls, tmp_path
    ) -> None:
        # An agent that never stops asking for tools
        fake_client.completions.responses = [
            tool_call_response("uniprot_lookup", {"accession": f"A{index}"}, f"call_{index}")
            for index in range(MAX_TOOL_ITERATIONS + 5)
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Search forever.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        calls = fake_client.completions.calls

        assert len(lookup_calls) == MAX_TOOL_ITERATIONS
        assert len(calls) == MAX_TOOL_ITERATIONS + 1
        # Tools are withheld on the final attempt to force a text answer, and none are run
        assert calls[-1]["tools"] is openai.NOT_GIVEN
        assert all(call["tools"] is not openai.NOT_GIVEN for call in calls[:-1])

    def test_no_tools_means_none_are_offered(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="No tools here.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
        )

        assert fake_client.completions.calls[0]["tools"] is openai.NOT_GIVEN


class TestToolRegistration:
    def test_pubmed_search_shorthand_registers_the_tool(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Search the literature.",
            save_dir=tmp_path,
            team_member=team_member,
            pubmed_search=True,
            num_rounds=0,
        )

        offered = fake_client.completions.calls[0]["tools"]

        assert [tool["function"]["name"] for tool in offered] == ["pubmed_search"]

    def test_custom_tools_combine_with_pubmed_search(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Search and look up.",
            save_dir=tmp_path,
            team_member=team_member,
            pubmed_search=True,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        offered = fake_client.completions.calls[0]["tools"]

        assert {tool["function"]["name"] for tool in offered} == {
            "pubmed_search",
            "uniprot_lookup",
        }

    def test_pubmed_tool_passed_twice_is_deduplicated(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Search the literature.",
            save_dir=tmp_path,
            team_member=team_member,
            pubmed_search=True,
            tools=(PUBMED_TOOL,),
            num_rounds=0,
        )

        offered = fake_client.completions.calls[0]["tools"]

        assert len(offered) == 1

    def test_duplicate_tool_names_are_rejected(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="unique"):
            run_meeting(
                meeting_type="individual",
                agenda="Ambiguous tools.",
                save_dir=tmp_path,
                team_member=team_member,
                tools=(lookup_tool, lookup_tool),
                num_rounds=0,
            )
