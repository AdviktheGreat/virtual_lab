"""Tests for the schemas that carry decisions between meetings."""

import json

import pytest
from pydantic import ValidationError

from virtual_lab.agent import Agent
from virtual_lab.constants import OUTPUT_DIR_NAME
from virtual_lab.run_meeting import run_meeting
from virtual_lab.schemas import AgentSpec, TeamRoster

from conftest import TEST_MODEL, FakeClient, parsed_response

IMMUNOLOGIST = AgentSpec(
    title="Immunologist",
    expertise="antibody engineering and immunogenicity",
    goal="design nanobodies with broad neutralising activity",
    role="advise on the immunogenicity of proposed designs",
)
STRUCTURAL_BIOLOGIST = AgentSpec(
    title="Structural Biologist",
    expertise="protein structure prediction",
    goal="model antibody-antigen complexes",
    role="assess whether proposed designs are structurally plausible",
)
ROSTER = TeamRoster(team_members=[IMMUNOLOGIST, STRUCTURAL_BIOLOGIST])


class TestAgentSpec:
    def test_builds_a_usable_agent(self) -> None:
        agent = IMMUNOLOGIST.to_agent(model=TEST_MODEL)

        assert isinstance(agent, Agent)
        assert agent.title == IMMUNOLOGIST.title
        assert agent.model == TEST_MODEL

    def test_fields_compose_the_system_prompt(self) -> None:
        # The field wording is chosen to complete these sentences, so a mismatch would read wrong
        prompt = IMMUNOLOGIST.to_agent(model=TEST_MODEL).prompt

        assert "You are a Immunologist." in prompt
        assert f"Your expertise is in {IMMUNOLOGIST.expertise}." in prompt
        assert f"Your goal is to {IMMUNOLOGIST.goal}." in prompt
        assert f"Your role is to {IMMUNOLOGIST.role}." in prompt

    def test_the_agent_has_a_valid_author_name(self) -> None:
        assert IMMUNOLOGIST.to_agent(model=TEST_MODEL).name == "Immunologist"
        assert STRUCTURAL_BIOLOGIST.to_agent(model=TEST_MODEL).name == "Structural_Biologist"

    def test_every_field_is_required(self) -> None:
        # Strict structured output mode rejects a schema with optional fields
        assert set(AgentSpec.model_fields) == {"title", "expertise", "goal", "role"}
        for field in AgentSpec.model_fields.values():
            assert field.is_required()

    def test_every_field_is_described_for_the_model(self) -> None:
        for name, field in AgentSpec.model_fields.items():
            assert field.description, f"{name} has no description to guide the model"


class TestTeamRoster:
    def test_builds_agents_in_order(self) -> None:
        agents = ROSTER.to_agents(model=TEST_MODEL)

        assert [agent.title for agent in agents] == ["Immunologist", "Structural Biologist"]
        assert all(agent.model == TEST_MODEL for agent in agents)

    def test_duplicate_titles_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Immunologist"):
            TeamRoster(team_members=[IMMUNOLOGIST, IMMUNOLOGIST])

    def test_duplicates_are_caught_before_a_meeting_would_fail(self) -> None:
        # run_meeting also rejects duplicate members, but only after the roster has been accepted
        with pytest.raises(ValidationError, match="unique"):
            TeamRoster(
                team_members=[
                    IMMUNOLOGIST,
                    AgentSpec(
                        title="Immunologist",
                        expertise="something else",
                        goal="something else",
                        role="something else",
                    ),
                ]
            )

    def test_agents_are_accepted_by_run_meeting(
        self, fake_client: FakeClient, team_lead: Agent, tmp_path
    ) -> None:
        # The point of the roster: its output can be handed straight to the next meeting
        run_meeting(
            meeting_type="team",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=ROSTER.to_agents(model=TEST_MODEL),
            num_rounds=1,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())
        speakers = {turn["agent"] for turn in discussion}

        assert {"Immunologist", "Structural Biologist"} <= speakers


class TestTeamSelectionMeeting:
    def test_a_meeting_can_hand_back_a_roster(
        self, fake_client: FakeClient, team_lead: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=ROSTER)]

        result = run_meeting(
            meeting_type="individual",
            agenda="Select a team of two scientists for this project.",
            save_dir=tmp_path,
            team_member=team_lead,
            num_rounds=0,
            output_schema=TeamRoster,
        )

        assert isinstance(result, TeamRoster)
        assert [member.title for member in result.team_members] == [
            "Immunologist",
            "Structural Biologist",
        ]

    def test_the_roster_survives_a_round_trip_through_disk(
        self, fake_client: FakeClient, team_lead: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=ROSTER)]

        run_meeting(
            meeting_type="individual",
            agenda="Select a team.",
            save_dir=tmp_path,
            team_member=team_lead,
            num_rounds=0,
            output_schema=TeamRoster,
        )

        saved = json.loads((tmp_path / OUTPUT_DIR_NAME / "discussion.json").read_text())
        reloaded = TeamRoster.model_validate(saved)

        assert reloaded == ROSTER
        assert reloaded.to_agents(model=TEST_MODEL)[0].title == "Immunologist"

    def test_selection_replaces_transcribing_agent_source_by_hand(
        self, fake_client: FakeClient, team_lead: Agent, tmp_path
    ) -> None:
        # Previously the agenda asked for literal "Agent(...)" source in prose for a human to
        # paste into a constants file. The roster arrives as data instead.
        fake_client.completions.parsed_responses = [parsed_response(parsed=ROSTER)]

        result = run_meeting(
            meeting_type="individual",
            agenda="Select a team.",
            save_dir=tmp_path,
            team_member=team_lead,
            num_rounds=0,
            output_schema=TeamRoster,
        )
        agents = result.to_agents(model=TEST_MODEL)

        assert all(isinstance(agent, Agent) for agent in agents)
        assert all(agent.prompt.startswith("You are a") for agent in agents)
