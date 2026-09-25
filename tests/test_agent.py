"""Tests for Agent, including author names and the model swap that keeps the critic on one model."""

import re

from virtual_lab.agent import Agent
from virtual_lab.constants import MAX_AGENT_NAME_LENGTH
from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC

from conftest import TEST_MODEL


def make_agent(title: str) -> Agent:
    return Agent(title=title, expertise="x", goal="y", role="z", model=TEST_MODEL)


def test_with_model_preserves_the_persona(team_member: Agent) -> None:
    swapped = team_member.with_model("o3-mini-2025-01-31")

    assert swapped.model == "o3-mini-2025-01-31"
    assert swapped.title == team_member.title
    assert swapped.expertise == team_member.expertise
    assert swapped.goal == team_member.goal
    assert swapped.role == team_member.role
    assert swapped.prompt == team_member.prompt


def test_with_model_does_not_mutate_the_original(team_member: Agent) -> None:
    team_member.with_model("o3-mini-2025-01-31")

    assert team_member.model == TEST_MODEL


class TestAuthorName:
    """The API only accepts letters, digits, underscores, and hyphens in an author name."""

    VALID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

    def test_spaces_become_underscores(self) -> None:
        assert make_agent("Principal Investigator").name == "Principal_Investigator"

    def test_punctuation_is_replaced(self) -> None:
        assert self.VALID.match(make_agent("Immunologist (Ph.D.), Lead").name)

    def test_no_leading_or_trailing_underscores(self) -> None:
        assert make_agent("  Scientific Critic!  ").name == "Scientific_Critic"

    def test_long_titles_are_truncated(self) -> None:
        name = make_agent("A" * 200).name

        assert len(name) == MAX_AGENT_NAME_LENGTH
        assert self.VALID.match(name)

    def test_title_without_usable_characters_falls_back(self) -> None:
        assert make_agent("!!!").name == "agent"

    def test_agents_with_different_titles_have_different_names(
        self, team_lead: Agent, team_member: Agent
    ) -> None:
        assert team_lead.name != team_member.name

    def test_name_survives_a_model_swap(self, team_member: Agent) -> None:
        assert team_member.with_model("o3-mini-2025-01-31").name == team_member.name

    def test_every_prompt_title_produces_a_valid_name(self) -> None:
        # The shipped personas must all be usable as author names without further escaping
        for agent in (PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC):
            assert self.VALID.match(agent.name), agent.title


def test_equality_includes_the_model(team_member: Agent) -> None:
    # Two agents differing only by model must not compare equal, which is why run_meeting
    # compares agent roles by identity rather than equality.
    assert team_member != team_member.with_model("o3-mini-2025-01-31")
    assert team_member == team_member.with_model(TEST_MODEL)
