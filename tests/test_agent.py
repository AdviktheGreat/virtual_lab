"""Tests for Agent, including the model swap used to keep the critic on one model."""

from virtual_lab.agent import Agent

from conftest import TEST_MODEL


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


def test_equality_includes_the_model(team_member: Agent) -> None:
    # Two agents differing only by model must not compare equal, which is why run_meeting
    # compares agent roles by identity rather than equality.
    assert team_member != team_member.with_model("o3-mini-2025-01-31")
    assert team_member == team_member.with_model(TEST_MODEL)
