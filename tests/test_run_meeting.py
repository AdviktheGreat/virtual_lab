"""Tests for meeting orchestration: model selection, retries, and checkpointing."""

import json

import pytest

from virtual_lab.agent import Agent
from virtual_lab.constants import DEFAULT_MAX_RETRIES, PARTIAL_MEETING_DIR_NAME
from virtual_lab.prompts import SCIENTIFIC_CRITIC
from virtual_lab.run_meeting import run_meeting

from conftest import TEST_MODEL, FakeClient, text_response

OTHER_MODEL = "o3-mini-2025-01-31"


def models_used(fake_client: FakeClient) -> list[str]:
    return [call["model"] for call in fake_client.completions.calls]


def agent_speakers(discussion: list[dict[str, str]]) -> list[str]:
    """Returns the agents who spoke, dropping the User prompt that precedes each turn."""
    return [turn["agent"] for turn in discussion if turn["agent"] != "User"]


class TestCriticModel:
    """The critic must not silently run a different model from the rest of the meeting."""

    def test_critic_defaults_to_the_team_member_model(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        member = team_member.with_model(OTHER_MODEL)

        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=member,
            num_rounds=1,
        )

        assert set(models_used(fake_client)) == {OTHER_MODEL}

    def test_critic_default_model_is_not_hardcoded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=1,
        )

        assert set(models_used(fake_client)) == {TEST_MODEL}
        assert SCIENTIFIC_CRITIC.model not in models_used(fake_client) or (
            SCIENTIFIC_CRITIC.model == TEST_MODEL
        )

    def test_explicit_critic_is_used(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            critic=SCIENTIFIC_CRITIC.with_model(OTHER_MODEL),
            num_rounds=1,
        )

        # Round 0: member then critic; final round: member only
        assert models_used(fake_client) == [TEST_MODEL, OTHER_MODEL, TEST_MODEL]

    def test_critic_appears_in_the_transcript_under_its_own_title(
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
        speakers = {turn["agent"] for turn in discussion}

        assert SCIENTIFIC_CRITIC.title in speakers
        assert team_member.title in speakers

    def test_team_meeting_rejects_a_critic(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="does not use a separate critic"):
            run_meeting(
                meeting_type="team",
                agenda="Design a nanobody.",
                save_dir=tmp_path,
                team_lead=team_lead,
                team_members=(team_member,),
                critic=SCIENTIFIC_CRITIC,
            )

    def test_critic_may_not_duplicate_the_team_member(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="separate from"):
            run_meeting(
                meeting_type="individual",
                agenda="Design a nanobody.",
                save_dir=tmp_path,
                team_member=team_member,
                critic=team_member.with_model(OTHER_MODEL),
            )


class TestRetries:
    def test_retries_are_delegated_to_the_client(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
        )

        assert fake_client.init_kwargs["max_retries"] == DEFAULT_MAX_RETRIES

    def test_max_retries_is_configurable(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
            max_retries=9,
        )

        assert fake_client.init_kwargs["max_retries"] == 9

    def test_default_retry_count_is_not_zero(self) -> None:
        assert DEFAULT_MAX_RETRIES > 0


class TestPartialSaveOnFailure:
    """A failure mid-meeting must not throw away the turns already paid for."""

    @staticmethod
    def run_failing_meeting(fake_client: FakeClient, team_member: Agent, save_dir) -> None:
        fake_client.completions.responses = [
            text_response("First answer."),
            text_response("A critique."),
            RuntimeError("API is down"),
        ]

        with pytest.raises(RuntimeError, match="API is down"):
            run_meeting(
                meeting_type="individual",
                agenda="Design a nanobody.",
                save_dir=save_dir,
                team_member=team_member,
                num_rounds=1,
            )

    def test_error_propagates(self, fake_client: FakeClient, team_member: Agent, tmp_path) -> None:
        self.run_failing_meeting(fake_client, team_member, tmp_path)

    def test_completed_turns_are_saved(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        self.run_failing_meeting(fake_client, team_member, tmp_path)

        partial = tmp_path / PARTIAL_MEETING_DIR_NAME / "discussion.json"

        assert partial.exists()
        assert (tmp_path / PARTIAL_MEETING_DIR_NAME / "discussion.md").exists()

        discussion = json.loads(partial.read_text())

        assert [turn["message"] for turn in discussion if turn["agent"] != "User"] == [
            "First answer.",
            "A critique.",
        ]

    def test_partial_is_not_mistaken_for_a_finished_meeting(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The notebooks collect prior meetings with glob("discussion_*.json") and feed the last
        # message of each to the next meeting as a summary. A truncated meeting must not match.
        self.run_failing_meeting(fake_client, team_member, tmp_path / "run")

        assert list((tmp_path / "run").glob("discussion*.json")) == []

    def test_usage_so_far_is_reported(
        self,
        fake_client: FakeClient,
        team_member: Agent,
        tmp_path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        self.run_failing_meeting(fake_client, team_member, tmp_path)

        output = capsys.readouterr().out

        assert PARTIAL_MEETING_DIR_NAME in output
        assert "Input token count" in output


class TestMeetingStructure:
    def test_individual_meeting_alternates_member_and_critic(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=2,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())

        assert discussion[0]["agent"] == "User"
        assert agent_speakers(discussion) == [
            team_member.title,
            SCIENTIFIC_CRITIC.title,
            team_member.title,
            SCIENTIFIC_CRITIC.title,
            team_member.title,
        ]

    def test_team_meeting_gives_every_member_a_turn(
        self,
        fake_client: FakeClient,
        team_lead: Agent,
        team_member: Agent,
        second_team_member: Agent,
        tmp_path,
    ) -> None:
        run_meeting(
            meeting_type="team",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=(team_member, second_team_member),
            num_rounds=1,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())

        # Round 0 goes lead, member, member; the final round is the lead's synthesis
        assert agent_speakers(discussion) == [
            team_lead.title,
            team_member.title,
            second_team_member.title,
            team_lead.title,
        ]

    def test_return_summary_returns_the_last_message(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [text_response("The final recommendation.")]

        summary = run_meeting(
            meeting_type="individual",
            agenda="Design a nanobody.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
            return_summary=True,
        )

        assert summary == "The final recommendation."
