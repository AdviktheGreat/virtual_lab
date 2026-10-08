"""Tests for steering a meeting as it goes: notes from the person following it, and pausing it."""

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from virtual_lab.agent import Agent
from virtual_lab.constants import HUMAN_SPEAKER
from virtual_lab.events import MeetingEvent, NextTurn
from virtual_lab.export import meeting_html
from virtual_lab.planning import run_project
from virtual_lab.project import Project
from virtual_lab.prompts import human_note_prompt
from virtual_lab.utils import count_discussion_tokens

from conftest import FakeClient, text_response
from test_events import CRITIC, GOAL, LEAD, individual, team
from test_planning import decide, plan, review, roster


def notes_for(*notes: str | None) -> tuple[list[NextTurn], Any]:
    """A steer that gives each note in turn, then none, and the turns it was asked about."""
    asked: list[NextTurn] = []
    queued = list(notes)

    def steer(turn: NextTurn) -> str | None:
        asked.append(turn)
        return queued.pop(0) if queued else None

    return asked, steer


def sent(fake_client: FakeClient, index: int) -> list[str]:
    return [message["content"] for message in fake_client.completions.calls[index]["messages"]]


class TestNotes:
    def test_steer_is_asked_before_every_turn(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        asked, steer = notes_for()

        team(team_lead, team_member, tmp_path, num_rounds=1, save_name="kickoff", steer=steer)

        assert asked == [
            NextTurn(meeting="kickoff", round=1, speaker=team_lead.title),
            NextTurn(meeting="kickoff", round=1, speaker=team_member.title),
            NextTurn(meeting="kickoff", round=2, speaker=team_lead.title),
        ]

    def test_a_note_is_read_by_the_next_speaker_and_everyone_after(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        _, steer = notes_for(None, "  Focus on stability, not affinity.\n")
        events: list[MeetingEvent] = []

        result = team(team_lead, team_member, tmp_path, num_rounds=1, steer=steer, on_event=events.append)

        note = human_note_prompt("Focus on stability, not affinity.")
        assert note == (
            "The human researcher overseeing this meeting has added a note:\n\nFocus on stability, not affinity."
        )
        # Not before it was given, and last of all for the agent it was given before
        assert note not in sent(fake_client, 0)
        assert sent(fake_client, 1)[-1] == note
        assert sent(fake_client, 2).count(note) == 1

        # In the transcript, after the prompt it follows, as said
        speakers = [turn["agent"] for turn in result.discussion]
        assert speakers == [
            "User",
            "User",
            team_lead.title,
            "User",
            HUMAN_SPEAKER,
            team_member.title,
            "User",
            team_lead.title,
        ]
        assert result.discussion[4]["message"] == "Focus on stability, not affinity."
        assert [turn.kind for turn in result.record.turns][3:6] == ["prompt", "note", "response"]
        assert result.record.turns[4].speaker == HUMAN_SPEAKER
        saved = json.loads(result.transcript_path.read_text(encoding="utf-8"))
        assert saved[4] == {"agent": HUMAN_SPEAKER, "message": "Focus on stability, not affinity."}

        # Told of as a message, before the turn it was given for begins
        told = [(event.kind, event.data.get("kind"), event.speaker) for event in events]
        at = told.index(("message", "note", HUMAN_SPEAKER))
        assert told[at + 1] == ("turn", None, team_member.title)
        assert events[at].round == 1

    def test_no_note_leaves_the_meeting_as_it_would_have_been(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        unsteered = individual(team_member, tmp_path / "a", num_rounds=1)
        calls = list(fake_client.completions.calls)
        fake_client.completions.calls.clear()

        _, steer = notes_for(None, "", "   \n")
        steered = individual(team_member, tmp_path / "b", num_rounds=1, steer=steer)

        assert steered.discussion == unsteered.discussion
        assert [call["messages"] for call in fake_client.completions.calls] == [call["messages"] for call in calls]

    def test_an_individual_meeting_is_steered_before_the_critic_too(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        asked, steer = notes_for(None, "Be harsh about the controls.")

        result = individual(team_member, tmp_path, num_rounds=1, steer=steer)

        assert [(turn.round, turn.speaker) for turn in asked] == [
            (1, team_member.title),
            (1, "Scientific Critic"),
            (2, team_member.title),
        ]
        assert sent(fake_client, 1)[-1] == human_note_prompt("Be harsh about the controls.")
        assert [turn["agent"] for turn in result.discussion][2:5] == ["User", HUMAN_SPEAKER, "Scientific Critic"]

    def test_a_note_is_shown_in_the_meeting_as_a_document(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        _, steer = notes_for("Use the **2023** data <only>.")
        result = individual(team_member, tmp_path, steer=steer)

        document = meeting_html(result.transcript_path)

        assert (
            '<div class="turn human"><div class="speaker">Human researcher</div>'
            '<div class="label">A note to the meeting</div>'
            "<div class=\"body\"><p>Use the <strong>2023</strong> data &lt;only&gt;.</p>\n</div></div>"
        ) in document

    def test_a_note_in_a_transcript_without_a_record_is_shown_as_a_note(self, tmp_path: Path) -> None:
        path = tmp_path / "discussion.json"
        transcript = [
            {"agent": "User", "message": "The agenda."},
            {"agent": HUMAN_SPEAKER, "message": "Keep it short."},
            {"agent": "Immunologist", "message": "Short."},
        ]
        path.write_text(json.dumps(transcript), encoding="utf-8")

        assert '<div class="turn human">' in meeting_html(path)

    def test_a_note_is_not_counted_as_written_by_a_model(self) -> None:
        discussion = [
            {"agent": "User", "message": "The agenda."},
            {"agent": HUMAN_SPEAKER, "message": "Keep it short, please, and cite your sources."},
            {"agent": "Immunologist", "message": "Short."},
        ]

        without_note = count_discussion_tokens([discussion[0], discussion[2]])
        with_note = count_discussion_tokens(discussion)

        assert with_note["output"] == without_note["output"]
        assert with_note["input"] > without_note["input"]


class TestPausingAndStopping:
    def test_a_meeting_waits_while_steer_does(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        waiting = threading.Event()
        resume = threading.Event()
        requests_while_paused: list[int] = []

        def steer(turn: NextTurn) -> str | None:
            if turn.speaker == "Scientific Critic":
                waiting.set()
                assert resume.wait(timeout=10)
                requests_while_paused.append(len(fake_client.completions.calls))
                return "Carry on."
            return None

        meeting = threading.Thread(
            target=individual, args=(team_member, tmp_path), kwargs={"num_rounds": 1, "steer": steer}
        )
        meeting.start()
        assert waiting.wait(timeout=10)
        assert len(fake_client.completions.calls) == 1
        resume.set()
        meeting.join(timeout=10)

        assert not meeting.is_alive()
        assert requests_while_paused == [1]
        assert len(fake_client.completions.calls) == 3
        assert (tmp_path / "discussion.json").is_file()

    def test_an_exception_from_steer_stops_the_meeting_and_its_work_is_kept(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        events: list[MeetingEvent] = []

        def steer(turn: NextTurn) -> str | None:
            if turn.round == 2:
                raise KeyboardInterrupt
            return "Note." if turn.speaker == "Scientific Critic" else None

        with pytest.raises(KeyboardInterrupt):
            individual(team_member, tmp_path, num_rounds=1, steer=steer, on_event=events.append)

        assert len(fake_client.completions.calls) == 2
        assert events[-1].kind == "failed"
        partial = json.loads((tmp_path / "partial" / "discussion.json").read_text(encoding="utf-8"))
        assert [turn["agent"] for turn in partial] == [
            "User",
            team_member.title,
            "User",
            HUMAN_SPEAKER,
            "Scientific Critic",
        ]
        record = json.loads((tmp_path / "partial" / "metadata" / "discussion.json").read_text(encoding="utf-8"))
        assert record["status"] == "failed"
        assert [turn["kind"] for turn in record["turns"]] == ["prompt", "response", "prompt", "note", "response"]

    def test_steer_must_return_a_note_or_none(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="steer returns a note or None, not list"):
            individual(team_member, tmp_path, steer=lambda turn: ["a note"])

        assert fake_client.completions.calls == []

    def test_steer_must_be_a_function(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="steer is a function"):
            individual(team_member, tmp_path, steer="Focus on stability.")


class TestProjects:
    def test_steer_given_to_a_project_steers_its_meetings_and_is_not_an_input(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        asked, steer = notes_for("Mind the budget.")
        first = Project(tmp_path, GOAL, steer=steer).meeting(
            "individual", "Compute a number.", name="first", team_member=team_member
        )

        assert [turn.meeting for turn in asked] == ["first"]
        assert first.discussion[1] == {"agent": HUMAN_SPEAKER, "message": "Mind the budget."}

        # Carried on without one, or with another, the meeting is read back, notes and all
        again = Project(tmp_path, GOAL).meeting(
            "individual", "Compute a number.", name="first", team_member=team_member
        )
        assert again.discussion == first.discussion
        assert len(fake_client.completions.calls) == 1

    def test_a_note_reaches_the_team_lead_as_it_decides_a_project_step(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses.extend(
            [
                roster("Immunologist"),
                plan("Task 1", "Task 2"),
                decide("finish", done=2, answer="Nanobody A binds better."),
                review(True),
            ]
        )
        asked: list[NextTurn] = []

        def steer(turn: NextTurn) -> str | None:
            asked.append(turn)
            return "Do not finish without binding data." if turn.meeting == "round_01_decision" else None

        report = run_project(Project(tmp_path, GOAL, steer=steer), team_lead=LEAD, critic=CRITIC)

        assert report.status == "finished"
        meetings = list(dict.fromkeys(turn.meeting for turn in asked))
        assert meetings == ["team", "plan", "round_01_decision", "round_01_review"]
        note = human_note_prompt("Do not finish without binding data.")
        given = [call for call in fake_client.completions.calls if note in [m["content"] for m in call["messages"]]]
        assert given and all(call["messages"][0]["content"] == LEAD.prompt for call in given)
        decision = json.loads((tmp_path / "meetings" / "round_01_decision.json").read_text(encoding="utf-8"))
        assert {"agent": HUMAN_SPEAKER, "message": "Do not finish without binding data."} in decision

    def test_a_steered_meeting_with_text_responses_still_ends_on_its_summary(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("First."),
            text_response("Critique."),
            text_response("Final."),
        ]
        _, steer = notes_for(None, None, "Wrap up.")

        result = individual(team_member, tmp_path, num_rounds=1, steer=steer)

        assert result.summary == "Final."
        assert result.discussion[-1] == {"agent": team_member.title, "message": "Final."}
        assert result.discussion[-2] == {"agent": HUMAN_SPEAKER, "message": "Wrap up."}
