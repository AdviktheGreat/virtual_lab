"""Tests for following a meeting or a project live: the events they tell of, and replies streamed
as they are written."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from virtual_lab.agent import Agent
from virtual_lab.events import MeetingEvent, ProjectEvent
from virtual_lab.llm import ask
from virtual_lab.planning import run_project
from virtual_lab.project import Project
from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.session import LocalSession, Session
from virtual_lab.tools import Tool

from conftest import TEST_MODEL, FakeClient, fake_llm, parsed_response, text_response, tool_call_response
from test_hold_meeting import bad_request
from test_planning import decide, found, plan, review, roster

LONG = "x" * 5_000


class Verdict(BaseModel):
    decision: str


def kinds(events: list[Any]) -> list[str]:
    return [event.kind for event in events]


def of(events: list[Any], kind: str) -> list[Any]:
    return [event for event in events if event.kind == kind]


def team(team_lead: Agent, team_member: Agent, save_dir: Path, **options: Any) -> Any:
    return hold_meeting(
        meeting_type="team",
        agenda="Design a nanobody.",
        save_dir=save_dir,
        team_lead=team_lead,
        team_members=(team_member,),
        **options,
    )


def individual(team_member: Agent, save_dir: Path, **options: Any) -> Any:
    return hold_meeting(
        meeting_type="individual",
        agenda="Compute a number.",
        save_dir=save_dir,
        team_member=team_member,
        **{"resources": "none", **options},
    )


class TestMeetingEvents:
    def test_a_team_meeting_is_told_of_turn_by_turn_in_order(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        events: list[MeetingEvent] = []

        result = team(team_lead, team_member, tmp_path, num_rounds=1, on_event=events.append, save_name="kickoff")

        assert kinds(events) == [
            "started",
            "message",
            *(["message", "turn", "usage", "message"] * 3),
            "finished",
        ]
        assert {event.meeting for event in events} == {"kickoff"}

        started = events[0]
        assert started.round is None
        assert started.data["meeting_type"] == "team"
        assert started.data["agenda"] == "Design a nanobody."
        assert started.data["num_rounds"] == 1
        assert [agent["title"] for agent in started.data["team"]] == [team_lead.title, team_member.title]
        assert started.data["session"] is None

        # Every message is the transcript's, by its place in it, and told of in its round
        messages = of(events, "message")
        assert [(event.speaker, event.text) for event in messages] == [
            (turn["agent"], turn["message"]) for turn in result.discussion
        ]
        assert [event.data["index"] for event in messages] == list(range(len(result.discussion)))
        assert [event.data["kind"] for event in messages] == [turn.kind for turn in result.record.turns]
        assert [event.round for event in messages] == [None, 1, 1, 1, 1, 2, 2]

        turns = of(events, "turn")
        assert [(event.speaker, event.round) for event in turns] == [
            (team_lead.title, 1),
            (team_member.title, 1),
            (team_lead.title, 2),
        ]
        assert turns[0].data == {"name": team_lead.name, "model": team_lead.model}

        usages = of(events, "usage")
        assert [event.data["num_calls"] for event in usages] == [1, 2, 3]
        assert usages[-1].data == result.usage.to_dict()

        finished = events[-1]
        assert finished.round is None
        assert finished.text == result.summary
        assert finished.data["transcript_path"] == str(result.transcript_path)
        assert finished.data["record_path"] == str(result.record_path)
        assert finished.data["output_path"] is None
        assert finished.data["output"] is None
        assert finished.data["usage"] == result.usage.to_dict()
        assert finished.data["elapsed_time"] == result.record.elapsed_seconds
        # Told of once the meeting is saved
        assert result.transcript_path.is_file()

    def test_no_reply_is_streamed_unless_asked_for(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        events: list[MeetingEvent] = []

        individual(team_member, tmp_path, on_event=events.append)

        assert "writing" not in kinds(events)
        assert not any(call.get("stream") for call in fake_client.completions.calls)

    def test_structured_output_is_told_of_as_a_message_and_in_the_end(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(Verdict(decision="go"))]
        events: list[MeetingEvent] = []

        result = individual(team_member, tmp_path, output_schema=Verdict, on_event=events.append)

        output = of(events, "message")[-1]
        assert output.data["kind"] == "structured_output"
        assert output.round is None
        assert json.loads(output.text) == {"decision": "go"}
        assert events[-1].data["output"] == {"decision": "go"}
        assert events[-1].data["output_path"] == str(result.output_path)

    def test_tools_are_told_of_before_they_run_with_all_their_arguments(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        events: list[MeetingEvent] = []
        told_before_running: list[bool] = []

        def look_up(query: str) -> str:
            told_before_running.append("tool_calls" in kinds(events))
            return "Found it."

        tool = Tool(
            name="look_up",
            description="Looks something up.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            function=look_up,
        )
        fake_client.completions.responses = [tool_call_response("look_up", {"query": LONG}), text_response("Done.")]

        individual(team_member, tmp_path, tools=(tool,), on_event=events.append)

        assert told_before_running == [True]
        (called,) = of(events, "tool_calls")
        assert called.speaker == team_member.title
        assert called.round == 1
        assert called.data["calls"] == [{"id": "call_1", "name": "look_up", "arguments": json.dumps({"query": LONG})}]
        assert [event.data["kind"] for event in of(events, "message")] == ["prompt", "tool_output", "response"]
        assert "cell" not in kinds(events)

    def test_an_exception_from_on_event_stops_the_meeting_and_its_work_is_kept(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        told: list[MeetingEvent] = []

        def on_event(event: MeetingEvent) -> None:
            told.append(event)
            if event.kind == "usage":
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            individual(team_member, tmp_path, num_rounds=1, on_event=on_event)

        assert len(fake_client.completions.calls) == 1
        failed = told[-1]
        assert failed.kind == "failed"
        assert failed.text == "KeyboardInterrupt"
        assert failed.data["type"] == "KeyboardInterrupt"
        assert failed.data["partial_path"] == str(tmp_path / "partial" / "discussion.json")
        assert Path(failed.data["partial_path"]).is_file()
        assert failed.data["usage"]["num_calls"] == 1

    def test_a_failure_is_told_of_with_the_error(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [RuntimeError("The server went away")]
        events: list[MeetingEvent] = []

        with pytest.raises(RuntimeError, match="went away"):
            individual(team_member, tmp_path, on_event=events.append, max_retries=0)

        assert kinds(events)[-1] == "failed"
        assert events[-1].text == "The server went away"
        assert events[-1].data["type"] == "RuntimeError"
        assert events[-1].round is None

    def test_on_event_failing_when_told_of_a_failure_does_not_hide_it(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_client.completions.responses = [RuntimeError("The server went away")]

        def on_event(event: MeetingEvent) -> None:
            if event.kind == "failed":
                raise ValueError("The interface closed")

        with pytest.raises(RuntimeError, match="went away"):
            individual(team_member, tmp_path, on_event=on_event, max_retries=0)

        warning = "on_event failed when told the meeting had failed: ValueError('The interface closed')"
        assert warning in capsys.readouterr().out

    def test_on_event_can_stop_a_meeting_before_its_first_request(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        events: list[MeetingEvent] = []

        def on_event(event: MeetingEvent) -> None:
            events.append(event)
            if event.kind == "message":
                raise RuntimeError("Stop")

        with pytest.raises(RuntimeError, match="Stop"):
            individual(team_member, tmp_path, on_event=on_event)

        # The first message was added before on_event stopped the meeting, so it is kept
        assert events[-1].data["partial_path"] is not None
        assert fake_client.completions.calls == []


class TestStreaming:
    def test_each_reply_is_told_of_as_it_is_written(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("Let us begin."),
            text_response("I agree with the plan."),
            text_response("We will make four nanobodies."),
        ]
        events: list[MeetingEvent] = []

        result = team(team_lead, team_member, tmp_path, num_rounds=1, on_event=events.append, stream=True)

        writing = of(events, "writing")
        assert [event.data["request"] for event in writing] == [1] * 3 + [2] * 5 + [3] * 5
        assert [event.text for event in writing if event.data["request"] == 2] == [
            "I ",
            "I agree ",
            "I agree with ",
            "I agree with the ",
            "I agree with the plan.",
        ]
        assert [event.speaker for event in writing][::5] == [team_lead.title, team_member.title, team_lead.title]
        assert writing[0].data["model"] == team_lead.model
        assert [event.round for event in writing if event.data["request"] == 3] == [2] * 5

        # Each reply is written before it is added to the transcript
        assert kinds(events)[kinds(events).index("turn") + 1 : kinds(events).index("turn") + 5] == [
            "writing",
            "writing",
            "writing",
            "usage",
        ]
        assert result.summary == "We will make four nanobodies."

        # Streamed with its usage, so that what it cost is still known
        assert all(call["stream"] is True for call in fake_client.completions.calls)
        assert all(call["stream_options"] == {"include_usage": True} for call in fake_client.completions.calls)
        assert result.usage.num_calls == 3
        assert result.usage.unreported_calls == 0
        assert result.cost is not None

    def test_a_request_sent_again_without_its_temperature_is_written_again(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [bad_request("temperature"), text_response("An answer.")]
        events: list[MeetingEvent] = []

        individual(team_member, tmp_path, on_event=events.append, stream=True)

        writing = of(events, "writing")
        assert [(event.data["request"], event.text) for event in writing] == [(1, "An "), (1, "An answer.")]
        assert len(fake_client.completions.calls) == 2

    def test_tool_calls_are_streamed_whole(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        received: list[str] = []
        tool = Tool(
            name="look_up",
            description="Looks something up.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            function=lambda query: received.append(query) or "Found it.",
        )
        fake_client.completions.responses = [tool_call_response("look_up", {"query": LONG}), text_response("Done.")]
        events: list[MeetingEvent] = []

        result = individual(team_member, tmp_path, tools=(tool,), on_event=events.append, stream=True)

        assert received == [LONG]
        assert result.summary == "Done."
        assert result.record.turns[-1].tool_calls[0]["name"] == "look_up"
        # The call itself has no text to tell of; the answer after it does
        assert [event.data["request"] for event in of(events, "writing")] == [2]

    def test_resources_are_chosen_without_being_streamed(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        with LocalSession(tmp_path / "work", warn=False, timeout=20) as session:
            fake_client.completions.responses = [text_response("Nothing is needed."), text_response("Done.")]
            events: list[MeetingEvent] = []

            result = individual(
                team_member, tmp_path, session=session, resources="retrieve", on_event=events.append, stream=True
            )

        assert [event.text for event in of(events, "writing")] == ["Done."]
        assert len(fake_client.completions.calls) == 2
        (chosen,) = of(events, "resources")
        assert chosen.data == result.record.resources
        assert kinds(events).index("resources") < kinds(events).index("turn")

    def test_stream_needs_on_event(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs an on_event"):
            individual(team_member, tmp_path, stream=True)

        assert fake_client.completions.calls == []

    def test_an_exception_while_a_reply_is_written_stops_the_meeting(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("A long answer that is cut short.")]
        told: list[MeetingEvent] = []

        def on_event(event: MeetingEvent) -> None:
            told.append(event)
            if event.kind == "writing" and event.text.startswith("A long"):
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            individual(team_member, tmp_path, num_rounds=1, on_event=on_event, stream=True)

        assert [event.text for event in told if event.kind == "writing"] == ["A ", "A long "]
        assert told[-1].kind == "failed"
        assert "usage" not in kinds(told)


class PlottingSession(LocalSession):
    """A session whose code is taken to have drawn a figure, since matplotlib is not installed here."""

    def run(self, code: str, language: str = "python", timeout: float | None = None) -> Any:
        result = super().run(code, language=language, timeout=timeout)
        drawn = dataclasses.replace(result, plots=("plots/figure_1.png",))
        self.history[-1] = drawn
        return drawn


@pytest.fixture
def session(tmp_path: Path):
    with PlottingSession(tmp_path / "work", warn=False, timeout=20) as opened:
        yield opened


class TestCode:
    def test_code_in_tags_is_told_of_before_and_after_it_runs(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("I will compute it.\n<execute>\nprint(6 * 7)\n</execute>"),
            text_response("The answer is 42."),
        ]
        events: list[MeetingEvent] = []

        individual(team_member, tmp_path, session=session, code_actions="tags", on_event=events.append)

        assert kinds(events)[kinds(events).index("code") :] == [
            "code",
            "cell",
            "message",
            "message",
            "usage",
            "message",
            "finished",
        ]
        (wrote,) = of(events, "code")
        assert wrote.speaker == team_member.title
        assert wrote.data == {"code": "print(6 * 7)", "language": "python"}
        assert "I will compute it." in wrote.text

        (ran,) = of(events, "cell")
        assert ran.data["code"] == "print(6 * 7)"
        assert ran.data["status"] == "ok"
        assert ran.data["output"].strip() == "42"
        assert ran.data["plots"] == ["plots/figure_1.png"]
        assert ran.data["plot_paths"] == [str(session.directory / "plots" / "figure_1.png")]
        assert ran.round == 1
        assert [event.data["kind"] for event in of(events, "message")][1:3] == ["code_action", "code_output"]

    def test_code_run_through_the_tool_is_told_of_once_it_has_run(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        session.run("earlier = 1")
        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "print('first')"}),
            tool_call_response("run_code", {"code": "print('second')"}),
            text_response("Done."),
        ]
        events: list[MeetingEvent] = []

        individual(team_member, tmp_path, session=session, on_event=events.append)

        # Only what the meeting ran, not what was run before it
        assert [event.data["code"] for event in of(events, "cell")] == ["print('first')", "print('second')"]
        assert kinds(events).count("tool_calls") == 2
        assert [kind for kind in kinds(events) if kind in ("tool_calls", "cell")] == ["tool_calls", "cell"] * 2
        # The session as it was when the meeting started
        assert of(events, "started")[0].data["session"] == {**session.describe(), "cells": 1}


LEAD = PRINCIPAL_INVESTIGATOR.with_model(TEST_MODEL)
CRITIC = SCIENTIFIC_CRITIC.with_model(TEST_MODEL)
GOAL = "Find which of two nanobodies binds KP.3 better."


class TestProjectEvents:
    def test_the_options_that_follow_a_meeting_do_not_make_it_another(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL)
        events: list[MeetingEvent] = []
        project.meeting("individual", "Compute a number.", name="first", team_member=team_member)

        result = Project(tmp_path, GOAL).meeting(
            "individual",
            "Compute a number.",
            name="first",
            team_member=team_member,
            on_event=events.append,
            stream=True,
        )

        assert len(fake_client.completions.calls) == 1
        (read_back,) = events
        assert read_back.kind == "read_back"
        assert read_back.meeting == "first"
        assert read_back.text == result.summary
        assert read_back.data == {
            "transcript_path": str(result.transcript_path),
            "record_path": str(result.record_path),
            "output_path": None,
            "usage": result.usage.to_dict(),
        }

    def test_a_project_meeting_streams_to_on_event_given_as_a_meeting_option(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        events: list[MeetingEvent] = []
        project = Project(tmp_path, GOAL, on_event=events.append, stream=True)

        project.meeting("individual", "Compute a number.", name="first", team_member=team_member)

        assert {event.meeting for event in events} == {"first"}
        assert "writing" in kinds(events)
        assert kinds(events)[-1] == "finished"

    def test_a_project_is_told_of_step_by_step_with_its_meetings(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses.extend(
            [
                roster("Immunologist"),
                plan("Task 1", "Task 2"),
                decide("team_meeting", participants=["Immunologist"], agenda="Compare the binding data."),
                found(),
                decide("finish", done=2, answer="Nanobody A binds better."),
                review(True),
            ]
        )
        own: list[MeetingEvent] = []
        events: list[MeetingEvent | ProjectEvent] = []

        def on_event(event: MeetingEvent | ProjectEvent) -> None:
            # The project's own on_event is called first
            if isinstance(event, MeetingEvent):
                assert own[-1] is event
            events.append(event)

        project = Project(tmp_path, GOAL, on_event=own.append)
        report = run_project(project, team_lead=LEAD, critic=CRITIC, on_event=on_event)

        told = [event for event in events if isinstance(event, ProjectEvent)]
        assert [(event.kind, event.round) for event in told] == [
            ("team", None),
            ("plan", None),
            ("decided", 1),
            ("round", 1),
            ("decided", 2),
            ("round", 2),
            ("finished", 2),
        ]
        assert [member["title"] for member in told[0].data["team"]] == ["Immunologist"]
        assert told[0].data["change"] == report.team_changes[0]
        assert told[1].data["plan"] == [{"task": "Task 1", "status": "to do"}, {"task": "Task 2", "status": "to do"}]
        assert told[2].data["proposed"]["action"] == "team_meeting"
        assert told[3].data["round"] == dataclasses.asdict(report.rounds[0])
        assert told[3].data["plan"] == report.rounds[0].proposed["plan"]
        assert told[-1].text == report.reason
        assert told[-1].data["report"] == report.to_dict()

        meetings = [event.meeting for event in events if isinstance(event, MeetingEvent) and event.kind == "started"]
        assert meetings == [
            "team",
            "plan",
            "round_01_decision",
            "round_01_meeting",
            "round_02_decision",
            "round_02_review",
        ]
        assert [event for event in own] == [event for event in events if isinstance(event, MeetingEvent)]

        # Each meeting is told of before the step it is part of
        decided = events.index(told[2])
        assert isinstance(events[decided - 1], MeetingEvent) and events[decided - 1].meeting == "round_01_decision"

    def test_code_run_for_a_project_is_told_of(self, fake_client: FakeClient, tmp_path: Path) -> None:
        from test_planning import code
        from virtual_lab.execution import LocalExecutor

        fake_client.completions.parsed_responses.extend(
            [
                roster("Immunologist"),
                plan("Task 1", "Task 2"),
                decide("write_code", participants=["Immunologist"], agenda="Compare them."),
                code("print('A binds better')"),
                found("A binds better"),
                decide("finish", done=2, answer="A.", findings=["F1"]),
                review(True),
            ]
        )
        events: list[MeetingEvent | ProjectEvent] = []

        run_project(
            Project(tmp_path, GOAL),
            team_lead=LEAD,
            critic=CRITIC,
            executor=LocalExecutor(warn=False),
            on_event=events.append,
        )

        (ran,) = [event for event in events if isinstance(event, ProjectEvent) and event.kind == "code"]
        assert ran.round == 1
        assert ran.data == {"name": "round_01_run", "succeeded": True}
        assert "A binds better" in ran.text


class TestAsk:
    def test_a_streamed_reply_is_the_reply(self, fake_client: FakeClient) -> None:
        fake_client.completions.responses = [text_response("Two words.")]
        written: list[str] = []

        reply = ask(fake_llm(fake_client), [{"role": "user", "content": "Hi"}], temperature=0.2, on_text=written.append)

        assert reply.content == "Two words."
        assert reply.finish_reason == "stop"
        assert reply.usage is not None and reply.usage.prompt_tokens == 100
        assert written == ["Two ", "Two words."]

    def test_a_model_told_not_to_stream_its_usage_is_not_overruled(self, fake_client: FakeClient) -> None:
        llm = fake_llm(fake_client).model_copy(update={"stream_usage": False})

        reply = ask(llm, [{"role": "user", "content": "Hi"}], temperature=0.2, on_text=lambda text: None)

        assert "stream_options" not in fake_client.completions.calls[-1]
        assert reply.usage is None

    def test_a_model_that_cannot_stream_sends_its_reply_at_once(self) -> None:
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage

        llm = FakeMessagesListChatModel(responses=[AIMessage(content="All at once.")])
        written: list[str] = []

        reply = ask(llm, [{"role": "user", "content": "Hi"}], temperature=None, on_text=written.append)

        assert reply.content == "All at once."
        assert written == ["All at once."]


class TestEndingEvents:
    def test_on_event_failing_when_told_of_the_end_does_not_undo_a_saved_meeting(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def on_event(event: MeetingEvent) -> None:
            if event.kind == "finished":
                raise ValueError("The interface closed")

        project = Project(tmp_path, GOAL)
        result = project.meeting(
            "individual", "Compute a number.", name="first", team_member=team_member, on_event=on_event
        )

        assert result.transcript_path.is_file()
        assert [step.status for step in project.steps] == ["completed"]
        assert "on_event failed when told the meeting had finished: ValueError('The interface closed')" in (
            capsys.readouterr().out
        )

        # Carried on, the meeting is read back rather than held and paid for again
        Project(tmp_path, GOAL).meeting("individual", "Compute a number.", name="first", team_member=team_member)
        assert len(fake_client.completions.calls) == 1

    def test_on_event_failing_when_told_the_project_ended_leaves_its_report_and_log_agreeing(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_client.completions.parsed_responses.extend(
            [plan("Task 1"), decide("finish", done=2, answer="A."), review(True)]
        )

        def on_event(event: MeetingEvent | ProjectEvent) -> None:
            if isinstance(event, ProjectEvent) and event.kind == "finished":
                raise ValueError("The interface closed")

        report = run_project(Project(tmp_path, GOAL), team_lead=LEAD, critic=CRITIC, team=(), on_event=on_event)

        assert report.status == "finished"
        assert json.loads((tmp_path / "research_log.json").read_text())["status"] == "finished"
        assert json.loads((tmp_path / "report.json").read_text())["status"] == "finished"
        assert "on_event failed when told the project had ended: ValueError('The interface closed')" in (
            capsys.readouterr().out
        )

    def test_on_event_failing_as_the_budget_runs_out_is_logged_as_the_error(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        from test_planning import CALL_COST

        fake_client.completions.parsed_responses.extend(
            [plan("Task 1"), decide("individual_meeting", participants=["Principal Investigator"], agenda="One.")]
        )

        def on_event(event: MeetingEvent | ProjectEvent) -> None:
            if isinstance(event, ProjectEvent) and event.kind == "round":
                raise ValueError("The interface closed")

        project = Project(tmp_path, GOAL, max_cost=7.5 * CALL_COST)
        with pytest.raises(ValueError, match="interface closed"):
            run_project(project, team_lead=LEAD, critic=CRITIC, team=(), on_event=on_event)

        log = json.loads((tmp_path / "research_log.json").read_text())
        assert log["status"] == "failed"
        assert log["error"] == {"type": "ValueError", "message": "The interface closed"}
        assert not (tmp_path / "report.json").exists()
