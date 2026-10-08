"""Tests for running a meeting or a project from the web interface: followed as it goes, and
steered with notes, a pause, a stop, and the person's approval of each decision."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from virtual_lab.approval import ApprovalRequest, ServerQuestion, asked_for
from virtual_lab.constants import HUMAN_SPEAKER
from virtual_lab.events import MeetingEvent, NextTurn, ProjectEvent
from virtual_lab.execution import DockerExecutor, LocalExecutor
from virtual_lab.environment import sandbox_image
from virtual_lab.planning import NextStep
from virtual_lab.session import DockerSession, LocalSession
from virtual_lab.tools import TOOL_REGISTRY
from virtual_lab.ui import library
from virtual_lab.ui.runs import (
    CodeSetup,
    Connections,
    Lab,
    MeetingSetup,
    ProjectSetup,
    Run,
    RunStopped,
    agent_data,
    changed_step,
    form_answer,
    read_setup,
)
from virtual_lab.ui.workspace import MEETING_NAME, PROJECT_SETUP_FILE_NAME, Workspace

from conftest import TEST_MODEL, FakeClient, text_response
from test_planning import GOAL, decide, found, plan, queue, review, roster, step

WAIT = 10


def person(title: str) -> dict[str, str]:
    return agent_data(library.scientist(title).with_model(TEST_MODEL))


def individual(**options: Any) -> MeetingSetup:
    options = {"agenda": "Choose the epitopes.", **options}
    return MeetingSetup(meeting_type="individual", lead=person("Immunologist"), **options)


def project(**options: Any) -> ProjectSetup:
    options = {"goal": GOAL, **options}
    return ProjectSetup(lead=person("Principal Investigator"), critic=person("Scientific Critic"), **options)


def finishing(fake_client: FakeClient) -> None:
    """Queues a project that is staffed, planned, holds one meeting, and finishes."""
    queue(
        fake_client,
        roster("Immunologist"),
        plan("Task 1", "Task 2"),
        decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Check the affinities."),
        found(),
        decide("finish", done=2, answer="Nanobody A binds better."),
        review(True),
    )


class Gate:
    """Holds every request to the fake model until let through, so that a test can act on a run
    at a known point."""

    def __init__(self, fake_client: FakeClient) -> None:
        self.open = threading.Event()
        self.requests = 0
        self.arrived = threading.Condition()
        for name in ("create", "parse"):
            original = getattr(fake_client.completions, name)
            setattr(fake_client.completions, name, self.held(original))

    def held(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            with self.arrived:
                self.requests += 1
                self.arrived.notify_all()
            assert self.open.wait(WAIT), "The gate was never opened"
            return original(**kwargs)

        return call

    def wait_for(self, requests: int) -> None:
        with self.arrived:
            assert self.arrived.wait_for(lambda: self.requests >= requests, WAIT)


def eventually(check: Callable[[], bool]) -> None:
    for _ in range(WAIT * 20):
        if check():
            return
        threading.Event().wait(0.05)
    raise AssertionError("It never happened")


def finished(run: Run) -> Run:
    assert run.wait(WAIT), "The run did not end"
    return run


def transcript(run: Run) -> list[dict[str, str]]:
    return json.loads((run.directory / f"{MEETING_NAME}.json").read_text())


class TestMeetingRun:
    def test_a_meeting_runs_in_its_own_directory_and_is_followed_as_it_goes(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        workspace = Workspace(tmp_path)
        fake_client.completions.responses = [text_response("The receptor binding domain.")]

        run = finished(Lab(workspace).start_meeting(individual(num_rounds=0)))
        view = run.view()

        assert view.status == "completed" and view.error is None
        assert view.directory.parent == workspace.meetings_dir
        assert view.result == run.directory / f"{MEETING_NAME}.json"
        assert [event.kind for event in view.events if event.kind in ("started", "finished")] == ["started", "finished"]
        assert not any(event.kind == "usage" for event in view.events)
        assert view.spent is not None and view.spent > 0
        assert read_setup(run.directory) == individual(num_rounds=0)
        assert [entry.status for entry in workspace.entries()] == ["completed"]

    def test_a_reply_being_written_is_kept_only_as_it_stands(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [text_response("Three words here.")]

        run = finished(Lab(Workspace(tmp_path)).start_meeting(individual(num_rounds=0, stream=True)))
        kinds = [event.kind for event in run.view().events]

        assert kinds.count("writing") == 1
        assert kinds[kinds.index("writing") + 1] == "message"

    def test_a_note_is_read_by_the_next_agent_to_speak(self, fake_client: FakeClient, tmp_path: Path) -> None:
        gate = Gate(fake_client)
        run = Lab(Workspace(tmp_path)).start_meeting(individual(num_rounds=1))
        gate.wait_for(1)

        assert run.add_note("  Consider the N-terminal domain too.  ")
        assert not run.add_note("   ")
        assert run.view().notes == ("Consider the N-terminal domain too.",)
        gate.open.set()
        finished(run)

        notes = [turn for turn in transcript(run) if turn["agent"] == HUMAN_SPEAKER]
        assert [turn["message"] for turn in notes] == ["Consider the N-terminal domain too."]
        assert run.view().notes == ()
        assert not run.add_note("Too late.")

    def test_a_paused_meeting_is_held_before_the_next_turn_until_resumed(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        gate = Gate(fake_client)
        run = Lab(Workspace(tmp_path)).start_meeting(individual(num_rounds=1))
        gate.wait_for(1)
        run.pause()
        gate.open.set()

        eventually(lambda: run.view().holding)
        held = run.view()
        assert held.paused and held.status == "running"
        assert held.next_turn is not None and held.next_turn.speaker == "Scientific Critic"
        requests = gate.requests
        threading.Event().wait(0.3)
        assert gate.requests == requests

        run.resume()
        assert finished(run).view().status == "completed"
        assert not run.view().holding

    def test_a_stopped_meeting_keeps_what_was_said(self, fake_client: FakeClient, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)
        gate = Gate(fake_client)
        run = Lab(workspace).start_meeting(individual(num_rounds=2))
        gate.wait_for(1)
        run.pause()
        gate.open.set()
        eventually(lambda: run.view().holding)

        run.stop()
        view = finished(run).view()

        assert view.status == "stopped"
        assert view.error == "Stopped by you. What was said until then is saved."
        assert [entry.status for entry in workspace.entries()] == ["stopped"]
        assert view.events[-1].kind == "failed"

    def test_a_meeting_writing_a_reply_is_stopped_in_the_middle_of_it(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        gate = Gate(fake_client)
        run = Lab(Workspace(tmp_path)).start_meeting(individual(num_rounds=1, stream=True))
        gate.wait_for(1)
        run.stop()
        gate.open.set()

        assert finished(run).view().status == "stopped"
        assert gate.requests == 1

    def test_a_run_that_fails_says_why(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [RuntimeError("The model is down")]

        view = finished(Lab(Workspace(tmp_path)).start_meeting(individual(num_rounds=0))).view()

        assert view.status == "failed"
        assert view.error == "RuntimeError: The model is down"

    def test_a_meeting_is_given_the_summaries_of_meetings_chosen(self, fake_client: FakeClient, tmp_path: Path) -> None:
        lab = Lab(Workspace(tmp_path))
        fake_client.completions.responses = [text_response("Epitope 3 is the one.")]
        earlier = finished(lab.start_meeting(individual(num_rounds=0)))

        finished(lab.start_meeting(individual(num_rounds=0, summaries=[str(earlier.view().result)])))

        assert "Epitope 3 is the one." in json.dumps(fake_client.completions.calls[-1]["messages"])


class TestResources:
    def test_a_meeting_is_held_with_the_resources_that_were_chosen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        held: list[dict[str, Any]] = []
        monkeypatch.setattr("virtual_lab.ui.runs.hold_meeting", lambda **options: held.append(options))
        lab = Lab(Workspace(tmp_path))

        for mode in ("all", "none"):
            finished(lab.start_meeting(individual(code=CodeSetup(resources=mode))))
        finished(lab.start_meeting(individual()))

        assert [options["resources"] for options in held] == ["all", "none", "retrieve"]

    def test_a_project_holds_its_meetings_with_the_resources_that_were_chosen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        made: list[dict[str, Any]] = []
        monkeypatch.setattr("virtual_lab.ui.runs.Project", lambda *args, **options: made.append(options))
        monkeypatch.setattr("virtual_lab.ui.runs.run_project", lambda *args, **options: None)
        lab = Lab(Workspace(tmp_path))

        finished(lab.start_project(project(code=CodeSetup(resources="all"))))
        finished(lab.start_project(project()))

        assert [options["resources"] for options in made] == ["all", "retrieve"]


class TestOneRunOfAKind:
    def started_at_once(self, start: Callable[[], Run], count: int = 8) -> tuple[list[Run], list[Exception]]:
        """Calls start from count threads released together, and returns what they started and what refused."""
        barrier = threading.Barrier(count)
        started: list[Run] = []
        refused: list[Exception] = []

        def attempt() -> None:
            barrier.wait()
            try:
                started.append(start())
            except RuntimeError as error:
                refused.append(error)

        threads = [threading.Thread(target=attempt) for _ in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(WAIT)
        return started, refused

    def test_meetings_started_at_once_are_one_meeting(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        release = threading.Event()
        monkeypatch.setattr("virtual_lab.ui.runs.hold_meeting", lambda **options: release.wait(WAIT))
        workspace = Workspace(tmp_path)
        lab = Lab(workspace)

        started, refused = self.started_at_once(lambda: lab.start_meeting(individual()))
        release.set()

        assert len(started) == 1 and len(refused) == 7
        assert all("A meeting is going on" in str(error) for error in refused)
        assert len(list(workspace.meetings_dir.iterdir())) == 1
        finished(started[0])

    def test_a_project_carried_on_at_once_from_two_pages_is_carried_on_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        release = threading.Event()
        monkeypatch.setattr("virtual_lab.ui.runs.Project", lambda *args, **options: SimpleNamespace(spent=0.0))
        monkeypatch.setattr("virtual_lab.ui.runs.run_project", lambda *args, **options: release.wait(WAIT))
        workspace = Workspace(tmp_path)
        lab = Lab(workspace)
        directory = workspace.new_directory("project", "Carried on.")

        started, refused = self.started_at_once(lambda: lab.start_project(project(), directory=directory))
        release.set()

        assert len(started) == 1 and len(refused) == 7
        finished(started[0])

    def test_a_meeting_and_a_project_go_on_together_and_another_starts_once_one_ends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        release = threading.Event()
        monkeypatch.setattr("virtual_lab.ui.runs.hold_meeting", lambda **options: release.wait(WAIT))
        monkeypatch.setattr("virtual_lab.ui.runs.Project", lambda *args, **options: SimpleNamespace(spent=0.0))
        monkeypatch.setattr("virtual_lab.ui.runs.run_project", lambda *args, **options: release.wait(WAIT))
        lab = Lab(Workspace(tmp_path))

        meeting = lab.start_meeting(individual())
        going = lab.start_project(project())
        with pytest.raises(RuntimeError, match="A project is going on"):
            lab.start_project(project())
        release.set()
        finished(meeting)
        finished(going)

        assert finished(lab.start_meeting(individual())).view().status == "completed"
        assert finished(lab.start_project(project())).view().status == "completed"


class TestProjectRun:
    def test_a_project_left_to_itself_runs_to_its_report(self, fake_client: FakeClient, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)
        finishing(fake_client)

        run = finished(Lab(workspace).start_project(project(autonomous=True)))
        view = run.view()

        assert view.status == "completed", view.error
        assert view.result == run.directory
        assert json.loads((run.directory / "report.json").read_text())["status"] == "finished"
        assert any(isinstance(event, ProjectEvent) and event.kind == "finished" for event in view.events)
        assert view.spent == pytest.approx(workspace.entries()[0].cost)
        assert read_setup(run.directory) == project(autonomous=True)

    def test_each_decision_waits_for_the_person_who_may_change_it(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        finishing(fake_client)
        run = Lab(Workspace(tmp_path)).start_project(project())

        eventually(lambda: bool(run.view().waiting))
        [waiting] = run.view().waiting
        assert waiting.kind == "decision" and waiting.round == 1
        assert isinstance(waiting.subject, NextStep)
        assert run.respond(waiting.id, changed_step(waiting.subject, agenda="Measure instead."))
        assert not run.respond(waiting.id, waiting.subject)

        eventually(lambda: bool(run.view().waiting))
        [second] = run.view().waiting
        assert second.round == 2
        run.set_autonomous(True)
        report = json.loads((finished(run).directory / "report.json").read_text())

        assert report["status"] == "finished"
        assert report["rounds"][0]["approved"]["agenda"] == "Measure instead."

    def test_a_decision_answered_with_none_stops_the_project_with_a_report(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        finishing(fake_client)
        run = Lab(Workspace(tmp_path)).start_project(project())

        eventually(lambda: bool(run.view().waiting))
        run.respond(run.view().waiting[0].id, None)

        assert finished(run).view().status == "completed"
        assert json.loads((run.directory / "report.json").read_text())["status"] == "stopped"

    def test_a_project_stopped_after_its_step_ends_with_a_report(self, fake_client: FakeClient, tmp_path: Path) -> None:
        finishing(fake_client)
        run = Lab(Workspace(tmp_path)).start_project(project())

        eventually(lambda: bool(run.view().waiting))
        run.stop(now=False)

        assert finished(run).view().status == "completed"
        assert run.view().stopping == "after_step"
        assert json.loads((run.directory / "report.json").read_text())["status"] == "stopped"

    def test_a_project_stopped_now_has_no_report_and_is_carried_on_later(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        workspace = Workspace(tmp_path)
        lab = Lab(workspace)
        finishing(fake_client)
        run = lab.start_project(project())
        eventually(lambda: bool(run.view().waiting))

        run.stop()

        assert finished(run).view().status == "stopped"
        assert not (run.directory / "report.json").exists()
        assert [entry.status for entry in workspace.entries()] == ["not finished"]

        carried = finished(lab.start_project(project(autonomous=True), directory=run.directory))
        assert carried.view().status == "completed", carried.view().error
        assert json.loads((run.directory / "report.json").read_text())["status"] == "finished"

    def test_a_project_running_already_is_not_started_again(self, fake_client: FakeClient, tmp_path: Path) -> None:
        lab = Lab(Workspace(tmp_path))
        finishing(fake_client)
        run = lab.start_project(project())
        eventually(lambda: bool(run.view().waiting))

        with pytest.raises(RuntimeError, match="running already"):
            lab.start_project(project(), directory=run.directory)

        assert lab.latest("project") is run
        run.stop()
        finished(run)

    def test_a_paused_project_waits_before_its_next_decision(self, fake_client: FakeClient, tmp_path: Path) -> None:
        finishing(fake_client)
        run = Lab(Workspace(tmp_path)).start_project(project(autonomous=True))
        run.pause()

        eventually(lambda: run.view().holding)
        assert run.status == "running"
        run.resume()

        assert finished(run).view().status == "completed"


class TestAsking:
    def asking(self, run: Run, ask: Callable[[], Any]) -> tuple[threading.Thread, list[Any]]:
        answers: list[Any] = []
        thread = threading.Thread(target=lambda: answers.append(ask()), daemon=True)
        thread.start()
        eventually(lambda: bool(run.view().waiting))
        return thread, answers

    def test_a_call_waiting_for_approval_is_made_only_if_approved(self, tmp_path: Path) -> None:
        run = Run("meeting", "Order.", tmp_path)
        request = ApprovalRequest(tool="lab_order", server="lab", server_tool="order", arguments={"n": 1})

        for approved in (True, False):
            thread, answers = self.asking(run, lambda: run.approve_call(request))
            [waiting] = run.view().waiting
            assert waiting.kind == "approval" and waiting.subject == request
            run.respond(waiting.id, approved)
            thread.join(WAIT)
            assert answers == [approved]

        assert run.view().waiting == ()

    def test_a_question_withdrawn_or_stopped_is_no_longer_waited_on(self, tmp_path: Path) -> None:
        run = Run("meeting", "Deploy.", tmp_path)
        question = ServerQuestion(server="proto", message="Deploy?", fields={"ok": {"type": "boolean"}})
        withdraw = threading.Event()

        def ask() -> Any:
            with asked_for(withdraw):
                return run.answer_question(question)

        thread, answers = self.asking(run, ask)
        withdraw.set()
        thread.join(WAIT)
        assert answers == [None] and run.view().waiting == ()

        thread, answers = self.asking(run, lambda: run.approve_call(ApprovalRequest("t", "s", "t", {})))
        run.stop()
        thread.join(WAIT)
        assert answers == [False]

    def test_a_form_is_read_as_its_fields_types(self) -> None:
        question = ServerQuestion(
            server="proto",
            message="Deploy?",
            fields={
                "name": {"type": "string"},
                "gpus": {"type": "integer"},
                "confirm": {"type": "boolean"},
                "region": {"enum": ["us", "eu"]},
            },
            required=("name",),
        )

        assert form_answer(question, {"name": "fold", "gpus": "2", "confirm": "yes", "region": "2"}) == (
            {"name": "fold", "gpus": 2, "confirm": True, "region": "eu"},
            None,
        )
        assert form_answer(question, {"name": " fold "}) == ({"name": "fold"}, None)
        assert form_answer(question, {"gpus": "2"}) == (None, "name must be filled in")
        assert form_answer(question, {"name": "x", "gpus": "two"}) == (None, "'two' is not a value gpus takes")

    def test_a_decision_is_changed_as_typed_on_the_page(self) -> None:
        proposed = step("team_meeting", participants=["Immunologist"], agenda="Compare.", agenda_questions=["Which?"])

        changed = changed_step(
            proposed,
            agenda=" Measure. ",
            agenda_questions="First?\n\n Second? ",
            participants="Immunologist, Geneticist,",
        )

        assert changed.agenda == "Measure."
        assert changed.agenda_questions == ["First?", "Second?"]
        assert changed.participants == ["Immunologist", "Geneticist"]
        assert changed.rationale == proposed.rationale
        assert changed_step(proposed) == proposed

    def test_a_decision_is_answered_only_with_a_step_or_none(self, tmp_path: Path) -> None:
        run = Run("project", GOAL, tmp_path, autonomous=False)
        thread, answers = self.asking(run, lambda: run.approve_step(1, step("finish", answer="A.")))

        with pytest.raises(TypeError):
            run.respond(run.view().waiting[0].id, True)
        run.set_autonomous(True)
        thread.join(WAIT)

        assert answers == [step("finish", answer="A.")]


class TestEvents:
    def test_writing_replaces_only_the_same_request_writing_last(self, tmp_path: Path) -> None:
        run = Run("meeting", "Agenda.", tmp_path)

        def writing(text: str, request: int) -> MeetingEvent:
            return MeetingEvent(kind="writing", meeting="m", text=text, data={"request": request})

        run.on_event(writing("A", 1))
        run.on_event(writing("A b", 1))
        run.on_event(writing("C", 2))
        run.on_event(MeetingEvent(kind="message", meeting="m", text="C d"))
        run.on_event(writing("E", 2))

        assert [event.text for event in run.view().events] == ["A b", "C", "C d", "E"]

    def test_a_stop_is_raised_only_from_events_before_work(self, tmp_path: Path) -> None:
        run = Run("project", GOAL, tmp_path)
        run.stop()

        with pytest.raises(RunStopped):
            run.on_event(MeetingEvent(kind="turn", meeting="m"))
        with pytest.raises(RunStopped):
            run.on_event(ProjectEvent(kind="round", round=1))
        with pytest.raises(RunStopped):
            run.steer(NextTurn(meeting="m", round=1, speaker="PI"))
        for event in (
            MeetingEvent(kind="finished", meeting="m", data={"transcript_path": "t.json"}),
            MeetingEvent(kind="failed", meeting="m"),
            MeetingEvent(kind="cell", meeting="m"),
            ProjectEvent(kind="finished"),
        ):
            run.on_event(event)

    def test_a_run_that_ended_ignores_a_stop(self, tmp_path: Path) -> None:
        run = Run("meeting", "Agenda.", tmp_path)
        run.start(lambda run: None)
        finished(run)

        run.stop()

        assert run.view().stopping is None and run.view().status == "completed"

    def test_cleanup_runs_even_when_the_work_fails(self, tmp_path: Path) -> None:
        cleaned = []
        run = Run("meeting", "Agenda.", tmp_path)

        def fail(run: Run) -> None:
            raise ValueError("Bad")

        run.start(fail, cleanup=[lambda: cleaned.append(True)])

        assert finished(run).view().error == "ValueError: Bad"
        assert cleaned == [True]


class TestSetup:
    def test_code_runs_where_it_is_asked_to(self, tmp_path: Path) -> None:
        assert CodeSetup().session(tmp_path) is None and CodeSetup().executor() is None

        local = CodeSetup(where="local", python="/usr/bin/python3")
        session = local.session(tmp_path / "local")
        assert isinstance(session, LocalSession) and session.python == "/usr/bin/python3"
        assert isinstance(local.executor(), LocalExecutor)

        plain = CodeSetup(where="docker", network=False)
        assert isinstance(plain.session(tmp_path / "plain"), DockerSession)
        executor = plain.executor()
        assert isinstance(executor, DockerExecutor)
        assert executor.image == DockerExecutor().image and not executor.allow_network
        assert CodeSetup(where="docker", sandbox="bio").docker_executor().image == sandbox_image("bio")

        with pytest.raises(ValueError):
            CodeSetup(where="cloud")
        with pytest.raises(ValueError):
            CodeSetup(sandbox="huge")

    def test_what_the_agents_are_told_of_biomnis_resources_is_one_of_three_and_leaves_the_lead_to_pick_by_default(
        self,
    ) -> None:
        assert CodeSetup().resources == "retrieve"
        assert [CodeSetup(resources=mode).resources for mode in ("retrieve", "all", "none")] == [
            "retrieve",
            "all",
            "none",
        ]
        with pytest.raises(ValueError, match="resources are one of retrieve, all, none, not 'some'"):
            CodeSetup(resources="some")

    def test_tools_are_given_by_name_and_unknown_ones_refused(self, tmp_path: Path) -> None:
        names = list(TOOL_REGISTRY)[:2]

        assert Connections(tools=names).open(Run("meeting", "A.", tmp_path), None) == tuple(  # type: ignore[arg-type]
            TOOL_REGISTRY[name] for name in names
        )
        with pytest.raises(ValueError, match="no tool"):
            Connections(tools=["astrology"])

    def test_a_server_neither_in_the_config_nor_a_preset_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no MCP server nowhere"):
            Connections(mcp_servers=["nowhere"]).open(Run("meeting", "A.", tmp_path), None)  # type: ignore[arg-type]

    def test_a_setup_is_saved_and_read_back_and_a_bad_one_is_not(self, tmp_path: Path) -> None:
        setup = individual(code={"where": "docker"}, connections={"tools": [next(iter(TOOL_REGISTRY))]})
        assert setup.code == CodeSetup(where="docker")

        (tmp_path / PROJECT_SETUP_FILE_NAME).write_text(json.dumps(setup.to_dict()))
        assert read_setup(tmp_path) == setup

        (tmp_path / PROJECT_SETUP_FILE_NAME).write_text(json.dumps({"meeting_type": "team", "agenda": "A."}))
        assert read_setup(tmp_path) is None

    def test_a_meeting_needs_an_agenda_and_a_team_meeting_members(self) -> None:
        with pytest.raises(ValueError, match="agenda"):
            individual(agenda=" ")
        with pytest.raises(ValueError, match="members"):
            MeetingSetup(meeting_type="team", agenda="A.", lead=person("Principal Investigator"))
        with pytest.raises(ValueError, match="goal"):
            project(goal="")
