"""Tests for the web interface's page: what its handlers do with what is typed on it, how it keeps
itself up to date, and who may reach it."""

import json
import sys
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from virtual_lab.approval import ApprovalRequest, ServerQuestion
import virtual_lab.ui as ui_package
from virtual_lab.ui import app as ui_app
from virtual_lab.ui.app import Interface, RunPanel, build_app, is_loopback, launch_ui, main
from virtual_lab.ui.runs import Lab, MeetingSetup, ProjectSetup, Run, Waiting, read_setup
from virtual_lab.ui.workspace import MEETING_NAME, Settings, Workspace

from conftest import TEST_MODEL, FakeClient, text_response
from test_planning import step
from test_ui_render import started, view

WAIT = 10


class Anything:
    def __eq__(self, other: object) -> bool:
        return True


ANY = Anything()


class StubRun:
    """A run as the page sees it: something that gives a view."""

    def __init__(self, current: Any) -> None:
        self.current = current

    def view(self) -> Any:
        return self.current


class StubLab:
    def __init__(self, run: StubRun | None) -> None:
        self.run = run

    def latest(self, kind: str) -> StubRun | None:
        return self.run


def panel(kind: str) -> RunPanel:
    built = RunPanel(kind)
    with gr.Blocks():
        built.build_feed()
        if kind == "project":
            built.build_decision()
        built.build_rail()
        if kind == "project":
            built.build_board()
    return built


def decision(identifier: int, action: str = "team_meeting", **fields: Any) -> Waiting:
    return Waiting(identifier, "decision", step(action, agenda="Compare.", participants=["Immunologist"], **fields), 1)


def interface(tmp_path: Path) -> Interface:
    return Interface(Lab(Workspace(tmp_path)))


def meeting_form(**changes: Any) -> dict[str, Any]:
    form: dict[str, Any] = {
        "meeting_type": "individual",
        "agenda": " Choose the epitopes. ",
        "questions": "Which?\n\n Why? ",
        "rules": "",
        "lead": "Immunologist",
        "members": [],
        "critic": "",
        "rounds": 0,
        "model": TEST_MODEL,
        "budget": 1.5,
        "temperature": 0.2,
        "tools": [],
        "servers": [],
        "code": "none",
        "resources": "retrieve",
        "summaries": [],
        "stream": False,
    }
    return {**form, **changes}


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


class TestStartingARun:
    def test_a_meeting_started_on_the_page_runs_with_what_was_typed(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("The receptor binding domain.")]
        page = interface(tmp_path)

        page.start_meeting(**meeting_form())
        run = page.run_of("meeting")
        assert run.wait(WAIT)
        setup = read_setup(run.directory)

        assert isinstance(setup, MeetingSetup)
        assert setup.agenda == "Choose the epitopes." and setup.agenda_questions == ["Which?", "Why?"]
        assert setup.max_cost == 1.5 and setup.temperature == 0.2 and setup.num_rounds == 0
        assert setup.lead["title"] == "Immunologist" and setup.lead["model"] == TEST_MODEL
        assert json.loads((run.directory / f"{MEETING_NAME}.json").read_text())

    def test_a_team_meeting_leaves_the_lead_out_of_the_members_and_an_individual_one_takes_a_critic(
        self, tmp_path: Path
    ) -> None:
        page = interface(tmp_path)
        captured: list[MeetingSetup] = []
        page.lab.start_meeting = captured.append  # type: ignore[method-assign, assignment]

        page.start_meeting(
            **meeting_form(
                meeting_type="team",
                lead="Principal Investigator",
                members=["Principal Investigator", "Immunologist"],
                critic="Scientific Critic",
                rounds=2,
            )
        )
        page.start_meeting(**meeting_form(critic="Scientific Critic"))

        team, individual = captured
        assert [member["title"] for member in team.members] == ["Immunologist"] and team.critic is None
        assert individual.members == [] and individual.critic is not None

    def test_where_the_code_runs_and_what_the_agents_are_told_of_resources_are_what_the_form_chose(
        self, tmp_path: Path
    ) -> None:
        page = interface(tmp_path)
        page.save_settings("", False, None, "docker", "bio", False, "", "")
        meetings: list[MeetingSetup] = []
        projects: list[ProjectSetup] = []
        page.lab.start_meeting = meetings.append  # type: ignore[method-assign, assignment]
        page.lab.start_project = lambda setup, directory=None: projects.append(setup)  # type: ignore[method-assign, assignment, misc]

        page.start_meeting(**meeting_form(code="docker", resources="all"))
        page.start_project(**project_form(code="local", resources="none"))

        assert (meetings[0].code.where, meetings[0].code.resources) == ("docker", "all")
        assert (meetings[0].code.sandbox, meetings[0].code.network) == ("bio", False)
        assert (projects[0].code.where, projects[0].code.resources) == ("local", "none")

    @pytest.mark.parametrize(
        ("changes", "said"),
        [
            ({"model": " "}, "Choose a model"),
            ({"lead": ""}, "Choose who works on it"),
            ({"lead": "Nobody"}, "no scientist titled"),
            ({"agenda": "  "}, "agenda"),
            ({"budget": -1}, "less than nothing"),
        ],
    )
    def test_a_meeting_without_what_it_needs_is_refused_with_the_reason(
        self, tmp_path: Path, changes: dict[str, Any], said: str
    ) -> None:
        with pytest.raises(gr.Error, match=said):
            interface(tmp_path).start_meeting(**meeting_form(**changes))

    def test_a_model_whose_key_is_not_set_is_refused_before_anything_starts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY")
        page = interface(tmp_path)

        with pytest.raises(gr.Error, match="OPENAI_API_KEY"):
            page.start_meeting(**meeting_form())

        assert page.lab.latest("meeting") is None
        assert not page.workspace.meetings_dir.exists() or not list(page.workspace.meetings_dir.iterdir())

    def test_a_second_run_of_a_kind_is_not_started_while_one_goes_on(self, tmp_path: Path) -> None:
        page = interface(tmp_path)
        page.lab._begin(Run("meeting", "Going.", tmp_path / "going"))
        page.lab._begin(Run("project", "Going.", tmp_path / "going-project"))

        with pytest.raises(gr.Error, match="A meeting is going on"):
            page.start_meeting(**meeting_form())
        with pytest.raises(gr.Error, match="A project is going on"):
            page.start_project(**project_form())
        with pytest.raises(gr.Error, match="A project is going on"):
            page.carry_on(str(tmp_path), 1, 10)

    def test_a_project_started_on_the_page_has_the_team_lead_and_critic_and_what_was_chosen(
        self, tmp_path: Path
    ) -> None:
        page = interface(tmp_path)
        captured: list[ProjectSetup] = []
        page.lab.start_project = lambda setup, directory=None: captured.append(setup)  # type: ignore[method-assign, assignment, misc]

        page.start_project(**project_form(autonomous=True, memory="all", budget=None))
        page.start_project(**project_form(choose_team=True, team=["Immunologist", "Geneticist"]))

        first, second = captured
        assert first.goal == "Find a nanobody." and first.autonomous and first.memory == "all"
        assert first.lead["title"] == "Principal Investigator" and first.critic["title"] == "Scientific Critic"
        assert first.max_cost is None and first.team == []
        assert [member["title"] for member in second.team] == ["Immunologist", "Geneticist"]

    def test_a_project_asked_to_have_the_team_chosen_by_the_person_needs_a_team(self, tmp_path: Path) -> None:
        with pytest.raises(gr.Error, match="Choose the team"):
            interface(tmp_path).start_project(**project_form(choose_team=True, team=[]))

    def test_a_project_with_no_goal_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(gr.Error, match="goal"):
            interface(tmp_path).start_project(**project_form(goal=" "))

    def test_a_project_not_started_on_the_page_cannot_be_carried_on_from_it(self, tmp_path: Path) -> None:
        directory = tmp_path / "other"
        directory.mkdir()

        with pytest.raises(gr.Error, match="not started here"):
            interface(tmp_path).carry_on(str(directory), 1, 10)
        with pytest.raises(gr.Error, match="Choose a project"):
            interface(tmp_path).carry_on("", 1, 10)


def project_form(**changes: Any) -> dict[str, Any]:
    form: dict[str, Any] = {
        "goal": " Find a nanobody. ",
        "autonomous": False,
        "choose_team": False,
        "team": [],
        "team_size": 3,
        "rounds": 10,
        "stalled": 3,
        "meeting_rounds": 1,
        "memory": "pick",
        "model": TEST_MODEL,
        "budget": 2,
        "tools": [],
        "servers": [],
        "code": "none",
        "resources": "retrieve",
        "stream": False,
    }
    return {**form, **changes}


class TestSteering:
    def going(self, tmp_path: Path, kind: str = "meeting") -> tuple[Interface, Run]:
        page = interface(tmp_path)
        run = Run(kind, "Going.", tmp_path / "going", autonomous=False)  # type: ignore[arg-type]
        page.lab._begin(run)
        return page, run

    def test_nothing_can_be_steered_before_a_run_is_started(self, tmp_path: Path) -> None:
        page = interface(tmp_path)

        for call in (
            lambda: page.add_note("meeting", "Hi."),
            lambda: page.toggle_pause("meeting"),
            lambda: page.stop("project", True),
        ):
            with pytest.raises(gr.Error, match="has been started"):
                call()
        page.set_autonomous(True)

    def test_a_note_needs_words_and_a_run_that_has_not_ended(self, tmp_path: Path) -> None:
        page, run = self.going(tmp_path)

        with pytest.raises(gr.Error, match="Write the note"):
            page.add_note("meeting", "  ")
        assert page.add_note("meeting", "Look at B.") == ""
        assert run.view().notes == ("Look at B.",)

        run.status = "completed"
        with pytest.raises(gr.Error, match="has ended"):
            page.add_note("meeting", "Too late.")

    def test_pause_turns_to_resume_and_back(self, tmp_path: Path) -> None:
        page, run = self.going(tmp_path)

        page.toggle_pause("meeting")
        assert run.view().paused
        page.toggle_pause("meeting")
        assert not run.view().paused

    def test_stop_asks_to_stop_now_or_after_the_step(self, tmp_path: Path) -> None:
        page, run = self.going(tmp_path, "project")

        page.stop("project", False)
        assert run.view().stopping == "after_step"
        page.stop("project", True)
        assert run.view().stopping == "now"

    def test_letting_a_project_run_on_its_own_is_remembered_by_the_run(self, tmp_path: Path) -> None:
        page, run = self.going(tmp_path, "project")

        page.set_autonomous(True)

        assert run.view().autonomous

    def test_an_approval_is_answered_only_while_it_is_the_one_the_page_showed(self, tmp_path: Path) -> None:
        page, run = self.going(tmp_path)
        request = ApprovalRequest("order", "lab", "place_order", {})
        answers = self.ask(run, lambda: run.approve_call(request))
        [waiting] = run.view().waiting

        with pytest.raises(gr.Error, match="no longer waiting"):
            page.answer_ask("meeting", {"meeting_ask": waiting.id + 1}, True, "")
        page.answer_ask("meeting", {"meeting_ask": waiting.id}, False, "")
        answers[0].join(WAIT)

        assert answers[1] == [False]
        with pytest.raises(gr.Error, match="no longer waiting"):
            page.answer_ask("meeting", {"meeting_ask": waiting.id}, True, "")

    def test_a_question_is_answered_from_the_form_typed_declined_or_done_at_its_page(self, tmp_path: Path) -> None:
        page, run = self.going(tmp_path)
        form = ServerQuestion("lab", "Which?", {"name": {"type": "string"}, "n": {"type": "integer"}}, ("name",))
        link = ServerQuestion("lab", "Sign in.", url="https://example.org")

        def answered(question: ServerQuestion, yes: bool, text: str = "") -> Any:
            thread, got = self.ask(run, lambda: run.answer_question(question))
            [waiting] = run.view().waiting
            page.answer_ask("meeting", {"meeting_ask": waiting.id}, yes, text)
            thread.join(WAIT)
            return got[0]

        assert answered(form, True, "name: fold\nn: 2\nstray line") == {"name": "fold", "n": 2}
        assert answered(form, False) is None
        assert answered(link, True) == {}

        thread, got = self.ask(run, lambda: run.answer_question(form))
        [waiting] = run.view().waiting
        with pytest.raises(gr.Error, match="name must be filled in"):
            page.answer_ask("meeting", {"meeting_ask": waiting.id}, True, "n: 2")
        page.answer_ask("meeting", {"meeting_ask": waiting.id}, False, "")
        thread.join(WAIT)

    def test_a_decision_is_approved_changed_or_stopped_as_the_page_showed_it(self, tmp_path: Path) -> None:
        page, run = self.going(tmp_path, "project")
        proposed = step("team_meeting", participants=["Immunologist"], agenda="Compare.")

        def decided(choice: str, **typed: str) -> Any:
            thread, got = self.ask(run, lambda: run.approve_step(1, proposed))
            [waiting] = run.view().waiting
            typed = {"agenda": "", "questions": "", "participants": "", "answer": "", **typed}
            page.decide({"project_decision": waiting.id}, choice, **typed)
            thread.join(WAIT)
            return got[0]

        assert decided("approve") == proposed
        assert decided("change", agenda="Measure.", participants="Immunologist, Geneticist").agenda == "Measure."
        assert decided("stop") is None

        with pytest.raises(gr.Error, match="no longer waiting"):
            page.decide({"project_decision": 99}, "approve", "", "", "", "")

    def ask(self, run: Run, call: Any) -> Any:
        answers: list[Any] = []
        thread = threading.Thread(target=lambda: answers.append(call()), daemon=True)
        thread.start()
        for _ in range(WAIT * 20):
            if run.view().waiting:
                break
            threading.Event().wait(0.05)
        return thread, answers


class TestSettingsExportAndHistory:
    def test_settings_are_saved_as_typed_and_a_blank_model_is_the_default(self, tmp_path: Path) -> None:
        page = interface(tmp_path)

        with pytest.warns(UserWarning, match="MCP config servers.json cannot be read"):
            page.save_settings(" ", True, 3, "docker", "bio", False, " /usr/bin/python3 ", " servers.json ")

        assert page.settings() == Settings(
            stream=True,
            max_cost=3.0,
            code="docker",
            sandbox="bio",
            network=False,
            python="/usr/bin/python3",
            mcp_config="servers.json",
        )
        page.save_settings("gpt-5.2", False, None, "none", "python", True, "", "")
        assert page.settings().max_cost is None and page.settings().model == "gpt-5.2"
        with pytest.raises(gr.Error, match="less than nothing"):
            page.save_settings("gpt-5.2", False, -1, "none", "python", True, "", "")

    def test_a_scientist_added_is_offered_and_used(self, tmp_path: Path) -> None:
        page = interface(tmp_path)
        page.workspace.save_scientist(
            {"title": "Virologist", "expertise": "viruses", "goal": "find escapes", "role": "judge designs"}
        )

        assert "Virologist" in page.scientist_titles()
        assert page.person("Virologist", "gpt-5.2")["model"] == "gpt-5.2"
        assert page.person("Immunologist", "gpt-5.2")["title"] == "Immunologist"

    def test_nothing_is_saved_without_a_result_and_a_project_without_a_report_is_not_exported(
        self, tmp_path: Path
    ) -> None:
        page = interface(tmp_path)

        with pytest.raises(gr.Error, match="no meeting to save"):
            page.export("meeting", None, False)
        with pytest.raises(gr.Error, match="no project to save"):
            page.export("project", "", True)
        with pytest.raises(gr.Error, match="no report yet"):
            page.export("project", str(tmp_path), False)

    def test_a_finished_meeting_is_exported_as_a_document_beside_its_transcript(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("The receptor binding domain.")]
        page = interface(tmp_path)
        page.start_meeting(**meeting_form())
        run = page.run_of("meeting")
        assert run.wait(WAIT)

        saved = Path(page.export("meeting", str(run.view().result), False))

        assert saved.suffix == ".html" and saved.parent == run.directory
        assert "The receptor binding domain." in saved.read_text()

    def test_the_history_lists_runs_and_marks_the_ones_going_on(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [text_response("Done.")]
        page = interface(tmp_path)
        page.start_meeting(**meeting_form())
        assert page.run_of("meeting").wait(WAIT)
        directory = page.workspace.new_directory("project", "Goal.")
        (directory / "project.json").write_text(json.dumps({"goal": "Goal.", "created_at": "2026-10-07T15:30:00"}))
        page.lab._begin(Run("project", "Goal.", directory))

        rows, paths = page.history("all")
        meetings, meeting_paths = page.history("meeting")

        assert len(rows) == len(paths) == 2
        assert {row[1]: row[3] for row in rows} == {"Meeting": "Completed", "Project": "Running"}
        assert [row[1] for row in meetings] == ["Meeting"] and len(meeting_paths) == 1

    def test_the_summaries_to_build_on_are_the_meetings_that_finished(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Done.")]
        page = interface(tmp_path)
        page.start_meeting(**meeting_form())
        assert page.run_of("meeting").wait(WAIT)

        [(label, path)] = page.summary_choices()

        assert "Choose the epitopes." in label and path.endswith(f"{MEETING_NAME}.json")


class TestKeepingThePageUpToDate:
    def tick(self, current: Any, built: RunPanel, seen: dict[str, Any]) -> dict[Any, Any]:
        return built.update(StubLab(StubRun(current)), seen)  # type: ignore[arg-type]

    def test_with_no_run_there_is_nothing_to_change(self) -> None:
        assert panel("meeting").update(StubLab(None), {}) == {}  # type: ignore[arg-type]

    def test_a_page_shown_a_run_for_the_first_time_is_told_all_of_it(self) -> None:
        built, seen = panel("meeting"), {}

        changes = self.tick(view([started()]), built, seen)

        assert {built.feed, built.status, built.note, built.send, built.stop, built.pause} <= set(changes)
        assert changes[built.note]["interactive"] is True
        assert changes[built.ask_group]["visible"] is False and changes[built.done_group]["visible"] is False

    def test_a_run_that_has_not_changed_is_told_only_its_clock_while_it_goes_on(self) -> None:
        built, seen = panel("meeting"), {}
        run = view([started()])
        self.tick(run, built, seen)

        assert set(self.tick(run, built, seen)) == {built.status}
        assert self.tick(view([started()], status="completed", version=1), built, seen) == {}

        changed = self.tick(view([started()], version=2), built, seen)
        assert built.feed in changed and built.status in changed

    def test_a_run_that_ends_turns_off_its_steering_and_offers_to_save_it(self) -> None:
        built, seen = panel("meeting"), {}
        self.tick(view([started()]), built, seen)

        changes = self.tick(
            view([started()], status="completed", version=2, result=Path("/x/transcript.json")), built, seen
        )

        assert changes[built.note]["interactive"] is False and changes[built.stop]["interactive"] is False
        assert changes[built.done_group]["visible"] is True

    def test_a_pause_changes_the_button_to_resume(self) -> None:
        built, seen = panel("meeting"), {}
        self.tick(view([started()]), built, seen)

        changes = self.tick(view([started()], paused=True, version=2), built, seen)

        assert changes[built.pause]["value"] == "Resume"

    def test_what_waits_for_the_person_is_shown_and_hidden_when_answered(self) -> None:
        built, seen = panel("meeting"), {}
        request = ApprovalRequest("order", "lab", "place_order", {})
        link = ServerQuestion("lab", "Sign in.", url="https://example.org")
        form = ServerQuestion("lab", "Which?", {"name": {"type": "string"}, "n": {"type": "integer"}})

        asked = self.tick(view([started()], waiting=(Waiting(1, "approval", request),), version=1), built, seen)
        assert asked[built.ask_group]["visible"] is True and asked[built.ask_yes]["value"] == "Approve"
        assert asked[built.ask_form]["visible"] is False

        linked = self.tick(view([started()], waiting=(Waiting(2, "question", link),), version=2), built, seen)
        assert linked[built.ask_yes]["value"] == "I have done it"

        filled = self.tick(view([started()], waiting=(Waiting(3, "question", form),), version=3), built, seen)
        assert filled[built.ask_form] == {"visible": True, "value": "name: \nn: ", "__type__": "update"}
        assert filled[built.ask_yes]["value"] == "Send"

        answered = self.tick(view([started()], version=4), built, seen)
        assert answered[built.ask_group]["visible"] is False

    def test_a_decision_fills_the_fields_and_shows_the_ones_its_step_uses(self) -> None:
        built, seen = panel("project"), {}

        meeting = self.tick(view([], kind="project", waiting=(decision(1),), version=1), built, seen)
        assert meeting[built.decision_group]["visible"] is True
        assert meeting[built.agenda] == {"value": "Compare.", "visible": True, "__type__": "update"}
        assert meeting[built.participants]["value"] == "Immunologist"
        assert meeting[built.answer]["visible"] is False

        finish = self.tick(
            view([], kind="project", waiting=(decision(2, "finish", answer="Nanobody A."),), version=2), built, seen
        )
        assert finish[built.answer] == {"value": "Nanobody A.", "visible": True, "__type__": "update"}
        assert finish[built.agenda]["visible"] is False

        gone = self.tick(view([], kind="project", version=3), built, seen)
        assert gone[built.decision_group]["visible"] is False

    def test_a_tab_opened_late_is_laid_out_again_without_losing_what_was_typed(self) -> None:
        """A tab not yet opened drops the updates that hide its parts, so opening it sends them again,
        but not the values, or what the person had typed into the fields would be lost."""
        built, seen = panel("project"), {}
        waiting = (decision(1),)
        self.tick(view([], kind="project", waiting=waiting, version=1), built, seen)
        assert self.tick(view([], kind="project", waiting=waiting, version=1), built, seen) == {built.status: ANY}

        built.show()
        again = self.tick(view([], kind="project", waiting=waiting, version=1), built, seen)

        assert again[built.decision_group]["visible"] is True
        assert again[built.answer] == {"visible": False, "__type__": "update"}
        assert all("value" not in again[field] for field in (built.agenda, built.questions, built.participants))
        assert built.feed in again and built.board in again
        assert self.tick(view([], kind="project", waiting=waiting, version=1), built, seen) == {built.status: ANY}

    def test_a_form_being_filled_in_is_not_reset_when_its_tab_is_opened(self) -> None:
        built, seen = panel("meeting"), {}
        form = ServerQuestion("lab", "Which?", {"name": {"type": "string"}})
        asking = view([started()], waiting=(Waiting(3, "question", form),), version=3)
        self.tick(asking, built, seen)

        built.show()
        again = self.tick(asking, built, seen)

        assert again[built.ask_group]["visible"] is True
        assert again[built.ask_form] == {"visible": True, "__type__": "update"}

    def test_a_new_run_replaces_what_the_page_showed(self) -> None:
        built, seen = panel("project"), {}
        self.tick(view([], kind="project", waiting=(decision(1),), version=1), built, seen)

        changes = self.tick(view([], kind="project", id="run-2", waiting=(decision(1),), version=1), built, seen)

        assert changes[built.agenda]["value"] == "Compare."
        assert changes[built.autonomous] == {"value": False, "__type__": "update"}


class TestWhoMayReachIt:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
    def test_this_machine_is_loopback(self, host: str) -> None:
        assert is_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.org", ""])
    def test_anywhere_else_is_not(self, host: str) -> None:
        assert not is_loopback(host)

    @pytest.fixture
    def launched(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: seen.update(kwargs))
        return seen

    @pytest.mark.parametrize("options", [{"share": True}, {"host": "0.0.0.0"}, {"host": "192.168.1.5", "password": ""}])
    def test_sharing_or_listening_beyond_this_machine_needs_a_password(
        self, options: dict[str, Any], launched: dict[str, Any], tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="give a password"):
            launch_ui(workspace=tmp_path, open_browser=False, block=False, **options)

        assert launched == {}
        assert list(tmp_path.iterdir()) == []

    def test_this_machine_alone_needs_no_password_and_nothing_else_is_served_but_the_workspace(
        self, launched: dict[str, Any], tmp_path: Path
    ) -> None:
        launch_ui(workspace=tmp_path, open_browser=False, block=False)

        assert launched["auth"] is None and launched["server_name"] == "127.0.0.1" and launched["share"] is False
        assert launched["allowed_paths"] == [str(tmp_path)]
        assert launched["prevent_thread_lock"] is True

    def test_a_password_lets_in_whoever_gives_it_with_any_name(self, launched: dict[str, Any], tmp_path: Path) -> None:
        launch_ui(workspace=tmp_path, host="0.0.0.0", password="s3cret", share=True, open_browser=False, block=False)
        auth = launched["auth"]

        assert auth("anyone", "s3cret") is True
        assert auth("anyone", "s3cre") is False and auth("anyone", "") is False and auth("anyone", "S3CRET") is False
        assert auth("anyone", "s3cret✓") is False
        assert launched["share"] is True

    def test_the_command_reads_its_password_from_the_environment_and_says_what_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(ui_app, "launch_ui", lambda **kwargs: calls.append(kwargs))
        monkeypatch.delenv("VIRTUAL_LAB_UI_PASSWORD", raising=False)

        main(["--workspace", str(tmp_path), "--no-browser", "--port", "7999"])
        assert calls[-1] == {
            "workspace": str(tmp_path),
            "host": "127.0.0.1",
            "port": 7999,
            "password": None,
            "share": False,
            "open_browser": False,
        }

        monkeypatch.setenv("VIRTUAL_LAB_UI_PASSWORD", "from-env")
        main(["--workspace", str(tmp_path), "--share"])
        assert calls[-1]["password"] == "from-env" and calls[-1]["share"] is True and calls[-1]["open_browser"] is True

    def test_the_command_refuses_with_a_usage_error_when_a_password_is_needed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("VIRTUAL_LAB_UI_PASSWORD", raising=False)
        monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: None)

        with pytest.raises(SystemExit) as stopped:
            main(["--workspace", str(tmp_path), "--share", "--no-browser"])

        assert stopped.value.code == 2
        assert "give a password" in capsys.readouterr().err


class TestThePage:
    def test_the_page_is_built_with_its_four_tabs_and_a_timer_that_keeps_it_up_to_date(self, tmp_path: Path) -> None:
        app = build_app(Lab(Workspace(tmp_path)))
        config = app.get_config_file()

        tabs = [item["props"]["label"] for item in config["components"] if item["type"] == "tabitem"]
        timers = [item for item in config["components"] if item["type"] == "timer"]

        assert tabs == ["Meeting room", "Project", "History", "Settings"]
        assert len(timers) == 1 and timers[0]["props"]["value"] == ui_app.TICK_SECONDS
        assert len(config["dependencies"]) > 30

    def test_the_header_shows_the_workspace_with_the_home_directory_as_a_tilde(self, tmp_path: Path) -> None:
        home = Path.home()

        assert ui_app.shown_path(home / "virtual_lab_workspace") == "~/virtual_lab_workspace"
        assert ui_app.shown_path(Path("/elsewhere/lab")) == "/elsewhere/lab"
        assert "&lt;" in ui_app.header_html(Workspace(tmp_path / "<b>"))

    def test_lines_and_budgets_are_read_as_typed(self) -> None:
        assert ui_app.lines(" a \n\n b ") == ["a", "b"] and ui_app.lines(None) == []
        assert ui_app.budget_of(None) is None and ui_app.budget_of("") is None  # type: ignore[arg-type]
        assert ui_app.budget_of(2) == 2.0
        with pytest.raises(gr.Error):
            ui_app.budget_of(-0.5)


class TestWithoutGradio:
    @pytest.fixture
    def reimported(self, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
        """The interface's module, to be imported again as if it had not been."""
        monkeypatch.delitem(sys.modules, "virtual_lab.ui.app")
        monkeypatch.delattr(ui_package, "app")

        def refuse(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("The interface was started")

        monkeypatch.setattr(gr.Blocks, "launch", refuse)
        return monkeypatch

    def test_a_missing_gradio_is_said_plainly_by_the_function_and_the_command(
        self, reimported: pytest.MonkeyPatch
    ) -> None:
        reimported.setitem(sys.modules, "gradio", None)

        with pytest.raises(ImportError, match=r"needs Gradio: pip install \"virtual-lab\[ui\]\""):
            ui_package.launch_ui()
        with pytest.raises(SystemExit) as stopped:
            ui_package.main([])

        assert str(stopped.value) == 'virtual-lab-ui: The web interface needs Gradio: pip install "virtual-lab[ui]"'

    def test_any_other_missing_module_is_not_hidden_behind_that_advice(self, reimported: pytest.MonkeyPatch) -> None:
        reimported.delattr(ui_package, "render")
        reimported.setitem(sys.modules, "virtual_lab.ui.render", None)

        with pytest.raises(ModuleNotFoundError, match="virtual_lab.ui.render"):
            ui_package.launch_ui()

    def test_the_command_passes_its_arguments_on(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(ui_app, "launch_ui", lambda **kwargs: calls.append(kwargs))

        ui_package.main(["--workspace", str(tmp_path), "--no-browser"])

        assert calls and calls[0]["workspace"] == str(tmp_path) and calls[0]["open_browser"] is False
