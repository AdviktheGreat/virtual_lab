"""Tests for the web interface's workspace, where meetings, projects, and settings are kept, and
for what it offers to choose from."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from virtual_lab.constants import DEFAULT_MODEL
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.tools import DATABASE_TOOLS
from virtual_lab.ui import library
from virtual_lab.ui.workspace import (
    MAX_SLUG_CHARS,
    MEETING_NAME,
    STOPPED_ERROR_TYPE,
    Settings,
    Workspace,
    slug,
)

from conftest import TEST_MODEL, FakeClient, text_response

NOW = datetime(2026, 10, 7, 15, 30)


def meeting(workspace: Workspace, agenda: str, fake_client: FakeClient, **options: object) -> Path:
    directory = workspace.new_directory("meeting", agenda, now=NOW)
    hold_meeting(
        meeting_type="individual",
        agenda=agenda,
        save_dir=directory,
        save_name=MEETING_NAME,
        team_member=library.scientist("Immunologist").with_model(TEST_MODEL),
        **options,  # type: ignore[arg-type]
    )
    return directory


def ledger(directory: Path, goal: str, created_at: str, steps: list[dict], report: dict | None = None) -> None:
    directory.mkdir(parents=True)
    (directory / "project.json").write_text(json.dumps({"goal": goal, "created_at": created_at, "steps": steps}))
    if report is not None:
        (directory / "report.json").write_text(json.dumps(report))


class TestSlug:
    def test_a_title_becomes_lower_case_words_joined_by_dashes(self) -> None:
        assert slug("Design Nanobodies for KP.3!") == "design-nanobodies-for-kp-3"

    def test_a_long_title_is_cut_at_a_word(self) -> None:
        name = slug("word " * 40)
        assert len(name) <= MAX_SLUG_CHARS
        assert name.endswith("word")

    def test_a_title_with_no_letters_is_untitled(self) -> None:
        assert slug("?!") == "untitled"


class TestWorkspace:
    def test_a_new_directory_is_named_after_when_and_what_and_never_reused(self, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)

        first = workspace.new_directory("meeting", "Choose the epitopes", now=NOW)
        second = workspace.new_directory("meeting", "Choose the epitopes", now=NOW)
        project = workspace.new_directory("project", "Design nanobodies", now=NOW)

        assert first == workspace.meetings_dir / "2026-10-07_1530_choose-the-epitopes"
        assert second.name == "2026-10-07_1530_choose-the-epitopes-2"
        assert project.parent == workspace.projects_dir
        assert first.is_dir() and second.is_dir()

    def test_settings_are_kept_and_unknown_or_unreadable_ones_ignored(self, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)
        assert workspace.load_settings() == Settings()

        workspace.save_settings(Settings(model=TEST_MODEL, max_cost=None, code="docker"))
        assert workspace.load_settings() == Settings(model=TEST_MODEL, max_cost=None, code="docker")

        workspace.settings_path.write_text(json.dumps({"stream": False, "colour": "blue"}))
        assert workspace.load_settings() == Settings(stream=False)

        workspace.settings_path.write_text("{not json")
        assert workspace.load_settings() == Settings()

    def test_a_finished_meeting_is_listed_with_its_agenda_status_and_cost(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        workspace = Workspace(tmp_path)
        directory = meeting(workspace, "Choose the epitopes.", fake_client)

        [entry] = workspace.entries()

        assert entry.kind == "meeting"
        assert entry.path == directory
        assert entry.title == "Choose the epitopes."
        assert entry.status == "completed"
        assert entry.cost is not None and entry.cost > 0
        assert entry.transcript == directory / f"{MEETING_NAME}.json"
        assert workspace.find(directory) == entry

    def test_a_meeting_that_failed_is_listed_with_what_it_said(self, fake_client: FakeClient, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)
        fake_client.completions.responses = [text_response("First."), RuntimeError("down")]
        with pytest.raises(RuntimeError):
            meeting(workspace, "Choose.", fake_client, num_rounds=1)

        [entry] = workspace.entries()

        assert entry.status == "failed"
        assert entry.transcript == entry.path / "partial" / f"{MEETING_NAME}.json"

    def test_a_meeting_the_person_stopped_is_listed_as_stopped(self, fake_client: FakeClient, tmp_path: Path) -> None:
        from virtual_lab.ui.runs import RunStopped

        assert RunStopped.__name__ == STOPPED_ERROR_TYPE
        workspace = Workspace(tmp_path)
        turns = []

        def steer(turn: object) -> None:
            turns.append(turn)
            if len(turns) == 2:
                raise RunStopped()

        with pytest.raises(RunStopped):
            meeting(workspace, "Choose.", fake_client, num_rounds=1, steer=steer)

        assert [entry.status for entry in workspace.entries()] == ["stopped"]

    def test_a_meeting_with_nothing_saved_yet_is_not_listed(self, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)
        workspace.new_directory("meeting", "Running.")

        assert workspace.entries() == []

    def test_projects_are_listed_with_how_they_stand_and_what_they_cost(self, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path)
        priced = [{"status": "completed", "usage": {"cost": 0.25}}, {"status": "completed", "usage": {"cost": 0.5}}]
        ledger(workspace.projects_dir / "a", "Finished.", "2026-10-01T00:00:00Z", priced, {"status": "finished"})
        ledger(workspace.projects_dir / "b", "Running.", "2026-10-02T00:00:00Z", [{"status": "running", "usage": {}}])
        ledger(workspace.projects_dir / "c", "Stopped.", "2026-10-03T00:00:00Z", [])
        (workspace.projects_dir / "d").mkdir()

        entries = workspace.entries()

        assert [(entry.title, entry.status) for entry in entries] == [
            ("Stopped.", "not finished"),
            ("Running.", "running"),
            ("Finished.", "finished"),
        ]
        assert [entry.cost for entry in entries] == [0, None, 0.75]

    def test_only_a_directory_of_the_workspace_is_found(self, tmp_path: Path) -> None:
        workspace = Workspace(tmp_path / "workspace")
        ledger(tmp_path / "elsewhere", "Goal.", "", [])

        assert workspace.find(tmp_path / "elsewhere") is None


class TestLibrary:
    def test_scientists_are_found_by_title_and_have_distinct_ones(self) -> None:
        titles = [agent.title for agent in library.SCIENTISTS]

        assert len(set(titles)) == len(titles)
        assert library.scientist("Biostatistician").title == "Biostatistician"
        with pytest.raises(KeyError):
            library.scientist("Astrologer")

    def test_the_default_model_comes_first_and_none_twice(self) -> None:
        models = library.models()

        assert models[0] == DEFAULT_MODEL
        assert len(set(models)) == len(models)
        assert TEST_MODEL in models

    def test_a_provider_is_ready_once_its_variables_are_set_and_no_value_is_read_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        statuses = {status.provider: status for status in library.provider_statuses()}

        assert statuses["OpenAI"].ready
        assert not statuses["Anthropic"].ready
        assert "sk-secret" not in repr(library.provider_statuses())
        assert library.missing_keys([TEST_MODEL, "claude-sonnet-4-5", "claude-opus-4-1"]) == ["ANTHROPIC_API_KEY"]

    def test_every_database_tool_is_offered_by_its_name(self) -> None:
        choices = library.tool_choices()

        assert [name for _, name in choices] == [tool.name for tool in DATABASE_TOOLS]
        assert all(label.startswith(f"{name}: ") for label, name in choices)
