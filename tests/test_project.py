"""Tests for a project: its ledger, its budget across meetings and repairs, and resuming it."""

import json
import math
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from virtual_lab.agent import Agent
from virtual_lab.artifacts import CodeArtifacts, CodeFile
from virtual_lab.execution import LocalExecutor
from virtual_lab.project import Project, ProjectBudgetExceededError, ProjectStateError, fingerprint_inputs
from virtual_lab.provenance import MeetingRecord
from virtual_lab.resources import KnowHow, Resources
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.tools import Tool
from virtual_lab.utils import BudgetExceededError, CostUnknownError, MeetingUsage, compute_token_cost

from conftest import TEST_MODEL, FakeClient, parsed_response, text_response

# What one fake response costs: 100 input tokens and 20 output
CALL_COST = compute_token_cost(TEST_MODEL, 100, 20)

GOAL = "Design nanobodies against KP.3."


class Verdict(BaseModel):
    decision: str


def script(contents: str) -> CodeArtifacts:
    return CodeArtifacts(
        files=[CodeFile(filename="analysis.py", language="python", description="Does the analysis.", contents=contents)]
    )


def ledger(save_dir: Path) -> dict[str, Any]:
    return json.loads((save_dir / "project.json").read_text())


def ask(project: Project, member: Agent, agenda: str = "Propose an epitope.", **options: Any) -> Any:
    return project.meeting("individual", agenda, team_member=member, **options)


class TestLedger:
    def test_each_meeting_is_saved_under_its_name_and_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("First."), text_response("Second.")]
        project = Project(tmp_path, GOAL)

        first = ask(project, team_member)
        second = ask(project, team_member, agenda="Rank the epitopes.")

        assert first.transcript_path == tmp_path / "meetings" / "meeting_001.json"
        assert second.transcript_path == tmp_path / "meetings" / "meeting_002.json"
        saved = ledger(tmp_path)
        assert saved["goal"] == GOAL
        assert [step["name"] for step in saved["steps"]] == ["meeting_001", "meeting_002"]
        step = saved["steps"][0]
        assert step["kind"] == "meeting"
        assert step["status"] == "completed"
        assert step["error"] is None
        assert step["usage"]["num_calls"] == 1
        assert step["usage"]["cost"] == pytest.approx(CALL_COST)
        assert step["files"] == {"transcript": "meetings/meeting_001.json", "record": "meetings/metadata/meeting_001.json"}
        assert set(step["files_sha256"]) == {"transcript", "record"}
        assert step["elapsed_seconds"] >= 0
        assert saved["spent"] == pytest.approx(2 * CALL_COST)
        assert project.spent == pytest.approx(2 * CALL_COST)

    def test_a_meeting_can_be_named(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        result = ask(Project(tmp_path, GOAL), team_member, name="epitopes")

        assert result.transcript_path == tmp_path / "meetings" / "epitopes.json"
        assert ledger(tmp_path)["steps"][0]["name"] == "epitopes"

    @pytest.mark.parametrize("name", ["../escape", "a/b", "", ".", "..", "with space"])
    def test_a_name_that_is_not_a_filename_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, name: str
    ) -> None:
        with pytest.raises(ValueError, match="name"):
            ask(Project(tmp_path, GOAL), team_member, name=name)

        assert fake_client.completions.calls == []

    def test_a_structured_output_is_recorded(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=Verdict(decision="go"))]

        result = ask(Project(tmp_path, GOAL), team_member, output_schema=Verdict)

        assert result.output == Verdict(decision="go")
        assert ledger(tmp_path)["steps"][0]["files"]["output"] == "meetings/outputs/meeting_001.json"

    def test_the_projects_meeting_options_apply_and_a_meetings_own_win(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL, temperature=0.7, max_completion_tokens=300)

        ask(project, team_member)
        ask(project, team_member, temperature=0.1)

        first, second = fake_client.completions.calls
        assert (first["temperature"], first["max_completion_tokens"]) == (0.7, 300)
        assert (second["temperature"], second["max_completion_tokens"]) == (0.1, 300)

    @pytest.mark.parametrize("option", ["save_dir", "save_name", "session", "chat_models", "client", "meeting_type"])
    def test_what_the_project_sets_cannot_be_given_to_a_meeting(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, option: str
    ) -> None:
        with pytest.raises(TypeError, match=option):
            project = Project(tmp_path, GOAL)
            project.meeting("individual", "Propose an epitope.", team_member=team_member, **{option: None})

        assert fake_client.completions.calls == []

    @pytest.mark.parametrize("option", ["save_dir", "save_name", "meeting_type", "agenda"])
    def test_what_each_meeting_sets_cannot_be_a_default(self, tmp_path: Path, option: str) -> None:
        with pytest.raises(TypeError, match=option):
            Project(tmp_path, GOAL, **{option: None})

    def test_an_option_hold_meeting_does_not_take_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="no option colour"):
            Project(tmp_path, GOAL, colour="blue")

    def test_a_goal_is_required(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="goal"):
            Project(tmp_path, "  ")

    @pytest.mark.parametrize("limit", [-1.0, float("inf"), float("nan")])
    def test_a_limit_that_is_not_one_is_refused(self, tmp_path: Path, limit: float) -> None:
        with pytest.raises(ValueError, match="max_cost"):
            Project(tmp_path, GOAL, max_cost=limit)

    def test_the_callers_own_callbacks_are_still_called(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        heard: list[int] = []
        checked: list[int] = []

        ask(
            Project(tmp_path, GOAL),
            team_member,
            num_rounds=1,
            on_usage=lambda usage: heard.append(usage.num_calls),
            before_request=lambda: checked.append(1),
        )

        assert heard == [1, 2, 3]
        assert len(checked) == 3

    def test_an_unpriced_model_has_an_unknown_cost_rather_than_none(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response(model="an-unreleased-model")]
        project = Project(tmp_path, GOAL)

        ask(project, team_member.with_model("an-unreleased-model"))

        assert project.spent is None
        assert project.remaining is None
        assert ledger(tmp_path)["steps"][0]["usage"]["cost"] is None


class TestBudget:
    def test_meetings_share_the_projects_budget(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL, max_cost=2.5 * CALL_COST)

        ask(project, team_member)
        ask(project, team_member)
        ask(project, team_member)
        with pytest.raises(ProjectBudgetExceededError, match="The project has cost") as error:
            ask(project, team_member)

        assert len(fake_client.completions.calls) == 3
        assert error.value.spent == pytest.approx(3 * CALL_COST)
        assert error.value.limit == 2.5 * CALL_COST
        # Nothing was started, so nothing is recorded
        assert len(ledger(tmp_path)["steps"]) == 3

    def test_a_project_at_exactly_its_limit_starts_nothing_more(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL, max_cost=2 * CALL_COST)
        ask(project, team_member)
        ask(project, team_member)

        with pytest.raises(ProjectBudgetExceededError):
            ask(project, team_member)

        assert len(fake_client.completions.calls) == 2
        assert len(ledger(tmp_path)["steps"]) == 2

    def test_a_meeting_is_stopped_when_the_project_runs_out(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL, max_cost=1.5 * CALL_COST)

        with pytest.raises(ProjectBudgetExceededError) as error:
            ask(project, team_member, num_rounds=2)

        assert len(fake_client.completions.calls) == 2
        # It is the project's limit that stopped it, and the meeting's own error says so
        assert isinstance(error.value.__cause__, BudgetExceededError)
        step = ledger(tmp_path)["steps"][0]
        assert step["status"] == "failed"
        assert step["error"]["type"] == "BudgetExceededError"
        assert step["usage"]["num_calls"] == 2
        assert step["files"] == {"partial": "meetings/partial/meeting_001.json"}
        assert (tmp_path / "meetings" / "partial" / "meeting_001.json").is_file()
        assert project.spent == pytest.approx(2 * CALL_COST)

    def test_the_projects_limit_is_named_even_when_the_sum_rounds_short_of_it(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        first = CALL_COST
        three_calls = compute_token_cost(TEST_MODEL, 300, 60)
        # The meeting is given limit - first, which it reaches, but first + what it cost is less
        limit = math.nextafter(first + three_calls, math.inf)
        assert three_calls >= limit - first and first + three_calls < limit
        project = Project(tmp_path, GOAL, max_cost=limit)
        ask(project, team_member)

        with pytest.raises(ProjectBudgetExceededError):
            ask(project, team_member, num_rounds=5)

        assert len(fake_client.completions.calls) == 4

    def test_a_meetings_own_limit_is_its_own(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL, max_cost=10 * CALL_COST)

        with pytest.raises(BudgetExceededError) as error:
            ask(project, team_member, num_rounds=2, max_cost=1.5 * CALL_COST)

        assert not isinstance(error.value, ProjectBudgetExceededError)
        assert len(fake_client.completions.calls) == 2
        # The project goes on
        ask(project, team_member)

    def test_the_meeting_is_given_what_is_left(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL, max_cost=5 * CALL_COST)

        ask(project, team_member)
        result = ask(project, team_member)

        assert result.record.max_cost == pytest.approx(4 * CALL_COST)

    def test_what_was_spent_carries_over_to_the_next_run(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        ask(Project(tmp_path, GOAL), team_member, num_rounds=1)

        project = Project(tmp_path, GOAL, max_cost=2 * CALL_COST)

        assert project.spent == pytest.approx(3 * CALL_COST)
        assert project.remaining == 0.0
        with pytest.raises(ProjectBudgetExceededError):
            ask(project, team_member, agenda="Something new.", name="next")
        assert len(fake_client.completions.calls) == 3

    def test_a_failed_meeting_still_counts(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("One."), RuntimeError("the provider fell over")]
        project = Project(tmp_path, GOAL)

        with pytest.raises(RuntimeError, match="fell over"):
            ask(project, team_member, num_rounds=1)

        assert project.spent == pytest.approx(CALL_COST)
        assert ledger(tmp_path)["steps"][0]["error"] == {"type": "RuntimeError", "message": "the provider fell over"}

    def test_a_limit_needs_every_model_priced(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        with pytest.raises(CostUnknownError, match="an-unreleased-model"):
            ask(Project(tmp_path, GOAL, max_cost=1.0), team_member.with_model("an-unreleased-model"))

        assert fake_client.completions.calls == []

    def test_a_limit_cannot_be_kept_once_a_cost_is_unknown(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response(model="an-unreleased-model")]
        ask(Project(tmp_path, GOAL), team_member.with_model("an-unreleased-model"))

        with pytest.raises(CostUnknownError, match="meeting_001"):
            ask(Project(tmp_path, GOAL, max_cost=1.0), team_member, agenda="Something new.", name="next")

        assert len(fake_client.completions.calls) == 1

    def test_meetings_at_the_same_time_are_stopped_by_the_shared_budget(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        # Each meeting is given the whole remaining budget when it starts, and three calls each
        # would fit in it, so only checking the project's total before every request stops them
        project = Project(tmp_path, GOAL, max_cost=4 * CALL_COST)
        both_answered_once = threading.Barrier(2, timeout=10)
        errors: list[BaseException] = []

        def wait_for_the_other(usage: MeetingUsage) -> None:
            if usage.num_calls == 1:
                both_answered_once.wait()

        def hold(name: str) -> None:
            try:
                project.meeting(
                    "team",
                    "Plan the screen.",
                    name=name,
                    team_lead=team_lead,
                    team_members=(team_member,),
                    num_rounds=1,
                    on_usage=wait_for_the_other,
                )
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=hold, args=(name,)) for name in ("first", "second")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Unchecked, the two would make six calls
        assert 4 <= len(fake_client.completions.calls) <= 5
        assert errors and all(isinstance(error, ProjectBudgetExceededError) for error in errors)
        assert project.spent == pytest.approx(len(fake_client.completions.calls) * CALL_COST)
        assert {step["name"] for step in ledger(tmp_path)["steps"]} == {"first", "second"}


class TestResuming:
    def test_a_finished_meeting_is_read_back_not_held_again(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("One."), text_response("Critique."), text_response("Two.")]
        held = ask(Project(tmp_path, GOAL), team_member, num_rounds=1)
        calls = len(fake_client.completions.calls)

        project = Project(tmp_path, GOAL)
        read = ask(project, team_member, num_rounds=1)

        assert len(fake_client.completions.calls) == calls
        assert read.summary == held.summary == "Two."
        assert read.discussion == held.discussion
        assert read.transcript_path == held.transcript_path
        assert read.record_path == held.record_path
        assert read.record.to_dict() == held.record.to_dict()
        assert read.usage.to_dict() == held.usage.to_dict()
        assert read.cost == pytest.approx(held.cost)
        # Read back, it costs nothing more and is recorded once
        assert project.spent == pytest.approx(3 * CALL_COST)
        assert len(ledger(tmp_path)["steps"]) == 1

    def test_a_structured_output_is_read_back_as_its_schema(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Go ahead.")]
        fake_client.completions.parsed_responses = [parsed_response(parsed=Verdict(decision="go"))]
        held = ask(Project(tmp_path, GOAL), team_member, output_schema=Verdict)

        read = ask(Project(tmp_path, GOAL), team_member, output_schema=Verdict)

        assert read.output == held.output == Verdict(decision="go")
        assert read.summary == held.summary == "Go ahead."
        assert read.output_path == held.output_path

    def test_a_script_run_again_carries_on_where_it_stopped(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        def script_of_meetings(project: Project) -> list[str]:
            return [ask(project, team_member, agenda=f"Step {number}.").summary for number in range(4)]

        fake_client.completions.responses = [text_response("A."), text_response("B.")]
        with pytest.raises(ProjectBudgetExceededError):
            script_of_meetings(Project(tmp_path, GOAL, max_cost=2 * CALL_COST))

        fake_client.completions.responses = [text_response("C."), text_response("D.")]
        summaries = script_of_meetings(Project(tmp_path, GOAL, max_cost=4 * CALL_COST))

        assert summaries == ["A.", "B.", "C.", "D."]
        assert len(fake_client.completions.calls) == 4

    def test_a_failed_meeting_is_held_again(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [RuntimeError("the provider fell over"), text_response("Done.")]
        with pytest.raises(RuntimeError):
            ask(Project(tmp_path, GOAL), team_member)

        project = Project(tmp_path, GOAL)
        result = ask(project, team_member)

        assert result.summary == "Done."
        assert [step["status"] for step in ledger(tmp_path)["steps"]] == ["failed", "completed"]
        # Read back from then on
        assert ask(Project(tmp_path, GOAL), team_member).summary == "Done."
        assert len(fake_client.completions.calls) == 2

    def test_a_meeting_asked_for_with_other_inputs_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        ask(Project(tmp_path, GOAL), team_member)

        with pytest.raises(ProjectStateError, match="different agenda") as error:
            ask(Project(tmp_path, GOAL), team_member, agenda="Something else.")

        assert "another name" in str(error.value)
        assert len(fake_client.completions.calls) == 1

    @pytest.mark.parametrize(
        ("change", "named"),
        [
            ({"num_rounds": 1}, "num_rounds"),
            ({"temperature": 0.9}, "temperature"),
            ({"contexts": ("A finding.",)}, "contexts"),
            ({"output_schema": Verdict}, "output_schema"),
        ],
    )
    def test_each_input_that_differs_is_named(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, change: dict[str, Any], named: str
    ) -> None:
        ask(Project(tmp_path, GOAL), team_member)

        with pytest.raises(ProjectStateError, match=named):
            ask(Project(tmp_path, GOAL), team_member, **change)

    def test_a_default_spelled_out_is_the_same_input(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        ask(Project(tmp_path, GOAL), team_member)

        ask(Project(tmp_path, GOAL), team_member, num_rounds=0, contexts=(), output_schema=None, code_actions="tool")

        assert len(fake_client.completions.calls) == 1

    def test_a_meeting_that_failed_before_it_began_points_to_nothing(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="num_rounds"):
            ask(Project(tmp_path, GOAL), team_member, num_rounds=-1)

        step = ledger(tmp_path)["steps"][0]
        assert step["status"] == "failed"
        assert step["files"] == {}

    def test_another_agent_is_another_input(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        ask(Project(tmp_path, GOAL), team_member)

        with pytest.raises(ProjectStateError, match="team_member"):
            ask(Project(tmp_path, GOAL), team_member.with_model("gpt-4o-mini"))

    def test_a_tool_described_differently_is_another_input(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        def lookup(query: str) -> str:
            return query

        def tool(description: str) -> Tool:
            return Tool(name="lookup", description=description, parameters={"type": "object"}, function=lookup)

        ask(Project(tmp_path, GOAL), team_member, tools=(tool("Looks a gene up."),))
        ask(Project(tmp_path, GOAL), team_member, tools=(tool("Looks a gene up."),))
        assert len(fake_client.completions.calls) == 1

        with pytest.raises(ProjectStateError, match="tools"):
            ask(Project(tmp_path, GOAL), team_member, tools=(tool("Looks a protein up."),))

    def test_resources_with_other_contents_are_another_input(self) -> None:
        def resources(content: str) -> Resources:
            return Resources(know_how=(KnowHow("guide", "A Guide", "How to.", content),))

        def fingerprint(given: Resources) -> dict[str, str]:
            return fingerprint_inputs(hold_meeting, {"resources": given})

        assert fingerprint(resources("Do this.")) == fingerprint(resources("Do this."))
        assert fingerprint(resources("Do this.")) != fingerprint(resources("Do that."))

    def test_a_schema_that_changed_is_another_input(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=Verdict(decision="go"))]
        ask(Project(tmp_path, GOAL), team_member, output_schema=Verdict)

        class Verdict2(BaseModel):  # the same name, as a schema edited between runs has
            decision: str
            reason: str

        Verdict2.__name__ = Verdict2.__qualname__ = "Verdict"
        Verdict2.__module__ = Verdict.__module__

        with pytest.raises(ProjectStateError, match="output_schema"):
            ask(Project(tmp_path, GOAL), team_member, output_schema=Verdict2)

    def test_what_only_limits_or_carries_a_meeting_is_not_an_input(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        ask(Project(tmp_path, GOAL), team_member)

        ask(
            Project(tmp_path, GOAL, max_cost=1.0),
            team_member,
            max_cost=0.5,
            max_retries=7,
            on_usage=lambda usage: None,
            before_request=lambda: None,
        )

        assert len(fake_client.completions.calls) == 1

    def test_another_goal_is_another_project(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        Project(tmp_path, GOAL)

        with pytest.raises(ProjectStateError, match="another goal"):
            Project(tmp_path, "Find a drug.")

    def test_a_transcript_changed_since_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        result = ask(Project(tmp_path, GOAL), team_member)
        result.transcript_path.write_text("[]")

        with pytest.raises(ProjectStateError, match="transcript .* has changed"):
            ask(Project(tmp_path, GOAL), team_member)

    def test_a_record_that_is_missing_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        result = ask(Project(tmp_path, GOAL), team_member)
        result.record_path.unlink()

        with pytest.raises(ProjectStateError, match="record .* is missing"):
            ask(Project(tmp_path, GOAL), team_member)

    def test_a_step_interrupted_without_saying_so_still_counts(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A process killed outright leaves its step running in the ledger, with the cost so far
        project = Project(tmp_path, GOAL)
        project._begin("meeting_001", "meeting", {"agenda": "x"})
        project._counter("meeting_001", None)(usage_of(calls=2))

        resumed = Project(tmp_path, GOAL, max_cost=10 * CALL_COST)

        step = resumed.steps[0]
        assert step.status == "interrupted"
        assert step.error is not None and step.error["type"] == "Interrupted"
        assert resumed.spent == pytest.approx(2 * CALL_COST)
        assert "meeting_001" in capsys.readouterr().out
        assert ledger(tmp_path)["steps"][0]["status"] == "interrupted"
        # And it is held again when asked for
        assert ask(resumed, team_member).summary == "A response."
        assert [step.status for step in resumed.steps] == ["interrupted", "completed"]

    def test_a_name_running_already_cannot_be_started_again(self, tmp_path: Path) -> None:
        project = Project(tmp_path, GOAL)
        project._begin("busy", "meeting", {})

        with pytest.raises(ValueError, match="already running"):
            project._begin("busy", "meeting", {})

    def test_a_name_taken_by_another_kind_of_step_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL)
        ask(project, team_member, name="analysis")

        with pytest.raises(ProjectStateError, match="meeting named"):
            project.repair(script("print(1)"), team_member, LocalExecutor(warn=False), name="analysis")

    def test_steps_are_copies(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        project = Project(tmp_path, GOAL)
        ask(project, team_member)

        project.steps[0].usage["cost"] = 100.0
        project.steps[0].status = "failed"

        assert project.steps[0].status == "completed"
        assert project.spent == pytest.approx(CALL_COST)


def usage_of(calls: int) -> MeetingUsage:
    from conftest import make_usage

    usage = MeetingUsage()
    for _ in range(calls):
        usage.add(model=TEST_MODEL, usage=make_usage())

    return usage


class TestRepair:
    def test_code_is_run_and_repaired_for_the_project(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=script("print('fixed')"))]
        project = Project(tmp_path, GOAL)

        outcome = project.repair(script("raise ValueError('boom')"), team_member, LocalExecutor(warn=False))

        assert outcome.succeeded and outcome.was_repaired
        assert outcome.paths == (tmp_path / "meetings" / "artifacts" / "repair_001" / "analysis.py",)
        step = ledger(tmp_path)["steps"][0]
        assert (step["name"], step["kind"], step["status"]) == ("repair_001", "repair", "completed")
        assert step["files"] == {
            "record": "meetings/executions/repair_001.json",
            "code:analysis.py": "meetings/artifacts/repair_001/analysis.py",
        }
        assert project.spent == pytest.approx(CALL_COST)

    def test_a_repair_already_run_is_read_back(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=script("print('fixed')"))]
        ran = Project(tmp_path, GOAL).repair(script("raise ValueError('boom')"), team_member, LocalExecutor(warn=False))

        read = Project(tmp_path, GOAL).repair(script("raise ValueError('boom')"), team_member, LocalExecutor(warn=False))

        assert len(fake_client.completions.parse_calls) == 1
        assert read.to_dict() == ran.to_dict()
        assert read.report() == ran.report()
        assert read.paths == ran.paths

    def test_code_changed_since_it_was_run_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        Project(tmp_path, GOAL).repair(script("print(1)"), team_member, LocalExecutor(warn=False), name="a")
        (tmp_path / "meetings" / "artifacts" / "a" / "analysis.py").write_text("print('changed')")

        with pytest.raises(ProjectStateError, match="code:analysis.py"):
            Project(tmp_path, GOAL).repair(script("print(1)"), team_member, LocalExecutor(warn=False), name="a")

    def test_other_code_under_the_same_name_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        Project(tmp_path, GOAL).repair(script("print(1)"), team_member, LocalExecutor(warn=False))

        with pytest.raises(ProjectStateError, match="artifacts"):
            Project(tmp_path, GOAL).repair(script("print(2)"), team_member, LocalExecutor(warn=False))

    def test_another_executor_is_another_input(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        Project(tmp_path, GOAL).repair(script("print(1)"), team_member, LocalExecutor(warn=False))

        with pytest.raises(ProjectStateError, match="executor"):
            Project(tmp_path, GOAL).repair(script("print(1)"), team_member, LocalExecutor(warn=False, timeout=5))

    def test_repairs_share_the_projects_budget(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        project = Project(tmp_path, GOAL, max_cost=1.5 * CALL_COST)
        ask(project, team_member)
        fake_client.completions.parsed_responses = [parsed_response(parsed=script("raise KeyError('again')"))]

        with pytest.raises(ProjectBudgetExceededError):
            project.repair(script("raise ValueError('boom')"), team_member, LocalExecutor(warn=False))

        assert len(fake_client.completions.parse_calls) == 1
        step = ledger(tmp_path)["steps"][1]
        assert step["status"] == "failed"
        assert step["usage"]["num_calls"] == 1
        assert step["files"] == {"partial": "meetings/partial/executions/repair_001.json"}
        assert (tmp_path / "meetings" / "partial" / "executions" / "repair_001.json").is_file()
        assert project.spent == pytest.approx(2 * CALL_COST)

    def test_the_projects_models_are_asked_for_repairs(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from conftest import fake_llm

        other = FakeClient()
        other.completions.parsed_responses = [parsed_response(parsed=script("print('fixed')"))]
        project = Project(tmp_path, GOAL, chat_models={TEST_MODEL: fake_llm(other)})

        project.repair(script("raise ValueError('boom')"), team_member, LocalExecutor(warn=False))

        assert len(other.completions.parse_calls) == 1
        assert fake_client.completions.parse_calls == []

    def test_an_option_run_with_repair_does_not_take_is_refused(self, team_member: Agent, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="save_dir"):
            Project(tmp_path, GOAL).repair(script("print(1)"), team_member, LocalExecutor(warn=False), save_dir=tmp_path)


class TestRebuilding:
    def test_a_meetings_usage_is_rebuilt_from_its_record(self) -> None:
        usage = usage_of(calls=3)

        assert MeetingUsage.from_dict(usage.to_dict()) == usage

    def test_a_meetings_record_is_rebuilt_from_its_file(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        result = ask(Project(tmp_path, GOAL), team_member, num_rounds=1)

        rebuilt = MeetingRecord.from_dict(json.loads(result.record_path.read_text()))

        assert rebuilt.to_dict() == result.record.to_dict()
