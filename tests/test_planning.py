"""Tests for running a project to its goal: the team, the plan, each round's decision, and how it ends."""

import json
from pathlib import Path
from typing import Any

import pytest

from virtual_lab.agent import Agent
from virtual_lab.artifacts import CodeArtifacts, CodeFile
from virtual_lab.execution import LocalExecutor
from virtual_lab.memory import Finding, Findings
from virtual_lab.planning import NextStep, PlanTask, ResearchPlan, Review, run_project
from virtual_lab.project import Project, ProjectStateError
from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC
from virtual_lab.schemas import AgentSpec, TeamRoster
from virtual_lab.utils import compute_token_cost

from conftest import TEST_MODEL, FakeClient, parsed_response

GOAL = "Find which of two nanobodies binds KP.3 better."

LEAD = PRINCIPAL_INVESTIGATOR.with_model(TEST_MODEL)
CRITIC = SCIENTIFIC_CRITIC.with_model(TEST_MODEL)

CALL_COST = compute_token_cost(TEST_MODEL, 100, 20)


def spec(title: str) -> AgentSpec:
    return AgentSpec(title=title, expertise=f"{title.lower()} work", goal="help the project", role="advise the team")


def member(title: str) -> Agent:
    return spec(title).to_agent(model=TEST_MODEL)


def roster(*titles: str) -> Any:
    return parsed_response(parsed=TeamRoster(team_members=[spec(title) for title in titles]))


def plan(*tasks: str) -> Any:
    return parsed_response(parsed=ResearchPlan(tasks=list(tasks)))


def tasks(done: int, total: int = 2) -> list[PlanTask]:
    return [PlanTask(task=f"Task {index}", status="done" if index <= done else "to do") for index in range(1, total + 1)]


def step(action: str, done: int = 0, **fields: Any) -> NextStep:
    values: dict[str, Any] = {
        "plan": tasks(done),
        "progress": "Some.",
        "action": action,
        "rationale": "It is next.",
        "participants": [],
        "agenda": "",
        "agenda_questions": [],
        "findings": [],
        "add_members": [],
        "remove_members": [],
        "answer": "",
    }
    return NextStep(**{**values, **fields})


def decide(action: str, done: int = 0, **fields: Any) -> Any:
    return parsed_response(parsed=step(action, done, **fields))


def found(*claims: str) -> Any:
    return parsed_response(parsed=Findings(findings=[Finding(claim=claim, evidence=f"Shown for {claim}") for claim in claims]))


def review(met: bool, *objections: str) -> Any:
    return parsed_response(parsed=Review(goal_met=met, objections=list(objections)))


def code(contents: str) -> Any:
    return parsed_response(
        parsed=CodeArtifacts(
            files=[CodeFile(filename="analysis.py", language="python", description="Compares them.", contents=contents)]
        )
    )


def queue(fake_client: FakeClient, *responses: Any) -> None:
    fake_client.completions.parsed_responses.extend(responses)


def run(save_dir: Path, project: Project | None = None, **options: Any) -> Any:
    return run_project(project or Project(save_dir, GOAL), team_lead=LEAD, critic=CRITIC, **options)


def sent(call: dict[str, Any]) -> str:
    return "\n".join(str(message.get("content")) for message in call["messages"])


def calls_mentioning(fake_client: FakeClient, text: str) -> list[dict[str, Any]]:
    return [call for call in fake_client.completions.calls if text in sent(call)]


def ledger(save_dir: Path) -> list[str]:
    return [entry["name"] for entry in json.loads((save_dir / "project.json").read_text())["steps"]]


class TestRun:
    def test_a_project_is_staffed_planned_worked_on_and_finished_when_the_critic_agrees(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("team_meeting", participants=["Immunologist"], agenda="Compare the binding data."),
            found(),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Check the affinities."),
            found(),
            decide("finish", done=2, answer="Nanobody A binds better."),
            review(True),
        )

        report = run(tmp_path)

        assert report.status == "finished"
        assert report.answer == "Nanobody A binds better."
        assert report.proposed_answer is None
        assert [member["title"] for member in report.team] == ["Immunologist"]
        assert [round_.outcome for round_ in report.rounds] == ["done", "done", "finished"]
        assert ledger(tmp_path) == [
            "team",
            "plan",
            "round_01_decision",
            "round_01_meeting",
            "round_02_decision",
            "round_02_meeting",
            "round_03_decision",
            "round_03_review",
        ]
        assert report.rounds[0].steps == ["round_01_meeting"]
        assert report.plan == [task.model_dump() for task in tasks(2)]
        plan_record = json.loads((tmp_path / "meetings" / "metadata" / "plan.json").read_text())
        assert [agent["title"] for agent in plan_record["team"]] == ["Principal Investigator", "Immunologist", "Scientific Critic"]
        assert report.spent == pytest.approx(len(fake_client.completions.calls) * CALL_COST)
        assert json.loads((tmp_path / "report.json").read_text())["status"] == "finished"
        assert "Nanobody A binds better." in (tmp_path / "report.md").read_text()
        log = json.loads((tmp_path / "research_log.json").read_text())
        assert log["status"] == "finished"
        assert [entry["outcome"] for entry in log["rounds"]] == ["done", "done", "finished"]

    def test_each_meeting_is_told_the_goal_and_plan_and_given_what_came_before(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Check the affinities."),
            found(),
            decide("finish", done=2, answer="A."),
            review(True),
        )

        run(tmp_path)

        decision = calls_mentioning(fake_client, "This is round 2 of the project")[0]
        assert GOAL in sent(decision)
        assert "1. [done] Task 1" in sent(decision)
        assert "Immunologist worked on: Check the affinities." in sent(decision)

    def test_the_team_lead_is_not_told_the_limits_that_can_be_raised(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."), review(True))

        run(tmp_path, project=Project(tmp_path, GOAL, max_cost=1.0), max_rounds=7, max_stalled_rounds=5)

        decision = sent(calls_mentioning(fake_client, "This is round 1 of the project")[0])
        assert "7" not in decision and "$" not in decision

    def test_a_given_team_is_used_without_choosing_one(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, plan("Task 1"), decide("finish", answer="A."), review(True))

        report = run(tmp_path, team=(member("Chemist"),))

        assert ledger(tmp_path)[0] == "plan"
        assert report.team_changes == [
            {"round": 0, "added": [report.team[0]], "removed": [], "why": "The team was given."}
        ]

    def test_without_a_team_the_team_lead_plans_alone(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, plan("Task 1"), decide("finish", answer="A."), review(True))

        report = run(tmp_path, team=())

        assert report.status == "finished"
        record = json.loads((tmp_path / "meetings" / "metadata" / "plan.json").read_text())
        assert record["meeting_type"] == "individual"

    def test_a_chosen_team_is_kept_to_its_size_and_to_new_titles(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Scientific Critic", "Chemist", " ", "chemist", "Biologist", "Physicist"),
            plan("Task 1"),
            decide("finish", answer="A."),
            review(True),
        )

        report = run(tmp_path, max_team_size=2)

        assert [member["title"] for member in report.team] == ["Chemist", "Biologist"]
        why = report.team_changes[0]["why"]
        assert "Scientific Critic, chemist left out, as already on the project" in why
        assert "1 left out, as without a title" in why
        assert "Physicist left out, as over the limit" in why


class TestDecisions:
    def test_a_decision_naming_someone_off_the_team_is_recorded_and_explained(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", participants=["Virologist"], agenda="Check it."),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Check it."),
            found(),
            decide("finish", done=2, answer="A."),
            review(True),
        )

        report = run(tmp_path)

        assert report.rounds[0].outcome == "invalid"
        assert '"Virologist" cannot take part' in report.rounds[0].note
        assert "round_01_meeting" not in ledger(tmp_path)
        assert calls_mentioning(fake_client, "This is round 2 of the project")
        assert all('"Virologist"' in sent(call) for call in calls_mentioning(fake_client, "This is round 2 of the project"))
        # Only the team lead is told: the meeting it decided on next is not
        meeting = [call for call in calls_mentioning(fake_client, "Check it.") if "This is round" not in sent(call)]
        assert meeting and not any('"Virologist"' in sent(call) for call in meeting)

    @pytest.mark.parametrize(
        ("decision", "problem"),
        [
            (step("team_meeting", agenda="A.", participants=[]), "at least one participant"),
            (step("team_meeting", agenda="A.", participants=["Immunologist", "immunologist"]), "more than once"),
            (step("individual_meeting", agenda="A.", participants=["Immunologist", "Scientific Critic"]), "exactly one"),
            (step("individual_meeting", agenda="A.", participants=["Scientific Critic"]), "cannot take part"),
            (step("individual_meeting", agenda=" ", participants=["Immunologist"]), "needs an agenda"),
            (step("write_code", agenda="A.", participants=["Immunologist"]), "nothing to run code with"),
            (step("change_team"), "someone to add or remove"),
            (step("change_team", remove_members=["Principal Investigator"]), "not a member of the team"),
            (step("change_team", add_members=[spec("Immunologist")]), "on the project already"),
            (step("change_team", add_members=[spec("Principal Investigator")]), "on the project already"),
            (step("change_team", add_members=[spec("Chemist"), spec("chemist")]), "the same title"),
            (step("change_team", add_members=[spec("  ")]), "needs a title"),
            (step("change_team", add_members=[spec("Chemist"), spec("Biologist")]), "at most 2"),
            (step("finish", answer=" "), "needs the project's answer"),
            (step("finish", answer="A.").model_copy(update={"plan": []}), "plan was left empty"),
        ],
    )
    def test_a_decision_that_cannot_be_carried_out_says_why(
        self, fake_client: FakeClient, tmp_path: Path, decision: NextStep, problem: str
    ) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), parsed_response(parsed=decision))

        report = run(tmp_path, max_rounds=1, max_team_size=2)

        assert report.rounds[0].outcome == "invalid"
        assert problem in report.rounds[0].note
        assert ledger(tmp_path) == ["team", "plan", "round_01_decision"]

    def test_the_team_can_be_filled_to_its_size(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            parsed_response(parsed=step("change_team", add_members=[spec("Chemist")])),
        )

        report = run(tmp_path, max_rounds=1, max_team_size=2)

        assert report.rounds[0].outcome == "done"
        assert [member["title"] for member in report.team] == ["Immunologist", "Chemist"]

    def test_the_critics_objections_are_what_the_next_round_goes_on(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("finish", done=1, answer="A, probably."),
            review(False, "No affinity was measured."),
            decide("individual_meeting", done=2, participants=["Immunologist"], agenda="Measure the affinity."),
            found(),
            decide("finish", done=2, answer="A, at 2 nM."),
            review(True),
        )

        report = run(tmp_path)

        assert report.status == "finished"
        assert report.answer == "A, at 2 nM."
        assert report.proposed_answer is None
        assert report.rounds[0].outcome == "objected"
        assert report.rounds[0].note == "No affinity was measured."
        # The team is shown the objection too, since it is work the project did
        assert calls_mentioning(fake_client, "Measure the affinity.")
        assert all("No affinity was measured." in sent(call) for call in calls_mentioning(fake_client, "Measure the affinity."))

    def test_the_team_can_be_changed_and_every_change_is_recorded(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist", "Chemist"),
            plan("Task 1", "Task 2"),
            decide("change_team", add_members=[spec("Virologist")], remove_members=["chemist"], rationale="Need virology."),
            decide("individual_meeting", done=1, participants=["Virologist"], agenda="Check escape."),
            found(),
            decide("finish", done=2, answer="A."),
            review(True),
        )

        report = run(tmp_path)

        assert [member["title"] for member in report.team] == ["Immunologist", "Virologist"]
        change = report.team_changes[1]
        assert (change["round"], change["removed"], change["why"]) == (1, ["Chemist"], "Need virology.")
        assert change["added"][0]["title"] == "Virologist"
        assert change["added"][0]["model"] == TEST_MODEL
        assert report.rounds[0].team == ["Immunologist", "Virologist"]
        assert report.rounds[0].steps == []
        assert any("Virologist" in sent(call) for call in calls_mentioning(fake_client, "Check escape."))

    def test_code_is_written_run_and_reported_back(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("write_code", done=1, participants=["Immunologist"], agenda="Compare the affinities."),
            code("print('A binds at 2 nM')"),
            found(),
            decide("finish", done=2, answer="A."),
            review(True),
        )

        report = run(tmp_path, executor=LocalExecutor(warn=False))

        assert report.rounds[0].steps == ["round_01_code", "round_01_run", "round_01_findings"]
        assert ledger(tmp_path)[3:6] == ["round_01_code", "round_01_run", "round_01_findings"]
        decision = calls_mentioning(fake_client, "This is round 2 of the project")[0]
        assert "A binds at 2 nM" in sent(decision)
        assert "The code ran successfully" in sent(decision)
        code_meeting = calls_mentioning(fake_client, "Compare the affinities.")[0]
        assert "no one to answer it" in sent(code_meeting)

    def test_write_code_is_offered_only_with_something_to_run_it(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."), review(True))

        run(tmp_path)

        assert "write_code:" not in sent(calls_mentioning(fake_client, "This is round 1 of the project")[0])


class TestStopping:
    def test_a_project_that_never_finishes_stops_at_its_rounds(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="One."),
            found(),
            decide("individual_meeting", done=2, participants=["Immunologist"], agenda="Two."),
            found(),
        )

        report = run(tmp_path, max_rounds=2)

        assert report.status == "out_of_rounds"
        assert report.answer is None
        assert len(report.rounds) == 2

    def test_a_project_whose_plan_stops_moving_is_stopped(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="One."),
            found(),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Two."),
            found(),
            decide("finish", done=1, answer="A."),
            review(False, "Task 2 is not done."),
        )

        report = run(tmp_path, max_stalled_rounds=2)

        assert report.status == "stalled"
        assert len(report.rounds) == 3
        assert report.proposed_answer == "A."
        assert report.objections == ["Task 2 is not done."]
        assert "did not accept" in report.to_markdown()

    def test_a_project_that_runs_out_of_budget_ends_with_a_report(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, plan("Task 1"), decide("individual_meeting", participants=["Principal Investigator"], agenda="One."))
        # The plan takes four requests and the decision two, and the meeting is stopped before its third
        project = Project(tmp_path, GOAL, max_cost=7.5 * CALL_COST)

        report = run(tmp_path, project=project, team=())

        assert report.status == "out_of_budget"
        assert "in round 1" in report.reason
        assert report.rounds[0].outcome == "out_of_budget"
        assert len(fake_client.completions.calls) == 8
        assert (tmp_path / "report.md").is_file()
        assert json.loads((tmp_path / "research_log.json").read_text())["status"] == "out_of_budget"

    def test_a_project_out_of_budget_before_it_starts_says_so(self, fake_client: FakeClient, tmp_path: Path) -> None:
        report = run(tmp_path, project=Project(tmp_path, GOAL, max_cost=0.0))

        assert report.status == "out_of_budget"
        assert "before its first round" in report.reason
        assert report.rounds == []
        assert fake_client.completions.calls == []


    def test_a_run_that_fails_leaves_no_report_and_says_why_in_its_log(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="One."),
            found(),
        )
        assert run(tmp_path, max_rounds=1).status == "out_of_rounds"
        assert (tmp_path / "report.md").is_file()

        fake_client.completions.responses.append(RuntimeError("The connection dropped."))
        queue(fake_client, decide("individual_meeting", participants=["Immunologist"], agenda="Two."))
        with pytest.raises(RuntimeError):
            run(tmp_path, max_rounds=2)

        assert not (tmp_path / "report.md").exists() and not (tmp_path / "report.json").exists()
        log = json.loads((tmp_path / "research_log.json").read_text())
        assert log["status"] == "failed"
        assert log["error"] == {"type": "RuntimeError", "message": "The connection dropped."}


class TestApproval:
    def test_the_hook_sees_each_decision_and_can_change_it(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="Guess."),
            found(),
            decide("finish", done=1, answer="A."),
            review(True),
        )
        seen: list[tuple[int, str]] = []

        def approve(number: int, decision: NextStep) -> NextStep:
            seen.append((number, decision.action))
            if decision.action == "individual_meeting":
                return decision.model_copy(update={"agenda": "Measure instead."})
            return decision

        report = run(tmp_path, approve=approve)

        assert seen == [(1, "individual_meeting"), (2, "finish")]
        assert report.rounds[0].proposed["agenda"] == "Guess."
        assert report.rounds[0].approved["agenda"] == "Measure instead."  # type: ignore[index]
        assert calls_mentioning(fake_client, "Measure instead.")
        assert not calls_mentioning(fake_client, "Guess.\n")

    def test_the_hook_is_given_a_copy_so_what_was_proposed_is_kept(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."), review(True))

        def approve(number: int, decision: NextStep) -> NextStep:
            decision.answer = "B."
            return decision

        report = run(tmp_path, approve=approve)

        assert report.rounds[0].proposed["answer"] == "A."
        assert report.answer == "B."

    def test_the_hook_can_stop_the_project(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."))

        report = run(tmp_path, approve=lambda number, decision: None)

        assert report.status == "stopped"
        assert report.rounds[0].outcome == "stopped"
        assert report.rounds[0].approved is None
        assert "round_01_review" not in ledger(tmp_path)

    def test_an_approval_that_returns_something_else_is_refused(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."))

        with pytest.raises(TypeError, match="NextStep or None"):
            run(tmp_path, approve=lambda number, decision: True)

    def test_a_changed_decision_must_still_be_one_that_can_be_carried_out(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."))

        report = run(tmp_path, max_rounds=1, approve=lambda number, decision: decision.model_copy(update={"answer": ""}))

        assert report.rounds[0].outcome == "invalid"


class TestResuming:
    def test_a_project_stopped_at_its_rounds_carries_on_when_they_are_raised(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="One."),
            found(),
        )
        assert run(tmp_path, max_rounds=1).status == "out_of_rounds"
        calls = len(fake_client.completions.calls)
        first_decision = len(calls_mentioning(fake_client, "This is round 1 of the project"))

        queue(fake_client, decide("finish", done=2, answer="A."), review(True))
        report = run(tmp_path, max_rounds=3)

        assert report.status == "finished"
        # Only round 2 is new: the team, the plan, and round 1 are read back
        assert len(calls_mentioning(fake_client, "This is round 1 of the project")) == first_decision
        # The decision and the review, two requests each
        assert len(fake_client.completions.calls) - calls == 2 + 2
        assert [round_.number for round_ in report.rounds] == [1, 2]

    def test_a_project_out_of_budget_carries_on_when_it_is_raised(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, plan("Task 1"), decide("individual_meeting", participants=["Principal Investigator"], agenda="One."))
        team = ()
        assert run(tmp_path, project=Project(tmp_path, GOAL, max_cost=7.5 * CALL_COST), team=team).status == "out_of_budget"

        queue(fake_client, found(), decide("finish", done=1, answer="A."), review(True))
        report = run(tmp_path, project=Project(tmp_path, GOAL, max_cost=100 * CALL_COST), team=team)

        assert report.status == "finished"
        assert [round_.outcome for round_ in report.rounds] == ["done", "finished"]

    def test_an_approved_decision_is_not_asked_about_again(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="Guess."),
            found(),
        )
        changed = lambda number, decision: decision.model_copy(update={"agenda": "Measure instead."})  # noqa: E731
        run(tmp_path, max_rounds=1, approve=changed)

        asked: list[int] = []

        def approve(number: int, decision: NextStep) -> NextStep:
            asked.append(number)
            return decision

        queue(fake_client, decide("finish", done=1, answer="A."), review(True))
        report = run(tmp_path, max_rounds=2, approve=approve)

        assert asked == [2]
        assert report.rounds[0].approved["agenda"] == "Measure instead."  # type: ignore[index]
        assert report.status == "finished"

    def test_an_approval_outlives_a_step_that_failed(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="Guess."),
        )
        changed = lambda number, decision: decision.model_copy(update={"agenda": "Measure instead."})  # noqa: E731
        original = fake_client.completions.create

        def fail_the_meeting(**kwargs: Any) -> Any:
            if "Measure instead." in sent(kwargs):
                raise RuntimeError("The connection dropped.")
            return original(**kwargs)

        fake_client.completions.create = fail_the_meeting  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="connection dropped"):
            run(tmp_path, approve=changed)
        fake_client.completions.create = original  # type: ignore[method-assign]

        asked: list[int] = []

        def approve(number: int, decision: NextStep) -> NextStep:
            asked.append(number)
            return decision

        queue(fake_client, found(), decide("finish", done=1, answer="A."), review(True))
        report = run(tmp_path, approve=approve)

        assert asked == [2]
        assert report.rounds[0].approved["agenda"] == "Measure instead."  # type: ignore[index]

    def test_an_approval_of_another_decision_is_not_used(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="Guess."),
            found(),
        )
        run(tmp_path, max_rounds=1)
        log_path = tmp_path / "research_log.json"
        log = json.loads(log_path.read_text())
        log["approvals"]["1"]["proposed"]["agenda"] = "Something else."
        log_path.write_text(json.dumps(log))

        asked: list[int] = []

        def approve(number: int, decision: NextStep) -> NextStep:
            asked.append(number)
            return decision

        run(tmp_path, max_rounds=1, approve=approve)

        assert asked == [1]

    def test_an_approval_is_kept_by_a_run_that_stops_before_its_round(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="One."),
            found(),
            decide("individual_meeting", done=2, participants=["Immunologist"], agenda="Two."),
            found(),
        )
        changed = lambda number, decision: decision.model_copy(update={"agenda": f"{decision.agenda} Carefully."})  # noqa: E731
        run(tmp_path, max_rounds=2, approve=changed)
        run(tmp_path, max_rounds=1, approve=changed)

        queue(fake_client, decide("finish", done=2, answer="A."), review(True))
        report = run(tmp_path, max_rounds=3)

        assert report.status == "finished"
        assert [round_.approved["agenda"] for round_ in report.rounds[:2]] == ["One. Carefully.", "Two. Carefully."]  # type: ignore[index]

    def test_a_decision_taken_without_a_hook_is_not_asked_about_when_one_is_added(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="One."),
            found(),
        )
        run(tmp_path, max_rounds=1)

        asked: list[int] = []

        def approve(number: int, decision: NextStep) -> NextStep:
            asked.append(number)
            return decision.model_copy(update={"agenda": "Something else."})

        queue(fake_client, decide("finish", done=1, answer="A."), review(True))
        report = run(tmp_path, max_rounds=2, approve=approve)

        assert asked == [2]
        assert report.status == "finished"

    def test_a_decision_the_hook_stopped_is_asked_about_again(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."))
        assert run(tmp_path, approve=lambda number, decision: None).status == "stopped"

        queue(fake_client, review(True))
        report = run(tmp_path)

        assert report.status == "finished"

    def test_a_project_run_with_another_team_lead_is_refused(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."), review(True))
        run(tmp_path)

        with pytest.raises(ProjectStateError, match="team_member"):
            run_project(Project(tmp_path, GOAL), team_lead=LEAD.with_model("gpt-4o-mini"), critic=CRITIC)


class TestArguments:
    @pytest.mark.parametrize(
        ("option", "value"),
        [("max_rounds", 0), ("max_stalled_rounds", 0), ("meeting_rounds", -1), ("max_team_size", -1)],
    )
    def test_a_limit_out_of_range_is_refused(self, tmp_path: Path, option: str, value: int) -> None:
        with pytest.raises(ValueError, match=option):
            run(tmp_path, **{option: value})

    def test_the_team_lead_and_critic_must_differ(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="different titles"):
            run_project(Project(tmp_path, GOAL), team_lead=LEAD, critic=LEAD)

    @pytest.mark.parametrize(
        ("team", "problem"),
        [
            ((member("Chemist"), member("chemist")), "different titles"),
            ((member("Scientific Critic"),), "critic's title"),
            ((member(" "),), "needs a title"),
        ],
    )
    def test_a_given_team_must_have_titles_of_its_own(self, tmp_path: Path, team: tuple[Agent, ...], problem: str) -> None:
        with pytest.raises(ValueError, match=problem):
            run(tmp_path, team=team)

    def test_an_unknown_repair_option_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="bogus"):
            run(tmp_path, repair_options={"bogus": 1})


def work_calls(fake_client: FakeClient, agenda: str) -> list[str]:
    """What the step with this agenda was sent, the team lead's decisions aside."""
    return [sent(call) for call in calls_mentioning(fake_client, agenda) if "This is round" not in sent(call)]


class TestMemory:
    def test_what_a_meeting_finds_is_kept_and_listed_for_the_team_lead(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Measure them."),
            found("A binds at 2 nM.", " ", "B binds at 9 nM."),
            decide("finish", done=2, answer="A.", findings=["F1", "f2"]),
            review(True),
        )

        report = run(tmp_path)

        assert report.rounds[0].found == ["F1", "F2"]
        assert report.rounds[1].findings == ["F1", "F2"]
        assert [(finding["id"], finding["claim"], finding["source"], finding["round"]) for finding in report.findings] == [
            ("F1", "A binds at 2 nM.", "round_01_meeting", 1),
            ("F2", "B binds at 9 nM.", "round_01_meeting", 1),
        ]
        decision = sent(calls_mentioning(fake_client, "This is round 2 of the project")[0])
        assert "[F1] A binds at 2 nM.\n[F2] B binds at 9 nM." in decision
        assert "Findings: F1, F2" in decision
        assert "name the ones the step needs" in decision
        assert "- [F1] A binds at 2 nM." in report.to_markdown()
        saved = json.loads((tmp_path / "memory.json").read_text())
        assert [finding["id"] for finding in saved["findings"]] == ["F1", "F2"]
        assert json.loads((tmp_path / "research_log.json").read_text())["memory"] == "pick"

    def test_a_claim_over_several_lines_is_one_item_of_the_reports_list(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="Measure them."),
            found("A binds\n\nweakly."),
        )

        report = run(tmp_path, max_rounds=1)

        assert "- [F1] A binds weakly.\n" in report.to_markdown()

    def test_before_any_findings_the_team_lead_is_told_there_are_none(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A."), review(True))

        run(tmp_path)

        decision = sent(calls_mentioning(fake_client, "This is round 1 of the project")[0])
        assert "no findings yet" in decision
        assert "listed above by id" not in decision
        assert "findings so far" not in decision

    def test_a_step_is_given_the_findings_the_team_lead_names_and_no_others(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Measure them."),
            found("A binds at 2 nM.", "B binds at 9 nM."),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Explain the gap.", findings=["F2"]),
            found(),
            decide("finish", done=2, answer="A.", findings=["F1"]),
            review(True),
        )

        report = run(tmp_path)

        meeting = work_calls(fake_client, "Explain the gap.")
        assert meeting
        assert all("[F2] B binds at 9 nM.\n\nEvidence: Shown for B binds at 9 nM." in text for text in meeting)
        assert not any("A binds at 2 nM." in text for text in meeting)
        # Nor is it given the summary of the meeting before it, which its findings stand in for
        assert not any("Immunologist worked on: Measure them." in text for text in meeting)
        assert report.rounds[1].findings == ["F2"]

        review_call = sent(calls_mentioning(fake_client, "proposes to end the project")[0])
        assert "[F1] A binds at 2 nM.\n\nEvidence:" in review_call
        assert "[F2] B binds at 9 nM.\n\nEvidence:" not in review_call
        # The critic is still shown every finding by its claim
        assert "[F2] B binds at 9 nM.\n" in review_call

    def test_the_team_lead_is_told_of_earlier_work_by_its_findings_and_the_latest_in_full(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="One."),
            found("A binds."),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Two."),
            found(),
            decide("finish", done=2, answer="A."),
            review(True),
        )

        run(tmp_path)

        second = sent(calls_mentioning(fake_client, "This is round 2 of the project")[0])
        third = sent(calls_mentioning(fake_client, "This is round 3 of the project")[0])
        assert "Immunologist worked on: One.\n\nA response.\n\nFindings: F1" in second
        assert "Immunologist worked on: One.\n\nFindings: F1" in third
        assert "Immunologist worked on: Two.\n\nA response.\n\nFindings: none" in third

    def test_a_finding_id_that_was_never_made_is_refused(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="Measure them."),
            found("A binds."),
            decide("finish", done=1, answer="A.", findings=["F1", "F9"]),
        )

        report = run(tmp_path, max_rounds=2)

        assert report.rounds[1].outcome == "invalid"
        assert report.rounds[1].note == "No finding has the id F9. The findings: F1."

    def test_a_change_to_the_team_needs_no_findings(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("change_team", add_members=[spec("Chemist")], findings=["F9"]),
        )

        report = run(tmp_path, max_rounds=1)

        assert report.rounds[0].outcome == "done"

    def test_code_that_was_run_is_turned_into_findings_by_its_author(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("write_code", done=1, participants=["Immunologist"], agenda="Compare the affinities."),
            code("print('A binds at 2 nM')"),
            found("A binds at 2 nM, as the comparison printed."),
            decide("finish", done=2, answer="A.", findings=["F1"]),
            review(True),
        )

        report = run(tmp_path, executor=LocalExecutor(warn=False))

        stated = sent(calls_mentioning(fake_client, "The code you wrote for this agenda was run")[0])
        assert "Compare the affinities." in stated and "A binds at 2 nM" in stated
        assert report.rounds[0].found == ["F1"]
        assert report.findings[0]["source"] == "round_01_findings"
        assert "round_01_findings" in ledger(tmp_path)

    def test_with_bm25_a_step_is_given_the_findings_that_match_its_agenda(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Measure them."),
            found("Nanobody A escapes KP.3.", "Nanobody B is soluble.", "KP.3 escape is common."),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Why the KP.3 escape?"),
            found(),
            decide("finish", done=2, answer="A, and B is soluble."),
            review(True),
        )

        report = run(tmp_path, memory="bm25", findings_per_step=1)

        meeting = work_calls(fake_client, "Why the KP.3 escape?")
        assert meeting and all("[F3] KP.3 escape is common.\n\nEvidence:" in text for text in meeting)
        assert not any("\n\nEvidence: Shown for Nanobody A" in text for text in meeting)
        assert report.rounds[1].findings == ["F3"]
        # A finish is matched by its answer
        assert report.rounds[2].findings == ["F2"]
        assert "Leave findings empty" in sent(calls_mentioning(fake_client, "This is round 1 of the project")[0])

    def test_with_bm25_named_findings_are_not_checked(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, roster("Immunologist"), plan("Task 1"), decide("finish", answer="A.", findings=["F9"]), review(True))

        assert run(tmp_path, memory="bm25").status == "finished"

    def test_with_summaries_each_step_is_given_every_summary_and_no_findings_are_kept(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("write_code", done=1, participants=["Immunologist"], agenda="Compare the affinities."),
            code("print('A binds at 2 nM')"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Explain it."),
            decide("finish", done=2, answer="A.", findings=["F9"]),
            review(True),
        )

        report = run(tmp_path, memory="summaries", executor=LocalExecutor(warn=False))

        assert report.status == "finished"
        assert report.rounds[0].steps == ["round_01_code", "round_01_run"]
        assert report.findings == []
        meeting = work_calls(fake_client, "Explain it.")
        assert meeting and all("A binds at 2 nM" in text and "The plan the team made" in text for text in meeting)
        assert not calls_mentioning(fake_client, "findings")
        assert json.loads((tmp_path / "memory.json").read_text()) == {"findings": []}

    def test_a_project_carried_on_remembers_what_it_found(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="One."),
            found("A binds.", "B binds."),
        )
        assert run(tmp_path, max_rounds=1).status == "out_of_rounds"

        queue(
            fake_client,
            decide("individual_meeting", done=1, participants=["Immunologist"], agenda="Two.", findings=["F2"]),
            found("C binds."),
            decide("finish", done=2, answer="A.", findings=["F3"]),
            review(True),
        )
        report = run(tmp_path, max_rounds=3)

        assert report.status == "finished"
        assert [finding["id"] for finding in report.findings] == ["F1", "F2", "F3"]
        assert all("[F2] B binds.\n\nEvidence:" in text for text in work_calls(fake_client, "Two."))

    def test_a_project_carried_on_with_another_memory_is_refused(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(
            fake_client,
            roster("Immunologist"),
            plan("Task 1"),
            decide("individual_meeting", participants=["Immunologist"], agenda="One."),
            found("A binds."),
        )
        run(tmp_path, max_rounds=1)

        with pytest.raises(ProjectStateError):
            run(tmp_path, max_rounds=1, memory="summaries")


class TestMemoryArguments:
    def test_an_unknown_memory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="memory must be"):
            run(tmp_path, memory="everything")

    def test_findings_per_step_below_one_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="findings_per_step"):
            run(tmp_path, findings_per_step=0)
