"""Tests for running a benchmark: solving, reading answers as Biomni reads them, scoring, and resuming."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from virtual_lab.agent import Agent
from virtual_lab.benchmarks import Benchmark, MultipleChoiceOutput, Question
from virtual_lab.evaluation import (
    EXTRACTION_PROMPT,
    Attempt,
    SingleAgent,
    SolverContext,
    TeamMeeting,
    describe_solver,
    run_benchmark,
    transcript_text,
)
from virtual_lab.session import LocalSession
from virtual_lab.tools import PUBMED_TOOL
from virtual_lab.utils import CostUnknownError, MeetingUsage, UsageTracker, combine_usage, compute_token_cost

from conftest import TEST_MODEL, FakeClient, make_usage, parsed_response, text_response

# What one fake response costs: 100 input tokens and 20 output
CALL_COST = compute_token_cost(TEST_MODEL, 100, 20)


def questions(count: int = 3, task: str = "quiz") -> list[Question]:
    return [Question(benchmark="toy", task=task, id=index, prompt=f"Question {index}? Options A or B.", answer="A") for index in range(count)]


class Toy(Benchmark):
    name = "toy"


class OtherSchema(BaseModel):
    """A schema that is not the benchmark's."""


def choice(letter: str | None) -> Any:
    return parsed_response(parsed=MultipleChoiceOutput(choice=letter))


@pytest.fixture
def toy() -> Toy:
    return Toy(questions())


def answer_each(fake_client: FakeClient, *letters: str | None) -> None:
    """Queues, for each question, the agent's answer and the extractor's reading of it."""
    fake_client.completions.responses = [text_response(f"[ANSWER]{letter}[/ANSWER]") for letter in letters]
    fake_client.completions.parsed_responses = [choice(letter) for letter in letters]


def run(toy: Benchmark, solver: Any, save_dir: Path, **kwargs: Any) -> Any:
    return run_benchmark(toy, solver, save_dir, extractor=TEST_MODEL, **kwargs)


class TestSingleAgent:
    def test_each_question_is_the_agenda_and_its_answer_is_read_from_the_transcript(
        self, fake_client: FakeClient, team_member: Agent, toy: Toy, tmp_path: Path
    ) -> None:
        answer_each(fake_client, "A", "B", "A")

        report = run(toy, SingleAgent(team_member), tmp_path)

        assert [result.answer for result in report.results] == ["A", "B", "A"]
        assert [result.score for result in report.results] == [1.0, 0.0, 1.0]
        assert report.metrics == {"accuracy": pytest.approx(2 / 3)}
        # The agent is asked each question in its own meeting, and answers once
        asked = [call for call in fake_client.completions.calls if "response_format" not in call]
        assert len(asked) == 3
        assert "Question 1? Options A or B." in asked[1]["messages"][-1]["content"]

    def test_the_answer_is_read_as_biomni_reads_it(
        self, fake_client: FakeClient, team_member: Agent, toy: Toy, tmp_path: Path
    ) -> None:
        answer_each(fake_client, "A", "B", "A")

        report = run(toy, SingleAgent(team_member), tmp_path)

        reading = fake_client.completions.parse_calls[1]
        assert reading["response_format"] is MultipleChoiceOutput
        assert reading["temperature"] == 0.0
        system, user = reading["messages"]
        assert system == {
            "role": "system",
            "content": (
                "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                "Review the entire history of messages provided. Here is the task output requirement: \n"
                "'Question 1? Options A or B.'.\n"
            ),
        }
        transcript = json.loads((tmp_path / report.results[1].transcript).read_text())
        assert user == {"role": "user", "content": transcript_text(transcript)}
        assert user["content"].endswith(f"{team_member.title}: [ANSWER]B[/ANSWER]")

    def test_each_question_is_saved_with_what_it_cost(
        self, fake_client: FakeClient, team_member: Agent, toy: Toy, tmp_path: Path
    ) -> None:
        answer_each(fake_client, "A", "B", "A")

        report = run(toy, SingleAgent(team_member), tmp_path)

        saved = json.loads((tmp_path / "results/quiz/1.json").read_text())
        assert saved == report.results[1].to_dict()
        assert saved["answer"] == "B"
        assert saved["correct_answer"] == "A"
        assert saved["output"] == {"choice": "B"}
        assert saved["error"] is None
        assert saved["usage"]["num_calls"] == 2
        assert saved["cost"] == pytest.approx(2 * CALL_COST)
        assert saved["transcript"] == "attempts/quiz/1/discussion.json"
        assert report.cost == pytest.approx(6 * CALL_COST)
        assert report.spent == pytest.approx(6 * CALL_COST)
        assert json.loads((tmp_path / "summary.json").read_text()) == {
            "benchmark": "toy",
            "metrics": {"accuracy": pytest.approx(2 / 3)},
            "questions": 3,
            "finished": 3,
            "failed": 0,
            "cost": pytest.approx(6 * CALL_COST),
            "spent": pytest.approx(6 * CALL_COST),
            "stopped": None,
        }

    def test_a_critic_and_options_are_passed_to_the_meeting(
        self, fake_client: FakeClient, team_member: Agent, team_lead: Agent, toy: Toy, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [choice("A")]

        run(Toy(questions(1)), SingleAgent(team_member, critic=team_lead, num_rounds=1, temperature=0.7), tmp_path)

        asked = [call for call in fake_client.completions.calls if "response_format" not in call]
        # Answer, criticism, revision
        assert len(asked) == 3
        assert {call["temperature"] for call in asked} == {0.7}
        assert team_lead.prompt in [call["messages"][0]["content"] for call in asked]

    def test_options_set_for_each_question_are_refused(self, team_member: Agent) -> None:
        with pytest.raises(TypeError, match="session"):
            SingleAgent(team_member, session=None)
        with pytest.raises(TypeError, match="output_schema"):
            SingleAgent(team_member, output_schema=MultipleChoiceOutput)

    def test_options_the_meeting_does_not_take_are_refused(self, team_member: Agent) -> None:
        with pytest.raises(TypeError, match="no option rounds"):
            SingleAgent(team_member, rounds=2)

    def test_describes_itself_completely(self, team_member: Agent) -> None:
        described = describe_solver(SingleAgent(team_member, num_rounds=2, tools=(PUBMED_TOOL,), code_actions="tags"))

        assert described == {
            "type": "single_agent",
            "agent": {
                "title": team_member.title,
                "name": team_member.name,
                "model": team_member.model,
                "expertise": team_member.expertise,
                "goal": team_member.goal,
                "role": team_member.role,
            },
            "critic": None,
            "num_rounds": 2,
            "options": {"tools": [PUBMED_TOOL.name], "code_actions": "tags"},
        }


class TestTeamMeeting:
    def test_the_team_discusses_each_question(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [choice("A")]

        report = run(Toy(questions(1)), TeamMeeting(team_lead, [team_member], num_rounds=1), tmp_path)

        assert report.results[0].score == 1.0
        transcript = json.loads((tmp_path / report.results[0].transcript).read_text())
        speakers = [turn["agent"] for turn in transcript if turn["agent"] != "User"]
        # Opening, a member's turn, and the lead's answer
        assert speakers == [team_lead.title, team_member.title, team_lead.title]
        assert "Question 0? Options A or B." in transcript[0]["message"]

    def test_describes_itself(self, team_lead: Agent, team_member: Agent) -> None:
        described = TeamMeeting(team_lead, (team_member,)).describe()

        assert described["type"] == "team_meeting"
        assert described["team_lead"]["title"] == team_lead.title
        assert [member["title"] for member in described["team_members"]] == [team_member.title]
        assert described["num_rounds"] == 1


class TestCustomSolvers:
    def test_a_transcript_is_read_and_kept(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")]

        # Long, so that a reader given only its end would be caught
        said = f"I worked on quiz/0 and chose A. {'Then I checked it again. ' * 400}"

        def solver(question: Question, context: SolverContext) -> str:
            return said

        report = run(Toy(questions(1)), solver, tmp_path)

        assert report.results[0].score == 1.0
        assert report.results[0].transcript == "attempts/quiz/0/transcript.txt"
        assert (tmp_path / "attempts/quiz/0/transcript.txt").read_text() == said
        assert fake_client.completions.parse_calls[0]["messages"][1]["content"] == said
        assert describe_solver(solver)["type"] == "custom"
        assert describe_solver(solver)["name"].endswith("solver")

    def test_an_answer_already_in_the_schema_is_not_read_again(self, fake_client: FakeClient, tmp_path: Path) -> None:
        def solver(question: Question, context: SolverContext) -> Attempt:
            return Attempt(transcript="Chose B.", output=MultipleChoiceOutput(choice="B"))

        report = run(Toy(questions(1)), solver, tmp_path)

        assert fake_client.completions.calls == []
        assert report.results[0].answer == "B"
        assert report.results[0].score == 0.0

    def test_an_answer_in_another_schema_fails(self, fake_client: FakeClient, tmp_path: Path) -> None:
        def solver(question: Question, context: SolverContext) -> Attempt:
            return Attempt(transcript="", output=OtherSchema())

        report = run(Toy(questions(1)), solver, tmp_path)

        assert report.results[0].error is not None
        assert "not a MultipleChoiceOutput" in report.results[0].error

    def test_a_solver_returning_something_else_fails(self, fake_client: FakeClient, tmp_path: Path) -> None:
        report = run(Toy(questions(1)), lambda question, context: 42, tmp_path)

        assert report.results[0].error == "TypeError: A solver returns a transcript or an Attempt, not int"

    def test_a_solvers_own_usage_counts(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")]

        def solver(question: Question, context: SolverContext) -> str:
            usage = MeetingUsage()
            context.count(usage)
            usage.add(TEST_MODEL, make_usage())
            return "A"

        report = run(Toy(questions(1)), solver, tmp_path)

        assert report.results[0].usage["num_calls"] == 2
        assert report.spent == pytest.approx(2 * CALL_COST)


class TestFailures:
    def test_a_failure_scores_zero_and_the_run_goes_on(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")]

        def solver(question: Question, context: SolverContext) -> str:
            if question.id == 0:
                raise RuntimeError("the agent fell over")
            return "A"

        report = run(Toy(questions(2)), solver, tmp_path)

        assert [result.error for result in report.results] == ["RuntimeError: the agent fell over", None]
        assert [result.score for result in report.results] == [0.0, 1.0]
        assert report.failed == 1
        assert report.metrics == {"accuracy": 0.5}

    def test_an_answer_that_cannot_be_read_is_a_failure_whose_cost_counts(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=None, refusal="No.")]

        report = run(Toy(questions(1)), lambda question, context: "A", tmp_path)

        result = report.results[0]
        assert result.error is not None and result.error.startswith("StructuredOutputError")
        assert result.answer is None and result.output is None
        assert result.usage["num_calls"] == 1
        assert result.cost == pytest.approx(CALL_COST)

    def test_too_many_failures_in_a_row_stop_the_run(self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        def solver(question: Question, context: SolverContext) -> str:
            raise RuntimeError("no key")

        report = run(Toy(questions(5)), solver, tmp_path, max_consecutive_failures=2)

        assert report.finished == 2
        assert report.stopped == "2 questions in a row failed, the last with RuntimeError: no key"
        assert "Run it again" in capsys.readouterr().out

    def test_a_success_resets_the_count_of_failures(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A"), choice("A")]

        def solver(question: Question, context: SolverContext) -> str:
            if question.id % 2 == 0:
                raise RuntimeError("flaky")
            return "A"

        report = run(Toy(questions(5)), solver, tmp_path, max_consecutive_failures=2)

        assert report.finished == 5
        assert report.stopped is None

    def test_an_interruption_leaves_what_was_finished(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")]

        def solver(question: Question, context: SolverContext) -> str:
            if question.id == 1:
                raise KeyboardInterrupt
            return "A"

        with pytest.raises(KeyboardInterrupt):
            run(Toy(questions(3)), solver, tmp_path)

        assert sorted(path.name for path in (tmp_path / "results/quiz").iterdir()) == ["0.json"]

    def test_an_unknown_cost_under_a_limit_stops_the_run(self, fake_client: FakeClient, tmp_path: Path) -> None:
        def solver(question: Question, context: SolverContext) -> str:
            usage = MeetingUsage()
            context.count(usage)
            usage.add(TEST_MODEL, None)
            context.remaining_budget()
            return "A"

        with pytest.raises(CostUnknownError):
            run(Toy(questions(2)), solver, tmp_path, max_cost=1.0)


class TestResuming:
    def test_finished_questions_are_not_asked_again(self, fake_client: FakeClient, team_member: Agent, toy: Toy, tmp_path: Path) -> None:
        answer_each(fake_client, "A", "B", "A")
        first = run(toy, SingleAgent(team_member), tmp_path)
        calls = len(fake_client.completions.calls)

        second = run(toy, SingleAgent(team_member), tmp_path)

        assert len(fake_client.completions.calls) == calls
        assert [result.to_dict() for result in second.results] == [result.to_dict() for result in first.results]
        assert second.metrics == first.metrics
        assert second.spent == 0.0

    def test_a_run_carries_on_where_it_stopped(self, fake_client: FakeClient, team_member: Agent, toy: Toy, tmp_path: Path) -> None:
        answer_each(fake_client, "A")
        run(toy, SingleAgent(team_member), tmp_path, questions=toy.questions[:1])
        answer_each(fake_client, "B", "A")

        report = run(toy, SingleAgent(team_member), tmp_path)

        assert [result.answer for result in report.results] == ["A", "B", "A"]
        assert "Question 0?" not in json.dumps(fake_client.completions.calls[-4:], default=str)

    def test_failures_are_asked_again_only_when_asked(self, fake_client: FakeClient, tmp_path: Path) -> None:
        attempts: list[str] = []

        def solver(question: Question, context: SolverContext) -> str:
            attempts.append(question.key)
            if len(attempts) == 1:
                raise RuntimeError("transient")
            return "A"

        fake_client.completions.parsed_responses = [choice("A")]
        run(Toy(questions(1)), solver, tmp_path)
        run(Toy(questions(1)), solver, tmp_path)
        assert attempts == ["quiz/0"]

        report = run(Toy(questions(1)), solver, tmp_path, retry_failed=True)

        assert attempts == ["quiz/0", "quiz/0"]
        assert report.results[0].error is None
        assert json.loads((tmp_path / "results/quiz/0.json").read_text())["score"] == 1.0

    def test_a_different_run_is_refused(self, fake_client: FakeClient, team_member: Agent, team_lead: Agent, toy: Toy, tmp_path: Path) -> None:
        answer_each(fake_client, "A")
        run(toy, SingleAgent(team_member), tmp_path, questions=toy.questions[:1])

        with pytest.raises(ValueError, match="different solver"):
            run(toy, SingleAgent(team_lead), tmp_path)
        with pytest.raises(ValueError, match="different benchmark"):
            run(Toy(questions(4)), SingleAgent(team_member), tmp_path)
        with pytest.raises(ValueError, match="different extractor"):
            run(toy, SingleAgent(team_member), tmp_path, extractor_temperature=0.5)

    def test_a_limit_is_not_part_of_what_the_run_is(self, fake_client: FakeClient, team_member: Agent, toy: Toy, tmp_path: Path) -> None:
        answer_each(fake_client, "A")
        run(toy, SingleAgent(team_member), tmp_path, questions=toy.questions[:1])
        answer_each(fake_client, "A", "A")

        report = run(toy, SingleAgent(team_member), tmp_path, max_cost=1.0, max_consecutive_failures=None)

        assert report.finished == 3


class TestBudget:
    def test_the_run_stops_when_its_budget_is_spent(self, fake_client: FakeClient, team_member: Agent, toy: Toy, tmp_path: Path) -> None:
        answer_each(fake_client, "A", "A", "A")

        # Enough for one question and its reading, and for the next to start
        report = run(toy, SingleAgent(team_member), tmp_path, max_cost=2.5 * CALL_COST)

        assert report.finished == 1
        assert report.stopped is not None
        assert "while answering quiz/1, which is left unfinished" in report.stopped
        assert not (tmp_path / "results/quiz/1.json").exists()
        # The unfinished question's spending is in what the run spent, not in a result
        assert report.spent == pytest.approx(3 * CALL_COST)
        assert report.cost == pytest.approx(2 * CALL_COST)

    def test_no_question_starts_once_the_budget_is_spent(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")]
        asked: list[str] = []

        def solver(question: Question, context: SolverContext) -> str:
            asked.append(question.key)
            return "A"

        report = run(Toy(questions(3)), solver, tmp_path, max_cost=CALL_COST)

        assert asked == ["quiz/0"]
        assert report.stopped is not None and report.stopped.startswith("The run spent")

    def test_the_reading_is_not_asked_for_past_the_budget(self, fake_client: FakeClient, tmp_path: Path) -> None:
        def solver(question: Question, context: SolverContext) -> str:
            usage = MeetingUsage()
            context.count(usage)
            usage.add(TEST_MODEL, make_usage())
            return "A"

        report = run(Toy(questions(2)), solver, tmp_path, max_cost_per_question=CALL_COST)

        assert fake_client.completions.calls == []
        assert [result.error for result in report.results] == [
            f"BudgetExceededError: The meeting has cost ${CALL_COST:.4f}, which reaches its limit of ${CALL_COST:.4f}, so it was stopped before the next request"
        ] * 2

    def test_a_question_over_its_own_budget_fails_and_the_run_goes_on(
        self, fake_client: FakeClient, team_member: Agent, team_lead: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses = [choice("A")] * 2

        # The critic's turn is the second request, which the question's limit leaves no room for
        report = run(
            Toy(questions(2)),
            SingleAgent(team_member, critic=team_lead, num_rounds=1),
            tmp_path,
            max_cost_per_question=CALL_COST,
        )

        assert [result.error is not None and result.error.startswith("BudgetExceededError") for result in report.results] == [True, True]
        # Two failures in a row are fewer than the three that stop a run by default
        assert report.stopped is None
        assert report.spent == pytest.approx(2 * CALL_COST)

    def test_a_meeting_is_given_what_the_question_has_left(self, fake_client: FakeClient, tmp_path: Path) -> None:
        budgets: list[float | None] = []
        fake_client.completions.parsed_responses = [choice("A")] * 2

        def solver(question: Question, context: SolverContext) -> str:
            budgets.append(context.remaining_budget())
            return "A"

        run(Toy(questions(2)), solver, tmp_path, max_cost=10 * CALL_COST, max_cost_per_question=4 * CALL_COST)
        assert budgets == [pytest.approx(4 * CALL_COST), pytest.approx(4 * CALL_COST)]

        budgets.clear()
        fake_client.completions.parsed_responses = [choice("A")] * 2
        run(Toy(questions(2)), solver, tmp_path / "tight", max_cost=1.5 * CALL_COST)
        assert budgets == [pytest.approx(1.5 * CALL_COST), pytest.approx(0.5 * CALL_COST)]

    def test_what_a_question_spends_comes_out_of_what_it_has_left(self, fake_client: FakeClient, tmp_path: Path) -> None:
        budgets: list[float | None] = []

        def solver(question: Question, context: SolverContext) -> str:
            usage = MeetingUsage()
            context.count(usage)
            for _ in range(3):
                usage.add(TEST_MODEL, make_usage())
                budgets.append(context.remaining_budget())
            return "A"

        run(Toy(questions(1)), solver, tmp_path, max_cost_per_question=2.5 * CALL_COST)

        # Never less than nothing, which hold_meeting would refuse as a limit
        assert budgets == [pytest.approx(1.5 * CALL_COST), pytest.approx(0.5 * CALL_COST), 0.0]

    def test_an_unpriced_extractor_cannot_be_held_to_a_limit(self, fake_client: FakeClient, tmp_path: Path) -> None:
        with pytest.raises(CostUnknownError, match="limit on spending"):
            run_benchmark(Toy(questions(1)), lambda question, context: "A", tmp_path, extractor="unpriced-model", max_cost=1.0)

        assert not (tmp_path / "run.json").exists()

    @pytest.mark.parametrize("limit", [-1.0, float("inf"), float("nan")])
    def test_a_limit_must_be_an_amount(self, limit: float, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="finite amount"):
            run(Toy(questions(1)), lambda question, context: "A", tmp_path, max_cost=limit)
        with pytest.raises(ValueError, match="finite amount"):
            run(Toy(questions(1)), lambda question, context: "A", tmp_path, max_cost_per_question=limit)


class TestSessions:
    def test_one_session_is_shared_by_every_question(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")] * 2
        session = LocalSession(directory=tmp_path / "work", warn=False)
        seen: list[Any] = []

        def solver(question: Question, context: SolverContext) -> str:
            seen.append(context.session)
            return "A"

        try:
            run(Toy(questions(2)), solver, tmp_path / "run", session=session)
        finally:
            session.close()

        assert seen == [session, session]

    def test_a_function_makes_a_session_per_question_and_each_is_closed(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")]
        made: list[Closable] = []

        def make() -> Any:
            made.append(Closable())
            return made[-1]

        seen: list[Any] = []

        def solver(question: Question, context: SolverContext) -> str:
            seen.append(context.session)
            if question.id == 1:
                raise RuntimeError("failed")
            return "A"

        run(Toy(questions(2)), solver, tmp_path, session=make)

        assert seen == made and len(made) == 2
        assert [session.closed for session in made] == [True, True]

    def test_anything_else_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="session must be"):
            run(Toy(questions(1)), lambda question, context: "A", tmp_path, session="docker")  # type: ignore[arg-type]

    def test_a_meeting_runs_its_code_in_the_shared_session(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        from conftest import tool_call_response

        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "seen = globals().get('seen', 0) + 1\nprint(seen)"}),
            text_response("A"),
            tool_call_response("run_code", {"code": "seen = globals().get('seen', 0) + 1\nprint(seen)"}),
            text_response("A"),
        ]
        fake_client.completions.parsed_responses = [choice("A")] * 2
        session = LocalSession(directory=tmp_path / "work", warn=False)
        try:
            run(Toy(questions(2)), SingleAgent(team_member, resources="none"), tmp_path / "run", session=session)
            outputs = [cell.output.strip() for cell in session.history]
        finally:
            session.close()

        # The second question sees what the first left, as Biomni's agent does
        assert outputs == ["1", "2"]


class Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestQuestions:
    def test_a_question_from_elsewhere_is_refused(self, toy: Toy, tmp_path: Path) -> None:
        stranger = Question(benchmark="toy", task="quiz", id=9, prompt="?", answer="A")

        with pytest.raises(ValueError, match="no question quiz/9"):
            run(toy, lambda question, context: "A", tmp_path, questions=[stranger])

    def test_a_question_asked_twice_is_refused(self, toy: Toy, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="twice"):
            run(toy, lambda question, context: "A", tmp_path, questions=[toy.questions[0]] * 2)

    def test_a_task_that_cannot_name_a_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="task's name"):
            run(Toy(questions(1, task="../escape")), lambda question, context: "A", tmp_path)

    def test_questions_are_reported_in_the_order_asked(self, fake_client: FakeClient, toy: Toy, tmp_path: Path) -> None:
        fake_client.completions.parsed_responses = [choice("A")] * 2

        report = run(toy, lambda question, context: "A", tmp_path, questions=[toy.questions[2], toy.questions[0]])

        assert [result.key for result in report.results] == ["quiz/2", "quiz/0"]
        assert report.questions == 2


def test_usage_is_added_up_model_by_model() -> None:
    first, second = MeetingUsage(), MeetingUsage()
    first.add("a", make_usage(100, 10, cached_tokens=5, reasoning_tokens=2))
    second.add("a", make_usage(300, 30))
    second.add("b", None)

    total = combine_usage([first, second])

    assert (total.per_model["a"].input_tokens, total.per_model["a"].output_tokens) == (400, 40)
    assert total.per_model["a"].cached_input_tokens == 5
    assert total.per_model["a"].reasoning_tokens == 2
    assert total.per_model["a"].max_input_tokens == 300
    assert total.per_model["a"].num_calls == 2
    assert total.per_model["b"].unreported_calls == 1


def test_a_meeting_reporting_its_usage_again_is_counted_once() -> None:
    tracker = UsageTracker()
    usage = MeetingUsage()
    usage.add(TEST_MODEL, make_usage())
    tracker.count(usage)
    usage.add(TEST_MODEL, make_usage())
    tracker.count(usage)

    assert tracker.total().num_calls == 2


def test_the_extraction_prompt_is_biomnis() -> None:
    assert EXTRACTION_PROMPT.format(task_intention="Pick {one}") == (
        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
        "Review the entire history of messages provided. "
        "Here is the task output requirement: \n'Pick {one}'.\n"
    )
