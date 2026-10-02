"""Runs a benchmark: asks each question of an agent or a team, reads the answer out of what was
said, as Biomni does, and scores it.

A solver answers one question: SingleAgent holds an individual meeting on it, TeamMeeting holds a
team meeting, and any function of a question and a SolverContext that returns a transcript can
stand in for either. Biomni's agents answer in prose and code, and a second request, to a model
told it is "evaluateGPT", reads the whole history of the attempt and returns the answer in the
schema the benchmark scores; run_benchmark asks for the answer the same way, with the same words.

Every finished question is saved as soon as it is scored, so a run that stops, for its budget, an
error, or a keyboard interrupt, carries on where it stopped when started again on the same
directory.
"""

import inspect
import json
import math
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tqdm import tqdm

from virtual_lab.agent import Agent
from virtual_lab.benchmarks import Benchmark, Question
from virtual_lab.completions import check_temperature
from virtual_lab.constants import DEFAULT_MAX_RETRIES, EXTRACTION_TEMPERATURE
from virtual_lab.llm import ModelSource, resolve_chat_models
from virtual_lab.provenance import describe_agent, describe_value
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.session import Session
from virtual_lab.structured import StructuredOutputError, request_structured_output
from virtual_lab.utils import (
    BudgetExceededError,
    CostUnknownError,
    MeetingUsage,
    UsageTracker,
    check_context_length,
    compute_token_cost,
    write_atomically,
)

# Biomni's words, from result_formatting in biomni/agent/a1.py at commit
# 400c1f366b96a35ca253e13c9b06c5076af41d65, Copyright the Biomni authors, used under the Apache
# License, Version 2.0. Biomni doubles the braces only because its prompt is a template.
EXTRACTION_PROMPT = (
    "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
    "Review the entire history of messages provided. "
    "Here is the task output requirement: \n"
    "'{task_intention}'.\n"
)

RUN_FILE = "run.json"
SUMMARY_FILE = "summary.json"
RESULTS_DIR = "results"
ATTEMPTS_DIR = "attempts"
TRANSCRIPT_FILE = "transcript.txt"

# Set by run_benchmark itself, so a solver given them would be overruled
CONTEXT_OPTIONS = frozenset(
    {
        "meeting_type",
        "agenda",
        "save_dir",
        "save_name",
        "team_lead",
        "team_members",
        "team_member",
        "critic",
        "num_rounds",
        "session",
        "chat_models",
        "client",
        "max_cost",
        "on_usage",
        "output_schema",
    }
)

SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Attempt:
    """What a solver did with a question.

    :param transcript: Everything said and run in answering it, which the answer is read from.
    :param output: The answer, already in the benchmark's schema, to be scored as it is rather
        than read from the transcript.
    :param transcript_path: Where the transcript was saved, if it was.
    """

    transcript: str
    output: BaseModel | None = None
    transcript_path: Path | None = None


@dataclass
class SolverContext:
    """What a solver is given to answer a question with.

    :param benchmark: The benchmark the question is from.
    :param save_dir: Where to save what the attempt produces. It is the question's own.
    :param session: The session to run code in, or None. Unless run_benchmark was given a function
        to make one per question, it is the same session for every question, as in Biomni, so it
        holds whatever earlier questions left in it.
    :param chat_models: The chat models to ask, as hold_meeting takes them.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param limit: The most the question may cost, in USD, or None.
    :param usage: What the question has used so far.
    :param run_usage: What the whole run has used so far, which the question's usage counts
        towards too.
    """

    benchmark: Benchmark
    save_dir: Path
    session: Session | None
    chat_models: ModelSource | None
    client: OpenAI | None
    limit: float | None
    usage: UsageTracker = field(default_factory=UsageTracker)
    run_usage: UsageTracker | None = None

    def count(self, usage: MeetingUsage) -> None:
        """Counts a meeting's usage towards the question. Pass it to hold_meeting as on_usage, or
        call it with a MeetingUsage a solver keeps itself, so that what it spends counts towards
        the budget and the record."""
        self.usage.count(usage)
        if self.run_usage is not None:
            self.run_usage.count(usage)

    def remaining_budget(self) -> float | None:
        """What the question may still spend, in USD, as hold_meeting takes it as max_cost."""
        if self.limit is None:
            return None

        return max(0.0, self.limit - self.usage.cost())


Solver = Callable[[Question, SolverContext], "Attempt | str"]


def transcript_text(discussion: Sequence[dict[str, str]]) -> str:
    """A meeting's transcript as one text, turn after turn, as Biomni gives its agent's log."""
    return "\n\n".join(f"{turn['agent']}: {turn['message']}" for turn in discussion)


def check_meeting_options(options: dict[str, Any]) -> None:
    """Refuses options hold_meeting does not take, or that each question sets itself."""
    if taken := sorted(set(options) & CONTEXT_OPTIONS):
        raise TypeError(f"{', '.join(taken)} cannot be given here: they are set for each question")

    accepted = inspect.signature(hold_meeting).parameters
    if unknown := sorted(set(options) - set(accepted)):
        raise TypeError(f"hold_meeting takes no option {', '.join(unknown)}")


class SingleAgent:
    """Answers each question in an individual meeting, with no critic unless num_rounds is given,
    as Biomni's single agent answers: the question is the agenda.

    :param agent: The agent who answers.
    :param critic: The critic, for num_rounds of criticism, defaulting to the Scientific Critic.
    :param num_rounds: How many rounds of criticism and revision follow the first answer.
    :param meeting_options: Anything else hold_meeting takes, such as code_actions, resources,
        tools, temperature, or max_completion_tokens.
    """

    def __init__(self, agent: Agent, critic: Agent | None = None, num_rounds: int = 0, **meeting_options: Any) -> None:
        check_meeting_options(meeting_options)
        self.agent = agent
        self.critic = critic
        self.num_rounds = num_rounds
        self.meeting_options = meeting_options

    def __call__(self, question: Question, context: SolverContext) -> Attempt:
        result = hold_meeting(
            meeting_type="individual",
            agenda=question.prompt,
            save_dir=context.save_dir,
            team_member=self.agent,
            critic=self.critic,
            num_rounds=self.num_rounds,
            session=context.session,
            chat_models=context.chat_models,
            client=context.client,
            max_cost=context.remaining_budget(),
            on_usage=context.count,
            **self.meeting_options,
        )

        return Attempt(transcript=transcript_text(result.discussion), transcript_path=result.transcript_path)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "single_agent",
            "agent": describe_agent(self.agent),
            "critic": describe_agent(self.critic) if self.critic is not None else None,
            "num_rounds": self.num_rounds,
            "options": describe_value(self.meeting_options),
        }


class TeamMeeting:
    """Answers each question in a team meeting: the question is the agenda, and the team lead's
    closing summary, with the discussion before it, is what the answer is read from.

    :param team_lead: The team lead, who opens and closes the meeting.
    :param team_members: The rest of the team.
    :param num_rounds: How many rounds the team discusses the question before the lead answers.
    :param meeting_options: Anything else hold_meeting takes.
    """

    def __init__(
        self, team_lead: Agent, team_members: Sequence[Agent], num_rounds: int = 1, **meeting_options: Any
    ) -> None:
        check_meeting_options(meeting_options)
        self.team_lead = team_lead
        self.team_members = tuple(team_members)
        self.num_rounds = num_rounds
        self.meeting_options = meeting_options

    def __call__(self, question: Question, context: SolverContext) -> Attempt:
        result = hold_meeting(
            meeting_type="team",
            agenda=question.prompt,
            save_dir=context.save_dir,
            team_lead=self.team_lead,
            team_members=self.team_members,
            num_rounds=self.num_rounds,
            session=context.session,
            chat_models=context.chat_models,
            client=context.client,
            max_cost=context.remaining_budget(),
            on_usage=context.count,
            **self.meeting_options,
        )

        return Attempt(transcript=transcript_text(result.discussion), transcript_path=result.transcript_path)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "team_meeting",
            "team_lead": describe_agent(self.team_lead),
            "team_members": [describe_agent(member) for member in self.team_members],
            "num_rounds": self.num_rounds,
            "options": describe_value(self.meeting_options),
        }


def describe_solver(solver: Solver) -> Any:
    """Describes a solver for the record of a run, by its describe method if it has one."""
    describe = getattr(solver, "describe", None)
    if callable(describe):
        return describe_value(describe())

    name = getattr(solver, "__qualname__", type(solver).__qualname__)

    return {"type": "custom", "name": f"{getattr(solver, '__module__', type(solver).__module__)}.{name}"}


def extraction_messages(question: Question, transcript: str) -> list[ChatCompletionMessageParam]:
    """The request that reads the answer out of an attempt: the question as the requirement, and
    the whole attempt as what to read."""
    return [
        {"role": "system", "content": EXTRACTION_PROMPT.format(task_intention=question.prompt)},
        {"role": "user", "content": transcript},
    ]


@dataclass
class QuestionResult:
    """How a question was answered and scored.

    :param task: The question's task.
    :param id: The question's id within its task.
    :param answer: The answer, as scored, or None if there was none.
    :param correct_answer: The benchmark's answer.
    :param score: The score, from 0 to 1.
    :param output: The answer as it was extracted, in the benchmark's schema.
    :param error: Why there was no answer, if there was none, as the type of error and its message.
    :param usage: The tokens used, extraction included, as MeetingUsage.to_dict gives them.
    :param cost: What answering cost in USD, or None if that cannot be worked out.
    :param elapsed: How long it took, in seconds.
    :param transcript: Where the transcript was saved, relative to the run's directory.
    """

    task: str
    id: int
    answer: str | None
    correct_answer: str
    score: float
    output: dict[str, Any] | None
    error: str | None
    usage: dict[str, Any]
    cost: float | None
    elapsed: float
    transcript: str | None

    @property
    def key(self) -> str:
        return f"{self.task}/{self.id}"

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestionResult":
        return cls(**{item.name: data[item.name] for item in fields(cls)})


@dataclass
class EvaluationReport:
    """What a run of a benchmark found.

    :param benchmark: The benchmark's name.
    :param results: Every question finished, in this run or an earlier one on the same directory,
        in the benchmark's order.
    :param metrics: The benchmark's measures over those questions, a question that failed
        counting as answered wrongly.
    :param questions: How many questions the run was asked to answer.
    :param spent: What this run spent in USD, unfinished questions included, or None if unknown.
    :param stopped: Why the run stopped before answering every question, or None if it did not.
    :param save_dir: Where the run was saved.
    """

    benchmark: str
    results: list[QuestionResult]
    metrics: dict[str, float | None]
    questions: int
    spent: float | None
    stopped: str | None
    save_dir: Path

    @property
    def finished(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return sum(result.failed for result in self.results)

    @property
    def cost(self) -> float | None:
        """What the finished questions cost in USD, or None if any one's cost is unknown."""
        costs = [result.cost for result in self.results]

        return None if any(cost is None for cost in costs) else sum(costs)  # type: ignore[misc]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "metrics": self.metrics,
            "questions": self.questions,
            "finished": self.finished,
            "failed": self.failed,
            "cost": self.cost,
            "spent": self.spent,
            "stopped": self.stopped,
        }


def save_json(path: Path, data: Any) -> None:
    write_atomically(path, json.dumps(data, indent=4).encode("utf-8"))


def relative_path(path: Path | None, directory: Path) -> str | None:
    """A path as it is recorded: relative to the run's directory when it is inside it."""
    if path is None:
        return None

    return path.relative_to(directory).as_posix() if path.is_relative_to(directory) else str(path)


def safe_name(name: str) -> str:
    """Refuses a task name that would put a question's files outside its directory."""
    if not SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"A task's name must be letters, digits, '_', '.', and '-' to name its files, not {name!r}")

    return name


def run_benchmark(
    benchmark: Benchmark,
    solver: Solver,
    save_dir: Path,
    extractor: str,
    questions: Iterable[Question] | None = None,
    session: Session | Callable[[], Session] | None = None,
    chat_models: ModelSource | None = None,
    client: OpenAI | None = None,
    max_cost: float | None = None,
    max_cost_per_question: float | None = None,
    extractor_temperature: float = EXTRACTION_TEMPERATURE,
    extractor_max_completion_tokens: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_failed: bool = False,
    max_consecutive_failures: int | None = 3,
) -> EvaluationReport:
    """Asks a solver each question of a benchmark, reads its answers, and scores them.

    Each question is answered by the solver, its answer is read out of the attempt by the
    extractor model, as Biomni reads it, into the benchmark's schema for the question, and it is
    scored by the benchmark's rule. Its result is saved under save_dir/results/ as soon as it is
    scored, and what the attempt saved goes under save_dir/attempts/. A question already saved is
    not asked again, so a run continues where an earlier one on the same directory stopped; the
    earlier one must have been the same benchmark, solver, and extractor.

    :param benchmark: The benchmark.
    :param solver: What answers each question: SingleAgent, TeamMeeting, or a function of a
        question and a SolverContext that returns its transcript, or an Attempt.
    :param save_dir: Where to save the run, and where an earlier run of it was saved.
    :param extractor: The model that reads the answers. Biomni asks the agent's own model.
    :param questions: The questions to ask, in order, defaulting to all of the benchmark's.
    :param session: A session for the solver to run code in. One session is shared by every
        question, as Biomni's agent keeps one, and is left running. Pass a function that makes a
        session instead for a new one per question, closed when the question is done.
    :param chat_models: The chat models to ask, as hold_meeting takes them, for the solver's
        agents and the extractor.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param max_cost: The most this run may spend, in USD. A question is not started once it has
        been spent, and one running when it is reached is stopped and left unfinished. As in
        hold_meeting it can be overrun by the cost of one request.
    :param max_cost_per_question: The most one question may spend, in USD. A question that
        reaches it fails, and the run goes on.
    :param extractor_temperature: The extractor's temperature.
    :param extractor_max_completion_tokens: The most tokens the extractor's answer may use.
    :param max_retries: Retries for a failed request to the extractor, if it is built here.
    :param retry_failed: Whether to ask again the questions an earlier run failed on.
    :param max_consecutive_failures: How many questions in a row may fail before the run stops,
        since that is more likely a mistake in setting it up, such as a missing key, than in the
        questions. None never stops.
    :raises ValueError: If save_dir holds a run of a different benchmark, solver, or extractor.
    :raises CostUnknownError: If a limit is given and what something cost cannot be worked out.
    :return: The results and the benchmark's measures over them.
    """
    for name, limit in (("max_cost", max_cost), ("max_cost_per_question", max_cost_per_question)):
        if limit is not None and not (math.isfinite(limit) and limit >= 0):
            raise ValueError(f"{name} must be a finite amount, zero or more, not {limit}")

    check_temperature(extractor_temperature)

    if max_consecutive_failures is not None and max_consecutive_failures < 1:
        raise ValueError(f"max_consecutive_failures must be at least 1, not {max_consecutive_failures}")

    if isinstance(session, Session):
        shared_session, make_session = session, None
    elif session is None or callable(session):
        shared_session, make_session = None, session
    else:
        raise TypeError(f"session must be a Session or a function that makes one, not {type(session).__name__}")

    in_benchmark = set(benchmark.questions)
    asked = list(benchmark.questions if questions is None else questions)
    if strangers := [question.key for question in asked if question not in in_benchmark]:
        raise ValueError(f"{benchmark.name} has no question {', '.join(strangers[:5])}")
    if len(set(asked)) != len(asked):
        raise ValueError("A question is asked twice")
    for task in {question.task for question in asked}:
        safe_name(task)

    if max_cost is not None or max_cost_per_question is not None:
        try:
            compute_token_cost(extractor, 0, 0)
        except CostUnknownError as error:
            raise CostUnknownError(f"{error}, so a limit on spending cannot be enforced") from error

    # Built before anything is asked, so that an extractor that cannot be built fails the run
    # before anything is spent
    extractor_llm = resolve_chat_models([extractor], chat_models=chat_models, client=client, max_retries=max_retries)[
        extractor
    ]

    save_dir = Path(save_dir)
    run = json.loads(
        json.dumps(
            {
                "benchmark": benchmark.describe(),
                "solver": describe_solver(solver),
                "extractor": {
                    "model": extractor,
                    "temperature": extractor_temperature,
                    "max_completion_tokens": extractor_max_completion_tokens,
                },
            }
        )
    )
    run_path = save_dir / RUN_FILE
    if run_path.is_file():
        earlier = json.loads(run_path.read_text(encoding="utf-8"))
        if earlier != run:
            different = ", ".join(part for part in run if earlier.get(part) != run[part])
            raise ValueError(
                f"{save_dir} holds a run with a different {different}, whose results cannot be "
                f"counted with this one's. Save this run somewhere else."
            )
    else:
        save_json(run_path, run)

    def result_path(question: Question) -> Path:
        return save_dir / RESULTS_DIR / question.task / f"{question.id}.json"

    def load(question: Question) -> QuestionResult | None:
        path = result_path(question)
        if not path.is_file():
            return None

        result = QuestionResult.from_dict(json.loads(path.read_text(encoding="utf-8")))

        return None if result.failed and retry_failed else result

    run_usage = UsageTracker()

    def run_spent() -> float | None:
        try:
            return run_usage.cost()
        except CostUnknownError:
            if max_cost is not None:
                raise
            return None

    def attempt(question: Question) -> QuestionResult:
        """Answers, reads, and scores one question. A failure is part of the result, except one
        that stops the run."""
        limits = [limit for limit in (max_cost_per_question,) if limit is not None]
        if max_cost is not None:
            limits.append(max(0.0, max_cost - run_usage.cost()))
        attempt_dir = save_dir / ATTEMPTS_DIR / question.task / str(question.id)
        context = SolverContext(
            benchmark=benchmark,
            save_dir=attempt_dir,
            session=shared_session,
            chat_models=chat_models,
            client=client,
            limit=min(limits) if limits else None,
            run_usage=run_usage,
        )

        start = time.time()
        output: BaseModel | None = None
        answer: str | None = None
        score = 0.0
        error: str | None = None
        transcript: Path | None = None
        own_session = None
        try:
            if make_session is not None:
                own_session = context.session = make_session()

            attempted = solver(question, context)
            if isinstance(attempted, str):
                attempted = Attempt(transcript=attempted)
            if not isinstance(attempted, Attempt):
                raise TypeError(f"A solver returns a transcript or an Attempt, not {type(attempted).__name__}")

            transcript = attempted.transcript_path
            if transcript is None:
                # What the answer is read from is kept, so that a score can be checked against it
                transcript = attempt_dir / TRANSCRIPT_FILE
                write_atomically(transcript, attempted.transcript.encode("utf-8"))

            schema = benchmark.output_schema(question)
            if attempted.output is not None:
                if not isinstance(attempted.output, schema):
                    raise TypeError(f"The solver's answer is a {type(attempted.output).__name__}, not a {schema.__name__}")
                output = attempted.output
            else:
                output = extract(question, attempted.transcript, schema, context)

            answer = benchmark.answer_from(question, output)
            score = benchmark.score(question, answer)
        except BudgetExceededError as failure:
            if max_cost is not None and run_usage.cost() >= max_cost:
                raise
            error = f"BudgetExceededError: {failure}"
        except CostUnknownError:
            raise
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        finally:
            if own_session is not None:
                own_session.close()

        if error is not None:
            answer = None

        total = context.usage.total()
        try:
            cost: float | None = total.compute_cost()
        except CostUnknownError:
            cost = None

        result = QuestionResult(
            task=question.task,
            id=question.id,
            answer=answer,
            correct_answer=question.answer,
            score=score,
            output=output.model_dump(mode="json") if output is not None and error is None else None,
            error=error,
            usage=total.to_dict(),
            cost=cost,
            elapsed=round(time.time() - start, 3),
            transcript=relative_path(transcript, save_dir),
        )
        save_json(result_path(question), result.to_dict())

        return result

    def extract(question: Question, transcript: str, schema: type[BaseModel], context: SolverContext) -> BaseModel:
        """Reads the answer out of an attempt, in one request, as Biomni's result_formatting does."""
        messages = extraction_messages(question, transcript)
        check_context_length(messages=messages, model=extractor)
        if context.limit is not None and (spent := context.usage.cost()) >= context.limit:
            raise BudgetExceededError(spent=spent, limit=context.limit)

        usage = MeetingUsage()
        context.count(usage)
        try:
            output, reply = request_structured_output(
                llm=extractor_llm,
                model=extractor,
                messages=messages,
                schema=schema,
                temperature=extractor_temperature,
                max_completion_tokens=extractor_max_completion_tokens,
            )
        except StructuredOutputError as error:
            usage.add(model=extractor, usage=error.usage)
            raise
        usage.add(model=extractor, usage=reply.usage)

        return output

    results: dict[Question, QuestionResult] = {}
    stopped: str | None = None
    failures_in_a_row = 0
    progress = tqdm(asked, desc=benchmark.name)
    for question in progress:
        if (saved := load(question)) is not None:
            results[question] = saved
            continue

        if max_cost is not None and run_usage.cost() >= max_cost:
            stopped = f"The run spent ${run_usage.cost():.4f}, which reaches its max_cost of ${max_cost:.4f}"
            break

        try:
            result = attempt(question)
        except BudgetExceededError:
            stopped = (
                f"The run spent ${run_usage.cost():.4f}, which reaches its max_cost of ${max_cost:.4f}, "
                f"while answering {question.key}, which is left unfinished"
            )
            break

        results[question] = result
        progress.set_postfix(score=f"{sum(r.score for r in results.values()) / len(results):.3f}")

        failures_in_a_row = failures_in_a_row + 1 if result.failed else 0
        if max_consecutive_failures is not None and failures_in_a_row >= max_consecutive_failures:
            stopped = f"{failures_in_a_row} questions in a row failed, the last with {result.error}"
            break
    progress.close()

    finished = [question for question in asked if question in results]
    report = EvaluationReport(
        benchmark=benchmark.name,
        results=[results[question] for question in finished],
        metrics=benchmark.metrics([(question, results[question].answer) for question in finished]),
        questions=len(asked),
        spent=run_spent(),
        stopped=stopped,
        save_dir=save_dir,
    )
    save_json(save_dir / SUMMARY_FILE, report.to_dict())

    if stopped is not None:
        print(f"Warning: {stopped}. Run it again on {save_dir} to carry on.")
    print(f"{benchmark.name}: {report.finished} of {report.questions} questions finished, {report.failed} failed")
    for measure, value in report.metrics.items():
        print(f"{measure}: {'n/a' if value is None else f'{value:.4f}'}")

    return report
