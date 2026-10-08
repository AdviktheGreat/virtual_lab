"""A research project: one goal, one budget, and a ledger of every step taken towards it.

A meeting can be limited in what it spends, but a project is many meetings, and code repaired
between them, so its limit has to be kept across all of them: across meetings held one after
another, meetings held at the same time, and runs of the same project in different processes.
Project keeps that limit, and a ledger, project.json, of every step it took, what each cost, and
where it was saved.

The ledger is also what lets a project be resumed. A step that finished is never paid for twice:
asked for again under the same name and with the same inputs, it is read back from disk instead.
A script that stopped partway, for its budget, an error, or a keyboard interrupt, therefore
carries on where it stopped when run again on the same directory, and a step asked for with
different inputs under a name already used is refused rather than silently replaced.
"""

import hashlib
import inspect
import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.artifacts import CodeArtifacts
from virtual_lab.constants import (
    ARTIFACT_DIR_NAME,
    EXECUTION_DIR_NAME,
    PARTIAL_MEETING_DIR_NAME,
    PROJECT_FILE_NAME,
    PROJECT_MEETINGS_DIR_NAME,
)
from virtual_lab.events import MeetingEvent
from virtual_lab.execution import Executor
from virtual_lab.llm import ModelSource, resolve_chat_models
from virtual_lab.provenance import MeetingRecord, describe_value, utc_timestamp
from virtual_lab.repair import RepairOutcome, describe_executor, run_with_repair, save_execution_record
from virtual_lab.resources import Resources
from virtual_lab.run_meeting import MeetingResult, hold_meeting
from virtual_lab.session import Session
from virtual_lab.tools import Tool
from virtual_lab.utils import BudgetExceededError, CostUnknownError, MeetingUsage, write_atomically

SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")

# Set by the project for every meeting it holds
PROJECT_OPTIONS = frozenset({"save_dir", "save_name", "session", "chat_models", "client"})

# What run_with_repair takes that a project leaves to the caller
REPAIR_OPTIONS = frozenset(
    {
        "model",
        "temperature",
        "max_attempts",
        "timeout",
        "max_retries",
        "chat_model",
        "max_cost",
        "on_usage",
        "before_request",
        "max_completion_tokens",
    }
)

# Options that decide what a step costs, how it is sent, or who is told of it as it goes, not what
# it is asked, so changing one does not make a finished step a different step. A person steering
# a meeting can change what it produces, but what they said is in its transcript
NOT_INPUTS = frozenset(
    {"max_cost", "on_usage", "before_request", "max_retries", "chat_model", "on_event", "stream", "steer"}
)


class ProjectBudgetExceededError(BudgetExceededError):
    """Raised before a step, or a request within one, that a project's limit leaves no room for."""

    def __init__(self, spent: float, limit: float) -> None:
        super().__init__(spent=spent, limit=limit, what="project")


class ProjectStateError(ValueError):
    """Raised when a project's directory holds a different project, or a step that differs."""


@dataclass
class ProjectStep:
    """One step a project took, as its ledger records it.

    :param name: The step's name, unique in the project, which its files are saved under.
    :param kind: "meeting" or "repair".
    :param status: "running", "completed", "failed", or "interrupted" if the process running it
        stopped before it could say how it ended.
    :param inputs: A SHA-256 of each input the step was given, by name, so that a step asked for
        again can be matched to this one, and the inputs that differ named if it does not match.
    :param usage: The tokens it used, as MeetingUsage.to_dict gives them, kept up to date after
        every response while it runs.
    :param error: What stopped it, as its type and message, if it did not complete.
    :param files: Where it was saved, relative to the project's directory, by what each file is.
    :param files_sha256: A SHA-256 of each file the step is read back from.
    """

    name: str
    kind: str
    status: str
    inputs: dict[str, str]
    started_at: str
    ended_at: str | None = None
    elapsed_seconds: float | None = None
    usage: dict[str, Any] = field(default_factory=lambda: MeetingUsage().to_dict())
    error: dict[str, str] | None = None
    files: dict[str, str] = field(default_factory=dict)
    files_sha256: dict[str, str] = field(default_factory=dict)

    @property
    def cost(self) -> float | None:
        """What the step cost in USD, or None if that cannot be worked out."""
        return self.usage.get("cost")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectStep":
        return cls(**{item.name: data[item.name] for item in fields(cls) if item.name in data})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_input(name: str, value: Any) -> Any:
    """Describes an input to a step completely enough that a change to it shows."""
    if isinstance(value, type) and issubclass(value, BaseModel):
        # Its name alone would not show a field added to it
        return {"schema": f"{value.__module__}.{value.__qualname__}", "fields": value.model_json_schema()}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if name == "executor":
        return describe_executor(value)
    # A record names tools and resources, but what the agents are told of them is what counts
    if isinstance(value, Tool):
        # What a partial binds, such as the directory a tool works in, is not what the agents
        # are told, and can differ from one run to the next
        function = value.function
        while isinstance(function, partial):
            function = function.func
        described = {
            "name": value.name,
            "description": value.description,
            "parameters": describe_value(value.parameters),
            "function": f"{getattr(function, '__module__', None)}.{getattr(function, '__qualname__', type(function).__qualname__)}",
        }
        # Only where there are any, so that a tool without them is described as it was before
        # tools could have them
        if value.instructions is not None:
            described["instructions"] = value.instructions
        return described
    if isinstance(value, Resources):
        return describe_value(asdict(value))
    # Which session it is, and where it runs, can differ from one run to the next; the tools its
    # code can call, and the data and software added to it, are part of what the agents are told.
    # Data is described by name, not by where it is on this machine. A session without any of
    # them is described as it was before sessions had them, so that a project's earlier steps
    # still match.
    if isinstance(value, Session) and (value.tools or value.data or value.software):
        described: dict[str, Any] = {
            "session": f"{type(value).__module__}.{type(value).__qualname__}",
            "tools": [describe_input(name, tool) for tool in value.tools],
        }
        if value.data:
            described["data"] = [{"name": item.name, "description": item.description} for item in value.data]
        if value.software:
            described["software"] = dict(value.software)
        return described
    if isinstance(value, list | tuple):
        return [describe_input(name, item) for item in value]

    return describe_value(value)


def fingerprint_inputs(function: Callable[..., Any], inputs: dict[str, Any]) -> dict[str, str]:
    """A SHA-256 of each input, by name, of what a step is asked to do.

    An input given at the function's default is left out, as if it had not been given, so that
    a step asked for with a default spelled out is the same step, and so is one asked for after
    an upgrade adds a parameter.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    return {
        name: sha256_text(json.dumps(describe_input(name, value), sort_keys=True))
        for name, value in sorted(inputs.items())
        if name not in NOT_INPUTS and not (name in defaults and describe_input(name, value) == describe_input(name, defaults[name]))
    }


def check_name(name: str) -> str:
    """Refuses a step name that would put its files somewhere other than its own."""
    if not SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"A step's name must be letters, digits, '_', '.', and '-' to name its files, not {name!r}")

    return name


def check_limit(name: str, limit: float | None) -> None:
    if limit is not None and not (math.isfinite(limit) and limit >= 0):
        raise ValueError(f"{name} must be a finite amount, zero or more, not {limit}")


def check_meeting_options(options: dict[str, Any]) -> None:
    """Refuses options hold_meeting does not take, or that the project sets itself."""
    if taken := sorted(set(options) & (PROJECT_OPTIONS | {"meeting_type", "agenda"})):
        raise TypeError(f"{', '.join(taken)} cannot be given here: the project sets them")

    accepted = inspect.signature(hold_meeting).parameters
    if unknown := sorted(set(options) - set(accepted)):
        raise TypeError(f"hold_meeting takes no option {', '.join(unknown)}")


def tightest(*limits: float | None) -> float | None:
    """The lowest of the limits given, or None if none is."""
    given = [limit for limit in limits if limit is not None]

    return min(given) if given else None


class Project:
    """Holds the meetings and repairs of one research project, within one budget, and resumes it.

    Every step is saved under project_dir/meetings/ by its name and recorded in project_dir's
    project.json as soon as it starts, with what it has cost kept up to date after every
    response, so that what was spent is known even if the process is killed. A step that
    finished is read back from disk when asked for again with the same name and inputs, and
    costs nothing; one that failed or was interrupted is held again, and what it cost the first
    time still counts towards the budget.

    Steps can be run from several threads at once, each under a name of its own; the budget is
    checked before every request any of them sends. Only one process should use a directory at
    a time.

    :param save_dir: The project's directory, where an earlier run of it was saved, if one was.
    :param goal: What the project is for. A directory holding a project with another goal is
        refused, so two projects are never mixed in one ledger.
    :param max_cost: The most the project may spend, in USD, over every run of it on this
        directory. Each request is checked against it, so it can be overrun by the cost of one
        request per step running at the time; max_completion_tokens bounds that. It may be
        raised for a later run, to let a project that stopped for its budget carry on.
    :param session: A session for every meeting's agents to run code in, as hold_meeting takes it.
        A session does not survive the process, so a resumed project's meetings that are read
        back from disk leave nothing in it; the files they wrote in its directory are still there.
    :param chat_models: The chat models to ask, as hold_meeting takes them, for meetings and
        repairs alike.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param meeting_options: Anything else hold_meeting takes, as the default for every meeting,
        such as temperature, code_actions, resources, or max_completion_tokens.
    :raises ProjectStateError: If save_dir holds a project with a different goal.
    """

    def __init__(
        self,
        save_dir: Path,
        goal: str,
        max_cost: float | None = None,
        session: Session | None = None,
        chat_models: ModelSource | None = None,
        client: OpenAI | None = None,
        **meeting_options: Any,
    ) -> None:
        check_limit("max_cost", max_cost)
        check_meeting_options(meeting_options)
        if not goal.strip():
            raise ValueError("A project needs a goal")

        self.save_dir = Path(save_dir)
        self.goal = goal
        self.max_cost = max_cost
        self.session = session
        self.chat_models = chat_models
        self.client = client
        self.meeting_options = meeting_options
        self._lock = threading.RLock()
        self._running: dict[str, ProjectStep] = {}
        self._started: dict[str, float] = {}
        self._counts: dict[str, int] = {}

        path = self.save_dir / PROJECT_FILE_NAME
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["goal"] != goal:
                raise ProjectStateError(
                    f"{self.save_dir} holds a project with another goal: {saved['goal']!r}. Save this one somewhere else."
                )
            self.created_at: str = saved["created_at"]
            self._steps = [ProjectStep.from_dict(step) for step in saved["steps"]]
        else:
            self.created_at = utc_timestamp()
            self._steps = []

        # Nothing is running yet in this process, so a step the ledger says is running was
        # running in one that stopped without saying how it ended
        interrupted = [step for step in self._steps if step.status == "running"]
        for step in interrupted:
            step.status = "interrupted"
            step.error = {
                "type": "Interrupted",
                "message": "The process running this step stopped before it ended. What it cost is what had "
                "been recorded by then, which leaves out any request it was waiting on.",
            }
        if interrupted:
            print(
                f"Warning: {len(interrupted)} step{'s' if len(interrupted) > 1 else ''} of this project "
                f"({', '.join(step.name for step in interrupted)}) did not end before the last run stopped. "
                f"What they cost counts towards the budget, and they are held again when asked for."
            )

        self._save()

    @property
    def meetings_dir(self) -> Path:
        """Where the project's meetings and repairs are saved."""
        return self.save_dir / PROJECT_MEETINGS_DIR_NAME

    @property
    def steps(self) -> tuple[ProjectStep, ...]:
        """Every step the project has taken, in the order they started, copied."""
        with self._lock:
            return tuple(replace(step, usage=dict(step.usage)) for step in self._steps)

    @property
    def spent(self) -> float | None:
        """What the project has spent in USD over every run of it, or None if that cannot be
        worked out, because a model is unpriced or a response did not report its usage."""
        with self._lock:
            costs = [step.cost for step in self._steps]

        return None if any(cost is None for cost in costs) else sum(costs)  # type: ignore[misc]

    @property
    def remaining(self) -> float | None:
        """What the project may still spend in USD, or None if it has no limit.

        :raises CostUnknownError: If it has a limit and what it has spent cannot be worked out.
        """
        if self.max_cost is None:
            return None

        return max(0.0, self.max_cost - self.checked_spent())

    def checked_spent(self) -> float:
        """What the project has spent, for a limit, which needs to know.

        :raises CostUnknownError: If that cannot be worked out.
        """
        spent = self.spent
        if spent is None:
            unknown = ", ".join(step.name for step in self.steps if step.cost is None)
            raise CostUnknownError(
                f"What the project has spent cannot be worked out, since what {unknown} cost is not known, so "
                f"its max_cost cannot be enforced"
            )

        return spent

    def check_budget(self) -> None:
        """Stops a step, or a request within one, that the project has no money left for.

        :raises ProjectBudgetExceededError: If the project has spent its max_cost.
        :raises CostUnknownError: If it has a limit and what it has spent cannot be worked out.
        """
        if self.max_cost is not None and (spent := self.checked_spent()) >= self.max_cost:
            raise ProjectBudgetExceededError(spent=spent, limit=self.max_cost)

    def meeting(
        self,
        meeting_type: Literal["team", "individual"],
        agenda: str,
        name: str | None = None,
        **options: Any,
    ) -> MeetingResult:
        """Holds a meeting for the project, or reads it back if the project already held it.

        :param meeting_type: The type of meeting.
        :param agenda: The agenda for the meeting.
        :param name: What to save the meeting under, unique in the project, defaulting to
            meeting_001, meeting_002, and so on, in the order meetings are asked for. Give each
            meeting held from another thread a name of its own, since that order is not fixed.
        :param options: Anything else hold_meeting takes, over the project's meeting_options.
            A max_cost here limits the meeting as well as the project's limit. An on_event is
            told of a meeting read back from disk by one "read_back" event.
        :raises ProjectBudgetExceededError: If the project has spent its max_cost, before the
            meeting or during it.
        :raises ProjectStateError: If the project already held a meeting under this name with
            other inputs, or its files have changed since.
        :return: Everything the meeting produced, as hold_meeting returns it.
        """
        check_meeting_options(options)
        arguments = {**self.meeting_options, **options}
        own_limit = arguments.pop("max_cost", None)
        check_limit("max_cost", own_limit)
        on_usage = arguments.pop("on_usage", None)
        before_request = arguments.pop("before_request", None)
        arguments = {"meeting_type": meeting_type, "agenda": agenda, **arguments, "session": self.session}

        name = self._name_step("meeting", name)
        done = self._begin(name, "meeting", fingerprint_inputs(hold_meeting, arguments))
        if done is not None:
            result = self._read_meeting(done, arguments.get("output_schema"))
            if (on_event := arguments.get("on_event")) is not None:
                on_event(
                    MeetingEvent(
                        kind="read_back",
                        meeting=name,
                        text=result.summary,
                        data={
                            "transcript_path": str(result.transcript_path),
                            "record_path": str(result.record_path),
                            "output_path": str(result.output_path) if result.output_path is not None else None,
                            "usage": result.usage.to_dict(),
                        },
                    )
                )
            return result

        project_limited = False
        try:
            limit, project_limited = self._step_limit(own_limit)
            result = hold_meeting(
                save_dir=self.meetings_dir,
                save_name=name,
                chat_models=self.chat_models,
                client=self.client,
                max_cost=limit,
                on_usage=self._counter(name, on_usage),
                before_request=self._guard(before_request),
                **arguments,
            )
        except BaseException as error:
            self._fail(name, error, {"partial": self.meetings_dir / PARTIAL_MEETING_DIR_NAME / f"{name}.json"})
            self._raise_for_project(error, project_limited)
            raise

        files = {
            "transcript": result.transcript_path,
            "record": result.record_path,
            **({"output": result.output_path} if result.output_path is not None else {}),
        }
        self._complete(name, result.usage, files)

        return result

    def repair(
        self,
        artifacts: CodeArtifacts,
        author: Agent,
        executor: Executor,
        name: str | None = None,
        **options: Any,
    ) -> RepairOutcome:
        """Runs code and has its author repair it, as run_with_repair does, for the project, or
        reads the outcome back if the project already did.

        The code is written under meetings/artifacts/<name>/ and the record of what happened to
        meetings/executions/<name>.json.

        :param artifacts: The files to run.
        :param author: The agent who wrote them, who is asked to fix them.
        :param executor: What to run them with.
        :param name: What to save the files and record under, unique in the project, defaulting
            to repair_001, repair_002, and so on.
        :param options: Anything else run_with_repair takes: model, temperature, max_attempts,
            timeout, max_retries, chat_model, max_cost, on_usage, before_request, or
            max_completion_tokens. The project's chat_models and client are used for the
            repairs unless chat_model is given.
        :raises ProjectBudgetExceededError: If the project has spent its max_cost, before the
            code is run or before a repair request.
        :raises ProjectStateError: If the project already ran code under this name with other
            inputs, or its record or code has changed since.
        :return: What happened.
        """
        if unknown := sorted(set(options) - REPAIR_OPTIONS):
            raise TypeError(f"run_with_repair takes no option {', '.join(unknown)} here")

        own_limit = options.pop("max_cost", None)
        check_limit("max_cost", own_limit)
        on_usage = options.pop("on_usage", None)
        before_request = options.pop("before_request", None)
        arguments = {"artifacts": artifacts, "author": author, "executor": executor, **options}

        name = self._name_step("repair", name)
        done = self._begin(name, "repair", fingerprint_inputs(run_with_repair, arguments))
        if done is not None:
            return self._read_repair(done)

        project_limited = False
        try:
            limit, project_limited = self._step_limit(own_limit)
            if arguments.get("chat_model") is None:
                model = arguments.get("model") or author.model
                retries = {"max_retries": arguments["max_retries"]} if "max_retries" in arguments else {}
                arguments["chat_model"] = resolve_chat_models(
                    [model], chat_models=self.chat_models, client=self.client, **retries
                )[model]

            outcome = run_with_repair(
                save_dir=self.meetings_dir,
                save_name=name,
                max_cost=limit,
                on_usage=self._counter(name, on_usage),
                before_request=self._guard(before_request),
                **arguments,
            )
            record_path = save_execution_record(
                save_dir=self.meetings_dir, save_name=name, outcome=outcome, author=author, executor=executor
            )
        except BaseException as error:
            self._fail(name, error, {"partial": self.meetings_dir / PARTIAL_MEETING_DIR_NAME / EXECUTION_DIR_NAME / f"{name}.json"})
            self._raise_for_project(error, project_limited)
            raise

        # The code is hashed too, since a later step may run it from these paths
        code_dir = self.meetings_dir / ARTIFACT_DIR_NAME / name
        code = {f"code:{path.relative_to(code_dir).as_posix()}": path for path in outcome.paths}
        self._complete(name, outcome.usage, {"record": record_path, **code})

        return outcome

    def _name_step(self, kind: str, name: str | None) -> str:
        """The name a step is saved under: the one given, or the next in the order asked for."""
        if name is not None:
            return check_name(name)

        with self._lock:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            return f"{kind}_{self._counts[kind]:03d}"

    def _begin(self, name: str, kind: str, inputs: dict[str, str]) -> ProjectStep | None:
        """Starts a step, or finds it finished.

        :return: The step, if it finished in an earlier run and can be read back, or None once
            it is recorded as running.
        """
        with self._lock:
            if name in self._running:
                raise ValueError(f'A step named "{name}" is already running in this project')

            finished = [step for step in self._steps if step.name == name and step.status == "completed"]
            if finished:
                step = finished[-1]
                if step.kind != kind:
                    raise ProjectStateError(f'The project has a {step.kind} named "{name}" already. Use another name.')
                if step.inputs != inputs:
                    different = sorted(key for key in step.inputs.keys() | inputs.keys() if step.inputs.get(key) != inputs.get(key))
                    raise ProjectStateError(
                        f'The project held "{name}" with a different {", ".join(different)}, and holding it again '
                        f"would replace what later steps may have been built on. Give it another name."
                    )
                self._check_files(step)
                return step

            self.check_budget()
            step = ProjectStep(name=name, kind=kind, status="running", inputs=inputs, started_at=utc_timestamp())
            self._steps.append(step)
            self._running[name] = step
            self._started[name] = time.monotonic()
            self._save()

        return None

    def _counter(
        self, name: str, on_usage: Callable[[MeetingUsage], None] | None
    ) -> Callable[[MeetingUsage], None]:
        """Keeps a running step's cost in the ledger, after every response, so it is never lost."""

        def count(usage: MeetingUsage) -> None:
            with self._lock:
                self._running[name].usage = usage.to_dict()
                self._save()
            if on_usage is not None:
                on_usage(usage)

        return count

    def _guard(self, before_request: Callable[[], None] | None) -> Callable[[], None]:
        """Checks the project's budget before every request a step sends."""

        def check() -> None:
            self.check_budget()
            if before_request is not None:
                before_request()

        return check

    def _complete(self, name: str, usage: MeetingUsage, files: dict[str, Path]) -> None:
        with self._lock:
            step = self._running.pop(name)
            step.status = "completed"
            step.usage = usage.to_dict()
            step.files = {role: self._relative(path) for role, path in files.items()}
            step.files_sha256 = {role: sha256_file(path) for role, path in files.items()}
            self._finish(step)

    def _fail(self, name: str, error: BaseException, files: dict[str, Path]) -> None:
        with self._lock:
            step = self._running.pop(name)
            step.status = "failed"
            step.error = {"type": type(error).__name__, "message": str(error)}
            # A step that failed before it began has left nothing to point to
            step.files = {role: self._relative(path) for role, path in files.items() if path.is_file()}
            self._finish(step)

    def _finish(self, step: ProjectStep) -> None:
        step.ended_at = utc_timestamp()
        step.elapsed_seconds = round(time.monotonic() - self._started.pop(step.name), 3)
        self._save()

    def _step_limit(self, own_limit: float | None) -> tuple[float | None, bool]:
        """The limit a step starts with: the tighter of its own and what the project has left.

        :return: The limit, and whether it is the project's.
        """
        remaining = self.remaining
        project_limited = remaining is not None and (own_limit is None or remaining <= own_limit)

        return tightest(own_limit, remaining), project_limited

    def _raise_for_project(self, error: BaseException, project_limited: bool) -> None:
        """Raises a step's budget error as the project's, if the project's limit is what it reached.

        A step is given the project's remaining budget as its own limit, so the step's error is
        the one that fires first when one step runs at a time. Whether that limit was the
        project's is known from when the step started, since adding what the step cost back to
        what was spent before it can come out a rounding error short of the project's limit.
        """
        if isinstance(error, ProjectBudgetExceededError) or not isinstance(error, BudgetExceededError):
            return

        if self.max_cost is None or (spent := self.spent) is None:
            return

        if project_limited or spent >= self.max_cost:
            raise ProjectBudgetExceededError(spent=spent, limit=self.max_cost) from error

    def _check_files(self, step: ProjectStep) -> None:
        """Refuses to read back a step whose files are missing or have changed since."""
        for role, relative in step.files.items():
            path = self.save_dir / relative
            if not path.is_file():
                raise ProjectStateError(f'The {role} of "{step.name}" is missing from {path}')
            if sha256_file(path) != step.files_sha256.get(role):
                raise ProjectStateError(
                    f'The {role} of "{step.name}" at {path} has changed since it was saved, so it is not what '
                    f"later steps were built on. Restore it, or hold the step again under another name."
                )

    def _read_meeting(self, step: ProjectStep, output_schema: type[BaseModel] | None) -> MeetingResult:
        """Reads back a meeting the project held in an earlier run."""
        transcript_path = self.save_dir / step.files["transcript"]
        record_path = self.save_dir / step.files["record"]
        output_path = self.save_dir / step.files["output"] if "output" in step.files else None

        discussion = tuple(json.loads(transcript_path.read_text(encoding="utf-8")))
        record = MeetingRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8")))
        output = (
            output_schema.model_validate_json(output_path.read_text(encoding="utf-8"))
            if output_schema is not None and output_path is not None
            else None
        )
        # The summary is the last thing an agent said, which a structured output follows
        responses = [turn.index for turn in record.turns if turn.kind == "response"]

        print(f'Read "{step.name}" back from {transcript_path}, since the project already held it')

        return MeetingResult(
            summary=discussion[responses[-1]]["message"] if responses else "",
            output=output,
            usage=MeetingUsage.from_dict(record.usage),
            record=record,
            discussion=discussion,
            transcript_path=transcript_path,
            record_path=record_path,
            output_path=output_path,
        )

    def _read_repair(self, step: ProjectStep) -> RepairOutcome:
        """Reads back the outcome of code the project ran in an earlier run."""
        record_path = self.save_dir / step.files["record"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        directory = self.meetings_dir / ARTIFACT_DIR_NAME / step.name
        paths = tuple(directory / file["filename"] for file in record["final_files"])

        print(f'Read "{step.name}" back from {record_path}, since the project already ran it')

        return RepairOutcome.from_dict(record, paths=paths)

    def _relative(self, path: Path) -> str:
        return Path(path).relative_to(self.save_dir).as_posix()

    def describe(self) -> dict[str, Any]:
        """The ledger, as project.json holds it."""
        with self._lock:
            return {
                "virtual_lab_version": __version__,
                "goal": self.goal,
                "created_at": self.created_at,
                "max_cost": self.max_cost,
                "spent": self.spent,
                "steps": [step.to_dict() for step in self._steps],
            }

    def _save(self) -> None:
        with self._lock:
            write_atomically(self.save_dir / PROJECT_FILE_NAME, json.dumps(self.describe(), indent=4).encode("utf-8"))

