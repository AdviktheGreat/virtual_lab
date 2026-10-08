"""Running a meeting or a project for the web interface: in a thread of its own, followed as it
goes, and steered by the person following it.

A Run holds everything that happened, for the page to show, and takes what the person does:
a note for the agents, which the next agent to speak reads first; a pause, which holds the
meeting before the next turn; a stop, which ends it there and keeps what was said; and, in a
project, the team lead's decisions, which wait for the person's approval unless the project is
left to run on its own. An MCP tool that waits for approval, or a server's question, is asked
on the page too, rather than at a terminal no one is watching.

Nothing here depends on Gradio, so that a run can be driven, and tested, without a page.
"""

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from virtual_lab.agent import Agent
from virtual_lab.approval import ApprovalRequest, ServerQuestion, read_value, withdrawn
from virtual_lab.constants import CONSISTENT_TEMPERATURE, PROJECT_FILE_NAME
from virtual_lab.events import MeetingEvent, NextTurn, ProjectEvent
from virtual_lab.execution import DockerExecutor, Executor, LocalExecutor
from virtual_lab.mcp_presets import MCP_PRESETS
from virtual_lab.planning import NextStep, run_project
from virtual_lab.project import Project
from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC
from virtual_lab.provenance import describe_agent
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.session import DockerSession, LocalSession, Session, session_executor
from virtual_lab.tools import TOOL_REGISTRY
from virtual_lab.ui.workspace import MEETING_NAME, PROJECT_SETUP_FILE_NAME, Workspace, read_json
from virtual_lab.utils import load_summaries, write_atomically

RunKind = Literal["meeting", "project"]
RunStatus = Literal["running", "completed", "failed", "stopped"]
WaitingKind = Literal["decision", "approval", "question"]

# The events a run can be stopped at by raising from on_event: those told of in the thread that
# runs the meeting or project, before the work they announce, and not its last word, which a
# meeting or project only warns about an error from
STOPPABLE_MEETING_EVENTS = frozenset({"turn", "writing", "message", "tool_calls", "code", "usage"})
STOPPABLE_PROJECT_EVENTS = frozenset({"team", "plan", "decided", "code", "round"})

# How often, in seconds, something waiting on the person looks again for a stop or a withdrawal
POLL_SECONDS = 0.2

CODE_PLACES = ("none", "docker", "local")
SANDBOXES = ("python", "base", "bio", "full")
RESOURCE_MODES = ("retrieve", "all", "none")


class RunStopped(Exception):
    """Raised in a meeting or project that the person following it stopped."""

    def __init__(self) -> None:
        super().__init__("Stopped by the person following it")


@dataclass
class Waiting:
    """Something a run is waiting on the person for.

    :param id: Its number in the run, to answer it by.
    :param kind: "decision" for a project's next step, "approval" for a call to a tool that
        waits for approval, or "question" for an MCP server's question.
    :param subject: The NextStep, ApprovalRequest, or ServerQuestion.
    :param round: For a decision, the project's round.
    """

    id: int
    kind: WaitingKind
    subject: NextStep | ApprovalRequest | ServerQuestion
    round: int | None = None
    answered: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    answer: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class RunView:
    """A run as it stands, for the page to show; a copy, so it can be read while the run goes on.

    :param events: Everything that happened, in order, with the parts of a reply being written
        kept only as the latest, and usage kept as spent.
    :param notes: Notes not yet read by an agent.
    :param paused: Whether the person paused the run.
    :param holding: Whether it is held now, at a pause, before next_turn.
    :param next_turn: The turn the run is about to take, or is held before.
    :param stopping: "now" or "after_step" if the person asked it to stop, else None.
    :param autonomous: Whether a project's decisions are carried out without asking.
    :param waiting: What it is waiting on the person for, oldest first.
    :param spent: What it has spent in USD, as far as is known, or None.
    :param result: Where its result is: a meeting's transcript, or a project's directory, once
        it has one.
    """

    id: str
    kind: RunKind
    title: str
    directory: Path
    status: RunStatus
    error: str | None
    events: tuple[MeetingEvent | ProjectEvent, ...]
    notes: tuple[str, ...]
    paused: bool
    holding: bool
    next_turn: NextTurn | None
    stopping: str | None
    autonomous: bool
    waiting: tuple[Waiting, ...]
    spent: float | None
    max_cost: float | None
    started_at: float
    ended_at: float | None
    version: int
    result: Path | None


class Run:
    """A meeting or project running in a thread of its own.

    :param kind: "meeting" or "project".
    :param title: Its agenda or goal.
    :param directory: Where it is saved.
    :param max_cost: Its budget in USD, or None.
    :param autonomous: For a project, whether the team lead's decisions are carried out without
        asking the person first.
    """

    def __init__(
        self,
        kind: RunKind,
        title: str,
        directory: Path,
        max_cost: float | None = None,
        autonomous: bool = True,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.directory = Path(directory)
        self.max_cost = max_cost
        self.autonomous = autonomous
        self.status: RunStatus = "running"
        self.error: str | None = None
        self.result: Path | None = None
        self.project: Project | None = None
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.thread: threading.Thread | None = None

        self._changed = threading.Condition()
        self._events: list[MeetingEvent | ProjectEvent] = []
        self._version = 0
        self._notes: list[str] = []
        self._paused = False
        self._holding = False
        self._next_turn: NextTurn | None = None
        self._stopping: str | None = None
        self._waiting: dict[int, Waiting] = {}
        self._asked = 0
        self._spent: float | None = None

    # What the meeting or project calls

    def on_event(self, event: MeetingEvent | ProjectEvent) -> None:
        """Keeps an event, and stops the run here if the person asked it to stop now."""
        with self._changed:
            if isinstance(event, MeetingEvent) and event.kind == "usage":
                if self.kind == "meeting":
                    self._spent = event.data.get("cost")
            elif isinstance(event, MeetingEvent) and event.kind == "writing" and self._replaces_writing(event):
                self._events[-1] = event
            else:
                self._events.append(event)
            if isinstance(event, MeetingEvent) and event.kind == "finished" and self.kind == "meeting":
                self.result = Path(event.data["transcript_path"])
            self._touch()

            stoppable = STOPPABLE_MEETING_EVENTS if isinstance(event, MeetingEvent) else STOPPABLE_PROJECT_EVENTS
            if self._stopping == "now" and event.kind in stoppable:
                raise RunStopped()

    def _replaces_writing(self, event: MeetingEvent) -> bool:
        last = self._events[-1] if self._events else None
        return (
            isinstance(last, MeetingEvent)
            and last.kind == "writing"
            and last.meeting == event.meeting
            and last.data.get("request") == event.data.get("request")
        )

    def steer(self, turn: NextTurn) -> str | None:
        """Holds the meeting while it is paused, then gives the next agent the notes waiting."""
        with self._changed:
            self._next_turn = turn
            self._touch()
            while self._paused and self._stopping != "now":
                self._holding = True
                self._touch()
                self._changed.wait(POLL_SECONDS)
            if self._holding:
                self._holding = False
                self._touch()
            if self._stopping == "now":
                raise RunStopped()

            notes, self._notes = self._notes, []
            if notes:
                self._touch()

            return "\n\n".join(notes) or None

    def approve_step(self, round_: int, step: NextStep) -> NextStep | None:
        """The team lead's decision as the person approved or changed it, or None to stop."""
        with self._changed:
            while self._paused and self._stopping is None:
                self._holding = True
                self._touch()
                self._changed.wait(POLL_SECONDS)
            if self._holding:
                self._holding = False
                self._touch()
            if self._stopping == "now":
                raise RunStopped()
            if self._stopping == "after_step":
                return None
            if self.autonomous:
                return step

        answer = self._ask("decision", step, round_)
        if answer is None:
            if self._stopping == "now":
                raise RunStopped()
            return None

        return answer

    def approve_call(self, request: ApprovalRequest) -> bool:
        """Whether the person approved a call to a tool that waits for approval."""
        return self._ask("approval", request) is True

    def answer_question(self, question: ServerQuestion) -> dict[str, Any] | None:
        """The person's answer to an MCP server's question, or None if they declined it."""
        answer = self._ask("question", question)
        return answer if isinstance(answer, dict) else None

    def _ask(self, kind: WaitingKind, subject: Any, round_: int | None = None) -> Any:
        """Waits for the person's answer; None if the run is stopped or the question withdrawn."""
        with self._changed:
            self._asked += 1
            waiting = Waiting(id=self._asked, kind=kind, subject=subject, round=round_)
            self._waiting[waiting.id] = waiting
            self._touch()
            try:
                while not waiting.answered.is_set():
                    if self._stopping == "now" or (kind != "decision" and withdrawn()):
                        return None
                    if kind == "decision" and self._stopping == "after_step":
                        return None
                    if kind == "decision" and self.autonomous:
                        return subject
                    self._changed.wait(POLL_SECONDS)
                return waiting.answer
            finally:
                self._waiting.pop(waiting.id, None)
                self._touch()

    # What the person does

    def add_note(self, note: str) -> bool:
        """Adds a note for the next agent to speak to read. False if there is nothing to add it to."""
        note = note.strip()
        with self._changed:
            if not note or self.status != "running":
                return False
            self._notes.append(note)
            self._touch()

        return True

    def withdraw_notes(self) -> None:
        with self._changed:
            self._notes = []
            self._touch()

    def pause(self) -> None:
        with self._changed:
            self._paused = True
            self._touch()

    def resume(self) -> None:
        with self._changed:
            self._paused = False
            self._touch()

    def stop(self, now: bool = True) -> None:
        """Stops the run: now, at the next point it can be, keeping what was done; or, for a
        project, after the step under way, before the next decision, so that it ends with a report."""
        with self._changed:
            if self.status != "running":
                return
            if now or self.kind == "meeting":
                self._stopping = "now"
            elif self._stopping is None:
                self._stopping = "after_step"
            self._touch()

    def set_autonomous(self, autonomous: bool) -> None:
        """Whether a project's decisions are carried out without asking, including one waiting now."""
        with self._changed:
            self.autonomous = autonomous
            self._touch()

    def respond(self, waiting_id: int, answer: Any) -> bool:
        """Answers what the run is waiting for. False if it no longer waits for it.

        :param answer: For a decision, the NextStep to carry out, or None to stop the project; for
            an approval, True or False; for a question, the form's fields, {} once the page was
            gone to, or None to decline.
        """
        with self._changed:
            waiting = self._waiting.get(waiting_id)
            if waiting is None or waiting.answered.is_set():
                return False
            if waiting.kind == "decision" and answer is not None and not isinstance(answer, NextStep):
                raise TypeError(f"A decision is answered with a NextStep or None, not {type(answer).__name__}")
            waiting.answer = answer
            waiting.answered.set()
            self._touch()

        return True

    # The run itself

    def start(self, work: Callable[["Run"], Any], cleanup: Iterable[Callable[[], None]] = ()) -> None:
        """Runs work in a thread of its own, with this run, then whatever cleans up after it."""
        cleanups = list(cleanup)

        def target() -> None:
            status: RunStatus = "completed"
            error = None
            try:
                work(self)
            except RunStopped:
                status = "stopped"
            # Whatever went wrong is shown on the page, which is the only place anyone is looking
            except BaseException as caught:
                status = "failed"
                error = f"{type(caught).__name__}: {caught}"
            finally:
                for clean in cleanups:
                    try:
                        clean()
                    except Exception as cleanup_error:
                        print(f"Warning: cleaning up after a run failed: {cleanup_error!r}")
            with self._changed:
                self.status, self.error = status, error
                if status == "stopped" and self.kind == "meeting":
                    self.error = "Stopped by you. What was said until then is saved."
                self.ended_at = time.time()
                self._paused = False
                self._holding = False
                self._touch()

        self.thread = threading.Thread(target=target, name=f"virtual-lab-{self.kind}-{self.id}", daemon=True)
        self.thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Waits for the run to end. False if it is still running after timeout seconds."""
        if self.thread is not None:
            self.thread.join(timeout)
        return self.status != "running"

    def view(self) -> RunView:
        with self._changed:
            spent = self._spent
            if self.project is not None:
                spent = self.project.spent
            return RunView(
                id=self.id,
                kind=self.kind,
                title=self.title,
                directory=self.directory,
                status=self.status,
                error=self.error,
                events=tuple(self._events),
                notes=tuple(self._notes),
                paused=self._paused,
                holding=self._holding,
                next_turn=self._next_turn,
                stopping=self._stopping,
                autonomous=self.autonomous,
                waiting=tuple(
                    sorted(
                        (item for item in self._waiting.values() if not item.answered.is_set()),
                        key=lambda item: item.id,
                    )
                ),
                spent=spent,
                max_cost=self.max_cost,
                started_at=self.started_at,
                ended_at=self.ended_at,
                version=self._version,
                result=self.result,
            )

    @property
    def version(self) -> int:
        with self._changed:
            return self._version

    def _touch(self) -> None:
        self._version += 1
        self._changed.notify_all()


# Answering on the page


def form_answer(question: ServerQuestion, values: Mapping[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    """A server's form, filled in on the page as text, as its fields' values.

    :param question: The question.
    :param values: What was typed for each field, by name.
    :return: The answer, or None and what is wrong with it.
    """
    answer: dict[str, Any] = {}
    for name, schema in question.fields.items():
        text = str(values.get(name, "")).strip()
        if not text:
            if name in question.required:
                return None, f"{name} must be filled in"
            continue
        ok, value = read_value(text, schema)
        if not ok:
            return None, f"{text!r} is not a value {name} takes"
        answer[name] = value

    return answer, None


def changed_step(
    step: NextStep,
    agenda: str | None = None,
    agenda_questions: str | None = None,
    participants: str | None = None,
    answer: str | None = None,
) -> NextStep:
    """The team lead's decision with what the person changed on the page.

    :param agenda_questions: One question a line.
    :param participants: Titles, separated by commas.
    """
    changes: dict[str, Any] = {}
    if agenda is not None:
        changes["agenda"] = agenda.strip()
    if agenda_questions is not None:
        changes["agenda_questions"] = [line.strip() for line in agenda_questions.splitlines() if line.strip()]
    if participants is not None:
        changes["participants"] = [title.strip() for title in participants.split(",") if title.strip()]
    if answer is not None:
        changes["answer"] = answer.strip()

    return NextStep.model_validate({**step.model_dump(mode="json"), **changes})


# What a run is set up with


def agent_from(data: Mapping[str, Any]) -> Agent:
    """An agent from its description, as describe_agent gives it, or as the page has it."""
    return Agent(
        title=str(data["title"]).strip(),
        expertise=str(data["expertise"]).strip(),
        goal=str(data["goal"]).strip(),
        role=str(data["role"]).strip(),
        model=str(data["model"]).strip(),
    )


def agent_data(agent: Agent) -> dict[str, str]:
    return {key: value for key, value in describe_agent(agent).items() if key != "name"}


@dataclass
class CodeSetup:
    """Where agents' code runs, if anywhere.

    :param where: "none" for nowhere, "docker" for a container, or "local" for this machine,
        with no isolation.
    :param sandbox: With Docker, "python" for a plain Python image, or the stage of Biomni's
        environment to run in: "base", "bio", or "full", which must have been built.
    :param network: With Docker, whether code can reach the network.
    :param python: On this machine, the interpreter to run, or "" for this one.
    :param resources: What of Biomni's tools, data, and know-how the agents are told they can use
        in the session: "retrieve" for what the agenda needs, as the lead picks it, "all", or "none".
    """

    where: str = "none"
    sandbox: str = "python"
    network: bool = True
    python: str = ""
    resources: str = "retrieve"

    def __post_init__(self) -> None:
        if self.where not in CODE_PLACES:
            raise ValueError(f"Code runs in one of {', '.join(CODE_PLACES)}, not {self.where!r}")
        if self.sandbox not in SANDBOXES:
            raise ValueError(f"The sandbox is one of {', '.join(SANDBOXES)}, not {self.sandbox!r}")
        if self.resources not in RESOURCE_MODES:
            raise ValueError(f"The resources are one of {', '.join(RESOURCE_MODES)}, not {self.resources!r}")

    def docker_executor(self) -> DockerExecutor:
        if self.sandbox == "python":
            return DockerExecutor(allow_network=self.network)
        return session_executor(self.sandbox, allow_network=self.network)

    def session(self, directory: Path) -> Session | None:
        """A session for agents to run code in, in directory, or None where code is not run."""
        if self.where == "docker":
            return DockerSession(directory, executor=self.docker_executor())
        if self.where == "local":
            return LocalSession(directory, python=self.python or None, warn=False)
        return None

    def executor(self) -> Executor | None:
        """What a project runs the code it writes with, or None where code is not run."""
        if self.where == "docker":
            return self.docker_executor()
        if self.where == "local":
            return LocalExecutor(warn=False)
        return None


@dataclass
class Connections:
    """The tools a run's agents may call.

    :param tools: The names of database tools, as TOOL_REGISTRY has them.
    :param mcp_config: A file of MCP servers, as connect_mcp reads it, or "".
    :param mcp_servers: The servers to connect to: those of the file, and presets, by name.
    """

    tools: list[str] = field(default_factory=list)
    mcp_config: str = ""
    mcp_servers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if unknown := [name for name in self.tools if name not in TOOL_REGISTRY]:
            raise ValueError(f"There is no tool {', '.join(unknown)}")

    def open(self, run: Run, stack: ExitStack) -> tuple[Any, ...]:
        """The tools, connected, with any MCP servers' approvals and questions asked on the page;
        the servers are closed with stack."""
        tools: list[Any] = [TOOL_REGISTRY[name] for name in self.tools]
        if self.mcp_servers:
            from virtual_lab.mcp_tools import connect_mcp, read_config

            config = self.mcp_config.strip() or None
            in_config = set(read_config(config)) if config is not None else set()
            presets = [name for name in self.mcp_servers if name not in in_config]
            if unknown := [name for name in presets if name not in MCP_PRESETS]:
                raise ValueError(f"There is no MCP server {', '.join(unknown)} in the config or the presets")
            mcp = connect_mcp(
                config,
                servers=self.mcp_servers,
                presets=presets,
                approve=run.approve_call,
                answer=run.answer_question,
            )
            stack.callback(mcp.close)
            tools += mcp.tools

        return tuple(tools)


@dataclass
class MeetingSetup:
    """A meeting, as the page sets it up.

    :param meeting_type: "team" or "individual".
    :param agenda: What the meeting is about.
    :param lead: The team lead of a team meeting, or the one who works in an individual meeting.
    :param members: The other members of a team meeting.
    :param critic: The critic of an individual meeting, or None for the Scientific Critic.
    :param summaries: Transcripts of earlier meetings whose summaries the meeting is given.
    :param max_cost: The budget in USD, or None.
    """

    meeting_type: str
    agenda: str
    lead: dict[str, str]
    members: list[dict[str, str]] = field(default_factory=list)
    critic: dict[str, str] | None = None
    agenda_questions: list[str] = field(default_factory=list)
    agenda_rules: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    num_rounds: int = 1
    temperature: float = CONSISTENT_TEMPERATURE
    max_cost: float | None = None
    stream: bool = True
    code: CodeSetup = field(default_factory=CodeSetup)
    connections: Connections = field(default_factory=Connections)

    def __post_init__(self) -> None:
        if self.meeting_type not in ("team", "individual"):
            raise ValueError(f'A meeting is "team" or "individual", not {self.meeting_type!r}')
        if not self.agenda.strip():
            raise ValueError("A meeting needs an agenda")
        if self.meeting_type == "team" and not self.members:
            raise ValueError("A team meeting needs members besides its lead")
        if self.num_rounds < 0:
            raise ValueError("The rounds cannot be fewer than none")
        if isinstance(self.code, dict):
            self.code = CodeSetup(**self.code)
        if isinstance(self.connections, dict):
            self.connections = Connections(**self.connections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeetingSetup":
        return cls(**{item.name: data[item.name] for item in fields(cls) if item.name in data})


@dataclass
class ProjectSetup:
    """A project, as the page sets it up. Carrying a project on needs the same team lead,
    critic, team, max_team_size, meeting_rounds, and memory, so they are kept with it.

    :param team: The team besides the team lead and critic, or [] for the team lead to choose.
    :param autonomous: Whether the team lead's decisions are carried out without asking.
    """

    goal: str
    lead: dict[str, str] = field(default_factory=lambda: agent_data(PRINCIPAL_INVESTIGATOR))
    critic: dict[str, str] = field(default_factory=lambda: agent_data(SCIENTIFIC_CRITIC))
    team: list[dict[str, str]] = field(default_factory=list)
    max_team_size: int = 3
    max_rounds: int = 10
    max_stalled_rounds: int = 3
    meeting_rounds: int = 1
    memory: str = "pick"
    max_cost: float | None = None
    autonomous: bool = False
    stream: bool = True
    code: CodeSetup = field(default_factory=CodeSetup)
    connections: Connections = field(default_factory=Connections)

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("A project needs a goal")
        if isinstance(self.code, dict):
            self.code = CodeSetup(**self.code)
        if isinstance(self.connections, dict):
            self.connections = Connections(**self.connections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectSetup":
        return cls(**{item.name: data[item.name] for item in fields(cls) if item.name in data})


def read_setup(directory: Path) -> MeetingSetup | ProjectSetup | None:
    """What a meeting or project run from the page was set up with, if it was."""
    data = read_json(Path(directory) / PROJECT_SETUP_FILE_NAME)
    if not isinstance(data, dict):
        return None
    try:
        if "goal" in data:
            return ProjectSetup.from_dict(data)
        return MeetingSetup.from_dict(data)
    except (TypeError, ValueError):
        return None


def save_setup(directory: Path, setup: MeetingSetup | ProjectSetup) -> None:
    write_atomically(Path(directory) / PROJECT_SETUP_FILE_NAME, json.dumps(setup.to_dict(), indent=4).encode("utf-8"))


class Lab:
    """The runs of one web interface, over one workspace.

    :param workspace: Where meetings and projects are saved.
    :param client: An OpenAI client for every run, as hold_meeting takes one, or None.
    :param chat_models: Chat models for every run, as hold_meeting takes them, or None.
    """

    def __init__(self, workspace: Workspace, client: Any = None, chat_models: Any = None) -> None:
        self.workspace = workspace
        self.client = client
        self.chat_models = chat_models
        self.runs: dict[str, Run] = {}
        self.current: dict[str, str] = {}
        self._lock = threading.Lock()

    def run(self, run_id: str | None) -> Run | None:
        return self.runs.get(run_id or "")

    def latest(self, kind: RunKind) -> Run | None:
        """The run of this kind started last, which the page shows."""
        return self.run(self.current.get(kind))

    def running(self, directory: Path) -> Run | None:
        """The run saving into this directory now, if there is one."""
        directory = Path(directory).resolve()
        return next(
            (run for run in self.runs.values() if run.status == "running" and run.directory.resolve() == directory),
            None,
        )

    def _begin(self, run: Run) -> None:
        with self._lock:
            self.runs[run.id] = run
            self.current[run.kind] = run.id

    def start_meeting(self, setup: MeetingSetup) -> Run:
        """Starts a meeting in a directory of its own in the workspace."""
        lead = agent_from(setup.lead)
        members = tuple(agent_from(member) for member in setup.members)
        critic = agent_from(setup.critic) if setup.critic is not None else None
        summaries = load_summaries([Path(path) for path in setup.summaries])

        directory = self.workspace.new_directory("meeting", setup.agenda)
        save_setup(directory, setup)
        run = Run("meeting", setup.agenda, directory, max_cost=setup.max_cost)

        def work(run: Run) -> None:
            with ExitStack() as stack:
                tools = setup.connections.open(run, stack)
                session = setup.code.session(directory / "workspace")
                if session is not None:
                    stack.callback(session.close)
                people: dict[str, Any] = (
                    {"team_lead": lead, "team_members": members}
                    if setup.meeting_type == "team"
                    else {"team_member": lead, "critic": critic}
                )
                hold_meeting(
                    meeting_type=setup.meeting_type,  # type: ignore[arg-type]
                    agenda=setup.agenda,
                    save_dir=directory,
                    save_name=MEETING_NAME,
                    agenda_questions=tuple(setup.agenda_questions),
                    agenda_rules=tuple(setup.agenda_rules),
                    summaries=summaries,
                    num_rounds=setup.num_rounds,
                    temperature=setup.temperature,
                    tools=tools,
                    client=self.client,
                    chat_models=self.chat_models,
                    max_cost=setup.max_cost,
                    session=session,
                    resources=setup.code.resources,  # type: ignore[arg-type]
                    on_event=run.on_event,
                    stream=setup.stream,
                    steer=run.steer,
                    **people,
                )

        self._begin(run)
        run.start(work)

        return run

    def start_project(self, setup: ProjectSetup, directory: Path | None = None) -> Run:
        """Starts a project in a directory of its own, or carries on the one in directory.

        :raises RuntimeError: If the project in directory is running already.
        """
        lead = agent_from(setup.lead)
        critic = agent_from(setup.critic)
        team = tuple(agent_from(member) for member in setup.team) or None

        if directory is None:
            directory = self.workspace.new_directory("project", setup.goal)
        elif self.running(directory) is not None:
            raise RuntimeError("This project is running already")
        save_setup(directory, setup)
        run = Run("project", setup.goal, directory, max_cost=setup.max_cost, autonomous=setup.autonomous)

        def work(run: Run) -> None:
            with ExitStack() as stack:
                tools = setup.connections.open(run, stack)
                session = setup.code.session(directory / "workspace")
                if session is not None:
                    stack.callback(session.close)
                project = Project(
                    directory,
                    setup.goal,
                    max_cost=setup.max_cost,
                    session=session,
                    chat_models=self.chat_models,
                    client=self.client,
                    tools=tools,
                    resources=setup.code.resources,
                    stream=setup.stream,
                    steer=run.steer,
                )
                run.project = project
                run.result = directory
                run_project(
                    project,
                    team_lead=lead,
                    critic=critic,
                    team=team,
                    max_team_size=setup.max_team_size,
                    max_rounds=setup.max_rounds,
                    max_stalled_rounds=setup.max_stalled_rounds,
                    meeting_rounds=setup.meeting_rounds,
                    executor=setup.code.executor(),
                    approve=run.approve_step,
                    memory=setup.memory,  # type: ignore[arg-type]
                    on_event=run.on_event,
                )

        self._begin(run)
        run.start(work)

        return run


def is_project_directory(directory: Path) -> bool:
    return (Path(directory) / PROJECT_FILE_NAME).is_file()
