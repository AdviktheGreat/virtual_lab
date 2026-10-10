"""A conversation with the head of a lab, who answers, runs code, or has the team work on it.

Meetings and projects are run to a plan: they are given an agenda or a goal and report when they
are done. A Chat is the other way round. A researcher talks to one agent, the lead, which answers
directly when it can, runs code in a session when the answer needs a computation, and, when a
question needs more than one kind of expertise, calls a tool to bring the team in: convene_team
for a discussion the lead leads, or consult for one scientist to work on a focused task with the
Scientific Critic reviewing it. Those are meetings, held as a Project holds them, in the same
session and within the same budget, and the lead is given what they concluded.

Everything that happens is told of through ChatEvents, kept in order, so that a page can follow a
turn as it goes and, after it is closed, find the conversation as it was. The conversation lives
in its directory: chat.json says what it is, messages.jsonl holds what the lead has been sent and
has said, events.jsonl what happened, uploads/ the files the researcher attached, and lab/ the
meetings the lead held, which are a Project. A Chat opened on a directory carries on from there.

One turn runs at a time, in the thread that calls send, or in one of its own with start. The
researcher can steer it from any other thread: add a note, which the lead, or the team if it is
working, reads before it goes on; pause it, to hold it before its next request; or stop it, which
ends it at the next point it can be and keeps what was done.
"""

import hashlib
import itertools
import json
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from virtual_lab.agent import Agent
from virtual_lab.completions import check_temperature, send_request
from virtual_lab.constants import (
    CHAT_EVENTS_FILE_NAME,
    CHAT_FILE_NAME,
    CHAT_LAB_DIR_NAME,
    CHAT_MESSAGES_FILE_NAME,
    CHAT_UPLOADS_DIR_NAME,
    CONSISTENT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    HUMAN_SPEAKER,
    MAX_CHAT_DELEGATION_ROUNDS,
    MAX_CHAT_DELEGATIONS,
    MAX_CHAT_TITLE_CHARS,
    MAX_TOOL_ITERATIONS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_NAME_CHARS,
    SESSION_MAX_TOOL_ITERATIONS,
)
from virtual_lab.events import ChatEvent, ChatEventKind, MeetingEvent, NextTurn, OnChatEvent
from virtual_lab.llm import ModelReply, ModelSource, ToolCall, ask, resolve_chat_models
from virtual_lab.project import Project, sha256_file
from virtual_lab.prompts import (
    SCIENTIFIC_CRITIC,
    chat_attachments_prompt,
    chat_note_prompt,
    chat_resources_prompt,
    chat_session_prompt,
    lab_chat_prompt,
)
from virtual_lab.provenance import describe_agent, utc_timestamp
from virtual_lab.resources import (
    Resources,
    available_resources,
    check_installed,
    parse_retrieval,
    resources_prompt,
    retrieval_prompt,
    select_resources,
)
from virtual_lab.run_meeting import MeetingResult, TruncatedResponseError
from virtual_lab.schemas import normalize_title
from virtual_lab.session import CODE_TOOL_NAME, Session, own_resources_prompt, session_tool
from virtual_lab.tools import PUBMED_TOOL, Tool, run_tool_calls, tool_instructions_prompt
from virtual_lab.utils import (
    BudgetExceededError,
    ContextLengthExceededError,
    CostUnknownError,
    MeetingUsage,
    check_context_length,
    combine_usage,
    compute_token_cost,
    count_message_tokens,
    get_max_input_tokens,
    write_atomically,
)

ChatState = Literal["idle", "running", "pausing", "paused", "stopping"]
ChatStatus = Literal["answered", "stopped", "failed"]

# The tools the lead has that are not its own to name, and the two that start a team session
FIND_RESOURCES_TOOL_NAME = "find_resources"
CONVENE_TOOL_NAME = "convene_team"
CONSULT_TOOL_NAME = "consult"
RESERVED_TOOL_NAMES = frozenset({CODE_TOOL_NAME, FIND_RESOURCES_TOOL_NAME, CONVENE_TOOL_NAME, CONSULT_TOOL_NAME})

# The events of a team session that a stop interrupts, since they are told of before the work
# they announce, in the thread that does it
STOPPABLE_LAB_EVENTS = frozenset({"turn", "writing", "message", "tool_calls", "code"})

# What a team session's events are not passed on for: its usage is the conversation's, told of
# as one, and its prompts, which hold all the resources it was told of, are in its transcript
UNTOLD_LAB_EVENTS = frozenset({"usage"})

STOPPED_REPLY = "I was stopped by the researcher before I finished."

# What the project that holds the team's meetings is for, which only has to be the same every time
CHAT_PROJECT_GOAL = "Conversation with the head of a lab"


class ChatBusyError(RuntimeError):
    """Raised when a message is sent to a conversation that is still working on the last one."""


class ChatClosedError(RuntimeError):
    """Raised when a message is sent to a conversation that was closed."""


class ChatStopped(Exception):
    """Raised in a turn the researcher stopped, at the point it was stopped."""

    def __init__(self) -> None:
        super().__init__("Stopped by the researcher")


class UploadTooLargeError(ValueError):
    """Raised when a file the researcher attached is larger than a conversation takes."""


@dataclass(frozen=True)
class Attachment:
    """A file the researcher attached to the conversation.

    :param name: Its name, in the uploads directory.
    :param path: Where it is, relative to the session's working directory, as code reads it.
    :param size: Its size in bytes.
    """

    name: str
    path: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "size": self.size}


@dataclass(frozen=True)
class ChatReply:
    """How a turn ended.

    :param status: "answered" if the lead replied, "stopped" if the researcher stopped it, or
        "failed" if it ended with an error, which is also told of as an event.
    :param text: The lead's answer, or an empty string if it did not give one.
    :param error: What stopped it, as its type and message, if it failed.
    :param usage: The usage of the whole conversation so far, as MeetingUsage.to_dict gives it.
    :param elapsed_time: Seconds the turn took.
    """

    status: ChatStatus
    text: str
    error: str | None
    usage: dict[str, Any]
    elapsed_time: float


def safe_filename(name: str) -> str:
    """Makes a name a file can be saved under in a directory, whatever it was uploaded as.

    :param name: The name the file came with, which may have a path, as a browser sends it.
    :return: Its last part, without control characters or a leading dot, and not too long.
    """
    name = re.sub(r"[\x00-\x1f\x7f]", "", name.replace("\\", "/").rsplit("/", 1)[-1])
    name = name.strip().lstrip(".").strip()
    if len(name) > MAX_UPLOAD_NAME_CHARS:
        stem, suffix = os.path.splitext(name)
        suffix = suffix[:20]
        name = stem[: MAX_UPLOAD_NAME_CHARS - len(suffix)] + suffix

    return name or "upload"


def describe_size(size: int) -> str:
    """A size in bytes as a person reads it."""
    if size < 1024:
        return f"{size} bytes"

    value = size / 1024
    for unit in ("KB", "MB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} GB"


def title_of(text: str, files: Sequence[Attachment]) -> str:
    """A conversation's title from the first thing the researcher said."""
    line = next((" ".join(line.split()) for line in text.splitlines() if line.strip()), "")
    line = line or (files[0].name if files else "")
    if len(line) > MAX_CHAT_TITLE_CHARS:
        line = line[: MAX_CHAT_TITLE_CHARS - 1].rstrip() + "…"

    return line


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Reads a file of JSON objects, one to a line, as the conversation saved it.

    A last line cut short, as a process killed while writing leaves, is dropped, and the file
    made whole again so that what is written next starts a line of its own.

    :raises ValueError: If any other line is not JSON, which is not what a crash does.
    """
    if not path.is_file():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number < len(lines):
                raise ValueError(f"Line {number} of {path} is not JSON: {error}") from error
            print(f"Warning: the last line of {path} was cut short, and is left out")
            write_atomically(path, "".join(f"{json.dumps(item, default=str)}\n" for item in items).encode("utf-8"))

    return items


def append_json_line(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def describe_call(call: ToolCall) -> dict[str, str]:
    return {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}


class Chat:
    """A conversation between a researcher and the head of a lab.

    :param save_dir: The conversation's directory, where an earlier run of it was saved, if there
        was one, which is carried on from.
    :param lead: The head of the lab, who the researcher talks to.
    :param team: The scientists the lead can bring in, each by its title. A Scientific Critic among
        them reviews a scientist the lead consults. With none, the lead works alone.
    :param session: A session for the lead and the team to run code in, as hold_meeting takes it,
        started on the first message and left running afterwards, so close it when done. It is
        also where the files the researcher attaches are saved, in its directory's uploads/,
        which is where code finds them. Without one nobody can run code.
    :param title: What to call the conversation; by default its first message, shortened.
    :param chat_models: The chat models to ask, as hold_meeting takes them.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param max_cost: The most the conversation may spend, in USD, the lead's requests and the team's
        together and over every run of it on this directory. It is checked before every request,
        so it can be overrun by the cost of one; max_completion_tokens bounds that. Every model
        must be priced for the limit to mean anything.
    :param temperature: The sampling temperature, as hold_meeting takes it.
    :param max_completion_tokens: The most tokens any one response may use, or None for the
        model's own limit.
    :param max_retries: The times to retry a failed API call, as hold_meeting takes it.
    :param tools: Tools the lead, and the team in its sessions, may call, as hold_meeting takes
        them. The tool that runs code is added if there is a session.
    :param pubmed_search: Whether to add the PubMed tool to them.
    :param resources: What of Biomni's environment the lead is told of in the session: with
        "retrieve", that it has a find_resources tool, which asks which suit a task, at the cost of
        one request of about 12,000 tokens when it does; with "all", every one, which is about
        40,000 tokens more in every request; or a Resources of your own. "none" lists nothing.
        The team's meetings are given the same. Only used with a session.
    :param commercial_mode: Whether to leave out data and know-how that may not be used
        commercially, as hold_meeting takes it.
    :param max_tool_iterations: The most tool calls the lead makes in answering one message before it
        is asked to answer without them. Defaults to 20 with a session, and 5 without one.
    :param max_delegations: The most team sessions the lead may start in answering one message.
        With none, the lead does not have the tools to start any.
    :param max_rounds: The most rounds of discussion the lead may ask a team session for.
    :param stream: Whether on_event, and anyone following the conversation, is also told of each
        reply as it is written, with "writing" events, as hold_meeting takes it.
    :param on_event: Called with every ChatEvent in the thread that caused it, which can be any
        thread that steers the conversation. It may call the conversation's own methods, but it must
        not wait for another thread that does. An exception it raises ends the turn the way any
        other failure does, except from an event that ends it, or one told of from a thread that
        did not start the turn, which are warned of.
    :param max_upload_bytes: The most bytes of one file the researcher may attach.
    :raises ValueError: If an argument is not valid, or the lead and the team do not have titles of
        their own.
    :raises CostUnknownError: If max_cost is given and a model is not priced.
    :raises ProjectStateError: If save_dir holds a project that is not this conversation's.
    """

    def __init__(
        self,
        save_dir: Path,
        lead: Agent,
        team: Sequence[Agent] = (),
        session: Session | None = None,
        title: str | None = None,
        chat_models: ModelSource | None = None,
        client: OpenAI | None = None,
        max_cost: float | None = None,
        temperature: float = CONSISTENT_TEMPERATURE,
        max_completion_tokens: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        tools: tuple[Tool, ...] = (),
        pubmed_search: bool = False,
        resources: Literal["retrieve", "all", "none"] | Resources = "retrieve",
        commercial_mode: bool = False,
        max_tool_iterations: int | None = None,
        max_delegations: int = MAX_CHAT_DELEGATIONS,
        max_rounds: int = MAX_CHAT_DELEGATION_ROUNDS,
        stream: bool = False,
        on_event: OnChatEvent | None = None,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        if max_cost is not None and not (math.isfinite(max_cost) and max_cost >= 0):
            raise ValueError(f"max_cost must be a finite amount, zero or more, not {max_cost}")
        check_temperature(temperature)
        if max_completion_tokens is not None and max_completion_tokens < 1:
            raise ValueError(f"max_completion_tokens must be at least 1, not {max_completion_tokens}")
        if max_tool_iterations is None:
            max_tool_iterations = SESSION_MAX_TOOL_ITERATIONS if session is not None else MAX_TOOL_ITERATIONS
        elif max_tool_iterations < 1:
            raise ValueError(f"max_tool_iterations must be at least 1, not {max_tool_iterations}")
        if max_delegations < 0:
            raise ValueError(f"max_delegations must be zero or more, not {max_delegations}")
        if max_rounds < 0:
            raise ValueError(f"max_rounds must be zero or more, not {max_rounds}")
        if max_upload_bytes < 1:
            raise ValueError(f"max_upload_bytes must be at least 1, not {max_upload_bytes}")
        if not_tools := [type(tool).__name__ for tool in tools if not isinstance(tool, Tool)]:
            raise TypeError(f"A conversation's tools are Tools, not {', '.join(not_tools)}")
        if isinstance(resources, Resources):
            if session is None:
                raise ValueError("resources lists what can be used in a session, so it needs a session")
        elif resources not in ("retrieve", "all", "none"):
            raise ValueError(f'resources must be "retrieve", "all", "none", or a Resources, not {resources!r}')

        team = tuple(team)
        titles = [normalize_title(agent.title) for agent in (lead, *team)]
        if not all(titles):
            raise ValueError("Everyone in a conversation needs a title")
        if len(set(titles)) != len(titles):
            raise ValueError("The lead and the team must have different titles")
        if normalize_title(HUMAN_SPEAKER) in titles:
            raise ValueError(f"{HUMAN_SPEAKER} is the title of the researcher's notes, so no agent may have it")

        chat_tools = tools
        if pubmed_search and PUBMED_TOOL.name not in {tool.name for tool in tools}:
            chat_tools = (PUBMED_TOOL, *tools)
        if taken := sorted({tool.name for tool in chat_tools} & RESERVED_TOOL_NAMES):
            raise ValueError(f"A conversation has tools of its own named {', '.join(taken)}, so no other may be")
        if len({tool.name for tool in chat_tools}) != len(chat_tools):
            raise ValueError("Tool names must be unique")

        if max_cost is not None:
            for agent in (lead, *team):
                try:
                    compute_token_cost(agent.model, 0, 0)
                except CostUnknownError as error:
                    raise CostUnknownError(
                        f"{error}, so a max_cost cannot be enforced. Add its prices to the tables in "
                        f"virtual_lab.constants, or run without a limit."
                    ) from error

        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.id = self.save_dir.name
        self.lead = lead
        self.team = team
        self.session = session
        self.max_cost = max_cost
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.resources = resources
        self.commercial_mode = commercial_mode
        self.max_tool_iterations = max_tool_iterations
        self.max_delegations = max_delegations
        self.max_rounds = max_rounds
        self.stream = stream
        self.on_event = on_event
        self.max_upload_bytes = max_upload_bytes
        self.tools = chat_tools

        # Built before the first request, so that a model whose provider cannot be worked out, or
        # whose package is not installed, fails here, before anything is spent
        self._llms = resolve_chat_models(
            [agent.model for agent in (lead, *team)],
            chat_models=chat_models,
            client=client,
            max_retries=max_retries,
        )

        critic = next(
            (agent for agent in team if normalize_title(agent.title) == normalize_title(SCIENTIFIC_CRITIC.title)), None
        )
        self._critic = critic
        self._consultants = tuple(agent for agent in team if agent is not critic)

        self._changed = threading.Condition()
        self._events: list[ChatEvent] = []
        self._live: dict[tuple[Any, ...], ChatEvent] = {}
        self._state_event: ChatEvent | None = None
        self._next_id = 1
        self._messages: list[dict[str, Any]] = []
        self._notes: list[str] = []
        self._state: ChatState = "idle"
        self._paused = False
        self._stopping = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self.last_reply: ChatReply | None = None

        self._turn = 0
        self._lab_calls = 0
        self._delegations = 0
        self._usage = MeetingUsage()
        self._requests = itertools.count(1)
        self._call_id: str | None = None

        # Worked out on the first message, since it starts the session
        self._system: str | None = None
        self._lead_tools: tuple[Tool, ...] = ()
        self._catalog: Resources | None = None
        self._cells_told = 0

        self.title = title or ""
        self.created_at = utc_timestamp()
        self._load()

        self.project = Project(
            save_dir=self.save_dir / CHAT_LAB_DIR_NAME,
            goal=CHAT_PROJECT_GOAL,
            session=session,
            chat_models=chat_models,
            client=client,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
            tools=chat_tools,
            resources=resources,
            commercial_mode=commercial_mode,
            stream=stream,
        )
        self._recover()

    # Loading and saving

    def _load(self) -> None:
        info_path = self.save_dir / CHAT_FILE_NAME
        self._interrupted = False
        numbered = 0
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            self.title = info.get("title") or self.title
            self.created_at = info.get("created_at", self.created_at)
            self._turn = info.get("turns", 0)
            self._lab_calls = info.get("lab_calls", 0)
            self._usage = MeetingUsage.from_dict(info.get("usage", {}))
            self._interrupted = bool(info.get("running"))
            numbered = info.get("events", 0)

        self._events = [ChatEvent.from_dict(saved) for saved in read_json_lines(self.save_dir / CHAT_EVENTS_FILE_NAME)]
        # The state the conversation was in is numbered like an event but not kept, so what was
        # numbered is saved too, and a page that saw that number is not told of a new event by it
        self._next_id = max(numbered, *(event.id for event in self._events), 0) + 1
        self._messages = read_json_lines(self.save_dir / CHAT_MESSAGES_FILE_NAME)

    def _recover(self) -> None:
        """Ends a turn that a process stopped in, so that the conversation can go on from it."""
        if not self._interrupted:
            return

        reason = "The process running this conversation stopped before the turn ended."
        self._close_open_calls(f"Not run. {reason}")
        self._emit("failed", text=reason, type="Interrupted")
        self._save_info()

    def _save_info(self) -> None:
        with self._changed:
            try:
                spent: float | None = self._total_usage().compute_cost()
            except CostUnknownError:
                spent = None
            info = {
                "id": self.id,
                "title": self.title,
                "created_at": self.created_at,
                "updated_at": utc_timestamp(),
                "lead": describe_agent(self.lead),
                "team": [describe_agent(agent) for agent in self.team],
                "session": self.session.describe() if self.session is not None else None,
                "max_cost": self.max_cost,
                "spent": spent,
                "turns": self._turn,
                "events": self._next_id - 1,
                "lab_calls": self._lab_calls,
                "running": self._state != "idle",
                "usage": self._usage.to_dict(),
            }
            write_atomically(self.save_dir / CHAT_FILE_NAME, json.dumps(info, indent=4, default=str).encode("utf-8"))

    def _append(self, message: dict[str, Any]) -> None:
        with self._changed:
            self._messages.append(message)
            append_json_line(self.save_dir / CHAT_MESSAGES_FILE_NAME, message)

    # Telling the caller

    def _record(self, kind: ChatEventKind, speaker: str | None, text: str, data: dict[str, Any]) -> ChatEvent:
        with self._changed:
            event = ChatEvent(
                kind=kind, id=self._next_id, turn=self._turn, speaker=speaker, text=text, data=data, time=time.time()
            )
            self._next_id += 1
            if kind == "writing":
                self._live[(data.get("meeting"), data.get("request"))] = event
            elif kind == "status":
                self._state_event = event
            else:
                # Whatever was being written is written, and is told of whole by this
                self._live.clear()
                self._events.append(event)
                append_json_line(self.save_dir / CHAT_EVENTS_FILE_NAME, event.to_dict())
            self._changed.notify_all()

        return event

    # Positional only, so that an event's own data can be called anything
    def _emit(self, kind: ChatEventKind, /, speaker: str | None = None, text: str = "", **data: Any) -> ChatEvent:
        event = self._record(kind, speaker, text, data)
        if self.on_event is not None:
            self.on_event(event)

        return event

    def _emit_ending(self, kind: ChatEventKind, text: str, **data: Any) -> None:
        """Tells of what ends a turn, which has been done and paid for by now, so that a caller's
        failure to take it does not undo that or hide the error that ended the turn."""
        try:
            self._emit(kind, text=text, **data)
        except Exception as error:
            print(f"Warning: on_event failed when told of the end of the turn: {error!r}")

    def _set_state(self, state: ChatState) -> None:
        with self._changed:
            if self._state == state:
                return
            self._state = state
            event = self._record("status", None, "", {"state": state})
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as error:
                print(f"Warning: on_event failed when told the conversation was {state}: {error!r}")

    def events_since(self, after: int = 0) -> list[ChatEvent]:
        """What has happened since the event numbered after, in order, for a page to catch up with.

        What is being written, and the state the conversation is in, are the latest of each, not
        what they were as they changed.
        """
        with self._changed:
            return self._since(after)

    def wait_for_events(self, after: int, timeout: float | None = None) -> list[ChatEvent]:
        """What has happened since the event numbered after, waiting up to timeout seconds for
        something to, or for the conversation to be closed. An empty list means nothing did."""
        with self._changed:
            self._changed.wait_for(lambda: self._next_id - 1 > after or self._closed, timeout)
            return self._since(after)

    def _since(self, after: int) -> list[ChatEvent]:
        latest = [self._state_event] if self._state_event is not None else []
        pending = [*self._events, *self._live.values(), *latest]

        return sorted((event for event in pending if event.id > after), key=lambda event: event.id)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """What the lead has been sent and has said, as it reads them, copied."""
        with self._changed:
            return [dict(message) for message in self._messages]

    @property
    def state(self) -> ChatState:
        with self._changed:
            return self._state

    @property
    def turns(self) -> int:
        """How many messages the researcher has sent."""
        with self._changed:
            return self._turn

    @property
    def usage(self) -> MeetingUsage:
        """What the lead and the team have used between them, in tokens, per model."""
        with self._changed:
            return self._total_usage()

    @property
    def spent(self) -> float | None:
        """What the conversation has cost in USD, or None if that cannot be worked out."""
        try:
            return self.usage.compute_cost()
        except CostUnknownError:
            return None

    def _total_usage(self) -> MeetingUsage:
        return combine_usage([self._usage, *(MeetingUsage.from_dict(step.usage) for step in self.project.steps)])

    def _usage_data(self) -> dict[str, Any]:
        return {**self._total_usage().to_dict(), "max_cost": self.max_cost}

    def _tell_usage(self) -> None:
        with self._changed:
            data = self._usage_data()
        self._emit("usage", **data)

    # What the researcher does

    def add_note(self, note: str) -> bool:
        """Adds a note, which the lead, or the team if it is working, reads before it goes on.

        :return: False if there is nothing to add it to, because no turn is running or the note is empty.
        """
        note = note.strip()
        with self._changed:
            if not note or self._state == "idle" or self._closed:
                return False
            self._notes.append(note)
            self._changed.notify_all()

        return True

    def pause(self) -> bool:
        """Holds the turn before its next request, or the next turn of a team session. A turn
        that is running code or a tool is carried through first. False if no turn is running."""
        with self._changed:
            if self._state not in ("running", "pausing", "paused"):
                return False
            self._paused = True
            if self._state == "running":
                self._set_state("pausing")

        return True

    def resume(self) -> bool:
        """Lets a paused turn go on. False if it was not paused."""
        with self._changed:
            if not self._paused:
                return False
            self._paused = False
            if self._state in ("pausing", "paused"):
                self._set_state("running")
            self._changed.notify_all()

        return True

    def stop(self) -> bool:
        """Ends the turn at the next point it can be, keeping what was done. Code or a tool that is
        running is not interrupted. False if no turn is running."""
        with self._changed:
            if self._state == "idle":
                return False
            self._stopping = True
            self._set_state("stopping")
            self._changed.notify_all()

        return True

    def rename(self, title: str) -> None:
        """Gives the conversation another title."""
        title = " ".join(title.split())
        with self._changed:
            self.title = title
            self._save_info()
        self._emit("title", text=title)

    def close(self, timeout: float | None = None) -> None:
        """Stops the turn, if one is running, and waits up to timeout seconds for it to end.

        The session is the caller's, and is left as it was.
        """
        self.stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        with self._changed:
            self._closed = True
            self._changed.notify_all()

    def __enter__(self) -> "Chat":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # Files

    @property
    def uploads_dir(self) -> Path:
        """Where the files the researcher attaches are kept: in the session's directory, where its code reads them."""
        return (self.session.directory if self.session is not None else self.save_dir) / CHAT_UPLOADS_DIR_NAME

    def save_upload(self, name: str, data: bytes | BinaryIO) -> Attachment:
        """Saves a file the researcher attached, to be mentioned when they send their message.

        A file is never replaced: one with a name already taken is kept under another, unless it
        is the same file, which is not saved again.

        :param name: The name it came with, which may have a path as a browser sends it.
        :param data: Its content.
        :raises UploadTooLargeError: If it is larger than max_upload_bytes.
        """
        filename = safe_filename(name)
        directory = self.uploads_dir
        directory.mkdir(parents=True, exist_ok=True)

        handle, temporary = tempfile.mkstemp(dir=directory, prefix=".upload.", suffix=".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(handle, "wb") as file:
                blocks: Iterable[bytes] = (
                    [data] if isinstance(data, bytes) else iter(lambda: data.read(1 << 20), b"")  # type: ignore[union-attr]
                )
                for block in blocks:
                    size += len(block)
                    if size > self.max_upload_bytes:
                        raise UploadTooLargeError(
                            f"{filename} is larger than the {describe_size(self.max_upload_bytes)} that can be attached"
                        )
                    digest.update(block)
                    file.write(block)

            stem, suffix = os.path.splitext(filename)
            for number in itertools.count(1):
                target = directory / (filename if number == 1 else f"{stem} ({number}){suffix}")
                if not target.exists():
                    os.replace(temporary, target)
                    break
                if sha256_file(target) == digest.hexdigest():
                    break
        finally:
            Path(temporary).unlink(missing_ok=True)

        return Attachment(name=target.name, path=f"{CHAT_UPLOADS_DIR_NAME}/{target.name}", size=size)

    def _attachments(self, items: Iterable[Attachment | str]) -> list[Attachment]:
        files: list[Attachment] = []
        for item in items:
            name = item.name if isinstance(item, Attachment) else item
            path = self.uploads_dir / name
            if name != safe_filename(name) or not path.is_file():
                raise ValueError(f"No file named {name!r} was attached to this conversation")
            files.append(Attachment(name=name, path=f"{CHAT_UPLOADS_DIR_NAME}/{name}", size=path.stat().st_size))

        return files

    # Sending a message

    def send(self, message: str, attachments: Iterable[Attachment | str] = ()) -> ChatReply:
        """Sends the lead a message, and waits for its answer.

        :param message: What the researcher says.
        :param attachments: Files attached with save_upload, as it returned them or by name.
        :raises ChatBusyError: If the last message is still being answered.
        :raises ChatClosedError: If the conversation was closed.
        :raises ValueError: If there is no message and no file, or a file was not attached.
        :return: How the turn ended. A turn that failed is reported in the reply, and told of as an
            event, rather than raised.
        """
        text, files = self._begin(message, attachments)

        return self._run_turn(text, files)

    def start(self, message: str, attachments: Iterable[Attachment | str] = ()) -> threading.Thread:
        """Sends the lead a message as send does, but answers in a thread of its own, followed through
        the events, and the last_reply once it has ended. The message is told of before this returns.

        :raises ChatBusyError: If the last message is still being answered.
        :raises ChatClosedError: If the conversation was closed.
        :raises ValueError: If there is no message and no file, or a file was not attached.
        :return: The thread, started.
        """
        text, files = self._begin(message, attachments)

        thread = threading.Thread(
            target=self._run_turn, args=(text, files), name=f"virtual-lab-chat-{self.id}", daemon=True
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._end_turn()
            raise

        return thread

    def _begin(self, message: str, attachments: Iterable[Attachment | str]) -> tuple[str, list[Attachment]]:
        """Takes the conversation for a turn, and tells of the message that begins it."""
        text = message.strip()
        files = self._attachments(attachments)
        if not text and not files:
            raise ValueError("A message needs something in it: some text or a file")

        with self._changed:
            if self._closed:
                raise ChatClosedError("This conversation was closed")
            if self._state != "idle":
                raise ChatBusyError("The lead is still answering the last message")
            self._turn += 1
            self._paused = False
            self._stopping = False
            self._notes = []
            self._delegations = 0
            self._set_state("running")
            self._save_info()

        try:
            self._emit("user", speaker=HUMAN_SPEAKER, text=text, attachments=[file.to_dict() for file in files])
            if not self.title:
                self.rename(title_of(text, files))
        except BaseException:
            self._end_turn()
            raise

        return text, files

    def _end_turn(self) -> None:
        with self._changed:
            self._paused = False
            self._stopping = False
            self._notes = []
            self._set_state("idle")
            self._save_info()

    def _run_turn(self, text: str, files: list[Attachment]) -> ChatReply:
        started = time.monotonic()
        status: ChatStatus = "answered"
        answer = ""
        error: str | None = None
        crash: BaseException | None = None

        try:
            self._prepare()
            self._append({"role": "user", "content": self._what_was_sent(text, files)})
            answer = self._converse()
        except ChatStopped:
            status = "stopped"
            self._close_open_calls("Not run: the researcher stopped the turn.")
            self._append({"role": "assistant", "name": self.lead.name, "content": STOPPED_REPLY})
        except BaseException as caught:
            status = "failed"
            error = f"{type(caught).__name__}: {caught}"
            self._close_open_calls(f"Not run: {error}")
            if not isinstance(caught, Exception):
                crash = caught

        with self._changed:
            unread, self._notes = self._notes, []
            usage = self._usage_data()
        for note in unread:
            self._emit_ending("note", note, to=None, read=False)

        elapsed = round(time.monotonic() - started, 3)
        if status == "answered":
            self._emit_ending("answer", answer, usage=usage, elapsed_time=elapsed)
        elif status == "stopped":
            self._emit_ending("stopped", "Stopped by you. What was done is kept.", elapsed_time=elapsed)
        else:
            assert error is not None
            self._emit_ending("failed", error, type=error.split(":", 1)[0], elapsed_time=elapsed)

        self._end_turn()
        self.last_reply = ChatReply(status=status, text=answer, error=error, usage=usage, elapsed_time=elapsed)
        if crash is not None:
            raise crash

        return self.last_reply

    def _what_was_sent(self, text: str, files: list[Attachment]) -> str:
        if not files:
            return text

        attached = chat_attachments_prompt(
            [(file.path, describe_size(file.size)) for file in files], can_open=self.session is not None
        )

        return f"{text}\n\n{attached}" if text else attached

    # Getting ready

    def _prepare(self) -> None:
        """Starts the session, and works out what the lead is told of it and of Biomni's resources. Once."""
        if self._system is not None:
            return

        session = self.session
        catalog: Resources | None = None
        not_found: list[str] = []
        if session is not None:
            # Before the first request, so that a session that cannot start, such as one whose image
            # was never built, fails the turn before anything is spent
            session.start()
            self._cells_told = len(session.history)
            if session.software:
                checked = check_installed(session, list(session.software), [], what="the software added to it")
                if checked is not None:
                    not_found = [name for name in session.software if name in checked[0]]
                    if not_found:
                        print(
                            f"Warning: the session does not have {', '.join(not_found)}: not as a Python "
                            "distribution or module, an R package, or a command. The lead is told of it "
                            "anyway, as not found."
                        )
            if isinstance(self.resources, Resources):
                catalog = self.resources
                if catalog.data_lake and catalog.data_lake_path is None:
                    catalog = replace(catalog, data_lake_path=session.data_lake_path())
            elif self.resources != "none":
                catalog = available_resources(session, commercial_mode=self.commercial_mode)

        tools: list[Tool] = list(self.tools)
        delegating = bool(self._consultants) and self.max_delegations > 0
        parts = [lab_chat_prompt(self.lead, self.team if delegating else (), can_run_code=session is not None)]

        if session is not None:
            tools.append(session_tool(session))
            parts.append(
                chat_session_prompt(session.where_code_runs(), session.can_reach_network(), session.data_lake_path())
            )
            if own := own_resources_prompt(session, not_found):
                parts.append(own)

        retrieving = catalog is not None and self.resources == "retrieve" and not catalog.is_empty()
        if catalog is not None:
            self._emit(
                "resources",
                mode=self.resources if isinstance(self.resources, str) else "given",
                commercial_mode=self.commercial_mode,
                available=catalog.counts(),
                not_installed=catalog.not_installed,
                selected=Resources().names() if retrieving else catalog.names(),
                retrieval=None,
            )
            if retrieving:
                self._catalog = catalog
                tools.append(self._find_resources_tool())
                parts.append(chat_resources_prompt(catalog.counts()))
            elif listed := resources_prompt(catalog, "tool", retrieved=False):
                parts.append(listed)

        if delegating:
            tools.extend(self._delegation_tools())

        if instructions := tool_instructions_prompt((*self.tools, *(session.tools if session is not None else ()))):
            parts.append(instructions)

        self._lead_tools = tuple(tools)
        self._system = "\n\n".join(parts)

    # The lead's turn

    def _converse(self) -> str:
        """Asks the lead, and runs what it calls, until it answers."""
        assert self._system is not None
        definitions = [tool.definition for tool in self._lead_tools]

        for iteration in range(self.max_tool_iterations + 1):
            # The tools are withheld on the final attempt, to force an answer
            final = iteration == self.max_tool_iterations

            if note := self._gate("lead"):
                self._append({"role": "user", "content": chat_note_prompt(note)})

            request: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": self._system},
                *self.messages,  # type: ignore[list-item]
            ]
            self._check_fits(request)
            self._check_budget()
            reply = self._ask(self.lead, request, definitions if definitions and not final else None)
            self._count(self.lead.model, reply)
            self._checkpoint()

            if reply.finish_reason == "length" and not reply.content.strip():
                raise TruncatedResponseError(
                    f"{self.lead.title} ran out of tokens before writing any of its answer. Allow more with "
                    "max_completion_tokens."
                )

            if final or not reply.tool_calls:
                if not reply.content.strip():
                    raise RuntimeError(f"{self.lead.title} sent no answer")
                if reply.finish_reason == "length":
                    print(f"Warning: {self.lead.title} ran out of tokens and its answer is cut short.")
                self._append({"role": "assistant", "name": self.lead.name, "content": reply.content})

                return reply.content

            self._append(
                {
                    "role": "assistant",
                    "name": self.lead.name,
                    "content": reply.content or None,
                    "tool_calls": [call.model_dump() for call in reply.tool_calls],
                }
            )
            self._emit(
                "tool_calls",
                speaker=self.lead.title,
                text=reply.content,
                calls=[describe_call(call) for call in reply.tool_calls],
            )
            self._run_calls(reply.tool_calls)

        raise AssertionError("The final attempt returns or raises")  # pragma: no cover

    def _run_calls(self, calls: Iterable[ToolCall]) -> None:
        """Runs the lead's tool calls one at a time, so that each is told of when it is done."""
        for call in calls:
            self._checkpoint()
            self._call_id = call.id
            outputs, messages = run_tool_calls([call], self._lead_tools)  # type: ignore[arg-type]
            self._call_id = None
            self._append(dict(messages[0]))
            self._tell_cells(call.id, own=call.function.name not in (CONVENE_TOOL_NAME, CONSULT_TOOL_NAME))
            self._emit("tool_output", text=outputs[0], call=call.id, name=call.function.name)

    def _tell_cells(self, call_id: str, own: bool) -> None:
        """Tells of the code the session has run since it was last told, by the call that ran it. A
        team's code is told of by the events of its meeting."""
        session = self.session
        if session is None:
            return
        cells = session.history[self._cells_told :]
        self._cells_told = len(session.history)
        if own:
            for cell in cells:
                plot_paths = [str(session.directory / plot) for plot in cell.plots]
                self._emit("cell", **cell.to_dict(), plot_paths=plot_paths, call=call_id)

    def _ask(
        self,
        agent: Agent,
        messages: list[ChatCompletionMessageParam],
        tools: list[Any] | None,
        speaking: bool = True,
    ) -> ModelReply:
        """Asks an agent for its next message, without a temperature if its model refuses one.

        :param speaking: Whether the reply is the agent's say, which is streamed to those following the
            conversation, rather than a request made for the conversation's sake.
        """
        on_text = None
        if self.stream and speaking:
            request = next(self._requests)

            def on_text(text: str) -> None:
                self._emit("writing", speaker=agent.title, text=text, request=request, model=agent.model)
                self._checkpoint()

        return send_request(
            lambda temperature: ask(
                self._llms[agent.model],
                messages,
                temperature=temperature,
                tools=tools or None,
                max_tokens=self.max_completion_tokens,
                reader=agent.name,
                on_text=on_text,
            ),
            model=agent.model,
            temperature=self.temperature,
        )

    def _count(self, model: str, reply: ModelReply) -> None:
        """Adds a response's usage to the lead's, and tells of the conversation's."""
        with self._changed:
            self._usage.add(model=model, usage=reply.usage)
            self._save_info()
        self._tell_usage()

    def _check_budget(self) -> None:
        """Stops the turn before a request that the conversation has no money left for."""
        if self.max_cost is None:
            return

        spent = self.usage.compute_cost()
        if spent >= self.max_cost:
            raise BudgetExceededError(spent=spent, limit=self.max_cost, what="conversation")

    def _check_fits(self, messages: list[ChatCompletionMessageParam]) -> None:
        try:
            check_context_length(messages=messages, model=self.lead.model)
        except ContextLengthExceededError as error:
            raise ContextLengthExceededError(
                f"This conversation has grown to about {count_message_tokens(messages):,} tokens, more than the "
                f'{get_max_input_tokens(self.lead.model):,} that "{self.lead.model}" can read. Start a new '
                "conversation, and say in its first message what you need from this one."
            ) from error

    # Steering

    def _checkpoint(self) -> None:
        """Ends the turn here if the researcher stopped it."""
        with self._changed:
            if self._stopping:
                raise ChatStopped()

    def _gate(self, to: Literal["lead", "lab"]) -> str | None:
        """Holds the turn while it is paused, ends it if it was stopped, and gives back the notes the
        researcher added, which whoever goes on reads first.

        :param to: Who is about to read them.
        """
        with self._changed:
            while self._paused and not self._stopping:
                if self._state == "pausing":
                    self._set_state("paused")
                self._changed.wait()
            if self._stopping:
                raise ChatStopped()
            notes, self._notes = self._notes, []

        for note in notes:
            self._emit("note", speaker=HUMAN_SPEAKER, text=note, to=to, read=True)

        return "\n\n".join(notes) or None

    def _close_open_calls(self, reason: str) -> None:
        """Answers the lead's tool calls that were never run, so that what it is sent next is valid."""
        with self._changed:
            index = next(
                (i for i in range(len(self._messages) - 1, -1, -1) if self._messages[i]["role"] == "assistant"), None
            )
            if index is None or not self._messages[index].get("tool_calls"):
                return
            answered = {message["tool_call_id"] for message in self._messages[index + 1 :] if message["role"] == "tool"}
            open_calls = [call for call in self._messages[index]["tool_calls"] if call["id"] not in answered]

        for call in open_calls:
            self._append({"role": "tool", "tool_call_id": call["id"], "content": reason})
            self._emit("tool_output", text=reason, call=call["id"], name=call["function"]["name"])

    # Resources

    def _find_resources_tool(self) -> Tool:
        return Tool(
            name=FIND_RESOURCES_TOOL_NAME,
            description=(
                "Find which of Biomni's tool functions, data lake files, software libraries, and know-how "
                "documents in the session's environment suit a task, and see how to use them. Call it before "
                "you write code for a task that may use them, and describe the task in a sentence or two, "
                "saying what is to be found out and from what. It does not run anything."
            ),
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "The task, in a sentence or two."}},
                "required": ["task"],
            },
            function=self._find_resources,
        )

    def _find_resources(self, task: str) -> str:
        assert self._catalog is not None
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": retrieval_prompt(task, self._catalog)}
        ]
        check_context_length(messages=messages, model=self.lead.model)
        self._check_budget()
        reply = self._ask(self.lead, messages, None, speaking=False)
        self._count(self.lead.model, reply)

        chosen = parse_retrieval(reply.content)
        if chosen is None:
            return (
                "Which resources suit that task could not be worked out, since the answer named none of the "
                "kinds. Describe the task more specifically and ask again."
            )

        selected = select_resources(self._catalog, chosen)
        self._emit(
            "resources",
            mode="retrieve",
            commercial_mode=self.commercial_mode,
            available=self._catalog.counts(),
            not_installed=self._catalog.not_installed,
            selected=selected.names(),
            retrieval={"task": task, "model": self.lead.model, "reply": reply.content},
        )

        return resources_prompt(selected, "tool", retrieved=True) or "None of the resources suit that task."

    # The team

    def _delegation_tools(self) -> tuple[Tool, ...]:
        titles = [agent.title for agent in self.team]
        consultants = [agent.title for agent in self._consultants]
        rounds = {
            "type": "integer",
            "minimum": 0,
            "maximum": self.max_rounds,
            "description": (
                "Rounds of discussion before the summary. 0 is one answer with no discussion. Defaults to 1; "
                f"at most {self.max_rounds}."
            ),
        }
        questions = {
            "type": "array",
            "items": {"type": "string"},
            "description": "Questions the team must answer by the end, if there are particular ones.",
        }
        warning = (
            "The team cannot see this conversation, so the agenda must have everything the work needs. "
            "A session costs many requests: use it when it will change the answer."
        )

        return (
            Tool(
                name=CONVENE_TOOL_NAME,
                description=(
                    "Have several scientists of your team discuss something, with you leading, and be given the "
                    "summary you reach. Use it when a question needs more than one kind of expertise, or "
                    "someone to push back on a plan. Include the Scientific Critic when the answer needs "
                    f"checking. {warning}"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "agenda": {
                            "type": "string",
                            "description": "What the team is to work on: the goal, the files and data, what has "
                            "been tried, and what you want back.",
                        },
                        "members": {
                            "type": "array",
                            "items": {"type": "string", "enum": titles},
                            "description": "Who takes part, by title, besides you.",
                        },
                        "questions": questions,
                        "rounds": rounds,
                    },
                    "required": ["agenda", "members"],
                },
                function=self._convene_team,
            ),
            Tool(
                name=CONSULT_TOOL_NAME,
                description=(
                    "Have one scientist of your team work on a focused task, with the Scientific Critic "
                    f"reviewing it, and be given the answer. {warning}"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "member": {"type": "string", "enum": consultants, "description": "Who, by title."},
                        "agenda": {
                            "type": "string",
                            "description": "The task: the goal, the files and data, what has been tried, and "
                            "what you want back.",
                        },
                        "questions": questions,
                        "rounds": rounds,
                    },
                    "required": ["member", "agenda"],
                },
                function=self._consult,
            ),
        )

    def _find_member(self, title: str) -> Agent | None:
        wanted = normalize_title(str(title))
        return next((agent for agent in self.team if normalize_title(agent.title) == wanted), None)

    def _convene_team(
        self, agenda: str, members: list[str], questions: list[str] | None = None, rounds: int = 1
    ) -> str:
        chosen: list[Agent] = []
        for title in members:
            if normalize_title(str(title)) == normalize_title(self.lead.title):
                return f"Not held: you lead the discussion, so name the others, from {self._roster()}."
            member = self._find_member(title)
            if member is None:
                return f'Not held: there is no "{title}" on your team. Your team is {self._roster()}.'
            if member not in chosen:
                chosen.append(member)
        if not chosen:
            return f"Not held: name at least one member of your team, from {self._roster()}."

        names = ", ".join(member.title for member in chosen)

        return self._hold(
            "team",
            agenda,
            questions,
            rounds,
            {"team_lead": self.lead, "team_members": tuple(chosen)},
            lambda result, rounds: (
                f"{names} took part in a discussion that you led, for {rounds} round{'s' if rounds != 1 else ''} "
                f"of discussion. The summary you reached with them:\n\n{result.summary}"
            ),
        )

    def _consult(self, member: str, agenda: str, questions: list[str] | None = None, rounds: int = 1) -> str:
        found = self._find_member(member)
        if found is None or found is self._critic:
            return (
                f'Not held: "{member}" cannot be consulted. Consult one of '
                f"{', '.join(agent.title for agent in self._consultants)}."
            )

        options: dict[str, Any] = {"team_member": found}
        reviewer = self._critic.title if self._critic is not None else SCIENTIFIC_CRITIC.title
        if self._critic is not None:
            options["critic"] = self._critic

        return self._hold(
            "individual",
            agenda,
            questions,
            rounds,
            options,
            lambda result, rounds: (
                f"{found.title} worked on it, with the {reviewer} reviewing, for {rounds} round"
                f"{'s' if rounds != 1 else ''} of review. Their answer:\n\n{result.summary}"
            ),
        )

    def _roster(self) -> str:
        return ", ".join(agent.title for agent in self.team)

    def _hold(
        self,
        meeting_type: Literal["team", "individual"],
        agenda: str,
        questions: list[str] | None,
        rounds: int,
        who: dict[str, Any],
        report: Callable[[MeetingResult, int], str],
    ) -> str:
        """Holds a meeting for the lead, as a project would, and writes what it found as the lead reads it."""
        if not str(agenda).strip():
            return "Not held: the agenda is empty. Say what the team is to work on."
        if self._delegations >= self.max_delegations:
            return (
                f"Not held: the team has already worked {self._delegations} time"
                f"{'s' if self._delegations != 1 else ''} on this message, which is the most allowed. Answer with "
                "what you have, and say what is left, or ask the researcher to carry on."
            )

        rounds = max(0, min(int(rounds), self.max_rounds))
        with self._changed:
            self._delegations += 1
            self._lab_calls += 1
            name = f"lab_{self._lab_calls:03d}"
            self._save_info()

        call = self._call_id
        read_by_team: list[str] = []

        def on_event(event: MeetingEvent) -> None:
            self._pass_on(call, event)

        def steer(turn: NextTurn) -> str | None:
            note = self._gate("lab")
            if note:
                read_by_team.append(note)

            return note

        try:
            result = self.project.meeting(
                meeting_type,
                str(agenda),
                name=name,
                agenda_questions=tuple(str(question) for question in questions or ()),
                num_rounds=rounds,
                on_event=on_event,
                steer=steer,
                on_usage=lambda usage: self._tell_usage(),
                before_request=self._check_budget,
                **who,
            )
        except ChatStopped:
            return "The researcher stopped the team before it finished, so there is no summary."

        text = report(result, rounds)
        if read_by_team:
            text += "\n\nWhile the team worked, the researcher told it:\n\n" + "\n\n".join(read_by_team)

        return text

    def _pass_on(self, call: str | None, event: MeetingEvent) -> None:
        """Tells of an event of a team session as the conversation's, and ends it if the researcher stopped it."""
        if event.kind in UNTOLD_LAB_EVENTS or (event.kind == "message" and event.data.get("kind") == "prompt"):
            return

        if event.kind == "writing":
            self._emit(
                "writing", speaker=event.speaker, text=event.text, **event.data, meeting=event.meeting, call=call
            )
        else:
            self._emit(
                "lab",
                speaker=event.speaker,
                text=event.text,
                call=call,
                event=event.kind,
                meeting=event.meeting,
                round=event.round,
                details=dict(event.data),
            )

        if event.kind in STOPPABLE_LAB_EVENTS:
            self._checkpoint()
