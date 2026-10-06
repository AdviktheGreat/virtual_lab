"""A running interpreter that a meeting's agents share, the way Biomni's agent has its REPL.

Running a finished script tells a meeting whether the script works. An analysis is not written
that way: it loads the data once, looks at it, and decides what to do next from what it saw.
A session keeps a Python interpreter running between pieces of code, so a table loaded by one
agent is still there for the next, and figures are saved as they are drawn. R and shell code
can be run in it too, each as a fresh process, as Biomni runs them.

The interpreter is virtual_lab/kernel.py, started in the sandbox with "python3 -c", so the
image needs nothing but a Python 3. DockerSession runs it under every restriction
DockerExecutor applies; LocalSession runs it on this machine with none of them.

A session can be given tools, which code in it calls like any function, but which run here,
outside the sandbox, the way Biomni's agent calls the tools added to it. The code's call comes
back over the same pipe as its output, the tool is run, and what it returns is sent back. It can
be given data and software too, as Biomni's agent is with add_data and add_software, and the
agents are told of all three first.
"""

import json
import keyword
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

from pydantic_core import to_jsonable_python

from virtual_lab.approval import asked_for
from virtual_lab.constants import (
    DEFAULT_EXECUTION_TIMEOUT,
    DEFAULT_SESSION_CPU_LIMIT,
    DEFAULT_SESSION_MEMORY_LIMIT,
    DEFAULT_SESSION_PIDS_LIMIT,
    DEFAULT_SESSION_TMPFS_SIZE,
    MAX_HOST_TOOL_RESULT_BYTES,
    MAX_RECORDED_ARGUMENT_CHARS,
    MAX_REPORTED_FILES,
    MAX_REPORTED_OUTPUT_CHARS,
    MAX_SESSION_RESPONSE_BYTES,
    MAX_SESSION_VALUE_BYTES,
    MAX_TOOL_ERROR_CHARACTERS,
    MAX_WRITTEN_FILE_BYTES,
    SANDBOX_DATA_LAKE_DIR,
    SANDBOX_IMAGE_NAME,
    SANDBOX_PLATFORM,
    SANDBOX_USER_DATA_DIR,
    SANDBOX_WORK_DIR,
    SESSION_CLOSE_TIMEOUT,
    SESSION_GRACE_SECONDS,
    SESSION_START_TIMEOUT,
)
from virtual_lab.environment import sandbox_image, sandbox_image_exists
from virtual_lab.execution import (
    DockerExecutor,
    ExecutionError,
    LocalExecutor,
    OutputTail,
    kill_process_group,
    list_files,
    truncate_tail,
)
from virtual_lab.records import truncate_text
from virtual_lab.repair import describe_executor
from virtual_lab.resources import format_item_with_description
from virtual_lab.tools import Tool

KERNEL_SOURCE = (files("virtual_lab") / "kernel.py").read_text(encoding="utf-8")

LANGUAGES = {
    "python": "python",
    "python3": "python",
    "py": "python",
    "r": "r",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
}


# What the tool that runs code in a session is called
CODE_TOOL_NAME = "run_code"


class SessionError(ExecutionError):
    """Raised when a session cannot be started, or is used after it was closed."""


@dataclass(frozen=True)
class CellResult:
    """What happened when a piece of code was run in a session.

    :param language: "python", "r", or "bash".
    :param code: The code.
    :param status: "ok"; "error" if it raised or exited non-zero; "timeout" if it was stopped
        at its time limit; or "lost" if the session itself ended while running it, which loses
        every variable the session held.
    :param output: What it printed, standard output and standard error together, in order.
    :param error: The error, in a line or two, if there was one.
    :param duration: Seconds it ran.
    :param plots: Figures saved from it, relative to the session's directory.
    :param produced_files: Every file that appeared in the directory while it ran, plots included.
    :param output_dropped: Bytes of output dropped from the middle to bound what is kept.
    :param start: Which start of the session ran it. A number higher than the last one's means
        the session was restarted in between, and nothing defined before is still there.
    :param sandboxed: Whether it ran in a container.
    :param tool_calls: Each call it made to one of the session's tools: the tool, its arguments
        as JSON, how it ended ("ok", "error", or "running" if the code finished first), the
        error if there was one, and how many seconds it took.
    :param value: For code run with Session.evaluate, the value of its last expression, in
        JSON's types; None otherwise, or if the code failed.
    """

    language: str
    code: str
    status: str
    output: str
    error: str | None
    duration: float
    plots: tuple[str, ...] = ()
    produced_files: tuple[str, ...] = ()
    output_dropped: int = 0
    start: int = 1
    sandboxed: bool = True
    tool_calls: tuple[dict[str, Any], ...] = ()
    value: Any = None

    @property
    def succeeded(self) -> bool:
        """Whether the code ran to the end without error."""
        return self.status == "ok"

    def report(self, max_chars: int = MAX_REPORTED_OUTPUT_CHARS) -> str:
        """Describes the run in the form an agent is shown.

        :param max_chars: The most characters of output to include, from its end.
        :return: The description.
        """
        took = f"{self.duration:.1f} seconds"
        headline = {
            "ok": f"Ran in {took}.",
            "error": f"Failed after {took}: {self.error}",
            "timeout": f"{self.error} (after {took})",
            "lost": f"The session stopped while running this code: {self.error}",
        }.get(self.status, f"{self.status}: {self.error}")
        sections = [headline]

        others = [name for name in self.produced_files if name not in self.plots]
        for title, names in (("Figures saved", self.plots), ("Files written", others)):
            if names:
                listed = list(names[:MAX_REPORTED_FILES])
                if (unlisted := len(names) - len(listed)) > 0:
                    listed.append(f"... and {unlisted:,} more")
                sections.append(f"{title}:\n" + "\n".join(listed))

        sections.append(
            f"Output:\n{truncate_tail(self.output, max_chars)}" if self.output.strip() else "Output: none"
        )

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Converts the result to JSON-safe types for a record."""
        return {
            "language": self.language,
            "code": self.code,
            "status": self.status,
            "error": self.error,
            "duration": round(self.duration, 3),
            "plots": list(self.plots),
            "produced_files": list(self.produced_files),
            "output": self.output,
            "output_dropped": self.output_dropped,
            "start": self.start,
            "sandboxed": self.sandboxed,
            "tool_calls": [
                {**call, "duration": None if call["duration"] is None else round(call["duration"], 3)}
                for call in self.tool_calls
            ],
            "value": self.value,
        }


def read_responses(stream: IO[bytes], answers: queue.Queue) -> None:
    """Passes each line the interpreter writes to the queue, then None once it stops."""
    with stream:
        while line := stream.readline(MAX_SESSION_RESPONSE_BYTES):
            if not line.endswith(b"\n"):
                # Longer than any answer the interpreter writes, so it is not one
                answers.put(b"")
                break
            answers.put(line)
    answers.put(None)


def check_session_tools(tools: Iterable[Tool]) -> tuple[Tool, ...]:
    """Refuses tools that code could not call by name, or that two would share a name."""
    checked = tuple(tools)
    for tool in checked:
        if not isinstance(tool, Tool):
            raise TypeError(
                f"A session's tools are Tools, not {type(tool).__name__}: make one of a function with "
                "tool_from_function"
            )
        if not tool.name.isidentifier() or keyword.iskeyword(tool.name):
            raise ValueError(
                f"Code calls a session's tools by name, and {tool.name!r} is not a name Python can call: use "
                "letters, digits, and '_', not starting with a digit"
            )

    names = [tool.name for tool in checked]
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"A session's tools need names of their own: {', '.join(repeated)} is given twice")

    return checked


@dataclass(frozen=True)
class SessionData:
    """A file or directory given to a session's code to read.

    :param name: Its file name, as it was given, which it is found by.
    :param source: Where it is on this machine, with any symbolic link followed.
    :param description: What it is, as the agents are told.
    """

    name: str
    source: Path
    description: str


def check_description(description: Any, what: str) -> str:
    if not isinstance(description, str):
        raise TypeError(f"What {what} is must be said in a str, not {type(description).__name__}")
    if not description.strip():
        raise ValueError(f"Say what {what} is: its description is empty")

    return description.strip()


def check_session_data(data: Mapping[str | os.PathLike[str], str] | None) -> tuple[SessionData, ...]:
    """Reads a session's data as Biomni's add_data takes it: each path, with what is there."""
    if data is None:
        return ()
    if not isinstance(data, Mapping):
        raise TypeError(
            "A session's data is a dict of each file or directory's path to what it is, such as "
            f"{{'expression.csv': 'Gene expression of ...'}}, not {type(data).__name__}"
        )

    checked = []
    for path, description in data.items():
        if not isinstance(path, str | os.PathLike):
            raise TypeError(f"A session's data is given by its path, not {type(path).__name__}")
        # Named as it was given, so that a link called latest.csv is found as latest.csv, not
        # under the name of the file it points to
        given = Path(os.path.abspath(Path(path).expanduser()))
        source = given.resolve()
        if not source.exists():
            raise FileNotFoundError(f"The session's data {str(path)!r} is not there: {source} does not exist")
        if not given.name:
            raise ValueError(f"{given} cannot be a session's data: give the files or directories in it instead")
        checked.append(SessionData(given.name, source, check_description(description, f"the data {given}")))

    names = [item.name for item in checked]
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(
            f"A session's data is found by its file name, and {', '.join(repeated)} names more than one of "
            "it: give each a name of its own, or the directories holding them instead"
        )

    return tuple(checked)


def check_session_software(software: Mapping[str, str] | None) -> dict[str, str]:
    """Reads a session's software as Biomni's add_software takes it: each name, with what it does."""
    if software is None:
        return {}
    if not isinstance(software, Mapping):
        raise TypeError(
            "A session's software is a dict of each library or program's name to what it does, such as "
            f"{{'pydeseq2': 'Differential expression with DESeq2'}}, not {type(software).__name__}"
        )

    checked = {}
    for name, description in software.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"A session's software is given by its name, not {name!r}")
        checked[name.strip()] = check_description(description, f"the software {name.strip()}")

    if len(checked) < len(software):
        raise ValueError("A session's software is given twice under the same name")

    return checked


def call_parameters(tool: Tool) -> list[dict[str, Any]]:
    """How code calls a tool: its parameters in the order its schema lists them.

    Each can be passed by position until a required parameter follows an optional one, which
    Python cannot express, so from there on each must be passed by name. A parameter that is
    optional is left out of the call unless given, so the tool applies its own default.
    """
    schema = tool.parameters if isinstance(tool.parameters, dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required") or [])

    parameters = []
    keyword_only = optional_seen = False
    for name, item in properties.items():
        is_required = name in required
        keyword_only = keyword_only or (is_required and optional_seen)
        optional_seen = optional_seen or not is_required
        parameter: dict[str, Any] = {"name": name, "required": is_required, "keyword_only": keyword_only}
        if not is_required and isinstance(item, dict) and "default" in item:
            parameter["default"] = repr(item["default"])
        parameters.append(parameter)

    return parameters


def call_signature(tool: Tool) -> str:
    """The tool's signature as code calls it, such as "(gene, limit=10, *, organism)"."""
    shown = []
    marked = False
    for parameter in call_parameters(tool):
        if parameter["keyword_only"] and not marked:
            shown.append("*")
            marked = True
        name = parameter["name"]
        shown.append(name if parameter["required"] else f"{name}={parameter.get('default', '...')}")

    return f"({', '.join(shown)})"


def describe_type(schema: Any) -> str:
    """Says in a few words what a JSON Schema accepts."""
    if not isinstance(schema, dict) or not schema:
        return "any"
    if "enum" in schema:
        return "one of " + ", ".join(json.dumps(value) for value in schema["enum"])
    for options in ("anyOf", "oneOf"):
        if isinstance(schema.get(options), list):
            return " or ".join(dict.fromkeys(describe_type(option) for option in schema[options]))
    kind = schema.get("type")
    if isinstance(kind, list):
        return " or ".join(str(item) for item in kind)
    if kind == "array" and isinstance(schema.get("items"), dict):
        return f"array of {describe_type(schema['items'])}"
    if kind:
        return str(kind)
    if isinstance(schema.get("$ref"), str):
        return schema["$ref"].rsplit("/", 1)[-1]

    return "object" if "properties" in schema else "any"


def session_tools_prompt(tools: tuple[Tool, ...]) -> str:
    """Tells a meeting which functions code in its session can call that run outside it.

    They are listed first and in full, as Biomni lists the tools added to its agent under its
    priority custom resources, since they were added for this work.

    :param tools: The session's tools.
    :return: The prompt, or an empty string if there are none.
    """
    if not tools:
        return ""

    entries = []
    for tool in tools:
        lines = [f"{tool.name}{call_signature(tool)}"]
        lines.extend(f"  {line}" if line.strip() else "" for line in tool.description.strip().splitlines())
        properties = tool.parameters.get("properties") or {}
        required = set(tool.parameters.get("required") or [])
        for name, item in properties.items():
            item = item if isinstance(item, dict) else {}
            detail = f"{describe_type(item)}, {'required' if name in required else 'optional'}"
            note = f": {item['description']}" if item.get("description") else ""
            default = f" [Default: {json.dumps(item['default'])}]" if "default" in item else ""
            lines.append(f"    - {name} ({detail}){note}{default}")
        entries.append("\n".join(lines))

    return (
        "- Functions added for this work, already defined in the session. Prefer them where they "
        "fit. Call them from Python code like any other function, without importing them; R and "
        "bash code cannot call them. They run outside the session, on the machine holding the "
        "meeting, so they can reach what the session's code cannot, and what they return is "
        "handed back to the code. Their arguments and results are passed as JSON, so pass "
        "numbers, strings, lists, and dicts, and a function that fails raises HostToolError.\n"
        "----\n" + "\n\n".join(entries) + "\n----"
    )


def own_resources_prompt(session: "Session", software_not_found: Iterable[str] = ()) -> str:
    """Tells a meeting what was added to its session for this work: tools, data, and software.

    They are listed before Biomni's resources, as Biomni lists what is added to its agent under
    its priority custom resources. Unlike Biomni, which lists a file it was given by name alone,
    each piece of data is listed by the path the code reads it at.

    :param session: The session.
    :param software_not_found: Software the session was found not to have, which is listed
        anyway, since it was added, but marked as not found.
    :return: The prompt, or an empty string if nothing was added.
    """
    sections = [session_tools_prompt(session.tools)] if session.tools else []

    if session.data:
        changed = " but not change" if session.sandboxed else ""
        items = "\n".join(
            format_item_with_description(session.data_path(item), item.description) for item in session.data
        )
        sections.append(
            f"- Data added for this work, which the session's code can read{changed}. Prefer it to "
            "other data where it fits. Each is listed by the path code reads it at, with what it is.\n"
            f"----\n{items}\n----"
        )

    if session.software:
        missing = set(software_not_found)
        items = "\n".join(
            format_item_with_description(
                name, f"{description} (not found in the session: check that it is there before relying on it)"
                if name in missing else description
            )
            for name, description in session.software.items()
        )
        sections.append(
            "- Software added for this work, for the session's code to use. Prefer it where it fits. "
            f"Each is listed with what it does.\n----\n{items}\n----"
        )

    return "\n\n".join(sections)


def jsonable(value: Any) -> Any:
    """Converts what a tool returned to JSON's types, writing out what JSON has none for as text."""
    try:
        return to_jsonable_python(value, fallback=str)
    except Exception:
        return str(value)


def stop_process(process: subprocess.Popen, stop: Callable[[subprocess.Popen], None]) -> None:
    """Stops an interpreter and whatever it started, then waits for it."""
    try:
        stop(process)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


class Session:
    """A Python interpreter that keeps its variables from one piece of code to the next.

    Use DockerSession or LocalSession. One session is meant to be shared by everyone in a
    meeting, so that what one agent computes, the others can use. Code is run one piece at a
    time, in the order it arrives.

    :param directory: The directory the code runs in, created if missing. Files the code writes
        there, figures included, stay after the session ends.
    :param timeout: Seconds each piece of code may run, unless it is given its own limit.
    :param start_timeout: Seconds to wait for the interpreter to start.
    :param tools: Tools for code in the session to call by name, as functions it need not
        import, which run here rather than in the session; see tool_from_function and
        connect_mcp. A call
        takes up the time of the code that made it, and one still running when that code
        stops at its limit is left to finish, its result unused.
    :param data: Files and directories for the code to read, each by its path here, with what
        it is, as Biomni's add_data takes them: {"counts.csv": "Read counts of ..."}. Each is
        found by its file name, so no two may share one.
    :param software: Libraries and programs the code can use, each by its name, with what it
        does, as Biomni's add_software takes them. They must be installed where the code runs;
        a meeting checks that they are, and warns of any it cannot find.
    """

    sandboxed = True

    def __init__(
        self,
        directory: Path,
        timeout: float = DEFAULT_EXECUTION_TIMEOUT,
        start_timeout: float = SESSION_START_TIMEOUT,
        tools: Iterable[Tool] = (),
        data: Mapping[str | os.PathLike[str], str] | None = None,
        software: Mapping[str, str] | None = None,
    ) -> None:
        self.tools = check_session_tools(tools)
        self.data = check_session_data(data)
        self.software = check_session_software(software)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.history: list[CellResult] = []
        self.starts = 0
        self.python_version: str | None = None
        self._process: subprocess.Popen | None = None
        self._answers: queue.Queue = queue.Queue()
        self._errors: OutputTail | None = None
        self._stop: Callable[[subprocess.Popen], None] = lambda process: None
        self._finalizer: weakref.finalize | None = None
        self._requests = 0
        self._lock = threading.Lock()
        # Replies to tools' calls are written from the threads that run them
        self._write_lock = threading.Lock()
        self._calls_lock = threading.Lock()
        self._tool_threads: set[int] = set()
        self._closed = False

    def spawn(self) -> tuple[list[str], dict[str, Any], Callable[[subprocess.Popen], None]]:
        """How to start the interpreter: its command, the options to start it with, and how to
        stop it and everything it started. The last must not refer to the session, which a
        finalizer holding it would then keep alive."""
        raise NotImplementedError

    @property
    def running(self) -> bool:
        """Whether the interpreter is running now."""
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Starts the interpreter if it is not running.

        :raises SessionError: If the session was closed, or the interpreter does not start.
        """
        if self._closed:
            raise SessionError("This session was closed. Start a new one.")
        if self.running:
            return

        self.shut_down()
        command, options, stop = self.spawn()

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **options,
            )
        except OSError as error:
            raise SessionError(f"Could not start the session's interpreter: {error}") from error

        self._process, self._stop = process, stop
        self._answers = queue.Queue()
        self._errors = OutputTail(process.stderr)  # type: ignore[arg-type]
        threading.Thread(
            target=read_responses, args=(process.stdout, self._answers), daemon=True
        ).start()
        # A session nobody closed must not leave its interpreter, or its container, running
        self._finalizer = weakref.finalize(self, stop_process, process, stop)
        self.starts += 1

        greeting = self.next_answer(time.monotonic() + self.start_timeout)
        if not isinstance(greeting, dict) or not greeting.get("ready"):
            detail = self.why_it_stopped() if greeting is None else "it did not answer"
            self.shut_down()
            raise SessionError(f"The session's interpreter did not start: {detail}")

        self.python_version = str(greeting.get("python"))

        if self.tools:
            self.define_tools()

    def define_tools(self) -> None:
        """Defines the session's tools in the interpreter, which it forgets when restarted."""
        self._requests += 1
        request = {
            "id": self._requests,
            "tools": [
                {"name": tool.name, "description": tool.description, "parameters": call_parameters(tool)}
                for tool in self.tools
            ],
            "max_call_bytes": MAX_SESSION_RESPONSE_BYTES,
        }
        deadline = time.monotonic() + self.start_timeout
        answer: dict | None | bool = None
        if self.write(self._process, json.dumps(request)):  # type: ignore[arg-type]
            answer = self.next_answer(deadline)
            while isinstance(answer, dict) and answer.get("id") != request["id"]:
                answer = self.next_answer(deadline)

        if not (isinstance(answer, dict) and answer.get("status") == "ok"):
            if isinstance(answer, dict):
                detail = str(answer.get("error"))
            elif answer is False:
                detail = "it did not answer"
            else:
                detail = self.why_it_stopped()
            self.shut_down()
            raise SessionError(f"The session's tools could not be defined in its interpreter: {detail}")

    def write(self, process: subprocess.Popen, line: str) -> bool:
        """Writes a line to an interpreter, returning whether it could be."""
        with self._write_lock:
            try:
                process.stdin.write((line + "\n").encode("utf-8"))  # type: ignore[union-attr]
                process.stdin.flush()  # type: ignore[union-attr]
            except (OSError, ValueError):
                return False

        return True

    def answer_call(self, call: dict, request: int, calls: list[dict[str, Any]], finished: threading.Event) -> None:
        """Runs a tool that code called, in a thread of its own, which writes back its reply.

        The thread lets the code's time limit stand: if the code is stopped while the tool runs,
        the session is answered and moves on, and the tool's reply, when it comes, is dropped.
        Whatever the tool asks a person is withdrawn once finished is set, when the code has
        finished.
        """
        name = call.get("tool")
        arguments = call.get("arguments")
        tool = next((tool for tool in self.tools if tool.name == name), None)
        record: dict[str, Any] = {
            "tool": str(name),
            "arguments": truncate_text(json.dumps(arguments), MAX_RECORDED_ARGUMENT_CHARS),
            "status": "running",
            "error": None,
            "duration": None,
        }
        with self._calls_lock:
            calls.append(record)

        problem = None
        if call.get("request") != request:
            problem = "the code that called it had finished"
        elif tool is None:
            problem = f"the session has no tool called {truncate_text(str(name), 100)}"
        elif not isinstance(arguments, dict):
            problem = "its arguments were not given by name"

        process = self._process
        if problem is not None or tool is None or not isinstance(arguments, dict):
            with self._calls_lock:
                record.update(status="error", error=problem, duration=0.0)
            self.write(process, json.dumps({"reply": call.get("call"), "error": problem}))  # type: ignore[arg-type]
            return

        threading.Thread(
            target=self.run_call,
            args=(process, call.get("call"), tool, arguments, record, finished),
            name=f"virtual-lab-tool-{tool.name}",
            daemon=True,
        ).start()

    def run_call(
        self,
        process: subprocess.Popen,
        call: Any,
        tool: Tool,
        arguments: dict,
        record: dict[str, Any],
        finished: threading.Event,
    ) -> None:
        started = time.monotonic()
        error = None
        self._tool_threads.add(threading.get_ident())
        try:
            with asked_for(finished):
                value = tool.function(**arguments)
            line = json.dumps({"reply": call, "result": jsonable(value)})
            if len(line) > MAX_HOST_TOOL_RESULT_BYTES:
                error = (
                    f"it returned {len(line):,} bytes of JSON, more than the {MAX_HOST_TOOL_RESULT_BYTES:,} a "
                    "call can return. Have it write what it found to a file the session can read, and "
                    "return the file's path."
                )
        except BaseException as exception:
            error = truncate_text(f"{type(exception).__name__}: {exception}", MAX_TOOL_ERROR_CHARACTERS)
        finally:
            self._tool_threads.discard(threading.get_ident())

        if error is not None:
            line = json.dumps({"reply": call, "error": error})
        with self._calls_lock:
            record.update(
                status="ok" if error is None else "error", error=error, duration=time.monotonic() - started
            )
        # To the interpreter that made the call, which may since have been replaced
        self.write(process, line)

    def next_answer(self, deadline: float) -> dict | None | bool:
        """Waits for the interpreter's next answer.

        :return: The answer; None if the interpreter has stopped or wrote something that is not
            an answer; or False if the deadline passed first.
        """
        try:
            line = self._answers.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            return False

        if not line:
            return None

        try:
            answer = json.loads(line)
        except ValueError:
            return None

        return answer if isinstance(answer, dict) else None

    def why_it_stopped(self) -> str:
        """Explains, as far as can be told, why the interpreter is gone."""
        process = self._process
        code = None
        if process is not None:
            try:
                code = process.wait(timeout=SESSION_CLOSE_TIMEOUT)
            except subprocess.TimeoutExpired:
                code = None

        if code is None:
            reason = "it stopped answering"
        elif code in (137, -signal.SIGKILL):
            reason = f"it was killed (exit code {code}), most likely for running out of memory"
        else:
            reason = f"it exited with code {code}"

        errors = self._errors.text(wait=1).strip() if self._errors is not None else ""

        return f"{reason}. {truncate_tail(errors, 1_000)}".strip() if errors else f"{reason}."

    def refuse_from_tool(self) -> None:
        """Refuses to run code for one of the session's own tools, which a call from the session's
        code is running: the code waits on the tool, so the tool would wait on the code."""
        if threading.get_ident() in self._tool_threads:
            raise SessionError("A session's tool cannot use the session whose code called it")

    def run(self, code: str, language: str = "python", timeout: float | None = None) -> CellResult:
        """Runs a piece of code in the session.

        :param code: The code.
        :param language: "python", whose variables persist; or "r" or "bash", run as a fresh
            process in the same directory.
        :param timeout: Seconds to allow, defaulting to the session's limit.
        :raises ValueError: If the language is not one of those.
        :raises SessionError: If the session was closed or cannot be started.
        :return: What happened. Code that fails, or runs out of time, is a result, not an error.
        """
        normalized = LANGUAGES.get(str(language).strip().casefold())
        if normalized is None:
            raise ValueError(f"Cannot run {language!r} code: use python, r, or bash.")

        self.refuse_from_tool()
        with self._lock:
            result = self.execute(code, normalized, self.timeout if timeout is None else timeout)
            self.history.append(result)

            return result

    def evaluate(self, code: str, timeout: float | None = None) -> CellResult:
        """Runs Python code in the session, as run does, and sends back the value of its last
        line, if that is an expression, rather than printing it.

        :param code: The code.
        :param timeout: Seconds to allow, defaulting to the session's limit.
        :raises SessionError: If the session was closed or cannot be started.
        :return: What happened, with the value in its value, in JSON's types. Numpy arrays and
            scalars, sets, and paths are converted; a value that is otherwise not JSON, or is more
            than MAX_SESSION_VALUE_BYTES of it, fails the code, though what it did is done.
        """
        self.refuse_from_tool()
        with self._lock:
            result = self.execute(code, "python", self.timeout if timeout is None else timeout, value=True)
            self.history.append(result)

            return result

    def check(self, code: str, timeout: float | None = None) -> CellResult:
        """Runs Python code that asks about the session rather than taking part in its analysis,
        such as what is installed, and keeps it out of the history.

        The code shares the interpreter with the analysis, so it should leave nothing behind.

        :param code: The code.
        :param timeout: Seconds to allow, defaulting to the session's limit.
        :raises SessionError: If the session was closed or cannot be started.
        :return: What happened.
        """
        self.refuse_from_tool()
        with self._lock:
            return self.execute(code, "python", self.timeout if timeout is None else timeout)

    def execute(self, code: str, normalized: str, limit: float, value: bool = False) -> CellResult:
        """Runs a piece of code without recording it, sending back the value of its last
        expression if asked. The caller holds the lock."""
        self.start()
        before = list_files(self.directory)
        started = time.monotonic()
        self._requests += 1
        request = {"id": self._requests, "language": normalized, "code": code, "timeout": limit}
        if value:
            request["value"] = MAX_SESSION_VALUE_BYTES
        answer: dict | None | bool = None
        calls: list[dict[str, Any]] = []
        # Set once the code has finished, so that a call it made that is still waiting for a
        # person's approval is not made after all
        finished = threading.Event()

        try:
            if self.write(self._process, json.dumps(request)):  # type: ignore[arg-type]
                deadline = started + limit + SESSION_GRACE_SECONDS
                answer = self.next_answer(deadline)
                # A call to a tool is answered, and an answer to some other request is not this
                # one's, and is not a reason to give up on the session while this one's may still
                # come
                while isinstance(answer, dict) and answer.get("id") != request["id"]:
                    if "call" in answer:
                        self.answer_call(answer, request["id"], calls, finished)
                    answer = self.next_answer(deadline)
        finally:
            finished.set()

        if isinstance(answer, dict) and answer.get("id") == request["id"]:
            result = self.result_from(answer, normalized, code, time.monotonic() - started)
        else:
            if answer is False:
                reason = (
                    f"the code did not stop at its time limit of {limit:g} seconds, so the "
                    "session was stopped."
                )
            else:
                reason = self.why_it_stopped()
            self.shut_down()
            result = CellResult(
                language=normalized,
                code=code,
                status="lost",
                output="",
                error=f"{reason} Every variable it held is gone; the next code will run in a "
                "fresh session.",
                duration=time.monotonic() - started,
                start=self.starts,
                sandboxed=self.sandboxed,
            )

        with self._calls_lock:
            tool_calls = tuple(dict(call) for call in calls)

        return replace(
            result, produced_files=tuple(sorted(list_files(self.directory) - before)), tool_calls=tool_calls
        )

    def result_from(self, answer: dict, language: str, code: str, duration: float) -> CellResult:
        """Reads the interpreter's answer, trusting none of its types."""
        plots = answer.get("plots")

        return CellResult(
            language=language,
            code=code,
            status=str(answer.get("status", "error")),
            output=str(answer.get("output", "")),
            error=None if answer.get("error") is None else str(answer["error"]),
            duration=duration,
            plots=tuple(str(plot) for plot in plots) if isinstance(plots, list) else (),
            output_dropped=int(answer.get("output_dropped") or 0),
            start=self.starts,
            sandboxed=self.sandboxed,
            value=answer.get("value"),
        )

    def restart(self) -> None:
        """Starts the interpreter afresh, discarding every variable it held."""
        self.refuse_from_tool()
        with self._lock:
            self.shut_down()
            self.start()

    def shut_down(self) -> None:
        """Stops the interpreter, if it is running, without closing the session."""
        process, self._process = self._process, None
        if process is None:
            return

        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None

        try:
            process.stdin.close()  # type: ignore[union-attr]
            process.wait(timeout=SESSION_CLOSE_TIMEOUT if process.poll() is None else 0)
        except (OSError, subprocess.TimeoutExpired):
            pass

        stop_process(process, self._stop)

    def close(self) -> None:
        """Stops the interpreter for good. Files it wrote are kept."""
        self.refuse_from_tool()
        with self._lock:
            self.shut_down()
            self._closed = True

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def describe(self) -> dict[str, Any]:
        """Describes how the session ran code, for a record."""
        return {
            "type": type(self).__name__,
            "sandboxed": self.sandboxed,
            "directory": str(self.directory),
            "timeout": self.timeout,
            "python_version": self.python_version,
            "starts": self.starts,
            "cells": len(self.history),
            "tools": [tool.name for tool in self.tools],
            "data": [
                {
                    "name": item.name,
                    "source": str(item.source),
                    "path": self.data_path(item),
                    "description": item.description,
                }
                for item in self.data
            ],
            "software": dict(self.software),
        }

    def where_code_runs(self) -> str:
        """The working directory as the code sees it."""
        return str(self.directory)

    def data_path(self, item: SessionData) -> str:
        """Where a piece of the session's data is, as the code sees it."""
        return str(item.source)

    def can_reach_network(self) -> bool:
        """Whether the code can reach the internet."""
        return True

    def data_lake_path(self) -> str | None:
        """Where Biomni's data lake is, as the code sees it, or None if it is not there."""
        return None

    def data_lake_files(self) -> list[str]:
        """The files in Biomni's data lake, by name, or none if it is not there."""
        return []

    def has_biomni_tools(self) -> bool:
        """Whether code can import Biomni's tools."""
        return False


def list_data_lake(directory: Path | None) -> list[str]:
    """The files in a data lake directory, leaving out downloads that did not finish."""
    if directory is None or not Path(directory).is_dir():
        return []

    return sorted(
        path.name for path in Path(directory).iterdir() if path.is_file() and not path.name.startswith(".")
    )


def session_executor(
    target: str = "full",
    data_lake: Path | None = None,
    allow_network: bool = True,
    timeout: float = DEFAULT_EXECUTION_TIMEOUT,
    biomni_tools: bool = True,
    forward_env: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
) -> DockerExecutor:
    """The container a session runs in by default: Biomni's environment, with the network.

    Biomni's agent works on the open internet, querying databases as it goes, so the network is
    on unless asked otherwise. Everything else stays shut: the host's filesystem beyond the
    session's directory and the data lake, its environment, and its API keys.

    :param target: Which stage of the sandbox image to use; see build_sandbox_image.
    :param data_lake: A directory of Biomni's data lake to mount read-only, if any.
    :param allow_network: Whether code can reach the network.
    :param timeout: Seconds each piece of code may run.
    :param biomni_tools: Whether code can import Biomni's tools, as Biomni's agent does.
    :param forward_env: Host environment variables to pass in, by name. About forty of Biomni's
        tools call a model, most of them the database tools that turn a question into a query,
        and need its API key: ANTHROPIC_API_KEY for Biomni's default model. Code can read what
        is passed, and send it anywhere over the network.
    :param environment: Variables to set, such as BIOMNI_LLM to choose the model Biomni's tools
        call. They are recorded, so they are not for secrets.
    :return: The executor, to pass to DockerSession.
    """
    return DockerExecutor(
        image=sandbox_image(target),
        platform=SANDBOX_PLATFORM,
        allow_network=allow_network,
        data_lake=data_lake,
        timeout=timeout,
        memory_limit=DEFAULT_SESSION_MEMORY_LIMIT,
        cpu_limit=DEFAULT_SESSION_CPU_LIMIT,
        pids_limit=DEFAULT_SESSION_PIDS_LIMIT,
        tmpfs_size=DEFAULT_SESSION_TMPFS_SIZE,
        biomni_tools=biomni_tools,
        forward_env=tuple(forward_env),
        environment=dict(environment or {}),
    )


class DockerSession(Session):
    """A session whose interpreter runs in a container.

    The container is the one DockerExecutor describes: a read-only root filesystem, no host
    environment, no capabilities, ceilings on memory, processes, and file size, and only the
    session's directory, and the data lake if one is given, visible from the host.

    :param directory: The directory the code runs in, mounted at /workspace.
    :param executor: The container's settings, defaulting to session_executor(): Biomni's full
        environment, with the network on.
    :param timeout: Seconds each piece of code may run, defaulting to the executor's.
    :param start_timeout: Seconds to wait for the interpreter to start.
    :param tools: Tools for code in the session to call, which run here, outside the container,
        with this machine's network, files, and keys; see Session.
    :param data: Files and directories for the code to read, as Session takes them, each
        mounted read-only at /data/ and its file name. None may be in the
        session's directory, where the code could change it, or hold it.
    :param software: Software installed in the image, for the code to use; see Session.
    """

    def __init__(
        self,
        directory: Path,
        executor: DockerExecutor | None = None,
        timeout: float | None = None,
        start_timeout: float = SESSION_START_TIMEOUT,
        tools: Iterable[Tool] = (),
        data: Mapping[str | os.PathLike[str], str] | None = None,
        software: Mapping[str, str] | None = None,
    ) -> None:
        self.executor = executor if executor is not None else session_executor()
        super().__init__(
            directory,
            timeout=self.executor.timeout if timeout is None else timeout,
            start_timeout=start_timeout,
            tools=tools,
            data=data,
            software=software,
        )
        # Refused now rather than when the session first starts, in the middle of a meeting
        self.executor.mount_arguments(self.directory, self.mounts())

    def mounts(self) -> list[tuple[Path, str]]:
        """The session's data, each with where it is mounted."""
        return [(item.source, self.data_path(item)) for item in self.data]

    def spawn(self) -> tuple[list[str], dict[str, Any], Callable[[subprocess.Popen], None]]:
        self.executor.check_available()

        image = self.executor.image
        if image.startswith(f"{SANDBOX_IMAGE_NAME}:") and not sandbox_image_exists(
            image, self.executor.docker_command
        ):
            raise SessionError(
                f"The sandbox image {image} has not been built. Build it with "
                'virtual_lab.build_sandbox_image, choosing the stage: "full" takes hours, '
                '"bio" about half an hour, and "base" minutes. Then pass '
                "DockerSession(executor=session_executor(target)) for the stage you built."
            )

        name = f"virtual-lab-session-{uuid4().hex[:12]}"
        command = self.executor.build_command(
            directory=self.directory,
            command=("python3", "-u", "-c", KERNEL_SOURCE),
            container_name=name,
            interactive=True,
            mounts=self.mounts(),
        )

        kill = self.executor.kill

        # Killing the docker client leaves the container running, so it is stopped by name
        return list(command), {}, lambda process: kill(name)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "executor": describe_executor(self.executor)}

    def where_code_runs(self) -> str:
        return SANDBOX_WORK_DIR

    def data_path(self, item: SessionData) -> str:
        return f"{SANDBOX_USER_DATA_DIR}/{item.name}"

    def can_reach_network(self) -> bool:
        return self.executor.allow_network

    def data_lake_path(self) -> str | None:
        return SANDBOX_DATA_LAKE_DIR if self.executor.data_lake is not None else None

    def data_lake_files(self) -> list[str]:
        return list_data_lake(self.executor.data_lake)

    def has_biomni_tools(self) -> bool:
        return self.executor.biomni_tools


class LocalSession(Session):
    """A session whose interpreter runs directly on this machine, with no isolation whatsoever.

    Like LocalExecutor, this is not a sandbox: the code can read and write anything you can and
    reach the network. What it has is a scrubbed environment, so the code does not inherit your
    API keys, a cap on the size of any file it writes, and its own process group, killed when the
    session ends. Use it where Docker is not available, or to run in an environment already
    installed here, such as Biomni's own biomni_e1.

    :param directory: The directory the code runs in.
    :param python: The interpreter to run, defaulting to this one.
    :param timeout: Seconds each piece of code may run.
    :param warn: Whether to warn that nothing is isolated.
    :param start_timeout: Seconds to wait for the interpreter to start.
    :param max_file_bytes: The largest file the code may write.
    :param biomni_tools: Whether code can import Biomni's tools. They need the libraries of
        Biomni's environment, so pass its interpreter as python.
    :param forward_env: Environment variables to pass through, by name, such as the API keys
        Biomni's tools use.
    :param environment: Variables to set, such as BIOMNI_LLM. They are recorded, so they are not
        for secrets.
    :param tools: Tools for code in the session to call, which run in this process; see Session.
    :param data: Files and directories for the code to read, where they are; see Session.
    :param software: Software installed for python, for the code to use; see Session.
    """

    sandboxed = False

    def __init__(
        self,
        directory: Path,
        python: str | None = None,
        timeout: float = DEFAULT_EXECUTION_TIMEOUT,
        warn: bool = True,
        start_timeout: float = SESSION_START_TIMEOUT,
        max_file_bytes: int = MAX_WRITTEN_FILE_BYTES,
        biomni_tools: bool = False,
        forward_env: tuple[str, ...] = (),
        environment: dict[str, str] | None = None,
        tools: Iterable[Tool] = (),
        data: Mapping[str | os.PathLike[str], str] | None = None,
        software: Mapping[str, str] | None = None,
    ) -> None:
        if warn:
            warnings.warn(
                "LocalSession runs model-authored code on this machine with no isolation. It "
                "can read and write any file you can. Prefer DockerSession.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(
            directory, timeout=timeout, start_timeout=start_timeout, tools=tools, data=data, software=software
        )
        self.python = python or sys.executable
        self.max_file_bytes = max_file_bytes
        self.biomni_tools = biomni_tools
        self.forward_env = tuple(forward_env)
        self.environment = dict(environment or {})

    def spawn(self) -> tuple[list[str], dict[str, Any], Callable[[subprocess.Popen], None]]:
        limits = LocalExecutor(
            warn=False,
            max_file_bytes=self.max_file_bytes,
            biomni_tools=self.biomni_tools,
            forward_env=self.forward_env,
            environment=self.environment,
        )
        options: dict[str, Any] = {
            "cwd": self.directory,
            "env": limits.build_environment(),
            "start_new_session": hasattr(os, "killpg"),
            "preexec_fn": limits.limit_file_size(),
        }

        return [self.python, "-u", "-c", KERNEL_SOURCE], options, kill_process_group

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "python": self.python,
            "biomni_tools": self.biomni_tools,
            "forward_env": list(self.forward_env),
            "environment": dict(self.environment),
        }

    def has_biomni_tools(self) -> bool:
        return self.biomni_tools


def session_notes(session: Session) -> list[str]:
    """What the tool that runs code in a session says of it, after saying who shares it."""
    data_lake = session.data_lake_path()

    notes = [
        "R and bash code runs as a fresh process each time, in the same directory, and can hand "
        "results to Python through files.",
        f"The working directory is {session.where_code_runs()}; files written there are kept. "
        "A matplotlib figure left open is saved to plots/ and reported back.",
        "The value of a final expression is printed, as in a notebook. Keep output short: "
        "print summaries, shapes, and heads rather than whole tables.",
        "Code can reach the internet." if session.can_reach_network() else "Code has no network access.",
    ]
    if data_lake is not None:
        notes.append(
            f"Biomni's data lake is mounted read-only at {data_lake}, also in the "
            "BIOMNI_DATA_LAKE environment variable."
        )

    return notes


def session_tool(session: Session) -> Tool:
    """Builds the tool that lets an agent run code in a session.

    :param session: The session, shared by everyone the tool is given to.
    :return: The tool.
    """
    notes = [
        "Run code in the meeting's shared interpreter and see what it prints. Python runs in "
        "one session that keeps its variables, imports, and loaded data from one call to the "
        "next, for every member of the meeting: what another agent defined is there for you, "
        "so check before loading something again.",
        *session_notes(session),
    ]

    def run_code(code: str, language: str = "python") -> str:
        return session.run(code, language=language).report()

    return Tool(
        name=CODE_TOOL_NAME,
        description=" ".join(notes),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The code to run."},
                "language": {
                    "type": "string",
                    "enum": ["python", "r", "bash"],
                    "description": "The language of the code. Defaults to python.",
                },
            },
            "required": ["code"],
        },
        function=run_code,
    )
