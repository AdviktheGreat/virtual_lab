"""A running interpreter that a meeting's agents share, the way Biomni's agent has its REPL.

Running a finished script tells a meeting whether the script works. An analysis is not written
that way: it loads the data once, looks at it, and decides what to do next from what it saw.
A session keeps a Python interpreter running between pieces of code, so a table loaded by one
agent is still there for the next, and figures are saved as they are drawn. R and shell code
can be run in it too, each as a fresh process, as Biomni runs them.

The interpreter is virtual_lab/kernel.py, started in the sandbox with "python3 -c", so the
image needs nothing but a Python 3. DockerSession runs it under every restriction
DockerExecutor applies; LocalSession runs it on this machine with none of them.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

from virtual_lab.constants import (
    DEFAULT_EXECUTION_TIMEOUT,
    DEFAULT_SESSION_CPU_LIMIT,
    DEFAULT_SESSION_MEMORY_LIMIT,
    DEFAULT_SESSION_PIDS_LIMIT,
    DEFAULT_SESSION_TMPFS_SIZE,
    MAX_REPORTED_FILES,
    MAX_REPORTED_OUTPUT_CHARS,
    MAX_SESSION_RESPONSE_BYTES,
    MAX_WRITTEN_FILE_BYTES,
    SANDBOX_DATA_LAKE_DIR,
    SANDBOX_IMAGE_NAME,
    SANDBOX_PLATFORM,
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
from virtual_lab.repair import describe_executor
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
    """

    sandboxed = True

    def __init__(
        self,
        directory: Path,
        timeout: float = DEFAULT_EXECUTION_TIMEOUT,
        start_timeout: float = SESSION_START_TIMEOUT,
    ) -> None:
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

        limit = self.timeout if timeout is None else timeout

        with self._lock:
            self.start()
            before = list_files(self.directory)
            started = time.monotonic()
            self._requests += 1
            request = {"id": self._requests, "language": normalized, "code": code, "timeout": limit}
            answer: dict | None | bool

            try:
                self._process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))  # type: ignore[union-attr]
                self._process.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, OSError):
                answer = None
            else:
                deadline = started + limit + SESSION_GRACE_SECONDS
                answer = self.next_answer(deadline)
                # An answer to some other request is not this one's, and is not a reason to
                # give up on the session while this one's may still come
                while isinstance(answer, dict) and answer.get("id") != request["id"]:
                    answer = self.next_answer(deadline)

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

            result = replace(
                result, produced_files=tuple(sorted(list_files(self.directory) - before))
            )
            self.history.append(result)

            return result

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
        )

    def restart(self) -> None:
        """Starts the interpreter afresh, discarding every variable it held."""
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
        }

    def where_code_runs(self) -> str:
        """The working directory as the code sees it."""
        return str(self.directory)

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
    """

    def __init__(
        self,
        directory: Path,
        executor: DockerExecutor | None = None,
        timeout: float | None = None,
        start_timeout: float = SESSION_START_TIMEOUT,
    ) -> None:
        self.executor = executor if executor is not None else session_executor()
        super().__init__(
            directory,
            timeout=self.executor.timeout if timeout is None else timeout,
            start_timeout=start_timeout,
        )

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
        )

        kill = self.executor.kill

        # Killing the docker client leaves the container running, so it is stopped by name
        return list(command), {}, lambda process: kill(name)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "executor": describe_executor(self.executor)}

    def where_code_runs(self) -> str:
        return SANDBOX_WORK_DIR

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
    :param environment: Variables to set, such as BIOMNI_LLM.
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
    ) -> None:
        if warn:
            warnings.warn(
                "LocalSession runs model-authored code on this machine with no isolation. It "
                "can read and write any file you can. Prefer DockerSession.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(directory, timeout=timeout, start_timeout=start_timeout)
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


def session_tool(session: Session) -> Tool:
    """Builds the tool that lets an agent run code in a session.

    :param session: The session, shared by everyone the tool is given to.
    :return: The tool.
    """
    data_lake = session.data_lake_path()

    notes = [
        "Run code in the meeting's shared interpreter and see what it prints. Python runs in "
        "one session that keeps its variables, imports, and loaded data from one call to the "
        "next, for every member of the meeting: what another agent defined is there for you, "
        "so check before loading something again. R and bash code runs as a fresh process "
        "each time, in the same directory, and can hand results to Python through files.",
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
