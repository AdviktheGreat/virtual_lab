"""Running model-authored code without giving it the machine.

A meeting that writes code and never runs it can only review its own reasoning, which is how a
stub returning random numbers passes for an implementation. Running that code is the point of
this module, and everything in it starts from the assumption that the code is untrusted: it was
written by a model, nobody read it, and it is about to be executed.

The default backend is a container with no network, no host filesystem beyond the meeting's own
output directory, and hard ceilings on memory, processes, and wall clock time. The local backend
exists for machines without Docker, offers none of that, and has to be asked for by name.
"""

import os
import re
import shutil
import signal
import subprocess
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import IO, Protocol
from uuid import uuid4

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

from virtual_lab.artifacts import CodeFile
from virtual_lab.constants import (
    DEFAULT_CPU_LIMIT,
    DEFAULT_EXECUTION_TIMEOUT,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_SANDBOX_IMAGE,
    DEFAULT_TMPFS_SIZE,
    SANDBOX_BIOMNI_PACKAGE_DIR,
    SANDBOX_BIOMNI_PATH,
    SANDBOX_DATA_LAKE_DIR,
    MAX_CAPTURED_OUTPUT_CHARS,
    MAX_REPORTED_FILES,
    MAX_REPORTED_OUTPUT_CHARS,
    MAX_WRITTEN_FILE_BYTES,
    OUTPUT_DRAIN_TIMEOUT,
    SANDBOX_WORK_DIR,
)


class ExecutionError(Exception):
    """Raised when code could not be run at all, as opposed to running and failing."""


class DockerUnavailableError(ExecutionError):
    """Raised when the sandbox cannot be used because Docker is missing or not running."""


ENVIRONMENT_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def biomni_package_directory() -> Path:
    """The directory holding the copy of Biomni's package that ships with virtual_lab.

    On the Python path, it makes Biomni's tools importable as Biomni's agent imports them:
    from biomni.tool.genomics import ...
    """
    return Path(str(files("virtual_lab") / "sandbox" / "biomni_package"))


def added_environment(
    forward_env: Sequence[str], environment: dict[str, str], already_set: Iterable[str]
) -> dict[str, str]:
    """Checks the variables a caller asked to give the code, and returns their values.

    :param forward_env: Names of variables to pass through from this process's environment.
    :param environment: Variables to set, by name.
    :param already_set: Names the executor sets itself, which neither may change.
    :raises ExecutionError: If a name is not a valid name, is given twice, is one the executor
        sets, or is to be passed through but is not set here.
    :return: Every added variable with its value.
    """
    names = [*forward_env, *environment]
    reserved = set(already_set)

    for name in names:
        if not isinstance(name, str) or not ENVIRONMENT_VARIABLE_NAME.match(name):
            raise ExecutionError(f"{name!r} is not a valid environment variable name.")
        if name in reserved:
            raise ExecutionError(f"{name} is set by the executor itself, so it cannot be given to the code.")

    if duplicated := sorted({name for name in names if names.count(name) > 1}):
        raise ExecutionError(f"{', '.join(duplicated)} is given more than once.")

    # A key that is silently absent would surface later as a tool failing to authenticate,
    # far from its cause
    if missing := [name for name in forward_env if name not in os.environ]:
        raise ExecutionError(
            f"Cannot pass {', '.join(missing)} to the code: not set in this process's environment."
        )

    return {**{name: os.environ[name] for name in forward_env}, **{name: str(value) for name, value in environment.items()}}


class UnsupportedLanguageError(ExecutionError):
    """Raised when there is no known way to run a file."""


# How to run a file of each language. Anything absent cannot be run, which is deliberate: a
# language is added here once there is a sandbox image that can actually execute it.
# Python is invoked as "python3" rather than "python" because the unsandboxed backend runs on
# the host, where a bare "python" often does not exist. Both names exist in the sandbox image.
LANGUAGE_TO_INTERPRETER = {
    "python": ("python3",),
    "python3": ("python3",),
    "py": ("python3",),
    "bash": ("bash",),
    "sh": ("sh",),
    "shell": ("sh",),
    "r": ("Rscript",),
}


def command_for(file: CodeFile) -> tuple[str, ...]:
    """Builds the command that runs a file.

    The path is written as "./name" so that a filename beginning with a dash cannot be read as
    an interpreter option instead of a script.

    :param file: The file to run.
    :raises UnsupportedLanguageError: If there is no known interpreter for the file's language.
    :return: The command, as an argument list.
    """
    language = file.language.strip().casefold()
    interpreter = LANGUAGE_TO_INTERPRETER.get(language)

    if interpreter is None:
        known = ", ".join(sorted(LANGUAGE_TO_INTERPRETER))
        raise UnsupportedLanguageError(
            f'Cannot run "{file.filename}": no interpreter for language "{file.language}". '
            f"Known languages: {known}."
        )

    return (*interpreter, f"./{file.filename}")


def truncate_tail(text: str, max_chars: int) -> str:
    """Shortens text to its last characters, noting what was dropped.

    The tail is kept rather than the head because the useful part of a traceback is its final
    lines, and the useful part of a script's output is usually what it printed last.

    :param text: The text to shorten.
    :param max_chars: The most characters to keep.
    :return: The text, shortened if it was too long.
    """
    if len(text) <= max_chars:
        return text

    dropped = len(text) - max_chars

    return f"[... {dropped:,} characters truncated ...]\n{text[-max_chars:]}"


@dataclass(frozen=True)
class ExecutionResult:
    """What happened when code was run."""

    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    produced_files: tuple[str, ...] = ()
    sandboxed: bool = True

    @property
    def succeeded(self) -> bool:
        """Whether the code ran to completion without error."""
        return self.exit_code == 0 and not self.timed_out

    def report(self, max_chars: int = MAX_REPORTED_OUTPUT_CHARS) -> str:
        """Describes the run in the form an agent is shown when asked to act on it.

        :param max_chars: The most characters of each output stream to include.
        :return: The description.
        """
        if self.timed_out:
            status = f"TIMED OUT after {self.duration:.1f} seconds"
        elif self.succeeded:
            status = f"SUCCEEDED in {self.duration:.1f} seconds"
        else:
            status = f"FAILED with exit code {self.exit_code} after {self.duration:.1f} seconds"

        sections = [f"Execution {status}."]

        if self.produced_files:
            listed = list(self.produced_files[:MAX_REPORTED_FILES])
            if (unlisted := len(self.produced_files) - len(listed)) > 0:
                listed.append(f"... and {unlisted:,} more")
            sections.append("Files written:\n" + "\n".join(listed))

        for name, stream in (("Standard output", self.stdout), ("Standard error", self.stderr)):
            sections.append(
                f"{name}:\n{truncate_tail(stream, max_chars)}" if stream.strip() else f"{name}: empty"
            )

        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        """Converts the result to JSON-safe types for the provenance record."""
        return {
            "command": list(self.command),
            "exit_code": self.exit_code,
            "succeeded": self.succeeded,
            "timed_out": self.timed_out,
            "duration": round(self.duration, 3),
            "sandboxed": self.sandboxed,
            "produced_files": list(self.produced_files),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionResult":
        """Rebuilds a result from what to_dict returned, with its duration as rounded there."""
        return cls(
            command=tuple(data["command"]),
            exit_code=data["exit_code"],
            stdout=data["stdout"],
            stderr=data["stderr"],
            duration=data["duration"],
            timed_out=data["timed_out"],
            produced_files=tuple(data["produced_files"]),
            sandboxed=data["sandboxed"],
        )


class Executor(Protocol):
    """Something that can run a command in a directory."""

    def run(
        self, directory: Path, command: Sequence[str], timeout: float | None = None
    ) -> ExecutionResult:
        """Runs a command with the directory as its working directory."""
        ...


def list_files(directory: Path) -> set[str]:
    """Lists every file under a directory, relative to it.

    :param directory: The directory to walk.
    :return: The relative paths, using forward slashes.
    """
    return {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }


class OutputTail:
    """Reads a pipe to its end in the background, keeping only the last of what came through.

    Untrusted code can print without bound, so output is neither held whole nor spooled to a
    file, either of which it could fill. What is dropped from the front is counted so the reader
    can be told.

    :param stream: The pipe to read. It is closed once it reaches its end.
    :param max_bytes: The most bytes to keep.
    """

    def __init__(self, stream: IO[bytes], max_bytes: int = MAX_CAPTURED_OUTPUT_CHARS) -> None:
        self.max_bytes = max_bytes
        self.dropped = 0
        self._tail = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self._thread.start()

    def _read(self, stream: IO[bytes]) -> None:
        with stream:
            while chunk := stream.read1(65536):  # type: ignore[attr-defined]
                with self._lock:
                    self._tail += chunk
                    if (excess := len(self._tail) - self.max_bytes) > 0:
                        del self._tail[:excess]
                        self.dropped += excess

    def text(self, wait: float = OUTPUT_DRAIN_TIMEOUT) -> str:
        """Returns the tail of the output, once the pipe has closed or the wait has run out.

        :param wait: The most seconds to wait for the pipe to close.
        :return: The decoded tail, with undecodable bytes replaced.
        """
        self._thread.join(wait)

        with self._lock:
            tail = bytes(self._tail)
            dropped = self.dropped

        if dropped:
            # The cut is at a byte count, so it can fall inside a character, whose remaining
            # bytes would decode as a replacement character at the start of the output
            start = 0
            while start < min(3, len(tail)) and 0x80 <= tail[start] <= 0xBF:
                start += 1
            text = tail[start:].decode("utf-8", errors="replace")

            return f"[... {dropped + start:,} bytes truncated ...]\n{text}"

        text = tail.decode("utf-8", errors="replace")

        return text


def run_bounded(
    arguments: Sequence[str],
    timeout: float,
    stop: Callable[[subprocess.Popen], None],
    **options,
) -> tuple[int | None, bool, str, str]:
    """Runs a command, capturing the tail of its output, and stops it once it is over.

    :param arguments: The command to run.
    :param timeout: Seconds to allow it.
    :param stop: Called once the command has exited or run out of time, to stop anything it
        left running. The command itself is killed afterwards if it is still running.
    :param options: Further arguments for subprocess.Popen.
    :return: The exit code (None on a timeout), whether it timed out, and the tails of standard
        output and standard error.
    """
    process = subprocess.Popen(
        list(arguments), stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options
    )
    out, err = OutputTail(process.stdout), OutputTail(process.stderr)  # type: ignore[arg-type]
    timed_out = False

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        stop(process)
        if process.poll() is None:
            process.kill()
        process.wait()

    # One wait for both pipes, which a process that escaped the kill can hold open together
    deadline = time.monotonic() + OUTPUT_DRAIN_TIMEOUT
    stdout = out.text(wait=max(0.0, deadline - time.monotonic()))
    stderr = err.text(wait=max(0.0, deadline - time.monotonic()))

    return None if timed_out else process.returncode, timed_out, stdout, stderr


def check_mountable(path: Path, advice: str) -> None:
    """Refuses a path that Docker's --mount cannot be given intact.

    --mount is parsed as a line of CSV. A comma in the path starts another field that Docker
    reads as an option, and a line break ends the value early, which would mount whatever
    directory the path's first line names.

    :param path: The host path to mount.
    :param advice: What to do instead.
    :raises ExecutionError: If the path cannot be mounted as it is.
    """
    if any(character in str(path) for character in ',"\n\r'):
        raise ExecutionError(
            f"Cannot run code in {str(path)!r}: Docker cannot mount a path containing a comma, "
            f"a quote, or a line break. {advice}"
        )


@dataclass
class DockerExecutor:
    """Runs code in a container that cannot reach the host.

    The container has no network unless asked for, a read-only root filesystem, a writable
    scratch space at /tmp, and exactly one path in common with the host: the directory being
    run, mounted as the working directory. No host environment variables are passed in, which
    matters because the API key is in the environment of whatever called this.

    Docker is driven through its command line rather than a client library, so that the exact
    command can be recorded and rerun by hand, and so that using the sandbox does not add a
    dependency.
    """

    image: str = DEFAULT_SANDBOX_IMAGE
    allow_network: bool = False
    timeout: float = DEFAULT_EXECUTION_TIMEOUT
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    cpu_limit: str = DEFAULT_CPU_LIMIT
    pids_limit: int = DEFAULT_PIDS_LIMIT
    tmpfs_size: str = DEFAULT_TMPFS_SIZE
    max_file_bytes: int = MAX_WRITTEN_FILE_BYTES
    docker_command: tuple[str, ...] = ("docker",)
    # A directory of Biomni's data lake (see virtual_lab.environment.download_data_lake), mounted
    # read-only so that code can read the tables but not change them for the next run
    data_lake: Path | None = None
    # The platform to run the image as, e.g. "linux/amd64" for a sandbox image built for x86-64
    # on an ARM machine. Without it Docker still runs such an image, but warns on the run's
    # stderr, which the meeting would then read as the code's own output.
    platform: str | None = None
    # Whether to mount the copy of Biomni's package, read-only, and put it on the Python path, so
    # that code can import Biomni's tools. They need the libraries of Biomni's sandbox image.
    biomni_tools: bool = False
    # Host environment variables to pass in, by name, such as the API keys some of Biomni's tools
    # use to call a model. Code run here can read them, and with the network, send them anywhere,
    # so pass only keys you accept that risk for. Their values never appear in the command.
    forward_env: tuple[str, ...] = ()
    # Variables to set in the container, such as BIOMNI_LLM for the model Biomni's tools call.
    # They appear in the command and in records, so they are not for secrets.
    environment: dict[str, str] = field(default_factory=dict)
    _checked: bool = field(default=False, init=False, repr=False)

    def build_command(
        self,
        directory: Path,
        command: Sequence[str],
        container_name: str,
        interactive: bool = False,
    ) -> tuple[str, ...]:
        """Builds the docker command that runs code under every restriction this class applies.

        :param directory: The host directory to mount as the working directory.
        :param command: The command to run inside the container.
        :param container_name: The name to give the container, so a timeout can kill it.
        :param interactive: Whether to keep the container's standard input open, for a
            session that is sent code as it goes rather than given a command to finish.
        :raises ExecutionError: If the directory's path cannot be passed to --mount intact.
        :return: The full command, as an argument list.
        """
        check_mountable(directory, "Save the meeting under another directory.")

        arguments = [
            *self.docker_command,
            "run",
            *(["--interactive"] if interactive else []),
            "--rm",
            # tini as the first process, so that orphaned children are reaped rather than left
            "--init",
            "--name",
            container_name,
            "--network",
            "bridge" if self.allow_network else "none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,size={self.tmpfs_size},mode=1777",
            "--mount",
            f"type=bind,source={directory},target={SANDBOX_WORK_DIR}",
            *self.data_lake_arguments(directory),
            *(["--platform", self.platform] if self.platform is not None else []),
            "--workdir",
            SANDBOX_WORK_DIR,
            "--memory",
            self.memory_limit,
            # Matching swap to memory is what makes the memory limit real; without it the
            # container may swap up to twice the limit instead of being stopped.
            "--memory-swap",
            self.memory_limit,
            "--cpus",
            self.cpu_limit,
            "--pids-limit",
            str(self.pids_limit),
            # The mount is on the host's disk, so a file written without bound would fill it
            "--ulimit",
            f"fsize={self.max_file_bytes}",
            # The daemon otherwise keeps its own copy of everything printed, on the host's disk,
            # until the container is removed. Attaching to the output does not depend on it.
            "--log-driver",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            # The only environment the code gets, with what environment_arguments adds. HOME
            # must be writable, and unbuffered output means a run that is killed still reports
            # what it had printed.
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            *self.biomni_arguments(),
            *self.environment_arguments(),
        ]

        if (user := self.container_user()) is not None:
            arguments += ["--user", user]

        return tuple([*arguments, self.image, *command])

    def data_lake_arguments(self, directory: Path) -> list[str]:
        """The mount and variable that give code the data lake, if there is one.

        :param directory: The host directory mounted as the working directory.
        :raises ExecutionError: If the data lake is missing, cannot be mounted, or overlaps the
            working directory.
        """
        if self.data_lake is None:
            return []

        source = Path(self.data_lake).resolve()
        if not source.is_dir():
            raise ExecutionError(f"The data lake {source} is not a directory. Fetch it with download_data_lake.")
        check_mountable(source, "Move the data lake under another directory.")

        # Inside the working directory it would be writable through that mount after all, and
        # around it, everything else in the directory containing it would be visible too
        work = Path(directory).resolve()
        if source.is_relative_to(work) or work.is_relative_to(source):
            raise ExecutionError(
                f"The data lake {source} overlaps the directory code runs in, {work}. Keep the "
                "data lake outside the meeting's save directory, and the meeting outside it."
            )

        return [
            "--mount",
            f"type=bind,source={source},target={SANDBOX_DATA_LAKE_DIR},readonly",
            "--env",
            f"BIOMNI_DATA_LAKE={SANDBOX_DATA_LAKE_DIR}",
        ]

    def biomni_arguments(self) -> list[str]:
        """The mount and variables that let code import Biomni's tools, if asked for.

        :raises ExecutionError: If the package's path cannot be mounted.
        """
        if not self.biomni_tools:
            return []

        source = biomni_package_directory()
        check_mountable(source, "Install virtual_lab under another directory.")

        return [
            "--mount",
            f"type=bind,source={source},target={SANDBOX_BIOMNI_PACKAGE_DIR},readonly",
            "--env",
            f"PYTHONPATH={SANDBOX_BIOMNI_PACKAGE_DIR}",
            # Biomni's tools find the data lake under their data path, as data_lake/
            "--env",
            f"BIOMNI_PATH={SANDBOX_BIOMNI_PATH}",
        ]

    def environment_arguments(self) -> list[str]:
        """The variables the caller asked to give the code.

        A variable passed through is named without its value, which docker then reads from its
        own environment, inherited from this process, so that a key is never in the command.

        :raises ExecutionError: If a variable cannot be given; see added_environment.
        """
        already_set = ["HOME", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"]
        if self.data_lake is not None:
            already_set.append("BIOMNI_DATA_LAKE")
        if self.biomni_tools:
            already_set += ["PYTHONPATH", "BIOMNI_PATH"]

        added_environment(self.forward_env, self.environment, already_set)
        arguments = []
        for name in self.forward_env:
            arguments += ["--env", name]
        for name, value in self.environment.items():
            arguments += ["--env", f"{name}={value}"]

        return arguments

    @staticmethod
    def container_user() -> str | None:
        """Returns the user the container should run as, so that written files belong to the host user.

        :return: The user in "uid:gid" form, or None on platforms without POSIX user ids.
        """
        if not hasattr(os, "getuid"):
            return None

        return f"{os.getuid()}:{os.getgid()}"

    def check_available(self) -> None:
        """Checks that Docker is installed and its daemon is reachable.

        :raises DockerUnavailableError: If the sandbox cannot be used, saying which of the two
            problems it is, since the fixes are different.
        """
        if self._checked:
            return

        executable = self.docker_command[0]

        if shutil.which(executable) is None:
            raise DockerUnavailableError(
                f'Cannot run code in a sandbox: "{executable}" was not found. Install Docker, or '
                "pass an explicitly unsandboxed LocalExecutor if you accept running "
                "model-authored code directly on this machine."
            )

        try:
            probe = subprocess.run(
                [*self.docker_command, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DockerUnavailableError(f"Cannot reach the Docker daemon: {error}") from error

        if probe.returncode != 0:
            detail = probe.stderr.decode("utf-8", errors="replace").strip().splitlines()
            raise DockerUnavailableError(
                "Cannot run code in a sandbox: Docker is installed but its daemon is not "
                "running. Start Docker and try again. "
                f"Docker said: {detail[0] if detail else 'no output'}"
            )

        self._checked = True

    def is_available(self) -> bool:
        """Whether code can currently be run in a sandbox.

        :return: True if Docker is installed and running.
        """
        try:
            self.check_available()
        except DockerUnavailableError:
            return False

        return True

    def run(
        self, directory: Path, command: Sequence[str], timeout: float | None = None
    ) -> ExecutionResult:
        """Runs a command in a container with the directory mounted as its working directory.

        :param directory: The directory to run in. It is the only host path the code can see.
        :param command: The command to run.
        :param timeout: Seconds to allow, defaulting to this executor's timeout.
        :raises DockerUnavailableError: If Docker is missing or not running.
        :raises NotADirectoryError: If the directory does not exist.
        :raises ExecutionError: If the directory's path cannot be mounted.
        :return: What happened, whether or not the code succeeded.
        """
        self.check_available()

        directory = Path(directory).resolve()

        if not directory.is_dir():
            raise NotADirectoryError(f"Cannot run code in {directory}: not a directory")

        limit = self.timeout if timeout is None else timeout
        container_name = f"virtual-lab-{uuid4().hex[:12]}"
        arguments = self.build_command(
            directory=directory, command=command, container_name=container_name
        )

        def stop(process: subprocess.Popen) -> None:
            # Killing the docker client leaves the container running, so it has to be stopped
            # by name. It is removed on exit by --rm.
            if process.poll() is None:
                self.kill(container_name)

        before = list_files(directory)
        start = time.monotonic()
        exit_code, timed_out, stdout, stderr = run_bounded(arguments, timeout=limit, stop=stop)
        duration = time.monotonic() - start

        return ExecutionResult(
            command=arguments,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            timed_out=timed_out,
            produced_files=tuple(sorted(list_files(directory) - before)),
            sandboxed=True,
        )

    def kill(self, container_name: str) -> None:
        """Stops a container, ignoring the case where it has already stopped.

        A kill that fails or hangs is ignored as well. It is only ever attempted when a run has
        already gone wrong, and raising here would lose the result of that run.

        :param container_name: The name of the container to stop.
        """
        try:
            subprocess.run(
                [*self.docker_command, "kill", container_name],
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


# Host variables a subprocess is allowed to inherit. Everything else is dropped, most
# importantly the API key, which is in the environment of whatever called this.
INHERITED_ENVIRONMENT_VARIABLES = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


def kill_process_group(process: subprocess.Popen) -> None:
    """Kills every process in the group a command was started as the leader of.

    This runs after a normal exit as well as after a timeout, because a child the code sent to
    the background is still running either way. While any member is alive the group keeps the
    leader's id, so it cannot have been handed to another group; once none is, there is nothing
    for the kill to find.

    :param process: The command, started in a new session.
    """
    if not hasattr(os, "killpg"):
        return

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass
class LocalExecutor:
    """Runs code directly on this machine, with no isolation whatsoever.

    This is not a sandbox. Code run through it can read and write anything the calling user can
    and reach the network. It exists so that a machine without Docker is not left with no option
    at all, and it has to be constructed by name so that nothing chooses it on a caller's behalf.

    The protections it does offer are a wall clock timeout, a cap on the size of any one file it
    writes, and a scrubbed environment, so that model-authored code does not inherit the API key
    of the process that started it. The code runs in its own process group, which is killed when
    the run ends, so children it started in the background do not outlive it. A child that puts
    itself in a new session escapes that, as it would escape any cleanup short of a container.
    """

    timeout: float = DEFAULT_EXECUTION_TIMEOUT
    warn: bool = True
    max_file_bytes: int = MAX_WRITTEN_FILE_BYTES
    # As for DockerExecutor: Biomni's tools on the Python path (they need Biomni's environment,
    # such as its biomni_e1), variables passed through by name, and variables to set
    biomni_tools: bool = False
    forward_env: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.warn:
            warnings.warn(
                "LocalExecutor runs model-authored code on this machine with no isolation. It "
                "can read and write any file you can. Prefer DockerExecutor.",
                UserWarning,
                stacklevel=3,
            )

    def build_environment(self) -> dict[str, str]:
        """Builds the environment the code will run with, keeping only what it needs.

        :raises ExecutionError: If a variable asked for cannot be given; see added_environment.
        :return: The environment variables to pass.
        """
        environment = {
            name: os.environ[name]
            for name in INHERITED_ENVIRONMENT_VARIABLES
            if name in os.environ
        }
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        if self.biomni_tools:
            environment["PYTHONPATH"] = str(biomni_package_directory())

        return {**environment, **added_environment(self.forward_env, self.environment, environment)}

    def limit_file_size(self) -> Callable[[], None] | None:
        """Builds what the child runs before the code starts to cap the size of files it writes.

        The hard limit is lowered as well as the soft one, so the code cannot raise it again.

        :return: The function to run in the child, or None where there are no resource limits.
        """
        if resource is None:
            return None

        _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        limit = self.max_file_bytes

        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)

        return partial(resource.setrlimit, resource.RLIMIT_FSIZE, (limit, limit))

    def run(
        self, directory: Path, command: Sequence[str], timeout: float | None = None
    ) -> ExecutionResult:
        """Runs a command with the directory as its working directory.

        :param directory: The directory to run in. The code is not confined to it.
        :param command: The command to run.
        :param timeout: Seconds to allow, defaulting to this executor's timeout.
        :raises NotADirectoryError: If the directory does not exist.
        :return: What happened, whether or not the code succeeded.
        """
        directory = Path(directory).resolve()

        if not directory.is_dir():
            raise NotADirectoryError(f"Cannot run code in {directory}: not a directory")

        limit = self.timeout if timeout is None else timeout
        before = list_files(directory)
        start = time.monotonic()

        try:
            exit_code, timed_out, stdout, stderr = run_bounded(
                command,
                timeout=limit,
                stop=kill_process_group,
                cwd=directory,
                env=self.build_environment(),
                start_new_session=hasattr(os, "killpg"),
                preexec_fn=self.limit_file_size(),
            )
        except FileNotFoundError as error:
            # 127 is what a shell reports for a command it cannot find, and a file in a language
            # whose interpreter is not installed is a failed run, not a reason to stop the meeting
            exit_code, timed_out, stdout = 127, False, ""
            stderr = f'Cannot run "{command[0]}": it is not installed on this machine ({error}).'

        duration = time.monotonic() - start

        return ExecutionResult(
            command=tuple(command),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            timed_out=timed_out,
            produced_files=tuple(sorted(list_files(directory) - before)),
            sandboxed=False,
        )


def run_files(
    directory: Path,
    files: Iterable[CodeFile],
    executor: Executor,
    timeout: float | None = None,
) -> tuple[tuple[CodeFile, ExecutionResult], ...]:
    """Runs each runnable file in a directory, stopping at the first failure.

    Files in languages with no interpreter are skipped rather than treated as failures, since a
    meeting's output legitimately includes data and documentation alongside its code.

    :param directory: The directory holding the files.
    :param files: The files to run, in order.
    :param executor: What to run them with.
    :param timeout: Seconds to allow each file.
    :return: Each file that was run, paired with its result.
    """
    results: list[tuple[CodeFile, ExecutionResult]] = []

    for file in files:
        try:
            command = command_for(file)
        except UnsupportedLanguageError:
            continue

        result = executor.run(directory=directory, command=command, timeout=timeout)
        results.append((file, result))

        if not result.succeeded:
            break

    return tuple(results)
