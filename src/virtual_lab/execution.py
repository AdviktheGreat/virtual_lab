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
import shutil
import subprocess
import tempfile
import time
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from virtual_lab.artifacts import CodeFile
from virtual_lab.constants import (
    DEFAULT_CPU_LIMIT,
    DEFAULT_EXECUTION_TIMEOUT,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_SANDBOX_IMAGE,
    DEFAULT_TMPFS_SIZE,
    MAX_CAPTURED_OUTPUT_CHARS,
    MAX_REPORTED_OUTPUT_CHARS,
    SANDBOX_WORK_DIR,
)


class ExecutionError(Exception):
    """Raised when code could not be run at all, as opposed to running and failing."""


class DockerUnavailableError(ExecutionError):
    """Raised when the sandbox cannot be used because Docker is missing or not running."""


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
            sections.append("Files written:\n" + "\n".join(self.produced_files))

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


def read_tail(stream, max_chars: int = MAX_CAPTURED_OUTPUT_CHARS) -> str:
    """Reads the end of a spooled output file.

    Output is spooled to a file rather than held in memory because untrusted code can print
    without bound, and only the tail is read back so that doing so cannot exhaust memory either.

    :param stream: The file object to read, positioned anywhere.
    :param max_chars: The most characters to return.
    :return: The decoded tail, with undecodable bytes replaced.
    """
    size = stream.seek(0, os.SEEK_END)
    dropped = max(0, size - max_chars)
    stream.seek(dropped)
    text = stream.read().decode("utf-8", errors="replace")

    if dropped:
        return f"[... {dropped:,} bytes truncated ...]\n{text}"

    return text


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
    docker_command: tuple[str, ...] = ("docker",)
    _checked: bool = field(default=False, init=False, repr=False)

    def build_command(
        self, directory: Path, command: Sequence[str], container_name: str
    ) -> tuple[str, ...]:
        """Builds the docker command that runs code under every restriction this class applies.

        :param directory: The host directory to mount as the working directory.
        :param command: The command to run inside the container.
        :param container_name: The name to give the container, so a timeout can kill it.
        :return: The full command, as an argument list.
        """
        arguments = [
            *self.docker_command,
            "run",
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
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            # The only environment the code gets. HOME must be writable, and unbuffered output
            # means a run that is killed still reports what it had printed.
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
        ]

        if (user := self.container_user()) is not None:
            arguments += ["--user", user]

        return tuple([*arguments, self.image, *command])

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

        before = list_files(directory)
        start = time.monotonic()

        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                completed = subprocess.run(
                    arguments, stdout=out, stderr=err, timeout=limit, check=False
                )
                exit_code: int | None = completed.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                # Killing the docker client leaves the container running, so it has to be
                # stopped by name. It is removed on exit by --rm.
                exit_code, timed_out = None, True
                self.kill(container_name)

            duration = time.monotonic() - start
            stdout, stderr = read_tail(out), read_tail(err)

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

        :param container_name: The name of the container to stop.
        """
        subprocess.run(
            [*self.docker_command, "kill", container_name],
            capture_output=True,
            timeout=60,
            check=False,
        )


# Host variables a subprocess is allowed to inherit. Everything else is dropped, most
# importantly the API key, which is in the environment of whatever called this.
INHERITED_ENVIRONMENT_VARIABLES = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


@dataclass
class LocalExecutor:
    """Runs code directly on this machine, with no isolation whatsoever.

    This is not a sandbox. Code run through it can read and write anything the calling user can,
    reach the network, and outlast the timeout by leaving children behind. It exists so that a
    machine without Docker is not left with no option at all, and it has to be constructed by
    name so that nothing chooses it on a caller's behalf.

    The two protections it does offer are a wall clock timeout and a scrubbed environment, so
    that model-authored code does not inherit the API key of the process that started it.
    """

    timeout: float = DEFAULT_EXECUTION_TIMEOUT
    warn: bool = True

    def __post_init__(self) -> None:
        if self.warn:
            warnings.warn(
                "LocalExecutor runs model-authored code on this machine with no isolation. It "
                "can read and write any file you can. Prefer DockerExecutor.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def build_environment() -> dict[str, str]:
        """Builds the environment the code will run with, keeping only what it needs.

        :return: The environment variables to pass.
        """
        environment = {
            name: os.environ[name]
            for name in INHERITED_ENVIRONMENT_VARIABLES
            if name in os.environ
        }
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        return environment

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

        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                completed = subprocess.run(
                    list(command),
                    cwd=directory,
                    env=self.build_environment(),
                    stdout=out,
                    stderr=err,
                    timeout=limit,
                    check=False,
                )
                exit_code: int | None = completed.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                exit_code, timed_out = None, True

            duration = time.monotonic() - start
            stdout, stderr = read_tail(out), read_tail(err)

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
