"""Tests for running model-authored code without giving it the machine.

The container flags are asserted against the command that would be run, so the isolation this
module claims is pinned without needing a Docker daemon. The handful of tests that do need one
skip when it is not reachable.
"""

import os
import sys
from pathlib import Path

import pytest

from virtual_lab.artifacts import CodeFile
from virtual_lab.execution import (
    DockerExecutor,
    DockerUnavailableError,
    ExecutionResult,
    LocalExecutor,
    UnsupportedLanguageError,
    command_for,
    list_files,
    run_files,
    truncate_tail,
)

DOCKER = DockerExecutor()

needs_docker = pytest.mark.skipif(
    not DOCKER.is_available(), reason="Docker daemon is not reachable"
)


def python_file(filename: str = "script.py", contents: str = "print('ran')") -> CodeFile:
    return CodeFile(
        filename=filename, language="python", description="A script.", contents=contents
    )


def local() -> LocalExecutor:
    """Builds a local executor without its warning, which is asserted on separately."""
    return LocalExecutor(warn=False)


def command_of(directory: Path = Path("/meeting"), command: tuple[str, ...] = ("python", "./a.py")):
    return DOCKER.build_command(directory=directory, command=command, container_name="probe")


class TestSandboxCommand:
    """The isolation is only as good as the flags, so each one is asserted."""

    def test_the_container_has_no_network_by_default(self) -> None:
        arguments = command_of()

        assert "--network" in arguments
        assert arguments[arguments.index("--network") + 1] == "none"

    def test_network_can_be_granted_explicitly(self) -> None:
        arguments = DockerExecutor(allow_network=True).build_command(
            directory=Path("/meeting"), command=("python", "./a.py"), container_name="probe"
        )

        assert arguments[arguments.index("--network") + 1] == "bridge"

    def test_the_root_filesystem_is_read_only(self) -> None:
        assert "--read-only" in command_of()

    def test_only_the_meeting_directory_is_mounted(self) -> None:
        arguments = command_of(directory=Path("/meeting/artifacts/discussion"))
        mounts = [
            arguments[index + 1] for index, item in enumerate(arguments) if item == "--mount"
        ]

        assert mounts == ["type=bind,source=/meeting/artifacts/discussion,target=/workspace"]

    def test_the_working_directory_is_the_mount(self) -> None:
        arguments = command_of()

        assert arguments[arguments.index("--workdir") + 1] == "/workspace"

    def test_memory_is_capped_and_swap_cannot_exceed_it(self) -> None:
        # Without --memory-swap the container may use twice the memory limit
        arguments = command_of()

        assert arguments[arguments.index("--memory") + 1] == "2g"
        assert arguments[arguments.index("--memory-swap") + 1] == "2g"

    def test_processes_and_cpus_are_capped(self) -> None:
        arguments = command_of()

        assert arguments[arguments.index("--cpus") + 1] == "2"
        assert arguments[arguments.index("--pids-limit") + 1] == "256"

    def test_capabilities_are_dropped(self) -> None:
        arguments = command_of()

        assert arguments[arguments.index("--cap-drop") + 1] == "ALL"
        assert arguments[arguments.index("--security-opt") + 1] == "no-new-privileges"

    def test_no_host_environment_is_passed(self) -> None:
        # The API key is in the environment of whatever called this, and must not reach the code
        arguments = command_of()
        passed = [arguments[index + 1] for index, item in enumerate(arguments) if item == "--env"]

        assert sorted(passed) == [
            "HOME=/tmp",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONUNBUFFERED=1",
        ]

    def test_the_container_is_removed_and_reaps_its_children(self) -> None:
        arguments = command_of()

        assert "--rm" in arguments
        assert "--init" in arguments

    def test_the_container_is_named_so_a_timeout_can_kill_it(self) -> None:
        arguments = command_of()

        assert arguments[arguments.index("--name") + 1] == "probe"

    def test_the_image_precedes_the_command(self) -> None:
        arguments = command_of(command=("python", "./a.py"))

        assert arguments[-3:] == ("python:3.12-slim", "python", "./a.py")

    @pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX only")
    def test_files_are_written_as_the_host_user(self) -> None:
        arguments = command_of()

        assert arguments[arguments.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


class TestCommandForFile:
    @pytest.mark.parametrize(
        "language,expected",
        [
            ("python", "python3"),
            ("Python", "python3"),
            ("  PYTHON  ", "python3"),
            ("py", "python3"),
            ("bash", "bash"),
            ("sh", "sh"),
            ("r", "Rscript"),
            ("R", "Rscript"),
        ],
    )
    def test_the_interpreter_follows_the_language(self, language: str, expected: str) -> None:
        file = CodeFile(
            filename="a.py", language=language, description="d", contents="print(1)"
        )

        assert command_for(file)[0] == expected

    def test_a_language_with_no_interpreter_is_refused(self) -> None:
        file = CodeFile(filename="a.txt", language="text", description="d", contents="notes")

        with pytest.raises(UnsupportedLanguageError, match="no interpreter"):
            command_for(file)

    def test_the_path_cannot_be_read_as_an_option(self) -> None:
        # "python -c" would execute the argument instead of reading a file
        file = python_file(filename="-c")

        assert command_for(file) == ("python3", "./-c")


class TestExecutionResult:
    def test_a_zero_exit_code_is_success(self) -> None:
        result = ExecutionResult(command=(), exit_code=0, stdout="", stderr="", duration=1.0)

        assert result.succeeded

    def test_a_nonzero_exit_code_is_failure(self) -> None:
        result = ExecutionResult(command=(), exit_code=1, stdout="", stderr="boom", duration=1.0)

        assert not result.succeeded

    def test_a_timeout_is_failure_even_without_an_exit_code(self) -> None:
        result = ExecutionResult(
            command=(), exit_code=None, stdout="", stderr="", duration=300.0, timed_out=True
        )

        assert not result.succeeded
        assert "TIMED OUT" in result.report()

    def test_the_report_names_the_failure(self) -> None:
        result = ExecutionResult(
            command=(), exit_code=2, stdout="", stderr="Traceback...", duration=0.5
        )
        report = result.report()

        assert "FAILED with exit code 2" in report
        assert "Traceback..." in report

    def test_the_report_lists_files_that_were_written(self) -> None:
        result = ExecutionResult(
            command=(),
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0,
            produced_files=("results.csv",),
        )

        assert "results.csv" in result.report()

    def test_the_report_keeps_the_end_of_a_long_traceback(self) -> None:
        # The informative part of a traceback is its last lines
        stderr = "noise\n" * 5000 + "ValueError: the real problem"
        result = ExecutionResult(command=(), exit_code=1, stdout="", stderr=stderr, duration=1.0)
        report = result.report(max_chars=500)

        assert "ValueError: the real problem" in report
        assert "truncated" in report
        assert len(report) < 2000

    def test_the_record_is_json_safe(self) -> None:
        import json

        result = ExecutionResult(
            command=("docker", "run"), exit_code=0, stdout="out", stderr="", duration=1.2345
        )
        restored = json.loads(json.dumps(result.to_dict()))

        assert restored["command"] == ["docker", "run"]
        assert restored["succeeded"] is True
        assert restored["duration"] == 1.234 or restored["duration"] == 1.235


class TestDockerUnavailable:
    def test_a_missing_docker_says_so(self, tmp_path) -> None:
        executor = DockerExecutor(docker_command=("definitely-not-docker-xyz",))

        with pytest.raises(DockerUnavailableError, match="was not found"):
            executor.run(directory=tmp_path, command=("python", "./a.py"))

    def test_a_missing_docker_names_the_alternative(self, tmp_path) -> None:
        executor = DockerExecutor(docker_command=("definitely-not-docker-xyz",))

        with pytest.raises(DockerUnavailableError, match="LocalExecutor"):
            executor.run(directory=tmp_path, command=("python", "./a.py"))

    def test_a_stopped_daemon_is_reported_differently(self, tmp_path, monkeypatch) -> None:
        # Installed but not running is a different problem with a different fix
        import subprocess

        import virtual_lab.execution as execution

        monkeypatch.setattr(execution.shutil, "which", lambda name: "/usr/local/bin/docker")
        monkeypatch.setattr(
            execution.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=1, stdout=b"", stderr=b"Cannot connect to the Docker daemon\n"
            ),
        )

        with pytest.raises(DockerUnavailableError, match="daemon is not running"):
            DockerExecutor().run(directory=tmp_path, command=("python", "./a.py"))

    def test_is_available_does_not_raise(self) -> None:
        assert DockerExecutor(docker_command=("definitely-not-docker-xyz",)).is_available() is False


class TestLocalExecutor:
    def test_it_warns_that_it_is_not_a_sandbox(self) -> None:
        with pytest.warns(UserWarning, match="no isolation"):
            LocalExecutor()

    def test_it_runs_code_and_captures_output(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("print('ran')")

        result = local().run(directory=tmp_path, command=(sys.executable, "script.py"))

        assert result.succeeded
        assert "ran" in result.stdout
        assert result.sandboxed is False

    def test_a_failure_carries_the_traceback(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("raise ValueError('the real problem')")

        result = local().run(directory=tmp_path, command=(sys.executable, "script.py"))

        assert not result.succeeded
        assert result.exit_code == 1
        assert "the real problem" in result.stderr

    def test_the_api_key_is_not_inherited(self, tmp_path, monkeypatch) -> None:
        # Model-authored code must not be handed the key of the process that started it
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-visible")
        (tmp_path / "script.py").write_text(
            "import os; print('KEY=' + os.environ.get('OPENAI_API_KEY', 'absent'))"
        )

        result = local().run(directory=tmp_path, command=(sys.executable, "script.py"))

        assert "KEY=absent" in result.stdout
        assert "sk-should-not-be-visible" not in result.stdout

    def test_a_runaway_script_is_stopped(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("while True: pass")

        result = local().run(directory=tmp_path, command=(sys.executable, "script.py"), timeout=2)

        assert result.timed_out
        assert not result.succeeded
        assert result.duration < 30

    def test_unbounded_output_does_not_exhaust_memory(self, tmp_path) -> None:
        # Output is spooled to disk and only its tail is read back
        (tmp_path / "script.py").write_text("for i in range(200000): print('x' * 50)")

        result = local().run(directory=tmp_path, command=(sys.executable, "script.py"))

        assert len(result.stdout) < 200_000
        assert "truncated" in result.stdout

    def test_files_the_code_wrote_are_reported(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("open('results.csv', 'w').write('a,b\\n')")

        result = local().run(directory=tmp_path, command=(sys.executable, "script.py"))

        assert "results.csv" in result.produced_files
        assert "script.py" not in result.produced_files

    def test_a_missing_directory_is_refused(self, tmp_path) -> None:
        with pytest.raises(NotADirectoryError):
            local().run(directory=tmp_path / "nope", command=("echo", "hi"))


class TestRunFiles:
    def test_every_runnable_file_is_run(self, tmp_path) -> None:
        for name in ("first.py", "second.py"):
            (tmp_path / name).write_text(f"print('{name}')")

        results = run_files(
            directory=tmp_path,
            files=[python_file("first.py"), python_file("second.py")],
            executor=local(),
        )

        assert len(results) == 2
        assert all(result.succeeded for _, result in results)

    def test_it_stops_at_the_first_failure(self, tmp_path) -> None:
        (tmp_path / "first.py").write_text("raise SystemExit(3)")
        (tmp_path / "second.py").write_text("print('should not run')")

        results = run_files(
            directory=tmp_path,
            files=[python_file("first.py"), python_file("second.py")],
            executor=local(),
        )

        assert len(results) == 1
        assert results[0][0].filename == "first.py"

    def test_data_and_documentation_are_skipped_not_failed(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("print('ran')")
        notes = CodeFile(
            filename="notes.md", language="text", description="Notes.", contents="# Notes"
        )

        results = run_files(
            directory=tmp_path, files=[notes, python_file("script.py")], executor=local()
        )

        assert [file.filename for file, _ in results] == ["script.py"]


class TestSandbox:
    """These require a running Docker daemon and are skipped without one."""

    @needs_docker
    def test_code_runs_in_the_container(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("print('ran inside')")

        result = DOCKER.run(directory=tmp_path, command=("python", "./script.py"))

        assert result.succeeded, result.stderr
        assert "ran inside" in result.stdout

    @needs_docker
    def test_the_host_filesystem_is_not_visible(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text(
            "import os; print(sorted(os.listdir('/')))\n"
            "print('home contents:', os.path.exists(os.path.expanduser('~/Desktop')))"
        )

        result = DOCKER.run(directory=tmp_path, command=("python", "./script.py"))

        assert "home contents: False" in result.stdout

    @needs_docker
    def test_the_network_is_unreachable(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 443), timeout=5)\n"
            "    print('NETWORK REACHED')\n"
            "except OSError as error:\n"
            "    print('blocked:', type(error).__name__)"
        )

        result = DOCKER.run(directory=tmp_path, command=("python", "./script.py"))

        assert "NETWORK REACHED" not in result.stdout
        assert "blocked:" in result.stdout

    @needs_docker
    def test_the_root_filesystem_cannot_be_written(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text(
            "try:\n"
            "    open('/usr/lib/planted', 'w').write('x')\n"
            "    print('WROTE TO ROOT')\n"
            "except OSError as error:\n"
            "    print('blocked:', type(error).__name__)"
        )

        result = DOCKER.run(directory=tmp_path, command=("python", "./script.py"))

        assert "WROTE TO ROOT" not in result.stdout
        assert "blocked:" in result.stdout

    @needs_docker
    def test_output_written_to_the_mount_reaches_the_host(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("open('results.csv', 'w').write('a,b\\n1,2\\n')")

        result = DOCKER.run(directory=tmp_path, command=("python", "./script.py"))

        assert result.succeeded, result.stderr
        assert (tmp_path / "results.csv").read_text() == "a,b\n1,2\n"
        assert "results.csv" in result.produced_files

    @needs_docker
    def test_a_memory_hog_is_stopped(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("x = bytearray(8 * 1024**3)\nprint('ALLOCATED')")

        result = DOCKER.run(
            directory=tmp_path, command=("python", "./script.py"), timeout=120
        )

        assert "ALLOCATED" not in result.stdout
        assert not result.succeeded

    @needs_docker
    def test_an_infinite_loop_is_stopped(self, tmp_path) -> None:
        (tmp_path / "script.py").write_text("while True: pass")

        result = DOCKER.run(directory=tmp_path, command=("python", "./script.py"), timeout=10)

        assert result.timed_out
        assert result.duration < 60

    @needs_docker
    def test_the_api_key_is_not_in_the_container(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-visible")
        (tmp_path / "script.py").write_text(
            "import os; print('KEY=' + os.environ.get('OPENAI_API_KEY', 'absent'))"
        )

        result = DOCKER.run(directory=tmp_path, command=("python", "./script.py"))

        assert "KEY=absent" in result.stdout


class TestHelpers:
    def test_truncate_tail_keeps_short_text_whole(self) -> None:
        assert truncate_tail("short", max_chars=100) == "short"

    def test_truncate_tail_notes_what_it_dropped(self) -> None:
        result = truncate_tail("a" * 500, max_chars=100)

        assert result.endswith("a" * 100)
        assert "400 characters truncated" in result

    def test_list_files_is_relative_and_recursive(self, tmp_path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "a.py").write_text("")
        (tmp_path / "src" / "b.py").write_text("")

        assert list_files(tmp_path) == {"a.py", "src/b.py"}
