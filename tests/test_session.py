"""Tests for the interpreter a meeting's agents share.

Almost everything is tested through LocalSession, which runs the same kernel as DockerSession,
only on this machine, so the kernel's behaviour is exercised for real without a Docker daemon.
The container flags are asserted on the command, and the few tests that need a daemon skip
without one.
"""

import gc
import io
import os
import queue
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

import virtual_lab.session as session_module
from virtual_lab.constants import (
    DEFAULT_SESSION_MEMORY_LIMIT,
    SANDBOX_DATA_LAKE_DIR,
    SANDBOX_PLATFORM,
    SANDBOX_USER_DATA_DIR,
    SANDBOX_WORK_DIR,
)
from virtual_lab.environment import sandbox_image
from virtual_lab.execution import DockerExecutor, ExecutionError
from virtual_lab.custom_tools import tool_from_function
from virtual_lab.session import (
    KERNEL_SOURCE,
    CellResult,
    DockerSession,
    LocalSession,
    SessionError,
    call_parameters,
    call_signature,
    check_session_data,
    check_session_software,
    check_session_tools,
    describe_type,
    own_resources_prompt,
    read_responses,
    session_executor,
    session_tool,
    session_tools_prompt,
)
from virtual_lab.tools import Tool

posix_only = pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX only")


@pytest.fixture
def session(tmp_path: Path):
    with LocalSession(tmp_path / "work", warn=False, timeout=20) as opened:
        yield opened


# A stand-in for matplotlib.pyplot, installed from inside the session, so that the kernel's
# handling of open figures is exercised without matplotlib being installed
FAKE_PYPLOT = textwrap.dedent(
    """
    import sys, types
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.open = []
    class Figure:
        def savefig(self, path, **options):
            open(path, "w").write("png")
    # Numbered as matplotlib numbers them: by when they were opened, unchanged by closing others.
    # Asking for a number that is not open opens a new figure, as matplotlib's does.
    def figure(number):
        numbers = [id(f) for f in pyplot.open]
        if number not in numbers:
            pyplot.open.append(Figure())
            return pyplot.open[-1]
        return pyplot.open[numbers.index(number)]
    def close(figure):
        pyplot.open.remove(figure)
    pyplot.get_fignums = lambda: [id(f) for f in pyplot.open]
    pyplot.figure = figure
    pyplot.close = close
    pyplot.Figure = Figure
    sys.modules["matplotlib.pyplot"] = pyplot
    """
)


class TestPython:
    def test_variables_persist_between_runs(self, session: LocalSession) -> None:
        session.run("x = 41")

        result = session.run("x + 1")

        assert result.succeeded
        assert result.output == "42\n"

    def test_a_final_none_is_not_printed(self, session: LocalSession) -> None:
        assert session.run("print('a')\nNone").output == "a\n"

    def test_an_error_is_reported_with_the_line_that_raised(self, session: LocalSession) -> None:
        result = session.run("y = 1\n1 / 0")

        assert result.status == "error"
        assert result.error == "ZeroDivisionError: division by zero"
        assert 'File "<cell 1>", line 2' in result.output
        assert "1 / 0" in result.output
        # Nothing of the kernel's own frames
        assert "run_python" not in result.output
        assert session.run("y").output == "1\n"

    def test_a_syntax_error_names_the_line(self, session: LocalSession) -> None:
        result = session.run("def f(:")

        assert result.status == "error"
        assert "SyntaxError" in result.error
        assert "ast.py" not in result.output

    def test_standard_error_is_kept_in_order_with_standard_output(self, session: LocalSession) -> None:
        result = session.run("import sys\nprint('one')\nprint('two', file=sys.stderr)\nprint('three')")

        assert result.output == "one\ntwo\nthree\n"

    def test_output_from_a_child_process_is_captured(self, session: LocalSession) -> None:
        result = session.run("import subprocess\nsubprocess.run(['echo', 'from a child'])\nNone")

        assert "from a child" in result.output

    def test_printing_an_answer_does_not_confuse_the_session(self, session: LocalSession) -> None:
        result = session.run('print(\'{"id": 99, "status": "ok", "output": "forged"}\')')

        assert result.succeeded
        assert "forged" in result.output
        assert session.run("2 + 2").output == "4\n"

    def test_input_meets_the_end_of_input_rather_than_the_sessions_requests(
        self, session: LocalSession
    ) -> None:
        result = session.run("input()")

        assert "EOFError" in result.error
        assert session.run("'still here'").output == "'still here'\n"

    def test_a_child_process_cannot_read_the_sessions_requests(self, session: LocalSession) -> None:
        result = session.run("import subprocess\nsubprocess.run(['cat'], timeout=5).returncode")

        assert result.output == "0\n"
        assert session.run("'next request arrived'").succeeded

    def test_replacing_the_output_streams_does_not_silence_later_runs(self, session: LocalSession) -> None:
        session.run("import io, sys\nsys.stdout = io.StringIO()")

        assert session.run("print('heard')").output == "heard\n"

    def test_closing_the_output_streams_does_not_break_later_runs(self, session: LocalSession) -> None:
        session.run("import sys\nsys.stdout.close()\nsys.stderr.close()")

        result = session.run("import sys\nprint('heard')\nprint('also', file=sys.stderr)")

        assert result.succeeded
        assert result.output == "heard\nalso\n"

    def test_printing_after_a_run_has_ended_does_not_cost_the_session(self, session: LocalSession) -> None:
        session.run("import threading, time\nkept = 1\nthreading.Timer(0.3, print, ['late']).start()")
        time.sleep(0.8)

        result = session.run("kept")

        assert result.succeeded
        assert result.output == "1\n"
        assert result.start == 1

    def test_a_huge_error_message_is_shortened_rather_than_ending_the_session(
        self, session: LocalSession
    ) -> None:
        session.run("kept = 1")

        result = session.run("raise ValueError('x' * 3_000_000)")

        assert result.status == "error"
        assert len(result.error) < 3_000
        assert "characters truncated" in result.error
        assert session.run("kept").start == 1

    def test_exit_does_not_end_the_session(self, session: LocalSession) -> None:
        session.run("kept = True")

        result = session.run("import sys\nsys.exit(2)")

        assert result.status == "error"
        assert "exit(2)" in result.error
        assert session.run("kept").output == "True\n"

    def test_output_is_bounded_at_both_ends(self, session: LocalSession) -> None:
        result = session.run("print('start' + 'x' * 200_000 + 'end')")

        assert result.output.startswith("start")
        assert result.output.rstrip().endswith("end")
        assert result.output_dropped > 100_000
        assert len(result.output) < 60_000
        assert "bytes of output truncated" in result.output


class TestTimeLimits:
    def test_code_is_stopped_at_its_limit_and_the_session_keeps_its_state(
        self, session: LocalSession
    ) -> None:
        session.run("before = 'kept'")

        result = session.run("while True:\n    pass", timeout=0.5)

        assert result.status == "timeout"
        assert "time limit" in result.error
        assert result.duration < 5
        assert session.run("before").output == "'kept'\n"
        assert result.start == session.history[-1].start == 1

    def test_catching_exception_does_not_swallow_the_limit(self, session: LocalSession) -> None:
        code = "while True:\n    try:\n        sum(range(1000))\n    except Exception:\n        pass"

        assert session.run(code, timeout=0.5).status == "timeout"

    @posix_only
    def test_code_that_ignores_the_limit_costs_the_session(
        self, session: LocalSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_module, "SESSION_GRACE_SECONDS", 0.5)
        session.run("lost = 'soon'")

        result = session.run(
            "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True:\n    pass",
            timeout=0.5,
        )

        assert result.status == "lost"
        assert "did not stop at its time limit" in result.error
        assert "fresh session" in result.error
        after = session.run("lost")
        assert "NameError" in after.error
        assert after.start == 2

    def test_bash_is_stopped_at_its_limit(self, session: LocalSession) -> None:
        result = session.run("sleep 30", language="bash", timeout=0.5)

        assert result.status == "timeout"
        assert result.duration < 5


class TestLostSessions:
    def test_an_interpreter_that_exits_is_reported_and_replaced(self, session: LocalSession) -> None:
        session.run("x = 1")

        result = session.run("import os\nos._exit(3)")

        assert result.status == "lost"
        assert "exited with code 3" in result.error
        after = session.run("'fresh'")
        assert after.succeeded
        assert after.start == 2
        assert session.starts == 2

    @posix_only
    def test_a_killed_interpreter_is_blamed_on_memory(self, session: LocalSession) -> None:
        result = session.run("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)")

        assert result.status == "lost"
        assert "most likely for running out of memory" in result.error

    def test_restart_discards_the_state(self, session: LocalSession) -> None:
        session.run("x = 1")

        session.restart()

        assert "NameError" in session.run("x").error

    def test_an_answer_to_another_request_is_skipped(self, session: LocalSession) -> None:
        session.start()
        session._answers.put(b'{"id": 999, "status": "ok", "output": "stale"}\n')

        result = session.run("'current'")

        assert result.succeeded
        assert result.output == "'current'\n"

    def test_an_overlong_answer_is_not_read_as_one(self) -> None:
        answers: queue.Queue = queue.Queue()
        stream = io.BytesIO(b"x" * (session_module.MAX_SESSION_RESPONSE_BYTES + 10))

        read_responses(stream, answers)

        assert answers.get() == b""
        assert answers.get() is None

    def test_an_answer_of_the_wrong_types_is_read_safely(self, session: LocalSession) -> None:
        result = session.result_from(
            {"status": None, "output": 5, "error": 7, "plots": "no", "output_dropped": None},
            "python",
            "code",
            1.0,
        )

        assert result.status == "None"
        assert result.output == "5"
        assert result.error == "7"
        assert result.plots == ()
        assert result.output_dropped == 0


class TestKernel:
    def test_an_unreadable_request_is_answered_rather_than_skipped(self, tmp_path: Path) -> None:
        requests = 'not json\n[1]\n"text"\n{"id": 4, "code": "1 + 1"}\n'

        answers = subprocess.run(
            [sys.executable, "-c", KERNEL_SOURCE],
            input=requests,
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=30,
        ).stdout.splitlines()

        assert len(answers) == 5
        assert '"error": "Unreadable request: Expecting value' in answers[1]
        assert '"error": "Unreadable request: not an object"' in answers[2]
        assert '"error": "Unreadable request: not an object"' in answers[3]
        assert '"id": 4' in answers[4] and '"output": "2\\n"' in answers[4]

    def test_a_reply_nothing_waits_for_is_dropped_without_an_answer(self, tmp_path: Path) -> None:
        requests = '{"reply": 5, "result": 1}\n{"id": 1, "code": "1 + 1"}\n'

        answers = subprocess.run(
            [sys.executable, "-c", KERNEL_SOURCE],
            input=requests,
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=30,
        ).stdout.splitlines()

        assert len(answers) == 2
        assert '"id": 1' in answers[1]


class TestOtherLanguages:
    def test_bash_runs_in_the_sessions_directory(self, session: LocalSession) -> None:
        result = session.run("pwd", language="bash")

        assert result.succeeded
        assert Path(result.output.strip()).resolve() == session.directory

    def test_a_failing_script_reports_its_exit_code(self, session: LocalSession) -> None:
        result = session.run("echo partial; exit 3", language="sh")

        assert result.status == "error"
        assert result.error == "bash exited with code 3."
        assert result.output == "partial\n"

    @posix_only
    def test_a_background_child_is_stopped_with_its_script(self, session: LocalSession) -> None:
        start = time.monotonic()

        result = session.run("sleep 60 &\necho started", language="bash")

        assert result.succeeded
        assert time.monotonic() - start < 2

    def test_a_missing_interpreter_is_an_error_not_a_crash(
        self, session: LocalSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if subprocess.run(["which", "Rscript"], capture_output=True).returncode == 0:
            pytest.skip("R is installed")

        result = session.run("print(1)", language="r")

        assert result.status == "error"
        assert "Rscript is not installed" in result.error
        assert session.run("'alive'").succeeded

    def test_an_unknown_language_is_refused(self, session: LocalSession) -> None:
        with pytest.raises(ValueError, match="use python, r, or bash"):
            session.run("x", language="julia")


class TestFiles:
    def test_files_written_are_listed(self, session: LocalSession) -> None:
        result = session.run("open('table.csv', 'w').write('a,b\\n')")

        assert result.produced_files == ("table.csv",)
        assert (session.directory / "table.csv").is_file()

    def test_open_figures_are_saved_and_closed(self, session: LocalSession) -> None:
        session.run(FAKE_PYPLOT)

        result = session.run("pyplot.open += [Figure(), Figure()]")

        assert result.plots == ("plots/figure_1.png", "plots/figure_2.png")
        assert set(result.produced_files) == set(result.plots)
        assert (session.directory / "plots" / "figure_2.png").read_text() == "png"
        assert session.run("len(pyplot.open)").output == "0\n"

    def test_the_figures_saved_from_one_run_are_capped(self, session: LocalSession) -> None:
        session.run(FAKE_PYPLOT)

        result = session.run("pyplot.open += [Figure() for _ in range(60)]")

        assert len(result.plots) == 50
        assert "Only the first 50 of 60" in result.output
        assert session.run("len(pyplot.open)").output == "0\n"

    def test_figures_are_saved_where_the_session_started(self, session: LocalSession) -> None:
        session.run(FAKE_PYPLOT)
        (session.directory / "elsewhere").mkdir()

        result = session.run("import os\nos.chdir('elsewhere')\npyplot.open.append(Figure())")

        assert result.plots == ("plots/figure_1.png",)
        assert (session.directory / "plots" / "figure_1.png").is_file()

    def test_a_restarted_session_does_not_overwrite_earlier_figures(self, session: LocalSession) -> None:
        (session.directory / "plots").mkdir()
        (session.directory / "plots" / "figure_1.png").write_text("earlier")
        session.run(FAKE_PYPLOT)

        result = session.run("pyplot.open.append(Figure())")

        assert result.plots == ("plots/figure_2.png",)
        assert (session.directory / "plots" / "figure_1.png").read_text() == "earlier"

    def test_the_report_lists_figures_apart_from_other_files(self) -> None:
        result = CellResult(
            language="python",
            code="",
            status="ok",
            output="done\n",
            error=None,
            duration=1.25,
            plots=("plots/figure_1.png",),
            produced_files=("plots/figure_1.png", "out.csv"),
        )

        report = result.report()

        assert report.startswith("Ran in 1.2 seconds.")
        assert "Figures saved:\nplots/figure_1.png" in report
        assert "Files written:\nout.csv" in report
        assert report.endswith("Output:\ndone\n")


class TestLifecycle:
    def test_the_environment_is_scrubbed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

        with LocalSession(tmp_path, warn=False) as opened:
            result = opened.run("import os\nos.environ.get('OPENAI_API_KEY')")

        assert result.output == ""

    def test_a_closed_session_refuses_code(self, tmp_path: Path) -> None:
        opened = LocalSession(tmp_path, warn=False)
        opened.run("1")
        process = opened._process

        opened.close()

        assert process.poll() is not None
        with pytest.raises(SessionError, match="closed"):
            opened.run("1")

    @posix_only
    def test_a_session_nobody_closed_is_stopped_when_it_is_collected(self, tmp_path: Path) -> None:
        opened = LocalSession(tmp_path, warn=False)
        opened.run("1")
        process = opened._process

        del opened
        gc.collect()

        assert process.wait(timeout=10) is not None

    def test_an_interpreter_that_will_not_start_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(SessionError, match="did not start"):
            LocalSession(tmp_path, python="false", warn=False).start()

    def test_an_interpreter_that_is_missing_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(SessionError, match="Could not start"):
            LocalSession(tmp_path, python=str(tmp_path / "absent"), warn=False).start()

    @posix_only
    def test_an_interpreter_that_never_answers_is_given_up_on(self, tmp_path: Path) -> None:
        silent = tmp_path / "silent"
        silent.write_text("#!/bin/sh\nexec sleep 30\n")
        silent.chmod(0o755)
        opened = LocalSession(tmp_path, python=str(silent), warn=False, start_timeout=0.5)

        with pytest.raises(SessionError, match="did not start"):
            opened.start()

        assert not opened.running

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
    def test_files_are_capped_in_size(self, tmp_path: Path) -> None:
        with LocalSession(tmp_path, warn=False, max_file_bytes=1_000) as opened:
            result = opened.run("with open('big.bin', 'wb') as file:\n    file.write(b'x' * 5_000)")

        assert result.status == "error"
        assert "File too large" in result.output
        assert (tmp_path / "big.bin").stat().st_size <= 1_000

    def test_it_warns_that_nothing_is_isolated(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match="no isolation"):
            LocalSession(tmp_path)

    def test_the_description_says_how_code_ran(self, session: LocalSession) -> None:
        session.run("1")

        description = session.describe()

        assert description["type"] == "LocalSession"
        assert description["sandboxed"] is False
        assert description["python"] == sys.executable
        assert description["starts"] == 1
        assert description["cells"] == 1
        assert description["python_version"] == sys.version.split()[0]


class TestDocker:
    def spawned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **settings) -> list[str]:
        executor = DockerExecutor(**settings)
        monkeypatch.setattr(DockerExecutor, "check_available", lambda self: None)
        command, options, stop = DockerSession(tmp_path, executor=executor).spawn()
        return command

    def test_the_kernel_runs_under_the_executors_restrictions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        command = self.spawned(tmp_path, monkeypatch)

        assert command[:3] == ["docker", "run", "--interactive"]
        for flag in ("--read-only", "--cap-drop", "--memory", "--pids-limit", "--ulimit"):
            assert flag in command
        assert command[command.index("--network") + 1] == "none"
        assert command[-4:] == ["python3", "-u", "-c", KERNEL_SOURCE]

    def test_the_default_is_biomnis_full_environment_on_the_network(self) -> None:
        executor = session_executor()

        assert executor.image == sandbox_image("full")
        assert executor.platform == SANDBOX_PLATFORM
        assert executor.allow_network is True
        assert executor.memory_limit == DEFAULT_SESSION_MEMORY_LIMIT
        assert DockerSession(Path("unused-session-dir")).executor == executor

    def test_an_unbuilt_sandbox_image_is_reported_with_how_to_build_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_module, "sandbox_image_exists", lambda tag, command: False)

        with pytest.raises(SessionError, match="build_sandbox_image"):
            self.spawned(tmp_path, monkeypatch, image=sandbox_image("bio"))

    def test_the_tool_describes_the_container(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        lake.mkdir()
        opened = DockerSession(tmp_path / "work", executor=DockerExecutor(data_lake=lake))

        description = session_tool(opened).description

        assert SANDBOX_WORK_DIR in description
        assert "no network access" in description
        assert SANDBOX_DATA_LAKE_DIR in description

    def test_the_description_records_the_container(self, tmp_path: Path) -> None:
        described = DockerSession(tmp_path, executor=DockerExecutor(image="img:1")).describe()

        assert described["executor"]["image"] == "img:1"
        assert described["sandboxed"] is True

    @pytest.mark.skipif(not DockerExecutor().is_available(), reason="Docker daemon is not reachable")
    def test_a_container_session_keeps_state_and_leaves_nothing_running(self, tmp_path: Path) -> None:
        with DockerSession(tmp_path, executor=DockerExecutor(), timeout=60) as opened:
            opened.run("import os\nx = os.getcwd()")
            result = opened.run("open('here.txt', 'w').write(x)\nx")
            names = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
            ).stdout

        assert result.output == f"'{SANDBOX_WORK_DIR}'\n"
        assert (tmp_path / "here.txt").read_text() == SANDBOX_WORK_DIR
        assert "virtual-lab-session-" in names
        left = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
        assert "virtual-lab-session-" not in left


class TestTool:
    def test_the_tool_runs_code_in_the_session(self, session: LocalSession) -> None:
        tool = session_tool(session)

        tool.function(code="shared = 7")
        report = tool.function(code="echo $0", language="bash")

        assert tool.name == "run_code"
        assert tool.parameters["required"] == ["code"]
        assert report.startswith("Ran in")
        assert session.run("shared").output == "7\n"

    def test_the_tool_says_the_session_is_shared_and_on_the_network(self, session: LocalSession) -> None:
        description = session_tool(session).description

        assert "every member of the meeting" in description
        assert "can reach the internet" in description
        assert str(session.directory) in description


def add(a: int, b: int = 2) -> int:
    """Adds two numbers.

    :param a: The first.
    :param b: The second.
    """
    return a + b


def lookup(gene: str) -> dict:
    """Looks a gene up."""
    return {"gene": gene, "length": 393, "aliases": ["p53"], "score": 0.5, "missing": None}


def fail(reason: str) -> None:
    """Always fails."""
    raise KeyError(reason)


def nap(seconds: float) -> str:
    """Sleeps."""
    time.sleep(seconds)
    return "awake"


def echo(value: Any = None) -> Any:
    """Returns what it is given."""
    return value


HOST_TOOLS = tuple(tool_from_function(function) for function in (add, lookup, fail, nap, echo))


@pytest.fixture
def tooled(tmp_path: Path):
    with LocalSession(tmp_path / "work", warn=False, timeout=20, tools=HOST_TOOLS) as opened:
        yield opened


class TestHostTools:
    def test_code_calls_a_tool_that_runs_on_the_host_and_gets_its_value(self, tooled: LocalSession) -> None:
        result = tooled.run("found = lookup('TP53')\nprint(add(1), add(1, b=5), add('3'), found['aliases'])")

        assert result.succeeded, result.output
        assert result.output == "3 6 5 ['p53']\n"
        assert tooled.run("found").output == "{'gene': 'TP53', 'length': 393, 'aliases': ['p53'], 'score': 0.5, 'missing': None}\n"

    def test_the_tool_runs_in_this_process_not_the_sessions(self, tmp_path: Path) -> None:
        import os

        def host_pid() -> int:
            """The host's process id."""
            return os.getpid()

        with LocalSession(tmp_path, warn=False, tools=(tool_from_function(host_pid),)) as opened:
            result = opened.run("import os\nprint(host_pid(), os.getpid())")

        host, session = result.output.split()
        assert int(host) == os.getpid()
        assert int(session) != os.getpid()

    def test_each_call_is_recorded_with_what_it_was_given(self, tooled: LocalSession) -> None:
        result = tooled.run("add(1)\ntry:\n    fail('x')\nexcept Exception:\n    pass")

        assert [(call["tool"], call["arguments"], call["status"]) for call in result.tool_calls] == [
            ("add", '{"a": 1}', "ok"),
            ("fail", '{"reason": "x"}', "error"),
        ]
        assert result.tool_calls[1]["error"] == "KeyError: 'x'"
        assert all(call["duration"] >= 0 for call in result.tool_calls)
        assert result.to_dict()["tool_calls"][0]["tool"] == "add"

    def test_a_failing_tool_raises_in_the_code_without_the_kernels_frames(self, tooled: LocalSession) -> None:
        result = tooled.run("fail('no such gene')")

        assert result.status == "error"
        assert result.error == "HostToolError: fail failed: KeyError: 'no such gene'"
        assert "<cell" in result.output
        assert "call_host" not in result.output and "stub" not in result.output

    def test_arguments_of_the_wrong_type_are_refused_on_the_host(self, tooled: LocalSession) -> None:
        result = tooled.run("add('one')")

        assert result.status == "error"
        assert "Invalid arguments for add: a: Input should be a valid integer" in result.error

    def test_a_call_that_does_not_fit_the_signature_is_refused_in_the_session(self, tooled: LocalSession) -> None:
        result = tooled.run("add(1, 2, 3)")

        assert result.error == "TypeError: add() too many positional arguments"
        assert result.tool_calls == ()

    def test_a_tool_is_a_function_with_its_signature_and_docstring(self, tooled: LocalSession) -> None:
        assert tooled.run("import inspect\nstr(inspect.signature(add))").output == "'(a, b=2)'\n"
        assert "Adds two numbers." in tooled.run("help(add)").output

    def test_an_optional_argument_left_out_is_not_sent(self, tooled: LocalSession) -> None:
        assert tooled.run("add(1)").tool_calls[0]["arguments"] == '{"a": 1}'

    def test_a_call_still_running_at_the_time_limit_is_left_and_its_reply_dropped(self, tooled: LocalSession) -> None:
        tooled.run("kept = 'still here'")
        started = time.monotonic()

        stopped = tooled.run("nap(2)", timeout=0.5)

        assert stopped.status == "timeout"
        assert time.monotonic() - started < 1.9
        assert stopped.tool_calls[0]["status"] == "running"
        assert tooled.run("kept").output == "'still here'\n"
        # The late reply arrives while other code runs, and is neither taken for an answer nor kept
        kernel = "next(cell.cell_contents for cell in add.__closure__ if hasattr(cell.cell_contents, 'replies'))"
        later = tooled.run(f"import time\ntime.sleep(2)\nprint(add(40), {kernel}.replies)")
        assert later.output == "42 {}\n"
        assert tooled.starts == 1

    def test_threads_the_code_starts_can_call_tools_at_once(self, tooled: LocalSession) -> None:
        code = (
            "import threading\n"
            "results = []\n"
            "threads = [threading.Thread(target=lambda i=i: results.append(add(i, i))) for i in range(8)]\n"
            "[thread.start() for thread in threads]\n"
            "[thread.join() for thread in threads]\n"
            "sorted(results)"
        )

        result = tooled.run(code)

        assert result.output == "[0, 2, 4, 6, 8, 10, 12, 14]\n"
        assert len(result.tool_calls) == 8

    def test_a_thread_cannot_call_a_tool_once_its_code_has_finished(self, tooled: LocalSession) -> None:
        tooled.run(
            "import threading, time\n"
            "errors = []\n"
            "def late():\n"
            "    time.sleep(0.3)\n"
            "    try:\n"
            "        add(1)\n"
            "    except Exception as error:\n"
            "        errors.append(str(error))\n"
            "threading.Thread(target=late).start()"
        )
        time.sleep(1)

        result = tooled.run("errors")

        assert "has finished" in result.output
        assert result.tool_calls == ()

    def test_a_call_still_waiting_when_its_code_finishes_is_ended(self, tooled: LocalSession) -> None:
        result = tooled.run(
            "import threading, time\n"
            "errors = []\n"
            "def waiting():\n"
            "    try:\n"
            "        nap(1)\n"
            "    except Exception as error:\n"
            "        errors.append(str(error))\n"
            "threading.Thread(target=waiting).start()\n"
            "time.sleep(0.3)"
        )
        time.sleep(1.5)

        assert [call["status"] for call in result.tool_calls] == ["running"]
        assert tooled.run("errors").output == "['nap was still running when the code that called it finished']\n"

    def test_values_json_cannot_hold_are_converted_or_refused(self, tooled: LocalSession) -> None:
        tooled.run("class Array:\n    def tolist(self):\n        return [1, 2]")

        assert tooled.run("echo(Array())").output == "[1, 2]\n"
        assert tooled.run("sorted(echo({3, 1}))").output == "[1, 3]\n"
        refused = tooled.run("echo(object())")
        assert refused.error.startswith("TypeError: The arguments to echo must be JSON")
        assert refused.tool_calls == ()

    def test_what_json_cannot_hold_is_returned_as_text(self, tmp_path: Path) -> None:
        def where() -> object:
            """Where."""
            return (Path("/data/x"), object.__new__(type("Opaque", (), {"__str__": lambda self: "opaque"})))

        with LocalSession(tmp_path, warn=False, tools=(tool_from_function(where),)) as opened:
            assert opened.run("where()").output == "['/data/x', 'opaque']\n"

    def test_arguments_too_large_to_send_are_refused_in_the_session(
        self, tooled: LocalSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_module, "MAX_SESSION_RESPONSE_BYTES", 500)
        tooled.restart()

        result = tooled.run("echo('x' * 1000)")

        assert "Write them to a file" in result.error
        assert tooled.run("echo('small')").output == "'small'\n"

    def test_a_result_too_large_to_return_is_an_error_saying_what_to_do(
        self, tooled: LocalSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_module, "MAX_HOST_TOOL_RESULT_BYTES", 100)

        result = tooled.run("echo('x' * 200)")

        assert result.status == "error"
        assert "return the file's path" in result.error

    def test_the_tools_are_defined_again_when_the_session_restarts(self, tooled: LocalSession) -> None:
        tooled.run("import os\nos._exit(1)")

        assert tooled.run("add(2)").output == "4\n"
        tooled.restart()
        assert tooled.run("add(3)").output == "5\n"

    def test_a_call_from_other_code_or_to_no_tool_is_answered_with_an_error(self, tooled: LocalSession) -> None:
        tooled.start()
        tooled._answers.put(b'{"call": 900, "request": 0, "tool": "add", "arguments": {"a": 1}}\n')
        tooled._answers.put(b'{"call": 901, "request": 3, "tool": "missing", "arguments": {}}\n')
        tooled._requests = 2

        result = tooled.run("'next'")

        assert result.output == "'next'\n"
        assert [(call["tool"], call["error"]) for call in result.tool_calls] == [
            ("add", "the code that called it had finished"),
            ("missing", "the session has no tool called missing"),
        ]

    def test_a_tool_cannot_use_the_session_whose_code_called_it(self, tmp_path: Path) -> None:
        opened: list[LocalSession] = []

        def inner() -> str:
            """Runs code in the session."""
            return opened[0].run("1").output

        with LocalSession(tmp_path, warn=False, tools=(tool_from_function(inner),)) as session:
            opened.append(session)
            started = time.monotonic()
            result = session.run("inner()", timeout=10)

        assert result.error == "HostToolError: inner failed: SessionError: A session's tool cannot use the session whose code called it"
        assert time.monotonic() - started < 5

    def test_the_description_lists_the_tools(self, tooled: LocalSession) -> None:
        assert tooled.describe()["tools"] == ["add", "lookup", "fail", "nap", "echo"]

    def test_tools_that_cannot_be_defined_stop_the_session_from_starting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            session_module,
            "call_parameters",
            lambda tool: [{"name": "a", "required": False}, {"name": "b", "required": True}],
        )
        opened = LocalSession(tmp_path, warn=False, tools=(HOST_TOOLS[0],))

        with pytest.raises(SessionError, match="tools could not be defined"):
            opened.start()

        assert not opened.running

    @pytest.mark.skipif(not DockerExecutor().is_available(), reason="Docker daemon is not reachable")
    def test_code_in_a_container_calls_a_tool_on_the_host(self, tmp_path: Path) -> None:
        import os

        def host_pid() -> int:
            """The host's process id."""
            return os.getpid()

        tools = (tool_from_function(host_pid), HOST_TOOLS[0])
        with DockerSession(tmp_path, executor=DockerExecutor(), timeout=60, tools=tools) as opened:
            result = opened.run("print(host_pid(), add(20, 22))")

        assert result.succeeded, result.output
        assert result.output.split() == [str(os.getpid()), "42"]


class TestCheckingTools:
    def test_a_function_must_be_made_a_tool_first(self) -> None:
        with pytest.raises(TypeError, match="tool_from_function"):
            check_session_tools([add])  # type: ignore[list-item]

    @pytest.mark.parametrize("name", ["with-dash", "class", "1st"])
    def test_a_name_code_cannot_call_is_refused(self, name: str) -> None:
        tool = Tool(name=name, description="", parameters={"type": "object"}, function=add)

        with pytest.raises(ValueError, match="not a name Python can call"):
            check_session_tools([tool])

    def test_two_tools_cannot_share_a_name(self) -> None:
        with pytest.raises(ValueError, match="add is given twice"):
            check_session_tools([HOST_TOOLS[0], HOST_TOOLS[0]])


class TestCallSignature:
    def tool(self, properties: dict, required: list[str]) -> Tool:
        return Tool(
            name="t",
            description="Does it.",
            parameters={"type": "object", "properties": properties, "required": required},
            function=lambda **arguments: arguments,
        )

    def test_a_required_parameter_after_an_optional_one_is_passed_by_name(self) -> None:
        tool = self.tool({"a": {}, "b": {"default": 1}, "c": {}, "d": {}, "e": {"default": "x"}}, ["a", "c", "d"])

        assert [(item["name"], item["keyword_only"]) for item in call_parameters(tool)] == [
            ("a", False),
            ("b", False),
            ("c", True),
            ("d", True),
            ("e", True),
        ]
        assert call_signature(tool) == "(a, b=1, *, c, d, e='x')"

    def test_an_optional_parameter_without_a_default_shows_an_ellipsis(self) -> None:
        assert call_signature(self.tool({"a": {}}, [])) == "(a=...)"

    def test_parameters_python_cannot_name_are_passed_with_stars(self, tmp_path: Path) -> None:
        tool = self.tool({"max-results": {}, "query": {}, "arguments": {"default": 0}}, ["max-results", "query"])

        with LocalSession(tmp_path, warn=False, tools=(tool,)) as opened:
            result = opened.run("import inspect\nprint(inspect.signature(t))\nt('q', **{'max-results': 3})")

        assert result.output == "(query, arguments=0, **_arguments)\n{'query': 'q', 'max-results': 3}\n"

    def test_types_are_described_in_a_few_words(self) -> None:
        assert describe_type({"type": "string"}) == "string"
        assert describe_type({"type": "array", "items": {"type": "integer"}}) == "array of integer"
        assert describe_type({"anyOf": [{"type": "string"}, {"type": "null"}]}) == "string or null"
        assert describe_type({"enum": ["a", 1]}) == 'one of "a", 1'
        assert describe_type({"type": ["string", "number"]}) == "string or number"
        assert describe_type({"$ref": "#/$defs/Tree"}) == "Tree"
        assert describe_type({"properties": {}}) == "object"
        assert describe_type({}) == "any"


class TestToolsPrompt:
    def test_each_tool_is_listed_with_how_to_call_it(self) -> None:
        prompt = session_tools_prompt(HOST_TOOLS[:2])

        assert prompt.startswith("- Functions added for this work, already defined in the session. Prefer them")
        assert "without importing them" in prompt
        assert "raises HostToolError" in prompt
        assert (
            "add(a, b=2)\n  Adds two numbers.\n    - a (integer, required): The first.\n"
            "    - b (integer, optional): The second. [Default: 2]"
        ) in prompt
        assert "lookup(gene)\n  Looks a gene up.\n    - gene (string, required)" in prompt

    def test_no_tools_no_prompt(self) -> None:
        assert session_tools_prompt(()) == ""


def own_data(tmp_path: Path) -> Path:
    """A directory of data to give a session, outside any session's directory."""
    data = tmp_path / "inputs"
    (data / "images").mkdir(parents=True)
    (data / "images" / "field1.txt").write_text("cells\n")
    (data / "counts.csv").write_text("gene,count\nTP53,5\n")
    return data


class TestSessionData:
    def test_a_local_session_reads_the_data_where_it_is(self, tmp_path: Path) -> None:
        data = own_data(tmp_path)

        with LocalSession(
            tmp_path / "work",
            warn=False,
            data={data / "counts.csv": " Read counts per gene. ", str(data / "images"): "Microscopy."},
        ) as opened:
            counts, images = opened.data
            result = opened.run(
                f"print(open({opened.data_path(counts)!r}).read().split()[1])\n"
                f"print(open({opened.data_path(images)!r} + '/field1.txt').read().strip())"
            )
            described = opened.describe()["data"]

        assert result.output.split() == ["TP53,5", "cells"]
        assert (counts.name, counts.source, counts.description) == (
            "counts.csv",
            (data / "counts.csv").resolve(),
            "Read counts per gene.",
        )
        assert opened.data_path(counts) == str((data / "counts.csv").resolve())
        assert described == [
            {
                "name": "counts.csv",
                "source": str((data / "counts.csv").resolve()),
                "path": str((data / "counts.csv").resolve()),
                "description": "Read counts per gene.",
            },
            {
                "name": "images",
                "source": str((data / "images").resolve()),
                "path": str((data / "images").resolve()),
                "description": "Microscopy.",
            },
        ]

    def test_paths_are_read_from_where_the_session_is_made(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data = own_data(tmp_path)
        monkeypatch.chdir(data)

        (item,) = check_session_data({"counts.csv": "Counts."})
        (parent,) = check_session_data({"images/..": "Everything."})

        assert item.source == (data / "counts.csv").resolve()
        assert (parent.name, parent.source) == ("inputs", data.resolve())

    def test_a_link_is_found_by_its_own_name(self, tmp_path: Path) -> None:
        data = own_data(tmp_path)
        (data / "latest.csv").symlink_to(data / "counts.csv")

        (item,) = check_session_data({data / "latest.csv": "The latest counts."})

        assert (item.name, item.source) == ("latest.csv", (data / "counts.csv").resolve())

    def test_none_is_nothing(self, tmp_path: Path) -> None:
        opened = LocalSession(tmp_path / "work", warn=False)

        assert opened.data == () and opened.software == {}
        assert opened.describe()["data"] == [] and opened.describe()["software"] == {}
        assert check_session_data(None) == () and check_session_software(None) == {}

    @pytest.mark.parametrize(
        ("given", "error", "message"),
        [
            (["counts.csv"], TypeError, "dict of each file or directory's path"),
            ({3: "Three."}, TypeError, "given by its path, not int"),
            ({"counts.csv": None}, TypeError, "said in a str, not NoneType"),
            ({"counts.csv": "  "}, ValueError, "its description is empty"),
            ({"missing.csv": "Gone."}, FileNotFoundError, "'missing.csv' is not there"),
            ({"/": "Everything."}, ValueError, "give the files or directories in it instead"),
        ],
    )
    def test_data_must_be_paths_that_exist_with_what_they_are(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: object, error: type, message: str
    ) -> None:
        monkeypatch.chdir(own_data(tmp_path))

        with pytest.raises(error, match=message):
            check_session_data(given)  # type: ignore[arg-type]

    def test_no_two_pieces_of_data_may_share_a_name(self, tmp_path: Path) -> None:
        data = own_data(tmp_path)
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "counts.csv").write_text("x\n")

        with pytest.raises(ValueError, match="counts.csv names more than one"):
            LocalSession(
                tmp_path / "work",
                warn=False,
                data={data / "counts.csv": "Counts.", tmp_path / "other" / "counts.csv": "Others."},
            )

    @pytest.mark.parametrize(
        ("given", "error", "message"),
        [
            (["pydeseq2"], TypeError, "dict of each library or program's name"),
            ({"": "Nothing."}, ValueError, "given by its name, not ''"),
            ({3: "Three."}, ValueError, "given by its name, not 3"),
            ({"pydeseq2": 2}, TypeError, "said in a str, not int"),
            ({"pydeseq2": ""}, ValueError, "its description is empty"),
            ({"pydeseq2": "One.", " pydeseq2 ": "Two."}, ValueError, "given twice"),
        ],
    )
    def test_software_must_be_names_with_what_they_do(self, given: object, error: type, message: str) -> None:
        with pytest.raises(error, match=message):
            check_session_software(given)  # type: ignore[arg-type]

    def test_software_is_kept_by_name_and_recorded(self, tmp_path: Path) -> None:
        opened = LocalSession(tmp_path / "work", warn=False, software={" pydeseq2 ": " Differential expression. "})

        assert opened.software == {"pydeseq2": "Differential expression."}
        assert opened.describe()["software"] == {"pydeseq2": "Differential expression."}


class TestDockerData:
    def test_each_piece_is_mounted_read_only_under_its_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data = own_data(tmp_path)
        monkeypatch.setattr(DockerExecutor, "check_available", lambda self: None)
        opened = DockerSession(
            tmp_path / "work", executor=DockerExecutor(), data={data / "counts.csv": "Counts.", data / "images": "Images."}
        )

        command, _, _ = opened.spawn()

        mounted = [command[index + 1] for index, item in enumerate(command) if item == "--mount"]
        assert f"type=bind,source={(data / 'counts.csv').resolve()},target={SANDBOX_USER_DATA_DIR}/counts.csv,readonly" in mounted
        assert f"type=bind,source={(data / 'images').resolve()},target={SANDBOX_USER_DATA_DIR}/images,readonly" in mounted
        assert [opened.data_path(item) for item in opened.data] == ["/data/counts.csv", "/data/images"]
        assert opened.describe()["data"][0]["path"] == "/data/counts.csv"
        assert opened.describe()["data"][0]["source"] == str((data / "counts.csv").resolve())

    @pytest.mark.parametrize("where", ["inside", "around"])
    def test_data_in_or_around_the_sessions_directory_is_refused(self, tmp_path: Path, where: str) -> None:
        data = own_data(tmp_path)
        directory = data / "work" if where == "around" else tmp_path / "work"
        given = data if where == "around" else tmp_path / "work" / "mine.csv"
        given.parent.mkdir(parents=True, exist_ok=True)
        if where == "inside":
            given.write_text("x\n")

        with pytest.raises(ExecutionError, match="overlaps the directory code runs in"):
            DockerSession(directory, executor=DockerExecutor(), data={given: "Mine."})

    @pytest.mark.parametrize("odd", ["in a,b/counts.csv", "a,b.csv"])
    def test_data_whose_path_cannot_be_mounted_is_refused(self, tmp_path: Path, odd: str) -> None:
        # The second is a link to a file whose path can be mounted, under a name that cannot
        clean = tmp_path / "clean.csv"
        clean.write_text("x\n")
        given = tmp_path / odd
        given.parent.mkdir(parents=True, exist_ok=True)
        if "/" in odd:
            given.write_text("x\n")
        else:
            given.symlink_to(clean)

        with pytest.raises(ExecutionError, match="cannot mount a path containing a comma"):
            DockerSession(tmp_path / "work", executor=DockerExecutor(), data={given: "Odd."})

    def test_data_gone_before_the_session_starts_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data = own_data(tmp_path)
        monkeypatch.setattr(DockerExecutor, "check_available", lambda self: None)
        opened = DockerSession(tmp_path / "work", executor=DockerExecutor(), data={data / "counts.csv": "Counts."})
        (data / "counts.csv").unlink()

        with pytest.raises(ExecutionError, match="counts.csv, given to the code to read, is not there any more"):
            opened.spawn()

    def test_without_data_nothing_more_is_mounted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(DockerExecutor, "check_available", lambda self: None)

        command, _, _ = DockerSession(tmp_path / "work", executor=DockerExecutor()).spawn()

        assert [command[index + 1] for index, item in enumerate(command) if item == "--mount"] == [
            f"type=bind,source={(tmp_path / 'work').resolve()},target={SANDBOX_WORK_DIR}"
        ]

    @pytest.mark.skipif(not DockerExecutor().is_available(), reason="Docker daemon is not reachable")
    def test_a_container_reads_the_data_and_cannot_change_it(self, tmp_path: Path) -> None:
        data = own_data(tmp_path)

        with DockerSession(
            tmp_path / "work",
            executor=DockerExecutor(),
            timeout=60,
            data={data / "counts.csv": "Counts.", data / "images": "Images."},
        ) as opened:
            read = opened.run(
                "import os\n"
                "print(sorted(os.listdir('/data')), open('/data/counts.csv').read().split()[1], "
                "open('/data/images/field1.txt').read().strip())"
            )
            written = opened.run("open('/data/images/new.txt', 'w').write('x')")

        assert read.output.strip() == "['counts.csv', 'images'] TP53,5 cells"
        assert written.status == "error" and "Read-only file system" in written.error
        assert not (data / "images" / "new.txt").exists()


class TestOwnResourcesPrompt:
    def test_data_is_listed_by_the_path_code_reads_it_at(self, tmp_path: Path) -> None:
        data = own_data(tmp_path)
        given = {data / "counts.csv": "Read counts per gene."}

        local = own_resources_prompt(LocalSession(tmp_path / "work", warn=False, data=given))
        docker = own_resources_prompt(DockerSession(tmp_path / "work", executor=DockerExecutor(), data=given))

        assert local.startswith("- Data added for this work, which the session's code can read. Prefer it")
        assert f"----\n{(data / 'counts.csv').resolve()}: Read counts per gene.\n----" in local
        assert docker.startswith("- Data added for this work, which the session's code can read but not change.")
        assert "----\n/data/counts.csv: Read counts per gene.\n----" in docker

    def test_software_is_listed_and_what_was_not_found_is_marked(self, tmp_path: Path) -> None:
        opened = LocalSession(tmp_path / "work", warn=False, software={"pytest": "Tests.", "nowhere-xyz": "Missing."})

        prompt = own_resources_prompt(opened, ["nowhere-xyz"])

        assert prompt.startswith("- Software added for this work, for the session's code to use. Prefer it")
        assert (
            "----\npytest: Tests.\nnowhere-xyz: Missing. (not found in the session: check that it is there "
            "before relying on it)\n----"
        ) in prompt
        assert "not found" not in own_resources_prompt(opened)

    def test_tools_come_first_then_data_then_software(self, tmp_path: Path) -> None:
        data = own_data(tmp_path)
        opened = LocalSession(
            tmp_path / "work",
            warn=False,
            tools=HOST_TOOLS[:1],
            data={data / "counts.csv": "Counts."},
            software={"pytest": "Tests."},
        )

        prompt = own_resources_prompt(opened)

        sections = prompt.split("\n\n- ")
        assert len(sections) == 3
        assert sections[0] == session_tools_prompt(HOST_TOOLS[:1])
        assert sections[1].startswith("Data added") and sections[2].startswith("Software added")

    def test_nothing_added_says_nothing(self, tmp_path: Path) -> None:
        assert own_resources_prompt(LocalSession(tmp_path / "work", warn=False)) == ""
