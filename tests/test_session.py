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

import pytest

import virtual_lab.session as session_module
from virtual_lab.constants import (
    DEFAULT_SESSION_MEMORY_LIMIT,
    SANDBOX_DATA_LAKE_DIR,
    SANDBOX_PLATFORM,
    SANDBOX_WORK_DIR,
)
from virtual_lab.environment import sandbox_image
from virtual_lab.execution import DockerExecutor
from virtual_lab.session import (
    KERNEL_SOURCE,
    CellResult,
    DockerSession,
    LocalSession,
    SessionError,
    read_responses,
    session_executor,
    session_tool,
)

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
    # Numbered as matplotlib numbers them: by when they were opened, unchanged by closing others
    def figure(number):
        return pyplot.open[[id(f) for f in pyplot.open].index(number)]
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
        requests = 'not json\n[1]\n{"id": 3, "code": "1 + 1"}\n'

        answers = subprocess.run(
            [sys.executable, "-c", KERNEL_SOURCE],
            input=requests,
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=30,
        ).stdout.splitlines()

        assert len(answers) == 4
        assert '"Unreadable request' in answers[1]
        assert '"Unreadable request' in answers[2]
        assert '"id": 3' in answers[3] and '"output": "2\\n"' in answers[3]


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
