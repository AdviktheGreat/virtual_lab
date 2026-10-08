"""Tests for the virtual-lab-functions command."""

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

from virtual_lab import functions_cli
from virtual_lab.execution import DEFAULT_SANDBOX_IMAGE, DockerExecutor, LocalExecutor
from virtual_lab.function_generator import FAILED, UNVERIFIED, VERIFIED, FunctionTask, GeneratedFunction, save_function
from virtual_lab.functions_cli import describe, executor_for, main
from virtual_lab.function_generator import FunctionRunReport

from conftest import TEST_MODEL, FakeClient
from test_function_generator import REQUEST_COST, code, fenced, queue, requests, without_docstring


def run(capsys: pytest.CaptureFixture[str], *arguments: str | Path) -> tuple[int, str, str]:
    status = main([str(argument) for argument in arguments])
    captured = capsys.readouterr()

    return status, captured.out, captured.err


def generate(tmp_path: Path, *arguments: str | Path) -> list[str | Path]:
    return ["generate", tmp_path / "functions", "--model", TEST_MODEL, *arguments]


class TestGenerate:
    def test_a_task_in_words_is_written_saved_and_said(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, fenced(code("count_reads")))

        status, out, err = run(capsys, *generate(tmp_path, "--task", "Count reads"))

        assert status == 0
        assert f"Wrote 1 of 1 functions (1 not run), spending ${REQUEST_COST:.4f}." in out
        assert f"Saved in {tmp_path / 'functions'}" in out
        assert "Function 1 of 1: count_reads" in err
        assert (tmp_path / "functions" / "count_reads.py").read_text() == code("count_reads") + "\n"

    def test_the_progress_is_not_shown_when_it_is_not_wanted(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, fenced(code("count_reads")))

        status, out, err = run(capsys, *generate(tmp_path, "--task", "Count reads", "--quiet"))

        assert status == 0 and err == "" and "Wrote 1 of 1" in out

    def test_tasks_come_from_files_and_words_together_and_each_is_written_once(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "one.json").write_text(json.dumps({"tasks": ["Count reads", "Align reads"]}))
        (tmp_path / "two.json").write_text(json.dumps({"tasks": ["count  READS", "Call variants"]}))
        queue(fake_client, *[fenced(code(name)) for name in ("count_reads", "align_reads", "call_variants", "plot_it")])

        status, out, _ = run(
            capsys,
            *generate(tmp_path, "--tasks", tmp_path / "one.json", "--tasks", tmp_path / "two.json"),
            *["--task", "Plot it", "--task", "ALIGN reads", "--quiet"],
        )

        assert status == 0 and "Wrote 4 of 4 functions" in out
        assert sorted(path.name for path in (tmp_path / "functions").glob("*.py")) == [
            "align_reads.py",
            "call_variants.py",
            "count_reads.py",
            "plot_it.py",
        ]

    def test_only_the_tasks_in_enough_papers_are_taken(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = {"tasks": {"Align reads": 5, "Count reads": 4, "Call variants": 3, "Plot it": 1}}
        (tmp_path / "frequency_summary.json").write_text(json.dumps(summary))
        queue(fake_client, fenced(code("align_reads")), fenced(code("count_reads")))

        status, out, _ = run(
            capsys, *generate(tmp_path, "--tasks", tmp_path / "frequency_summary.json", "--min-papers", "4", "--quiet")
        )

        assert status == 0 and "Wrote 2 of 2 functions" in out
        assert sorted(path.name for path in (tmp_path / "functions").glob("*.py")) == [
            "align_reads.py",
            "count_reads.py",
        ]

    def test_only_the_first_few_tasks_of_each_file_are_taken(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        for name in ("one", "two"):
            (tmp_path / f"{name}.json").write_text(json.dumps({"tasks": [f"Task {name} a", f"Task {name} b"]}))
        queue(fake_client, fenced(code("task_one_a")), fenced(code("task_two_a")))

        status, out, _ = run(
            capsys,
            *generate(tmp_path, "--tasks", tmp_path / "one.json", "--tasks", tmp_path / "two.json", "--limit", "1"),
            "--quiet",
        )

        assert status == 0 and "Wrote 2 of 2 functions" in out
        assert sorted(path.name for path in (tmp_path / "functions").glob("*.py")) == ["task_one_a.py", "task_two_a.py"]

    def test_every_option_is_passed_on_to_the_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passed: dict[str, Any] = {}

        def fake_run(tasks: Any, save_dir: Path, **options: Any) -> FunctionRunReport:
            passed.update(options, save_dir=save_dir, tasks=list(tasks))

            return FunctionRunReport(results=[], tasks=1, spent=0.0, stopped=None, save_dir=save_dir)

        monkeypatch.setattr(functions_cli, "generate_functions", fake_run)

        status, _, _ = run(
            capsys,
            *generate(tmp_path, "--task", "Count reads", "--quiet", "--temperature", "0.9", "--max-attempts", "5"),
            *[
                "--max-completion-tokens",
                "800",
                "--timeout",
                "7.5",
                "--max-cost",
                "3",
                "--max-cost-per-function",
                "0.5",
            ],
            *["--retry-failed", "--max-consecutive-failures", "4", "--verify", "local"],
        )

        assert status == 0 and passed.pop("on_progress") is None
        assert isinstance(passed.pop("executor"), LocalExecutor)
        assert passed == {
            "save_dir": tmp_path / "functions",
            "tasks": [FunctionTask("Count reads", "Count reads")],
            "model": TEST_MODEL,
            "temperature": 0.9,
            "max_attempts": 5,
            "timeout": 7.5,
            "max_completion_tokens": 800,
            "max_cost": 3.0,
            "max_cost_per_function": 0.5,
            "retry_failed": True,
            "max_consecutive_failures": 4,
        }

    def test_no_tasks_is_not_an_error_and_asks_the_model_nothing(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "none.json").write_text('{"tasks": []}')

        status, out, _ = run(capsys, *generate(tmp_path, "--tasks", tmp_path / "none.json"))

        assert status == 0 and out == "There are no tasks to write functions for.\n"
        assert fake_client.completions.calls == [] and not (tmp_path / "functions").exists()

    def test_a_run_with_nothing_to_write_is_told_to_give_tasks(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, out, err = run(capsys, *generate(tmp_path))

        assert (
            status == 2
            and out == ""
            and err == "virtual-lab-functions: ValueError: Give the tasks, with --tasks or --task\n"
        )

    def test_a_function_that_cannot_be_made_to_pass_is_named_with_why_and_is_not_a_stop(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, fenced(without_docstring("count_reads")), fenced(code("align_reads")))

        status, out, _ = run(
            capsys, *generate(tmp_path, "--task", "Count reads", "--task", "Align reads", "--max-attempts", "1")
        )

        assert status == 0
        assert "Wrote 1 of 2 functions (1 not run, 1 failed)" in out
        assert (
            "  Failed: count_reads: FunctionGenerationError: No function that can be used was written in 1 attempt"
            in out
        )

    def test_a_run_that_stops_for_its_limit_says_so_and_carries_on_when_run_again(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, fenced(code("count_reads")))
        arguments = generate(tmp_path, "--task", "Count reads", "--task", "Align reads", "--quiet")

        first = run(capsys, *arguments, "--max-cost", str(REQUEST_COST / 2))

        assert first[0] == 1 and "Wrote 1 of 2 functions" in first[1] and "  Stopped: The run spent $" in first[1]
        queue(fake_client, fenced(code("align_reads")))

        second = run(capsys, *arguments)

        assert second[0] == 0 and "Wrote 2 of 2 functions (2 not run)" in second[1]
        assert len(fake_client.completions.calls) == 2

    def test_failures_in_a_row_stop_the_run_unless_it_is_told_to_keep_going(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, *["No code."] * 8)
        tasks = [argument for name in ("a", "b", "c", "d") for argument in ("--task", f"Task {name}")]

        stopped = run(capsys, *generate(tmp_path, *tasks, "--max-attempts", "1", "--quiet"))

        assert stopped[0] == 1 and "Stopped: 3 functions in a row failed" in stopped[1]
        assert len(fake_client.completions.calls) == 3

        keeping_going = run(
            capsys, *generate(tmp_path, *tasks, "--max-attempts", "1", "--quiet", "--keep-going", "--retry-failed")
        )

        assert keeping_going[0] == 0 and "Wrote 0 of 4 functions (4 failed)" in keeping_going[1]
        assert len(fake_client.completions.calls) == 3 + 4

    def test_a_failed_function_is_tried_again_only_if_asked(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, "No code.")
        arguments = generate(tmp_path, "--task", "Count reads", "--max-attempts", "1", "--quiet")
        run(capsys, *arguments)

        run(capsys, *arguments)
        assert len(fake_client.completions.calls) == 1

        queue(fake_client, fenced(code("count_reads")))
        status, out, _ = run(capsys, *arguments, "--retry-failed")

        assert status == 0 and "Wrote 1 of 1 functions" in out and len(fake_client.completions.calls) == 2

    def test_a_function_is_imported_here_to_verify_it_when_asked(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, fenced(code("count_reads")))

        status, out, _ = run(capsys, *generate(tmp_path, "--task", "Count reads", "--verify", "local", "--quiet"))

        assert status == 0 and "Wrote 1 of 1 functions, spending" in out and "not run" not in out
        record = json.loads((tmp_path / "functions" / "records" / "count_reads.json").read_text())
        assert record["status"] == VERIFIED

    def test_a_library_that_is_not_installed_is_found_out_when_verifying_and_the_model_is_told(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(
            fake_client,
            fenced(code("count_reads", extra="import no_such_library_here\n\n\n")),
            fenced(code("count_reads")),
        )

        status, out, _ = run(capsys, *generate(tmp_path, "--task", "Count reads", "--verify", "local", "--quiet"))

        assert status == 0 and "Wrote 1 of 1 functions" in out
        assert "ModuleNotFoundError: No module named 'no_such_library_here'" in requests(fake_client)[1]["user"]

    def test_the_model_is_asked_as_the_options_say(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, fenced(code("count_reads")), fenced(code("align_reads")))

        run(
            capsys,
            *generate(
                tmp_path, "--task", "Count reads", "--temperature", "0.7", "--max-completion-tokens", "900", "--quiet"
            ),
        )
        run(capsys, *generate(tmp_path, "--task", "Align reads", "--model-temperature", "--quiet"))

        first, second = fake_client.completions.calls
        assert first["temperature"] == 0.7 and 900 in {value for key, value in first.items() if "token" in key}
        assert "temperature" not in second

    def test_the_default_temperature_is_the_librarys(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, fenced(code("count_reads")))

        run(capsys, *generate(tmp_path, "--task", "Count reads", "--quiet"))

        assert fake_client.completions.calls[0]["temperature"] == 0.2


class TestExecutor:
    def arguments(self, **fields: Any) -> argparse.Namespace:
        return argparse.Namespace(**{"verify": None, "image": None, "platform": None, **fields})

    def test_nothing_is_run_unless_asked(self) -> None:
        assert executor_for(self.arguments()) is None

    def test_local_is_the_machines_own_python(self) -> None:
        assert isinstance(executor_for(self.arguments(verify="local")), LocalExecutor)

    def test_docker_is_the_sandbox_image_unless_another_is_named(self) -> None:
        default = executor_for(self.arguments(verify="docker"))
        named = executor_for(self.arguments(verify="docker", image="my/image", platform="linux/amd64"))

        assert (
            isinstance(default, DockerExecutor) and default.image == DEFAULT_SANDBOX_IMAGE and default.platform is None
        )
        assert isinstance(named, DockerExecutor) and (named.image, named.platform) == ("my/image", "linux/amd64")

    def test_docker_that_is_not_running_fails_the_run_before_anything_is_asked(
        self,
        fake_client: FakeClient,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            DockerExecutor, "check_available", lambda self: (_ for _ in ()).throw(RuntimeError("no docker"))
        )

        status, _, err = run(capsys, *generate(tmp_path, "--task", "Count reads", "--verify", "docker"))

        assert status == 2 and err == "virtual-lab-functions: RuntimeError: no docker\n"
        assert fake_client.completions.calls == [] and not (tmp_path / "functions").exists()


class TestDescribe:
    def report(
        self, tmp_path: Path, *statuses: str, spent: float | None = 0.5, stopped: str | None = None
    ) -> FunctionRunReport:
        task = FunctionTask("a", "a")
        results = [GeneratedFunction(task=task, name=f"f{n}", status=status) for n, status in enumerate(statuses)]

        return FunctionRunReport(
            results=results, tasks=len(statuses) + 1, spent=spent, stopped=stopped, save_dir=tmp_path
        )

    def test_what_was_written_is_counted_with_what_was_not_run_and_what_failed(self, tmp_path: Path) -> None:
        text = describe(self.report(tmp_path, VERIFIED, UNVERIFIED, UNVERIFIED, FAILED))

        assert text == "Wrote 3 of 5 functions (2 not run, 1 failed), spending $0.5000."

    def test_what_stopped_the_run_is_said(self, tmp_path: Path) -> None:
        text = describe(self.report(tmp_path, VERIFIED, stopped="Out of money"))

        assert text == "Wrote 1 of 2 functions, spending $0.5000.\n  Stopped: Out of money"

    def test_a_cost_that_cannot_be_worked_out_is_said_so(self, tmp_path: Path) -> None:
        assert describe(self.report(tmp_path, VERIFIED, spent=None)).endswith(
            "spending an amount that cannot be worked out."
        )


class TestList:
    def test_each_function_is_listed_with_its_status_and_what_it_needs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        task = FunctionTask("a", "a")
        save_function(tmp_path, GeneratedFunction(task=task, name="a_task", status=VERIFIED, source="x = 1\n"))
        save_function(
            tmp_path,
            GeneratedFunction(
                task=task, name="b_task", status=UNVERIFIED, source="x = 1\n", modules=("pandas", "scipy")
            ),
        )
        save_function(tmp_path, GeneratedFunction(task=task, name="c_task", status=FAILED, error="x" * 300))

        status, out, _ = run(capsys, "list", tmp_path)

        assert status == 0
        assert out.splitlines() == [
            "verified   a_task",
            "unverified b_task (needs pandas, scipy)",
            "failed     c_task",
            "           " + "x" * 200,
        ]

    def test_a_directory_with_nothing_in_it_says_so(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        status, out, _ = run(capsys, "list", tmp_path)

        assert status == 0 and out == f"No functions are saved in {tmp_path}.\n"


class TestErrors:
    def test_a_tasks_path_that_is_not_there_is_refused_with_no_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, out, err = run(capsys, *generate(tmp_path, "--tasks", tmp_path / "missing.json"))

        assert status == 2 and out == ""
        assert err.startswith("virtual-lab-functions: ValueError:") and "is neither a file nor a directory" in err

    def test_an_error_that_is_not_expected_is_reported_by_its_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("The api_key client option must be set")

        monkeypatch.setattr(functions_cli, "generate_functions", broken)

        status, _, err = run(capsys, *generate(tmp_path, "--task", "Count reads"))

        assert status == 2 and err == "virtual-lab-functions: RuntimeError: The api_key client option must be set\n"

    def test_an_interruption_says_the_command_carries_on(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(functions_cli, "generate_functions", interrupted)

        status, out, err = run(capsys, *generate(tmp_path, "--task", "Count reads"))

        assert status == 130 and out == "" and "the same command carries on from there" in err

    def test_a_limit_that_cannot_be_enforced_is_refused(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, _, err = run(
            capsys,
            "generate",
            tmp_path / "functions",
            *["--task", "Count reads", "--model", "a-model-nobody-priced", "--max-cost", "1"],
        )

        assert status == 2 and "CostUnknownError" in err and not (tmp_path / "functions").exists()

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit) as caught:
            main([])

        assert caught.value.code == 2

    def test_keys_are_taken_from_an_env_file_that_is_there(
        self,
        fake_client: FakeClient,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Set and removed so that the test puts back whatever load_env sets
        monkeypatch.setenv("VL_FUNCTIONS_CLI_TEST", "unset")
        monkeypatch.delenv("VL_FUNCTIONS_CLI_TEST")
        (tmp_path / "keys.env").write_text("VL_FUNCTIONS_CLI_TEST=from-the-file\n")
        queue(fake_client, fenced(code("count_reads")))

        status, _, _ = run(
            capsys, *generate(tmp_path, "--task", "Count reads", "--env-file", tmp_path / "keys.env", "--quiet")
        )

        assert status == 0 and os.environ["VL_FUNCTIONS_CLI_TEST"] == "from-the-file"

    def test_an_env_file_that_is_not_there_is_refused_before_anything_is_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, _, err = run(
            capsys, *generate(tmp_path, "--task", "Count reads", "--env-file", tmp_path / "nothing.env")
        )

        assert (
            status == 2 and "FileNotFoundError: There is no .env file" in err and not (tmp_path / "functions").exists()
        )


def test_the_command_is_installed_by_the_package() -> None:
    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["virtual-lab-functions"] == "virtual_lab.functions_cli:main"
