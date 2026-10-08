"""Tests for writing a function for a task, checking it, and saving it."""

import json
from pathlib import Path
from typing import Any

import pytest

from virtual_lab.execution import ExecutionError, ExecutionResult, LocalExecutor
from virtual_lab.function_checks import script_name
from virtual_lab.function_generator import (
    AGAIN,
    CHECK_FILE,
    CONFLICT,
    CORRECTION_PROMPT,
    FAILED,
    REUSE,
    SYSTEM_PROMPT,
    UNVERIFIED,
    VERIFIED,
    VERIFY,
    FunctionBudgetExceededError,
    FunctionGenerationError,
    FunctionRunReport,
    FunctionTask,
    GeneratedFunction,
    brief_of,
    check_function_name,
    check_generation_options,
    function_names,
    function_tasks,
    generate_function,
    generate_functions,
    import_in,
    load_function,
    resume_decision,
    save_function,
    saved_functions,
    unique_tasks,
)
from virtual_lab.paper_batch import (
    COMBINED_FILE,
    FREQUENCY_FILE,
    Paper,
    PaperResult,
    file_digest,
    save_json,
    summarize_papers,
)
from virtual_lab.papers import PaperFindings, PaperTask
from virtual_lab.utils import CostUnknownError, MeetingUsage, compute_token_cost

from conftest import TEST_MODEL, FakeClient, make_usage, text_response

REQUEST_COST = compute_token_cost(TEST_MODEL, 100, 20)


def code(name: str = "count_reads", parameters: str = "path: str", extra: str = "") -> str:
    return (
        f'{extra}def {name}({parameters}) -> int:\n    """Counts the reads in a file.\n\n'
        f'    Args:\n        path: The file.\n    """\n    return 1'
    )


def fenced(source: str) -> str:
    return f"Here it is.\n```python\n{source}\n```\n"


def without_docstring(name: str = "count_reads") -> str:
    return f"def {name}(path: str) -> int:\n    return 1"


def with_a_path_hint(name: str = "count_reads") -> str:
    return code(name, "path: Path", "from pathlib import Path\n\n\n")


def without_the_function(name: str = "count_reads") -> str:
    return code(name + "_other")


def queue(client: FakeClient, *replies: Any) -> None:
    """Queues the replies a model gives, which are text or a response to give as it is."""
    for reply in replies:
        client.completions.responses.append(text_response(reply) if isinstance(reply, str) else reply)


def requests(client: FakeClient) -> list[dict[str, str]]:
    """What the model was sent, as its system prompt and what it was asked, for each request."""
    return [
        {"system": call["messages"][0]["content"], "user": call["messages"][1]["content"]}
        for call in client.completions.calls
    ]


def reply_of(reason: str, content: str) -> Any:
    response = text_response(content)
    response.choices[0].finish_reason = reason

    return response


def succeeded() -> ExecutionResult:
    return ExecutionResult(command=(), exit_code=0, stdout="", stderr="", duration=0.1)


def failed(stderr: str = "Traceback\nModuleNotFoundError: No module named 'scanpy'") -> ExecutionResult:
    return ExecutionResult(command=(), exit_code=1, stdout="", stderr=stderr, duration=0.1)


class StubExecutor:
    """Stands in for an executor, keeping what it was asked to run and answering from a queue."""

    def __init__(self, *results: ExecutionResult, unavailable: bool = False) -> None:
        self.results = list(results)
        self.runs: list[dict[str, Any]] = []
        self.unavailable = unavailable
        self.checked = 0

    def check_available(self) -> None:
        self.checked += 1
        if self.unavailable:
            raise ExecutionError("Docker is not running")

    def run(self, directory: Path, command: Any, timeout: float | None = None) -> ExecutionResult:
        files = {
            path.relative_to(directory).as_posix(): path.read_text() for path in directory.rglob("*") if path.is_file()
        }
        self.runs.append({"command": tuple(command), "files": files, "timeout": timeout, "directory": directory})

        return self.results.pop(0) if self.results else succeeded()


class TestPrompt:
    def test_biomnis_requirements_are_asked_and_then_what_a_tool_needs(self) -> None:
        system = SYSTEM_PROMPT.format(name="count_reads")

        assert system.startswith("You are a senior Python engineer. Generate robust, idiomatic Python code")
        for requirement in (
            "1. Output ONLY Python code, ideally inside a single triple-backtick code block.",
            "2. Include minimal inline comments and a small docstring.",
            "3. Add a `main()` and an `if __name__ == '__main__':` guard when appropriate.",
            "4. Avoid external dependencies unless necessary; if used, show `pip` installs in comments.",
            "5. Do not include prose before or after the code.",
            "6. When applicable, prioritize the use of codes on public repositories, such as HuggingFace or Github",
        ):
            assert requirement in system
        assert "one function named\n   `count_reads`" in system and "must only define things" in system

    def test_a_correction_shows_the_code_and_what_is_wrong_with_it(self) -> None:
        text = CORRECTION_PROMPT.format(task_prompt="TASK", source="CODE", problems="1. WRONG")

        assert text.startswith("TASK\n\nThis is the code that was written for it:\n\n```python\nCODE\n```")
        assert "1. WRONG" in text and text.endswith("and nothing else.")


class TestFunctionTask:
    def test_a_task_in_words_is_named_by_them_and_told_to_the_model_as_they_are(self) -> None:
        task = FunctionTask.of("  Count  the reads\nin a file  ")

        assert (task.name, task.brief, task.papers) == (
            "Count the reads in a file",
            "Count  the reads\nin a file",
            None,
        )

    def test_a_task_already_is_one_and_anything_else_is_refused(self) -> None:
        task = FunctionTask("a", "b", 3)

        assert FunctionTask.of(task) is task
        with pytest.raises(TypeError, match="A task is a str or a FunctionTask, not int"):
            FunctionTask.of(5)  # type: ignore[arg-type]

    def test_a_task_is_saved_and_read_back(self) -> None:
        task = FunctionTask("Align reads", "Align reads\nWhat it does: aligns.", 4)

        assert FunctionTask.from_dict(task.to_dict()) == task

    @pytest.mark.parametrize(
        ("name", "brief", "message"),
        [("", "x", "needs a name"), ("  ", "x", "needs a name"), ("a", " ", "nothing to say what it is")],
    )
    def test_a_task_needs_a_name_and_something_to_say_what_it_is(self, name: str, brief: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            FunctionTask(name, brief)

    def test_a_task_too_long_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("virtual_lab.function_generator.MAX_FUNCTION_TASK_CHARS", 10)

        with pytest.raises(ValueError, match="is 20 characters, more than the 10 a function may be written for"):
            FunctionTask("a" * 20, "a" * 20)


class TestGeneratedFunction:
    def test_it_is_saved_and_read_back_with_everything_about_it(self) -> None:
        function = GeneratedFunction(
            task=FunctionTask("Count reads", "Count reads", 2),
            name="count_reads",
            status=VERIFIED,
            source="x = 1\n",
            attempts=2,
            problems=("a", "b"),
            modules=("pandas",),
            model="gpt-4o",
            usage={"gpt-4o": {"input": 1}},
            cost=0.5,
            error=None,
            elapsed=1.5,
            digest="abc",
        )

        assert GeneratedFunction.from_dict(json.loads(json.dumps(function.to_dict()))) == function

    def test_it_is_one_of_three_statuses_and_only_a_failed_one_is_not_usable(self) -> None:
        task = FunctionTask("a", "a")

        with pytest.raises(ValueError, match="status is one of verified, unverified, failed, not 'done'"):
            GeneratedFunction(task=task, name="a", status="done")
        assert [
            GeneratedFunction(task=task, name="a", status=status).usable for status in (VERIFIED, UNVERIFIED, FAILED)
        ] == [
            True,
            True,
            False,
        ]


class TestOptions:
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"temperature": 3.0}, "temperature"),
            ({"max_attempts": 0}, "max_attempts must be at least 1, not 0"),
            ({"max_completion_tokens": 0}, "max_completion_tokens must be at least 1, not 0"),
            ({"max_cost": -1.0}, "max_cost must be a finite amount, zero or more, not -1.0"),
            ({"max_cost": float("nan")}, "max_cost must be a finite amount"),
            ({"max_cost": float("inf")}, "max_cost must be a finite amount"),
        ],
    )
    def test_an_option_out_of_range_is_refused(self, options: dict[str, Any], message: str) -> None:
        arguments = {
            "model": TEST_MODEL,
            "temperature": 0.2,
            "max_attempts": 3,
            "max_completion_tokens": None,
            "max_cost": None,
            **options,
        }

        with pytest.raises(ValueError, match=message):
            check_generation_options(**arguments)

    def test_a_limit_on_spending_is_refused_for_a_model_whose_price_is_not_known(self) -> None:
        with pytest.raises(CostUnknownError, match="so a max_cost cannot be enforced"):
            check_generation_options("no-such-model", 0.2, 3, None, 1.0)

        check_generation_options("no-such-model", 0.2, 3, None, None)

    @pytest.mark.parametrize("name", ["count_reads", "a", "x1", "_private", "a" * 64])
    def test_a_name_that_is_a_lowercase_identifier_is_a_function_name(self, name: str) -> None:
        check_function_name(name)

    @pytest.mark.parametrize("name", ["Count", "1abc", "a b", "a-b", "é", "a" * 65, "", "a.b"])
    def test_any_other_name_is_not(self, name: str) -> None:
        with pytest.raises(ValueError, match="cannot name a function"):
            check_function_name(name)


class TestGenerateFunction:
    def test_a_model_is_asked_as_biomnis_function_generator_asks_it(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(code()))

        generate_function("Count reads in a file", name="count_reads", model=TEST_MODEL, temperature=0.7)

        (request,) = requests(fake_client)
        assert request["system"] == SYSTEM_PROMPT.format(name="count_reads")
        assert request["user"] == "Generate Python codes for the following task:\nCount reads in a file"
        assert fake_client.completions.calls[0]["temperature"] == 0.7

    def test_a_function_that_can_be_a_tool_is_returned_unverified_with_what_it_cost(
        self, fake_client: FakeClient
    ) -> None:
        queue(fake_client, fenced(code(extra="import pandas as pd\n\n\n")))

        function = generate_function("Count reads in a file", name="count_reads", model=TEST_MODEL)

        assert function.status == UNVERIFIED and function.usable
        assert function.source == code(extra="import pandas as pd\n\n\n") + "\n"
        assert (function.name, function.attempts, function.modules, function.model) == (
            "count_reads",
            1,
            ("pandas",),
            TEST_MODEL,
        )
        assert function.cost == pytest.approx(REQUEST_COST) and function.usage
        assert function.task == FunctionTask("Count reads in a file", "Count reads in a file")

    def test_it_is_named_as_biomni_names_a_file_unless_it_is_given_a_name(self, fake_client: FakeClient) -> None:
        task = "Perform differential expression analysis on RNA-seq data, using DESeq2"
        named = script_name(task)
        queue(fake_client, fenced(code(named)), fenced(code("my_function")))

        assert (
            generate_function(task, model=TEST_MODEL).name
            == named
            == "perform_differential_expression_analysis_on_rnaseq"
        )
        assert generate_function(task, name="my_function", model=TEST_MODEL).name == "my_function"

    def test_what_is_refused_is_refused_before_the_model_is_asked(self, fake_client: FakeClient) -> None:
        for options in ({"name": "Bad Name"}, {"max_attempts": 0}, {"temperature": 5}, {"max_cost": -1}):
            with pytest.raises(ValueError):
                generate_function("Count reads", model=TEST_MODEL, **options)
        with pytest.raises(ValueError, match="needs a name"):
            generate_function(" ", model=TEST_MODEL)

        assert fake_client.completions.calls == []

    def test_a_reply_that_is_only_code_is_taken_as_it_is(self, fake_client: FakeClient) -> None:
        queue(fake_client, code())

        assert generate_function("Count reads", name="count_reads", model=TEST_MODEL).attempts == 1

    def test_code_that_cannot_be_a_tool_is_given_back_with_what_is_wrong_and_corrected(
        self, fake_client: FakeClient
    ) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(code()))

        function = generate_function("Count reads", name="count_reads", model=TEST_MODEL)

        first, second = requests(fake_client)
        assert second["system"] == first["system"]
        assert second["user"] == CORRECTION_PROMPT.format(
            task_prompt=first["user"],
            source=without_docstring() + "\n",
            problems="1. count_reads has no docstring. Give it one that says what it does, what each parameter is, "
            "and what it returns.",
        )
        assert function.attempts == 2 and function.source == code() + "\n"
        assert function.cost == pytest.approx(2 * REQUEST_COST)

    def test_every_fault_in_the_code_is_given_back_at_once(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced("def count_reads(path, n: set):\n    return 1"), fenced(code()))

        generate_function("Count reads", name="count_reads", model=TEST_MODEL)

        problems = requests(fake_client)[1]["user"]
        assert "1. count_reads has no docstring" in problems
        assert "2. Parameter path has no type hint" in problems and "3. The type hint of parameter n" in problems

    def test_a_reply_with_no_code_is_asked_again_with_why(self, fake_client: FakeClient) -> None:
        queue(fake_client, "I am sorry, I cannot do that.", fenced(code()))

        function = generate_function("Count reads", name="count_reads", model=TEST_MODEL)

        retry = requests(fake_client)[1]["user"]
        assert retry.startswith(
            "Generate Python codes for the following task:\nCount reads\n\nYour last reply could not"
        )
        assert "The reply had no Python code. Reply with the code in one ```python code block." in retry
        assert function.attempts == 2

    def test_a_reply_that_ran_out_of_tokens_is_asked_again_for_less(self, fake_client: FakeClient) -> None:
        queue(fake_client, reply_of("length", fenced(code())), fenced(code()))

        generate_function("Count reads", name="count_reads", model=TEST_MODEL)

        assert "ran out of tokens before the code was finished. Write less code" in requests(fake_client)[1]["user"]

    def test_a_code_block_that_was_never_closed_is_asked_again_for_less(self, fake_client: FakeClient) -> None:
        queue(fake_client, "```python\n" + code(), fenced(code()))

        generate_function("Count reads", name="count_reads", model=TEST_MODEL)

        assert "ended before the code block was closed" in requests(fake_client)[1]["user"]

    def test_the_last_code_that_was_readable_is_kept_if_what_follows_is_not_code(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()), "Sorry.", "Sorry again, differently.")

        with pytest.raises(FunctionGenerationError) as raised:
            generate_function("Count reads", name="count_reads", model=TEST_MODEL)

        assert raised.value.source == without_docstring() + "\n"

    def test_it_stops_after_the_attempts_it_may_make_and_says_what_each_wrote(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(with_a_path_hint()), fenced(without_the_function()))

        with pytest.raises(FunctionGenerationError) as raised:
            generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_attempts=3)

        error = raised.value
        assert len(fake_client.completions.calls) == 3 and error.attempts == 3
        assert str(error).startswith(
            "No function that can be used was written in 3 attempts: The code defines no function"
        )
        assert error.problems and "defines no function named count_reads" in error.problems[0]
        assert error.source == without_the_function() + "\n"
        assert error.usage is not None and error.usage.compute_cost() == pytest.approx(3 * REQUEST_COST)

    def test_it_says_attempt_and_not_attempts_of_one(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()))

        with pytest.raises(FunctionGenerationError, match="written in 1 attempt: count_reads has no docstring"):
            generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_attempts=1)

    def test_a_model_that_repeats_a_fault_is_not_asked_again(self, fake_client: FakeClient) -> None:
        queue(fake_client, *[fenced(without_docstring())] * 5)

        with pytest.raises(FunctionGenerationError) as raised:
            generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_attempts=5)

        assert len(fake_client.completions.calls) == 2 and raised.value.attempts == 2

    def test_a_fault_that_is_not_the_same_each_time_is_corrected_again(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(with_a_path_hint()), fenced(code()))

        assert generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_attempts=3).attempts == 3

    def test_a_limit_on_spending_is_checked_before_each_request(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(code()))

        with pytest.raises(FunctionBudgetExceededError) as raised:
            generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_cost=REQUEST_COST / 2)

        assert len(fake_client.completions.calls) == 1
        assert raised.value.source == without_docstring() + "\n" and raised.value.spent == pytest.approx(REQUEST_COST)
        assert raised.value.usage is not None and raised.value.usage.compute_cost() == pytest.approx(REQUEST_COST)

    def test_what_a_caller_has_already_spent_counts_towards_the_limit(self, fake_client: FakeClient) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage(1_000_000, 0))

        with pytest.raises(FunctionBudgetExceededError):
            generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_cost=1.0, usage=usage)

        assert fake_client.completions.calls == []

    def test_what_the_requests_use_is_added_to_the_count_a_caller_keeps(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(code()))
        usage = MeetingUsage()

        generate_function("Count reads", name="count_reads", model=TEST_MODEL, usage=usage)

        assert usage.compute_cost() == pytest.approx(2 * REQUEST_COST)

    def test_a_limit_cannot_be_kept_if_a_response_does_not_say_what_it_used(self, fake_client: FakeClient) -> None:
        silent = text_response(fenced(without_docstring()))
        silent.usage = None
        queue(fake_client, silent, fenced(code()))

        with pytest.raises(
            FunctionGenerationError, match="did not report what it used, so max_cost cannot be enforced"
        ) as raised:
            generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_cost=1.0)

        assert raised.value.source == without_docstring() + "\n"

    def test_without_a_price_a_function_is_written_and_its_cost_is_not_known(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(code()))

        function = generate_function("Count reads", name="count_reads", model="no-such-model")

        assert function.cost is None and function.status == UNVERIFIED

    def test_the_most_an_answer_may_use_is_asked_of_the_model(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(code()))

        generate_function("Count reads", name="count_reads", model=TEST_MODEL, max_completion_tokens=777)

        assert 777 in {value for key, value in fake_client.completions.calls[0].items() if "token" in key}

    def test_progress_is_said_before_each_request(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(code()))
        lines: list[str] = []

        generate_function("Count reads", name="count_reads", model=TEST_MODEL, on_progress=lines.append)

        assert lines[0] == "Writing count_reads"
        assert lines[1].startswith("  Attempt 2 of 3, to correct: count_reads has no docstring")


class TestVerifyingInAnExecutor:
    def test_the_file_is_imported_in_the_executor_and_the_function_is_verified(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(code()))
        executor = StubExecutor()

        function = generate_function("Count reads", name="count_reads", model=TEST_MODEL, executor=executor, timeout=30)

        assert function.status == VERIFIED
        (run,) = executor.runs
        assert run["command"] == ("python3", f"./{CHECK_FILE}", "function/count_reads.py", "count_reads")
        assert run["files"]["function/count_reads.py"] == code() + "\n" and CHECK_FILE in run["files"]
        assert run["timeout"] == 30

    def test_code_that_cannot_be_read_is_not_run(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(code()))
        executor = StubExecutor()

        generate_function("Count reads", name="count_reads", model=TEST_MODEL, executor=executor)

        assert len(executor.runs) == 1

    def test_a_failed_import_is_given_back_to_the_model_with_what_the_executor_said(
        self, fake_client: FakeClient
    ) -> None:
        queue(fake_client, fenced(code()), fenced(code(extra="import json\n\n\n")))
        executor = StubExecutor(failed())

        function = generate_function("Count reads", name="count_reads", model=TEST_MODEL, executor=executor)

        correction = requests(fake_client)[1]["user"]
        assert "Importing the file failed. Execution FAILED with exit code 1" in correction
        assert "ModuleNotFoundError: No module named 'scanpy'" in correction
        assert function.status == VERIFIED and function.attempts == 2

    def test_an_import_that_fails_the_same_way_twice_is_not_tried_a_third_time(self, fake_client: FakeClient) -> None:
        queue(fake_client, *[fenced(code())] * 4)
        executor = StubExecutor(failed(), failed())

        with pytest.raises(FunctionGenerationError) as raised:
            generate_function("Count reads", name="count_reads", model=TEST_MODEL, executor=executor, max_attempts=4)

        assert len(fake_client.completions.calls) == 2 and raised.value.attempts == 2
        assert "Importing the file failed" in raised.value.problems[0]
        assert raised.value.source == code() + "\n"

    def test_an_import_that_fails_differently_is_corrected_again(self, fake_client: FakeClient) -> None:
        queue(fake_client, *[fenced(code())] * 3)
        executor = StubExecutor(failed("ModuleNotFoundError: No module named 'a'"), failed("ImportError: b"))

        function = generate_function("Count reads", name="count_reads", model=TEST_MODEL, executor=executor)

        assert function.attempts == 3 and function.status == VERIFIED

    def test_progress_says_when_the_import_is_made(self, fake_client: FakeClient) -> None:
        queue(fake_client, fenced(code()))
        lines: list[str] = []

        generate_function(
            "Count reads", name="count_reads", model=TEST_MODEL, executor=StubExecutor(), on_progress=lines.append
        )

        assert lines == ["Writing count_reads", "  Importing it in the executor"]

    def test_what_is_being_corrected_is_said_on_one_line_however_many_the_executor_wrote(
        self, fake_client: FakeClient
    ) -> None:
        queue(fake_client, fenced(code()), fenced(code()))
        lines: list[str] = []

        generate_function(
            "Count reads",
            name="count_reads",
            model=TEST_MODEL,
            executor=StubExecutor(failed("Traceback:\n  line 1\n\nImportError: no")),
            on_progress=lines.append,
        )

        (attempt,) = [line for line in lines if "Attempt 2" in line]
        assert "\n" not in attempt and attempt.startswith("  Attempt 2 of 3, to correct: Importing the file failed.")


class TestImportIn:
    """The check script, run for real by an executor that is this machine's own Python."""

    def run(self, source: str, name: str = "count_reads") -> ExecutionResult:
        return import_in(LocalExecutor(warn=False), source, name, timeout=60)

    def test_a_file_that_defines_the_function_is_imported(self) -> None:
        assert self.run(code()).succeeded

    def test_a_library_the_file_needs_and_this_machine_lacks_is_found_out(self) -> None:
        result = self.run(code(extra="import virtual_lab_no_such_library\n\n\n"))

        assert not result.succeeded and "ModuleNotFoundError" in result.stderr

    def test_code_that_fails_when_imported_is_found_out(self) -> None:
        result = self.run(code(extra="raise RuntimeError('on import')\n\n\n"))

        assert not result.succeeded and "RuntimeError: on import" in result.stderr

    def test_a_file_that_does_not_define_the_function_is_found_out(self) -> None:
        result = self.run(code("other"))

        assert not result.succeeded and "defines no function named count_reads" in result.stderr

    def test_a_file_named_for_a_library_is_not_taken_for_it(self) -> None:
        source = 'import json\nassert hasattr(json, "dumps")\n\n\ndef json():\n    return 1\n'

        assert self.run(source, "json").succeeded

    def test_the_file_is_not_left_behind(self) -> None:
        executor = StubExecutor()

        import_in(executor, code(), "count_reads")  # type: ignore[arg-type]

        assert executor.runs[0]["files"].keys() == {"function/count_reads.py", CHECK_FILE}
        assert not executor.runs[0]["directory"].exists()


class TestFunctionNames:
    def test_a_task_is_named_as_biomni_names_its_file(self) -> None:
        tasks = [FunctionTask.of("Count reads"), FunctionTask.of("Align reads to a genome with STAR now please")]

        assert function_names(tasks) == ["count_reads", "align_reads_to_a_genome_with"]

    def test_tasks_that_share_a_name_are_each_told_apart_by_a_few_characters_of_their_own(self) -> None:
        tasks = [
            FunctionTask.of("Align reads to the genome with STAR"),
            FunctionTask.of("Align reads to the genome with BWA"),
            FunctionTask.of("Count reads"),
        ]

        names = function_names(tasks)

        assert len(set(names)) == 3 and names[2] == "count_reads"
        assert names[0].startswith("align_reads_to_the_genome_with_") and names[0] != names[1]
        assert names == function_names(tasks)
        for name in names:
            check_function_name(name)

    def test_a_long_name_keeps_room_for_what_tells_it_apart(self) -> None:
        tasks = [FunctionTask.of("a" * 100 + " one"), FunctionTask.of("a" * 100 + " two")]

        names = function_names(tasks)

        assert len(set(names)) == 2 and all(len(name) <= 64 for name in names)


class TestSavingAFunction:
    def function(self, **fields: Any) -> GeneratedFunction:
        values: dict[str, Any] = {
            "task": FunctionTask("Count reads", "Count reads"),
            "name": "count_reads",
            "status": UNVERIFIED,
            "source": code() + "\n",
        }

        return GeneratedFunction(**{**values, **fields})

    def test_a_function_is_saved_as_a_file_and_a_record_that_says_what_the_file_was(self, tmp_path: Path) -> None:
        saved = save_function(tmp_path, self.function())

        assert (tmp_path / "count_reads.py").read_text() == code() + "\n"
        assert saved.digest == file_digest(tmp_path / "count_reads.py") != ""
        assert load_function(tmp_path, "count_reads") == saved

    def test_a_function_that_failed_is_saved_as_a_record_only(self, tmp_path: Path) -> None:
        saved = save_function(tmp_path, self.function(status=FAILED, error="No good"))

        assert not (tmp_path / "count_reads.py").exists() and saved.digest == ""
        assert load_function(tmp_path, "count_reads") == saved and saved.source == code() + "\n"

    def test_a_function_that_passed_has_code_to_save(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="passed, and has no code to save"):
            save_function(tmp_path, self.function(source=None))

        assert list(tmp_path.iterdir()) == []

    def test_the_file_is_written_before_its_record_so_a_stop_leaves_no_record_of_a_file_that_is_not_there(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def stop(path: Path, data: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("virtual_lab.function_generator.save_json", stop)

        with pytest.raises(KeyboardInterrupt):
            save_function(tmp_path, self.function())

        assert (tmp_path / "count_reads.py").is_file() and load_function(tmp_path, "count_reads") is None

    def test_a_record_that_is_missing_or_cannot_be_read_is_none(self, tmp_path: Path) -> None:
        assert load_function(tmp_path, "count_reads") is None
        for text in ("not json", "[]", '{"name": "x"}', "{}"):
            (tmp_path / "records").mkdir(exist_ok=True)
            (tmp_path / "records" / "count_reads.json").write_text(text)

            assert load_function(tmp_path, "count_reads") is None

    def test_every_record_is_listed_in_the_order_of_its_name_and_one_that_cannot_be_read_is_left_out(
        self, tmp_path: Path
    ) -> None:
        assert saved_functions(tmp_path) == []
        for name in ("b_task", "a_task"):
            save_function(tmp_path, self.function(name=name))
        (tmp_path / "records" / "c_task.json").write_text("garbage")

        assert [function.name for function in saved_functions(tmp_path)] == ["a_task", "b_task"]


class TestResumeDecision:
    def setup(self, tmp_path: Path, status: str = VERIFIED, file: str | None = "kept") -> GeneratedFunction:
        saved = GeneratedFunction(
            task=FunctionTask("Count reads", "Count reads"), name="count_reads", status=status, source=code() + "\n"
        )
        if status == FAILED:
            save_function(tmp_path, saved)
            if file is not None:
                (tmp_path / "count_reads.py").write_text(file)
            return load_function(tmp_path, "count_reads")  # type: ignore[return-value]
        saved = save_function(tmp_path, saved)
        if file is None:
            (tmp_path / "count_reads.py").unlink()
        elif file != "kept":
            (tmp_path / "count_reads.py").write_text(file)

        return saved

    def decide(
        self, tmp_path: Path, saved: GeneratedFunction | None, brief: str = "Count reads", **options: Any
    ) -> str:
        arguments = {"retry_failed": False, "has_executor": False, **options}

        return resume_decision(saved, FunctionTask("Count reads", brief), "count_reads", tmp_path, **arguments)

    def test_nothing_saved_is_written(self, tmp_path: Path) -> None:
        assert self.decide(tmp_path, None) == AGAIN

    def test_a_file_that_no_record_speaks_of_is_not_written_over(self, tmp_path: Path) -> None:
        (tmp_path / "count_reads.py").write_text("mine")

        assert self.decide(tmp_path, None) == CONFLICT

    def test_a_function_saved_for_the_same_task_is_reused(self, tmp_path: Path) -> None:
        assert self.decide(tmp_path, self.setup(tmp_path)) == REUSE

    def test_a_function_saved_without_being_run_is_run_only_if_there_is_something_to_run_it_in(
        self, tmp_path: Path
    ) -> None:
        saved = self.setup(tmp_path, UNVERIFIED)

        assert self.decide(tmp_path, saved) == REUSE
        assert self.decide(tmp_path, saved, has_executor=True) == VERIFY

    def test_a_verified_function_is_not_run_again(self, tmp_path: Path) -> None:
        assert self.decide(tmp_path, self.setup(tmp_path), has_executor=True) == REUSE

    def test_a_function_whose_file_is_gone_is_written_again(self, tmp_path: Path) -> None:
        assert self.decide(tmp_path, self.setup(tmp_path, file=None)) == AGAIN

    def test_a_function_that_failed_is_left_as_it_was_unless_asked_to_try_again(self, tmp_path: Path) -> None:
        saved = self.setup(tmp_path, FAILED, file=None)

        assert self.decide(tmp_path, saved) == REUSE
        assert self.decide(tmp_path, saved, retry_failed=True) == AGAIN

    def test_a_file_put_where_a_function_failed_is_not_written_over(self, tmp_path: Path) -> None:
        saved = self.setup(tmp_path, FAILED, file="mine")

        assert self.decide(tmp_path, saved, retry_failed=True) == CONFLICT

    def test_another_brief_is_written_again_over_a_file_that_is_as_it_was_saved(self, tmp_path: Path) -> None:
        assert self.decide(tmp_path, self.setup(tmp_path), brief="Count reads better") == AGAIN

    def test_another_brief_is_not_written_over_a_file_that_has_been_changed(self, tmp_path: Path) -> None:
        saved = self.setup(tmp_path, file="edited by hand")

        assert self.decide(tmp_path, saved, brief="Count reads better") == CONFLICT


def good(name: str) -> str:
    return fenced(code(name))


class TestGenerateFunctions:
    def directory(self, tmp_path: Path) -> Path:
        return tmp_path / "functions"

    def run(self, tmp_path: Path, tasks: list[Any], **options: Any) -> FunctionRunReport:
        return generate_functions(tasks, self.directory(tmp_path), model=TEST_MODEL, **{"max_attempts": 2, **options})

    def test_each_function_is_saved_with_its_record_and_the_run_with_a_report(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"), good("align_reads"))

        report = self.run(tmp_path, ["Count reads", "Align reads"])

        folder = self.directory(tmp_path)
        assert [result.name for result in report.results] == ["count_reads", "align_reads"]
        assert (folder / "count_reads.py").read_text() == code("count_reads") + "\n"
        assert (folder / "align_reads.py").read_text() == code("align_reads") + "\n"
        assert [function.name for function in saved_functions(folder)] == ["align_reads", "count_reads"]
        assert (report.tasks, report.verified, report.unverified, report.failed) == (2, 0, 2, 0)
        assert report.stopped is None and report.spent == pytest.approx(2 * REQUEST_COST)
        saved = json.loads((folder / "report.json").read_text())
        assert saved["finished"] == 2 and saved["unverified"] == 2 and saved["stopped"] is None
        assert [result["name"] for result in saved["results"]] == ["count_reads", "align_reads"]

    def test_a_second_run_asks_the_model_for_nothing_that_is_already_saved(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"), good("align_reads"))
        first = self.run(tmp_path, ["Count reads", "Align reads"])

        second = self.run(tmp_path, ["Count reads", "Align reads"])

        assert len(fake_client.completions.calls) == 2
        assert [r.digest for r in second.results] == [r.digest for r in first.results]
        assert second.spent == 0

    def test_a_run_carries_on_with_the_tasks_that_are_not_saved(self, tmp_path: Path, fake_client: FakeClient) -> None:
        queue(fake_client, good("count_reads"))
        self.run(tmp_path, ["Count reads"])
        before = (self.directory(tmp_path) / "count_reads.py").stat().st_mtime_ns
        queue(fake_client, good("align_reads"))

        report = self.run(tmp_path, ["Count reads", "Align reads"])

        assert len(fake_client.completions.calls) == 2
        assert [result.name for result in report.results] == ["count_reads", "align_reads"]
        assert (self.directory(tmp_path) / "count_reads.py").stat().st_mtime_ns == before

    def test_a_function_that_cannot_be_made_to_pass_is_saved_as_failed_with_its_last_code_and_the_run_goes_on(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, fenced(without_docstring()), fenced(with_a_path_hint()), good("align_reads"))

        report = self.run(tmp_path, ["Count reads", "Align reads"])

        folder = self.directory(tmp_path)
        record = load_function(folder, "count_reads")
        assert record is not None and record.status == FAILED and record.attempts == 2
        assert record.source == with_a_path_hint() + "\n" and "Path is not one of the types" in record.problems[0]
        assert record.error is not None and record.error.startswith("FunctionGenerationError: No function")
        assert record.cost == pytest.approx(2 * REQUEST_COST)
        assert not (folder / "count_reads.py").exists() and (folder / "align_reads.py").is_file()
        assert (report.failed, report.unverified) == (1, 1)

    def test_a_function_that_failed_is_not_written_again_unless_asked_to(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, "No code.", "Still none.")
        self.run(tmp_path, ["Count reads"])
        assert len(fake_client.completions.calls) == 2

        self.run(tmp_path, ["Count reads"])
        assert len(fake_client.completions.calls) == 2

        queue(fake_client, good("count_reads"))
        report = self.run(tmp_path, ["Count reads"], retry_failed=True)

        assert len(fake_client.completions.calls) == 3 and report.unverified == 1
        assert (self.directory(tmp_path) / "count_reads.py").is_file()

    def test_a_task_that_is_said_differently_is_written_again(self, tmp_path: Path, fake_client: FakeClient) -> None:
        queue(fake_client, good("count_reads"))
        self.run(tmp_path, ["Count reads"])
        queue(fake_client, fenced(code("count_reads", "path: str, minimum: int = 1")))

        report = self.run(tmp_path, [FunctionTask("Count reads", "Count reads, and only the long ones")])

        assert len(fake_client.completions.calls) == 2
        assert "minimum" in (self.directory(tmp_path) / "count_reads.py").read_text()
        assert report.results[0].task.brief == "Count reads, and only the long ones"

    def test_a_file_changed_by_hand_is_not_written_over_and_nothing_at_all_is_written_or_asked(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"))
        self.run(tmp_path, ["Count reads"])
        (self.directory(tmp_path) / "count_reads.py").write_text("mine")

        with pytest.raises(
            ValueError, match="holds count_reads.py, which this run did not write or has since been changed"
        ):
            self.run(tmp_path, [FunctionTask("Count reads", "Count reads differently"), "Align reads"])

        assert len(fake_client.completions.calls) == 1
        assert (self.directory(tmp_path) / "count_reads.py").read_text() == "mine"
        assert not (self.directory(tmp_path) / "align_reads.py").exists()

    def test_a_file_with_no_record_is_not_written_over(self, tmp_path: Path, fake_client: FakeClient) -> None:
        self.directory(tmp_path).mkdir()
        (self.directory(tmp_path) / "count_reads.py").write_text("mine")

        with pytest.raises(ValueError, match="holds count_reads.py"):
            self.run(tmp_path, ["Count reads"])

        assert (
            fake_client.completions.calls == [] and (self.directory(tmp_path) / "count_reads.py").read_text() == "mine"
        )
        assert not (self.directory(tmp_path) / "report.json").exists()

    def test_many_conflicts_are_listed_by_a_few(self, tmp_path: Path, fake_client: FakeClient) -> None:
        self.directory(tmp_path).mkdir()
        names = [f"task_number_{letter}" for letter in "abcdefg"]
        for name in names:
            (self.directory(tmp_path) / f"{name}.py").write_text("mine")

        with pytest.raises(ValueError, match="task_number_e.py, and 2 more, which"):
            self.run(tmp_path, [name.replace("_", " ") for name in names])

    def test_tasks_whose_first_words_are_the_same_are_each_written_to_a_file_of_their_own(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        tasks = [
            FunctionTask.of("Align reads to the genome with STAR"),
            FunctionTask.of("Align reads to the genome with BWA"),
        ]
        first, second = function_names(tasks)
        queue(fake_client, good(first), good(second))

        report = self.run(tmp_path, tasks)

        assert {path.name for path in self.directory(tmp_path).glob("*.py")} == {f"{first}.py", f"{second}.py"}
        assert [result.task.name for result in report.results] == [task.name for task in tasks]

    def test_the_same_task_twice_is_refused_before_anything_is_asked(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        with pytest.raises(ValueError, match="Two tasks have the same name"):
            self.run(tmp_path, ["Count reads", "Count reads"])

        assert fake_client.completions.calls == [] and not self.directory(tmp_path).exists()

    def test_the_run_stops_when_it_has_spent_its_limit_and_what_it_wrote_is_kept(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"), good("align_reads"))

        report = self.run(tmp_path, ["Count reads", "Align reads"], max_cost=REQUEST_COST / 2)

        assert len(fake_client.completions.calls) == 1 and [r.name for r in report.results] == ["count_reads"]
        assert report.stopped is not None and "reaches its max_cost of" in report.stopped
        assert (self.directory(tmp_path) / "count_reads.py").is_file()
        assert not (self.directory(tmp_path) / "align_reads.py").exists()
        assert json.loads((self.directory(tmp_path) / "report.json").read_text())["stopped"] == report.stopped

    def test_a_function_the_limit_stopped_is_left_unsaved_so_that_the_next_run_writes_it(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, fenced(without_docstring()), good("count_reads"))

        report = self.run(tmp_path, ["Count reads"], max_cost=REQUEST_COST / 2)

        assert (
            report.results == []
            and report.stopped is not None
            and "while writing count_reads, which is left unfinished" in report.stopped
        )
        assert load_function(self.directory(tmp_path), "count_reads") is None
        assert report.spent == pytest.approx(REQUEST_COST)

        again = self.run(tmp_path, ["Count reads"])

        assert again.unverified == 1

    def test_a_function_that_reaches_its_own_limit_fails_and_the_run_goes_on(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, fenced(without_docstring()), good("align_reads"))

        report = self.run(tmp_path, ["Count reads", "Align reads"], max_cost_per_function=REQUEST_COST / 2)

        record = load_function(self.directory(tmp_path), "count_reads")
        assert record is not None and record.status == FAILED
        assert record.error is not None and record.error.startswith(
            "reached its limit on spending: FunctionBudgetExceededError"
        )
        assert record.cost == pytest.approx(REQUEST_COST) and record.source == without_docstring() + "\n"
        assert (report.failed, report.unverified, report.stopped) == (1, 1, None)

    def test_several_failures_in_a_row_stop_the_run(self, tmp_path: Path, fake_client: FakeClient) -> None:
        queue(fake_client, *["No code."] * 6)

        report = self.run(tmp_path, ["Task one", "Task two", "Task three", "Task four"], max_attempts=1)

        assert len(fake_client.completions.calls) == 3 and report.failed == 3
        assert report.stopped is not None and report.stopped.startswith("3 functions in a row failed, the last with ")

    def test_a_success_between_failures_starts_the_count_again(self, tmp_path: Path, fake_client: FakeClient) -> None:
        queue(fake_client, "No.", "No.", good("task_three"), "No.", "No.")

        report = self.run(tmp_path, ["Task one", "Task two", "Task three", "Task four", "Task five"], max_attempts=1)

        assert report.stopped is None and (report.failed, report.unverified) == (4, 1)

    def test_a_run_may_be_told_never_to_stop_for_failures(self, tmp_path: Path, fake_client: FakeClient) -> None:
        queue(fake_client, *["No."] * 5)

        report = self.run(tmp_path, [f"Task {n}" for n in "abcde"], max_attempts=1, max_consecutive_failures=None)

        assert report.failed == 5 and report.stopped is None

    def test_an_error_that_is_not_about_the_task_is_reported_and_not_remembered(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, RuntimeError("the key is wrong"), good("align_reads"))

        report = self.run(tmp_path, ["Count reads", "Align reads"])

        first = report.results[0]
        assert first.status == FAILED and first.error == "RuntimeError: the key is wrong"
        assert load_function(self.directory(tmp_path), "count_reads") is None
        assert not (self.directory(tmp_path) / "count_reads.py").exists()
        queue(fake_client, good("count_reads"))

        assert self.run(tmp_path, ["Count reads", "Align reads"]).unverified == 2

    def test_errors_that_are_not_about_the_task_count_towards_stopping_the_run(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, *[RuntimeError("down")] * 4)

        report = self.run(tmp_path, ["Task one", "Task two", "Task three", "Task four"])

        assert (
            len(fake_client.completions.calls) == 3
            and report.stopped is not None
            and "RuntimeError: down" in report.stopped
        )
        assert saved_functions(self.directory(tmp_path)) == []

    def test_an_interrupt_still_leaves_a_report_of_what_was_written(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"), KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            self.run(tmp_path, ["Count reads", "Align reads"])

        report = json.loads((self.directory(tmp_path) / "report.json").read_text())
        assert report["finished"] == 1 and report["tasks"] == 2
        assert (self.directory(tmp_path) / "count_reads.py").is_file()

    def test_an_executor_that_cannot_run_fails_the_run_before_anything_is_saved(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        executor = StubExecutor(unavailable=True)

        with pytest.raises(ExecutionError, match="Docker is not running"):
            self.run(tmp_path, ["Count reads"], executor=executor)

        assert fake_client.completions.calls == [] and not self.directory(tmp_path).exists()

    def test_with_an_executor_each_function_is_imported_and_verified(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"), good("align_reads"))
        executor = StubExecutor(succeeded(), succeeded())

        report = self.run(tmp_path, ["Count reads", "Align reads"], executor=executor, timeout=5)

        assert executor.checked == 1 and len(executor.runs) == 2 and executor.runs[0]["timeout"] == 5
        assert report.verified == 2 and load_function(self.directory(tmp_path), "align_reads").status == VERIFIED  # type: ignore[union-attr]

    def test_functions_saved_without_being_run_are_run_later_without_asking_the_model_again(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"))
        first = self.run(tmp_path, ["Count reads"])
        executor = StubExecutor()

        second = self.run(tmp_path, ["Count reads"], executor=executor)

        assert len(fake_client.completions.calls) == 1 and len(executor.runs) == 1
        assert second.verified == 1 and second.spent == 0
        assert load_function(self.directory(tmp_path), "count_reads").status == VERIFIED  # type: ignore[union-attr]
        assert second.results[0].digest == first.results[0].digest

    def test_a_function_that_does_not_import_is_written_again(self, tmp_path: Path, fake_client: FakeClient) -> None:
        queue(fake_client, good("count_reads"))
        self.run(tmp_path, ["Count reads"])
        queue(fake_client, fenced(code("count_reads", "path: str, n: int = 1")))
        executor = StubExecutor(failed(), succeeded())

        report = self.run(tmp_path, ["Count reads"], executor=executor)

        assert len(fake_client.completions.calls) == 2 and report.verified == 1
        assert "n: int" in (self.directory(tmp_path) / "count_reads.py").read_text()

    def test_a_function_changed_by_hand_that_does_not_import_is_left_as_it_is(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"))
        self.run(tmp_path, ["Count reads"])
        edited = code("count_reads", extra="import not_installed\n\n\n") + "\n"
        (self.directory(tmp_path) / "count_reads.py").write_text(edited)
        lines: list[str] = []

        report = self.run(tmp_path, ["Count reads"], executor=StubExecutor(failed()), on_progress=lines.append)

        assert len(fake_client.completions.calls) == 1 and report.unverified == 1
        assert (self.directory(tmp_path) / "count_reads.py").read_text() == edited
        assert any(
            "has been changed since it was saved, and does not pass, so it is left as it is" in line for line in lines
        )

    def test_a_function_changed_by_hand_that_imports_is_verified_as_it_now_is(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"))
        self.run(tmp_path, ["Count reads"])
        edited = code("count_reads", extra="import json\n\n\n") + "\n"
        (self.directory(tmp_path) / "count_reads.py").write_text(edited)

        report = self.run(tmp_path, ["Count reads"], executor=StubExecutor())

        (result,) = report.results
        assert result.status == VERIFIED and result.source == edited
        assert result.digest == file_digest(self.directory(tmp_path) / "count_reads.py")

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"max_cost": float("nan")}, "max_cost must be a finite amount"),
            ({"max_cost_per_function": -1.0}, "max_cost_per_function must be a finite amount"),
            ({"max_cost_per_function": float("inf")}, "max_cost_per_function must be a finite amount"),
            ({"max_consecutive_failures": 0}, "max_consecutive_failures must be at least 1, not 0"),
            ({"max_attempts": 0}, "max_attempts must be at least 1"),
        ],
    )
    def test_an_option_that_would_fail_every_task_is_refused_before_any_is_tried(
        self, tmp_path: Path, fake_client: FakeClient, options: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            self.run(tmp_path, ["Count reads"], **options)

        assert fake_client.completions.calls == [] and not self.directory(tmp_path).exists()

    def test_a_limit_on_spending_for_a_model_with_no_price_is_refused(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        with pytest.raises(CostUnknownError, match="so a limit on spending cannot be enforced"):
            generate_functions(
                ["Count reads"], self.directory(tmp_path), model="no-such-model", max_cost_per_function=1.0
            )

    def test_what_the_run_uses_is_added_to_the_count_a_caller_keeps(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"))
        usage = MeetingUsage()

        self.run(tmp_path, ["Count reads"], usage=usage)

        assert usage.compute_cost() == pytest.approx(REQUEST_COST)

    def test_what_a_caller_has_already_spent_counts_towards_the_runs_limit_but_not_its_own_spending(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage(1_000_000, 0))

        report = self.run(tmp_path, ["Count reads"], usage=usage, max_cost=1.0)

        assert fake_client.completions.calls == [] and report.spent == 0
        assert report.stopped is not None and "reaches its max_cost" in report.stopped

    def test_progress_says_which_function_of_how_many(self, tmp_path: Path, fake_client: FakeClient) -> None:
        queue(fake_client, good("count_reads"), fenced(without_docstring("align_reads")), good("align_reads"))
        lines: list[str] = []

        self.run(tmp_path, ["Count reads", "Align reads"], on_progress=lines.append)

        assert lines[0] == "Function 1 of 2: count_reads" and "  Writing count_reads" in lines
        assert "Function 2 of 2: align_reads" in lines and any(line.startswith("  Attempt 2 of 2") for line in lines)

    def test_a_task_may_be_a_function_task_with_what_is_known_of_it(
        self, tmp_path: Path, fake_client: FakeClient
    ) -> None:
        queue(fake_client, good("count_reads"))
        task = FunctionTask(
            "Count reads", brief_of("Count reads", {"inputs": "A BAM file.", "outputs": "A number."}), 4
        )

        report = self.run(tmp_path, [task])

        assert "What it takes: A BAM file.\nWhat it gives: A number." in requests(fake_client)[0]["user"]
        assert report.results[0].task.papers == 4


class TestRunReport:
    def test_it_counts_what_was_written_and_what_it_cost(self, tmp_path: Path) -> None:
        task = FunctionTask("a", "a")
        results = [
            GeneratedFunction(task=task, name="a", status=VERIFIED, cost=0.25),
            GeneratedFunction(task=task, name="b", status=UNVERIFIED, cost=0.5),
            GeneratedFunction(task=task, name="c", status=FAILED, cost=0.125, error="No good"),
        ]

        report = FunctionRunReport(results=results, tasks=4, spent=0.875, stopped="Out of money", save_dir=tmp_path)

        assert (report.verified, report.unverified, report.failed, report.cost) == (1, 1, 1, 0.875)
        assert report.to_dict() == {
            "tasks": 4,
            "finished": 3,
            "verified": 1,
            "unverified": 1,
            "failed": 1,
            "cost": 0.875,
            "spent": 0.875,
            "stopped": "Out of money",
            "results": [
                {"name": "a", "status": VERIFIED, "cost": 0.25, "error": None},
                {"name": "b", "status": UNVERIFIED, "cost": 0.5, "error": None},
                {"name": "c", "status": FAILED, "cost": 0.125, "error": "No good"},
            ],
        }

    def test_a_cost_that_cannot_be_worked_out_makes_the_whole_unknown(self, tmp_path: Path) -> None:
        task = FunctionTask("a", "a")
        results = [GeneratedFunction(task=task, name="a", status=VERIFIED, cost=None)]

        assert FunctionRunReport(results, 1, None, None, tmp_path).cost is None


def task_in_papers(name: str, **fields: str) -> PaperTask:
    values = {
        "description": "Finds genes that differ.",
        "inputs": "A count matrix.",
        "outputs": "A table of results.",
        "code_implementation": "pydeseq2.",
        "frequency": "Very common.",
        "standard_methods": "Negative binomial models.",
        "example": "Used on the tumour samples.",
    }

    return PaperTask(task_name=name, **{**values, **fields})


def read_papers(directory: Path, *tasks_by_paper: list[PaperTask]) -> None:
    for number, tasks in enumerate(tasks_by_paper):
        paper = Paper(key=f"p{number}", title=f"Paper {number}", doi=f"10.1/{number}")
        findings = PaperFindings(tasks=tasks, databases=[], software=[])
        save_json(
            directory / "results" / f"p{number}.json",
            PaperResult(paper=paper, status="read", findings=findings).to_dict(),
        )
    summarize_papers(directory)


class TestFunctionTasks:
    def test_biomnis_file_of_descriptions_gives_one_task_for_each(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text(json.dumps({"tasks": ["Count reads in a BAM file", "  Align   reads "]}))

        tasks = function_tasks(path)

        assert tasks == [
            FunctionTask("Count reads in a BAM file", "Count reads in a BAM file"),
            FunctionTask("Align reads", "Align   reads"),
        ]

    def test_a_list_or_what_a_file_holds_is_as_good_as_a_file(self) -> None:
        assert (
            function_tasks(["Count reads"])
            == function_tasks({"tasks": ["Count reads"]})
            == [FunctionTask("Count reads", "Count reads")]
        )
        assert function_tasks(("Count reads",)) == function_tasks(["Count reads"])

    def test_a_summary_of_names_and_counts_gives_the_tasks_in_order_with_the_papers_they_were_found_in(self) -> None:
        tasks = function_tasks({"tasks": {"Align reads": 5, "Count reads": 3, "Plot it": 1}})

        assert [(task.name, task.papers) for task in tasks] == [("Align reads", 5), ("Count reads", 3), ("Plot it", 1)]

    def test_only_the_tasks_in_enough_papers_are_taken_and_a_count_that_is_not_known_is_not_held_against_one(
        self,
    ) -> None:
        source = {"tasks": {"Align reads": 5, "Count reads": 1, "Odd": "many", "Flag": True}}

        assert [task.name for task in function_tasks(source, min_papers=2)] == ["Align reads", "Odd", "Flag"]
        assert [task.name for task in function_tasks(source, min_papers=6)] == ["Odd", "Flag"]
        assert [task.name for task in function_tasks(source, min_papers=1)] == [
            "Align reads",
            "Count reads",
            "Odd",
            "Flag",
        ]

    def test_only_the_first_few_are_taken_if_asked(self) -> None:
        source = {"tasks": {"One": 3, "Two": 2, "Three": 1}}

        assert [task.name for task in function_tasks(source, limit=2)] == ["One", "Two"]
        assert function_tasks(source, limit=0) == []
        assert function_tasks(source, min_papers=2, limit=5) == function_tasks(source, min_papers=2)

    def test_an_object_says_what_the_task_does_takes_and_gives(self) -> None:
        source = {
            "tasks": [{"task_name": "Align reads", "description": "Aligns.", "inputs": "FASTQ.", "outputs": "BAM."}]
        }

        (task,) = function_tasks(source)

        assert task.name == "Align reads"
        assert task.brief == "Align reads\nWhat it does: Aligns.\nWhat it takes: FASTQ.\nWhat it gives: BAM."

    def test_an_object_may_be_named_by_name_instead(self) -> None:
        assert function_tasks([{"name": "Align reads", "description": "Aligns."}])[0].name == "Align reads"

    def test_tasks_that_differ_only_as_names_are_one_and_the_first_is_kept(self) -> None:
        tasks = function_tasks(["DESeq2 analysis", "deseq2 Analysis", "Other"])

        assert [task.name for task in tasks] == ["DESeq2 analysis", "Other"]

    def test_blank_tasks_are_passed_over(self) -> None:
        assert [task.name for task in function_tasks(["  ", "A task", ""])] == ["A task"]

    def test_a_directory_of_papers_gives_its_commonest_tasks_with_what_the_papers_say_of_them(
        self, tmp_path: Path
    ) -> None:
        read_papers(
            tmp_path,
            [task_in_papers("DESeq2 analysis", description="Short."), task_in_papers("PCA")],
            [task_in_papers("deseq2 analysis", description="A much longer account of what it does.")],
        )

        tasks = function_tasks(tmp_path)

        assert [(task.name, task.papers) for task in tasks] == [("DESeq2 analysis", 2), ("PCA", 1)]
        assert "What it does: A much longer account of what it does." in tasks[0].brief
        assert (
            "What it takes: A count matrix." in tasks[0].brief and "How it is implemented: pydeseq2." in tasks[0].brief
        )
        assert [task.name for task in function_tasks(tmp_path, min_papers=2)] == ["DESeq2 analysis"]

    def test_a_directory_whose_papers_were_combined_gives_the_combined_tasks(self, tmp_path: Path) -> None:
        read_papers(tmp_path, [task_in_papers("PCA")])
        (tmp_path / COMBINED_FILE).write_text(json.dumps({"tasks": {"Combined task": 9}}))

        assert [(task.name, task.papers) for task in function_tasks(tmp_path)] == [("Combined task", 9)]

    def test_what_the_papers_in_a_directory_below_say_is_used_too(self, tmp_path: Path) -> None:
        read_papers(tmp_path / "run_one", [task_in_papers("PCA", description="Reduces dimensions.")])
        (tmp_path / "tasks.json").write_text("{}")
        (tmp_path / FREQUENCY_FILE).write_text(json.dumps({"tasks": {"PCA": 1}}))

        (task,) = function_tasks(tmp_path)

        assert "What it does: Reduces dimensions." in task.brief

    def test_a_directory_with_no_summary_says_how_to_make_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=f"has no {FREQUENCY_FILE} or {COMBINED_FILE}. Read papers into it"):
            function_tasks(tmp_path)

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            ({"jobs": []}, 'The tasks are under "tasks"'),
            ({"tasks": 5}, "The tasks are a list, or a mapping of names to counts"),
            ("just words", "is neither a file nor a directory"),
            ([5], "A task is a description or an object, not int"),
            ([{"description": "x"}], "A task that is an object needs a task_name or a name"),
        ],
    )
    def test_a_source_that_is_not_tasks_says_what_is_wrong_with_it(self, source: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            function_tasks(source)

    def test_a_file_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "tasks.json").write_text("{not json")

        with pytest.raises(ValueError, match="tasks.json cannot be read"):
            function_tasks(tmp_path / "tasks.json")

    def test_a_task_too_long_is_refused_with_its_name(self) -> None:
        with pytest.raises(ValueError, match="more than the"):
            function_tasks(["a" * 20_000])


class TestUniqueTasks:
    def test_the_first_of_tasks_with_one_name_is_kept_across_lists(self) -> None:
        first = function_tasks(["Align reads", "Count reads"])
        second = function_tasks(["align  READS", "Plot"])

        assert [task.name for task in unique_tasks([*first, *second])] == ["Align reads", "Count reads", "Plot"]
