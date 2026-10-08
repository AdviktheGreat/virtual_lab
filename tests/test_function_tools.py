"""Tests for making tools of the functions written for tasks, and running them."""

import ast
import json
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from virtual_lab.function_generator import (
    FAILED,
    UNVERIFIED,
    VERIFIED,
    FunctionTask,
    GeneratedFunction,
    save_function,
)
from virtual_lab.function_tools import (
    SESSION_CACHE,
    FunctionToolError,
    tool_from_source,
    tools_from_saved_functions,
    run_in_session,
    session_call_code,
)
from virtual_lab.session import CellResult, LocalSession

SOURCE = '''"""Reads files."""
import json

CALLS = []


def count_lines(path: str, skip: int = 0, label: str | None = None) -> dict:
    """Counts the lines in a file.

    Args:
        path: The file to read.
        skip: How many lines to leave out.
        label: What to call the count.
    """
    CALLS.append(path)
    with open(path) as file:
        lines = file.read().splitlines()[skip:]
    print("counted")
    return {"lines": len(lines), "label": label, "calls": len(CALLS), "path_type": type(path).__name__}


def main():
    print(count_lines(__file__))


if __name__ == "__main__":
    main()
'''


def cell(value: Any = None, output: str = "", error: str | None = None, status: str = "ok") -> CellResult:
    return CellResult(language="python", code="", status=status, output=output, error=error, duration=0.1, value=value)


class SpySession:
    """Stands in for a session, keeping the code it was asked to run and answering from a queue."""

    def __init__(self, *results: CellResult) -> None:
        self.results = list(results)
        self.codes: list[str] = []

    def evaluate(self, code: str, timeout: float | None = None) -> CellResult:
        self.codes.append(code)

        return self.results.pop(0) if self.results else cell()


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LocalSession]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with LocalSession(tmp_path_factory.mktemp("work"), warn=False, timeout=60) as opened:
            yield opened


@pytest.fixture
def data(tmp_path: Path) -> Path:
    path = tmp_path / "data.txt"
    path.write_text("a\nb\nc\n")

    return path


class TestAToolFromAFunction:
    def test_it_is_named_described_and_given_parameters_as_the_function_is(self) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=SpySession())  # type: ignore[arg-type]

        assert tool.name == "count_lines" and tool.description.startswith("Counts the lines in a file.")
        assert tool.parameters["required"] == ["path"]
        properties = tool.parameters["properties"]
        assert properties["path"]["type"] == "string" and "The file to read." in properties["path"]["description"]
        assert properties["skip"]["type"] == "integer" and properties["skip"]["default"] == 0

    def test_making_it_runs_nothing(self) -> None:
        spy = SpySession()

        tool_from_source(SOURCE, "count_lines", session=spy)  # type: ignore[arg-type]

        assert spy.codes == []

    def test_it_must_be_said_where_the_function_runs(self) -> None:
        with pytest.raises(ValueError, match="Say where the function runs"):
            tool_from_source(SOURCE, "count_lines")
        with pytest.raises(ValueError, match="Not both"):
            tool_from_source(SOURCE, "count_lines", session=SpySession(), run_here=True)  # type: ignore[arg-type]

    def test_code_that_cannot_be_a_tool_is_refused_with_why(self) -> None:
        with pytest.raises(ValueError, match="count_lines cannot be a tool: .*defines no function named count_lines"):
            tool_from_source("def other():\n    pass\n", "count_lines", run_here=True)

    def test_a_call_with_arguments_that_are_not_of_the_types_is_refused_before_the_function_is_reached(self) -> None:
        spy = SpySession()
        tool = tool_from_source(SOURCE, "count_lines", session=spy)  # type: ignore[arg-type]

        with pytest.raises(Exception, match="skip"):
            tool.function(path="x", skip="many")

        assert spy.codes == []


class TestCallingInASession:
    def test_the_function_is_called_with_the_arguments_as_json_in_code_that_ends_in_its_value(self) -> None:
        spy = SpySession(cell({"lines": 2}))
        tool = tool_from_source(SOURCE, "count_lines", session=spy)  # type: ignore[arg-type]

        value = tool.function(path='it\'s "here"\n', skip=1)

        assert value == {"lines": 2}
        (code,) = spy.codes
        tree = ast.parse(code)
        final = tree.body[-1]
        assert isinstance(final, ast.Expr) and isinstance(final.value, ast.Call)
        (spread,) = final.value.keywords
        assert spread.arg is None and isinstance(spread.value, ast.Call)
        assert json.loads(ast.literal_eval(spread.value.args[0])) == {"path": 'it\'s "here"\n', "skip": 1}

    def test_what_a_function_prints_is_what_it_gave_if_it_returned_nothing(self) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=SpySession(cell(None, output="3 lines\n")))  # type: ignore[arg-type]

        assert tool.function(path="x") == "3 lines\n"

    def test_a_function_that_returned_nothing_and_printed_nothing_gives_none(self) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=SpySession(cell()))  # type: ignore[arg-type]

        assert tool.function(path="x") is None

    def test_a_value_is_not_replaced_by_what_was_printed(self) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=SpySession(cell(0, output="noise")))  # type: ignore[arg-type]

        assert tool.function(path="x") == 0

    def test_a_failure_raises_with_its_error_and_the_end_of_what_was_printed(self) -> None:
        spy = SpySession(
            cell(error="FileNotFoundError: no such file", output="a\n" * 5_000 + "last line\n", status="error")
        )
        tool = tool_from_source(SOURCE, "count_lines", session=spy)  # type: ignore[arg-type]

        with pytest.raises(FunctionToolError) as raised:
            tool.function(path="x")

        message = str(raised.value)
        assert message.startswith("count_lines failed: FileNotFoundError: no such file\n\nWhat it printed:\n")
        assert message.endswith("last line") and len(message) < 2_300

    def test_a_failure_that_printed_nothing_says_only_the_error(self) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=SpySession(cell(error="Timeout", status="timeout")))  # type: ignore[arg-type]

        with pytest.raises(FunctionToolError) as raised:
            tool.function(path="x")

        assert str(raised.value) == "count_lines failed: Timeout"

    def test_arguments_that_cannot_be_sent_as_json_are_refused(self) -> None:
        spy = SpySession()
        run = run_in_session(spy, "count_lines", SOURCE)  # type: ignore[arg-type]

        with pytest.raises(
            FunctionToolError, match="count_lines was called with arguments that cannot be sent as JSON"
        ):
            run(path=object())

        assert spy.codes == []

    def test_the_code_defines_the_function_once_in_a_module_of_its_own(self) -> None:
        code = session_call_code("count_lines", SOURCE, {"path": "x"})

        assert code.count(SESSION_CACHE) == 3 and "virtual_lab_generated_count_lines" in code
        assert "globals().setdefault" in code

    def test_the_code_names_the_source_it_defines_the_function_from(self) -> None:
        corrected = SOURCE.replace("return {", "return {'x': 1, ")

        assert session_call_code("count_lines", SOURCE, {}) != session_call_code("count_lines", corrected, {})
        assert session_call_code("count_lines", SOURCE, {}) == session_call_code("count_lines", SOURCE, {})


class TestCallingInASessionThatRuns:
    """The same, in a session that runs code, which is this machine's own."""

    def test_a_function_is_called_with_json_and_returns_json(self, session: LocalSession, data: Path) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=session)

        value = tool.function(path=str(data), skip=1, label="x")

        assert value["lines"] == 2 and value["label"] == "x" and value["path_type"] == "str"

    def test_it_is_defined_once_and_adds_nothing_to_the_names_of_the_session(
        self, session: LocalSession, data: Path
    ) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=session)

        first, second = tool.function(path=str(data)), tool.function(path=str(data))

        assert second["calls"] == first["calls"] + 1
        names = session.evaluate("sorted(name for name in globals() if not name.startswith('_'))").value
        assert "count_lines" not in names and "CALLS" not in names and "json" not in names

    def test_a_function_that_was_corrected_is_defined_again_in_a_session_that_is_still_open(
        self, session: LocalSession, data: Path
    ) -> None:
        name = "count_corrected"
        before = SOURCE.replace("count_lines", name)
        after = before.replace('"lines": len(lines)', '"lines": len(lines) * 10')

        first = tool_from_source(before, name, session=session).function(path=str(data))
        corrected = tool_from_source(after, name, session=session).function(path=str(data))
        again = tool_from_source(after, name, session=session).function(path=str(data))

        assert first["lines"] == 3 and first["calls"] == 1
        assert corrected["lines"] == 30 and corrected["calls"] == 1
        assert again["lines"] == 30 and again["calls"] == 2

    def test_a_function_whose_correction_fails_to_define_is_tried_again_and_not_taken_for_defined(
        self, session: LocalSession, data: Path
    ) -> None:
        name = "count_unfixed"
        before = SOURCE.replace("count_lines", name)
        broken = before.replace("import json", "import no_such_library_anywhere")
        tool_from_source(before, name, session=session).function(path=str(data))

        for _ in range(2):
            with pytest.raises(FunctionToolError, match="no_such_library_anywhere"):
                tool_from_source(broken, name, session=session).function(path=str(data))

    def test_what_a_function_printed_before_it_failed_is_in_the_error(
        self, session: LocalSession, tmp_path: Path
    ) -> None:
        tool = tool_from_source(SOURCE, "count_lines", session=session)

        with pytest.raises(FunctionToolError) as raised:
            tool.function(path=str(tmp_path / "missing.txt"))

        assert "count_lines failed:" in str(raised.value) and "FileNotFoundError" in str(raised.value)

    def test_two_functions_do_not_share_a_module(self, session: LocalSession, data: Path) -> None:
        other = SOURCE.replace("count_lines", "count_more").replace("CALLS", "OTHER_CALLS")
        first = tool_from_source(SOURCE, "count_lines", session=session)
        second = tool_from_source(other, "count_more", session=session)

        assert first.function(path=str(data))["calls"] >= 1 and second.function(path=str(data))["calls"] == 1


class TestCallingHere:
    def test_it_is_called_with_the_arguments_converted_and_run_in_this_process(self, data: Path) -> None:
        tool = tool_from_source(SOURCE, "count_lines", run_here=True, filename="count_lines.py")

        value = tool.function(path=str(data), skip="1")

        assert value["lines"] == 2 and value["path_type"] == "str"

    def test_it_is_defined_when_first_called_and_not_before_and_only_once(self, data: Path) -> None:
        name = "count_lines_here"
        tool = tool_from_source(SOURCE.replace("count_lines", name), name, run_here=True)
        module = f"virtual_lab_generated_{name}"

        assert module not in sys.modules
        first, second = tool.function(path=str(data)), tool.function(path=str(data))

        assert module in sys.modules and second["calls"] == first["calls"] + 1
        del sys.modules[module]

    def test_code_that_fails_when_it_is_defined_fails_the_call_and_leaves_no_module(self, data: Path) -> None:
        name = "count_lines_broken"
        source = SOURCE.replace("count_lines", name).replace("import json", "import no_such_library_anywhere")
        tool = tool_from_source(source, name, run_here=True)

        for _ in range(2):
            with pytest.raises(ModuleNotFoundError, match="no_such_library_anywhere"):
                tool.function(path=str(data))

        assert f"virtual_lab_generated_{name}" not in sys.modules

    def test_the_file_is_named_in_the_module_so_that_what_reads_it_finds_it(self) -> None:
        body = '    """Says where it is.\n\n    Args:\n        path: Not used.\n    """\n    return __file__\n'
        named = tool_from_source(
            f"def where_am_i(path: str) -> str:\n{body}", "where_am_i", run_here=True, filename="/s/w.py"
        )
        unnamed = tool_from_source(f"def where_too(path: str) -> str:\n{body}", "where_too", run_here=True)

        assert named.function(path="x") == "/s/w.py" and unnamed.function(path="x") == "where_too.py"
        for name in ("where_am_i", "where_too"):
            sys.modules.pop(f"virtual_lab_generated_{name}", None)

    def test_an_error_in_the_function_is_the_functions_error(self, tmp_path: Path) -> None:
        name = "count_lines_failing"
        tool = tool_from_source(SOURCE.replace("count_lines", name), name, run_here=True, filename="/x/count.py")

        with pytest.raises(FileNotFoundError):
            tool.function(path=str(tmp_path / "missing"))

        sys.modules.pop(f"virtual_lab_generated_{name}", None)


class TestToolsFromADirectory:
    def save(self, directory: Path, name: str, status: str = UNVERIFIED, source: str | None = None) -> None:
        text = source if source is not None else SOURCE.replace("count_lines", name)
        function = GeneratedFunction(
            task=FunctionTask(name, name), name=name, status=status, source=None if status == FAILED else text
        )
        save_function(directory, function)

    def test_each_function_saved_is_a_tool_in_the_order_of_their_names(self, tmp_path: Path) -> None:
        for name in ("b_task", "a_task"):
            self.save(tmp_path, name)

        tools = tools_from_saved_functions(tmp_path, session=SpySession())  # type: ignore[arg-type]

        assert [tool.name for tool in tools] == ["a_task", "b_task"]

    def test_a_function_that_failed_is_not_a_tool(self, tmp_path: Path) -> None:
        self.save(tmp_path, "a_task")
        self.save(tmp_path, "b_task", FAILED)

        assert [tool.name for tool in tools_from_saved_functions(tmp_path, run_here=True)] == ["a_task"]

    def test_only_the_functions_that_were_run_may_be_asked_for(self, tmp_path: Path) -> None:
        self.save(tmp_path, "a_task", VERIFIED)
        self.save(tmp_path, "b_task", UNVERIFIED)

        assert [tool.name for tool in tools_from_saved_functions(tmp_path, run_here=True, verified_only=True)] == [
            "a_task"
        ]

    def test_functions_may_be_chosen_by_name(self, tmp_path: Path) -> None:
        for name in ("a_task", "b_task", "c_task"):
            self.save(tmp_path, name)

        assert [
            tool.name for tool in tools_from_saved_functions(tmp_path, run_here=True, names=["c_task", "a_task"])
        ] == [
            "a_task",
            "c_task",
        ]
        assert [tool.name for tool in tools_from_saved_functions(tmp_path, run_here=True, names="b_task")] == ["b_task"]

    def test_a_name_that_is_not_a_saved_function_is_refused_with_all_of_them(self, tmp_path: Path) -> None:
        self.save(tmp_path, "a_task")
        self.save(tmp_path, "b_task", FAILED)

        with pytest.raises(ValueError, match=r"holds no function 'b_task', 'c_task' that can be a tool"):
            tools_from_saved_functions(tmp_path, run_here=True, names=["a_task", "b_task", "c_task"])

    def test_the_function_is_the_file_as_it_is_now_so_a_correction_by_hand_is_used(self, tmp_path: Path) -> None:
        self.save(tmp_path, "a_task")
        path = tmp_path / "a_task.py"
        path.write_text(path.read_text().replace("How many lines to leave out.", "Lines to skip, corrected."))

        (tool,) = tools_from_saved_functions(tmp_path, run_here=True)

        assert "Lines to skip, corrected." in tool.parameters["properties"]["skip"]["description"]

    def test_a_file_that_can_no_longer_be_a_tool_is_refused_by_name(self, tmp_path: Path) -> None:
        self.save(tmp_path, "a_task")
        (tmp_path / "a_task.py").write_text("x = 1\n")

        with pytest.raises(ValueError, match="a_task cannot be a tool"):
            tools_from_saved_functions(tmp_path, run_here=True)

    def test_a_file_that_is_gone_is_an_os_error(self, tmp_path: Path) -> None:
        self.save(tmp_path, "a_task")
        (tmp_path / "a_task.py").unlink()

        with pytest.raises(OSError):
            tools_from_saved_functions(tmp_path, run_here=True)

    def test_where_they_run_must_be_said(self, tmp_path: Path) -> None:
        self.save(tmp_path, "a_task")

        with pytest.raises(ValueError, match="Say where the function runs"):
            tools_from_saved_functions(tmp_path)

    def test_a_directory_with_nothing_saved_has_no_tools(self, tmp_path: Path) -> None:
        assert tools_from_saved_functions(tmp_path, run_here=True) == ()
