"""Tests for reading a function that a model wrote, without running it."""

import inspect
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from virtual_lab.custom_tools import tool_from_function
from virtual_lab.function_checks import (
    GENERATED_MODULE,
    UNREADABLE_DEFAULT,
    FunctionSourceError,
    extract_code,
    fenced_blocks,
    read_function,
    script_name,
)

GOOD = '''
"""Counts things."""
import json
from typing import Literal, Optional

import pandas as pd

LIMIT = 5


def count_reads(
    path: str, minimum: int = -1, label: Optional[str] = None, scale: float = LIMIT, mode: Literal["a", "b"] = "a",
    tags: list[str] | None = None, *, strict: bool = False, **extra: Any,
) -> dict:
    """Counts the reads in a file.

    Args:
        path: The file to read.
        minimum: The fewest reads to count.
    """
    return {}


def main():
    print(count_reads("x"))


if __name__ == "__main__":
    main()
'''


def problems_of(source: str, name: str = "count_reads") -> list[str]:
    with pytest.raises(FunctionSourceError) as raised:
        read_function(source, name)

    return raised.value.problems


def function_with(
    parameters: str = "path: str", body: str = '    """Does it."""\n    return 1\n', extra: str = ""
) -> str:
    return f"{extra}\ndef count_reads({parameters}):\n{body}"


class TestFencedBlocks:
    def test_each_block_has_its_language_and_whether_it_was_closed(self) -> None:
        text = "Here:\n```python\nx = 1\n```\nand\n```bash\npip install x\n```\nlast\n```\nopen"

        blocks = fenced_blocks(text)

        assert [(block.language, block.text, block.closed) for block in blocks] == [
            ("python", "x = 1", True),
            ("bash", "pip install x", True),
            ("", "open", False),
        ]

    def test_a_longer_fence_is_closed_only_by_one_as_long(self) -> None:
        blocks = fenced_blocks("````python\nprint('''\n```\n''')\n````\n")

        assert len(blocks) == 1 and blocks[0].text == "print('''\n```\n''')" and blocks[0].closed

    def test_a_tilde_fence_is_a_fence_and_is_not_closed_by_backticks(self) -> None:
        blocks = fenced_blocks("~~~py\nx = 1\n```\n~~~\n")

        assert len(blocks) == 1 and blocks[0].text == "x = 1\n```" and blocks[0].language == "py"

    def test_text_without_a_fence_has_no_blocks(self) -> None:
        assert fenced_blocks("def f():\n    pass\n") == []


class TestExtractingCode:
    @pytest.mark.parametrize("language", ["python", "Python", "py", "python3", ""])
    def test_a_block_marked_as_python_or_not_marked_is_the_code(self, language: str) -> None:
        code = extract_code(f"Sure.\n```{language}\nx = 1\n```\nDone.")

        assert code is not None and code.source == "x = 1" and code.complete

    def test_a_block_of_another_language_is_not_taken_for_the_code(self) -> None:
        reply = "```bash\npip install pandas\n```\n```python\nimport pandas\n```"

        code = extract_code(reply)

        assert code is not None and code.source == "import pandas"

    def test_a_reply_with_only_another_language_has_no_code(self) -> None:
        assert extract_code("```bash\npip install pandas\n```") is None

    def test_the_first_block_is_taken_and_an_empty_one_is_skipped(self) -> None:
        code = extract_code("```python\n\n```\n```python\na = 1\n```\n```python\nb = 2\n```")

        assert code is not None and code.source == "a = 1"

    def test_a_block_the_reply_ended_inside_is_incomplete(self) -> None:
        code = extract_code("```python\ndef f():\n    return 1\n")

        assert code is not None and code.source == "def f():\n    return 1" and not code.complete

    def test_an_indented_block_is_dedented(self) -> None:
        code = extract_code("```python\n    def f():\n        return 1\n```")

        assert code is not None and code.source == "def f():\n    return 1"

    def test_a_reply_that_is_python_without_a_fence_is_the_code(self) -> None:
        code = extract_code("\n\ndef f():\n    return 1\n")

        assert code is not None and code.source == "def f():\n    return 1" and code.complete

    @pytest.mark.parametrize("reply", ["Here is the code you asked for.", "", "   \n", "def f(:"])
    def test_a_reply_that_is_neither_fenced_nor_python_has_no_code(self, reply: str) -> None:
        assert extract_code(reply) is None


def biomni_filename(task_description: str, max_words: int = 6) -> str:
    """Biomni's FunctionGenerator._generate_script_filename, as it is in biomni/agent/function_generator.py."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", task_description.lower())
    words = cleaned.split()
    selected_words = words[:max_words] if words else ["script"]
    base_name = "_".join(selected_words)

    return f"{base_name}.py"


class TestScriptName:
    @pytest.mark.parametrize(
        "task",
        [
            "RNA-seq differential expression analysis with DESeq2",
            "Perform a two-way ANOVA with Tukey's post-hoc test using SciPy",
            "Align reads",
            "  Multiple   spaces\tand\nnewlines in a task of many words ",
            "Compute (log2) fold-change & p-values!",
            "Résumé of café naïve analysis",
        ],
    )
    def test_it_is_the_name_biomni_gives_the_file_without_its_extension(self, task: str) -> None:
        assert script_name(task) + ".py" == biomni_filename(task)

    def test_a_task_with_no_words_is_named_script_as_in_biomni(self) -> None:
        assert [script_name(task) for task in ["", "!!!", "   "]] == ["script"] * 3
        assert biomni_filename("!!!") == "script.py"

    def test_the_words_may_be_counted_differently(self) -> None:
        assert script_name("one two three four", max_words=2) == "one_two"

    def test_a_name_that_begins_with_a_digit_is_given_a_prefix_so_that_it_can_be_imported(self) -> None:
        assert script_name("16S rRNA amplicon analysis") == "task_16s_rrna_amplicon_analysis"

    @pytest.mark.parametrize("task", ["json", "random analysis", "class", "typing"])
    def test_a_keyword_or_standard_module_is_given_a_suffix_so_that_the_file_does_not_shadow_it(
        self, task: str
    ) -> None:
        name = script_name(task, max_words=1)

        assert name == f"{task.split()[0]}_task" and name.isidentifier()

    def test_a_name_is_cut_to_what_a_tool_may_be_called(self) -> None:
        name = script_name("x" * 200)

        assert len(name) == 64 and name == "x" * 64

    def test_a_reserved_name_cut_to_length_still_ends_in_its_suffix(self) -> None:
        name = script_name("a" * 64, max_words=1)
        assert name == "a" * 64

        # Cutting can only leave a reserved name when the task is itself one, which is short; the suffix
        # is still counted in the length wherever it is added
        assert len(script_name("class")) <= 64

    def test_every_name_can_be_a_tool_and_a_module(self) -> None:
        for task in ["16S", "def", "os", "$$$", "Ünïcödé", "a" * 100, "1 2 3"]:
            name = script_name(task)

            assert name.isidentifier() and len(name) <= 64 and name.isascii()
            tool_from_function(lambda: None, name=name, description="Does it.")


class TestReadingAFunction:
    def test_a_function_that_can_be_a_tool_is_read_into_its_name_signature_and_docstring(self) -> None:
        reading = read_function(GOOD, "count_reads")

        assert reading.name == "count_reads"
        assert reading.docstring.startswith("Counts the reads in a file.")
        assert list(reading.signature.parameters) == [
            "path",
            "minimum",
            "label",
            "scale",
            "mode",
            "tags",
            "strict",
            "extra",
        ]
        assert reading.signature.parameters["minimum"].default == -1
        assert reading.signature.parameters["label"].default is None
        assert reading.signature.parameters["strict"].kind is inspect.Parameter.KEYWORD_ONLY
        assert reading.signature.parameters["extra"].kind is inspect.Parameter.VAR_KEYWORD

    def test_a_default_that_is_not_a_literal_is_marked_and_left_to_the_function(self) -> None:
        reading = read_function(GOOD, "count_reads")

        assert reading.signature.parameters["scale"].default is UNREADABLE_DEFAULT

    def test_the_modules_that_must_be_installed_are_those_that_are_not_pythons_own(self) -> None:
        assert read_function(GOOD, "count_reads").modules == ("pandas",)

    def test_a_tool_is_made_of_it_with_its_parameters_and_their_descriptions(self) -> None:
        reading = read_function(GOOD, "count_reads")
        seen: list[dict[str, Any]] = []

        tool = tool_from_function(
            reading.stand_in(lambda **arguments: seen.append(arguments) or "done"), name="count_reads"
        )

        assert tool.name == "count_reads" and tool.description == "Counts the reads in a file."
        properties = tool.parameters["properties"]
        assert properties["path"] == {"description": "The file to read.", "type": "string"}
        assert properties["minimum"]["default"] == -1 and properties["minimum"]["type"] == "integer"
        assert properties["mode"]["enum"] == ["a", "b"]
        assert "default" not in properties["scale"]
        assert tool.parameters["required"] == ["path"]
        assert tool.function(path="reads.bam", minimum=3) == "done"
        assert seen == [{"path": "reads.bam", "minimum": 3}]

    def test_the_stand_in_is_named_for_the_function_and_the_module_generated_functions_come_from(self) -> None:
        reading = read_function(GOOD, "count_reads")

        function = reading.stand_in(lambda **arguments: None)

        assert (function.__name__, function.__qualname__, function.__module__) == (
            "count_reads",
            "count_reads",
            GENERATED_MODULE,
        )
        assert inspect.signature(function) == reading.signature

    def test_nothing_in_the_code_is_run(self, tmp_path: Path) -> None:
        marker = tmp_path / "ran"
        at_the_top = f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n" + function_with()
        in_a_hint = function_with(
            f"path: __import__('pathlib').Path({str(marker)!r}).write_text('y')", extra="import pathlib\n"
        )
        in_a_default = function_with(f"path: str = open({str(marker)!r}, 'w')")
        in_a_guard = f"import os\nif os.name:\n    open({str(marker)!r}, 'w')\n" + function_with()

        for source in (at_the_top, in_a_hint, in_a_default, in_a_guard):
            with pytest.raises(FunctionSourceError):
                read_function(source, "count_reads")

        assert not marker.exists()

    def test_a_default_that_is_a_call_is_refused_since_it_is_computed_once_on_import(self) -> None:
        (problem,) = problems_of(function_with("folder: str = os.getcwd()", extra="import os\n"))

        assert "default of parameter folder, os.getcwd()," in problem and "Default it to None" in problem

    def test_a_default_that_is_a_name_or_a_literal_is_kept(self) -> None:
        source = function_with("a: int = LIMIT, b: list = [1, 2], c: str = 'x'", extra="LIMIT = 5\n")

        parameters = read_function(source, "count_reads").signature.parameters

        assert parameters["b"].default == [1, 2] and parameters["c"].default == "x"
        assert parameters["a"].default is UNREADABLE_DEFAULT

    def test_a_function_written_twice_is_the_last_one(self) -> None:
        source = function_with("a: str") + function_with("b: int")

        assert list(read_function(source, "count_reads").signature.parameters) == ["b"]

    def test_the_main_guard_is_not_code_that_runs_on_import(self) -> None:
        source = function_with(extra='if __name__ == "__main__":\n    print(count_reads("x"))\n')

        assert read_function(source, "count_reads").name == "count_reads"
        assert read_function(function_with(extra='if "__main__" == __name__:\n    print(1)\n'), "count_reads")

    def test_a_docstring_is_what_describes_the_tool_and_its_parameters_in_any_style(self) -> None:
        source = function_with(
            "path: str, top: int = 5",
            body='    """Counts.\n\n    :param path: The file.\n    :param top: How many to show.\n    """\n',
        )

        tool = tool_from_function(read_function(source, "count_reads").stand_in(lambda **a: 1), name="count_reads")

        assert tool.parameters["properties"]["top"]["description"] == "How many to show."


class TestWhatIsWrongWithAFunction:
    def test_code_that_is_not_python_says_where(self) -> None:
        (problem,) = problems_of("def count_reads(:\n    pass\n")

        assert problem.startswith("The code is not valid Python:") and "on line 1" in problem

    def test_code_with_a_null_byte_is_refused(self) -> None:
        (problem,) = problems_of("x = 1\x00")

        assert "is not valid Python" in problem

    def test_code_too_long_is_refused_before_it_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("virtual_lab.function_checks.MAX_FUNCTION_SOURCE_CHARS", 50)

        (problem,) = problems_of(function_with() + "# " + "x" * 100)

        assert "more than the 50" in problem

    def test_code_nested_too_deep_for_the_parser_is_refused(self) -> None:
        (problem,) = problems_of("x = " + "(" * 500 + "1" + ")" * 500)

        assert problem.startswith("The code")

    def test_a_file_without_the_function_names_what_it_does_define(self) -> None:
        (problem,) = problems_of("def helper():\n    pass\n\ndef other():\n    pass\n")

        assert "defines no function named count_reads" in problem and "It defines helper, other." in problem

    def test_a_function_inside_a_class_or_a_function_is_not_the_function(self) -> None:
        (problem,) = problems_of("class A:\n    def count_reads(self):\n        pass\n")

        assert "defines no function named count_reads" in problem and "It defines" not in problem

    def test_an_async_function_is_refused(self) -> None:
        (problem,) = problems_of("async def count_reads(path: str):\n    '''Does it.'''\n")

        assert "async function" in problem

    def test_a_decorated_function_is_refused(self) -> None:
        (problem,) = problems_of("@cache\ndef count_reads(path: str):\n    '''Does it.'''\n")

        assert "has a decorator" in problem

    def test_a_function_without_a_docstring_or_with_an_empty_one_is_refused(self) -> None:
        for body in ["    return 1\n", '    """  """\n    return 1\n']:
            (problem,) = problems_of(function_with(body=body))

            assert problem.startswith("count_reads has no docstring")

    def test_a_parameter_without_a_type_hint_is_named(self) -> None:
        (problem,) = problems_of(function_with("path, top: int = 1"))

        assert problem.startswith("Parameter path has no type hint")

    def test_every_problem_found_is_reported_not_only_the_first(self) -> None:
        problems = problems_of(function_with("a, b", body="    return 1\n") + "print('hi')\n")

        assert len(problems) == 4
        assert "no docstring" in problems[0] and "Parameter a" in problems[1] and "Parameter b" in problems[2]
        assert "Line " in problems[3] and "print('hi')" in problems[3]

    @pytest.mark.parametrize(
        ("hint", "extra", "why"),
        [
            ("Path", "from pathlib import Path\n", "Path is not one of the types a tool takes"),
            ("Path", "", "Path is not one of the types a tool takes"),
            ("Optional[str]", "", "Optional is not imported. Add `from typing import Optional`"),
            ("tuple[int, int]", "", "tuple is not one of the types a tool takes"),
            ("set[str]", "", "set is not one of the types a tool takes"),
            ("bytes", "", "bytes is not one of the types a tool takes"),
            ("list[tuple[int]]", "", "tuple is not one of the types a tool takes"),
            ("Literal[b'x']", "from typing import Literal\n", "a Literal may hold only"),
            ("Sequence[str]", "from typing import Sequence\n", "is not one of the types JSON has"),
            ("DataFrame", "", "DataFrame is not one of the types a tool takes"),
            ("pd.DataFrame", "import pandas as pd\n", "pd is not one of the types a tool takes"),
            ("Mode", "class Mode:\n    pass\n", "Mode is not one of the types a tool takes"),
        ],
    )
    def test_a_type_hint_that_is_not_one_of_the_types_json_has_is_refused_with_advice(
        self, hint: str, extra: str, why: str
    ) -> None:
        (problem,) = problems_of(function_with(f"path: {hint}", extra=extra))

        assert why in problem and "A tool is called with JSON" in problem and "take a file's path as a str" in problem

    @pytest.mark.parametrize(
        "hint", ["str", "int", "float", "bool", "list", "dict", "list[str]", "dict[str, float]", "str | None", "Any"]
    )
    def test_the_types_json_has_are_accepted(self, hint: str) -> None:
        extra = "from typing import Any\n"

        assert read_function(function_with(f"x: {hint}", extra=extra), "count_reads")

    @pytest.mark.parametrize(
        ("hint", "extra"),
        [
            ("Optional[str]", "from typing import Optional\n"),
            ("t.Optional[str]", "import typing as t\n"),
            ("typing.Union[int, str]", "import typing\n"),
            ("Optional[List[Dict[str, Any]]]", "from typing import *\n"),
            ("'list[str]'", ""),
            ("Literal['a', 1, True, None]", "from typing import Literal\n"),
            ("Literal[-1, 2]", "from typing import Literal\n"),
        ],
    )
    def test_hints_from_typing_are_read_however_they_are_imported_or_written(self, hint: str, extra: str) -> None:
        assert read_function(function_with(f"x: {hint}", extra=extra), "count_reads")

    def test_a_hint_that_calls_something_is_refused_and_not_run(self, tmp_path: Path) -> None:
        marker = tmp_path / "ran"
        hint = f"__import__('pathlib').Path({str(marker)!r}).touch()"

        (problem,) = problems_of(function_with(f"x: {hint}"))

        assert "made of more than names" in problem and not marker.exists()

    @pytest.mark.parametrize(
        ("hint", "why"),
        [
            ("str.__class__", "not public"),
            ("list[str] + list[int]", "only | may join types"),
            ("[str]", "made of more than names"),
            ("lambda: 1", "made of more than names"),
            ("'not a type('", "is not a type"),
            ("~str", "only a minus sign may stand before a number"),
            ("-str", "TypeError"),
        ],
    )
    def test_a_hint_made_of_anything_but_names_and_subscripts_is_refused(self, hint: str, why: str) -> None:
        (problem,) = problems_of(function_with(f"x: {hint}"))

        assert why in problem

    def test_a_parameter_that_can_only_be_passed_by_position_is_refused(self) -> None:
        problems = problems_of(function_with("path: str, /, top: int = 1"))

        assert problems == [
            "Parameter path is positional-only, but a tool's parameters are passed by name. "
            "Remove the / from the signature."
        ]

    @pytest.mark.parametrize(
        ("source", "line"),
        [
            ("print('hi')\n", "print('hi')"),
            ("main()\n", "main()"),
            ("if True:\n    main()\n", "main()"),
            ("try:\n    main()\nexcept Exception:\n    pass\n", "main()"),
            ("with open('x') as f:\n    f.read()\n", "f.read()"),
            ("if False:\n    pass\nelse:\n    main()\n", "main()"),
        ],
    )
    def test_code_that_runs_when_the_file_is_imported_is_refused(self, source: str, line: str) -> None:
        (problem,) = problems_of(function_with(extra=source))

        assert f"runs {line} at the top level" in problem and "__main__" in problem

    @pytest.mark.parametrize("source", ["for i in range(3):\n    pass\n", "while False:\n    pass\n"])
    def test_a_loop_at_the_top_level_is_refused(self, source: str) -> None:
        (problem,) = problems_of(function_with(extra=source))

        assert "loops at the top level" in problem

    @pytest.mark.parametrize(
        "source",
        [
            "import os\nLIMIT = 3\nNAMES = ['a']\n",
            "class Helper:\n    x = print('only when it is used')\n    def run(self):\n        main()\n",
            "def helper():\n    main()\n    for i in range(3):\n        pass\n",
            "'''A docstring.'''\n",
            "if __name__ == '__main__':\n    main()\n    for i in range(3):\n        pass\n",
            "try:\n    import numpy\nexcept ImportError:\n    numpy = None\n",
        ],
    )
    def test_what_only_defines_things_is_not_refused(self, source: str) -> None:
        assert read_function(function_with(extra=source), "count_reads")

    def test_a_function_whose_type_cannot_be_a_tool_is_refused_in_the_words_of_tool_from_function(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refusing(function: Any, name: str | None = None, **options: Any) -> Any:
            raise TypeError("A parameter of count_reads has a type a model cannot pass as JSON")

        monkeypatch.setattr("virtual_lab.function_checks.tool_from_function", refusing)

        (problem,) = problems_of(function_with())

        assert problem == "A parameter of count_reads has a type a model cannot pass as JSON"

    def test_a_python_that_the_interpreter_warns_of_is_still_read_quietly(self) -> None:
        source = function_with(body='    """Does it."""\n    return "\\d"\n')

        assert read_function(source, "count_reads")


def test_the_modules_listed_are_found_anywhere_in_the_file_and_not_relative_imports() -> None:
    source = function_with(
        body=(
            '    """Does it."""\n    import scipy.stats\n    from . import sibling\n'
            "    from sklearn.cluster import KMeans\n"
        ),
        extra="import json\nimport os.path\nfrom __future__ import annotations\n",
    )

    assert read_function(source.replace("from __future__ import annotations\n", ""), "count_reads").modules == (
        "scipy",
        "sklearn",
    )
    assert sys.version_info >= (3, 12)
