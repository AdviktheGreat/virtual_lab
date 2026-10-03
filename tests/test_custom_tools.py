"""Tests for making tools of functions of one's own."""

import asyncio
import enum
import json
import warnings
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import pytest
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from pydantic import BaseModel

from virtual_lab.custom_tools import (
    FunctionNotes,
    ParameterNote,
    clean_schema,
    inline_references,
    parse_docstring,
    tool_from_function,
)
from virtual_lab.tools import run_tool_calls, tool_output_text

from conftest import FakeClient, parsed_response


class Strand(enum.Enum):
    FORWARD = "+"
    REVERSE = "-"


class Region(BaseModel):
    chromosome: str
    start: int
    end: int


class Tree(BaseModel):
    name: str
    children: list["Tree"] = []


def find_gene(symbol: str, limit: int = 10, organism: str = "human") -> dict[str, Any]:
    """Finds a gene by its symbol.

    Searches every assembly.

    :param symbol: The gene's symbol, such as
        TP53 or BRCA1.
    :param limit: How many matches to return.
    :type limit: int
    :raises ValueError: If the symbol is empty.
    :return: The matches, best first.
    """
    return {"symbol": symbol, "limit": limit, "organism": organism}


def call_of(name: str, arguments: dict[str, Any]) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="call_1", type="function", function=Function(name=name, arguments=json.dumps(arguments))
    )


class TestDocstrings:
    def test_rest_fields_describe_the_parameters_and_the_return_stays(self) -> None:
        description, parameters = parse_docstring(find_gene.__doc__)

        assert description == "Finds a gene by its symbol.\n\nSearches every assembly.\n\nReturns: The matches, best first."
        assert parameters == {"symbol": "The gene's symbol, such as TP53 or BRCA1.", "limit": "How many matches to return."}

    def test_a_rest_parameter_with_its_type_in_the_field(self) -> None:
        assert parse_docstring(":param str symbol: The symbol.")[1] == {"symbol": "The symbol."}

    def test_google_args_with_types_and_continued_lines(self) -> None:
        docstring = """Aligns reads.

        Args:
            reads (list[str]): The reads,
                as sequences.
            mismatches: How many to allow.
            **options: Anything else.

        Returns:
            The alignments.
        """

        description, parameters = parse_docstring(docstring)

        assert description == "Aligns reads.\n\nReturns:\n    The alignments."
        assert parameters == {"reads": "The reads, as sequences.", "mismatches": "How many to allow.", "options": "Anything else."}

    def test_numpy_parameters_including_two_named_together(self) -> None:
        docstring = """Scores a pair.

        Parameters
        ----------
        first, second : str
            The sequences
            to compare.
        gap : float, optional
            The gap penalty.

        Returns
        -------
        float
            The score.
        """

        description, parameters = parse_docstring(docstring)

        assert parameters == {"first": "The sequences to compare.", "second": "The sequences to compare.", "gap": "The gap penalty."}
        assert description == "Scores a pair.\n\nReturns\n-------\nfloat\n    The score."

    def test_no_docstring_describes_nothing(self) -> None:
        assert parse_docstring(None) == ("", {})
        assert parse_docstring("   ") == ("", {})

    def test_a_returns_section_alone_is_not_taken_for_parameters(self) -> None:
        assert parse_docstring("Counts.\n\nReturns:\n    count: how many.")[1] == {}


class TestSchema:
    def test_the_parameters_are_the_signatures(self) -> None:
        tool = tool_from_function(find_gene)

        assert tool.name == "find_gene"
        assert tool.description.startswith("Finds a gene by its symbol.")
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "The gene's symbol, such as TP53 or BRCA1."},
                "limit": {"type": "integer", "default": 10, "description": "How many matches to return."},
                "organism": {"type": "string", "default": "human"},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        }

    def test_types_are_read_from_the_hints(self) -> None:
        def typed(
            names: list[str],
            weights: dict[str, float],
            mode: Literal["fast", "exact"],
            strand: Strand,
            where: Path,
            on: date,
            flag: bool,
            region: Region,
        ) -> None:
            """Typed."""

        properties = tool_from_function(typed).parameters["properties"]

        assert properties["names"] == {"type": "array", "items": {"type": "string"}}
        assert properties["weights"] == {"type": "object", "additionalProperties": {"type": "number"}}
        assert properties["mode"] == {"type": "string", "enum": ["fast", "exact"]}
        assert properties["strand"] == {"type": "string", "enum": ["+", "-"]}
        assert properties["where"]["type"] == "string"
        assert properties["on"] == {"type": "string", "format": "date"}
        assert properties["flag"] == {"type": "boolean"}
        # Written out in place, since not every provider follows a reference
        assert properties["region"]["properties"]["start"] == {"type": "integer"}
        assert "$defs" not in tool_from_function(typed).parameters

    def test_a_schema_that_refers_to_itself_keeps_its_definitions(self) -> None:
        def grow(tree: Tree) -> None:
            """Grows a tree."""

        parameters = tool_from_function(grow).parameters

        assert parameters["properties"]["tree"] == {"$ref": "#/$defs/Tree"}
        assert "Tree" in parameters["$defs"]

    def test_a_parameter_defaulting_to_none_takes_none(self) -> None:
        def optional(name: str = None) -> None:  # type: ignore[assignment]
            """Optional."""

        assert tool_from_function(optional).parameters["properties"]["name"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
        }

    def test_a_parameter_without_a_hint_takes_anything(self) -> None:
        def loose(value, other=None) -> None:  # type: ignore[no-untyped-def]
            """Loose."""

        properties = tool_from_function(loose).parameters["properties"]

        assert properties == {"value": {}, "other": {"default": None}}

    def test_star_arguments_and_bound_arguments_are_left_out(self) -> None:
        def spread(directory: str, query: str, *args: str, **kwargs: str) -> None:
            """Spread."""

        tool = tool_from_function(partial(spread, "/data"))

        assert list(tool.parameters["properties"]) == ["query"]
        assert tool.parameters["required"] == ["query"]
        assert tool.name == "spread"

    def test_a_default_that_is_not_json_is_left_out_of_the_schema(self) -> None:
        marker = object()

        def odd(value: Any = marker) -> None:
            """Odd."""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tool = tool_from_function(odd)

        assert tool.parameters["properties"]["value"] == {}

    def test_a_parameter_called_title_keeps_its_name(self) -> None:
        def label(title: str) -> None:
            """Labels."""

        assert tool_from_function(label).parameters["properties"] == {"title": {"type": "string"}}

    def test_only_titles_that_are_keywords_are_dropped(self) -> None:
        schema = {"title": "X", "properties": {"title": {"title": "Title", "default": {"title": "kept"}}}}

        assert clean_schema(schema) == {"properties": {"title": {"default": {"title": "kept"}}}}

    def test_inlining_keeps_what_sits_beside_a_reference(self) -> None:
        schema = {"properties": {"a": {"$ref": "#/$defs/A", "description": "An A."}}, "$defs": {"A": {"type": "string"}}}

        assert inline_references(schema) == {"properties": {"a": {"type": "string", "description": "An A."}}}

    def test_a_type_a_model_cannot_send_is_refused_naming_it(self) -> None:
        class Frame:
            pass

        def table(frame: Frame) -> None:
            """Tables."""

        with pytest.raises(TypeError, match="cannot pass as JSON"):
            tool_from_function(table)

    def test_a_hint_that_cannot_be_resolved_is_warned_of_and_takes_anything(self) -> None:
        def unresolved(value: "Missing") -> None:  # type: ignore[name-defined]  # noqa: F821
            """Unresolved."""

        with pytest.warns(UserWarning, match="type hints of unresolved"):
            tool = tool_from_function(unresolved)

        assert tool.parameters["properties"] == {"value": {}}

    def test_parameters_may_have_names_a_pydantic_model_reserves(self) -> None:
        def reserved(model_config: str, schema: int, _private: bool = False) -> dict:
            """Reserved."""
            return {"model_config": model_config, "schema": schema, "_private": _private}

        tool = tool_from_function(reserved)

        assert list(tool.parameters["properties"]) == ["model_config", "schema", "_private"]
        assert tool.function(model_config="a", schema="2", _private=True) == {"model_config": "a", "schema": 2, "_private": True}


class TestCalling:
    def test_arguments_are_converted_to_the_hints(self) -> None:
        def convert(count: int, strand: Strand, where: Path, region: Region, pair: tuple[int, str], on: date) -> list:
            """Converts."""
            return [count, strand, where, region, pair, on]

        result = tool_from_function(convert).function(
            count="5", strand="-", where="/data", region={"chromosome": "1", "start": 1, "end": 9}, pair=[1, "a"], on="2024-01-31"
        )

        assert result == [5, Strand.REVERSE, Path("/data"), Region(chromosome="1", start=1, end=9), (1, "a"), date(2024, 1, 31)]

    def test_wrong_arguments_are_listed_and_the_function_is_not_called(self) -> None:
        called = []

        def strict(count: int) -> None:
            """Strict."""
            called.append(count)

        with pytest.raises(ValueError) as raised:
            tool_from_function(strict).function(count="many", extra=1)

        assert str(raised.value) == (
            "Invalid arguments for strict: count: Input should be a valid integer, unable to parse string as an "
            "integer; extra: Extra inputs are not permitted"
        )
        assert called == []

    def test_a_missing_argument_is_named(self) -> None:
        with pytest.raises(ValueError, match="symbol: Field required"):
            tool_from_function(find_gene).function()

    def test_only_what_was_given_is_passed_so_the_functions_defaults_apply(self) -> None:
        seen: list[dict] = []

        def defaults(first: str, items: list[int] = []) -> None:  # noqa: B006
            """Defaults."""
            seen.append({"first": first, "items_is_default": items is defaults.__defaults__[0]})

        tool_from_function(defaults).function(first="a")

        assert seen == [{"first": "a", "items_is_default": True}]

    def test_positional_only_parameters_are_passed_by_position(self) -> None:
        def ordered(first: int, second: int = 2, /, *, third: int) -> tuple:
            """Ordered."""
            return (first, second, third)

        tool = tool_from_function(ordered)

        assert tool.function(first=1, third=3) == (1, 2, 3)
        assert tool.function(second=5, first=1, third=3) == (1, 5, 3)

    def test_an_async_function_is_waited_for(self) -> None:
        async def doubled(value: int) -> int:
            """Doubles."""
            await asyncio.sleep(0)
            return value * 2

        tool = tool_from_function(doubled)

        assert tool.function(value=4) == 8

        async def inside_a_loop() -> int:
            return tool.function(value=5)

        assert asyncio.run(inside_a_loop()) == 10

    def test_a_callable_object_is_described_by_its_call(self) -> None:
        class Counter:
            """Counts things, and keeps the count."""

            def __call__(self, step: int = 1) -> int:
                """Adds to the count.

                :param step: How much to add.
                """
                return step

        tool = tool_from_function(Counter(), name="count")

        assert tool.description == "Adds to the count."
        assert tool.parameters["properties"]["step"]["description"] == "How much to add."
        assert tool.function(step="3") == 3
        assert tool.function.__qualname__.endswith("Counter")

    def test_the_tool_is_named_for_its_function_in_records(self) -> None:
        function = tool_from_function(find_gene).function

        assert (function.__module__, function.__qualname__) == (__name__, "find_gene")


class TestNamesAndDescriptions:
    def test_a_name_and_description_can_be_given(self) -> None:
        tool = tool_from_function(find_gene, name="gene_lookup", description="Looks genes up.")

        assert (tool.name, tool.description) == ("gene_lookup", "Looks genes up.")
        assert tool.parameters["properties"]["symbol"]["description"].startswith("The gene's symbol")

    @pytest.mark.parametrize("name", ["has space", "", "x" * 65, "dotted.name"])
    def test_a_name_no_provider_accepts_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="cannot name a tool"):
            tool_from_function(find_gene, name=name)

    def test_a_lambda_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="Pass name="):
            tool_from_function(lambda x: x, description="Echoes.")

        assert tool_from_function(lambda x: x, name="echo", description="Echoes.").function(x=1) == 1

    def test_something_that_is_not_callable_is_refused(self) -> None:
        with pytest.raises(TypeError, match="not str"):
            tool_from_function("find_gene")  # type: ignore[arg-type]

    def test_a_function_without_a_docstring_needs_a_description_or_a_model(self) -> None:
        def bare(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="no docstring"):
            tool_from_function(bare)

        assert tool_from_function(bare, description="Echoes.").description == "Echoes."

    def test_a_model_describes_a_function_without_a_docstring(
        self, fake_client: FakeClient, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def bare(sequence: str, window: int = 5) -> float:
            return len(sequence) / window

        fake_client.completions.parsed_responses.append(
            parsed_response(
                FunctionNotes(
                    description="Divides a sequence's length by a window.",
                    parameters=[
                        ParameterNote(name="sequence", description="A DNA sequence."),
                        ParameterNote(name="invented", description="Not a parameter."),
                    ],
                )
            )
        )

        tool = tool_from_function(bare, model="gpt-4o-2024-08-06")

        assert tool.description == "Divides a sequence's length by a window."
        assert tool.parameters["properties"] == {
            "sequence": {"type": "string", "description": "A DNA sequence."},
            "window": {"type": "integer", "default": 5},
        }
        # The model is shown the source, and only describes: the parameters are the signature's
        sent = fake_client.completions.parse_calls[0]["messages"][-1]["content"]
        assert "return len(sequence) / window" in sent
        assert "Signature: bare(sequence: str, window: int = 5) -> float" in sent
        assert "Add this to the function as its docstring" in capsys.readouterr().out

    def test_a_given_description_means_no_model_is_asked(self, fake_client: FakeClient) -> None:
        def bare(x: int) -> int:
            return x

        tool_from_function(bare, description="Echoes.", model="gpt-4o-2024-08-06")

        assert fake_client.completions.calls == []


class TestShownToTheModel:
    def test_what_a_tool_returns_is_written_as_json_where_it_is_data(self) -> None:
        assert tool_output_text("text") == "text"
        assert tool_output_text({"a": [1, 2.5], "é": None}) == '{"a": [1, 2.5], "é": null}'
        assert tool_output_text((Path("/x"), Strand.FORWARD)) == '["/x", "+"]'
        assert tool_output_text(Region(chromosome="X", start=1, end=2)) == '{"chromosome":"X","start":1,"end":2}'
        assert tool_output_text(None) == "None"
        assert tool_output_text(3.5) == "3.5"

    def test_a_meeting_runs_the_tool_and_hears_what_was_wrong(self) -> None:
        tool = tool_from_function(find_gene)

        outputs, _ = run_tool_calls([call_of("find_gene", {"symbol": "TP53", "limit": "3"})], (tool,))
        wrong, _ = run_tool_calls([call_of("find_gene", {"limit": "three"})], (tool,))

        assert json.loads(outputs[0]) == {"symbol": "TP53", "limit": 3, "organism": "human"}
        assert wrong[0].startswith('Error running tool "find_gene": ValueError: Invalid arguments for find_gene: symbol: Field required')


def test_optional_hints_written_either_way_are_the_same() -> None:
    def old(value: Optional[int] = None) -> None:  # noqa: UP007
        """Old."""

    def new(value: int | None = None) -> None:
        """New."""

    assert tool_from_function(old).parameters == tool_from_function(new).parameters
