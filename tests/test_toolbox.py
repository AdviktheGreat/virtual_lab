"""Tests for Biomni's tool functions as tools run in a session.

The calls run for real, in a LocalSession: Biomni's protocols module, which needs only the
standard library, for Biomni's own functions, and a module of stand-ins written to the session's
directory for what Biomni's need libraries to show.
"""

import ast
import json
import textwrap
import warnings
from pathlib import Path

import pytest

import virtual_lab.toolbox as toolbox_module
from virtual_lab.resources import BIOMNI_TOOL_MODULES, biomni_tools
from virtual_lab.session import LocalSession
from virtual_lab.toolbox import (
    BiomniToolError,
    biomni_conversion,
    biomni_session_tools,
    biomni_tool,
    biomni_type_schema,
    call_code,
    fits,
    parameter_schema,
)

STAND_INS = textwrap.dedent(
    """
    def describe(**arguments):
        return {name: type(value).__name__ for name, value in arguments.items()}

    def shape(values):
        return list(values.shape)

    def columns(table):
        return sorted(table.columns)

    def only_prints(text):
        print("found", text)

    def returns_nothing():
        return None

    def fails():
        print("about to fail")
        raise RuntimeError("it broke")
    """
)


@pytest.fixture
def session(tmp_path: Path):
    with LocalSession(tmp_path / "work", warn=False, timeout=30, biomni_tools=True) as opened:
        (opened.directory / "stand_ins.py").write_text(STAND_INS)
        yield opened


def stand_in(session: LocalSession, name: str, parameters: list[dict], optional: list[dict] | None = None):
    return biomni_tool(
        session,
        {
            "name": name,
            "description": f"The stand-in {name}.",
            "required_parameters": parameters,
            "optional_parameters": optional or [],
            "module": "stand_ins",
        },
    )


class TestTypeSchema:
    @pytest.mark.parametrize(
        ("type_name", "schema"),
        [
            ("str", {"type": "string"}),
            ("int", {"type": "integer"}),
            ("float", {"type": "number"}),
            ("bool", {"type": "boolean"}),
            ("dict", {"type": "object"}),
            ("list", {"type": "array"}),
            ("tuple", {"type": "array"}),
            ("array-like", {"type": "array"}),
            ("numpy.ndarray", {"type": "array"}),
            ("pd.DataFrame", {"type": "array", "items": {"type": "object"}}),
            ("List[str]", {"type": "array", "items": {"type": "string"}}),
            ("list[str]", {"type": "array", "items": {"type": "string"}}),
            ("list of str", {"type": "array", "items": {"type": "string"}}),
            ("list of dict", {"type": "array", "items": {"type": "object"}}),
            ("List[dict]", {"type": "array", "items": {"type": "object"}}),
            (
                "List[Dict[str, str]]",
                {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}},
            ),
            (
                "Optional[List[Dict[str, str]]]",
                {
                    "anyOf": [
                        {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}},
                        {"type": "null"},
                    ]
                },
            ),
            ("Tuple[int, int]", {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}),
            (
                "Tuple[str, int]",
                {
                    "type": "array",
                    "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                    "minItems": 2,
                    "maxItems": 2,
                },
            ),
            ("Tuple[int, ...]", {"type": "array", "items": {"type": "integer"}}),
            ("Sequence[str]", {"type": "array", "items": {"type": "string"}}),
            ("Set[int]", {"type": "array", "items": {"type": "integer"}}),
            ("tuple or None", {"anyOf": [{"type": "array"}, {"type": "null"}]}),
            ("list or numpy.ndarray", {"type": "array"}),
            ("str or dict", {"anyOf": [{"type": "string"}, {"type": "object"}]}),
            ("int|str", {"anyOf": [{"type": "integer"}, {"type": "string"}]}),
            ("str|list[str]", {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}),
            ("Union[str, List[str]]", {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}),
            ("dict or array-like", {"anyOf": [{"type": "object"}, {"type": "array"}]}),
            ("Dict[str, float]", {"type": "object", "additionalProperties": {"type": "number"}}),
            ("Any", {}),
            ("AnnData", {}),
            ("str or AnnData", {}),
        ],
    )
    def test_biomni_types_are_read_into_json_schema(self, type_name: str, schema: dict) -> None:
        assert biomni_type_schema(type_name) == schema

    def test_a_callable_cannot_be_passed(self) -> None:
        assert biomni_type_schema("callable") is None

    def test_a_callable_among_others_is_left_out(self) -> None:
        assert biomni_type_schema("str or callable") == {"type": "string"}

    @pytest.mark.parametrize(
        ("type_name", "conversion"),
        [
            ("numpy.ndarray", '__import__("numpy").asarray'),
            ("pd.DataFrame", '__import__("pandas").DataFrame'),
            ("tuple", '__import__("builtins").tuple'),
            ("Tuple[str, str]", '__import__("builtins").tuple'),
            ("tuple or None", '__import__("builtins").tuple'),
            ("list or numpy.ndarray", None),
            ("numpy.ndarray or list", None),
            ("pd.DataFrame or str", None),
            ("List[float] or numpy.ndarray", None),
            ("array-like", None),
            ("List[str]", None),
            ("str", None),
        ],
    )
    def test_only_a_type_a_list_is_not_is_converted(self, type_name: str, conversion: str | None) -> None:
        assert biomni_conversion(type_name) == conversion

    def test_a_function_whose_name_cannot_be_called_is_not_a_tool(self) -> None:
        api = {"name": "not-a-name", "module": "m", "required_parameters": [], "optional_parameters": []}

        assert biomni_tool(object(), api) is None  # type: ignore[arg-type]

    def test_a_parameter_whose_name_cannot_be_passed_is_left_out(self) -> None:
        api = {
            "name": "f",
            "module": "m",
            "required_parameters": [
                {"name": "two words", "type": "str", "description": "T"},
                {"name": "kept", "type": "str", "description": "K"},
            ],
            "optional_parameters": [],
        }

        tool = biomni_tool(object(), api)  # type: ignore[arg-type]

        assert list(tool.parameters["properties"]) == ["kept"]
        assert tool.parameters["required"] == ["kept"]

    def test_every_biomni_type_is_valid_json_schema(self) -> None:
        jsonschema = pytest.importorskip("jsonschema")

        for api in biomni_tools():
            tool = biomni_tool(object(), api)  # type: ignore[arg-type]
            assert tool is not None
            jsonschema.Draft202012Validator.check_schema(tool.parameters)


class TestFits:
    @pytest.mark.parametrize(
        ("value", "schema", "expected"),
        [
            ("a", {"type": "string"}, True),
            (1, {"type": "string"}, False),
            (1, {"type": "integer"}, True),
            (True, {"type": "integer"}, False),
            (1.5, {"type": "number"}, True),
            (1, {"type": "number"}, True),
            ("1", {"type": "number"}, False),
            (True, {"type": "number"}, False),
            (False, {"type": "boolean"}, True),
            (None, {"type": "null"}, True),
            ({"a": "b"}, {"type": "object", "additionalProperties": {"type": "string"}}, True),
            ({"a": 1}, {"type": "object", "additionalProperties": {"type": "string"}}, False),
            ([1, 2], {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, True),
            ([1], {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, False),
            ([1, 2, 3], {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, False),
            ([1, "a"], {"type": "array", "items": {"type": "integer"}}, False),
            ("a", {"anyOf": [{"type": "integer"}, {"type": "string"}]}, True),
            ([], {"anyOf": [{"type": "integer"}, {"type": "string"}]}, False),
            (object(), {}, True),
        ],
    )
    def test_values_are_checked_against_the_schemas_written(self, value: object, schema: dict, expected: bool) -> None:
        assert fits(value, schema) is expected


class TestParameterSchema:
    def test_the_description_is_kept(self) -> None:
        schema = parameter_schema({"name": "gene", "type": "str", "description": "A gene symbol"}, required=True)

        assert schema == {"description": "A gene symbol", "type": "string"}

    def test_a_required_parameter_has_no_default(self) -> None:
        schema = parameter_schema({"name": "n", "type": "int", "default": None, "description": "N"}, required=True)
        listed = parameter_schema({"name": "n", "type": "int", "default": 10, "description": "N"}, required=True)

        assert "default" not in schema
        assert listed == {"description": "N", "type": "integer"}

    def test_a_whole_number_is_a_default_for_a_float(self) -> None:
        schema = parameter_schema({"name": "x", "type": "float", "default": "1", "description": "X"}, required=False)

        assert schema["default"] == 1

    def test_a_default_that_fits_is_given(self) -> None:
        schema = parameter_schema({"name": "n", "type": "int", "default": 10, "description": "N"}, required=False)

        assert schema["default"] == 10

    def test_a_default_written_as_code_is_read(self) -> None:
        quoted = parameter_schema({"name": "m", "type": "str", "default": "'LSODA'", "description": "M"}, False)
        pair = parameter_schema({"name": "t", "type": "tuple", "default": "(0, 100)", "description": "T"}, False)

        assert quoted["default"] == "LSODA"
        assert pair["default"] == [0, 100]

    def test_a_string_default_that_reads_as_something_else_stays_a_string(self) -> None:
        # Biomni's generate_transcriptformer_embeddings really does default use_raw to "None"
        schema = parameter_schema({"name": "u", "type": "str", "default": "None", "description": "U"}, False)
        number = parameter_schema({"name": "v", "type": "str", "default": "100", "description": "V"}, False)

        assert schema["default"] == "None"
        assert number["default"] == "100"

    def test_a_default_that_does_not_fit_is_described_instead(self) -> None:
        schema = parameter_schema({"name": "n", "type": "int", "default": "many", "description": "N."}, False)

        assert "default" not in schema
        assert schema["description"] == "N. Defaults to many."

    def test_a_none_default_is_not_given(self) -> None:
        schema = parameter_schema({"name": "s", "type": "str", "default": None, "description": "S"}, False)

        assert "default" not in schema

    def test_a_type_that_cannot_be_read_is_named_in_the_description(self) -> None:
        schema = parameter_schema({"name": "a", "type": "AnnData", "description": "The data."}, True)

        assert schema == {"description": "The data. Biomni's type: AnnData."}

    def test_a_table_says_how_to_pass_it(self) -> None:
        schema = parameter_schema({"name": "t", "type": "pd.DataFrame", "description": "Counts."}, True)

        assert schema["description"] == "Counts. Pass the table's rows, each an object of column to value."

    def test_a_callable_cannot_be_a_parameter(self) -> None:
        assert parameter_schema({"name": "f", "type": "callable", "description": "F"}, False) is None


class TestCallCode:
    def test_the_call_is_one_expression(self) -> None:
        code = call_code("biomni.tool.genetics", "align", {"seq": "ACGT", "n": 2}, {})

        assert code == (
            '__import__("importlib").import_module(\'biomni.tool.genetics\').align('
            '**__import__("json").loads(\'{"seq": "ACGT", "n": 2}\'))'
        )
        compile(code, "<call>", "eval")

    def test_arguments_cannot_break_out_of_the_call(self) -> None:
        code = call_code("m", "f", {"text": "') + __import__('os').system('x') + ('"}, {})

        call = ast.parse(code, mode="eval").body
        assert call.func.attr == "f"
        assert len(call.args) == 0 and len(call.keywords) == 1

    def test_a_list_for_a_converted_parameter_is_converted(self) -> None:
        code = call_code("m", "f", {"x": [1, 2], "y": 3}, {"x": '__import__("numpy").asarray'})

        assert 'x=__import__("numpy").asarray(__import__("json").loads(\'[1, 2]\'))' in code
        assert '**__import__("json").loads(\'{"y": 3}\')' in code

    def test_none_for_a_converted_parameter_is_passed_as_it_is(self) -> None:
        code = call_code("m", "f", {"x": None}, {"x": '__import__("builtins").tuple'})

        assert "tuple" not in code

    def test_a_call_without_arguments(self) -> None:
        assert call_code("m", "f", {}, {}) == "__import__(\"importlib\").import_module('m').f()"


class TestCalls:
    def test_a_biomni_function_runs_in_the_session(self, session: LocalSession) -> None:
        (tool,) = biomni_session_tools(session, modules=["protocols"], names=["list_local_protocols"])

        listed = tool.function(source="addgene")

        assert isinstance(listed, dict)
        assert listed["protocols"] and all(item["source"] == "addgene" for item in listed["protocols"])
        assert session.history[-1].value == listed
        assert "list_local_protocols" in session.history[-1].code

    def test_a_function_that_fails_says_why_and_what_it_printed(self, session: LocalSession) -> None:
        tool = stand_in(session, "fails", [])

        with pytest.raises(BiomniToolError) as raised:
            tool.function()

        message = str(raised.value)
        assert message.startswith("fails failed: RuntimeError: it broke")
        assert "about to fail" in message
        assert "Traceback" in message

    def test_a_biomni_function_that_fails(self, session: LocalSession) -> None:
        (tool,) = biomni_session_tools(session, modules="protocols", names="read_local_protocol")

        with pytest.raises(BiomniToolError, match="FileNotFoundError"):
            tool.function(filename="no such protocol.txt")

    def test_what_a_function_that_returns_nothing_printed_is_returned(self, session: LocalSession) -> None:
        tool = stand_in(session, "only_prints", [{"name": "text", "type": "str", "description": "T"}])

        assert tool.function(text="TP53") == "found TP53\n"

    def test_a_function_that_returns_nothing_and_prints_nothing_returns_none(self, session: LocalSession) -> None:
        assert stand_in(session, "returns_nothing", []).function() is None

    def test_json_arguments_arrive_as_json_types(self, session: LocalSession) -> None:
        tool = stand_in(
            session,
            "describe",
            [
                {"name": "text", "type": "str", "description": "T"},
                {"name": "number", "type": "float", "description": "N"},
                {"name": "names", "type": "List[str]", "description": "L"},
                {"name": "mapping", "type": "dict", "description": "D"},
            ],
        )

        assert tool.function(text="a", number=1.5, names=["x"], mapping={"k": 1}) == {
            "text": "str",
            "number": "float",
            "names": "list",
            "mapping": "dict",
        }

    def test_a_list_for_an_array_arrives_as_a_numpy_array(self, session: LocalSession) -> None:
        pytest.importorskip("numpy")
        tool = stand_in(session, "shape", [{"name": "values", "type": "numpy.ndarray", "description": "V"}])

        assert tool.function(values=[[1, 2, 3], [4, 5, 6]]) == [2, 3]

    def test_rows_for_a_table_arrive_as_a_data_frame(self, session: LocalSession) -> None:
        pytest.importorskip("pandas")
        tool = stand_in(session, "columns", [{"name": "table", "type": "pd.DataFrame", "description": "T"}])

        assert tool.function(table=[{"gene": "TP53", "count": 3}, {"gene": "MYC", "count": 1}]) == ["count", "gene"]

    def test_a_list_for_a_tuple_arrives_as_a_tuple(self, session: LocalSession) -> None:
        tool = stand_in(
            session,
            "describe",
            [{"name": "pair", "type": "Tuple[str, str]", "description": "P"}],
            [{"name": "span", "type": "tuple or None", "default": None, "description": "S"}],
        )

        assert tool.function(pair=["a", "b"], span=None) == {"pair": "tuple", "span": "NoneType"}

    def test_an_argument_it_does_not_take_is_refused_before_running(self, session: LocalSession) -> None:
        tool = stand_in(session, "describe", [{"name": "text", "type": "str", "description": "T"}])

        with pytest.raises(BiomniToolError, match="describe has no parameter other; it takes text"):
            tool.function(text="a", other=1)
        assert session.history == []

    def test_a_missing_argument_is_refused_before_running(self, session: LocalSession) -> None:
        tool = stand_in(session, "describe", [{"name": "text", "type": "str", "description": "T"}])

        with pytest.raises(BiomniToolError, match="describe needs text"):
            tool.function()
        assert session.history == []

    def test_a_callable_parameter_is_left_out(self, session: LocalSession) -> None:
        tool = stand_in(
            session,
            "describe",
            [],
            [{"name": "ode_function", "type": "callable", "default": None, "description": "F"}],
        )

        assert tool.parameters["properties"] == {}
        with pytest.raises(BiomniToolError, match="no parameter ode_function"):
            tool.function(ode_function="lambda t, y: y")

    def test_a_call_leaves_no_name_behind_in_the_session(self, session: LocalSession) -> None:
        before = session.evaluate("sorted(globals())").value

        stand_in(session, "returns_nothing", []).function()

        assert session.evaluate("sorted(name for name in globals() if name != '_')").value == [
            name for name in before if name != "_"
        ]


class TestChoosing:
    def test_a_session_whose_code_cannot_import_biomni_is_refused(self, tmp_path: Path) -> None:
        with LocalSession(tmp_path, warn=False) as session:
            with pytest.raises(ValueError, match="cannot import Biomni's tools"):
                biomni_session_tools(session)

    def test_every_function_is_offered_by_default(self, session: LocalSession) -> None:
        tools = biomni_session_tools(session, check=False)

        assert [tool.name for tool in tools] == [api["name"] for api in biomni_tools()]
        assert not session.running

    def test_modules_are_chosen_with_or_without_their_package(self, session: LocalSession) -> None:
        short = biomni_session_tools(session, modules=["protocols"], check=False)
        full = biomni_session_tools(session, modules=["biomni.tool.protocols"], check=False)

        assert [tool.name for tool in short] == [tool.name for tool in full]
        assert {tool.function.__module__ for tool in short} == {"biomni.tool.protocols"}

    def test_an_unknown_module_is_refused(self, session: LocalSession) -> None:
        with pytest.raises(ValueError, match="no tool module 'genomes'"):
            biomni_session_tools(session, modules=["genomes"])

    def test_a_module_must_be_named(self, session: LocalSession) -> None:
        with pytest.raises(TypeError, match="named by str"):
            biomni_session_tools(session, modules=[3])  # type: ignore[list-item]

    def test_functions_are_chosen_by_name(self, session: LocalSession) -> None:
        tools = biomni_session_tools(session, names=["read_local_protocol", "list_local_protocols"], check=False)

        assert [tool.name for tool in tools] == ["list_local_protocols", "read_local_protocol"]

    def test_an_unknown_name_is_refused(self, session: LocalSession) -> None:
        with pytest.raises(ValueError, match="no 'read_protocol' among the modules chosen"):
            biomni_session_tools(session, modules=["protocols"], names=["read_protocol"])

    def test_modules_that_import_are_all_kept(self, session: LocalSession) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tools = biomni_session_tools(session, modules=["protocols", "support_tools"])

        assert {tool.function.__module__ for tool in tools} == {"biomni.tool.protocols", "biomni.tool.support_tools"}

    def test_a_module_that_fails_to_import_is_left_out_with_a_warning(
        self, session: LocalSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked = []

        def check_installed(session, libraries, modules, what):
            asked.append(modules)
            return set(), {"biomni.tool.support_tools": "ModuleNotFoundError: No module named 'synapseclient'"}

        monkeypatch.setattr(toolbox_module, "check_installed", check_installed)

        with pytest.warns(UserWarning, match="biomni.tool.support_tools \\(ModuleNotFoundError"):
            tools = biomni_session_tools(session, modules=["protocols", "support_tools"])

        assert asked == [["biomni.tool.support_tools", "biomni.tool.protocols"]]
        assert {tool.function.__module__ for tool in tools} == {"biomni.tool.protocols"}

    def test_every_function_is_kept_if_the_check_cannot_be_done(
        self, session: LocalSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(toolbox_module, "check_installed", lambda *arguments, **options: None)

        tools = biomni_session_tools(session, modules=["protocols"])

        assert len(tools) == len([api for api in biomni_tools() if api["module"] == "biomni.tool.protocols"])

    def test_the_tools_are_described_as_biomni_describes_them(self, session: LocalSession) -> None:
        (tool,) = biomni_session_tools(session, names=["get_protocol_details"], check=False)
        (api,) = [api for api in biomni_tools() if api["name"] == "get_protocol_details"]

        assert tool.description == api["description"]
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "protocol_id": {"description": "Numeric protocol ID from protocols.io", "type": "integer"},
                "timeout": {"description": "Request timeout in seconds", "type": "integer", "default": 30},
            },
            "required": ["protocol_id"],
        }
        json.dumps(tool.parameters)

    def test_every_module_has_functions(self) -> None:
        modules = {api["module"] for api in biomni_tools()}

        assert modules == {f"biomni.tool.{module}" for module in BIOMNI_TOOL_MODULES}
