"""Turning a Python function of your own into a tool that agents can call.

Biomni's add_tool asks a model to read a function's source and write its description and
parameters. Here the parameters come from the function's signature and type hints, which say
exactly what it takes, and the descriptions from its docstring, which says what its author
meant. A model is asked only for a function with no docstring, and then only to describe it:
the parameters it can be called with are still the ones its signature declares.

The arguments a model sends are checked against the type hints, and converted to them, before
the function is called, so a function annotated to take an Enum, a Path, a pydantic model, or a
tuple gets one, and one called with an argument of the wrong type is not called at all: the model
is told what was wrong instead.
"""

import asyncio
import functools
import inspect
import json
import re
import typing
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError
from pydantic.json_schema import PydanticJsonSchemaWarning

from virtual_lab.llm import ModelSource, resolve_chat_models
from virtual_lab.structured import request_structured_output
from virtual_lab.tools import Tool

# The names the OpenAI API accepts for a function, which the other providers accept too
TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")

# The most characters of a function's source a model is shown to describe it
MAX_SOURCE_CHARS = 20_000

RST_FIELD = re.compile(r"^:(?P<kind>[A-Za-z]+)(?P<rest>[^:]*):\s*(?P<text>.*)$")
RST_PARAMETER_KINDS = frozenset({"param", "parameter", "arg", "argument", "key", "keyword"})
RST_RETURN_KINDS = frozenset({"return", "returns"})

GOOGLE_PARAMETER_SECTIONS = frozenset(
    {"args", "arguments", "parameters", "params", "keyword args", "keyword arguments", "other parameters"}
)
GOOGLE_ITEM = re.compile(r"^\*{0,2}(?P<name>\w+)\s*(?:\([^)]*\))?\s*:\s*(?P<text>.*)$")

NUMPY_PARAMETER_SECTIONS = frozenset({"parameters", "other parameters"})
NUMPY_ITEM = re.compile(r"^\*{0,2}(?P<names>\w+(?:\s*,\s*\*{0,2}\w+)*)\s*(?::.*)?$")
NUMPY_UNDERLINE = re.compile(r"^-{3,}\s*$")

# Keywords whose value is a schema, or a mapping or list of schemas, as opposed to data such as
# a default or an enum, which a "title" inside must not be stripped from
SCHEMA_KEYWORDS = frozenset({"items", "additionalProperties", "not", "contains", "propertyNames", "if", "then", "else"})
SCHEMA_MAP_KEYWORDS = frozenset({"properties", "$defs", "definitions", "patternProperties"})
SCHEMA_LIST_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})


def join_lines(lines: list[str]) -> str:
    return " ".join(" ".join(line.split()) for line in lines if line.strip())


def parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Separates a docstring's description from its parameters' descriptions.

    Parameters are read in the reST style (":param name: text"), the Google style (an "Args:"
    section), and the NumPy style (a "Parameters" section underlined with dashes). What the
    function returns is kept in the description, since it is part of what a model needs to know
    to decide whether to call it; the other reST fields, such as ":raises:" and ":type:", are
    left out.

    :param docstring: The docstring.
    :return: The description, and each parameter's description by name.
    """
    if not docstring or not docstring.strip():
        return "", {}

    lines = inspect.cleandoc(docstring).splitlines()
    kept: list[str] = []
    returns: list[str] = []
    parameters: dict[str, str] = {}
    index = 0

    def indentation(line: str) -> int:
        return len(line) - len(line.lstrip())

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # reST: a field, and the indented lines that continue it
        if (field := RST_FIELD.match(stripped)) is not None and indentation(line) == 0:
            text = [field["text"]]
            index += 1
            while index < len(lines) and lines[index].strip() and indentation(lines[index]) > 0:
                text.append(lines[index])
                index += 1
            kind = field["kind"].lower()
            words = field["rest"].split()
            if kind in RST_PARAMETER_KINDS and words:
                parameters[words[-1].lstrip("*")] = join_lines(text)
            elif kind in RST_RETURN_KINDS:
                returns.append(join_lines(text))
            continue

        # Google: a section header, then items indented under it
        if indentation(line) == 0 and stripped.endswith(":") and stripped[:-1].lower() in GOOGLE_PARAMETER_SECTIONS:
            index += 1
            name: str | None = None
            item_indent: int | None = None
            while index < len(lines) and (not lines[index].strip() or indentation(lines[index]) > 0):
                current = lines[index]
                if current.strip():
                    if item_indent is None:
                        item_indent = indentation(current)
                    item = GOOGLE_ITEM.match(current.strip())
                    if indentation(current) <= item_indent and item is not None:
                        name = item["name"]
                        parameters[name] = item["text"].strip()
                    elif name is not None:
                        parameters[name] = join_lines([parameters[name], current])
                index += 1
            continue

        # NumPy: a section name underlined with dashes, then unindented items, each described
        # by the indented lines under it
        if (
            stripped.lower() in NUMPY_PARAMETER_SECTIONS
            and index + 1 < len(lines)
            and NUMPY_UNDERLINE.match(lines[index + 1].strip())
        ):
            index += 2
            names: list[str] = []
            while index < len(lines):
                current = lines[index]
                if not current.strip():
                    index += 1
                    continue
                if indentation(current) == 0:
                    # Another section starts where an unindented line is underlined
                    if index + 1 < len(lines) and NUMPY_UNDERLINE.match(lines[index + 1].strip()):
                        break
                    item = NUMPY_ITEM.match(current.strip())
                    if item is None:
                        break
                    names = [part.strip().lstrip("*") for part in item["names"].split(",")]
                    for each in names:
                        parameters[each] = ""
                else:
                    for each in names:
                        parameters[each] = join_lines([parameters[each], current])
                index += 1
            continue

        kept.append(line)
        index += 1

    description = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    if returns:
        description = f"{description}\n\nReturns: {' '.join(returns)}".strip()

    return description, parameters


def clean_schema(schema: Any) -> Any:
    """Drops the titles pydantic gives every schema, which repeat the names they sit under."""
    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key in SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            cleaned[key] = {name: clean_schema(item) for name, item in value.items()}
        elif key in SCHEMA_LIST_KEYWORDS and isinstance(value, list):
            cleaned[key] = [clean_schema(item) for item in value]
        elif key in SCHEMA_KEYWORDS:
            cleaned[key] = clean_schema(value)
        else:
            cleaned[key] = value

    return cleaned


class RecursiveSchema(Exception):
    """Raised when a schema refers to itself, so that its references cannot be written out."""


def inline_references(schema: dict[str, Any]) -> dict[str, Any]:
    """Writes each reference to a definition out in place, where none of them is recursive.

    Not every provider follows a "$ref" in a tool's parameters, but all of them read a schema
    written out in full. A recursive one cannot be, so it is left as it is.
    """
    definitions = schema.get("$defs")
    if not definitions:
        return schema

    def inline(value: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(value, list):
            return [inline(item, seen) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference[len("#/$defs/") :]
            if name in seen:
                raise RecursiveSchema(name)
            siblings = {key: item for key, item in value.items() if key != "$ref"}
            return {**inline(definitions[name], (*seen, name)), **inline(siblings, seen)}
        return {
            key: (
                {name: inline(item, seen) for name, item in item_value.items()}
                if key in SCHEMA_MAP_KEYWORDS and isinstance(item_value, dict)
                else inline(item_value, seen)
                if key in SCHEMA_KEYWORDS | SCHEMA_LIST_KEYWORDS
                else item_value
            )
            for key, item_value in value.items()
        }

    try:
        return inline({key: value for key, value in schema.items() if key != "$defs"}, ())
    except RecursiveSchema:
        return schema


def run_awaitable(awaitable: Any) -> Any:
    """Waits for what an async function returned, from code that is not itself async."""

    async def wait() -> Any:
        return await awaitable

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(wait())

    # A loop is already running in this thread, and a second cannot be run inside it
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, wait()).result()


def describe_errors(error: ValidationError) -> str:
    """Lists what was wrong with the arguments, one problem to a line, without pydantic's links."""
    problems = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"])
        problems.append(f"{location}: {problem['msg']}" if location else problem["msg"])

    return "; ".join(problems)


def innermost(function: Callable[..., Any]) -> Callable[..., Any]:
    while isinstance(function, functools.partial):
        function = function.func

    return function


def what_is_called(function: Callable[..., Any]) -> Any:
    """What runs when the callable is called: itself, a class's __init__, or an object's __call__."""
    target = innermost(function)
    if inspect.isclass(target):
        return target.__init__
    if inspect.isroutine(target):
        return target

    return type(target).__call__


def type_hints(function: Callable[..., Any], name: str) -> dict[str, Any]:
    """The function's type hints, resolved, or none where they cannot be."""
    try:
        return typing.get_type_hints(what_is_called(function), include_extras=True)
    except Exception as error:
        warnings.warn(
            f"Could not resolve the type hints of {name} ({type(error).__name__}: {error}), so the "
            "parameters whose hints cannot be read accept anything.",
            UserWarning,
            stacklevel=3,
        )
        return {}


class ParameterNote(BaseModel):
    """What one parameter of a function is for."""

    name: str = Field(description="The parameter's name, exactly as in the function's signature.")
    description: str = Field(description="What the parameter is, and what values it takes, in a sentence.")


class FunctionNotes(BaseModel):
    """A description of a function for a model deciding whether, and how, to call it."""

    description: str = Field(
        description="What the function does and what it returns, in one to three sentences, "
        "including anything a caller must know to use it correctly."
    )
    parameters: list[ParameterNote] = Field(description="Each of the function's parameters.")


def describe_with_model(
    function: Callable[..., Any],
    name: str,
    signature: inspect.Signature,
    model: str,
    chat_models: ModelSource | None,
    client: Any,
) -> tuple[str, dict[str, str]]:
    """Asks a model to describe a function from its source, as Biomni's add_tool does."""
    try:
        source = inspect.getsource(innermost(function))
    except (OSError, TypeError):
        source = f"def {name}{signature}: ..."
    if len(source) > MAX_SOURCE_CHARS:
        source = source[:MAX_SOURCE_CHARS] + "\n# ... the rest of the source is not shown"

    llm = resolve_chat_models([model], chat_models=chat_models, client=client)[model]
    notes, _ = request_structured_output(
        llm=llm,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You document Python functions for language models that will call them as tools.",
            },
            {
                "role": "user",
                "content": (
                    f"Describe the function {name} for a model deciding whether to call it, and with "
                    "what arguments. Describe only the parameters in its signature; do not invent "
                    "others. Be as clear and succinct as possible.\n\n"
                    f"Signature: {name}{signature}\n\nSource:\n{source}"
                ),
            },
        ],
        schema=FunctionNotes,
        temperature=None,
    )

    return notes.description.strip(), {note.name: note.description.strip() for note in notes.parameters}


def tool_from_function(
    function: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    model: str | None = None,
    chat_models: ModelSource | None = None,
    client: Any = None,
) -> Tool:
    """Makes a tool of a Python function, for a meeting's agents to call, or for code in a
    session to call, or both.

    The tool's parameters are the function's: those without a default are required, and each
    is typed by its type hint, which may be any type pydantic can read from JSON, such as str,
    int, float, bool, list[str], dict[str, float], Literal["a", "b"], an Enum, a Path, a date,
    or a pydantic model. A parameter without a hint accepts anything, and *args and **kwargs are
    left out, since a model cannot be told what they take. The arguments a model sends are
    converted to these types before the function is called, and arguments that cannot be are
    reported back to the model instead of being passed on. A function that is async is waited
    for.

    The descriptions come from the docstring: its text describes the tool, and its parameter
    documentation, in the reST, Google, or NumPy style, describes each parameter. Only if the
    function has no docstring, and no description is given, is a model asked to write them from
    its source. That costs one request, made now and counted towards no budget, and what it
    writes is printed: a model can describe the same function differently each time, and a
    project resumed with a tool described differently holds its meetings again, so keep a
    description you are happy with by adding it to the function as its docstring.

    :param function: The function, or any callable, such as a functools.partial binding some
        of its arguments, which are then left out of the tool's parameters.
    :param name: What the tool is called, defaulting to the function's name.
    :param description: What the tool does, in place of the docstring's description.
    :param model: A model to describe a function with no docstring, such as "gpt-4o". Without
        one, such a function needs a description.
    :param chat_models: The chat model to ask, as hold_meeting takes it.
    :param client: An OpenAI client, as hold_meeting takes it.
    :raises ValueError: If the function has no usable name, or no docstring, description, or
        model to describe it.
    :raises TypeError: If a parameter's type is one that cannot be read from JSON, such as a
        pandas DataFrame, and so cannot be passed by a model.
    :return: The tool. What the function returns is what code in a session receives; a model is
        shown it as text, with dicts, lists, and pydantic models written as JSON.
    """
    if not callable(function):
        raise TypeError(f"A tool is made of a function, not {type(function).__name__}")

    target = innermost(function)
    tool_name = name if name is not None else getattr(target, "__name__", type(target).__name__)
    if not TOOL_NAME.fullmatch(tool_name):
        raise ValueError(
            f"{tool_name!r} cannot name a tool: use letters, digits, '_', and '-', at most 64 of them. "
            "Pass name= to give the tool another."
        )

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Cannot read the parameters of {tool_name}: {error}") from error

    hints = type_hints(function, tool_name)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    # A callable object's docstring is its class's, which describes the object; its __call__'s
    # describes the call, where it has one
    docstring = inspect.getdoc(target)
    if not (inspect.isroutine(target) or inspect.isclass(target)):
        docstring = inspect.getdoc(type(target).__call__) or docstring
    text, documented = parse_docstring(docstring)

    if description is None and not text:
        if model is None:
            raise ValueError(
                f"{tool_name} has no docstring to describe it. Give it one, pass description=, or pass "
                "model= to have a model write one."
            )
        text, documented = describe_with_model(function, tool_name, signature, model, chat_models, client)
        print(
            f"{tool_name} has no docstring, so {model} described it as: {text}\n"
            f"Parameters: {json.dumps(documented)}\n"
            "Add this to the function as its docstring, or pass it as description, so that it is "
            "the same from one run to the next."
        )

    fields: dict[str, Any] = {}
    for position, parameter in enumerate(parameters):
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty or isinstance(annotation, str):
            annotation = Any
        has_default = parameter.default is not inspect.Parameter.empty
        # A parameter defaulting to None takes None, whatever its hint says
        if has_default and parameter.default is None and annotation is not Any:
            annotation = typing.Optional[annotation]  # noqa: UP007
        options: dict[str, Any] = {"alias": parameter.name}
        if has_default:
            options["default"] = parameter.default
        if documented.get(parameter.name):
            options["description"] = documented[parameter.name]
        # Fields get names of their own, and the parameters' names as aliases, so that a
        # parameter may be called anything, including what a pydantic model calls its methods
        fields[f"p{position}"] = (annotation, Field(**options))

    try:
        arguments_model: type[BaseModel] = create_model(  # type: ignore[call-overload]
            f"{tool_name}_arguments", __config__=ConfigDict(extra="forbid"), **fields
        )
        with warnings.catch_warnings():
            # A default that is not JSON is left out of the schema, which is all that warns of
            warnings.simplefilter("ignore", PydanticJsonSchemaWarning)
            schema = arguments_model.model_json_schema()
    except (PydanticSchemaGenerationError, PydanticUserError, TypeError) as error:
        raise TypeError(
            f"A parameter of {tool_name} has a type a model cannot pass as JSON, so it cannot be a tool's: "
            f"{error}. Take its value in a form that can be, such as a path to a file, and build it in the "
            "function."
        ) from error

    schema = inline_references(clean_schema(schema))
    schema.setdefault("properties", {})

    positional_only = [
        parameter.name for parameter in parameters if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ]

    def call(**arguments: Any) -> Any:
        try:
            validated = arguments_model.model_validate(arguments)
        except ValidationError as error:
            raise ValueError(f"Invalid arguments for {tool_name}: {describe_errors(error)}") from None

        # Only what was given is passed on, so that the function's own defaults apply
        given = {
            parameters[int(field_name[1:])].name: getattr(validated, field_name)
            for field_name in validated.model_fields_set
        }
        positional = []
        last_given = max((index for index, item in enumerate(positional_only) if item in given), default=-1)
        for item in positional_only[: last_given + 1]:
            positional.append(given.pop(item) if item in given else signature.parameters[item].default)

        result = function(*positional, **given)

        return run_awaitable(result) if inspect.isawaitable(result) else result

    # Named after the function, so that a record, and a project's fingerprint of the tool, say
    # which function it runs
    functools.update_wrapper(call, target, updated=())
    if not (inspect.isroutine(target) or inspect.isclass(target)):
        call.__module__ = type(target).__module__
        call.__qualname__ = type(target).__qualname__
    call.__wrapped__ = function  # type: ignore[attr-defined]

    return Tool(
        name=tool_name,
        description=description if description is not None else text,
        parameters=schema,
        function=call,
    )
