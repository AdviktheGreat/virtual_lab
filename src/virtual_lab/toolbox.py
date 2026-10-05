"""Biomni's tool functions as tools, each run in a session.

Biomni's agent imports its tool functions into the interpreter its code runs in, and Biomni's MCP
server calls them in the server's own process, beside everything else it holds. Here each is a
Tool whose call runs the function in a session: in the sandbox, with Biomni's environment, where
its libraries are installed and what it does is contained, with the session's files and
variables at hand. Every call is in the session's history, and what the function returns is
sent back whole. A model can call one as it calls any tool, in a meeting or through serve_mcp.

Biomni types each parameter in words, such as "List[str]", "numpy.ndarray", or "str or dict",
and its MCP server passes everything it does not recognise as a string. Here each type is read
into a JSON Schema, and a list sent for an array, a tuple, or a table is made one before the
function is called.
"""

import ast
import contextlib
import json
import re
import warnings
from collections.abc import Iterable
from typing import Any

from virtual_lab.execution import truncate_tail
from virtual_lab.resources import BIOMNI_TOOL_MODULES, biomni_tools, check_installed
from virtual_lab.session import Session
from virtual_lab.tools import Tool

# Most characters of what a failed function printed that its error carries, from the end, where
# the traceback is
MAX_FAILURE_OUTPUT_CHARS = 2_000

# Biomni's types that JSON has, by the names Biomni writes them in, lowercased
SCALAR_SCHEMAS: dict[str, dict[str, Any]] = {
    "str": {"type": "string"},
    "string": {"type": "string"},
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "float": {"type": "number"},
    "number": {"type": "number"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
    "none": {"type": "null"},
    "dict": {"type": "object"},
    "list": {"type": "array"},
    "tuple": {"type": "array"},
    "array-like": {"type": "array"},
    "any": {},
}

# Types sent as a list and made something else in the session, with the code that makes them
CONVERSIONS = {
    "numpy.ndarray": '__import__("numpy").asarray',
    "np.ndarray": '__import__("numpy").asarray',
    "ndarray": '__import__("numpy").asarray',
    "pd.dataframe": '__import__("pandas").DataFrame',
    "pandas.dataframe": '__import__("pandas").DataFrame',
    "dataframe": '__import__("pandas").DataFrame',
    "tuple": '__import__("builtins").tuple',
}

# Types no JSON can stand for
UNPASSABLE = frozenset({"callable", "function"})

GENERIC = re.compile(r"^(?P<name>[\w.]+)\s*\[(?P<arguments>.*)\]$", re.DOTALL)
LIST_OF = re.compile(r"^(?:list|array)\s+of\s+(?P<item>.+)$", re.IGNORECASE | re.DOTALL)


class BiomniToolError(RuntimeError):
    """Raised when a Biomni function run in a session fails, or cannot be run."""


def split_top_level(text: str, separator: str) -> list[str]:
    """Splits text at a separator, except inside brackets."""
    parts, depth, start = [], 0, 0
    for index, character in enumerate(text):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])

    return [part.strip() for part in parts]


def alternatives(type_name: str) -> list[str]:
    """The types a Biomni type allows, which "or", "|", Union, and Optional join."""
    text = re.sub(r"\s+or\s+", "|", type_name.strip())
    found = []
    for part in split_top_level(text, "|"):
        generic = GENERIC.match(part)
        name = generic["name"].casefold() if generic else ""
        if name in ("union", "typing.union"):
            found.extend(item for argument in split_top_level(generic["arguments"], ",") for item in alternatives(argument))
        elif name in ("optional", "typing.optional"):
            found.extend([*alternatives(generic["arguments"]), "None"])
        elif part:
            found.append(part)

    return found


def single_schema(type_name: str) -> dict[str, Any] | None:
    """The JSON Schema of one of the types a Biomni type allows, or None if JSON has none."""
    lowered = type_name.strip().casefold()
    if lowered in UNPASSABLE:
        return None
    if lowered in SCALAR_SCHEMAS:
        return dict(SCALAR_SCHEMAS[lowered])
    if lowered in CONVERSIONS:
        if "dataframe" in lowered:
            return {"type": "array", "items": {"type": "object"}}
        return {"type": "array"}

    if listed := LIST_OF.match(type_name.strip()):
        return array_of(listed["item"])

    generic = GENERIC.match(type_name.strip())
    if generic is None:
        return {}
    name = generic["name"].casefold().removeprefix("typing.")
    arguments = split_top_level(generic["arguments"], ",")
    if name in ("list", "sequence", "set"):
        return array_of(arguments[0])
    if name in ("dict", "mapping"):
        values = biomni_type_schema(arguments[1]) if len(arguments) == 2 else {}
        return {"type": "object", **({"additionalProperties": values} if values else {})}
    if name == "tuple":
        if len(arguments) == 2 and arguments[1] == "...":
            return array_of(arguments[0])
        items = combined([biomni_type_schema(argument) for argument in arguments])
        return {
            "type": "array",
            **({"items": items} if items else {}),
            "minItems": len(arguments),
            "maxItems": len(arguments),
        }

    return {}


def array_of(item_type: str) -> dict[str, Any]:
    items = biomni_type_schema(item_type)
    return {"type": "array", **({"items": items} if items else {})}


def combined(schemas: list[dict[str, Any] | None]) -> dict[str, Any]:
    """The schema allowing any of the schemas, of which None allows nothing and {} anything."""
    options: list[dict[str, Any]] = []
    for schema in schemas:
        if schema is None:
            continue
        if not schema:
            return {}
        for option in schema["anyOf"] if set(schema) == {"anyOf"} else [schema]:
            if option not in options:
                options.append(option)

    if not options:
        return {}

    return options[0] if len(options) == 1 else {"anyOf": options}


def biomni_type_schema(type_name: str) -> dict[str, Any] | None:
    """The JSON Schema of a parameter that Biomni types as type_name.

    :param type_name: The type, as Biomni's tool descriptions write it, such as "List[str]".
    :return: The schema, which is {} for a type it cannot tell; or None if no JSON can stand for
        a value of the type, such as a callable.
    """
    found = alternatives(type_name)
    schemas = [single_schema(item) for item in found]
    if found and all(schema is None for schema in schemas):
        return None

    return combined(schemas)


def biomni_conversion(type_name: str) -> str | None:
    """The code that makes a list sent for a parameter of the type what the function takes,
    such as a numpy array, or None if a list is already what it takes."""
    found = [item for item in alternatives(type_name) if item.casefold() != "none"]
    if len(found) != 1:
        return None
    lowered = found[0].casefold()
    generic = GENERIC.match(found[0])
    if generic and generic["name"].casefold().removeprefix("typing.") == "tuple":
        lowered = "tuple"

    return CONVERSIONS.get(lowered)


def fits(value: Any, schema: dict[str, Any]) -> bool:
    """Whether a JSON value is one of those a schema written by biomni_type_schema allows."""
    if not schema:
        return True
    if "anyOf" in schema:
        return any(fits(value, option) for option in schema["anyOf"])

    kind = schema.get("type")
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "object":
        values = schema.get("additionalProperties")
        return isinstance(value, dict) and (
            not isinstance(values, dict) or all(fits(item, values) for item in value.values())
        )
    if kind == "array":
        items = schema.get("items")
        return (
            isinstance(value, list)
            and schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value))
            and (not isinstance(items, dict) or all(fits(item, items) for item in value))
        )

    return True


def jsonable_default(value: Any) -> Any:
    """A default as JSON writes it, or a value no JSON value equals if it cannot be."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return NotImplemented


def parameter_schema(parameter: dict[str, Any], required: bool) -> dict[str, Any] | None:
    """The JSON Schema of one of a Biomni function's parameters, or None if it cannot be passed."""
    type_name = str(parameter.get("type") or "")
    schema = biomni_type_schema(type_name) if type_name else {}
    if schema is None:
        return None

    notes = [str(parameter.get("description") or "").strip()]
    if not schema and type_name:
        notes.append(f"Biomni's type: {type_name}.")
    if "dataframe" in type_name.casefold():
        notes.append("Pass the table's rows, each an object of column to value.")

    default = parameter.get("default")
    if not required and default is not None:
        # Biomni writes some defaults as the code that makes them, such as "'LSODA'" or "(0, 100)"
        candidates = [default]
        if isinstance(default, str):
            with contextlib.suppress(ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                candidates.insert(0, ast.literal_eval(default))
        written = next(
            (
                value
                for value in map(jsonable_default, candidates)
                if value is not NotImplemented and value is not None and fits(value, schema)
            ),
            NotImplemented,
        )
        if written is not NotImplemented:
            schema["default"] = written
        else:
            notes.append(f"Defaults to {default}.")

    description = " ".join(note for note in notes if note)
    if description:
        schema = {"description": description, **schema}

    return schema


def call_code(module: str, function: str, arguments: dict[str, Any], conversions: dict[str, str]) -> str:
    """The code that calls a Biomni function with the arguments, a single expression whose value
    is what the function returns, and which leaves no name behind in the session."""
    converted = {name: value for name, value in arguments.items() if name in conversions and isinstance(value, list)}
    plain = {name: value for name, value in arguments.items() if name not in converted}

    passed = [f'**__import__("json").loads({json.dumps(plain)!r})'] if plain else []
    passed.extend(
        f'{name}={conversions[name]}(__import__("json").loads({json.dumps(value)!r}))'
        for name, value in converted.items()
    )

    return f'__import__("importlib").import_module({module!r}).{function}({", ".join(passed)})'


def biomni_tool(session: Session, api: dict[str, Any]) -> Tool | None:
    """One of Biomni's functions as a tool run in the session, or None if it cannot be one."""
    name, module = str(api.get("name") or ""), str(api.get("module") or "")
    if not name.isidentifier():
        return None

    properties: dict[str, Any] = {}
    required: list[str] = []
    conversions: dict[str, str] = {}
    for kind in ("required_parameters", "optional_parameters"):
        for parameter in api.get(kind) or []:
            parameter_name = str(parameter.get("name") or "")
            if not parameter_name.isidentifier():
                continue
            schema = parameter_schema(parameter, required=kind == "required_parameters")
            if schema is None:
                continue
            properties[parameter_name] = schema
            if kind == "required_parameters":
                required.append(parameter_name)
            if conversion := biomni_conversion(str(parameter.get("type") or "")):
                conversions[parameter_name] = conversion

    def call(**arguments: Any) -> Any:
        if unknown := sorted(set(arguments) - set(properties)):
            raise BiomniToolError(
                f"{name} has no parameter {', '.join(unknown)}; it takes {', '.join(properties) or 'none'}"
            )
        if missing := [parameter for parameter in required if parameter not in arguments]:
            raise BiomniToolError(f"{name} needs {', '.join(missing)}")

        result = session.evaluate(call_code(module, name, arguments, conversions))
        if not result.succeeded:
            printed = truncate_tail(result.output.strip(), MAX_FAILURE_OUTPUT_CHARS)
            raise BiomniToolError(
                f"{name} failed: {result.error}" + (f"\n\nWhat it printed:\n{printed}" if printed else "")
            )

        # A function that returns nothing may have printed what it found instead
        if result.value is None and result.output.strip():
            return result.output

        return result.value

    call.__name__ = call.__qualname__ = name
    call.__module__ = module

    return Tool(
        name=name,
        description=str(api.get("description") or "").strip(),
        parameters={"type": "object", "properties": properties, "required": required},
        function=call,
    )


def biomni_session_tools(
    session: Session,
    modules: Iterable[str] | None = None,
    names: Iterable[str] | None = None,
    check: bool = True,
) -> tuple[Tool, ...]:
    """Biomni's tool functions, each a tool that runs it in the session.

    Each is named and described as Biomni names and describes it, and takes the parameters
    Biomni lists for it, typed from the types Biomni gives them. A call runs the function in
    the session, as code the session's history records, so a path it is given or writes to is
    one the session's code sees, and it can use the session's files. A function that fails
    raises BiomniToolError, with its error and the end of what it printed. A function that
    returns nothing returns what it printed instead.

    :param session: The session, whose code must be able to import Biomni's tools: a
        DockerSession, by default, or a LocalSession with biomni_tools=True and Biomni's
        environment as its python.
    :param modules: The modules of biomni.tool to take functions from, such as "genetics", or
        None for all of them.
    :param names: The functions to take, by name, or None for every one in the modules.
    :param check: Whether to start the session, if it is not running, and leave out the functions
        of any module that fails to import in it, with a warning. A stage of the sandbox image
        other than "full" has only some of Biomni's libraries.
    :raises ValueError: If the session's code cannot import Biomni's tools, or a module or name is
        not one of Biomni's.
    :return: The tools.
    """
    if not session.has_biomni_tools():
        raise ValueError(
            "Code in this session cannot import Biomni's tools. Use a DockerSession, whose image has "
            "them, or a LocalSession with biomni_tools=True and Biomni's environment as its python."
        )

    every = biomni_tools()
    chosen_modules = None
    if modules is not None:
        given = [modules] if isinstance(modules, str) else list(modules)
        if not all(isinstance(module, str) for module in given):
            raise TypeError("Biomni's tool modules are named by str, such as 'genetics'")
        if unknown := [module for module in given if module.removeprefix("biomni.tool.") not in BIOMNI_TOOL_MODULES]:
            raise ValueError(
                f"Biomni has no tool module {', '.join(map(repr, unknown))}: its modules are "
                f"{', '.join(BIOMNI_TOOL_MODULES)}"
            )
        chosen_modules = {f"biomni.tool.{module.removeprefix('biomni.tool.')}" for module in given}
        every = tuple(api for api in every if api["module"] in chosen_modules)

    if names is not None:
        wanted = [names] if isinstance(names, str) else list(names)
        known = {api["name"] for api in every}
        if unknown := [name for name in wanted if name not in known]:
            where = "the modules chosen" if chosen_modules is not None else "Biomni's tools"
            raise ValueError(f"There is no {', '.join(map(repr, unknown))} among {where}")
        every = tuple(api for api in every if api["name"] in set(wanted))

    if check and every:
        modules_used = list(dict.fromkeys(api["module"] for api in every))
        checked = check_installed(session, [], modules_used, what="Biomni's tool modules")
        if checked is not None and (failed := checked[1]):
            warnings.warn(
                "Left out the functions of Biomni's modules that fail to import in the session: "
                + "; ".join(f"{module} ({error})" for module, error in failed.items()),
                UserWarning,
                stacklevel=2,
            )
            every = tuple(api for api in every if api["module"] not in failed)

    return tuple(tool for api in every if (tool := biomni_tool(session, api)) is not None)
