"""Tools made of the functions written for tasks, run in a session, or here.

Biomni's generated functions are files that a person reads, and then adds to its tool modules by
hand. Here a saved function is made a Tool as it stands, by tool_from_function, from the function's
own signature and docstring, which function_checks has already read and checked from its code. That
reading runs nothing, so making the tool does not run the model's code.

What runs it, when a model calls the tool, is chosen. A function written by a model that read a
paper is not code anyone has looked at, so by default it runs in a session, in the sandbox, with
its files and libraries, and what it does is contained. Running it in this process is possible, and
has to be asked for.
"""

import hashlib
import json
import sys
import threading
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic_core import PydanticSerializationError, to_jsonable_python

from virtual_lab.custom_tools import tool_from_function
from virtual_lab.execution import truncate_tail
from virtual_lab.function_checks import GENERATED_MODULE, FunctionReading, FunctionSourceError, read_function
from virtual_lab.function_generator import VERIFIED, function_path, saved_functions
from virtual_lab.session import Session
from virtual_lab.tools import Tool

# Most characters of what a failed function printed that its error carries, from the end, where
# the traceback is
MAX_FAILURE_OUTPUT_CHARS = 2_000

# Where a session keeps the modules of the functions it has run, by function, each with a digest
# of the source it was defined from, so that one is defined once and not on every call
SESSION_CACHE = "__virtual_lab_functions__"


class FunctionToolError(RuntimeError):
    """Raised when a function that is a tool fails, or cannot be run."""


def session_call_code(name: str, source: str, arguments: dict[str, Any]) -> str:
    """The code that calls a function in a session: it defines the function from its source, as a
    module of its own so that it adds nothing else to the session's names, the first time, and
    again if the source is not the one defined before, such as a function a person has corrected
    in a session that is still open. It ends in a single expression whose value is what the
    function returns."""
    module_name = f"{GENERATED_MODULE}_{name}"
    digest = hashlib.sha256(source.encode()).hexdigest()

    return (
        f"if globals().setdefault({SESSION_CACHE!r}, {{}}).get({name!r}, (None,))[0] != {digest!r}:\n"
        f"    {SESSION_CACHE}[{name!r}] = ({digest!r}, (lambda module: (\n"
        f"        __import__('sys').modules.__setitem__({module_name!r}, module),\n"
        f"        exec(compile({source!r}, {f'{name}.py'!r}, 'exec'), module.__dict__),\n"
        f"        module,\n"
        f"    )[2])(__import__('types').ModuleType({module_name!r})))\n"
        f"getattr({SESSION_CACHE}[{name!r}][1], {name!r})(**__import__('json').loads({json.dumps(arguments)!r}))\n"
    )


def run_in_session(session: Session, name: str, source: str) -> Callable[..., Any]:
    """What calls a function in a session, with the arguments a tool was called with."""

    def run(**arguments: Any) -> Any:
        try:
            sent = to_jsonable_python(arguments)
        except PydanticSerializationError as error:
            raise FunctionToolError(f"{name} was called with arguments that cannot be sent as JSON: {error}") from None

        result = session.evaluate(session_call_code(name, source, sent))
        if not result.succeeded:
            printed = truncate_tail(result.output.strip(), MAX_FAILURE_OUTPUT_CHARS)
            raise FunctionToolError(
                f"{name} failed: {result.error}" + (f"\n\nWhat it printed:\n{printed}" if printed else "")
            )

        # A function that returns nothing may have printed what it found instead
        if result.value is None and result.output.strip():
            return result.output

        return result.value

    return run


class LocalFunction:
    """A function that is run in this process, defined from its source the first time it is called,
    and not before, so that making a tool of it runs nothing."""

    def __init__(self, name: str, source: str, filename: str) -> None:
        self.name = name
        self.source = source
        self.filename = filename
        self.function: Callable[..., Any] | None = None
        self.lock = threading.Lock()

    def load(self) -> Callable[..., Any]:
        with self.lock:
            if self.function is None:
                module = types.ModuleType(f"{GENERATED_MODULE}_{self.name}")
                module.__file__ = self.filename
                sys.modules[module.__name__] = module
                try:
                    exec(compile(self.source, self.filename, "exec"), module.__dict__)  # noqa: S102
                    self.function = getattr(module, self.name)
                except BaseException:
                    sys.modules.pop(module.__name__, None)
                    raise

            return self.function

    def __call__(self, **arguments: Any) -> Any:
        return self.load()(**arguments)


def tool_from_source(
    source: str,
    name: str,
    session: Session | None = None,
    run_here: bool = False,
    filename: str | None = None,
) -> Tool:
    """A function's code as a tool, without running any of it.

    The tool is named and described as the function is, by its name and docstring, and takes its
    parameters, each as its type hint says, which are read from the code's syntax tree. When a
    model calls it, the function runs where it is told to:

    - in a session, with the session's files and libraries: the sandbox, if it is a DockerSession.
      The arguments are sent as JSON, and what the function returns must be JSON, as a session's
      evaluate sends back. A function that fails raises FunctionToolError with its error and the
      end of what it printed.
    - in this process, if run_here is True, with whatever the code does done as you. Do this only
      for code you have read.

    :param source: The function's file, as function_generator saved it.
    :param name: The function's name.
    :param session: The session to run it in.
    :param run_here: Whether to run it in this process, which is not done unless asked for.
    :param filename: What to call the file in an error's traceback, defaulting to name.py.
    :raises ValueError: If neither a session nor run_here is given, or both are, or the code cannot
        be a tool's function: see read_function.
    :return: The tool.
    """
    if (session is None) == (not run_here):
        raise ValueError(
            "Say where the function runs: in a session, such as a DockerSession, whose sandbox contains what "
            "it does, or in this process with run_here=True, which runs code a model wrote as you. Not both."
        )

    try:
        reading: FunctionReading = read_function(source, name)
    except FunctionSourceError as error:
        raise ValueError(f"{name} cannot be a tool: {error}") from error

    run: Callable[..., Any] = (
        run_in_session(session, name, source)
        if session is not None
        else LocalFunction(name, source, filename or f"{name}.py")
    )

    return tool_from_function(reading.stand_in(run), name=name)


def tools_from_saved_functions(
    directory: Path | str,
    session: Session | None = None,
    run_here: bool = False,
    names: Iterable[str] | None = None,
    verified_only: bool = False,
) -> tuple[Tool, ...]:
    """The functions saved in a directory by generate_functions, each as a tool.

    Each is read from its file as it is now, so that a function a person has corrected is the one
    that is used. A function that failed has no file and is not a tool.

    :param directory: Where the functions were saved.
    :param session: The session to run them in, or None.
    :param run_here: Whether to run them in this process, as tool_from_source says, which is not done
        unless asked for.
    :param names: The functions to take, by name, or None for all of them.
    :param verified_only: Whether to take only the functions that were imported in an executor.
    :raises ValueError: If where they run is not given, a name was not saved, or a file can no longer
        be a tool's function.
    :raises OSError: If a file cannot be read.
    :return: The tools, in the order of the functions' names.
    """
    directory = Path(directory)
    saved = [function for function in saved_functions(directory) if function.usable]
    if verified_only:
        saved = [function for function in saved if function.status == VERIFIED]
    if names is not None:
        wanted = [names] if isinstance(names, str) else list(names)
        if unknown := [name for name in wanted if name not in {function.name for function in saved}]:
            raise ValueError(f"{directory} holds no function {', '.join(map(repr, unknown))} that can be a tool")
        saved = [function for function in saved if function.name in wanted]

    tools = []
    for function in saved:
        path = function_path(directory, function.name)
        tools.append(
            tool_from_source(path.read_text(encoding="utf-8"), function.name, session, run_here, filename=str(path))
        )

    return tuple(tools)
