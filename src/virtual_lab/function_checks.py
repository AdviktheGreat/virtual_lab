"""Reading a function that a model wrote, without running it.

Biomni's FunctionGenerator takes the first fenced block of the model's reply as the function and
names its file for the first six words of the task. It does not look at what it saved: a reply
without a block is saved as None, which fails, and one with a block is saved whatever is in it,
so a file may not parse, may define nothing, or may run the whole task the moment it is imported.

What is checked here is what a tool needs of a function, and all of it is read from the code's
syntax tree, so that nothing the model wrote is run to find it out. A function is a tool's if the
file parses, defines a function named for the task, with a docstring, and a type hint for each
parameter, of a type a model can send as JSON, and nothing in the file runs the task when it is
imported. The parameters are read into a signature without evaluating the file: a type hint is
evaluated only if it is made of names from a few standard modules and nothing else, so that no
call, and no attribute that is not public, is ever made.
"""

import ast
import builtins
import importlib
import inspect
import keyword
import re
import sys
import textwrap
import types
import typing
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from virtual_lab.constants import FUNCTION_NAME_WORDS, MAX_FUNCTION_NAME_CHARS, MAX_FUNCTION_SOURCE_CHARS
from virtual_lab.custom_tools import tool_from_function

# The module that every function written for a task is said to come from, so that the tool made
# of it, and a project's record of the tool, say what it is
GENERATED_MODULE = "virtual_lab_generated"

# The name a function's file is given when the task has no words to name it from, as Biomni names it
UNNAMED_SCRIPT = "script"

# Added to a name that is a keyword or the name of a standard module, which a file of that name
# could not be imported as, or would shadow
RESERVED_SUFFIX = "_task"

OPENING_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?P<language>[^\s`]*)")

# The languages a fenced block may be marked with and still be Python, as a model writes them. A
# block with no mark is taken to be, since Biomni's is
PYTHON_LANGUAGES = frozenset({"", "python", "python3", "py", "py3"})

# The one standard module a type hint may name, which is all that is imported to read one
HINT_MODULES = frozenset({"typing"})

# The types a parameter may have, which are the types JSON has. A tool is called with JSON, and a
# function run in a session is sent JSON, so a parameter of any other type, such as a Path or a
# tuple, would be sent as something else than its hint says
JSON_TYPES = (str, int, float, bool, list, dict, type(None))
HINT_BUILTINS = ("str", "int", "float", "bool", "list", "dict")

# What a type hint may be made of: names, the attributes of modules, subscripts, constants (None,
# the strings of a Literal, and the ... of a tuple), a union written with |, and a negative number
HINT_NODES = (
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.Tuple,
    ast.BinOp,
    ast.UnaryOp,
    ast.operator,
    ast.unaryop,
    ast.expr_context,
)


HINT_ADVICE = (
    "A tool is called with JSON, so use str, int, float, bool, list, dict, or an Optional, union, or Literal of "
    "them, and take a file's path as a str."
)


@dataclass(frozen=True)
class Code:
    """The code in a model's reply.

    :param source: The code, without the fence it was in.
    :param complete: False if the reply ended inside the fence, as one does that ran out of tokens.
    """

    source: str
    complete: bool


@dataclass(frozen=True)
class Block:
    language: str
    text: str
    closed: bool


class FunctionSourceError(ValueError):
    """Raised when code cannot be a tool's function.

    :param problems: What is wrong, one problem to a sentence, worded for the model that wrote it.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__(" ".join(problems))
        self.problems = list(problems)


class UnreadableDefault:
    """What stands for a default that is not a literal, such as a constant defined in the file.

    A tool does not send it: what is not given is left to the function, so the function's own
    default applies. It is only that the schema cannot say what it is.
    """

    def __repr__(self) -> str:
        return "<a default computed by the function>"


UNREADABLE_DEFAULT = UnreadableDefault()


def fenced_blocks(text: str) -> list[Block]:
    """The fenced code blocks of some Markdown, in order. A block never closed is one too.

    A block ends at a line of the same fence, at least as long, so that a block fenced with four
    backticks may hold lines of three.
    """
    blocks: list[Block] = []
    opening: tuple[str, str] | None = None
    lines: list[str] = []

    for line in text.splitlines():
        if opening is None:
            if found := OPENING_FENCE.match(line):
                opening, lines = (found["fence"], found["language"].lower()), []
            continue

        fence, language = opening
        if re.fullmatch(rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*", line):
            blocks.append(Block(language, "\n".join(lines), True))
            opening = None
        else:
            lines.append(line)

    if opening is not None:
        blocks.append(Block(opening[1], "\n".join(lines), False))

    return blocks


def parses(source: str) -> bool:
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return False

    return True


def extract_code(reply: str) -> Code | None:
    """The Python in a model's reply: its first fenced block that is not marked as another
    language, or, if it has no fences, the whole reply if that is Python.

    Biomni takes the first fenced block of any language, so a reply that begins with a block of
    shell commands to install a library is saved as a script of those.

    :param reply: What the model said.
    :return: The code, or None if there is none.
    """
    blocks = fenced_blocks(reply)
    for block in blocks:
        if block.language in PYTHON_LANGUAGES and block.text.strip():
            return Code(textwrap.dedent(block.text).strip(), block.closed)

    if not blocks and reply.strip() and parses(textwrap.dedent(reply).strip()):
        return Code(textwrap.dedent(reply).strip(), True)

    return None


def script_name(task: str, max_words: int = FUNCTION_NAME_WORDS) -> str:
    """What a task's function and its file are called, made as Biomni makes the file's name.

    The first six words of the task, in lowercase, with anything that is not a letter or a digit
    left out of them, joined with underscores. A name that Python could not import a file of, or
    would take for another module, is changed: one that begins with a digit is given "task_", and
    a keyword or the name of a standard module is given "_task". It is at most 64 characters.

    :param task: What the function is to do.
    :param max_words: How many of the task's words the name is made of.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", task.lower())
    name = "_".join(cleaned.split()[:max_words] or [UNNAMED_SCRIPT])
    if name[0].isdigit():
        name = f"task_{name}"
    name = name[:MAX_FUNCTION_NAME_CHARS]

    if keyword.iskeyword(name) or name in sys.stdlib_module_names:
        name = f"{name[: MAX_FUNCTION_NAME_CHARS - len(RESERVED_SUFFIX)]}{RESERVED_SUFFIX}"

    return name


def is_main_guard(statement: ast.stmt) -> bool:
    """Whether a statement is `if __name__ == "__main__":`, whose body runs only when the file is run."""
    if not (isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare)):
        return False

    test = statement.test
    operands = [test.left, *test.comparators]

    return (
        len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and any(isinstance(operand, ast.Name) and operand.id == "__name__" for operand in operands)
        and any(isinstance(operand, ast.Constant) and operand.value == "__main__" for operand in operands)
    )


def module_level(statements: list[ast.stmt]) -> list[ast.stmt]:
    """The statements that run when a file is imported: those at its top level, and in the blocks
    of if, try, and with there, but not in a function or a class, or under a main guard."""
    found: list[ast.stmt] = []
    for statement in statements:
        found.append(statement)
        if isinstance(statement, ast.With) or (isinstance(statement, ast.If) and not is_main_guard(statement)):
            found += module_level(statement.body)
        if isinstance(statement, ast.If):
            found += module_level(statement.orelse)
        if isinstance(statement, ast.Try):
            found += module_level(statement.body + statement.orelse + statement.finalbody)
            for handler in statement.handlers:
                found += module_level(handler.body)

    return found


def imported_modules(tree: ast.AST) -> list[str]:
    """The modules a file imports, by their top-level names, anywhere in it, sorted."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])

    return sorted(names)


def third_party(modules: list[str]) -> list[str]:
    """Those of the modules that are not part of Python, and so must be installed."""
    return [module for module in modules if module not in sys.stdlib_module_names]


def hint_names(tree: ast.Module) -> dict[str, Any]:
    """The names a type hint in the file may use: the types Python has, and what the file imports
    from the few standard modules type hints are made of. Nothing else is imported."""
    names: dict[str, Any] = {name: getattr(builtins, name) for name in HINT_BUILTINS}

    for statement in module_level(tree.body):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name in HINT_MODULES:
                    module = importlib.import_module(alias.name)
                    if alias.asname:
                        names[alias.asname] = module
                    else:
                        top = alias.name.split(".")[0]
                        names[top] = importlib.import_module(top)
        elif isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module in HINT_MODULES:
            module = importlib.import_module(statement.module)
            for alias in statement.names:
                if alias.name == "*":
                    names.update({name: getattr(module, name) for name in getattr(module, "__all__", ())})
                elif not alias.name.startswith("_") and hasattr(module, alias.name):
                    names[alias.asname or alias.name] = getattr(module, alias.name)

    return names


def hint_value(node: ast.expr, names: dict[str, Any]) -> Any:
    """What a type hint stands for, evaluated only if it is made of names and nothing else.

    :raises ValueError: If the hint is made of anything more, or names something that is not one
        of the types Python has or the file imports from a standard module.
    """
    # A hint written as a string stands for the hint inside it
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value.strip(), mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"{node.value!r} is not a type") from error

    for part in ast.walk(node):
        if not isinstance(part, HINT_NODES):
            raise ValueError(f"it is made of more than names: {ast.unparse(node)}")
        if isinstance(part, ast.Attribute) and part.attr.startswith("_"):
            raise ValueError(f"it names {part.attr}, which is not public")
        if isinstance(part, ast.BinOp) and not isinstance(part.op, ast.BitOr):
            raise ValueError("only | may join types")
        if isinstance(part, ast.UnaryOp) and not isinstance(part.op, ast.USub):
            raise ValueError("only a minus sign may stand before a number")

    expression = ast.fix_missing_locations(ast.Expression(body=node))
    try:
        # Nothing in it can call anything, and there are no builtins for a name to reach
        return eval(compile(expression, "<type hint>", "eval"), {"__builtins__": {}}, names)  # noqa: S307
    except NameError as error:
        if error.name in typing.__all__:
            raise ValueError(f"{error.name} is not imported. Add `from typing import {error.name}`") from None
        raise ValueError(f"{error.name} is not one of the types a tool takes") from None
    except Exception as error:
        raise ValueError(f"{type(error).__name__}: {error}") from None


def not_json(hint: Any) -> str | None:
    """Why a type hint is not of the types JSON has, or None if it is. A list or dict may be of
    any of them, and a union or Optional may be of them, and a Literal of strings, numbers, and booleans."""
    if hint is Any or hint in JSON_TYPES:
        return None

    origin, arguments = typing.get_origin(hint), typing.get_args(hint)
    if origin in (list, dict):
        return next((why for argument in arguments if (why := not_json(argument))), None)
    if origin in (typing.Union, types.UnionType):
        return next((why for argument in arguments if (why := not_json(argument))), None)
    if origin is typing.Literal:
        if all(value is None or isinstance(value, str | int | float | bool) for value in arguments):
            return None
        return "a Literal may hold only strings, numbers, booleans, and None"

    return f"{hint!r} is not one of the types JSON has"


def default_value(node: ast.expr | None) -> Any:
    if node is None:
        return inspect.Parameter.empty
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return UNREADABLE_DEFAULT


def read_signature(function: ast.FunctionDef, names: dict[str, Any]) -> tuple[inspect.Signature | None, list[str]]:
    """The signature of a function as its code declares it, and what is wrong with it for a tool."""
    problems: list[str] = []
    arguments = function.args
    kinds = inspect.Parameter

    def read(argument: ast.arg, default: ast.expr | None, kind: Any) -> inspect.Parameter | None:
        if argument.annotation is None:
            problems.append(
                f"Parameter {argument.arg} has no type hint. Give every parameter one, such as str or list[str]."
            )
            return None
        try:
            annotation = hint_value(argument.annotation, names)
        except ValueError as error:
            problems.append(
                f"The type hint of parameter {argument.arg}, {ast.unparse(argument.annotation)}, cannot be used: "
                f"{error}. {HINT_ADVICE}"
            )
            return None
        if (why := not_json(annotation)) is not None:
            problems.append(
                f"The type hint of parameter {argument.arg}, {ast.unparse(argument.annotation)}, cannot be used: "
                f"{why}. {HINT_ADVICE}"
            )
            return None
        if default is not None and any(isinstance(part, ast.Call) for part in ast.walk(default)):
            problems.append(
                f"The default of parameter {argument.arg}, {ast.unparse(default)[:60]}, is computed when the file is "
                "imported, once, not each time the function is called. Default it to None and compute it inside."
            )
            return None

        return kinds(argument.arg, kind, default=default_value(default), annotation=annotation)

    for argument in arguments.posonlyargs:
        problems.append(
            f"Parameter {argument.arg} is positional-only, but a tool's parameters are passed by name. "
            "Remove the / from the signature."
        )

    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    parameters = [
        read(argument, default, kinds.POSITIONAL_OR_KEYWORD)
        for argument, default in zip(positional, defaults, strict=True)
        if argument not in arguments.posonlyargs
    ]
    # *args and **kwargs are not a tool's parameters, and are kept so that the signature is the function's
    if arguments.vararg:
        parameters.append(kinds(arguments.vararg.arg, kinds.VAR_POSITIONAL))
    parameters += [
        read(argument, default, kinds.KEYWORD_ONLY)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
    ]
    if arguments.kwarg:
        parameters.append(kinds(arguments.kwarg.arg, kinds.VAR_KEYWORD))

    if problems:
        return None, problems

    try:
        return inspect.Signature([parameter for parameter in parameters if parameter is not None]), []
    except ValueError as error:
        return None, [f"The parameters of the function cannot be read: {error}."]


@dataclass(frozen=True)
class FunctionReading:
    """A function as its code declares it, which is all a tool is made of.

    :param name: The function's name, which is also the file's and the tool's.
    :param signature: Its parameters, with their type hints and defaults.
    :param docstring: Its docstring, which describes the tool and its parameters.
    :param modules: The modules the file imports that are not Python's own, and so must be
        installed for the function to run.
    """

    name: str
    signature: inspect.Signature
    docstring: str
    modules: tuple[str, ...]

    def stand_in(self, run: Callable[..., Any]) -> Callable[..., Any]:
        """A function with this one's name, signature, and docstring that runs what it is given
        in place of its body, so that a tool can be made of it with tool_from_function, which
        reads its parameters from them, without the function's code having been run.

        :param run: What is called, with the arguments the tool was called with, by name.
        """

        def function(**arguments: Any) -> Any:
            return run(**arguments)

        function.__signature__ = self.signature  # type: ignore[attr-defined]
        function.__annotations__ = {
            parameter.name: parameter.annotation
            for parameter in self.signature.parameters.values()
            if parameter.annotation is not inspect.Parameter.empty
        }
        function.__name__ = function.__qualname__ = self.name
        function.__module__ = GENERATED_MODULE
        function.__doc__ = self.docstring

        return function


def refuse_to_run(**arguments: Any) -> Any:
    raise RuntimeError("This stands for a function that has not been loaded")


def parse_function_source(source: str) -> ast.Module:
    """The syntax tree of a function's code, or why there is none, as a FunctionSourceError."""
    if len(source) > MAX_FUNCTION_SOURCE_CHARS:
        raise FunctionSourceError(
            [
                f"The code is {len(source):,} characters, more than the {MAX_FUNCTION_SOURCE_CHARS:,} "
                "a function may be. Write less."
            ]
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(source)
    except SyntaxError as error:
        where = f" on line {error.lineno}" if error.lineno else ""
        raise FunctionSourceError([f"The code is not valid Python: {error.msg}{where}."]) from error
    except (ValueError, RecursionError, MemoryError) as error:
        raise FunctionSourceError([f"The code cannot be read as Python: {error}."]) from error


def import_problems(tree: ast.Module) -> list[str]:
    """What the file does when it is imported that it should only do when it is run."""
    problems: list[str] = []
    for statement in module_level(tree.body):
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            problems.append(
                f"Line {statement.lineno} runs {ast.unparse(statement.value)[:60]} at the top level, so importing "
                "the file runs it. Importing must only define things: put it under `if __name__ == '__main__':`."
            )
        elif isinstance(statement, ast.For | ast.While):
            problems.append(
                f"Line {statement.lineno} loops at the top level, so importing the file runs it. Put it in a function."
            )

    return problems


def declared_function(tree: ast.Module, name: str) -> tuple[FunctionReading | None, list[str]]:
    """The function a file defines under a name, as a tool's, and what is wrong with it if it is not."""
    defined = [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    named = [node for node in defined if node.name == name]
    if not named:
        others = f" It defines {', '.join(node.name for node in defined)}." if defined else ""
        missing = f"The code defines no function named {name}. Name the function that does the task {name}."
        return None, [missing + others]

    # The last, which is the one a file that defines it twice has
    function = named[-1]
    if isinstance(function, ast.AsyncFunctionDef):
        return None, [f"{name} is an async function. Define it with def: a tool is called as an ordinary function."]
    if function.decorator_list:
        return None, [f"{name} has a decorator. A tool is a plain function: remove it."]

    problems: list[str] = []
    docstring = ast.get_docstring(function)
    if not docstring:
        problems.append(
            f"{name} has no docstring. Give it one that says what it does, what each parameter is, and what it returns."
        )
    signature, signature_problems = read_signature(function, hint_names(tree))
    problems += signature_problems
    if problems or signature is None:
        return None, problems

    return (
        FunctionReading(
            name=name,
            signature=signature,
            docstring=docstring or "",
            modules=tuple(third_party(imported_modules(tree))),
        ),
        [],
    )


def read_function(source: str, name: str) -> FunctionReading:
    """Reads the function a task's code defines, without running anything in it.

    :param source: The file's code.
    :param name: What the function must be called.
    :raises FunctionSourceError: If the code cannot be a tool's function: it is not Python, or
        too long, or defines no function of that name, or the function is async or has a
        decorator, or has no docstring, or has a parameter that has no type hint, or one that
        is not of a type a model can send as JSON, or the file does something when it is
        imported rather than only defining things. Every problem found is in the error.
    :return: What the function takes, and what it is for.
    """
    tree = parse_function_source(source)
    reading, problems = declared_function(tree, name)
    problems = [*problems, *import_problems(tree)]

    if reading is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tool_from_function(reading.stand_in(refuse_to_run), name=name)
        except (TypeError, ValueError) as error:
            problems.insert(0, str(error))

    if problems or reading is None:
        raise FunctionSourceError(problems)

    return reading
