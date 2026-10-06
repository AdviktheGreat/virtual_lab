"""Asking a person before a tool does what costs money or cannot be undone, and passing on to a
person what an MCP server asks while one of its tools runs.

A tool of an MCP server can be made to wait for a person's approval of every call, as the
presets for Adaptyv's lab and for Proto make those of their tools that order experiments, pay,
or deploy. A server can also ask a person something itself, which MCP calls elicitation, as
Proto's server asks before it deploys a tool to Modal. Both are asked at the terminal unless
connect_mcp is given functions of its own to ask with. Where there is no terminal to ask at, a
call that needs approval is not made and a server's question is declined, so that nothing is
done that no one agreed to.
"""

import json
import math
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from virtual_lab.constants import APPROVAL_MAX_ARGUMENT_CHARS

# One question at a time, since calls to tools run at the same time and each could ask
TERMINAL = threading.Lock()

YES = frozenset({"y", "yes"})
TRUE = frozenset({"y", "yes", "true", "1"})
FALSE = frozenset({"n", "no", "false", "0"})


class ApprovalDeclined(RuntimeError):
    """Raised in place of a call to a tool that a person did not approve."""


@dataclass(frozen=True)
class ApprovalRequest:
    """A call to a tool that waits for a person's approval.

    :param tool: The tool's name, as agents and code call it.
    :param server: The name of the MCP server it is a tool of.
    :param server_tool: Its name on that server.
    :param arguments: What it is to be called with.
    :param description: What the tool does, as its server describes it.
    """

    tool: str
    server: str
    server_tool: str
    arguments: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class ServerQuestion:
    """What an MCP server asks a person while one of its tools runs.

    A question is answered with a form, whose fields are given, or at a web page, such as one to
    sign in or pay at, whose URL is given.

    :param server: The name of the MCP server that asks.
    :param message: What it asks.
    :param fields: The form's fields, by name, each a JSON Schema of the text, number, true or
        false, or choice it takes. Empty for a question answered at a web page.
    :param required: The names of the fields that must be filled in.
    :param url: The web page, for a question answered at one; otherwise None.
    """

    server: str
    message: str
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    url: str | None = None


# Decides whether a call is made: True makes it, and anything else does not. Raising
# ApprovalDeclined declines it with the reason given
Approve = Callable[[ApprovalRequest], bool]

# Answers a question: the form's fields, by name, for one answered with a form, or {} for one
# answered at a web page, once the person has gone to it; or None to decline it
Answer = Callable[[ServerQuestion], dict[str, Any] | None]


def terminal_available() -> bool:
    """Whether there is a person at a terminal to ask."""
    try:
        return sys.stdin is not None and sys.stdin.isatty() and sys.stderr is not None
    except (AttributeError, ValueError):
        return False


def say(text: str) -> None:
    # On stderr, which is where a meeting shows its progress, and not on stdout, which a
    # caller may be reading
    sys.stderr.write(text)
    sys.stderr.flush()


def read_reply(prompt: str) -> str | None:
    """A line typed in reply, or None once input has ended."""
    say(prompt)
    line = sys.stdin.readline()
    if not line:
        say("\n")
        return None

    return line.strip()


def shown_arguments(arguments: Mapping[str, Any]) -> str:
    text = json.dumps(dict(arguments), indent=2, ensure_ascii=False, default=str)
    if len(text) > APPROVAL_MAX_ARGUMENT_CHARS:
        left = len(text) - APPROVAL_MAX_ARGUMENT_CHARS
        text = f"{text[:APPROVAL_MAX_ARGUMENT_CHARS]}\n... and {left:,} more characters"

    return text


def first_paragraph(text: str) -> str:
    return text.strip().split("\n\n", 1)[0].strip()


def approve_in_terminal(request: ApprovalRequest) -> bool:
    """Asks at the terminal whether a call to a tool may be made.

    :raises ApprovalDeclined: If there is no terminal to ask at.
    :return: Whether the person approved it.
    """
    if not terminal_available():
        raise ApprovalDeclined(
            f"Each call to {request.tool} waits for a person's approval, and there is no terminal to ask at, "
            "so it was not made. To approve calls another way, pass approve to connect_mcp."
        )
    with TERMINAL:
        about = first_paragraph(request.description)
        say(
            f"\n{request.tool}, a tool of the MCP server {request.server}, waits for your approval of each call."
            + (f"\n{about}" if about else "")
            + f"\nIt is to be called with:\n{shown_arguments(request.arguments)}\n"
        )
        reply = read_reply("Allow this call? [y/N] ")

    return reply is not None and reply.lower() in YES


def choices(schema: Mapping[str, Any]) -> list[Any] | None:
    """The values a field may take, where it is a choice."""
    if isinstance(schema.get("enum"), list):
        return list(schema["enum"])
    for key in ("oneOf", "anyOf"):
        options = schema.get(key)
        if isinstance(options, list) and options and all(isinstance(option, Mapping) and "const" in option for option in options):
            return [option["const"] for option in options]

    return None


def field_kind(schema: Mapping[str, Any]) -> str:
    kind = schema.get("type")
    if kind == "array":
        return "array"
    if choices(schema) is not None:
        return "choice"

    return kind if kind in ("boolean", "integer", "number") else "string"


def read_value(text: str, schema: Mapping[str, Any]) -> tuple[bool, Any]:
    """The value of a field typed as text: whether it is one, and the value."""
    kind = field_kind(schema)
    if kind == "boolean":
        if text.lower() in TRUE:
            return True, True
        if text.lower() in FALSE:
            return True, False
        return False, None
    if kind == "integer":
        try:
            return True, int(text)
        except ValueError:
            return False, None
    if kind == "number":
        try:
            value = float(text)
        except ValueError:
            return False, None
        return math.isfinite(value), value
    if kind == "choice":
        allowed = choices(schema) or []
        for option in allowed:
            if text == str(option):
                return True, option
        if text.isdigit() and 1 <= int(text) <= len(allowed):
            return True, allowed[int(text) - 1]
        return False, None
    if kind == "array":
        items = schema.get("items") if isinstance(schema.get("items"), Mapping) else {}
        values = []
        for part in (part.strip() for part in text.split(",")):
            if part:
                ok, value = read_value(part, items)
                if not ok:
                    return False, None
                values.append(value)
        return True, values

    return True, text


def field_prompt(name: str, schema: Mapping[str, Any], required: bool) -> str:
    label = str(schema.get("title") or name)
    kind = field_kind(schema)
    if kind == "choice":
        options = choices(schema) or []
        detail = "one of " + ", ".join(f"{number}. {option}" for number, option in enumerate(options, 1))
    elif kind == "array":
        items = schema.get("items") if isinstance(schema.get("items"), Mapping) else {}
        options = choices(items)
        detail = "a list, separated by commas" + (f", of {', '.join(map(str, options))}" if options else "")
    else:
        detail = {"boolean": "yes or no", "integer": "a whole number", "number": "a number"}.get(kind, "text")
    if not required:
        default = f", or Enter for {json.dumps(schema['default'])}" if "default" in schema else ", or Enter to leave it out"
        detail += default
    description = f"\n  {schema['description']}" if isinstance(schema.get("description"), str) else ""

    return f"{label}{description}\n  ({detail}): "


def answer_in_terminal(question: ServerQuestion) -> dict[str, Any] | None:
    """Asks at the terminal what an MCP server asks.

    :return: The answer, or None if the person declined it, or there is no terminal to ask at.
    """
    if not terminal_available():
        return None
    with TERMINAL:
        say(f"\nThe MCP server {question.server} asks:\n{question.message.strip()}\n")
        if question.url is not None:
            say(f"It asks you to go to {question.url}\n")
            reply = read_reply("Open this page? [y/N] ")
            if reply is None or reply.lower() not in YES:
                return None
            try:
                webbrowser.open(question.url)
            except Exception:  # noqa: BLE001, S110
                # The address is shown above, to be opened by hand
                pass
            return {}

        fields = question.fields
        if not fields:
            reply = read_reply("Agree? [y/N] ")
            return {} if reply is not None and reply.lower() in YES else None

        # A question that is only yes or no, such as whether to go ahead, is asked as one
        if len(fields) == 1:
            ((name, schema),) = fields.items()
            if field_kind(schema) == "boolean":
                label = str(schema.get("title") or name)
                reply = read_reply(f"{label}? [y/N] ")
                if reply is None:
                    return None
                return {name: reply.lower() in YES}

        reply = read_reply("Answer it? [y/N] ")
        if reply is None or reply.lower() not in YES:
            return None
        answer: dict[str, Any] = {}
        for name, schema in fields.items():
            required = name in question.required
            while True:
                text = read_reply(field_prompt(name, schema, required))
                if text is None:
                    return None
                if not text and not required:
                    if "default" in schema:
                        answer[name] = schema["default"]
                    break
                ok, value = read_value(text, schema) if text else (False, None)
                if ok:
                    answer[name] = value
                    break
                say("That is not an answer this takes.\n")

    return answer
