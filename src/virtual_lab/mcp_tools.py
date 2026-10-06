"""Tools from MCP servers, for a meeting's agents to call, or for code in a session to call.

Biomni's add_mcp reads servers from a YAML file, starts each one afresh for every call to one of
its tools, over stdio only, and gives back the first piece of what the tool returned, whether or
not the tool said it had failed. Here a server is started once and kept running, so that what it
loads or holds lasts from one call to the next, and calls to it can run at the same time. It can
be a program started here, over stdio, or a server reached at a URL, over streamable HTTP or
SSE. A tool's parameters are the ones the server declares; what it returns is given back whole,
as the structured result where the server gives one; and a tool that fails says so, so that the
agent is told the call failed rather than handed the error as if it were the answer.

A config is Biomni's YAML, under mcp_servers, or the JSON that Claude Desktop, Claude Code,
Cursor, and others write, under mcpServers.
"""

import asyncio
import atexit
import base64
import binascii
import concurrent.futures
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import tempfile
import threading
import warnings
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path
from typing import Any, Self
from urllib.parse import quote, quote_plus, urlsplit

from virtual_lab.__about__ import __version__
from virtual_lab.approval import (
    Answer,
    ApprovalDeclined,
    ApprovalRequest,
    Approve,
    ServerQuestion,
    answer_in_terminal,
    approve_in_terminal,
    asked_for,
    withdrawn,
)
from virtual_lab.constants import (
    MCP_CALL_TIMEOUT,
    MCP_LOG_TAIL_CHARS,
    MCP_MAX_INSTRUCTIONS_CHARS,
    MCP_MAX_TOOL_PAGES,
    MCP_START_TIMEOUT,
    MCP_STOP_TIMEOUT,
)
from virtual_lab.custom_tools import clean_schema, inline_references
from virtual_lab.mcp_presets import MCP_PRESETS, preset_entry
from virtual_lab.records import truncate_text
from virtual_lab.tools import Tool

# Where a config lists its servers: Biomni's YAML, and the JSON of Claude Desktop and the rest
CONFIG_KEYS = ("mcp_servers", "mcpServers")

# What a server's entry can say. Anything else is warned of, since it is not used
SERVER_KEYS = frozenset(
    {
        "command",
        "args",
        "env",
        "cwd",
        "url",
        "httpUrl",
        "headers",
        "type",
        "transport",
        "enabled",
        "disabled",
        "tools",
        "description",
        "preset",
        "approval",
        "instructions",
    }
)

# What a server's approval can say: the tools that wait for a person's approval of each call, and
# those of them that do not after all
APPROVAL_KEYS = frozenset({"ask", "allow"})

STREAMABLE_HTTP_TYPES = frozenset({"http", "streamable-http", "streamable_http", "streamableHttp"})

# ${NAME}, or ${NAME:-default} for a value to use where NAME is not set
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# The shortest value of a variable that messages hide. A shorter one could hardly be a secret,
# and hiding everywhere it appears would garble the message
MIN_HIDDEN_CHARS = 4

# The longest name a tool may have, which is what OpenAI's API accepts
MAX_TOOL_NAME_CHARS = 64

# Seconds beyond a call's own time limit to wait for the server's answer, before giving up on a
# connection that cannot even say it timed out
CALL_TIMEOUT_MARGIN = 5.0


class MCPToolError(RuntimeError):
    """Raised when an MCP server says that a call to one of its tools failed."""


class MCPServerError(ConnectionError):
    """Raised when an MCP server cannot be started or reached, or stops while a tool runs."""


class UnsetVariableError(ValueError):
    """Raised when a config uses an environment variable that is not set, and gives no default."""


@dataclass(frozen=True)
class MCPServerConfig:
    """How to reach one MCP server, read from a config.

    :param name: The server's name in the config.
    :param prefix: What the names of its tools start with, followed by "_".
    :param transport: "stdio" for a program started here, or "http" or "sse" for a URL.
    :param shown: The command or URL as the config gives it, before variables are replaced, to
        name the server by without showing a secret a variable holds.
    :param command: The program to start, for stdio.
    :param args: Its arguments.
    :param env: Variables it is given, besides the few it gets from this process.
    :param cwd: The directory it starts in, or None for this process's.
    :param url: Where the server is, for http and sse.
    :param headers: Headers sent with every request to it, such as one carrying an API key.
    :param tools: The names of the tools to offer, or None for all of them.
    :param descriptions: Descriptions the config gives some of them, by name, in place of the
        server's.
    :param hidden: The values of the environment variables that replaced ${NAME} in the
        config, each with the ${NAME} it replaced, which messages show in its place.
    :param ask: Patterns, as fnmatch reads them, of the names on the server of the tools that
        wait for a person's approval of each call.
    :param allow: Patterns of the names of those of them that do not after all.
    :param instructions: What the agents are told of using the server's tools, besides what the
        server says itself, or None.
    :param setup: What to do before the server can be connected to, said when it cannot be, as a
        preset says it; or None.
    """

    name: str
    prefix: str
    transport: str
    shown: str
    command: str | None = field(default=None, repr=False)
    args: tuple[str, ...] = field(default=(), repr=False)
    env: dict[str, str] = field(default_factory=dict, repr=False)
    cwd: str | None = None
    url: str | None = field(default=None, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    tools: tuple[str, ...] | None = None
    descriptions: dict[str, str] = field(default_factory=dict)
    hidden: dict[str, str] = field(default_factory=dict, repr=False)
    ask: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()
    instructions: str | None = field(default=None, repr=False)
    setup: str | None = field(default=None, repr=False)

    def needs_approval(self, tool: str) -> bool:
        """Whether each call to a tool, by its name on the server, waits for a person's approval."""
        return any(fnmatchcase(tool, pattern) for pattern in self.ask) and not any(
            fnmatchcase(tool, pattern) for pattern in self.allow
        )

    def redact(self, text: str) -> str:
        """The text with each value of a variable the config used, as it is or as it appears
        in a URL, replaced with the ${NAME} it took the place of."""
        forms: dict[str, str] = {}
        for value, reference in self.hidden.items():
            if len(value) >= MIN_HIDDEN_CHARS:
                for form in (value, quote(value), quote(value, safe=""), quote_plus(value)):
                    forms.setdefault(form, reference)
        # Longest first, so that a value inside another is not replaced in it first
        for form in sorted(forms, key=len, reverse=True):
            text = text.replace(form, forms[form])

        return text


def read_config(config: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """The servers a config lists, by name, each as the config describes it.

    :param config: A path to a YAML or JSON file, or the config itself as a dict. Its servers are
        under mcp_servers or mcpServers, or, where it has neither, it is the servers themselves.
    """
    if isinstance(config, Mapping):
        data: Any = config
        where = "The MCP config"
    elif isinstance(config, str | os.PathLike):
        path = Path(config).expanduser()
        where = f"The MCP config {path}"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"There is no MCP config at {path}") from None
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where} is not valid JSON: {error}") from None
        else:
            import yaml

            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as error:
                raise ValueError(f"{where} is not valid YAML: {error}") from None
            if data is None:
                data = {}
    else:
        raise TypeError(f"An MCP config is a path to a YAML or JSON file, or a dict, not {type(config).__name__}")

    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be a mapping of servers, not {type(data).__name__}")

    present = [key for key in CONFIG_KEYS if key in data]
    if len(present) > 1:
        raise ValueError(f"{where} lists servers under both {' and '.join(present)}: list them under one")
    servers = data[present[0]] if present else data
    if servers is None:
        return {}
    if not isinstance(servers, Mapping):
        raise ValueError(
            f"{where} lists its servers as a mapping of each one's name to how to reach it, not {type(servers).__name__}"
        )

    return dict(servers)


def substitute(text: str, server: str, hidden: dict[str, str] | None = None) -> str:
    """Replaces each ${NAME} with the environment variable NAME, or ${NAME:-default} with its
    default where NAME is not set.

    :param hidden: Where to record each variable's value with the ${NAME} it replaced, for
        messages to show in its place.
    """

    def value(match: re.Match[str]) -> str:
        name, default = match[1], match[2]
        if name in os.environ:
            if hidden is not None:
                hidden.setdefault(os.environ[name], f"${{{name}}}")
            return os.environ[name]
        if default is not None:
            return default
        raise UnsetVariableError(
            f"The MCP server {server} uses ${{{name}}}, which is not set: set it, or give a default as "
            f"${{{name}:-default}}"
        )

    return VARIABLE.sub(value, text)


def scalar(value: Any, server: str, what: str) -> str:
    """A value from a config as the text a command line or an environment variable holds."""
    if isinstance(value, bool):
        # As YAML writes it, which is how a config that says DEBUG: true means it
        return "true" if value else "false"
    if isinstance(value, str | int | float):
        return str(value)

    raise TypeError(f"The {what} of the MCP server {server} must be text or a number, not {type(value).__name__}")


def text_mapping(value: Any, server: str, what: str, hidden: dict[str, str]) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"The {what} of the MCP server {server} is a mapping of names to values, not {type(value).__name__}")
    mapping = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"The {what} of the MCP server {server} are named by text, not {key!r}")
        mapping[key] = substitute(scalar(item, server, f"{what} {key}"), server, hidden)

    return mapping


def tool_selection(value: Any, server: str) -> tuple[tuple[str, ...] | None, dict[str, str]]:
    """The tools a server's entry chooses, and descriptions it gives them.

    Each is named, or, in Biomni's form, given as a mapping with its name under biomni_name or
    name, and its description. The parameters Biomni's form lists are not used: the server
    declares them itself.
    """
    if value is None or value == []:
        return None, {}
    if not isinstance(value, list):
        raise TypeError(f"The tools of the MCP server {server} are a list of their names, not {type(value).__name__}")

    names: list[str] = []
    descriptions: dict[str, str] = {}
    ignored_parameters = False
    for item in value:
        if isinstance(item, Mapping):
            name = item.get("biomni_name", item.get("name"))
            description = item.get("description")
            if description is not None:
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"The description of {name!r}, a tool of the MCP server {server}, must be text")
                descriptions[str(name)] = description.strip()
            ignored_parameters = ignored_parameters or "parameters" in item
        else:
            name = item
        if not isinstance(name, str) or not name:
            raise ValueError(f"The tools of the MCP server {server} are named by text, not {name!r}")
        if name in names:
            raise ValueError(f"The MCP server {server} lists its tool {name} twice")
        names.append(name)

    if ignored_parameters:
        warnings.warn(
            f"The parameters the config lists for the tools of the MCP server {server} are not used: each tool "
            "takes the parameters the server declares for it.",
            UserWarning,
            stacklevel=5,
        )

    return tuple(names), descriptions


def patterns(value: Any, server: str, what: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(
            f"{what} of the MCP server {server} lists the names of tools, or patterns such as list_*, not "
            f"{type(value).__name__}"
        )
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{what} of the MCP server {server} lists tools by name, not {item!r}")

    return tuple(item.strip() for item in value)


def approval_rules(value: Any, server: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tools of a server that wait for a person's approval of each call, and those of them
    that do not after all, each as names or patterns.

    A server's approval is a list of the tools that wait, or a mapping with them under ask and
    those that do not under allow, so that {ask: ["*"], allow: ["list_*"]} has every tool wait but
    those that list.
    """
    if value is None:
        return (), ()
    if isinstance(value, list):
        return patterns(value, server, "The approval"), ()
    if not isinstance(value, Mapping):
        raise TypeError(
            f"The approval of the MCP server {server} is a list of the tools that wait for a person's approval, or "
            f"a mapping with them under ask and those that do not under allow, not {type(value).__name__}"
        )
    if unknown := sorted(str(key) for key in set(value) - APPROVAL_KEYS):
        raise ValueError(
            f"The approval of the MCP server {server} says {', '.join(unknown)}, which it does not take: it takes ask "
            "and allow"
        )

    return (
        patterns(value.get("ask", []), server, "The approval's ask"),
        patterns(value.get("allow", []), server, "The approval's allow"),
    )


def parse_server(name: Any, entry: Any) -> MCPServerConfig | None:
    """How to reach a server, from its entry in a config, or None if the entry disables it.

    An entry that names a preset is the preset's, with what else it gives in place of the
    preset's.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"An MCP server is named by text, not {name!r}")
    if not isinstance(entry, Mapping):
        raise TypeError(
            f"The MCP server {name} is described by a mapping, such as {{'command': ['python', 'server.py']}}, "
            f"not {type(entry).__name__}"
        )
    entry, preset = preset_entry(name, dict(entry))
    try:
        parsed = parse_entry(name, entry)
    except UnsetVariableError as error:
        if preset is None:
            raise
        raise UnsetVariableError(f"{error}. {preset.setup}") from None
    if parsed is None or preset is None:
        return parsed

    return replace(parsed, setup=preset.setup)


def parse_entry(name: str, entry: Mapping[str, Any]) -> MCPServerConfig | None:
    """How to reach a server, from its entry in a config with any preset filled in."""
    for key in ("enabled", "disabled"):
        if key in entry and not isinstance(entry[key], bool):
            raise TypeError(f"{key} of the MCP server {name} must be true or false, not {entry[key]!r}")
    if not entry.get("enabled", True) or entry.get("disabled", False):
        return None

    if unknown := sorted(str(key) for key in set(entry) - SERVER_KEYS):
        warnings.warn(
            f"The MCP server {name} has settings that are not used: {', '.join(unknown)}.",
            UserWarning,
            stacklevel=4,
        )

    prefix = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not prefix.isidentifier():
        raise ValueError(
            f"The tools of the MCP server {name} are named {prefix}_ and theirs, which must start with a letter "
            "or '_': rename the server in the config"
        )

    transport = entry.get("type", entry.get("transport"))
    url = entry.get("url")
    if "httpUrl" in entry:
        if url is not None:
            raise ValueError(f"The MCP server {name} gives both url and httpUrl: give one")
        url = entry["httpUrl"]
        transport = transport or "http"
    command = entry.get("command")
    if command is not None and url is not None:
        raise ValueError(f"The MCP server {name} gives both a command and a URL: give one")
    if transport is None:
        if command is not None:
            transport = "stdio"
        elif isinstance(url, str):
            # A server that only speaks SSE is conventionally found at /sse
            transport = "sse" if urlsplit(url).path.rstrip("/").endswith("/sse") else "http"
        else:
            raise ValueError(f"The MCP server {name} needs a command to start it, or a url to reach it at")
    if not isinstance(transport, str):
        raise TypeError(f"The type of the MCP server {name} is text, not {type(transport).__name__}")
    if transport in STREAMABLE_HTTP_TYPES:
        transport = "http"
    if transport not in ("stdio", "http", "sse"):
        raise ValueError(f'The MCP server {name} has the type {transport!r}: use "stdio", "http", or "sse"')

    tools, descriptions = tool_selection(entry.get("tools"), name)
    ask, allow = approval_rules(entry.get("approval"), name)
    instructions = entry.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise TypeError(f"The instructions of the MCP server {name} are text, not {type(instructions).__name__}")
    notes = (instructions or "").strip() or None
    hidden: dict[str, str] = {}

    if transport == "stdio":
        if command is None:
            raise ValueError(f"The MCP server {name} is started by a command, which it does not give")
        if misplaced := sorted({"headers"} & set(entry)):
            raise ValueError(f"The MCP server {name} is started here, so {misplaced[0]} is not used: remove it")
        if isinstance(command, str):
            parts = [command]
        elif isinstance(command, list) and command:
            parts = [scalar(part, name, "command") for part in command]
        else:
            raise TypeError(f"The command of the MCP server {name} is text, or a list of the program and its arguments")
        args = entry.get("args")
        if args is not None:
            if not isinstance(args, list):
                raise TypeError(f"The args of the MCP server {name} are a list, not {type(args).__name__}")
            parts.extend(scalar(part, name, "args") for part in args)
        shown = " ".join(parts)
        parts = [substitute(part, name, hidden) for part in parts]
        if not parts[0].strip():
            raise ValueError(f"The command of the MCP server {name} is empty")
        cwd = entry.get("cwd")
        if cwd is not None:
            if not isinstance(cwd, str):
                raise TypeError(f"The cwd of the MCP server {name} is a path, not {type(cwd).__name__}")
            cwd = str(Path(substitute(cwd, name, hidden)).expanduser())
            if not Path(cwd).is_dir():
                raise FileNotFoundError(f"The MCP server {name} starts in {entry['cwd']}, which is not a directory")
        env = text_mapping(entry.get("env"), name, "environment variables", hidden)

        return MCPServerConfig(
            name=name,
            prefix=prefix,
            transport="stdio",
            shown=shown,
            command=parts[0],
            args=tuple(parts[1:]),
            env=env,
            cwd=cwd,
            tools=tools,
            descriptions=descriptions,
            hidden=hidden,
            ask=ask,
            allow=allow,
            instructions=notes,
        )

    if not isinstance(url, str):
        raise ValueError(f"The MCP server {name} is reached at a URL, which it does not give")
    if misplaced := sorted({"command", "args", "env", "cwd"} & set(entry)):
        raise ValueError(f"The MCP server {name} is reached at a URL, so {misplaced[0]} is not used: remove it")
    address = substitute(url, name, hidden)
    if urlsplit(address).scheme not in ("http", "https") or not urlsplit(address).netloc:
        raise ValueError(f"The MCP server {name} is reached at {url!r}, which is not an http:// or https:// URL")

    return MCPServerConfig(
        name=name,
        prefix=prefix,
        transport=transport,
        shown=url,
        url=address,
        headers=text_mapping(entry.get("headers"), name, "headers", hidden),
        tools=tools,
        descriptions=descriptions,
        hidden=hidden,
        ask=ask,
        allow=allow,
        instructions=notes,
    )


def tool_name(prefix: str, name: str) -> str:
    """What a server's tool is called here: the server's name, then the tool's, as one name that
    code can call and every provider accepts."""
    full = f"{prefix}_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
    if len(full) <= MAX_TOOL_NAME_CHARS:
        return full

    # Shortened, but kept apart from any other name shortened to the same start
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:8]
    return f"{full[: MAX_TOOL_NAME_CHARS - len(digest) - 1]}_{digest}"


def tool_parameters(schema: Any) -> dict[str, Any]:
    """A tool's input schema as a model is given it: written out in full, without titles."""
    parameters = {key: value for key, value in dict(schema or {}).items() if key != "$schema"}
    parameters = inline_references(clean_schema(parameters))
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})

    return parameters


def wraps_result(tool: Any) -> bool:
    """Whether a tool's structured result is its return value wrapped as {"result": ...}.

    The MCP Python SDK wraps a value that is not an object that way, under a schema with only
    that property, titled after the function with "Output", and FastMCP marks the schema it wraps
    one with.
    """
    schema = getattr(tool, "output_schema", None)
    if not isinstance(schema, dict):
        return False
    if schema.get("x-fastmcp-wrap-result") is True:
        return True
    properties = schema.get("properties")

    title = schema.get("title")

    return (
        isinstance(properties, dict)
        and list(properties) == ["result"]
        and isinstance(title, str)
        and title.endswith("Output")
    )


def decoded_size(data: str) -> str:
    try:
        return f"{len(base64.b64decode(data, validate=True)):,} bytes"
    except (binascii.Error, ValueError):
        return "size unknown"


def content_text(content: Iterable[Any]) -> str:
    """What a tool returned as text, with what cannot be text said in words."""
    parts = []
    for item in content:
        kind = getattr(item, "type", None)
        if kind == "text":
            parts.append(item.text)
        elif kind in ("image", "audio"):
            what = "An image" if kind == "image" else "A recording"
            parts.append(f"[{what} ({item.mime_type}, {decoded_size(item.data)}), which cannot be shown as text]")
        elif kind == "resource_link":
            about = ", ".join(str(part) for part in (item.name, item.description) if part)
            parts.append(f"[A resource at {item.uri}{f': {about}' if about else ''}]")
        elif kind == "resource":
            resource = item.resource
            if isinstance(getattr(resource, "text", None), str):
                parts.append(f"[{resource.uri}]\n{resource.text}")
            else:
                kind_of_file = resource.mime_type or "binary"
                size = decoded_size(getattr(resource, "blob", ""))
                parts.append(f"[A file at {resource.uri} ({kind_of_file}, {size}), which cannot be shown as text]")
        else:
            parts.append(f"[Content of the type {kind}, which cannot be shown as text]")

    return "\n".join(parts)


def tool_result(result: Any, name: str, wrapped: bool) -> Any:
    """What a call to a tool returned, as code is given it.

    :param result: The result of the call.
    :param name: The tool's name on its server.
    :param wrapped: Whether the tool's structured result wraps its value as {"result": ...}.
    :return: The structured result, where there is one and the rest is text, since otherwise it
        would leave out an image or a file the text describes; or else the text.
    :raises MCPToolError: If the result says the call failed.
    """
    if result.is_error:
        raise MCPToolError(content_text(result.content) or f"{name} failed without saying why")
    if result.structured_content is not None and all(getattr(item, "type", None) == "text" for item in result.content):
        value = result.structured_content
        if wrapped and isinstance(value, dict) and list(value) == ["result"]:
            return value["result"]
        return value

    return content_text(result.content)


def failure_text(error: BaseException) -> str:
    """What went wrong, from the errors inside an exception group as well as the group itself."""
    leaves: list[BaseException] = []

    def collect(item: BaseException) -> None:
        if isinstance(item, BaseExceptionGroup):
            for inner in item.exceptions:
                collect(inner)
        else:
            leaves.append(item)

    collect(error)
    shown = dict.fromkeys(f"{type(leaf).__name__}: {leaf}".rstrip(": ") for leaf in leaves)

    return "; ".join(shown)


def connection_ended(client: Any) -> bool:
    """Whether the SDK has found that a connection ended, as it does when a server started here
    exits while no call is waiting on it. The task that holds the connection open is not told,
    so without this the next call would be sent to the server that is gone, and fail.

    The SDK says so only in its dispatcher's private state. Where that is not found, the
    connection is taken to be open, and a call on one that has ended fails as it would have.
    """
    try:
        dispatcher = client.session._dispatcher
    except Exception:
        return False

    return getattr(dispatcher, "_closed", False) is True


class EventLoop:
    """An event loop in a thread of its own, which the servers' connections live on, so that a
    tool is called the same way from any thread, inside an event loop or not."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="virtual_lab MCP", daemon=True)
        self.thread.start()

    def submit(self, coroutine: Any) -> concurrent.futures.Future[Any]:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def stop(self) -> None:
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(MCP_STOP_TIMEOUT)
        if not self.thread.is_alive():
            self.loop.close()


class Holding:
    """One attempt at a connection, which a task holds open until told to stop.

    It is ended by its stop event, or else by cancelling its anyio cancel scope. Cancelling the
    task itself could cut short the shielded shutdown in which the SDK stops a server it started,
    and leave the server running.
    """

    def __init__(self) -> None:
        self.ready: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self.stop = asyncio.Event()
        self.scope: Any = None
        self.cancelled = False
        self.task: concurrent.futures.Future[Any] | None = None

    def cancel(self) -> None:
        """Cancels the attempt. Called on the event loop."""
        self.cancelled = True
        if self.scope is not None:
            self.scope.cancel()


class MCPConnection:
    """A connection to one MCP server, made again if the server stops."""

    # How the version of the protocol is agreed, as the SDK's Client takes it: "auto" speaks the
    # newest the server does, and "legacy" the one from before 2026, under which a server asks
    # its questions with requests of its own rather than in a tool's result
    protocol_mode = "auto"

    def __init__(
        self,
        server: MCPServerConfig,
        loop: EventLoop,
        log_dir: Path,
        timeout: float | None,
        start_timeout: float,
        approve: Approve = approve_in_terminal,
        answer: Answer = answer_in_terminal,
    ) -> None:
        self.server = server
        self.loop = loop
        self.approve = approve
        self.answer = answer
        self.log_path = log_dir / f"{server.prefix}.log" if server.transport == "stdio" else None
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.lock = threading.Lock()
        self.client: Any = None
        self.holding: Holding | None = None
        self.generation = 0
        self.stopped = False
        self.closed = False
        self.listed: list[Any] = []
        self.wrapped: dict[str, bool] = {}
        # What the server says of how to use its tools, when it is connected to
        self.instructions: str | None = None
        # Each tool offered, by its name on the server, with its name and description here
        self.offered: dict[str, tuple[str, str]] = {}

    @property
    def described(self) -> str:
        how = "started with" if self.server.transport == "stdio" else "at"
        return f"the MCP server {self.server.name} ({how} {self.server.shown})"

    def log_tail(self) -> str:
        """The end of what the server wrote to stderr, to say why it failed."""
        if self.log_path is None:
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        if not text:
            return " It wrote nothing to stderr."
        # Before it is cut short, so that no part of a secret is left at the start
        text = self.server.redact(text)
        if len(text) > MCP_LOG_TAIL_CHARS:
            text = "..." + text[-MCP_LOG_TAIL_CHARS:]

        return f" What it wrote to stderr ends:\n{text}"

    def setup_hint(self) -> str:
        return f"\n{self.server.setup}" if self.server.setup else ""

    async def elicited(self, context: Any, params: Any) -> Any:
        """Passes on to a person what the server asks while one of its tools runs, and gives
        back their answer, or that they declined, which is also the answer where no one is
        asked."""
        import anyio
        from mcp_types import ElicitResult

        if getattr(params, "mode", "form") == "url":
            question = ServerQuestion(server=self.server.name, message=params.message, url=params.url)
        else:
            schema = params.requested_schema if isinstance(params.requested_schema, dict) else {}
            properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            required = schema.get("required") if isinstance(schema.get("required"), list) else []
            question = ServerQuestion(
                server=self.server.name,
                message=params.message,
                fields={str(name): dict(item) if isinstance(item, dict) else {} for name, item in properties.items()},
                required=tuple(str(name) for name in required),
            )
        finished = threading.Event()

        def ask() -> dict[str, Any] | None:
            with asked_for(finished):
                return self.answer(question)

        try:
            # In a thread, since a person may take a while, and the connections to every server
            # are served on this loop meanwhile. The thread is left behind if the call is
            # cancelled, and the question is withdrawn, so that the answer is not taken for
            # one to whatever is asked next
            answer = await anyio.to_thread.run_sync(ask, abandon_on_cancel=True)
        except Exception as error:
            warnings.warn(
                f"The question the MCP server {self.server.name} asked could not be answered, so it was declined: "
                f"{failure_text(error)}",
                UserWarning,
                stacklevel=1,
            )
            return ElicitResult(action="decline")
        finally:
            finished.set()
        if answer is None:
            return ElicitResult(action="decline")
        if question.url is not None:
            return ElicitResult(action="accept")

        return ElicitResult(action="accept", content=dict(answer))

    async def transport(self, stack: AsyncExitStack, log: Any) -> Any:
        server = self.server
        read_timeout = None if self.timeout is None else max(300.0, self.timeout)
        if server.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            parameters = StdioServerParameters(
                command=server.command, args=list(server.args), env=dict(server.env), cwd=server.cwd
            )
            return stdio_client(parameters, errlog=log)
        if server.transport == "sse":
            from mcp.client.sse import sse_client

            return sse_client(server.url, headers=dict(server.headers), sse_read_timeout=read_timeout)

        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        http = await stack.enter_async_context(
            httpx2.AsyncClient(headers=dict(server.headers), timeout=httpx2.Timeout(30.0, read=read_timeout))
        )
        return streamable_http_client(server.url, http_client=http)

    async def hold(self, holding: Holding, log: Any) -> None:
        """Keeps the connection open until told to stop. A connection is opened and closed in
        the same task, as the SDK requires."""
        import anyio
        from mcp import Client, Implementation

        try:
            with anyio.CancelScope() as scope:
                holding.scope = scope
                # Cancelled before it began, so nothing is started that would have to be stopped
                if holding.cancelled:
                    return
                async with AsyncExitStack() as stack:
                    client = await stack.enter_async_context(
                        Client(
                            await self.transport(stack, log),
                            client_info=Implementation(name="virtual-lab", version=__version__),
                            elicitation_callback=self.elicited,
                            mode=self.protocol_mode,
                        )
                    )
                    if not holding.ready.done():
                        holding.ready.set_result(client)
                    await holding.stop.wait()
        except Exception as error:
            if not holding.ready.done():
                holding.ready.set_exception(error)
        finally:
            if log is not None:
                log.close()
            if not holding.ready.done():
                holding.ready.set_exception(ConnectionError("The connection was closed before it was made"))

    async def list_tools(self, client: Any) -> list[Any]:
        tools: list[Any] = []
        cursor = None
        seen: set[str] = set()
        for _ in range(MCP_MAX_TOOL_PAGES):
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools
            if cursor in seen:
                raise MCPServerError(f"{self.described} lists its tools in pages that repeat")
            seen.add(cursor)

        raise MCPServerError(f"{self.described} lists its tools in more than {MCP_MAX_TOOL_PAGES:,} pages")

    def start(self) -> None:
        """Starts the server, or connects to it, and lists its tools."""
        log = open(self.log_path, "a", encoding="utf-8") if self.log_path is not None else None  # noqa: SIM115
        holding = Holding()
        holding.task = self.loop.submit(self.hold(holding, log))
        starting = "start" if self.server.transport == "stdio" else "connect to"
        try:
            client = holding.ready.result(self.start_timeout)
            listing = self.loop.submit(self.list_tools(client))
            try:
                tools = listing.result(self.start_timeout)
            except concurrent.futures.TimeoutError:
                listing.cancel()
                raise
        except concurrent.futures.TimeoutError:
            self.end(holding, graceful=False)
            if log is not None:
                log.close()
            raise MCPServerError(
                f"Could not {starting} {self.described}: it did not answer within {self.start_timeout:g} seconds."
                f"{self.log_tail()}{self.setup_hint()}"
            ) from None
        except Exception as error:
            self.end(holding, graceful=False)
            if log is not None:
                log.close()
            if isinstance(error, MCPServerError):
                raise
            raise MCPServerError(
                f"Could not {starting} {self.described}: {self.server.redact(failure_text(error))}.{self.log_tail()}"
                f"{self.setup_hint()}"
            ) from None

        self.client, self.holding = client, holding
        self.generation += 1
        self.stopped = False
        self.listed = tools
        self.wrapped = {tool.name: wraps_result(tool) for tool in tools}
        instructions = getattr(client, "instructions", None)
        self.instructions = (instructions.strip() or None) if isinstance(instructions, str) else None

    def end(self, holding: Holding | None, graceful: bool = True) -> None:
        """Closes a connection, and stops the server if it was started here: by asking, where
        graceful, and by cancelling it where that is not, or it does not end in time."""
        if holding is None or holding.task is None or self.loop.loop.is_closed():
            return
        if graceful:
            self.loop.loop.call_soon_threadsafe(holding.stop.set)
            try:
                holding.task.result(MCP_STOP_TIMEOUT)
                return
            except concurrent.futures.TimeoutError:
                pass
            except Exception:
                return
        self.loop.loop.call_soon_threadsafe(holding.cancel)
        try:
            holding.task.result(MCP_STOP_TIMEOUT)
        except Exception:  # noqa: S110
            # Ended or not, there is nothing more to do: a server that outlives this is stopped
            # when this process exits, as its stdin closes
            pass

    def connected(self) -> tuple[Any, int]:
        """The connection to call a tool on, made again first if the server has stopped."""
        with self.lock:
            if self.closed:
                raise RuntimeError(f"The MCP server {self.server.name} was closed: connect to it again with connect_mcp")
            if (
                self.client is None
                or self.stopped
                or (self.holding is not None and self.holding.task is not None and self.holding.task.done())
                or connection_ended(self.client)
            ):
                if self.server.transport == "stdio":
                    warnings.warn(
                        f"The MCP server {self.server.name} stopped, so it is started again: whatever it held from "
                        "the calls before, such as what it had loaded, is gone.",
                        UserWarning,
                        stacklevel=4,
                    )
                else:
                    warnings.warn(
                        f"The connection to the MCP server {self.server.name} was lost, so it is made again: "
                        "whatever the server held for the connection before, such as a session, may be gone.",
                        UserWarning,
                        stacklevel=4,
                    )
                self.end(self.holding, graceful=False)
                self.client = self.holding = None
                self.start()

            return self.client, self.generation

    def call(self, name: str, /, **arguments: Any) -> Any:
        """Calls one of the server's tools, by its name on the server.

        :return: The tool's structured result, where it gives one and nothing else, or else what
            it returned as text.
        :raises MCPToolError: If the server says the call failed.
        :raises MCPServerError: If the server stopped during the call. It is started again, or
            connected to again, at the next call to one of its tools.
        :raises TimeoutError: If the tool did not answer within the time limit.
        """
        import anyio
        from mcp import MCPError
        from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

        if self.server.needs_approval(name):
            self.approved(name, arguments)
        client, generation = self.connected()
        call = self.loop.submit(client.call_tool(name, arguments, read_timeout_seconds=self.timeout))
        too_slow = TimeoutError(f"The MCP server {self.server.name} did not answer within {self.timeout} seconds")
        try:
            result = call.result(None if self.timeout is None else self.timeout + CALL_TIMEOUT_MARGIN)
        except concurrent.futures.TimeoutError:
            call.cancel()
            raise too_slow from None
        except MCPError as error:
            if error.code == REQUEST_TIMEOUT:
                raise too_slow from None
            if error.code != CONNECTION_CLOSED:
                raise MCPToolError(
                    f"The MCP server {self.server.name} refused the call: {self.server.redact(error.message)}"
                ) from None
            self.lost(generation)
            raise self.stopped_error() from None
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream):
            self.lost(generation)
            raise self.stopped_error() from None

        try:
            return tool_result(result, name, self.wrapped.get(name, False))
        except MCPToolError as error:
            # A server's error can quote what it was given, such as a URL with its API key
            raise MCPToolError(self.server.redact(str(error))) from None

    def approved(self, name: str, arguments: dict[str, Any]) -> None:
        """Asks a person whether a call may be made.

        :raises ApprovalDeclined: If it may not.
        """
        here, description = self.offered.get(name, (tool_name(self.server.prefix, name), ""))
        request = ApprovalRequest(
            tool=here, server=self.server.name, server_tool=name, arguments=dict(arguments), description=description
        )
        late = ApprovalDeclined(
            f"The code that called {here} stopped waiting for a person to approve the call, so it was not made."
        )
        if withdrawn():
            raise late
        # Only True approves, so that an approver that answers anything else, such as "no", does not
        approval = self.approve(request)
        # An approval that comes once the caller has gone would make a call whose result no one
        # receives, and that the caller may well make again
        if withdrawn():
            raise late
        if approval is not True:
            raise ApprovalDeclined(
                f"A person declined this call to {here}, so it was not made. Do not make it again unchanged."
            )

    def lost(self, generation: int) -> None:
        with self.lock:
            # A call that failed on a connection made before the current one says nothing of it
            if generation == self.generation:
                self.stopped = True

    def stopped_error(self) -> MCPServerError:
        again = "started again" if self.server.transport == "stdio" else "connected to again"
        return MCPServerError(
            f"The MCP server {self.server.name} stopped before it answered. It is {again} at the next call to "
            f"one of its tools.{self.log_tail()}"
        )

    def begin_closing(self) -> None:
        """Refuses calls from now on, and asks the server to stop."""
        with self.lock:
            self.closed = True
            if self.holding is not None and not self.loop.loop.is_closed():
                self.loop.loop.call_soon_threadsafe(self.holding.stop.set)

    def close(self) -> None:
        self.begin_closing()
        with self.lock:
            self.end(self.holding)
            self.client = self.holding = None


class MCPTools:
    """The tools of the MCP servers that connect_mcp started or connected to, which stay
    running until closed. Close them when done, or use them in a with statement; any still
    open are closed when Python exits.

    :param tools: Every server's tools, to give a meeting or a session.
    :param servers: Each server's tools, by the server's name in the config.
    """

    def __init__(
        self,
        connections: list[MCPConnection],
        tools: dict[str, tuple[Tool, ...]],
        loop: EventLoop | None,
        log_dir: Path | None,
    ) -> None:
        self.connections = connections
        self.servers = tools
        self.tools: tuple[Tool, ...] = tuple(tool for server in tools.values() for tool in server)
        self.loop = loop
        self.log_dir = log_dir
        self.closed = False
        atexit.register(self.close)

    def close(self) -> None:
        """Stops the servers started here, and closes the connections to the others."""
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self.close)
        try:
            # Every server is asked to stop before any is waited for, so that they stop together,
            # without threads, which cannot be started once Python has begun to exit
            for connection in self.connections:
                connection.begin_closing()
            for connection in self.connections:
                connection.close()
        finally:
            if self.loop is not None:
                self.loop.stop()
            if self.log_dir is not None:
                shutil.rmtree(self.log_dir, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def __repr__(self) -> str:
        servers = ", ".join(f"{name}: {len(tools)} tools" for name, tools in self.servers.items())
        return f"MCPTools({servers}{', closed' if self.closed else ''})"


def check_timeout(name: str, value: float | None, optional: bool) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not (math.isfinite(value) and value > 0):
        allowed = "a number of seconds above zero" + (", or None for no limit" if optional else "")
        raise ValueError(f"{name} must be {allowed}, not {value!r}")


def server_instructions(connection: MCPConnection) -> str | None:
    """What the agents are told of using a server's tools: what its config says, and what the
    server says itself, or None if neither says anything."""
    server = connection.server
    parts = [server.instructions] if server.instructions else []
    if connection.instructions:
        said = truncate_text(server.redact(connection.instructions), MCP_MAX_INSTRUCTIONS_CHARS)
        parts.append(f"The server says of them:\n{said}")
    if not parts:
        return None

    heading = f"The tools of the MCP server {server.name}, named {server.prefix}_ and then their names on the server:"

    return heading + "\n" + "\n\n".join(parts)


def unused_approvals(server: MCPServerConfig, listed: Iterable[str]) -> list[str]:
    """The tools a server's approval names, not as patterns, that the server does not have, which
    is likely a mistake, since a call to the tool meant would then not wait."""
    names = set(listed)

    return [pattern for pattern in server.ask if not any(mark in pattern for mark in "*?[") and pattern not in names]


def connect_mcp(
    config: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    servers: Iterable[str] | None = None,
    timeout: float | None = MCP_CALL_TIMEOUT,
    start_timeout: float = MCP_START_TIMEOUT,
    presets: Iterable[str] | None = None,
    approve: Approve | None = None,
    answer: Answer | None = None,
) -> MCPTools:
    """Starts, or connects to, the MCP servers a config lists, and makes tools of their tools,
    for a meeting's agents to call, or for code in a session to call, or both, as Biomni's
    add_mcp does.

    A config is a YAML or JSON file, or a dict, listing servers by name, under mcp_servers as
    Biomni's does or under mcpServers as Claude Desktop's does. A server is started here from
    its command, a list of the program and its arguments, or a program with its args, and is
    given env, its environment variables, besides the few it takes from this process, such as
    PATH and HOME, and it starts in cwd. Or it is reached at its url, over streamable HTTP, or
    over SSE where its type says "sse" or the URL ends in /sse, with headers sent with every
    request. ${NAME} anywhere in these is replaced with the environment variable NAME, and
    ${NAME:-default} with the default where NAME is not set; errors show ${NAME} in place of
    its value. A server that says enabled: false,
    or disabled: true, is left out. tools lists the names of the tools to offer, where not all of
    them; an entry in Biomni's form, with biomni_name and a description, gives the tool that
    description in place of the server's.

        mcp_servers:
          genes:
            command: ["python", "-m", "gene_server"]
            env: {GENE_API_KEY: "${GENE_API_KEY}"}
          search:
            url: "https://search.example.org/mcp"
            headers: {Authorization: "Bearer ${SEARCH_TOKEN}"}

    Each tool is named after its server and itself, joined by "_", so that two servers' tools
    never share a name, with anything but letters, digits, and "_" made "_": the tool lookup of
    the server genes is genes_lookup. Its parameters are the ones the server declares, and the
    server checks the arguments. What it returns is its structured result where it gives one,
    such as a dict, and otherwise its text, with any image, recording, or file in it described;
    a model is shown either as text. A tool the server says failed raises MCPToolError, which a
    meeting's agent is told of as an error, as is a server that stops, which is started again
    at the next call to one of its tools, with a warning, since anything it held is gone.

    A server is kept running, and connected to, until the tools are closed, and calls to it can
    run at the same time. What a server started here writes to stderr is kept out of the way,
    and its end is shown if the server fails.

    A server's approval lists the tools, by their names on the server, or by patterns such as
    create_*, that wait for a person's approval of each call: approval: [submit, pay], or
    approval: {ask: ["*"], allow: ["list_*", "get_*"]} for every tool to wait but those that only
    read, which also holds for any tool the server adds later. A call that is not approved is not
    made, and fails with ApprovalDeclined, which the agent is told of. A server's instructions,
    and what the server says itself of how to use its tools, are told to the agents of a meeting
    given its tools, or a session given them. What a server asks a person while one of its tools
    runs, as Proto's server asks before it deploys a tool, is passed on to a person to answer.

    A preset is a server's entry written out already, for Paperclip's literature, Adaptyv's lab,
    or Proto's design tools: {"lab": {"preset": "adaptyv"}} in a config, with anything else the
    entry gives used in place of the preset's, or presets=["adaptyv"] for a server named after
    its preset. MCP_PRESETS lists them, with what each needs, such as an API key.

        with connect_mcp(presets=["paperclip", "proto"]) as mcp:
            run_meeting(..., tools=mcp.tools)

    :param config: The config: a path to a YAML or JSON file, or the config as a dict; or None
        for presets alone.
    :param servers: The names of the servers to connect to, or None for every one the config
        enables, and the presets.
    :param timeout: The most seconds a call to a tool may take, or None for no limit.
    :param start_timeout: The most seconds a server may take to start, or be connected to, and
        list its tools. A server run with npx or docker may first have to be downloaded.
    :param presets: Presets to connect to as well, each as a server named after it.
    :param approve: Called with an ApprovalRequest before each call to a tool that waits for
        approval, and makes the call only if it returns True; it may raise ApprovalDeclined to
        say why not. None asks at the terminal, and declines every call where there is none. A
        call from code in a session that stops before the call is approved is not made.
    :param answer: Called with a ServerQuestion when a server asks a person something, and returns
        the answer: the form's fields by name, {} once the person has gone to a web page the
        question asks them to, or None to decline. None asks at the terminal, and declines
        where there is none. A question whose call stops waiting before it is answered is
        withdrawn, and asking at the terminal then declines it.
    :raises ImportError: If the MCP SDK is not installed: pip install "virtual-lab[mcp]".
    :raises ValueError: If the config is not one, or uses a variable that is not set, or names
        a preset that is not one, or two tools would share a name.
    :raises MCPServerError: If a server cannot be started or reached. Those that could are
        stopped again.
    :return: The tools, which hold the servers open until they are closed.
    """
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise ImportError(
            'Connecting to MCP servers needs the MCP SDK, which installs with: pip install "virtual-lab[mcp]"'
        ) from None

    check_timeout("timeout", timeout, optional=True)
    check_timeout("start_timeout", start_timeout, optional=False)
    for what, value in (("approve", approve), ("answer", answer)):
        if value is not None and not callable(value):
            raise TypeError(f"{what} is a function, or None to ask at the terminal, not {type(value).__name__}")

    if config is None and presets is None:
        raise ValueError("connect_mcp connects to the servers of a config, or to presets, or both: give one")
    entries = read_config(config) if config is not None else {}
    for preset in [presets] if isinstance(presets, str) else list(presets or ()):
        if preset not in MCP_PRESETS:
            raise ValueError(f"There is no MCP preset {preset!r}: the presets are {', '.join(MCP_PRESETS)}")
        if preset in entries:
            raise ValueError(
                f"The MCP config already has a server {preset}: give it preset: {preset} there, or name it otherwise"
            )
        entries[preset] = {"preset": preset}
    if servers is None:
        chosen = list(entries)
    else:
        chosen = [servers] if isinstance(servers, str) else list(servers)
        if unknown := [name for name in chosen if name not in entries]:
            raise ValueError(
                f"The MCP config has no server {', '.join(map(str, unknown))}: it has {', '.join(map(str, entries)) or 'none'}"
            )
    configs = []
    for name in dict.fromkeys(chosen):
        parsed = parse_server(name, entries[name])
        if parsed is None and servers is not None:
            raise ValueError(f"The MCP server {name} is disabled in the config: enable it to connect to it")
        if parsed is not None:
            configs.append(parsed)

    prefixes: dict[str, str] = {}
    for parsed in configs:
        if parsed.prefix in prefixes:
            raise ValueError(
                f"The MCP servers {prefixes[parsed.prefix]} and {parsed.name} would both name their tools "
                f"{parsed.prefix}_...: rename one in the config"
            )
        prefixes[parsed.prefix] = parsed.name

    if not configs:
        warnings.warn("The MCP config enables no servers, so there are no tools from them.", UserWarning, stacklevel=2)
        return MCPTools([], {}, None, None)

    loop = EventLoop()
    log_dir = Path(tempfile.mkdtemp(prefix="virtual_lab_mcp_"))
    connections = [
        MCPConnection(
            parsed,
            loop,
            log_dir,
            timeout,
            start_timeout,
            approve=approve if approve is not None else approve_in_terminal,
            answer=answer if answer is not None else answer_in_terminal,
        )
        for parsed in configs
    ]
    opened = MCPTools(connections, {}, loop, log_dir)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(connections)) as pool:
            starts = [pool.submit(connection.start) for connection in connections]
        failures = [start.exception() for start in starts if start.exception() is not None]
        if failures:
            if len(failures) == 1:
                raise failures[0]
            raise MCPServerError("\n\n".join(str(failure) for failure in failures))

        by_server: dict[str, tuple[Tool, ...]] = {}
        names: dict[str, str] = {}
        for connection in connections:
            parsed = connection.server
            offered = {tool.name: tool for tool in connection.listed}
            if parsed.tools is not None:
                if missing := [name for name in parsed.tools if name not in offered]:
                    raise ValueError(
                        f"The MCP server {parsed.name} has no tool {', '.join(missing)}: it has "
                        f"{', '.join(offered) or 'none'}"
                    )
                offered = {name: offered[name] for name in parsed.tools}
            if unused := unused_approvals(parsed, (tool.name for tool in connection.listed)):
                warnings.warn(
                    f"The approval of the MCP server {parsed.name} names {', '.join(unused)}, which it has no tool "
                    f"called, so no call waits for approval on {'its' if len(unused) == 1 else 'their'} account. "
                    "To have every tool wait but those allowed, ask for \"*\".",
                    UserWarning,
                    stacklevel=2,
                )
            instructions = server_instructions(connection)
            made = []
            for tool in offered.values():
                name = tool_name(parsed.prefix, tool.name)
                if name in names:
                    raise ValueError(
                        f"The tools {names[name]} and {tool.name} of the MCP server {parsed.name} would both be "
                        f"called {name}: choose one with the server's tools in the config"
                    )
                names[name] = tool.name
                description = (
                    parsed.descriptions.get(tool.name)
                    or inspect.cleandoc(tool.description or "")
                    or (getattr(tool, "title", None) or "").strip()
                    or f"The tool {tool.name} of the MCP server {parsed.name}."
                )
                connection.offered[tool.name] = (name, description)
                if parsed.needs_approval(tool.name):
                    description += "\n\nEach call waits for a person to approve it, and fails if they do not."
                made.append(
                    Tool(
                        name=name,
                        description=description,
                        parameters=tool_parameters(tool.input_schema),
                        function=partial(connection.call, tool.name),
                        instructions=instructions,
                    )
                )
            by_server[parsed.name] = tuple(made)
    except BaseException:
        opened.close()
        raise

    opened.servers = by_server
    opened.tools = tuple(tool for server in by_server.values() for tool in server)

    return opened
