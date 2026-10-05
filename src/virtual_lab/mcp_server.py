"""Tools served over MCP, for Claude Desktop, Claude Code, Cursor, another agent, or connect_mcp.

Biomni's create_mcp_server serves its tool functions from the server's own process, passing each
parameter whose type it does not recognise as a string, and reports a function that fails as if
it had returned its error. Here any tools can be served: Biomni's functions, each run in a
session in the sandbox, with typed parameters (see biomni_session_tools); the tool that runs code
in that session, which keeps its variables from one call to the next; virtual_lab's own database
and literature tools; and tools of your own. Arguments are checked against each tool's
parameters before it is called, what a tool returns is sent back whole, as the structured result
where it is a dict, and a tool that fails says so.

A server is reached over stdio, started by the client, or over streamable HTTP, at
http://127.0.0.1:8000/mcp by default. Over HTTP it answers only this machine unless told to listen
elsewhere, which it does only with a token that every request must carry.

Run it with virtual-lab-mcp, or python -m virtual_lab.mcp_server; see main.
"""

import argparse
import contextlib
import hmac
import importlib
import ipaddress
import json
import os
import signal
import socket
import sys
import threading
import warnings
from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from virtual_lab.__about__ import __version__
from virtual_lab.constants import (
    DEFAULT_EXECUTION_TIMEOUT,
    MAX_TOOL_OUTPUT_CHARS,
    MCP_SERVER_HOST,
    MCP_SERVER_MIN_TOKEN_CHARS,
    MCP_SERVER_PATH,
    MCP_SERVER_PORT,
    MCP_SERVER_TOKEN_VARIABLE,
)
from virtual_lab.execution import ExecutionError
from virtual_lab.records import truncate_text
from virtual_lab.session import CODE_TOOL_NAME, Session, jsonable, session_notes, session_tool
from virtual_lab.tools import TOOL_REGISTRY, Tool, tool_output_text

# What the name of a tool may be, as the MCP specification allows it
MAX_SERVED_NAME_CHARS = 128
NAME_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.")

DEFAULT_SERVER_NAME = "virtual-lab"

# The most problems with a call's arguments that are listed
MAX_ARGUMENT_PROBLEMS = 10

TRANSPORTS = {"stdio": "stdio", "http": "http", "streamable-http": "http", "streamable_http": "http"}


def require_mcp() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise ImportError(
            'Serving tools over MCP needs the MCP SDK, which installs with: pip install "virtual-lab[mcp]"'
        ) from None


def check_served_tools(tools: Iterable[Tool]) -> tuple[Tool, ...]:
    """Refuses what is not a tool, a name MCP does not allow, and two tools of one name."""
    checked = tuple(tools)
    for tool in checked:
        if not isinstance(tool, Tool):
            raise TypeError(
                f"A server serves Tools, not {type(tool).__name__}: make one of a function with tool_from_function"
            )
        if not (0 < len(tool.name) <= MAX_SERVED_NAME_CHARS and set(tool.name) <= NAME_CHARACTERS):
            raise ValueError(
                f"MCP cannot serve a tool called {tool.name!r}: use letters, digits, '_', '-', and '.', "
                f"at most {MAX_SERVED_NAME_CHARS} of them"
            )

    names = [tool.name for tool in checked]
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"A server's tools need names of their own: {', '.join(repeated)} is given twice")

    return checked


def argument_validator(tool: Tool) -> Any:
    """Checks a call's arguments against the tool's parameters, or None if its schema is not
    one that can check them, in which case the tool checks them itself."""
    from jsonschema import SchemaError, validators

    try:
        validator = validators.validator_for(tool.parameters)
        validator.check_schema(tool.parameters)
    except (SchemaError, TypeError, ValueError):
        return None

    return validator(tool.parameters)


def argument_problems(validator: Any, arguments: dict[str, Any]) -> list[str]:
    """What is wrong with the arguments, one problem to an item, each saying where."""
    problems = []
    for error in sorted(validator.iter_errors(arguments), key=lambda error: list(map(str, error.path))):
        where = ".".join(str(part) for part in error.path)
        problems.append(f"{where}: {error.message}" if where else error.message)

    if len(problems) > MAX_ARGUMENT_PROBLEMS:
        problems[MAX_ARGUMENT_PROBLEMS:] = [f"and {len(problems) - MAX_ARGUMENT_PROBLEMS:,} more"]

    return problems


def tool_result(value: Any) -> Any:
    """What a tool returned, as MCP sends it: as text, and as the structured result if a dict."""
    import mcp_types

    # A pydantic model is written out as a dict
    structured = jsonable(value)

    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=tool_output_text(value))],
        structured_content=structured if isinstance(structured, dict) else None,
    )


def error_result(text: str) -> Any:
    import mcp_types

    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=truncate_text(text, MAX_TOOL_OUTPUT_CHARS))],
        is_error=True,
    )


def create_mcp_server(tools: Iterable[Tool], name: str = DEFAULT_SERVER_NAME, instructions: str | None = None) -> Any:
    """Makes an MCP server of tools.

    Each tool is listed with its name, description, and parameters. A call's arguments are
    checked against its parameters, and a call whose arguments do not fit is refused, with what
    is wrong, rather than run. The tool runs in a thread of its own, so calls can run at the same
    time, and what it returns is sent back as text, a dict or list as JSON, and a dict as the
    structured result too. A tool that raises is reported as having failed, with its error.

    :param tools: The tools.
    :param name: The server's name, as clients are told it.
    :param instructions: What clients are told of the server and how to use its tools.
    :raises ImportError: If the MCP SDK is not installed: pip install "virtual-lab[mcp]".
    :raises ValueError: If a tool's name is not one MCP allows, or two tools share a name.
    :return: The server, a low-level server of the MCP SDK, to serve with serve_mcp.
    """
    require_mcp()
    import anyio
    import mcp_types
    from mcp import MCPError
    from mcp.server.lowlevel import Server

    served = check_served_tools(tools)
    by_name = {tool.name: tool for tool in served}
    validators = {tool.name: argument_validator(tool) for tool in served}
    listed = [
        mcp_types.Tool(name=tool.name, description=tool.description, input_schema=tool.parameters)
        for tool in served
    ]

    async def list_tools(context: Any, params: Any) -> Any:
        return mcp_types.ListToolsResult(tools=listed)

    async def call_tool(context: Any, params: Any) -> Any:
        tool = by_name.get(params.name)
        if tool is None:
            raise MCPError(
                code=mcp_types.INVALID_PARAMS,
                message=f"Unknown tool: {truncate_text(str(params.name), MAX_SERVED_NAME_CHARS)}",
            )

        arguments = params.arguments or {}
        validator = validators[tool.name]
        if validator is not None and (problems := argument_problems(validator, arguments)):
            return error_result(f"Invalid arguments for {tool.name}: " + "; ".join(problems))

        try:
            value = await anyio.to_thread.run_sync(partial(tool.function, **arguments), abandon_on_cancel=True)
        except anyio.get_cancelled_exc_class():
            raise
        except BaseException as error:
            return error_result(f"{type(error).__name__}: {error}")

        return tool_result(value)

    return Server(
        name,
        version=__version__,
        instructions=instructions,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def is_loopback(host: str) -> bool:
    """Whether a host to listen at is reachable from this machine alone."""
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_token(token: str | None, host: str) -> None:
    if token is None:
        if not is_loopback(host):
            raise ValueError(
                f"Listening at {host} lets other machines call these tools, and run code through them, so "
                f"every request must carry a token: set {MCP_SERVER_TOKEN_VARIABLE}, to one made with "
                "python -c \"import secrets; print(secrets.token_urlsafe(32))\", and have clients send it "
                "as the header Authorization: Bearer <token>"
            )
        return

    if not isinstance(token, str):
        raise TypeError(f"A token is a str, not {type(token).__name__}")
    if len(token) < MCP_SERVER_MIN_TOKEN_CHARS:
        raise ValueError(
            f"A token must be at least {MCP_SERVER_MIN_TOKEN_CHARS} characters, to be hard to guess: make "
            "one with python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    if not (token.isascii() and token.isprintable()) or any(character.isspace() for character in token):
        raise ValueError("A token is sent in a header, so it must be printable ASCII without spaces")


class RequireToken:
    """Lets through only the HTTP requests that carry the token, as Authorization: Bearer."""

    def __init__(self, app: Callable[..., Any], token: str) -> None:
        self.app = app
        self.token = token.encode("ascii")

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope["type"] == "http" and not self.authorized(scope):
            body = json.dumps({"error": "unauthorized", "error_description": "A valid bearer token is required"})
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b'Bearer error="invalid_token"'),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body.encode()})
            return

        await self.app(scope, receive, send)

    def authorized(self, scope: dict[str, Any]) -> bool:
        given = [value for key, value in scope.get("headers", []) if key.lower() == b"authorization"]
        if len(given) != 1:
            return False
        scheme, _, credentials = given[0].strip().partition(b" ")

        return scheme.lower() == b"bearer" and hmac.compare_digest(credentials.strip(), self.token)


def listening_socket(host: str, port: int) -> socket.socket:
    """A socket listening at the host and port, which is chosen if 0."""
    try:
        family = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)[0][0]
        return socket.create_server((host, port), family=family)
    except OSError as error:
        raise OSError(f"Cannot listen at {host}, port {port}: {error}") from error


def serve_mcp(
    tools: Iterable[Tool],
    transport: str = "stdio",
    host: str = MCP_SERVER_HOST,
    port: int = MCP_SERVER_PORT,
    path: str = MCP_SERVER_PATH,
    token: str | None = None,
    name: str = DEFAULT_SERVER_NAME,
    instructions: str | None = None,
) -> None:
    """Serves tools over MCP until the client goes, or the server is stopped.

    :param tools: The tools; see create_mcp_server.
    :param transport: "stdio", for a client that starts the server and speaks to it over its
        standard input and output; or "http", for streamable HTTP at host, port, and path.
    :param host: Where to listen, for HTTP. The default answers this machine alone, and refuses
        requests that name another host, so that a web page cannot reach the server by DNS
        rebinding. Anywhere else needs a token.
    :param port: The port, for HTTP, or 0 for any that is free.
    :param path: Where the server is, for HTTP.
    :param token: A token every HTTP request must carry, as the header Authorization: Bearer,
        of at least MCP_SERVER_MIN_TOKEN_CHARS characters. Optional on this machine alone.
    :param name: The server's name, as clients are told it.
    :param instructions: What clients are told of the server and how to use its tools.
    :raises ImportError: If the MCP SDK is not installed: pip install "virtual-lab[mcp]".
    :raises ValueError: If the transport is not one of these, or listening elsewhere than this
        machine without a token, or the token is too short.
    :raises OSError: If the server cannot listen at the host and port.
    """
    chosen = TRANSPORTS.get(str(transport).strip().casefold())
    if chosen is None:
        raise ValueError(f"Cannot serve over {transport!r}: use stdio or http")
    if chosen == "stdio" and token is not None:
        raise ValueError("A token is for serving over HTTP: over stdio, the client starts the server itself")
    if chosen == "http":
        check_token(token, host)
        if not str(path).startswith("/"):
            raise ValueError(f"The server's path must start with /, as /mcp does, not {path!r}")

    tools = tuple(tools)
    server = create_mcp_server(tools, name=name, instructions=instructions)
    served = f"{len(tools)} tool{'' if len(tools) == 1 else 's'}"

    if chosen == "stdio":
        import anyio
        from mcp.server.stdio import stdio_server

        async def run() -> None:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        print(f"Serving {served} over stdio", file=sys.stderr, flush=True)
        anyio.run(run)
        return

    import uvicorn

    app = server.streamable_http_app(streamable_http_path=path, host=host)
    if token is not None:
        app = RequireToken(app, token)

    listener = listening_socket(host, port)
    bound_host, bound_port = listener.getsockname()[:2]
    shown_host = f"[{bound_host}]" if ":" in bound_host else bound_host
    print(f"Serving {served} at http://{shown_host}:{bound_port}{path}", file=sys.stderr, flush=True)
    try:
        uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on")).run(sockets=[listener])
    finally:
        listener.close()


def server_code_tool(session: Session, biomni: bool = False) -> Tool:
    """The tool that runs code in the session, described for the clients of a server.

    :param session: The session.
    :param biomni: Whether the server's Biomni tools run in the same session.
    :return: The tool.
    """
    tool = session_tool(session)
    shared = " and for Biomni's tools, which run in it too" if biomni else ""
    opening = (
        "Run code in the server's shared interpreter and see what it prints. Python runs in one "
        "session that keeps its variables, imports, and loaded data from one call to the next, for "
        f"everyone using this server{shared}: what was defined before is there for you, so check "
        "before loading something again."
    )

    return replace(tool, description=" ".join([opening, *session_notes(session)]))


def import_tools(spec: str) -> tuple[Tool, ...]:
    """The tools a --tool option names: one of virtual_lab's tools by name, or module:attribute
    for a Tool, an iterable of them, such as what connect_mcp returns, or a function, made a
    tool with tool_from_function."""
    if spec in TOOL_REGISTRY:
        return (TOOL_REGISTRY[spec],)

    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(
            f"{spec!r} is not one of virtual_lab's tools ({', '.join(sorted(TOOL_REGISTRY))}), nor "
            "module:attribute, such as my_tools:TOOLS"
        )

    # A console script does not put the working directory on the path, as python -m does
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    value: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)

    if isinstance(value, Tool):
        return (value,)
    if hasattr(value, "tools") and not callable(value):
        value = value.tools
    if callable(value) and not isinstance(value, Iterable):
        from virtual_lab.custom_tools import tool_from_function

        return (tool_from_function(value),)
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        found = tuple(value)
        if all(isinstance(item, Tool) for item in found):
            return found

    raise TypeError(f"{spec} is not a Tool, an iterable of Tools, or a function")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="virtual-lab-mcp",
        description=(
            "Serve tools over MCP: Biomni's tool functions, each run in a session, the tool that runs "
            "code in that session, virtual_lab's database and literature tools, and tools of your own."
        ),
        epilog=(
            "Environment variables are read from .env in the working directory first, as Biomni reads "
            f"them. Over HTTP, a token is read from {MCP_SERVER_TOKEN_VARIABLE}."
        ),
    )
    served = parser.add_argument_group("what to serve (at least one)")
    served.add_argument(
        "--biomni-tools", action="store_true", help="Biomni's tool functions, each run in the session"
    )
    served.add_argument(
        "--biomni-module",
        action="append",
        metavar="MODULE",
        help="only the functions of this module of biomni.tool, such as genetics; may be repeated",
    )
    served.add_argument("--code-tool", action="store_true", help=f"{CODE_TOOL_NAME}, to run code in the session")
    served.add_argument("--database-tools", action="store_true", help="virtual_lab's database and literature tools")
    served.add_argument(
        "--tool",
        action="append",
        metavar="TOOL",
        help="one of virtual_lab's tools by name, or module:attribute for a Tool, a list of them, or a "
        "function of your own; may be repeated",
    )

    session = parser.add_argument_group("the session that Biomni's tools and run_code use")
    session.add_argument(
        "--session",
        choices=("docker", "local"),
        default="docker",
        help="docker, in the sandbox (the default), or local, on this machine with no isolation",
    )
    session.add_argument("--directory", type=Path, help="the session's working directory, where files are kept")
    session.add_argument(
        "--image-target",
        choices=("full", "bio", "base"),
        default="full",
        help="which stage of the sandbox image to use, for docker (default: full)",
    )
    session.add_argument("--data-lake", type=Path, help="Biomni's data lake, mounted read-only, for docker")
    session.add_argument("--no-network", action="store_true", help="keep the session's code off the network, for docker")
    session.add_argument(
        "--forward-env",
        action="append",
        default=[],
        metavar="NAME",
        help="an environment variable to pass to the session, such as an API key Biomni's tools use; "
        "may be repeated",
    )
    session.add_argument("--python", help="the interpreter to run, for local, such as Biomni's environment's")
    session.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_EXECUTION_TIMEOUT,
        help=f"seconds each call may run in the session (default: {DEFAULT_EXECUTION_TIMEOUT:g})",
    )

    serving = parser.add_argument_group("how to serve")
    serving.add_argument("--transport", choices=("stdio", "http"), default="stdio", help="default: stdio")
    serving.add_argument("--host", default=MCP_SERVER_HOST, help=f"where to listen, for http (default: {MCP_SERVER_HOST})")
    serving.add_argument("--port", type=int, default=MCP_SERVER_PORT, help=f"for http (default: {MCP_SERVER_PORT})")
    serving.add_argument("--path", default=MCP_SERVER_PATH, help=f"for http (default: {MCP_SERVER_PATH})")
    serving.add_argument("--name", default=DEFAULT_SERVER_NAME, help=f"the server's name (default: {DEFAULT_SERVER_NAME})")
    serving.add_argument("--env-file", type=Path, help="another .env file to read, after the one in the working directory")

    return parser


def make_session(arguments: argparse.Namespace) -> Session:
    if arguments.directory is None:
        raise ValueError("Biomni's tools and run_code run in a session: give its working directory with --directory")

    if arguments.session == "local":
        from virtual_lab.session import LocalSession

        print(
            "virtual-lab-mcp: warning: a local session runs the code it is sent on this machine, with no "
            "isolation: it can read and write any file you can. Prefer --session docker.",
            file=sys.stderr,
        )
        return LocalSession(
            arguments.directory,
            python=arguments.python,
            warn=False,
            timeout=arguments.timeout,
            biomni_tools=arguments.biomni_tools,
            forward_env=tuple(arguments.forward_env),
        )

    from virtual_lab.session import DockerSession, session_executor

    return DockerSession(
        arguments.directory,
        executor=session_executor(
            target=arguments.image_target,
            data_lake=arguments.data_lake,
            allow_network=not arguments.no_network,
            timeout=arguments.timeout,
            forward_env=tuple(arguments.forward_env),
        ),
    )


def server_instructions(session: Session | None, biomni: bool, code_tool: bool) -> str:
    notes = ["Tools from Virtual Lab, an AI research team for science."]
    if session is not None:
        what = "Biomni's tool functions run" if biomni else "Code runs"
        where = "in a sandbox" if session.sandboxed else "on the server's machine, with no isolation"
        notes.append(
            f"{what} in one session, {where}, whose working directory is {session.where_code_runs()}: "
            "paths are read there, and files written there are kept."
        )
        if biomni and code_tool:
            notes.append(f"{CODE_TOOL_NAME} runs code in the same session, so it can read what they write.")

    return " ".join(notes)


def collect_tools(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[list[Tool], Session | None]:
    if not (arguments.biomni_tools or arguments.code_tool or arguments.database_tools or arguments.tool):
        parser.error("say what to serve: --biomni-tools, --code-tool, --database-tools, or --tool")
    if arguments.biomni_module and not arguments.biomni_tools:
        parser.error("--biomni-module chooses among Biomni's tools: give --biomni-tools too")

    tools: list[Tool] = []
    for spec in arguments.tool or []:
        tools.extend(import_tools(spec))
    if arguments.database_tools:
        tools.extend(tool for tool in TOOL_REGISTRY.values() if tool not in tools)

    session = None
    if arguments.biomni_tools or arguments.code_tool:
        session = make_session(arguments)
        try:
            if arguments.biomni_tools:
                from virtual_lab.toolbox import biomni_session_tools

                tools.extend(biomni_session_tools(session, modules=arguments.biomni_module))
            if arguments.code_tool:
                tools.append(server_code_tool(session, biomni=arguments.biomni_tools))
        except BaseException:
            session.close()
            raise

    return tools, session


def main(argv: list[str] | None = None) -> int:
    """Serves tools over MCP, as the command virtual-lab-mcp.

    For a client that starts the server, such as Claude Desktop, add to its config:
    {"mcpServers": {"virtual-lab": {"command": "virtual-lab-mcp", "args": ["--biomni-tools",
    "--code-tool", "--directory", "/path/to/work"]}}}

    :param argv: The arguments, defaulting to the command line's.
    :return: The exit status.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    token = os.environ.get(MCP_SERVER_TOKEN_VARIABLE) or None
    session = None
    # A client stops a server it started with SIGTERM, which would otherwise end the process
    # without stopping the session's container
    previous = signal.signal(signal.SIGTERM, stop_on_signal) if in_main_thread() else None
    formatwarning = warnings.formatwarning
    warnings.formatwarning = command_warning
    try:
        # Over stdio, standard output carries the protocol, and nothing else may be written there
        with contextlib.redirect_stdout(sys.stderr) if arguments.transport == "stdio" else contextlib.nullcontext():
            if arguments.env_file is not None:
                from virtual_lab.env_file import load_env

                load_env(arguments.env_file)
                token = os.environ.get(MCP_SERVER_TOKEN_VARIABLE) or None
            if arguments.transport == "http":
                check_token(token, arguments.host)
            require_mcp()
            tools, session = collect_tools(arguments, parser)

        serve_mcp(
            tools,
            transport=arguments.transport,
            host=arguments.host,
            port=arguments.port,
            path=arguments.path,
            token=token if arguments.transport == "http" else None,
            name=arguments.name,
            instructions=server_instructions(session, arguments.biomni_tools, arguments.code_tool),
        )
    except KeyboardInterrupt:
        return 130
    except (ExecutionError, ImportError, OSError, TypeError, ValueError, AttributeError) as error:
        print(f"virtual-lab-mcp: error: {error}", file=sys.stderr)
        return 1
    finally:
        if session is not None:
            session.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        warnings.formatwarning = formatwarning

    return 0


def command_warning(message: Warning | str, category: type[Warning], *args: Any, **kwargs: Any) -> str:
    """A warning as the command shows it, without the line of virtual_lab that gave it."""
    return f"virtual-lab-mcp: warning: {message}\n"


def in_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def stop_on_signal(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    sys.exit(main())
