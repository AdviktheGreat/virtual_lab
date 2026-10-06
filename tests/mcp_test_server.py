"""An MCP server for the tests of connect_mcp, run as its own process.

python mcp_test_server.py [--pid-file PATH] [--sleep SECONDS] [--say TEXT] [--instructions TEXT]
    [stdio | http PORT | sse PORT | die]
"""

import argparse
import os
import sys
import time
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context, Elicit, ElicitationResult, Resolve
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from pydantic import BaseModel

server = MCPServer("Test genes", instructions="Look genes up by symbol.")
# The Authorization header of the last HTTP request, for the tests to check what was sent
seen = {"authorization": "<none>"}
# How many times count was called, for the tests to check that a call was not made
counted = {"calls": 0}


class Gene(BaseModel):
    symbol: str
    length: int


@server.tool()
def lookup(symbol: str) -> Gene:
    """Looks a gene up by its symbol.

    Only TP53 is known.
    """
    if symbol != "TP53":
        raise ToolError(f"There is no gene {symbol}")
    return Gene(symbol=symbol, length=393)


@server.tool()
def echo(text: str, times: int = 1) -> str:
    """Says the text back."""
    return text * times


@server.tool()
def total(values: list[int]) -> int:
    """Adds the values up."""
    return sum(values)


@server.tool()
def lengths(symbols: list[str]) -> dict[str, int]:
    """How long each gene's name is."""
    return {symbol: len(symbol) for symbol in symbols}


@server.tool()
def picture():
    """A tiny picture."""
    return Image(data=b"\x89PNG\r\n\x1a\n0123456789", format="png")


@server.tool()
def variable(name: str) -> str:
    """The value of an environment variable of the server's, or <unset>."""
    return os.environ.get(name, "<unset>")


@server.tool()
def directory() -> str:
    """The directory the server runs in."""
    return os.getcwd()


@server.tool()
def process() -> int:
    """The server's process ID."""
    return os.getpid()


@server.tool()
def wait(seconds: float) -> str:
    """Waits, then says so."""
    time.sleep(seconds)
    return f"waited {seconds}"


@server.tool()
def crash() -> str:
    """Stops the server."""
    print("the server is crashing", file=sys.stderr, flush=True)
    os._exit(3)


@server.tool()
def authorization() -> str:
    """The Authorization header the last request carried."""
    return seen["authorization"]


class Outcome(BaseModel):
    result: str


@server.tool()
def outcome() -> Outcome:
    """An object whose one field is called result."""
    return Outcome(result="positive")


@server.tool(name="find-genes.v2")
def find_genes(query: str) -> list[str]:
    """Finds genes whose symbols start with the query."""
    return [symbol for symbol in ("TP53", "TP63", "TP73", "BRCA1") if symbol.startswith(query)]


@server.tool()
def count() -> int:
    """Counts the calls made to it."""
    counted["calls"] += 1
    return counted["calls"]


class GoAhead(BaseModel):
    approve: bool


def ask_to_go_ahead() -> Elicit[GoAhead]:
    return Elicit("Go ahead with the test?", GoAhead)


@server.tool()
def confirm(answer: Annotated[ElicitationResult[GoAhead], Resolve(ask_to_go_ahead)]) -> str:
    """Asks a person whether to go ahead, and says what they answered."""
    return f"{answer.action} {answer.data.approve}" if answer.action == "accept" else answer.action


class Order(BaseModel):
    name: str
    kind: Literal["protein", "dna"]
    copies: int = 1


def ask_what_to_order() -> Elicit[Order]:
    return Elicit("What is to be ordered?", Order)


@server.tool()
def order(answer: Annotated[ElicitationResult[Order], Resolve(ask_what_to_order)]) -> dict:
    """Asks a person what to order, and says what they answered."""
    return {"action": answer.action, "order": answer.data.model_dump() if answer.action == "accept" else None}


@server.tool()
async def visit(ctx: Context) -> str:
    """Asks a person to go to a web page, and says whether they agreed to. Only over the protocol
    before 2026, which lets a server send a request of its own while a tool runs."""
    answer = await ctx.elicit_url("Pay for the test at this page.", "https://pay.example.org/test", "payment-1")
    return answer.action


def remember_headers(app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            for key, value in scope["headers"]:
                if key == b"authorization":
                    seen["authorization"] = value.decode()
        await app(scope, receive, send)

    return wrapped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--say", help="Text to write to stderr as the server starts")
    parser.add_argument("--instructions", help="What the server says of its tools in place of its own, or '' for nothing")
    parser.add_argument("mode", nargs="?", default="stdio")
    parser.add_argument("port", nargs="?", type=int)
    arguments = parser.parse_args()
    if arguments.instructions is not None:
        server._lowlevel_server.instructions = arguments.instructions or None

    print("the test server is starting", file=sys.stderr, flush=True)
    if arguments.say:
        print(arguments.say, file=sys.stderr, flush=True)
    if arguments.pid_file:
        with open(arguments.pid_file, "w") as file:
            file.write(str(os.getpid()))
    time.sleep(arguments.sleep)

    if arguments.mode == "die":
        print("the test server has no config, so it stops", file=sys.stderr, flush=True)
        sys.exit(2)
    if arguments.mode in ("http", "sse"):
        import uvicorn

        app = server.streamable_http_app() if arguments.mode == "http" else server.sse_app()
        uvicorn.run(remember_headers(app), host="127.0.0.1", port=arguments.port, log_level="warning")
    else:
        server.run()
