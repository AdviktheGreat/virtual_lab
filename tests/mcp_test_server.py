"""An MCP server for the tests of connect_mcp, run as its own process.

python mcp_test_server.py [--pid-file PATH] [--sleep SECONDS] [--say TEXT] [--instructions TEXT]
    [--oauth [--token-seconds SECONDS] [--token-path PATH] [--deny]] [stdio | http PORT | sse PORT | die]

With --oauth, a server over HTTP or SSE is signed in to with OAuth, from an authorization server
of its own that registers any client and signs anyone in at once, as a browser would find once a
person had signed in. Its tokens are issued at --token-path, away from the MCP server's own
/token, as an authorization server on another host issues them. GET /oauth-stats says how many
clients were registered, sign-ins made, and tokens refreshed.
"""

import argparse
import os
import secrets
import sys
import time
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.server.auth.routes import build_metadata
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.mcpserver import Context, Elicit, ElicitationResult, Resolve
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl, BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

parser = argparse.ArgumentParser()
parser.add_argument("--pid-file")
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--say", help="Text to write to stderr as the server starts")
parser.add_argument("--instructions", help="What the server says of its tools in place of its own, or '' for nothing")
parser.add_argument("--oauth", action="store_true", help="Be signed in to with OAuth")
parser.add_argument("--token-seconds", type=int, default=3600, help="Seconds an access token lasts")
parser.add_argument("--token-path", default="/token", help="Where tokens are issued")
parser.add_argument("--deny", action="store_true", help="Turn every sign-in down, as a person might")
parser.add_argument("mode", nargs="?", default="stdio")
parser.add_argument("port", nargs="?", type=int)
arguments = parser.parse_args()


class SignIns:
    """An authorization server kept in memory, which signs anyone in at once."""

    def __init__(self, token_seconds: int, deny: bool) -> None:
        self.token_seconds = token_seconds
        self.deny = deny
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.codes: dict[str, AuthorizationCode] = {}
        self.access: dict[str, AccessToken] = {}
        self.refresh: dict[str, RefreshToken] = {}
        self.stats = {"registered": 0, "signed_in": 0, "refreshed": 0, "redirect_uris": []}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.clients[client_info.client_id] = client_info
        self.stats["registered"] += 1
        self.stats["redirect_uris"] += [str(uri) for uri in client_info.redirect_uris or []]

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if self.deny:
            return construct_redirect_uri(
                str(params.redirect_uri), error="access_denied", error_description="The person said no", state=params.state
            )
        code = secrets.token_urlsafe(16)
        self.codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 300,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        self.stats["signed_in"] += 1
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(self, client: OAuthClientInformationFull, code: str) -> AuthorizationCode | None:
        found = self.codes.get(code)
        return found if found is not None and found.client_id == client.client_id else None

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, code: AuthorizationCode) -> OAuthToken:
        del self.codes[code.code]
        return self.issue(client, code.scopes, code.resource)

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        found = self.refresh.get(refresh_token)
        return found if found is not None and found.client_id == client.client_id else None

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        del self.refresh[refresh_token.token]
        self.stats["refreshed"] += 1
        return self.issue(client, scopes or refresh_token.scopes, refresh_token.resource)

    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.access.get(token)

    async def revoke_token(self, token: Any) -> None:
        self.access.pop(token.token, None)
        self.refresh.pop(token.token, None)

    def issue(self, client: OAuthClientInformationFull, scopes: list[str], resource: str | None) -> OAuthToken:
        access, refresh = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
        self.access[access] = AccessToken(
            token=access,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=int(time.time()) + self.token_seconds,
            resource=resource,
        )
        self.refresh[refresh] = RefreshToken(token=refresh, client_id=client.client_id, scopes=scopes, resource=resource)
        return OAuthToken(
            access_token=access, expires_in=self.token_seconds, refresh_token=refresh, scope=" ".join(scopes) or None
        )


sign_ins = SignIns(arguments.token_seconds, arguments.deny) if arguments.oauth else None
origin = f"http://127.0.0.1:{arguments.port}"
auth = (
    AuthSettings(
        issuer_url=origin,
        resource_server_url=f"{origin}/{'mcp' if arguments.mode == 'http' else 'sse'}",
        client_registration_options=ClientRegistrationOptions(enabled=True),
        validate_token_resource=False,
    )
    if sign_ins is not None
    else None
)
server = MCPServer("Test genes", instructions="Look genes up by symbol.", auth_server_provider=sign_ins, auth=auth)
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


@server.custom_route("/oauth-stats", methods=["GET"])
async def oauth_stats(request: Request) -> JSONResponse:
    return JSONResponse(sign_ins.stats if sign_ins is not None else {})


def remember_headers(app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            for key, value in scope["headers"]:
                if key == b"authorization":
                    seen["authorization"] = value.decode()
            if auth is not None and arguments.token_path != "/token":
                if scope["path"] == "/.well-known/oauth-authorization-server":
                    metadata = build_metadata(
                        auth.issuer_url, None, auth.client_registration_options, RevocationOptions()
                    )
                    metadata.token_endpoint = AnyHttpUrl(f"{origin}{arguments.token_path}")
                    response = JSONResponse(metadata.model_dump(mode="json", exclude_none=True))
                    return await response(scope, receive, send)
                if scope["path"] == "/token":
                    return await JSONResponse({"error": "not here"}, status_code=404)(scope, receive, send)
                if scope["path"] == arguments.token_path:
                    scope = {**scope, "path": "/token", "raw_path": b"/token"}
        await app(scope, receive, send)

    return wrapped


if __name__ == "__main__":
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
