"""Keeping the server to the person who started it.

The server runs on this machine, with the keys and the code of whoever started it behind it, and any web
page open in the same browser can send it requests. So:

- A request must name this machine in its Host header. This keeps a page at another address that was
  made to point at this machine, by DNS rebinding, from reading anything.
- A request to change something must come from the server's own pages: its Origin, if it has one, is the
  server's own. A page at another address is turned away, whatever it holds.
- The API wants the token the server was started with, in the cookie the link the server prints sets, or
  as a bearer token. The cookie is SameSite=Strict, so it is not sent for a page at another address either.
"""

import secrets
from collections.abc import Collection
from urllib.parse import parse_qsl, urlencode, urlsplit

from starlette.datastructures import Headers
from starlette.requests import cookie_parser
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from virtual_lab.server.errors import ApiError

TOKEN_COOKIE = "vl_token"
TOKEN_PARAMETER = "token"

LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")

# Requests that only look at things, which need no check of where they come from
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# The only thing the API says without the token, so that a page can tell the server is there
OPEN_PATHS = frozenset({"/api/health"})


def new_token() -> str:
    """What the API is started wanting, which cannot be guessed."""
    return secrets.token_urlsafe(32)


def host_name(host: str) -> str:
    """A Host header without its port: "localhost:8000" is "localhost", "[::1]:8000" is "[::1]"."""
    if host.startswith("["):
        return host[: host.find("]") + 1].lower()

    return host.rsplit(":", 1)[0].lower()


class Guard:
    """Turns away requests that do not come from the person who started the server.

    :param app: The application it guards.
    :param token: What the API wants from every request, or None to want nothing.
    :param allowed_hosts: The names the server may be reached by, which are this machine's unless it was
        told of others.
    """

    def __init__(self, app: ASGIApp, token: str | None, allowed_hosts: Collection[str] = LOCAL_HOSTS) -> None:
        self.app = app
        self.token = token
        self.allowed_hosts = frozenset(host.lower() for host in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = [(name, value) for name, value in message["headers"]]
                given = {name.lower() for name, _ in response_headers}
                ours = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                ]
                if scope["path"].startswith("/api/"):
                    ours.append((b"cache-control", b"no-store"))
                # What a route says for itself is left as it said it
                response_headers += [(name, value) for name, value in ours if name not in given]
                message = {**message, "headers": response_headers}
            await send(message)

        headers = Headers(scope=scope)
        try:
            self.check_host(headers)
            if scope["method"] not in SAFE_METHODS:
                self.check_origin(headers)
            if (entry := self.token_entry(scope)) is not None:
                await entry(scope, receive, with_headers)
                return
            self.check_token(scope, headers)
        except ApiError as error:
            await JSONResponse(error.body(), status_code=error.status)(scope, receive, with_headers)
            return

        await self.app(scope, receive, with_headers)

    def check_host(self, headers: Headers) -> None:
        if host_name(headers.get("host", "")) not in self.allowed_hosts:
            raise ApiError(403, "forbidden_host", "This server is for this machine only: use its own address")

    def check_origin(self, headers: Headers) -> None:
        origin = headers.get("origin")
        if origin is not None and urlsplit(origin).netloc.lower() != headers.get("host", "").lower():
            raise ApiError(403, "forbidden_origin", "A page of another address may not change anything here")
        if headers.get("sec-fetch-site") == "cross-site":
            raise ApiError(403, "forbidden_origin", "A page of another address may not change anything here")

    def token_entry(self, scope: Scope) -> ASGIApp | None:
        """What to answer a page that is opened with the token in its address: remember it, and show the
        page without it in its address, so that it is not left in the history."""
        if scope["method"] != "GET" or scope["path"].startswith("/api/"):
            return None
        query = parse_qsl(scope["query_string"].decode("latin-1"), keep_blank_values=True)
        given = [value for name, value in query if name == TOKEN_PARAMETER]
        if not given or not self.matches(given[0]):
            return None

        rest = urlencode([(name, value) for name, value in query if name != TOKEN_PARAMETER])
        # Not "//", which a browser would take for another address
        path = "/" + scope["path"].lstrip("/")
        response = RedirectResponse(path + (f"?{rest}" if rest else ""), status_code=303)
        response.set_cookie(TOKEN_COOKIE, self.token, httponly=True, samesite="strict", path="/api")

        return response

    def check_token(self, scope: Scope, headers: Headers) -> None:
        if self.token is None or not scope["path"].startswith("/api/") or scope["path"] in OPEN_PATHS:
            return
        given = cookie_parser(headers.get("cookie", "")).get(TOKEN_COOKIE, "")
        authorization = headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            given = authorization[7:].strip()
        if not self.matches(given):
            raise ApiError(
                401, "unauthorized", "Open the link the server printed when it started, which has the token in it"
            )

    def matches(self, given: str) -> bool:
        return self.token is not None and secrets.compare_digest(given.encode(), self.token.encode())
