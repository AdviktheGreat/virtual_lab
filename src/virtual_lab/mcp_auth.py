"""Signing in to an MCP server at a URL with OAuth, as Proto's hosted server requires, and as
Paperclip's and Adaptyv's servers allow in place of an API key.

A server whose config says auth: oauth is signed in to in a browser the first time it is
connected to: the server's sign-in page opens, and once the person has signed in, the browser
is sent back to an address on this machine, 127.0.0.1, where the sign-in is received. The MCP
SDK does the rest: it finds the server's authorization server, registers this client with it,
and exchanges what the browser brings back for tokens, with PKCE. The tokens are kept in a file
of their own, readable by this user alone, so that the next connection needs no sign-in, and
are refreshed as they expire.

Where no browser can be opened, as over SSH, the address of the sign-in page is shown at the
terminal, to be opened anywhere, and the address the browser ends at is pasted back.
"""

import asyncio
import hashlib
import html
import json
import os
import re
import socket
import tempfile
import threading
import time
import warnings
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from virtual_lab.__about__ import __version__
from virtual_lab.approval import TERMINAL, asked_for, read_reply, say, terminal_available
from virtual_lab.constants import MCP_AUTH_DIRECTORY, MCP_AUTH_DIRECTORY_VARIABLE, MCP_SIGN_IN_TIMEOUT

# Where the browser is sent back to, on this machine, once the person has signed in
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PATH = "/callback"

# Seconds a connection to the address the browser is sent back to may take to say what it asks
CALLBACK_READ_TIMEOUT = 10.0


class SignInError(RuntimeError):
    """Raised when signing in to an MCP server cannot be done, or was not."""


def auth_directory() -> Path:
    """Where the sign-ins to MCP servers are kept: VIRTUAL_LAB_MCP_AUTH_DIR, or else
    ~/.virtual_lab/mcp_auth."""
    return Path(os.environ.get(MCP_AUTH_DIRECTORY_VARIABLE) or MCP_AUTH_DIRECTORY).expanduser()


def sign_in_path(url: str) -> Path:
    """The file the sign-in to the server at a URL is kept in, named after its host so that it
    can be found, and after the whole URL so that two servers on one host are kept apart."""
    host = re.sub(r"[^A-Za-z0-9.-]", "_", urlsplit(url).netloc) or "server"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    return auth_directory() / f"{host}-{digest}.json"


class SignInStorage:
    """The SDK's token storage, kept in a file readable by this user alone.

    The file holds the client registered with the server's authorization server, the tokens,
    and what the SDK does not keep: when the access token expires, and where the authorization
    server, which may be on another host than the MCP server, refreshes it.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.path = sign_in_path(url)
        self.lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            warnings.warn(
                f"The sign-in to the MCP server at {self.url}, in {self.path}, could not be read, so it will be "
                f"signed in to again: {error}",
                UserWarning,
                stacklevel=2,
            )
            return {}
        # A file of another server's, which a hash shared by two URLs would give, is not used
        if not isinstance(data, dict) or data.get("server_url") != self.url:
            return {}

        return data

    def update(self, **values: Any) -> None:
        with self.lock:
            data = {**self.read(), **values, "server_url": self.url}
            # Readable by this user alone where it is made here; a directory named that exists
            # already is left as it is
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            # Written whole and then moved into place, so that a sign-in is never left half
            # written; mkstemp makes the file readable by this user alone
            descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".signing-in-", suffix=".json")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=2)
                os.replace(temporary, self.path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise

    def forget_client(self) -> None:
        """Forgets the registered client, and with it the tokens issued to it, so that a client is
        registered again."""
        self.update(client_info=None, tokens=None, expires_at=None, discovered=None)

    @property
    def expires_at(self) -> float | None:
        """When the access token kept expires, as time.time() counts, or None if it is not known."""
        value = self.read().get("expires_at")
        return float(value) if isinstance(value, int | float) else None

    def stored_client(self) -> dict[str, Any] | None:
        client = self.read().get("client_info")
        return client if isinstance(client, dict) else None

    def discovered(self) -> tuple[Any, Any]:
        """The authorization server's metadata, and the MCP server's, found when the tokens were
        issued, or None for either that is not kept."""
        from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata
        from pydantic import ValidationError

        found = self.read().get("discovered")
        if not isinstance(found, dict):
            return None, None
        models: list[Any] = []
        for key, model in (("authorization_server", OAuthMetadata), ("resource", ProtectedResourceMetadata)):
            try:
                models.append(model.model_validate(found[key]) if isinstance(found.get(key), dict) else None)
            except ValidationError:
                models.append(None)

        return models[0], models[1]

    def remember_discovered(self, authorization_server: Any, resource: Any) -> None:
        self.update(
            discovered={
                "authorization_server": authorization_server.model_dump(mode="json", exclude_none=True)
                if authorization_server is not None
                else None,
                "resource": resource.model_dump(mode="json", exclude_none=True) if resource is not None else None,
            }
        )

    async def get_tokens(self) -> Any:
        from mcp.shared.auth import OAuthToken
        from pydantic import ValidationError

        tokens = self.read().get("tokens")
        if not isinstance(tokens, dict):
            return None
        try:
            return OAuthToken.model_validate(tokens)
        except ValidationError:
            return None

    async def set_tokens(self, tokens: Any) -> None:
        expires_at = time.time() + tokens.expires_in if tokens.expires_in is not None else None
        self.update(tokens=tokens.model_dump(mode="json", exclude_none=True), expires_at=expires_at)

    async def get_client_info(self) -> Any:
        from mcp.shared.auth import OAuthClientInformationFull
        from pydantic import ValidationError

        client = self.stored_client()
        if client is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate(client)
        except ValidationError:
            return None

    async def set_client_info(self, client_info: Any) -> None:
        self.update(client_info=client_info.model_dump(mode="json", exclude_none=True))


class PausedClock:
    """The seconds spent waiting for a person to sign in, which are not counted against the time
    a server has to be connected to, or a call to finish."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total = 0.0
        self.since: float | None = None

    def pause(self) -> None:
        with self.lock:
            if self.since is None:
                self.since = time.monotonic()

    def resume(self) -> None:
        with self.lock:
            if self.since is not None:
                self.total += time.monotonic() - self.since
                self.since = None

    @property
    def paused(self) -> bool:
        with self.lock:
            return self.since is not None

    def seconds(self) -> float:
        with self.lock:
            return self.total + (time.monotonic() - self.since if self.since is not None else 0.0)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((CALLBACK_HOST, 0))
        return int(probe.getsockname()[1])


def callback_port(storage: SignInStorage) -> int:
    """The port the browser is sent back to: the one the client kept was registered with, since a
    server may send it only to the address it was registered with; or a free one, for a client
    yet to be registered."""
    client = storage.stored_client()
    if client is not None:
        for uri in client.get("redirect_uris") or []:
            parts = urlsplit(str(uri))
            if parts.hostname == CALLBACK_HOST and parts.path == CALLBACK_PATH and parts.port:
                return parts.port
        # Registered for somewhere else, which this cannot receive a sign-in at
        storage.forget_client()

    return free_port()


def open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:  # noqa: BLE001
        return False


def page(title: str, text: str) -> bytes:
    body = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head>"
        f"<body style='font-family: sans-serif; margin: 3em'><h1>{html.escape(title)}</h1>"
        f"<p>{html.escape(text)}</p></body></html>"
    )
    return body.encode("utf-8")


async def respond(writer: asyncio.StreamWriter, status: str, body: bytes) -> None:
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode("ascii")
        + body
    )
    await writer.drain()


def first_values(query: str) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(query).items() if values}


class SignIn:
    """Signs a person in to one MCP server: what the SDK's OAuth provider is given to send them
    to the server's sign-in page, and to receive where the browser is sent back to."""

    def __init__(self, server: str, url: str, clock: PausedClock) -> None:
        self.server = server
        self.url = url
        self.clock = clock
        self.storage = SignInStorage(url)
        self.port = callback_port(self.storage)
        self.redirect_uri = f"http://{CALLBACK_HOST}:{self.port}{CALLBACK_PATH}"
        self.listener: asyncio.Server | None = None
        self.received: asyncio.Future[dict[str, str]] | None = None
        self.state: str | None = None
        self.pasting = False

    def provider(self) -> Any:
        """The SDK's OAuth provider for the server, as httpx's auth."""
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata
        from pydantic import AnyUrl

        storage = self.storage

        class Provider(OAuthClientProvider):
            async def _initialize(self) -> None:
                await super()._initialize()
                # The SDK takes a token it loads to be valid however old it is, and so would send
                # one that has expired, and on being refused have the person sign in again rather
                # than use the refresh token kept with it. It refreshes a token at the token
                # endpoint of the authorization server it found, which it would otherwise take to
                # be on the MCP server's host, where Proto's is not
                if self.context.current_tokens is not None:
                    self.context.token_expiry_time = storage.expires_at
                    authorization_server, resource = storage.discovered()
                    if self.context.oauth_metadata is None:
                        self.context.oauth_metadata = authorization_server
                    if self.context.protected_resource_metadata is None:
                        self.context.protected_resource_metadata = resource

            async def _handle_token_response(self, response: Any) -> None:
                await super()._handle_token_response(response)
                storage.remember_discovered(self.context.oauth_metadata, self.context.protected_resource_metadata)

        # No token_endpoint_auth_method is asked for, so that the server registers the client
        # with one it takes, as RFC 7591 lets it, and the SDK checks that it can use it
        metadata = OAuthClientMetadata(
            client_name="Virtual Lab",
            redirect_uris=[AnyUrl(self.redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            software_version=__version__,
        )

        return Provider(
            server_url=self.url,
            client_metadata=metadata,
            storage=storage,
            redirect_handler=self.redirect,
            callback_handler=self.callback,
        )

    async def redirect(self, authorization_url: str) -> None:
        """Sends the person to the server's sign-in page, having first begun to listen where the
        browser is sent back to."""
        self.clock.pause()
        try:
            self.state = first_values(urlsplit(authorization_url).query).get("state")
            self.received = asyncio.get_running_loop().create_future()
            try:
                self.listener = await asyncio.start_server(self.handle, CALLBACK_HOST, self.port)
            except OSError as error:
                # Registered again at the next connection, with a port that is free then
                self.storage.forget_client()
                raise SignInError(
                    f"Signing in to the MCP server {self.server} needs port {self.port} of this machine, which is in "
                    f"use: {error}. Connect again to sign in at another."
                ) from None
            say(
                f"\nThe MCP server {self.server} needs you to sign in. Its sign-in page is opening in your browser; "
                f"if it does not open, go to:\n{authorization_url}\n"
            )
            if await asyncio.to_thread(open_browser, authorization_url):
                return
            if not terminal_available():
                raise SignInError(
                    f"Signing in to the MCP server {self.server} needs a browser, and none could be opened here, nor "
                    "is there a terminal to paste the address it ends at into. Connect once where one can be, and the "
                    f"sign-in is kept in {self.storage.path} for the next connection."
                )
            self.pasting = True
        except BaseException:
            await self.close()
            self.clock.resume()
            raise

    async def callback(self) -> Any:
        """Waits for the browser to be sent back, and gives the SDK what it brings."""
        from mcp.shared.auth import AuthorizationCodeResult

        assert self.received is not None
        try:
            if self.pasting:
                # The browser the page was opened in may yet be sent back here, as over SSH with
                # the port forwarded, so whichever comes first is taken
                pasting = asyncio.ensure_future(self.pasted())
                try:
                    done, _ = await asyncio.wait({pasting, self.received}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    # One that ended as the browser came back has what it raised looked at, so
                    # that asyncio does not warn that it never was
                    if not pasting.cancel() and not pasting.cancelled():
                        pasting.exception()
                answer = self.received.result() if self.received in done else pasting.result()
            else:
                try:
                    answer = await asyncio.wait_for(self.received, MCP_SIGN_IN_TIMEOUT)
                except TimeoutError:
                    raise SignInError(
                        f"No one signed in to the MCP server {self.server} within {MCP_SIGN_IN_TIMEOUT:g} seconds"
                    ) from None
        finally:
            await self.close()
            self.clock.resume()

        if "error" in answer:
            detail = f": {answer['error_description']}" if answer.get("error_description") else ""
            raise SignInError(f"Signing in to the MCP server {self.server} did not succeed: {answer['error']}{detail}")

        return AuthorizationCodeResult(code=answer.get("code", ""), state=answer.get("state"), iss=answer.get("iss"))

    async def pasted(self) -> dict[str, str]:
        """What the address the person pastes brings, where no browser could be opened here."""
        import anyio

        finished = threading.Event()

        def ask() -> str | None:
            with asked_for(finished), TERMINAL:
                return read_reply("Once you have signed in, paste the address your browser ends at: ")

        try:
            with anyio.fail_after(MCP_SIGN_IN_TIMEOUT):
                text = await anyio.to_thread.run_sync(ask, abandon_on_cancel=True)
        except TimeoutError:
            raise SignInError(
                f"No one signed in to the MCP server {self.server} within {MCP_SIGN_IN_TIMEOUT:g} seconds"
            ) from None
        finally:
            finished.set()
        if not text:
            raise SignInError(f"No address was pasted, so the MCP server {self.server} was not signed in to")
        answer = first_values(urlsplit(text).query or text.lstrip("?"))
        if "error" not in answer and ("code" not in answer or answer.get("state") != self.state):
            raise SignInError(
                f"The address pasted is not the one signing in to the MCP server {self.server} ends at, which begins "
                f"{self.redirect_uri}?code="
            )

        return answer

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Answers a request to where the browser is sent back to. Only the one that brings this
        sign-in's state is taken, so that another program on this machine cannot end it."""
        try:
            line = await asyncio.wait_for(reader.readline(), CALLBACK_READ_TIMEOUT)
            # The headers, which say nothing needed, are read to the blank line that ends them
            while await asyncio.wait_for(reader.readline(), CALLBACK_READ_TIMEOUT) not in (b"\r\n", b"\n", b""):
                pass
            method, target, *_ = line.decode("latin-1").split() + ["", ""]
            parts = urlsplit(target)
            answer = first_values(parts.query)
            if method != "GET" or parts.path != CALLBACK_PATH:
                await respond(writer, "404 Not Found", page("Not found", "This address only receives a sign-in."))
                return
            if answer.get("state") is None or answer.get("state") != self.state:
                await respond(
                    writer,
                    "400 Bad Request",
                    page("Not this sign-in", "This is not the sign-in that is waiting. Sign in again from the start."),
                )
                return
            if self.received is not None and not self.received.done():
                self.received.set_result(answer)
            if "error" in answer:
                await respond(
                    writer,
                    "200 OK",
                    page("Not signed in", f"Signing in to {self.server} did not succeed: {answer['error']}."),
                )
            else:
                await respond(
                    writer, "200 OK", page("Signed in", f"You are signed in to {self.server}. You can close this tab.")
                )
        except (TimeoutError, ConnectionError, ValueError):
            pass
        finally:
            writer.close()

    async def close(self) -> None:
        if self.listener is not None:
            self.listener.close()
            self.listener = None


def sign_out_mcp(server: str) -> bool:
    """Forgets the sign-in to an MCP server, so that the next connection to it signs in again, as
    another account, say.

    :param server: The name of a preset, such as "proto_hosted", or the server's URL.
    :raises ValueError: If the preset is of a server started here, which is not signed in to, or
        what is given is neither a preset nor an http:// or https:// URL.
    :return: Whether there was a sign-in to forget.
    """
    from virtual_lab.mcp_presets import MCP_PRESETS

    url = server
    if server in MCP_PRESETS:
        url = MCP_PRESETS[server].entry.get("url")
        if not isinstance(url, str):
            raise ValueError(f"The preset {server} is of a server started here, which is not signed in to")
    elif not isinstance(server, str) or urlsplit(server).scheme not in ("http", "https") or not urlsplit(server).netloc:
        raise ValueError(
            f"sign_out_mcp is given a preset, such as proto_hosted, or the URL of an MCP server, not {server!r}"
        )
    path = sign_in_path(url)
    existed = path.exists()
    path.unlink(missing_ok=True)

    return existed
