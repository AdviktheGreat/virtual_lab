"""Tests for signing in to MCP servers with OAuth, against the test server run with an
authorization server of its own, and a browser that does what one would once a person had signed in."""

import asyncio
import concurrent.futures
import http.client
import io
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

pytest.importorskip("mcp")

from virtual_lab import MCP_PRESETS, SignInError, connect_mcp, sign_out_mcp  # noqa: E402
from virtual_lab.mcp_auth import (  # noqa: E402
    PausedClock,
    SignIn,
    SignInStorage,
    auth_directory,
    callback_port,
    sign_in_path,
)
from virtual_lab.mcp_presets import preset_entry  # noqa: E402
from virtual_lab.mcp_tools import MCPConnection, MCPServerError, parse_server  # noqa: E402

SERVER = str(Path(__file__).with_name("mcp_test_server.py"))


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def get(url: str) -> tuple[int, str | None, str]:
    """A GET of the URL, without following a redirect: its status, where it redirects to, and its body."""
    parts = urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.request("GET", parts.path + (f"?{parts.query}" if parts.query else ""))
        response = connection.getresponse()
        return response.status, response.getheader("Location"), response.read().decode()
    finally:
        connection.close()


class OAuthServer:
    """The test server, signed in to with OAuth, over HTTP or SSE in a process of its own."""

    def __init__(self, mode: str = "http", *options: str) -> None:
        self.mode = mode
        self.options = options
        self.port = free_port()
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def url(self) -> str:
        return f"{self.origin}/{'mcp' if self.mode == 'http' else 'sse'}"

    def config(self, **entry: Any) -> dict[str, Any]:
        return {"mcp_servers": {"genes": {"url": self.url, "auth": "oauth", "transport": self.mode, **entry}}}

    def stats(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.origin}/oauth-stats", timeout=10) as response:
            return json.loads(response.read())

    def start(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, SERVER, "--oauth", *self.options, self.mode, str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError(f"The test server did not start on port {self.port}")

    def stop(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.process.wait()
            self.process = None


def running(*options: str, mode: str = "http") -> Iterator[OAuthServer]:
    server = OAuthServer(mode, *options)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def oauth_server() -> Iterator[OAuthServer]:
    # Its tokens are issued away from its /token, as Proto's are, by an authorization server on
    # another host
    yield from running("--token-path", "/auth/token")


@pytest.fixture(autouse=True)
def auth_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "mcp_auth"
    monkeypatch.setenv("VIRTUAL_LAB_MCP_AUTH_DIR", str(directory))
    return directory


class Browser:
    """In place of webbrowser.open: goes to the sign-in page, which sends it back at once, as it
    would once a person had signed in, and follows it back."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.pages: list[tuple[int, str]] = []
        self.works = True
        self.delay = 0.0

    def open(self, url: str, *args: Any, **kwargs: Any) -> bool:
        self.opened.append(url)
        if not self.works:
            return False
        time.sleep(self.delay)
        status, location, body = get(url)
        assert status in (302, 307), (status, body)
        assert location is not None
        status, _, body = get(location)
        self.pages.append((status, body))
        return True


@pytest.fixture
def browser(monkeypatch: pytest.MonkeyPatch) -> Browser:
    opened = Browser()
    monkeypatch.setattr(webbrowser, "open", opened.open)
    return opened


def stored(directory: Path) -> tuple[Path, dict[str, Any]]:
    [path] = list(directory.glob("*.json"))
    return path, json.loads(path.read_text())


def authorization(tools: Any) -> str:
    return {tool.name: tool for tool in tools.tools}["genes_authorization"].function()


class TestSigningIn:
    def test_a_person_signs_in_once_and_the_sign_in_is_kept_for_the_next_connection(
        self, oauth_server: OAuthServer, browser: Browser, auth_dir: Path
    ) -> None:
        with connect_mcp(oauth_server.config(), start_timeout=30) as tools:
            first = authorization(tools)
        assert first.startswith("Bearer ")
        assert len(browser.opened) == 1
        assert browser.pages[0][0] == 200 and "You are signed in to genes" in browser.pages[0][1]

        path, kept = stored(auth_dir)
        assert path == sign_in_path(oauth_server.url)
        assert kept["server_url"] == oauth_server.url
        assert f"Bearer {kept['tokens']['access_token']}" == first
        assert kept["tokens"]["refresh_token"]
        assert kept["client_info"]["client_name"] == "Virtual Lab"
        assert kept["expires_at"] == pytest.approx(time.time() + 3600, abs=60)
        assert kept["discovered"]["authorization_server"]["token_endpoint"] == f"{oauth_server.origin}/auth/token"
        # Readable by this user alone
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(auth_dir.stat().st_mode) == 0o700

        with connect_mcp(oauth_server.config(), start_timeout=30) as tools:
            assert authorization(tools) == first
        assert len(browser.opened) == 1
        assert oauth_server.stats() | {"redirect_uris": None} == {
            "registered": 1,
            "signed_in": 1,
            "refreshed": 0,
            "redirect_uris": None,
        }

    def test_a_token_that_has_expired_is_refreshed_where_the_authorization_server_issues_tokens(
        self, oauth_server: OAuthServer, browser: Browser, auth_dir: Path
    ) -> None:
        with connect_mcp(oauth_server.config(), start_timeout=30) as tools:
            first = authorization(tools)
        # Still taken by the server, so only a client that knows it has expired refreshes it
        path, kept = stored(auth_dir)
        path.write_text(json.dumps({**kept, "expires_at": time.time() - 10}))

        with connect_mcp(oauth_server.config(), start_timeout=30) as tools:
            second = authorization(tools)
        assert second != first
        assert len(browser.opened) == 1
        assert oauth_server.stats()["refreshed"] == 1
        assert oauth_server.stats()["signed_in"] == 1
        assert stored(auth_dir)[1]["expires_at"] > time.time() + 3000

    def test_a_token_the_server_no_longer_takes_is_refreshed_without_signing_in_again(
        self, browser: Browser, auth_dir: Path
    ) -> None:
        for server in running("--token-seconds", "2", "--token-path", "/auth/token"):
            with connect_mcp(server.config(), start_timeout=30) as tools:
                authorization(tools)
            time.sleep(2.5)
            with connect_mcp(server.config(), start_timeout=30) as tools:
                assert authorization(tools).startswith("Bearer ")
            assert server.stats()["signed_in"] == 1
            assert server.stats()["refreshed"] >= 1
            assert len(browser.opened) == 1

    def test_a_server_over_sse_is_signed_in_to(self, browser: Browser) -> None:
        for server in running(mode="sse"):
            with connect_mcp(server.config(), start_timeout=30) as tools:
                assert authorization(tools).startswith("Bearer ")
            assert len(browser.opened) == 1

    def test_the_time_a_person_takes_to_sign_in_is_not_counted_against_start_timeout(
        self, oauth_server: OAuthServer, browser: Browser
    ) -> None:
        browser.delay = 3.0
        with connect_mcp(oauth_server.config(), start_timeout=1.5) as tools:
            assert authorization(tools).startswith("Bearer ")

    def test_a_sign_in_turned_down_says_so(self, browser: Browser) -> None:
        for server in running("--deny"):
            with pytest.raises(MCPServerError, match="did not succeed: access_denied: The person said no"):
                connect_mcp(server.config(), start_timeout=30)
            assert browser.pages[0][0] == 200 and "did not succeed" in browser.pages[0][1]

    def test_signing_out_has_the_next_connection_sign_in_again(
        self, oauth_server: OAuthServer, browser: Browser, auth_dir: Path
    ) -> None:
        with connect_mcp(oauth_server.config(), start_timeout=30):
            pass
        assert sign_out_mcp(oauth_server.url) is True
        assert not list(auth_dir.glob("*.json"))
        assert sign_out_mcp(oauth_server.url) is False
        with connect_mcp(oauth_server.config(), start_timeout=30):
            pass
        assert len(browser.opened) == 2
        assert oauth_server.stats()["registered"] == 2

    def test_signing_in_again_uses_the_client_registered_and_its_port(
        self, oauth_server: OAuthServer, browser: Browser, auth_dir: Path
    ) -> None:
        with connect_mcp(oauth_server.config(), start_timeout=30):
            pass
        path, kept = stored(auth_dir)
        path.write_text(json.dumps({**kept, "tokens": None, "expires_at": None}))
        with connect_mcp(oauth_server.config(), start_timeout=30):
            pass
        stats = oauth_server.stats()
        assert (stats["registered"], stats["signed_in"]) == (1, 2)
        [redirect_uri] = stats["redirect_uris"]
        assert all(f"redirect_uri={redirect_uri.replace(':', '%3A').replace('/', '%2F')}" in url for url in browser.opened)

    def test_where_no_browser_opens_nor_is_there_a_terminal_it_says_how_to_sign_in(
        self, oauth_server: OAuthServer, browser: Browser, auth_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        browser.works = False
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        with pytest.raises(MCPServerError, match="needs a browser, and none could be opened here") as raised:
            connect_mcp(oauth_server.config(), start_timeout=30)
        assert str(sign_in_path(oauth_server.url)) in str(raised.value)

    def test_where_no_browser_opens_the_address_it_ends_at_is_pasted(
        self, oauth_server: OAuthServer, browser: Browser, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        browser.works = False

        class Pasting(io.StringIO):
            def isatty(self) -> bool:
                return True

            def readline(self, *args: Any) -> str:
                # Where the sign-in page sends the browser, which here is not listened at
                status, location, _ = get(browser.opened[-1])
                assert status in (302, 307) and location is not None
                return f"{location}\n"

        monkeypatch.setattr(sys, "stdin", Pasting())
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
        with connect_mcp(oauth_server.config(), start_timeout=30) as tools:
            assert authorization(tools).startswith("Bearer ")
        said = capsys.readouterr().err
        assert "The MCP server genes needs you to sign in" in said
        assert browser.opened[0] in said
        assert "paste the address your browser ends at" in said

    def test_a_browser_sent_back_here_while_the_address_is_asked_for_is_taken(
        self, oauth_server: OAuthServer, browser: Browser, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        # As over SSH, with the port forwarded: the page is opened elsewhere, and its browser is
        # sent back here
        browser.works = False
        typed = threading.Event()

        class Waiting(io.StringIO):
            def isatty(self) -> bool:
                return True

            def readline(self, *args: Any) -> str:
                status, location, _ = get(browser.opened[-1])
                assert status in (302, 307) and location is not None
                assert get(location)[0] == 200
                typed.wait(10)
                return "\n"

        monkeypatch.setattr(sys, "stdin", Waiting())
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
        try:
            with connect_mcp(oauth_server.config(), start_timeout=30) as tools:
                assert authorization(tools).startswith("Bearer ")
        finally:
            typed.set()
        deadline = time.monotonic() + 10
        said = ""
        while "stopped waiting for the answer" not in said and time.monotonic() < deadline:
            time.sleep(0.05)
            said += capsys.readouterr().err
        assert "stopped waiting for the answer" in said

    def test_a_pasted_address_of_another_sign_in_is_refused(
        self, oauth_server: OAuthServer, browser: Browser, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        browser.works = False

        class Pasting(io.StringIO):
            def isatty(self) -> bool:
                return True

            def readline(self, *args: Any) -> str:
                return "http://127.0.0.1:1/callback?code=stolen&state=another\n"

        monkeypatch.setattr(sys, "stdin", Pasting())
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
        with pytest.raises(MCPServerError, match="The address pasted is not the one signing in"):
            connect_mcp(oauth_server.config(), start_timeout=30)

    def test_a_preset_signed_in_to_is_not_sent_its_api_key(
        self, oauth_server: OAuthServer, browser: Browser, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PAPERCLIP_API_KEY", raising=False)
        changes = {"url": oauth_server.url, "auth": "oauth", "instructions": ""}
        with connect_mcp(presets={"paperclip": changes}, start_timeout=30) as tools:
            seen = {tool.name: tool for tool in tools.tools}["paperclip_authorization"].function()
        assert seen.startswith("Bearer ")
        assert len(browser.opened) == 1


class TestConfig:
    def test_auth_is_oauth_or_left_out(self) -> None:
        with pytest.raises(ValueError, match='The auth of the MCP server s is "oauth"'):
            parse_server("s", {"url": "https://example.org/mcp", "auth": "basic"})
        assert parse_server("s", {"url": "https://example.org/mcp", "auth": "oauth"}).auth == "oauth"
        assert parse_server("s", {"url": "https://example.org/mcp"}).auth is None

    def test_a_server_started_here_is_not_signed_in_to(self) -> None:
        with pytest.raises(ValueError, match="started here, so auth is not used"):
            parse_server("s", {"command": "true", "auth": "oauth"})

    def test_a_server_signed_in_to_is_not_sent_an_authorization_header_of_its_own(self) -> None:
        with pytest.raises(ValueError, match="sent an Authorization header of its own"):
            parse_server(
                "s", {"url": "https://example.org/mcp", "auth": "oauth", "headers": {"authorization": "Bearer x"}}
            )
        assert parse_server(
            "s", {"url": "https://example.org/mcp", "auth": "oauth", "headers": {"X-Other": "y"}}
        ).headers == {"X-Other": "y"}

    @pytest.mark.parametrize(("preset", "header"), [("paperclip", "X-API-Key"), ("adaptyv", "Authorization")])
    def test_a_preset_signed_in_to_needs_no_api_key(
        self, preset: str, header: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for variable in ("PAPERCLIP_API_KEY", "FOUNDRY_API_TOKEN"):
            monkeypatch.delenv(variable, raising=False)
        entry, _ = preset_entry("s", {"preset": preset, "auth": "oauth"})
        assert header not in entry.get("headers", {})
        server = parse_server("s", {"preset": preset, "auth": "oauth"})
        assert server.auth == "oauth"
        assert header not in server.headers
        assert "signed in to" in server.setup and "sign_out_mcp" in server.setup
        # Without auth: oauth, the key is needed, and how to sign in instead is said
        with pytest.raises(ValueError, match="sign in instead, with auth: oauth"):
            parse_server("s", {"preset": preset})

    def test_the_preset_of_protos_hosted_server_is_signed_in_to_and_asks_before_deploying(self) -> None:
        preset = MCP_PRESETS["proto_hosted"]
        server = parse_server("proto_hosted", {"preset": "proto_hosted"})
        assert server.url == "https://mcp.evodesign.org/mcp"
        assert server.auth == "oauth"
        assert "https://proto.evodesign.org" in preset.setup
        assert server.needs_approval("deploy_tool")
        assert not server.needs_approval("run_tool")

    def test_presets_are_given_by_name_or_with_changes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(TypeError, match="The changes to the preset paperclip are a mapping"):
            connect_mcp(presets={"paperclip": "oauth"})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="There is no MCP preset 'nothing'"):
            connect_mcp(presets={"nothing": {}})


class TestStorage:
    def test_the_directory_is_virtual_lab_mcp_auth_dir_or_else_in_the_home_directory(
        self, auth_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert auth_directory() == auth_dir
        monkeypatch.delenv("VIRTUAL_LAB_MCP_AUTH_DIR")
        assert auth_directory() == Path.home() / ".virtual_lab" / "mcp_auth"

    def test_each_server_has_a_file_named_after_its_host(self, auth_dir: Path) -> None:
        one = sign_in_path("https://mcp.example.org/mcp")
        assert one.parent == auth_dir
        assert one.name.startswith("mcp.example.org-") and one.suffix == ".json"
        assert sign_in_path("https://mcp.example.org/other") != one
        assert sign_in_path("http://127.0.0.1:8000/mcp").name.startswith("127.0.0.1_8000-")

    def test_a_file_of_another_servers_or_one_that_cannot_be_read_is_not_used(self) -> None:
        storage = SignInStorage("https://one.example.org/mcp")
        storage.update(tokens={"access_token": "a"})
        assert storage.read()["tokens"] == {"access_token": "a"}
        storage.path.write_text(json.dumps({"server_url": "https://two.example.org/mcp", "tokens": {}}))
        assert storage.read() == {}
        storage.path.write_text("{not json")
        with pytest.warns(UserWarning, match="could not be read, so it will be signed in to again"):
            assert storage.read() == {}

    def test_forgetting_the_client_forgets_its_tokens(self) -> None:
        storage = SignInStorage("https://one.example.org/mcp")
        storage.update(client_info={"client_id": "c"}, tokens={"access_token": "a"}, expires_at=5.0, discovered={})
        assert storage.expires_at == 5.0
        storage.forget_client()
        kept = storage.read()
        assert (kept["client_info"], kept["tokens"], kept["expires_at"], kept["discovered"]) == (None, None, None, None)
        assert storage.expires_at is None
        assert asyncio.run(storage.get_tokens()) is None
        assert asyncio.run(storage.get_client_info()) is None

    def test_what_is_not_a_token_or_a_client_is_not_used(self) -> None:
        storage = SignInStorage("https://one.example.org/mcp")
        storage.update(client_info={"redirect_uris": 5}, tokens={"token_type": "Bearer"})
        assert asyncio.run(storage.get_tokens()) is None
        assert asyncio.run(storage.get_client_info()) is None
        assert storage.discovered() == (None, None)

    def test_the_port_is_the_one_the_client_kept_was_registered_with(self) -> None:
        storage = SignInStorage("https://one.example.org/mcp")
        storage.update(client_info={"client_id": "c", "redirect_uris": ["http://127.0.0.1:43123/callback"]})
        assert callback_port(storage) == 43123
        # One registered for somewhere this cannot receive a sign-in is forgotten
        storage.update(client_info={"client_id": "c", "redirect_uris": ["https://elsewhere.example.org/callback"]})
        assert callback_port(storage) != 43123
        assert storage.stored_client() is None


class TestSignOut:
    def test_a_preset_is_signed_out_of_by_its_name(self) -> None:
        url = MCP_PRESETS["proto_hosted"].entry["url"]
        SignInStorage(url).update(tokens={"access_token": "a"})
        assert sign_out_mcp("proto_hosted") is True
        assert not sign_in_path(url).exists()
        assert sign_out_mcp("proto_hosted") is False

    def test_a_server_started_here_or_what_is_not_a_url_is_refused(self) -> None:
        with pytest.raises(ValueError, match="The preset proto is of a server started here"):
            sign_out_mcp("proto")
        with pytest.raises(ValueError, match="sign_out_mcp is given a preset"):
            sign_out_mcp("genes")


def sign_in(port: int | None = None) -> SignIn:
    signing_in = SignIn("genes", "https://mcp.example.org/mcp", PausedClock())
    if port is not None:
        signing_in.port = port
    return signing_in


async def request(port: int, line: str) -> tuple[int, str]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"{line}\r\nHost: 127.0.0.1\r\n\r\n".encode())
    await writer.drain()
    answer = (await reader.read()).decode()
    writer.close()
    return int(answer.split()[1]), answer


class TestTheAddressSentBackTo:
    def test_only_the_sign_in_waiting_is_taken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: True)
        signing_in = sign_in(free_port())

        async def run() -> None:
            await signing_in.redirect("https://auth.example.org/authorize?state=right&client_id=c")
            assert signing_in.clock.paused
            port = signing_in.port
            assert (await request(port, "GET /elsewhere?state=right&code=x HTTP/1.1"))[0] == 404
            assert (await request(port, "POST /callback?state=right&code=x HTTP/1.1"))[0] == 404
            assert (await request(port, "GET /callback?state=wrong&code=x HTTP/1.1"))[0] == 400
            assert (await request(port, "GET /callback?code=x HTTP/1.1"))[0] == 400
            assert not signing_in.received.done()
            status, page = await request(port, "GET /callback?state=right&code=abc HTTP/1.1")
            assert status == 200 and "You are signed in to genes" in page
            result = await signing_in.callback()
            assert (result.code, result.state) == ("abc", "right")
            assert not signing_in.clock.paused
            assert signing_in.listener is None

        asyncio.run(run())

    def test_a_port_in_use_says_so_and_another_is_used_next_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: True)
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            signing_in = sign_in(taken.getsockname()[1])
            signing_in.storage.update(client_info={"client_id": "c"})
            with pytest.raises(SignInError, match="which is in use"):
                asyncio.run(signing_in.redirect("https://auth.example.org/authorize?state=s"))
        assert signing_in.storage.stored_client() is None
        assert not signing_in.clock.paused

    def test_where_no_browser_opens_nor_is_there_a_terminal_the_listener_is_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: False)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        signing_in = sign_in(free_port())
        with pytest.raises(SignInError, match="needs a browser"):
            asyncio.run(signing_in.redirect("https://auth.example.org/authorize?state=s"))
        assert signing_in.listener is None
        assert not signing_in.clock.paused
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", signing_in.port))

    def test_a_browser_that_cannot_be_opened_is_one_that_did_not_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(url: str, *args: Any, **kwargs: Any) -> bool:
            raise webbrowser.Error("no browser")

        monkeypatch.setattr(webbrowser, "open", broken)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        with pytest.raises(SignInError, match="needs a browser"):
            asyncio.run(sign_in(free_port()).redirect("https://auth.example.org/authorize?state=s"))


class TestTimeSpentSigningIn:
    def test_the_clock_counts_the_time_paused(self) -> None:
        clock = PausedClock()
        assert clock.seconds() == 0.0 and not clock.paused
        clock.pause()
        time.sleep(0.1)
        clock.pause()
        assert clock.paused
        clock.resume()
        clock.resume()
        spent = clock.seconds()
        assert 0.1 <= spent < 1.0
        time.sleep(0.05)
        assert clock.seconds() == spent

    def test_a_wait_is_longer_by_the_time_spent_signing_in(self) -> None:
        connection = MCPConnection.__new__(MCPConnection)
        connection.signing_in = PausedClock()
        future: concurrent.futures.Future[str] = concurrent.futures.Future()

        def sign_in_then_finish() -> None:
            connection.signing_in.pause()
            time.sleep(1.5)
            connection.signing_in.resume()
            time.sleep(0.2)
            future.set_result("done")

        threading.Thread(target=sign_in_then_finish, daemon=True).start()
        assert connection.wait(future, 0.5) == "done"

    def test_a_wait_begun_while_a_person_signs_in_lasts_until_they_have(self) -> None:
        connection = MCPConnection.__new__(MCPConnection)
        connection.signing_in = PausedClock()
        connection.signing_in.pause()
        future: concurrent.futures.Future[str] = concurrent.futures.Future()

        def finish() -> None:
            time.sleep(0.8)
            connection.signing_in.resume()
            future.set_result("done")

        threading.Thread(target=finish, daemon=True).start()
        assert connection.wait(future, 0.0) == "done"

    def test_a_wait_without_signing_in_times_out(self) -> None:
        connection = MCPConnection.__new__(MCPConnection)
        connection.signing_in = PausedClock()
        started = time.monotonic()
        with pytest.raises(concurrent.futures.TimeoutError):
            connection.wait(concurrent.futures.Future(), 0.3)
        assert 0.3 <= time.monotonic() - started < 1.5

    def test_a_future_that_timed_out_itself_is_not_taken_for_one_waited_too_long_for(self) -> None:
        connection = MCPConnection.__new__(MCPConnection)
        connection.signing_in = PausedClock()
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        future.set_exception(TimeoutError("the server's own"))
        with pytest.raises(TimeoutError, match="the server's own"):
            connection.wait(future, 10.0)


def test_the_sign_in_directory_is_not_written_to_until_a_sign_in_is_kept(auth_dir: Path) -> None:
    SignIn("genes", "https://mcp.example.org/mcp", PausedClock())
    assert not auth_dir.exists()
    assert os.environ["VIRTUAL_LAB_MCP_AUTH_DIR"] == str(auth_dir)
