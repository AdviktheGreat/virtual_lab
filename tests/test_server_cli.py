"""Tests of starting the server."""

import socket
import sys
from pathlib import Path
from typing import Any

import pytest
import uvicorn

import virtual_lab.server as server_package
from virtual_lab.server import cli
from virtual_lab.server.cli import address_of, is_loopback, listen, main, serve


def require_ipv6_loopback() -> None:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("This machine has no IPv6 loopback")


class TestAddresses:
    @pytest.mark.parametrize(
        ("host", "loopback"),
        [("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.1.2.3", True)]
        + [("0.0.0.0", False), ("192.168.1.5", False), ("lab.example", False), ("::", False), ("", False)],
    )
    def test_only_this_machine_is_this_machine(self, host: str, loopback: bool) -> None:
        assert is_loopback(host) is loopback

    def test_an_address_names_the_port_that_was_taken(self) -> None:
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            port = taken.getsockname()[1]

            assert address_of("127.0.0.1", taken) == f"http://127.0.0.1:{port}"

    def test_an_address_on_ipv6_is_in_brackets(self) -> None:
        require_ipv6_loopback()

        with listen("::1", 0) as sock:
            assert sock.family == socket.AF_INET6
            assert address_of("::1", sock) == f"http://[::1]:{sock.getsockname()[1]}"


class TestListening:
    def test_the_port_asked_for_is_the_port_taken(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        with listen("127.0.0.1", port) as sock:
            assert sock.getsockname()[1] == port

    def test_a_port_asked_for_that_is_taken_is_an_error_and_not_another_port(self) -> None:
        with listen("127.0.0.1", 0) as taken:
            port = taken.getsockname()[1]

            with pytest.raises(OSError):
                listen("127.0.0.1", port)

    def test_with_no_port_asked_for_the_first_free_one_is_taken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            first = probe.getsockname()[1]
        monkeypatch.setattr(cli, "FIRST_PORT", first)

        with listen("127.0.0.1", None) as one, listen("127.0.0.1", None) as two:
            assert one.getsockname()[1] == first
            assert two.getsockname()[1] > first

    def test_with_no_port_free_it_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with listen("127.0.0.1", 0) as taken:
            monkeypatch.setattr(cli, "FIRST_PORT", taken.getsockname()[1])
            monkeypatch.setattr(cli, "PORTS_TRIED", 1)

            with pytest.raises(OSError):
                listen("127.0.0.1", None)

    def test_a_port_that_is_taken_is_let_go_of_and_the_one_that_is_kept_can_be_taken_again_and_handed_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        made: list[Any] = []
        real = socket.socket

        class Recording(real):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.options: list[tuple[Any, ...]] = []
                self.let_go = False
                made.append(self)

            def setsockopt(self, *args: Any) -> None:
                self.options.append(args)
                super().setsockopt(*args)

            def close(self) -> None:
                self.let_go = True
                super().close()

        with listen("127.0.0.1", 0) as taken:
            monkeypatch.setattr(cli, "FIRST_PORT", taken.getsockname()[1])
            monkeypatch.setattr(cli.socket, "socket", Recording)

            with listen("127.0.0.1", None) as kept:
                pass

        assert len(made) == 2
        failed, chosen = made
        assert chosen is kept
        assert failed.let_go
        assert (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) in chosen.options

    def test_the_socket_that_is_kept_is_handed_on_to_the_server_it_starts(self) -> None:
        with listen("127.0.0.1", 0) as sock:
            assert sock.get_inheritable()

    def test_the_socket_is_listening_so_that_whoever_comes_early_waits(self) -> None:
        with listen("127.0.0.1", 0) as sock:
            with socket.create_connection(sock.getsockname(), timeout=2):
                pass


class RecordingServer:
    """Stands in for uvicorn's server, which would not return."""

    started: list["RecordingServer"] = []

    def __init__(self, config: uvicorn.Config) -> None:
        self.config = config
        self.sockets: list[socket.socket] | None = None
        RecordingServer.started.append(self)

    def run(self, sockets: list[socket.socket] | None = None) -> None:
        self.sockets = sockets


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> type[RecordingServer]:
    RecordingServer.started = []
    monkeypatch.setattr(cli.uvicorn, "Server", RecordingServer)

    return RecordingServer


class TestServing:
    def test_the_server_listens_on_a_socket_it_has_and_prints_the_link_that_signs_you_in(
        self, tmp_path: Path, recording: type[RecordingServer], capsys: pytest.CaptureFixture[str]
    ) -> None:
        pages = tmp_path / "pages"
        pages.mkdir()

        serve(tmp_path / "workspace", token="chosen-token", static_dir=pages, open_browser=False, check_keys=False)

        started = recording.started[0]
        assert started.sockets is not None and len(started.sockets) == 1
        port = started.sockets[0].getsockname()[1]
        assert f"Open http://127.0.0.1:{port}/?token=chosen-token" in capsys.readouterr().out
        # What is asked of the server carries no token to a log, and a page that stays is let go of when it stops
        assert started.config.access_log is False
        assert started.config.log_level == "warning"
        assert started.config.timeout_graceful_shutdown == cli.SHUTDOWN_GRACE_SECONDS
        started.sockets[0].close()

    def test_what_the_conversations_are_to_ask_with_is_handed_to_them(
        self, tmp_path: Path, recording: type[RecordingServer]
    ) -> None:
        client, models = object(), object()

        serve(tmp_path / "ws", token="t", open_browser=False, check_keys=False, client=client, chat_models=models)
        serve(tmp_path / "ws", token="t", open_browser=False)

        given, plain = (started.config.app.state.conversations for started in recording.started)
        assert (given.check_keys, given.client, given.chat_models) == (False, client, models)
        assert (plain.check_keys, plain.client, plain.chat_models) == (True, None, None)
        for started in recording.started:
            assert started.sockets is not None
            started.sockets[0].close()

    def test_a_token_is_made_when_none_is_given(
        self, tmp_path: Path, recording: type[RecordingServer], capsys: pytest.CaptureFixture[str]
    ) -> None:
        pages = tmp_path / "pages"
        pages.mkdir()

        serve(tmp_path / "ws", static_dir=pages, open_browser=False, check_keys=False)
        serve(tmp_path / "ws", static_dir=pages, open_browser=False, check_keys=False)

        links = [line.split("?token=")[1] for line in capsys.readouterr().out.splitlines() if "?token=" in line]
        assert len(links) == 2 and links[0] != links[1] and all(len(token) >= 32 for token in links)
        for started in recording.started:
            assert started.sockets is not None
            started.sockets[0].close()

    def test_without_pages_it_says_it_has_only_its_api_and_opens_nothing(
        self,
        tmp_path: Path,
        recording: type[RecordingServer],
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        opened: list[Any] = []
        monkeypatch.setattr(cli.threading, "Timer", lambda *args, **kwargs: opened.append(args) or None)
        monkeypatch.setattr(cli, "PACKAGED_PAGES", tmp_path / "none-built")

        serve(tmp_path / "ws", token="t", open_browser=True, check_keys=False)

        output = capsys.readouterr().out
        assert "no pages to show" in output and "/api" in output
        assert "token=" not in output
        assert opened == []
        assert recording.started[0].sockets is not None
        recording.started[0].sockets[0].close()

    def test_the_page_is_opened_in_a_browser_when_there_is_one_to_show(
        self, tmp_path: Path, recording: type[RecordingServer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pages = tmp_path / "pages"
        pages.mkdir()
        timers: list[Any] = []

        class FakeTimer:
            def __init__(self, interval: float, function: Any, args: tuple[Any, ...]) -> None:
                timers.append((function, args))

            def start(self) -> None:
                pass

        monkeypatch.setattr(cli.threading, "Timer", FakeTimer)

        serve(tmp_path / "ws", token="t", static_dir=pages, open_browser=True, check_keys=False)
        serve(tmp_path / "ws", token="t", static_dir=pages, open_browser=False, check_keys=False)

        assert len(timers) == 1
        function, args = timers[0]
        assert function is cli.webbrowser.open
        assert args[0].endswith("/?token=t")
        for started in recording.started:
            assert started.sockets is not None
            started.sockets[0].close()

    def test_the_pages_the_install_has_are_the_pages_shown(
        self, tmp_path: Path, recording: type[RecordingServer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        packaged = tmp_path / "packaged"
        packaged.mkdir()
        (packaged / "index.html").write_text("packaged")
        monkeypatch.setattr(cli, "PACKAGED_PAGES", packaged)

        serve(tmp_path / "ws", token="t", open_browser=False, check_keys=False)

        app = recording.started[0].config.app
        from fastapi.testclient import TestClient

        with TestClient(app, base_url="http://localhost", headers={"Authorization": "Bearer t"}) as opened:
            assert opened.get("/").text == "packaged"
        assert recording.started[0].sockets is not None
        recording.started[0].sockets[0].close()

    def test_a_server_that_listens_beyond_this_machine_must_be_told_what_it_is_reached_by(
        self, tmp_path: Path, recording: type[RecordingServer]
    ) -> None:
        with pytest.raises(ValueError, match="allowed hosts"):
            serve(tmp_path / "ws", host="0.0.0.0", open_browser=False)

        assert recording.started == []

    def test_a_server_that_listens_beyond_this_machine_says_so_and_knows_its_names(
        self, tmp_path: Path, recording: type[RecordingServer], capsys: pytest.CaptureFixture[str]
    ) -> None:
        serve(
            tmp_path / "ws",
            host="0.0.0.0",
            port=0,
            token="t",
            allowed_hosts=["lab.example"],
            open_browser=False,
            check_keys=False,
        )

        assert "listens beyond this machine" in capsys.readouterr().out
        app = recording.started[0].config.app
        from fastapi.testclient import TestClient

        with TestClient(app, base_url="http://lab.example:9000", headers={"Authorization": "Bearer t"}) as named:
            assert named.get("/api/settings").status_code == 200
        with TestClient(app, base_url="http://other.example", headers={"Authorization": "Bearer t"}) as other:
            assert other.get("/api/settings").status_code == 403
        assert recording.started[0].sockets is not None
        recording.started[0].sockets[0].close()


class TestCommandLine:
    def test_the_options_are_passed_on(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        given: dict[str, Any] = {}
        monkeypatch.setattr(cli, "serve", lambda **kwargs: given.update(kwargs))

        main(
            [
                "--workspace",
                str(tmp_path),
                "--host",
                "0.0.0.0",
                "--port",
                "9001",
                "--token",
                "abc",
                "--allowed-host",
                "lab.example",
                "--allowed-host",
                "other.example",
                "--static-dir",
                str(tmp_path / "pages"),
                "--no-browser",
            ]
        )

        assert given == {
            "workspace": str(tmp_path),
            "host": "0.0.0.0",
            "port": 9001,
            "token": "abc",
            "allowed_hosts": ["lab.example", "other.example"],
            "static_dir": str(tmp_path / "pages"),
            "open_browser": False,
        }

    def test_the_defaults_are_this_machine_and_a_port_that_is_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        given: dict[str, Any] = {}
        monkeypatch.setattr(cli, "serve", lambda **kwargs: given.update(kwargs))
        monkeypatch.delenv("VIRTUAL_LAB_SERVER_TOKEN", raising=False)

        main([])

        assert given["host"] == "127.0.0.1"
        assert given["port"] is None
        assert given["token"] is None
        assert given["allowed_hosts"] == []
        assert given["open_browser"] is True

    def test_the_token_may_come_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        given: dict[str, Any] = {}
        monkeypatch.setattr(cli, "serve", lambda **kwargs: given.update(kwargs))
        monkeypatch.setenv("VIRTUAL_LAB_SERVER_TOKEN", "from-the-environment")

        main([])

        assert given["token"] == "from-the-environment"

    @pytest.mark.parametrize("failure", [ValueError("not like this"), OSError("port is taken")])
    def test_what_cannot_be_done_is_said_and_the_exit_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: Exception
    ) -> None:
        def refuse(**kwargs: Any) -> None:
            raise failure

        monkeypatch.setattr(cli, "serve", refuse)

        with pytest.raises(SystemExit) as caught:
            main([])

        assert caught.value.code == 2
        assert str(failure) in capsys.readouterr().err


class TestInstalling:
    def test_the_package_starts_the_server_without_importing_what_it_needs_until_then(self) -> None:
        assert server_package.__all__ == ["main", "serve"]
        assert server_package.launcher() is cli

    def test_without_fastapi_or_uvicorn_it_says_what_to_install(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setitem(sys.modules, "uvicorn", None)
        monkeypatch.delitem(sys.modules, "virtual_lab.server.cli")
        monkeypatch.delattr(server_package, "cli")

        with pytest.raises(ImportError, match=r'pip install "virtual-lab\[server\]"'):
            server_package.launcher()
        with pytest.raises(ImportError, match="FastAPI and Uvicorn"):
            server_package.serve()
        with pytest.raises(SystemExit) as caught:
            server_package.main([])

        assert str(caught.value) == f"virtual-lab-server: {server_package.NEEDS}"

    def test_a_package_that_is_missing_for_another_reason_is_not_hidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "virtual_lab.server.app", None)
        monkeypatch.delitem(sys.modules, "virtual_lab.server.cli")
        monkeypatch.delattr(server_package, "cli")

        with pytest.raises(ModuleNotFoundError):
            server_package.launcher()

    def test_the_command_is_declared(self) -> None:
        import tomllib

        project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())["project"]

        assert project["scripts"]["virtual-lab-server"] == "virtual_lab.server:main"
        assert {requirement.split(">=")[0] for requirement in project["optional-dependencies"]["server"]} == {
            "fastapi",
            "uvicorn",
        }
        assert any("server" in requirement for requirement in project["optional-dependencies"]["dev"])
