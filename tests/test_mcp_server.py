"""Tests for serving tools over MCP: in-process, and as the command, over stdio and HTTP, to real
clients, the MCP SDK's own and connect_mcp."""

import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client, MCPError  # noqa: E402
from mcp.types import INVALID_PARAMS  # noqa: E402

import virtual_lab  # noqa: E402
import virtual_lab.mcp_server as server_module  # noqa: E402
from virtual_lab.constants import MCP_SERVER_TOKEN_VARIABLE  # noqa: E402
from virtual_lab.mcp_server import (  # noqa: E402
    RequireToken,
    build_parser,
    check_served_tools,
    check_token,
    create_mcp_server,
    import_tools,
    is_loopback,
    main,
    make_session,
    serve_mcp,
    server_code_tool,
    server_instructions,
    tool_result,
)
from virtual_lab.mcp_tools import MCPServerError, MCPToolError, connect_mcp  # noqa: E402
from virtual_lab.environment import sandbox_image  # noqa: E402
from virtual_lab.session import DockerSession, LocalSession  # noqa: E402
from virtual_lab.tools import TOOL_REGISTRY, Tool  # noqa: E402

SOURCE = str(Path(virtual_lab.__file__).parent.parent)
TOKEN = "a-token-long-enough-to-pass-1234"


class Found(BaseModel):
    gene: str
    count: int


def raise_value_error() -> None:
    raise ValueError("no such gene")


def raise_system_exit() -> None:
    raise SystemExit(3)


def nap(seconds: float) -> dict[str, float]:
    time.sleep(seconds)
    return {"slept": seconds}


TOOLS = (
    Tool(
        "add",
        "Adds two integers.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer", "default": 1}},
            "required": ["a"],
        },
        lambda a, b=1: a + b,
    ),
    Tool("lookup", "Looks up a gene.", {"type": "object", "properties": {}}, lambda: {"gene": "TP53", "count": 2}),
    Tool("model", "Returns a model.", {"type": "object", "properties": {}}, lambda: Found(gene="MYC", count=1)),
    Tool("listing", "Returns a list.", {"type": "object", "properties": {}}, lambda: ["a", "b"]),
    Tool("text", "Returns text.", {"type": "object", "properties": {}}, lambda: "plain text"),
    Tool("nothing", "Returns None.", {"type": "object", "properties": {}}, lambda: None),
    Tool("fails", "Fails.", {"type": "object", "properties": {}}, raise_value_error),
    Tool("exits", "Exits.", {"type": "object", "properties": {}}, raise_system_exit),
    Tool("nap", "Sleeps.", {"type": "object", "properties": {"seconds": {"type": "number"}}}, nap),
    # Not a schema jsonschema can check, so the tool is called with whatever it is sent
    Tool("loose", "Takes anything.", {"type": "object", "properties": {"x": {"type": 5}}}, lambda **given: given),
)


def run_client(server: Any, work: Any) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await work(client)

    return anyio.run(go)


def call(server: Any, name: str, arguments: dict[str, Any] | None = None) -> Any:
    return run_client(server, lambda client: client.call_tool(name, arguments or {}))


@pytest.fixture(scope="module")
def server() -> Any:
    return create_mcp_server(TOOLS, name="test-lab", instructions="Use these.")


class TestCheckingTools:
    def test_what_is_not_a_tool_is_refused(self) -> None:
        with pytest.raises(TypeError, match="serves Tools, not function"):
            check_served_tools([lambda: None])  # type: ignore[list-item]

    @pytest.mark.parametrize("name", ["", "has space", "slash/name", "x" * 129])
    def test_a_name_mcp_does_not_allow_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="MCP cannot serve a tool called"):
            check_served_tools([Tool(name, "D.", {"type": "object"}, lambda: None)])

    def test_a_name_mcp_allows(self) -> None:
        tool = Tool("ns.tool-name_2", "D.", {"type": "object"}, lambda: None)

        assert check_served_tools([tool]) == (tool,)

    def test_two_tools_of_one_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="add is given twice"):
            check_served_tools([TOOLS[0], TOOLS[0]])


class TestServer:
    def test_the_tools_are_listed_with_their_descriptions_and_parameters(self, server: Any) -> None:
        listed = run_client(server, lambda client: client.list_tools()).tools

        assert [tool.name for tool in listed] == [tool.name for tool in TOOLS]
        assert listed[0].description == "Adds two integers."
        assert listed[0].input_schema == TOOLS[0].parameters

    def test_the_server_is_named_and_says_how_to_use_it(self, server: Any) -> None:
        async def initialize(client: Any) -> Any:
            return client.server_info, client.instructions

        info, instructions = run_client(server, initialize)

        assert info.name == "test-lab"
        assert info.version == virtual_lab.__version__
        assert instructions == "Use these."

    def test_a_call_returns_what_the_tool_returned(self, server: Any) -> None:
        result = call(server, "add", {"a": 2, "b": 3})

        assert not result.is_error
        assert [item.text for item in result.content] == ["5"]
        assert result.structured_content is None

    def test_a_dict_is_the_structured_result_as_well_as_text(self, server: Any) -> None:
        result = call(server, "lookup")

        assert result.structured_content == {"gene": "TP53", "count": 2}
        assert json.loads(result.content[0].text) == {"gene": "TP53", "count": 2}

    def test_a_pydantic_model_is_structured_too(self, server: Any) -> None:
        result = call(server, "model")

        assert result.structured_content == {"gene": "MYC", "count": 1}

    def test_a_list_is_json_text(self, server: Any) -> None:
        result = call(server, "listing")

        assert result.content[0].text == '["a", "b"]'
        assert result.structured_content is None

    def test_text_is_sent_as_it_is(self, server: Any) -> None:
        assert call(server, "text").content[0].text == "plain text"

    def test_a_tool_that_raises_is_reported_as_failed(self, server: Any) -> None:
        result = call(server, "fails")

        assert result.is_error
        assert result.content[0].text == "ValueError: no such gene"

    def test_a_tool_that_exits_is_reported_as_failed_and_the_server_carries_on(self, server: Any) -> None:
        async def both(client: Any) -> Any:
            return await client.call_tool("exits", {}), await client.call_tool("text", {})

        exited, after = run_client(server, both)

        assert exited.is_error
        assert exited.content[0].text == "SystemExit: 3"
        assert after.content[0].text == "plain text"

    def test_arguments_that_do_not_fit_are_refused_without_calling_the_tool(self) -> None:
        called = []
        tool = Tool(
            "strict",
            "D.",
            {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
            lambda n: called.append(n),
        )

        result = call(create_mcp_server([tool]), "strict", {"n": "three"})

        assert result.is_error
        assert result.content[0].text == "Invalid arguments for strict: n: 'three' is not of type 'integer'"
        assert called == []

    def test_every_problem_is_listed(self) -> None:
        tool = Tool(
            "two",
            "D.",
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
                "required": ["a", "b", "c"],
            },
            lambda **given: given,
        )

        text = call(create_mcp_server([tool]), "two", {"a": "x", "b": 1}).content[0].text

        assert text.startswith("Invalid arguments for two: ")
        assert "'c' is a required property" in text
        assert "a: 'x' is not of type 'integer'" in text
        assert "b: 1 is not of type 'string'" in text

    def test_past_ten_problems_the_rest_are_counted(self) -> None:
        names = [f"p{index:02d}" for index in range(13)]
        tool = Tool(
            "many",
            "D.",
            {"type": "object", "properties": {name: {"type": "integer"} for name in names}},
            lambda **given: given,
        )

        text = call(create_mcp_server([tool]), "many", dict.fromkeys(names, "x")).content[0].text

        problems = text.removeprefix("Invalid arguments for many: ").split("; ")
        assert problems[:10] == [f"{name}: 'x' is not of type 'integer'" for name in names[:10]]
        assert problems[10:] == ["and 3 more"]

    def test_a_missing_required_argument_is_refused(self, server: Any) -> None:
        result = call(server, "add", {"b": 1})

        assert result.is_error
        assert "'a' is a required property" in result.content[0].text

    def test_a_schema_that_cannot_check_arguments_leaves_them_to_the_tool(self, server: Any) -> None:
        result = call(server, "loose", {"x": "anything"})

        assert not result.is_error
        assert result.structured_content == {"x": "anything"}

    def test_an_unknown_tool_is_a_protocol_error(self, server: Any) -> None:
        async def unknown(client: Any) -> MCPError:
            with pytest.raises(MCPError) as caught:
                await client.call_tool("missing", {})
            return caught.value

        error = run_client(server, unknown)

        assert error.code == INVALID_PARAMS
        assert error.message == "Unknown tool: missing"

    def test_calls_run_at_the_same_time(self, server: Any) -> None:
        async def three(client: Any) -> float:
            started = time.monotonic()
            async with anyio.create_task_group() as group:
                for _ in range(3):
                    group.start_soon(client.call_tool, "nap", {"seconds": 1.0})
            return time.monotonic() - started

        assert run_client(server, three) < 2.5

    def test_an_unwritable_return_is_written_as_text(self) -> None:
        result = tool_result({"when": object()})

        assert isinstance(result.structured_content, dict)
        assert result.structured_content["when"].startswith("<object object")

    def test_a_long_error_is_shortened(self) -> None:
        def huge() -> None:
            raise RuntimeError("x" * 100_000)

        text = call(create_mcp_server([Tool("huge", "D.", {"type": "object"}, huge)]), "huge").content[0].text

        assert len(text) < 30_000


class TestTokens:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "LocalHost", "::1", "[::1]"])
    def test_loopback_hosts(self, host: str) -> None:
        assert is_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.org", "10.0.0.1"])
    def test_other_hosts(self, host: str) -> None:
        assert not is_loopback(host)

    def test_no_token_is_needed_on_this_machine(self) -> None:
        check_token(None, "127.0.0.1")

    def test_a_token_is_needed_elsewhere(self) -> None:
        with pytest.raises(ValueError, match=f"set {MCP_SERVER_TOKEN_VARIABLE}"):
            check_token(None, "0.0.0.0")

    def test_a_short_token_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 16 characters"):
            check_token("x" * 15, "0.0.0.0")
        check_token("x" * 16, "0.0.0.0")

    @pytest.mark.parametrize("token", ["has a space in it ok", "tab\there-is-long-enough", "naïve-but-long-enough"])
    def test_a_token_that_cannot_be_a_header_is_refused(self, token: str) -> None:
        with pytest.raises(ValueError, match="printable ASCII"):
            check_token(token, "127.0.0.1")

    def test_a_token_is_a_string(self) -> None:
        with pytest.raises(TypeError, match="not bytes"):
            check_token(b"x" * 20, "127.0.0.1")  # type: ignore[arg-type]


def asgi_call(app: Any, scope: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    sent: list[dict[str, Any]] = []
    reached = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["type"])

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    anyio.run(RequireToken(inner, TOKEN), scope, receive, send)

    return sent, bool(reached)


def http_scope(*headers: tuple[bytes, bytes]) -> dict[str, Any]:
    return {"type": "http", "method": "POST", "path": "/mcp", "headers": list(headers)}


class TestLoopbackSecurity:
    @pytest.mark.parametrize(
        ("host", "written"),
        [("127.0.0.2", "127.0.0.2"), ("LOCALHOST", "localhost"), ("0:0:0:0:0:0:0:1", "[0:0:0:0:0:0:0:1]")],
    )
    def test_the_host_listened_at_is_allowed_as_well_as_the_usual_names(self, host: str, written: str) -> None:
        security = server_module.loopback_security(host)

        assert security.enable_dns_rebinding_protection
        for name in ("127.0.0.1", "localhost", "[::1]", written):
            assert f"{name}:*" in security.allowed_hosts
            assert f"http://{name}:*" in security.allowed_origins
        assert not any("evil" in allowed for allowed in security.allowed_hosts + security.allowed_origins)

    def test_each_name_is_listed_once(self) -> None:
        security = server_module.loopback_security("127.0.0.1")

        assert len(security.allowed_hosts) == len(set(security.allowed_hosts)) == 6


class TestRequireToken:
    def test_a_request_with_the_token_is_let_through(self) -> None:
        sent, reached = asgi_call(None, http_scope((b"authorization", f"Bearer {TOKEN}".encode())))

        assert reached and sent == []

    def test_the_scheme_is_read_without_regard_to_case(self) -> None:
        _, reached = asgi_call(None, http_scope((b"authorization", f"bearer {TOKEN}".encode())))

        assert reached

    @pytest.mark.parametrize(
        "headers",
        [
            (),
            ((b"authorization", b"Bearer wrong-token-wrong-token-1234"),),
            ((b"authorization", TOKEN.encode()),),
            ((b"authorization", f"Basic {TOKEN}".encode()),),
            ((b"authorization", f"Bearer {TOKEN}x".encode()),),
            ((b"authorization", f"Bearer {TOKEN[:-1]}".encode()),),
            ((b"authorization", f"Bearer {TOKEN}".encode()), (b"authorization", b"Bearer other")),
        ],
    )
    def test_any_other_request_is_refused(self, headers: tuple) -> None:
        sent, reached = asgi_call(None, http_scope(*headers))

        assert not reached
        assert sent[0]["status"] == 401
        assert (b"www-authenticate", b'Bearer error="invalid_token"') in sent[0]["headers"]
        assert json.loads(sent[1]["body"])["error"] == "unauthorized"

    def test_the_lifespan_is_let_through(self) -> None:
        _, reached = asgi_call(None, {"type": "lifespan"})

        assert reached


class TestServeChecks:
    def test_an_unknown_transport_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Cannot serve over 'sse'"):
            serve_mcp(TOOLS, transport="sse")

    def test_a_token_over_stdio_is_refused(self) -> None:
        with pytest.raises(ValueError, match="A token is for serving over HTTP"):
            serve_mcp(TOOLS, transport="stdio", token=TOKEN)

    def test_listening_elsewhere_without_a_token_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lets other machines call these tools"):
            serve_mcp(TOOLS, transport="http", host="0.0.0.0")

    def test_a_path_must_start_with_a_slash(self) -> None:
        with pytest.raises(ValueError, match="must start with /"):
            serve_mcp(TOOLS, transport="http", path="mcp")

    def test_a_port_in_use_is_reported(self) -> None:
        import socket

        with socket.create_server(("127.0.0.1", 0)) as taken:
            port = taken.getsockname()[1]
            with pytest.raises(OSError, match=f"Cannot listen at 127.0.0.1, port {port}"):
                serve_mcp(TOOLS, transport="http", port=port)


class TestImportTools:
    @pytest.fixture(autouse=True)
    def own_module(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "my_lab_tools.py").write_text(
            textwrap.dedent(
                """
                from types import SimpleNamespace
                from virtual_lab.tools import Tool

                ONE = Tool("one", "One.", {"type": "object", "properties": {}}, lambda: 1)
                TWO = Tool("two", "Two.", {"type": "object", "properties": {}}, lambda: 2)
                BOTH = [ONE, TWO]
                CONNECTED = SimpleNamespace(tools=(ONE, TWO))
                NOT_TOOLS = [1, 2]

                def count_genes(genes: list[str]) -> int:
                    '''Counts genes.

                    :param genes: The genes.
                    '''
                    return len(genes)

                class Holder:
                    inner = ONE
                """
            )
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "path", [path for path in sys.path if path not in ("", str(tmp_path))])
        monkeypatch.delitem(sys.modules, "my_lab_tools", raising=False)

    def test_one_of_virtual_labs_tools_by_name(self) -> None:
        assert import_tools("pubmed_search") == (TOOL_REGISTRY["pubmed_search"],)

    def test_a_tool(self) -> None:
        assert [tool.name for tool in import_tools("my_lab_tools:ONE")] == ["one"]

    def test_a_list_of_tools(self) -> None:
        assert [tool.name for tool in import_tools("my_lab_tools:BOTH")] == ["one", "two"]

    def test_what_connect_mcp_returns(self) -> None:
        assert [tool.name for tool in import_tools("my_lab_tools:CONNECTED")] == ["one", "two"]

    def test_a_function_is_made_a_tool(self) -> None:
        (tool,) = import_tools("my_lab_tools:count_genes")

        assert tool.name == "count_genes"
        assert tool.description == "Counts genes."
        assert tool.function(genes=["TP53", "MYC"]) == 2

    def test_a_dotted_attribute(self) -> None:
        assert [tool.name for tool in import_tools("my_lab_tools:Holder.inner")] == ["one"]

    def test_what_is_not_tools_is_refused(self) -> None:
        with pytest.raises(TypeError, match="is not a Tool, an iterable of Tools, or a function"):
            import_tools("my_lab_tools:NOT_TOOLS")

    def test_a_spec_that_is_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nor module:attribute"):
            import_tools("no_such_tool")

    def test_a_missing_module_is_reported(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            import_tools("no_such_module:TOOLS")


class TestCodeTool:
    def test_it_is_described_for_a_servers_clients(self, tmp_path: Path) -> None:
        with LocalSession(tmp_path, warn=False) as session:
            alone = server_code_tool(session)
            beside = server_code_tool(session, biomni=True)

            assert alone.description.startswith("Run code in the server's shared interpreter")
            assert "for everyone using this server: what was defined before" in alone.description
            assert "meeting" not in alone.description
            assert "and for Biomni's tools, which run in it too" in beside.description
            assert f"The working directory is {session.directory}" in alone.description
            assert alone.function(code="1 + 1").endswith("Output:\n2\n")

    def test_the_instructions_say_where_code_runs(self, tmp_path: Path) -> None:
        with LocalSession(tmp_path, warn=False) as session:
            text = server_instructions(session, biomni=True, code_tool=True)

        assert "Biomni's tool functions run in one session, on the server's machine, with no isolation" in text
        assert f"working directory is {session.directory}" in text
        assert "run_code runs code in the same session" in text
        assert server_instructions(None, biomni=False, code_tool=False) == (
            "Tools from Virtual Lab, an AI research team for science."
        )


class TestMain:
    def test_nothing_to_serve_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exited:
            main([])

        assert exited.value.code == 2
        assert "say what to serve" in capsys.readouterr().err

    def test_a_module_without_biomnis_tools_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exited:
            main(["--code-tool", "--biomni-module", "genetics"])

        assert exited.value.code == 2
        assert "give --biomni-tools too" in capsys.readouterr().err

    def test_a_session_needs_a_directory(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--code-tool"]) == 1
        assert "give its working directory with --directory" in capsys.readouterr().err

    def test_listening_elsewhere_needs_a_token_before_anything_starts(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(MCP_SERVER_TOKEN_VARIABLE, raising=False)
        started = []
        monkeypatch.setattr(server_module, "collect_tools", lambda *arguments: started.append(1))

        assert main(["--database-tools", "--transport", "http", "--host", "0.0.0.0"]) == 1
        assert "every request must carry a token" in capsys.readouterr().err
        assert started == []

    def test_a_missing_env_file_is_reported(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        assert main(["--database-tools", "--env-file", str(tmp_path / "none.env")]) == 1
        assert "There is no .env file" in capsys.readouterr().err

    def test_a_docker_session_is_made_as_asked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VL_TEST_KEY", "set")
        lake = tmp_path / "lake"
        lake.mkdir()
        arguments = build_parser().parse_args(
            [
                "--code-tool",
                "--directory",
                str(tmp_path / "work"),
                "--image-target",
                "bio",
                "--data-lake",
                str(lake),
                "--no-network",
                "--forward-env",
                "VL_TEST_KEY",
                "--timeout",
                "12",
            ]
        )

        session = make_session(arguments)

        assert isinstance(session, DockerSession)
        assert not session.running
        assert session.executor.image == sandbox_image("bio")
        assert session.executor.data_lake == lake
        assert session.executor.allow_network is False
        assert session.executor.forward_env == ("VL_TEST_KEY",)
        assert session.timeout == 12
        session.close()

    def test_a_docker_session_has_the_network_and_the_full_image_by_default(self, tmp_path: Path) -> None:
        session = make_session(build_parser().parse_args(["--code-tool", "--directory", str(tmp_path)]))

        assert isinstance(session, DockerSession)
        assert session.executor.image == sandbox_image("full")
        assert session.executor.allow_network is True
        session.close()

    def test_a_local_session_is_made_as_asked(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        arguments = build_parser().parse_args(
            [
                "--biomni-tools",
                "--session",
                "local",
                "--directory",
                str(tmp_path),
                "--python",
                sys.executable,
                "--forward-env",
                "VL_TEST_KEY",
                "--timeout",
                "7",
            ]
        )

        session = make_session(arguments)

        assert isinstance(session, LocalSession)
        assert session.python == sys.executable
        assert session.biomni_tools is True
        assert session.forward_env == ("VL_TEST_KEY",)
        assert session.timeout == 7
        assert "with no isolation" in capsys.readouterr().err
        session.close()

    def test_the_session_is_closed_if_serving_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sessions = []
        original = server_module.make_session

        def make_session(arguments: Any) -> Any:
            sessions.append(original(arguments))
            return sessions[-1]

        def fail(*arguments: Any, **options: Any) -> None:
            raise OSError("Cannot listen")

        monkeypatch.setattr(server_module, "make_session", make_session)
        monkeypatch.setattr(server_module, "serve_mcp", fail)

        assert main(["--code-tool", "--session", "local", "--directory", str(tmp_path)]) == 1
        assert sessions and sessions[0]._closed

    def test_the_session_is_closed_if_biomnis_tools_cannot_be_made(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sessions = []
        original = server_module.make_session

        def make_session(arguments: Any) -> Any:
            sessions.append(original(arguments))
            return sessions[-1]

        monkeypatch.setattr(server_module, "make_session", make_session)

        code = main(["--biomni-tools", "--biomni-module", "genomes", "--session", "local", "--directory", str(tmp_path)])

        assert code == 1
        assert "no tool module 'genomes'" in capsys.readouterr().err
        assert sessions[0]._closed

    def test_the_handler_for_sigterm_is_put_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(signum: int, frame: Any) -> None:
            pass

        before = signal.signal(signal.SIGTERM, handler)
        monkeypatch.setattr(server_module, "serve_mcp", lambda *arguments, **options: None)

        try:
            assert main(["--database-tools"]) == 0
            assert signal.getsignal(signal.SIGTERM) is handler
        finally:
            signal.signal(signal.SIGTERM, before)

    def test_sigterm_stops_serving_as_an_interrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def serve(*arguments: Any, **options: Any) -> None:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)

        monkeypatch.setattr(server_module, "serve_mcp", serve)

        assert main(["--database-tools"]) == 130

    def test_setup_writes_nothing_to_standard_output_over_stdio(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def collect(arguments: Any, parser: Any) -> Any:
            print("chatter while setting up")
            return [TOOLS[0]], None

        monkeypatch.setattr(server_module, "collect_tools", collect)
        monkeypatch.setattr(server_module, "serve_mcp", lambda *arguments, **options: None)

        assert main(["--database-tools"]) == 0
        written = capsys.readouterr()
        assert written.out == ""
        assert "chatter while setting up" in written.err

    def test_a_warning_is_shown_as_the_commands_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shown = []

        def collect(arguments: Any, parser: Any) -> Any:
            shown.append(warnings.formatwarning("Left out a module", UserWarning, "toolbox.py", 1))
            return [TOOLS[0]], None

        def formatwarning(*arguments: Any, **options: Any) -> str:
            return "as it was\n"

        monkeypatch.setattr(server_module, "collect_tools", collect)
        monkeypatch.setattr(server_module, "serve_mcp", lambda *arguments, **options: None)
        monkeypatch.setattr(warnings, "formatwarning", formatwarning)

        assert main(["--database-tools"]) == 0
        assert shown == ["virtual-lab-mcp: warning: Left out a module\n"]
        assert warnings.formatwarning is formatwarning

    def test_the_database_tools_are_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served = {}

        def serve(tools: Any, **options: Any) -> None:
            served.update(tools=tools, **options)

        monkeypatch.setattr(server_module, "serve_mcp", serve)
        monkeypatch.setenv(MCP_SERVER_TOKEN_VARIABLE, TOKEN)

        assert main(["--database-tools", "--tool", "pubmed_search", "--name", "mine"]) == 0
        assert [tool.name for tool in served["tools"]] == list(TOOL_REGISTRY)
        assert served["name"] == "mine"
        assert served["transport"] == "stdio"
        # A token is for HTTP, and is not passed over stdio
        assert served["token"] is None


class TestPackage:
    def test_the_command_is_installed(self) -> None:
        import tomllib

        with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as file:
            scripts = tomllib.load(file)["project"]["scripts"]

        assert scripts["virtual-lab-mcp"] == "virtual_lab.mcp_server:main"

    def test_serving_is_exported_from_the_package(self) -> None:
        assert virtual_lab.create_mcp_server is create_mcp_server
        assert virtual_lab.serve_mcp is serve_mcp
        assert "create_mcp_server" in virtual_lab.__all__ and "serve_mcp" in virtual_lab.__all__

    def test_another_name_is_still_missing(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
            virtual_lab.no_such_name  # noqa: B018

    def test_running_the_module_warns_of_nothing(self, tmp_path: Path) -> None:
        ran = subprocess.run(
            [sys.executable, "-W", "error", "-m", "virtual_lab.mcp_server", "--help"],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": SOURCE},
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert ran.returncode == 0, ran.stderr
        assert ran.stderr == ""
        assert ran.stdout.startswith("usage: virtual-lab-mcp")


class ServerProcess:
    """The command, run as a client would run it, with what it writes to stderr read as it comes."""

    def __init__(self, arguments: list[str], cwd: Path, **environment: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "virtual_lab.mcp_server", *arguments],
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": SOURCE, **environment},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.lines: queue.Queue = queue.Queue()
        self.seen: list[str] = []
        threading.Thread(target=self.read, daemon=True).start()

    def read(self) -> None:
        for line in self.process.stderr:  # type: ignore[union-attr]
            self.lines.put(line)
        self.lines.put(None)

    def wait_for(self, pattern: str, timeout: float = 120) -> re.Match:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                break
            self.seen.append(line)
            if match := re.search(pattern, line):
                return match
        raise AssertionError(f"The server never wrote {pattern!r}: {''.join(self.seen)}")

    def stop(self) -> int:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            return self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return self.process.wait()


@pytest.fixture
def started(tmp_path: Path):
    processes: list[ServerProcess] = []

    def start(*arguments: str, **environment: str) -> ServerProcess:
        processes.append(ServerProcess(list(arguments), tmp_path, **environment))
        return processes[-1]

    yield start
    for process in processes:
        process.stop()


def stdio_config(tmp_path: Path, *arguments: str) -> dict[str, Any]:
    return {
        "mcpServers": {
            "lab": {
                "command": sys.executable,
                "args": ["-m", "virtual_lab.mcp_server", *arguments],
                "env": {"PYTHONPATH": SOURCE},
                "cwd": str(tmp_path),
            }
        }
    }


def kernel_pid(run_code: Tool) -> int:
    return int(run_code.function(code="import os\nprint(os.getpid())").split("Output:\n")[1])


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestOverStdio:
    def test_biomnis_tools_and_the_code_tool_share_one_session(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        config = stdio_config(
            tmp_path,
            "--biomni-tools",
            "--biomni-module",
            "protocols",
            "--code-tool",
            "--session",
            "local",
            "--directory",
            str(work),
        )

        with connect_mcp(config) as tools:
            by_name = {tool.name: tool for tool in tools.tools}
            assert set(by_name) == {
                "lab_search_protocols",
                "lab_get_protocol_details",
                "lab_list_local_protocols",
                "lab_read_local_protocol",
                "lab_run_code",
            }
            assert by_name["lab_get_protocol_details"].parameters["properties"]["protocol_id"]["type"] == "integer"

            listed = by_name["lab_list_local_protocols"].function(source="addgene")
            assert listed["protocols"]
            assert {protocol["source"] for protocol in listed["protocols"]} == {"addgene"}

            # The function's result is the session's last value, there for code to use
            report = by_name["lab_run_code"].function(code="len(_['protocols'])")
            assert report.endswith(f"Output:\n{len(listed['protocols'])}\n")

            with pytest.raises(MCPToolError, match="Invalid arguments for read_local_protocol"):
                by_name["lab_read_local_protocol"].function()
            with pytest.raises(MCPToolError, match="read_local_protocol failed: FileNotFoundError"):
                by_name["lab_read_local_protocol"].function(filename="missing.txt")

            by_name["lab_run_code"].function(code="open('note.txt', 'w').write('kept')")
            pid = kernel_pid(by_name["lab_run_code"])

        assert (work / "note.txt").read_text() == "kept"
        deadline = time.monotonic() + 10
        while process_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not process_exists(pid)

    def test_tools_of_your_own_are_served(self, tmp_path: Path) -> None:
        (tmp_path / "own_tools.py").write_text(
            textwrap.dedent(
                """
                print("imported, and printing to standard output")

                def shout(text: str) -> str:
                    '''Shouts.

                    :param text: What to shout.
                    '''
                    print("this print must not reach the client")
                    return text.upper()
                """
            )
        )

        with connect_mcp(stdio_config(tmp_path, "--tool", "own_tools:shout")) as tools:
            (shout,) = tools.tools
            assert shout.name == "lab_shout"
            assert shout.function(text="tp53") == "TP53"

    def test_one_tool_is_one_tool(self, started: Any) -> None:
        server = started("--tool", "pubmed_search")

        server.wait_for("^Serving 1 tool over stdio$")
        assert server.process.wait(timeout=60) == 0

    def test_a_module_that_does_not_import_is_left_out_with_a_warning(self, started: Any, tmp_path: Path) -> None:
        # genomics needs esm, which the test environment does not have
        server = started(
            "--biomni-tools",
            "--biomni-module",
            "protocols",
            "--biomni-module",
            "genomics",
            "--session",
            "local",
            "--directory",
            str(tmp_path / "work"),
        )
        server.wait_for("Serving 4 tools over stdio")

        # Its standard input is empty, so the server stops as it would when its client goes
        assert server.process.wait(timeout=60) == 0
        assert [line.rstrip("\n") for line in server.seen[1:]] == [
            "virtual-lab-mcp: warning: Left out the functions of Biomni's modules that fail to import in the "
            "session: biomni.tool.genomics (ModuleNotFoundError: No module named 'esm')",
            "Serving 4 tools over stdio",
        ]


def has_ipv6_loopback() -> bool:
    try:
        with socket.create_server(("::1", 0), family=socket.AF_INET6):
            return True
    except OSError:
        return False


class TestOverHTTP:
    @pytest.mark.skipif(not has_ipv6_loopback(), reason="needs IPv6's loopback address")
    def test_an_ipv6_address_is_written_in_brackets(self, started: Any) -> None:
        server = started("--database-tools", "--transport", "http", "--host", "::1", "--port", "0")
        url = server.wait_for(r"Serving 14 tools at (http://\[::1\]:\d+/mcp)$")[1]

        with connect_mcp({"mcpServers": {"lab": {"url": url}}}) as tools:
            assert len(tools.tools) == 14

    def test_tools_are_served_at_the_address_written(self, started: Any) -> None:
        server = started("--database-tools", "--transport", "http", "--port", "0")
        url = server.wait_for(r"Serving 14 tools at (http://127\.0\.0\.1:\d+/mcp)")[1]

        with connect_mcp({"mcpServers": {"lab": {"url": url}}}) as tools:
            assert [tool.name for tool in tools.tools] == [f"lab_{name}" for name in TOOL_REGISTRY]

    def test_a_request_naming_another_host_is_refused(self, started: Any) -> None:
        server = started("--database-tools", "--transport", "http", "--port", "0")
        url = server.wait_for(r"at (http://\S+)")[1]
        request = urllib.request.Request(
            url,
            data=b"{}",
            headers={"host": "evil.example", "content-type": "application/json", "accept": "application/json"},
        )

        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=30)

        assert refused.value.code == 421

    @pytest.mark.parametrize(
        "host",
        [
            "LOCALHOST",
            pytest.param(
                "0:0:0:0:0:0:0:1",
                marks=pytest.mark.skipif(not has_ipv6_loopback(), reason="needs IPv6's loopback address"),
            ),
        ],
    )
    def test_every_way_of_writing_this_machine_refuses_other_hosts(self, started: Any, host: str) -> None:
        server = started("--database-tools", "--transport", "http", "--host", host, "--port", "0")
        url = server.wait_for(r"at (http://\S+)")[1]

        def refused(**headers: str) -> int:
            request = urllib.request.Request(
                url,
                data=b"{}",
                headers={"content-type": "application/json", "accept": "application/json", **headers},
            )
            with pytest.raises(urllib.error.HTTPError) as refusal:
                urllib.request.urlopen(request, timeout=30)
            return refusal.value.code

        assert refused(host="evil.example:8000") == 421
        assert refused(origin="http://evil.example") == 403
        with connect_mcp({"mcpServers": {"lab": {"url": url}}}) as tools:
            assert len(tools.tools) == 14

    def test_a_token_is_required_once_set(self, started: Any) -> None:
        server = started(
            "--database-tools", "--transport", "http", "--port", "0", **{MCP_SERVER_TOKEN_VARIABLE: TOKEN}
        )
        url = server.wait_for(r"at (http://\S+)")[1]

        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(urllib.request.Request(url, data=b"{}"), timeout=30)
        assert refused.value.code == 401

        with pytest.raises(MCPServerError):
            connect_mcp({"mcpServers": {"lab": {"url": url}}}, start_timeout=30)

        config = {"mcpServers": {"lab": {"url": url, "headers": {"Authorization": f"Bearer {TOKEN}"}}}}
        with connect_mcp(config) as tools:
            assert len(tools.tools) == len(TOOL_REGISTRY)

    @pytest.mark.parametrize("env_file", ["server.env", ".env"])
    def test_the_token_can_come_from_an_env_file(self, started: Any, tmp_path: Path, env_file: str) -> None:
        (tmp_path / env_file).write_text(f"{MCP_SERVER_TOKEN_VARIABLE}={TOKEN}\n")
        arguments = ["--database-tools", "--transport", "http", "--port", "0"]
        if env_file != ".env":
            arguments += ["--env-file", env_file]

        url = started(*arguments).wait_for(r"at (http://\S+)")[1]

        # Refused without it, so it was read
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(urllib.request.Request(url, data=b"{}"), timeout=30)
        assert refused.value.code == 401
        config = {"mcpServers": {"lab": {"url": url, "headers": {"Authorization": f"Bearer {TOKEN}"}}}}
        with connect_mcp(config) as tools:
            assert len(tools.tools) == len(TOOL_REGISTRY)

    def test_stopping_the_server_stops_its_session(self, started: Any, tmp_path: Path) -> None:
        server = started(
            "--code-tool", "--session", "local", "--directory", str(tmp_path / "work"), "--transport", "http", "--port", "0"
        )
        url = server.wait_for(r"at (http://\S+)")[1]
        assert any("no isolation" in line for line in server.seen)

        with connect_mcp({"mcpServers": {"lab": {"url": url}}}) as tools:
            pid = kernel_pid(tools.tools[0])
            assert process_exists(pid)

        assert server.stop() == 130
        assert not process_exists(pid)
