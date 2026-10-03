"""Tests for the tools of MCP servers, against real servers run over stdio, HTTP, and SSE."""

import asyncio
import atexit
import base64
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
import warnings
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

pytest.importorskip("mcp")

from mcp import MCPError  # noqa: E402
from mcp_types import (  # noqa: E402
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from virtual_lab import LocalSession  # noqa: E402
from virtual_lab.mcp_tools import (  # noqa: E402
    MCPConnection,
    MCPServerConfig,
    MCPServerError,
    MCPToolError,
    MCPTools,
    connect_mcp,
    content_text,
    failure_text,
    parse_server,
    read_config,
    tool_name,
    tool_parameters,
    tool_result,
    wraps_result,
)
from virtual_lab.project import describe_input  # noqa: E402
from virtual_lab.tools import run_tool_calls  # noqa: E402

mcp_tools_module = import_module("virtual_lab.mcp_tools")

SERVER = str(Path(__file__).with_name("mcp_test_server.py"))


def stdio_config(*arguments: str, **entry: Any) -> dict[str, Any]:
    return {"mcp_servers": {"genes": {"command": [sys.executable, SERVER, *arguments], **entry}}}


def call_of(name: str, arguments: dict[str, Any]) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="call_1", type="function", function=Function(name=name, arguments=json.dumps(arguments))
    )


def by_name(tools: MCPTools) -> dict[str, Any]:
    return {tool.name: tool for tool in tools.tools}


def process_running(pid: int) -> bool:
    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        # Not a child of this process, so it can only be asked whether it exists
        pass
    else:
        if finished == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False

    return True


def wait_until_stopped(pid: int, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not process_running(pid):
            return True
        time.sleep(0.05)

    return False


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class HTTPServer:
    """The test server run over HTTP or SSE in a process of its own, which a test can stop and
    start again on the same port."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.port = free_port()
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{'mcp' if self.mode == 'http' else 'sse'}"

    def start(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, SERVER, self.mode, str(self.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
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


@pytest.fixture(scope="module")
def genes() -> Iterator[MCPTools]:
    with connect_mcp(stdio_config(), timeout=10) as tools:
        yield tools


@pytest.fixture
def http_server() -> Iterator[HTTPServer]:
    server = HTTPServer("http")
    server.start()
    yield server
    server.stop()


class TestReadingAConfig:
    def test_biomni_servers_are_read_from_mcp_servers(self) -> None:
        assert read_config({"mcp_servers": {"a": {"command": ["x"]}}}) == {"a": {"command": ["x"]}}

    def test_claude_desktop_servers_are_read_from_mcpServers_beside_other_settings(self) -> None:
        config = {"globalShortcut": "Ctrl+Space", "mcpServers": {"a": {"command": "x"}}}

        assert read_config(config) == {"a": {"command": "x"}}

    def test_a_dict_without_either_key_is_the_servers_themselves(self) -> None:
        assert read_config({"a": {"url": "http://x"}}) == {"a": {"url": "http://x"}}

    def test_servers_under_both_keys_are_refused(self) -> None:
        with pytest.raises(ValueError, match="under both mcp_servers and mcpServers"):
            read_config({"mcp_servers": {}, "mcpServers": {}})

    def test_a_yaml_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp_config.yaml"
        path.write_text('mcp_servers:\n  genes:\n    command: ["python", "-m", "genes"]\n    enabled: true\n')

        assert read_config(path) == {"genes": {"command": ["python", "-m", "genes"], "enabled": True}}

    def test_a_json_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "claude_desktop_config.json"
        path.write_text(json.dumps({"mcpServers": {"genes": {"command": "python", "args": ["-m", "genes"]}}}))

        assert read_config(str(path)) == {"genes": {"command": "python", "args": ["-m", "genes"]}}

    def test_an_empty_yaml_file_or_server_list_has_no_servers(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        none = tmp_path / "none.yaml"
        none.write_text("mcp_servers:\n")

        assert read_config(empty) == {}
        assert read_config(none) == {}

    def test_a_missing_file_says_where_it_was_looked_for(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="There is no MCP config at .*missing.yaml"):
            read_config(tmp_path / "missing.yaml")

    def test_invalid_json_and_yaml_are_refused(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{")
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("mcp_servers: [unclosed")

        with pytest.raises(ValueError, match="is not valid JSON"):
            read_config(bad_json)
        with pytest.raises(ValueError, match="is not valid YAML"):
            read_config(bad_yaml)

    def test_a_config_that_is_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- genes\n")

        with pytest.raises(ValueError, match="must be a mapping of servers, not list"):
            read_config(path)
        with pytest.raises(ValueError, match="as a mapping of each one's name"):
            read_config({"mcp_servers": ["genes"]})
        with pytest.raises(TypeError, match="not int"):
            read_config(3)  # type: ignore[arg-type]


class TestReadingAServer:
    def test_biomnis_command_list_is_the_program_and_its_arguments(self) -> None:
        server = parse_server("genes", {"command": ["python", "-m", "genes"]})

        assert server is not None
        assert (server.transport, server.command, server.args) == ("stdio", "python", ("-m", "genes"))
        assert server.prefix == "genes"

    def test_claudes_command_and_args_are_joined(self) -> None:
        server = parse_server("genes", {"command": "python", "args": ["-m", "genes", 8080, 0.5, True]})

        assert server is not None
        assert (server.command, server.args) == ("python", ("-m", "genes", "8080", "0.5", "true"))

    def test_args_follow_a_command_list(self) -> None:
        server = parse_server("genes", {"command": ["python", "-m"], "args": ["genes"]})

        assert server is not None
        assert (server.command, server.args) == ("python", ("-m", "genes"))

    def test_a_command_with_spaces_is_one_program(self) -> None:
        server = parse_server("genes", {"command": "/Applications/Gene Tools/server"})

        assert server is not None
        assert (server.command, server.args) == ("/Applications/Gene Tools/server", ())

    @pytest.mark.parametrize(
        "entry",
        [{"command": ["x"], "enabled": False}, {"command": ["x"], "disabled": True}],
    )
    def test_a_disabled_server_is_left_out(self, entry: dict[str, Any]) -> None:
        assert parse_server("genes", entry) is None

    def test_enabled_must_be_true_or_false(self) -> None:
        with pytest.raises(TypeError, match="enabled of the MCP server genes must be true or false"):
            parse_server("genes", {"command": ["x"], "enabled": "no"})

    def test_settings_that_are_not_used_are_warned_of(self) -> None:
        with pytest.warns(UserWarning, match="settings that are not used: alwaysAllow, autoApprove"):
            parse_server("genes", {"command": ["x"], "autoApprove": [], "alwaysAllow": []})

    def test_a_description_is_a_setting_that_is_expected(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parse_server("genes", {"command": ["x"], "description": "Gene lookups"})

    def test_a_server_name_that_cannot_start_a_tool_name_is_refused(self) -> None:
        assert parse_server("brave-search", {"command": ["x"]}).prefix == "brave_search"  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="must start with a letter or '_': rename the server"):
            parse_server("1password", {"command": ["x"]})

    @pytest.mark.parametrize("name", ["", 3, None])
    def test_a_server_is_named_by_text(self, name: Any) -> None:
        with pytest.raises(ValueError, match="An MCP server is named by text"):
            parse_server(name, {"command": ["x"]})

    def test_an_entry_must_be_a_mapping(self) -> None:
        with pytest.raises(TypeError, match="is described by a mapping"):
            parse_server("genes", ["python"])

    @pytest.mark.parametrize(
        ("entry", "transport"),
        [
            ({"url": "https://example.org/mcp"}, "http"),
            ({"url": "https://example.org/sse"}, "sse"),
            ({"url": "https://example.org/sse/"}, "sse"),
            ({"url": "https://example.org/mcp", "type": "sse"}, "sse"),
            ({"url": "https://example.org/sse", "type": "http"}, "http"),
            ({"url": "https://example.org/x", "type": "streamable-http"}, "http"),
            ({"url": "https://example.org/x", "type": "streamableHttp"}, "http"),
            ({"url": "https://example.org/x", "transport": "streamable_http"}, "http"),
            ({"httpUrl": "https://example.org/sse"}, "http"),
            ({"command": "x", "type": "stdio"}, "stdio"),
        ],
    )
    def test_how_a_server_is_reached(self, entry: dict[str, Any], transport: str) -> None:
        server = parse_server("genes", entry)

        assert server is not None
        assert server.transport == transport

    @pytest.mark.parametrize(
        ("entry", "error", "message"),
        [
            ({"command": "x", "url": "https://x"}, ValueError, "both a command and a URL"),
            ({"url": "https://x", "httpUrl": "https://y"}, ValueError, "both url and httpUrl"),
            ({}, ValueError, "needs a command to start it, or a url"),
            ({"command": "x", "type": "websocket"}, ValueError, 'type \'websocket\': use "stdio", "http", or "sse"'),
            ({"command": "x", "type": 3}, TypeError, "type of the MCP server genes is text"),
            ({"type": "stdio"}, ValueError, "started by a command, which it does not give"),
            ({"type": "http"}, ValueError, "reached at a URL, which it does not give"),
            ({"command": "x", "headers": {"a": "b"}}, ValueError, "so headers is not used"),
            ({"url": "https://x", "env": {"a": "b"}}, ValueError, "so env is not used"),
            ({"url": "https://x", "cwd": "/"}, ValueError, "so cwd is not used"),
            ({"url": "ftp://example.org"}, ValueError, "not an http:// or https:// URL"),
            ({"url": "https://"}, ValueError, "not an http:// or https:// URL"),
            ({"command": []}, TypeError, "a list of the program and its arguments"),
            ({"command": [""]}, ValueError, "command of the MCP server genes is empty"),
            ({"command": "x", "args": "-m genes"}, TypeError, "args of the MCP server genes are a list"),
            ({"command": ["x", None]}, TypeError, "command of the MCP server genes must be text or a number"),
            ({"command": "x", "cwd": 3}, TypeError, "cwd of the MCP server genes is a path"),
            ({"command": "x", "env": ["A=b"]}, TypeError, "mapping of names to values"),
            ({"command": "x", "env": {"": "b"}}, ValueError, "named by text"),
        ],
    )
    def test_entries_that_say_nothing_usable_are_refused(
        self, entry: dict[str, Any], error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=message.replace("(", r"\(").replace(")", r"\)")):
            parse_server("genes", entry)

    def test_cwd_must_be_a_directory(self, tmp_path: Path) -> None:
        server = parse_server("genes", {"command": "x", "cwd": str(tmp_path)})

        assert server is not None
        assert server.cwd == str(tmp_path)
        with pytest.raises(FileNotFoundError, match="which is not a directory"):
            parse_server("genes", {"command": "x", "cwd": str(tmp_path / "missing")})

    def test_variables_are_replaced_everywhere_a_server_is_described(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")
        monkeypatch.setenv("GENE_PORT", "8080")
        monkeypatch.delenv("GENE_MISSING", raising=False)

        stdio = parse_server(
            "genes",
            {
                "command": ["${GENE_HOME:-/opt}/server", "--token", "${GENE_TOKEN}"],
                "env": {"TOKEN": "${GENE_TOKEN}", "LEVEL": "${GENE_MISSING:-info}", "DEBUG": False, "EMPTY": "${GENE_MISSING:-}"},
            },
        )
        http = parse_server(
            "search",
            {"url": "http://localhost:${GENE_PORT}/mcp", "headers": {"Authorization": "Bearer ${GENE_TOKEN}"}},
        )

        assert stdio is not None and http is not None
        assert (stdio.command, stdio.args) == ("/opt/server", ("--token", "secret-token"))
        assert stdio.env == {"TOKEN": "secret-token", "LEVEL": "info", "DEBUG": "false", "EMPTY": ""}
        assert http.url == "http://localhost:8080/mcp"
        assert http.headers == {"Authorization": "Bearer secret-token"}

    def test_a_server_is_shown_as_the_config_gives_it_so_that_secrets_are_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")

        stdio = parse_server("genes", {"command": ["server", "--token", "${GENE_TOKEN}"], "env": {"T": "${GENE_TOKEN}"}})
        http = parse_server(
            "search", {"url": "https://x.org/mcp?key=${GENE_TOKEN}", "headers": {"Authorization": "${GENE_TOKEN}"}}
        )

        assert stdio is not None and http is not None
        assert stdio.shown == "server --token ${GENE_TOKEN}"
        assert http.shown == "https://x.org/mcp?key=${GENE_TOKEN}"
        assert "secret-token" not in repr(http)
        assert "secret-token" not in repr(stdio)
        program = parse_server("genes", {"command": "/opt/${GENE_TOKEN}/server"})
        assert program is not None and program.command == "/opt/secret-token/server"
        assert "secret-token" not in repr(program)

    def test_the_values_of_variables_are_kept_to_hide_from_messages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")
        monkeypatch.setenv("GENE_HOST", "genes.example.org")
        monkeypatch.setenv("GENE_HOME", str(tmp_path))
        monkeypatch.delenv("GENE_MISSING", raising=False)

        stdio = parse_server(
            "genes",
            {
                "command": ["server", "--token", "${GENE_TOKEN}"],
                "env": {"A": "${GENE_HOST}", "B": "${GENE_MISSING:-x}"},
                "cwd": "${GENE_HOME}",
            },
        )
        http = parse_server(
            "search",
            {"url": "https://${GENE_HOST}/mcp", "headers": {"Authorization": "Bearer ${GENE_TOKEN}"}},
        )

        assert stdio is not None and http is not None
        assert stdio.hidden == {
            "secret-token": "${GENE_TOKEN}",
            str(tmp_path): "${GENE_HOME}",
            "genes.example.org": "${GENE_HOST}",
        }
        assert http.hidden == {"genes.example.org": "${GENE_HOST}", "secret-token": "${GENE_TOKEN}"}
        assert parse_server("genes", {"command": "server"}).hidden == {}  # type: ignore[union-attr]

    def test_a_cwd_that_is_not_a_directory_is_named_as_the_config_gives_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")

        with pytest.raises(FileNotFoundError, match=r"starts in /x/\$\{GENE_TOKEN\}, which is not a directory$"):
            parse_server("genes", {"command": "x", "cwd": "/x/${GENE_TOKEN}"})

    def test_messages_show_a_variable_in_place_of_its_value(self) -> None:
        server = MCPServerConfig(
            name="genes",
            prefix="genes",
            transport="http",
            shown="x",
            hidden={"to/ken =&": "${TOKEN}", "to/ken =&+more": "${LONGER}", "abc": "${SHORT}", "wxyz": "${FOUR}"},
        )

        assert server.redact("a to/ken =& b") == "a ${TOKEN} b"
        assert server.redact("?key=to/ken%20%3D%26&") == "?key=${TOKEN}&"
        assert server.redact("?key=to%2Fken%20%3D%26&") == "?key=${TOKEN}&"
        assert server.redact("?key=to%2Fken+%3D%26&") == "?key=${TOKEN}&"
        assert server.redact("to/ken =&+more") == "${LONGER}"
        # Too short to be a secret, and too likely to be part of something else
        assert server.redact("abc wxyz") == "abc ${FOUR}"

    def test_a_variable_that_is_not_set_is_refused_rather_than_left_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GENE_MISSING", raising=False)

        with pytest.raises(ValueError, match=r"uses \$\{GENE_MISSING\}, which is not set"):
            parse_server("genes", {"command": "x", "env": {"KEY": "${GENE_MISSING}"}})

    def test_a_variable_set_to_nothing_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GENE_EMPTY", "")

        server = parse_server("genes", {"command": "x", "env": {"KEY": "${GENE_EMPTY:-fallback}"}})

        assert server is not None
        assert server.env == {"KEY": ""}

    def test_tools_are_chosen_by_name_or_in_biomnis_form(self) -> None:
        named = parse_server("genes", {"command": "x", "tools": ["lookup", "echo"]})
        with pytest.warns(UserWarning, match="parameters the config lists for the tools of the MCP server genes"):
            biomni = parse_server(
                "genes",
                {
                    "command": "x",
                    "tools": [
                        {"biomni_name": "lookup", "description": "Find a gene.", "parameters": {"symbol": {}}},
                        {"name": "echo"},
                    ],
                },
            )
        every = parse_server("genes", {"command": "x", "tools": []})

        assert named is not None and biomni is not None and every is not None
        assert named.tools == ("lookup", "echo")
        assert (biomni.tools, biomni.descriptions) == (("lookup", "echo"), {"lookup": "Find a gene."})
        assert every.tools is None

    @pytest.mark.parametrize(
        ("tools", "error", "message"),
        [
            ("lookup", TypeError, "are a list of their names"),
            (["lookup", "lookup"], ValueError, "lists its tool lookup twice"),
            ([{"description": "x"}], ValueError, "named by text"),
            ([3], ValueError, "named by text"),
            ([{"name": "lookup", "description": " "}], ValueError, "must be text"),
        ],
    )
    def test_tools_that_are_not_named_are_refused(self, tools: Any, error: type[Exception], message: str) -> None:
        with pytest.raises(error, match=message):
            parse_server("genes", {"command": "x", "tools": tools})


class TestNamingAndDescribing:
    def test_a_tool_is_named_after_its_server_and_itself(self) -> None:
        assert tool_name("genes", "lookup") == "genes_lookup"
        assert tool_name("brave_search", "web-search.v2") == "brave_search_web_search_v2"

    def test_a_long_name_is_shortened_to_one_every_provider_accepts_and_kept_apart(self) -> None:
        first = tool_name("genes", "a" * 80)
        second = tool_name("genes", "a" * 81)

        assert len(first) == len(second) == 64
        assert first != second
        assert first.isidentifier()
        assert first == tool_name("genes", "a" * 80)

    def test_a_name_of_exactly_the_limit_is_kept(self) -> None:
        assert tool_name("genes", "a" * 58) == "genes_" + "a" * 58

    def test_parameters_are_written_out_in_full_without_titles(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "lookupArguments",
            "type": "object",
            "properties": {"region": {"$ref": "#/$defs/Region", "title": "Region"}},
            "required": ["region"],
            "$defs": {"Region": {"title": "Region", "type": "object", "properties": {"start": {"title": "Start", "type": "integer"}}}},
        }

        assert tool_parameters(schema) == {
            "type": "object",
            "properties": {"region": {"type": "object", "properties": {"start": {"type": "integer"}}}},
            "required": ["region"],
        }

    def test_parameters_are_always_an_object_with_properties(self) -> None:
        assert tool_parameters({}) == {"type": "object", "properties": {}}
        assert tool_parameters(None) == {"type": "object", "properties": {}}

    @pytest.mark.parametrize(
        ("schema", "wrapped"),
        [
            ({"type": "object", "properties": {"result": {"type": "string"}}, "title": "echoOutput"}, True),
            ({"type": "object", "properties": {"result": {"type": "string"}}, "title": "find_genesOutput"}, True),
            ({"type": "object", "properties": {"result": {"type": "string"}}, "title": "Answer"}, False),
            ({"type": "object", "properties": {"result": {"type": "string"}}}, False),
            ({"type": "object", "properties": {"result": {}, "other": {}}, "title": "echoOutput"}, False),
            ({"type": "object", "properties": {"x": {}}, "x-fastmcp-wrap-result": True}, True),
            (None, False),
        ],
    )
    def test_a_result_is_unwrapped_only_where_the_server_wrapped_it(self, schema: Any, wrapped: bool) -> None:
        assert wraps_result(SimpleNamespace(name="echo", output_schema=schema)) is wrapped

    def test_content_is_text_with_what_cannot_be_text_described(self) -> None:
        png = base64.b64encode(b"\x89PNG" + bytes(96)).decode()
        content = [
            TextContent(type="text", text="A plot of the expression."),
            ImageContent(type="image", data=png, mime_type="image/png"),
            AudioContent(type="audio", data=base64.b64encode(bytes(10)).decode(), mime_type="audio/wav"),
            ResourceLink(type="resource_link", uri="file:///genes.csv", name="genes.csv", description="All genes"),
            ResourceLink(type="resource_link", uri="file:///bare", name="bare"),
            EmbeddedResource(type="resource", resource=TextResourceContents(uri="file:///notes.txt", text="Notes")),
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(uri="file:///data.bin", blob=base64.b64encode(bytes(2048)).decode()),
            ),
            SimpleNamespace(type="hologram"),
        ]

        assert content_text(content).splitlines() == [
            "A plot of the expression.",
            "[An image (image/png, 100 bytes), which cannot be shown as text]",
            "[A recording (audio/wav, 10 bytes), which cannot be shown as text]",
            "[A resource at file:///genes.csv: genes.csv, All genes]",
            "[A resource at file:///bare: bare]",
            "[file:///notes.txt]",
            "Notes",
            "[A file at file:///data.bin (binary, 2,048 bytes), which cannot be shown as text]",
            "[Content of the type hologram, which cannot be shown as text]",
        ]

    @pytest.mark.parametrize("data", ["not base64!", "abcd efgh"])
    def test_data_that_is_not_base64_has_no_size(self, data: str) -> None:
        image = ImageContent(type="image", data=data, mime_type="image/png")

        assert content_text([image]) == "[An image (image/png, size unknown), which cannot be shown as text]"

    def test_a_structured_result_is_given_where_the_rest_is_text(self) -> None:
        text = [TextContent(type="text", text='{"symbol": "TP53"}')]
        result = SimpleNamespace(is_error=False, structured_content={"symbol": "TP53"}, content=text)

        assert tool_result(result, "lookup", wrapped=False) == {"symbol": "TP53"}

    def test_a_wrapped_value_is_unwrapped_and_an_object_with_a_result_field_is_not(self) -> None:
        text = [TextContent(type="text", text="positive")]
        result = SimpleNamespace(is_error=False, structured_content={"result": "positive"}, content=text)
        two = SimpleNamespace(is_error=False, structured_content={"result": 1, "other": 2}, content=text)

        assert tool_result(result, "test", wrapped=True) == "positive"
        assert tool_result(result, "test", wrapped=False) == {"result": "positive"}
        assert tool_result(two, "test", wrapped=True) == {"result": 1, "other": 2}

    def test_the_text_is_given_where_a_structured_result_would_leave_something_out(self) -> None:
        content = [
            TextContent(type="text", text="A plot"),
            ImageContent(type="image", data=base64.b64encode(bytes(4)).decode(), mime_type="image/png"),
        ]
        result = SimpleNamespace(is_error=False, structured_content={"plotted": True}, content=content)

        assert tool_result(result, "plot", wrapped=False) == (
            "A plot\n[An image (image/png, 4 bytes), which cannot be shown as text]"
        )

    def test_without_a_structured_result_the_text_is_given(self) -> None:
        result = SimpleNamespace(is_error=False, structured_content=None, content=[TextContent(type="text", text="ok")])

        assert tool_result(result, "echo", wrapped=True) == "ok"

    def test_a_failure_raises_what_the_server_said_or_that_it_said_nothing(self) -> None:
        said = SimpleNamespace(is_error=True, structured_content=None, content=[TextContent(type="text", text="No such gene")])
        silent = SimpleNamespace(is_error=True, structured_content={"x": 1}, content=[])

        with pytest.raises(MCPToolError, match="^No such gene$"):
            tool_result(said, "lookup", wrapped=False)
        with pytest.raises(MCPToolError, match="^lookup failed without saying why$"):
            tool_result(silent, "lookup", wrapped=False)

    def test_a_failure_says_what_went_wrong_inside_the_groups_that_hold_it(self) -> None:
        error = ExceptionGroup("outer", [ExceptionGroup("inner", [ConnectionError("refused")]), ValueError("bad")])

        assert failure_text(error) == "ConnectionError: refused; ValueError: bad"
        assert failure_text(ExceptionGroup("x", [OSError("same"), OSError("same")])) == "OSError: same"
        assert failure_text(TimeoutError()) == "TimeoutError"


class FakeClient:
    """A client listing tools in pages, as a server with many tools does."""

    def __init__(self, cursors: list[str | None]) -> None:
        self.cursors = cursors
        self.asked: list[str | None] = []

    async def list_tools(self, cursor: str | None = None) -> Any:
        self.asked.append(cursor)
        page = len(self.asked) - 1
        return SimpleNamespace(tools=[f"tool{page}"], next_cursor=self.cursors[page])


def unconnected(transport: str = "http", **fields: Any) -> MCPConnection:
    server = MCPServerConfig(name="genes", prefix="genes", transport=transport, shown="server", url="https://x", **fields)
    return MCPConnection(server, None, Path("."), None, 1.0)  # type: ignore[arg-type]


class TestListingTools:
    def connection(self) -> MCPConnection:
        return unconnected()

    def test_every_page_is_listed(self) -> None:
        client = FakeClient(["a", "b", None])

        assert asyncio.run(self.connection().list_tools(client)) == ["tool0", "tool1", "tool2"]
        assert client.asked == [None, "a", "b"]

    def test_pages_that_repeat_are_refused(self) -> None:
        with pytest.raises(MCPServerError, match="lists its tools in pages that repeat"):
            asyncio.run(self.connection().list_tools(FakeClient(["a", "b", "a", None])))

    def test_pages_that_never_end_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mcp_tools_module, "MCP_MAX_TOOL_PAGES", 3)

        with pytest.raises(MCPServerError, match="lists its tools in more than 3 pages"):
            asyncio.run(self.connection().list_tools(FakeClient(["a", "b", "c", None])))
        assert asyncio.run(self.connection().list_tools(FakeClient(["a", "b", None]))) == ["tool0", "tool1", "tool2"]


class TestWhatAConnectionKeeps:
    def test_a_call_that_failed_on_an_earlier_connection_says_nothing_of_the_current_one(self) -> None:
        connection = unconnected()
        connection.generation = 2

        connection.lost(1)
        assert connection.stopped is False
        connection.lost(2)
        assert connection.stopped is True

    def test_a_connection_cancelled_before_it_is_made_is_never_made(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"
        connection = unconnected("stdio", command=sys.executable, args=(SERVER, "--pid-file", str(pid_file)))
        holding = mcp_tools_module.Holding()
        holding.cancel()

        asyncio.run(asyncio.wait_for(connection.hold(holding, None), 10))

        with pytest.raises(ConnectionError, match="The connection was closed before it was made"):
            holding.ready.result(0)
        assert not pid_file.exists()

    @pytest.mark.parametrize(
        ("raised", "error", "message", "stopped"),
        [
            (lambda: __import__("anyio").ClosedResourceError(), MCPServerError, "genes stopped before it answered", True),
            (lambda: __import__("anyio").EndOfStream(), MCPServerError, "genes stopped before it answered", True),
            (lambda: MCPError(code=-32602, message="Invalid params"), MCPToolError, "genes refused the call: Invalid params", False),
            (lambda: MCPError(code=-32001, message="timed out"), TimeoutError, "genes did not answer within 1.0 seconds", False),
        ],
    )
    def test_what_goes_wrong_in_a_call_is_said(
        self, raised: Any, error: type[Exception], message: str, stopped: bool
    ) -> None:
        class Refusing:
            async def call_tool(self, name: str, arguments: dict[str, Any], read_timeout_seconds: float | None) -> Any:
                raise raised()

        loop = mcp_tools_module.EventLoop()
        connection = unconnected()
        connection.loop, connection.timeout = loop, 1.0
        connection.client, connection.holding, connection.generation = Refusing(), mcp_tools_module.Holding(), 1
        connection.holding.task = concurrent.futures.Future()
        try:
            with pytest.raises(error, match=message):
                connection.call("lookup", symbol="TP53")
        finally:
            loop.stop()

        assert connection.stopped is stopped

    def test_the_end_of_what_a_server_wrote_is_shown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mcp_tools_module, "MCP_LOG_TAIL_CHARS", 12)
        connection = unconnected("stdio")
        connection.log_path = tmp_path / "server.log"
        connection.log_path.write_text("the start of it, then the end of it\n")

        assert connection.log_tail() == " What it wrote to stderr ends:\n...he end of it"

    def test_what_a_server_wrote_shows_a_variable_in_place_of_its_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mcp_tools_module, "MCP_LOG_TAIL_CHARS", 20)
        connection = unconnected("stdio", hidden={"secret-token-0123456789": "${GENE_TOKEN}"})
        connection.log_path = tmp_path / "server.log"
        connection.log_path.write_text("the key is secret-token-0123456789\n")

        assert connection.log_tail() == " What it wrote to stderr ends:\n...key is ${GENE_TOKEN}"

    @pytest.mark.parametrize(
        ("answer", "message"),
        [
            (
                MCPError(code=-32602, message="No key secret-token"),
                "^The MCP server genes refused the call: No key \\$\\{GENE_TOKEN\\}$",
            ),
            (
                SimpleNamespace(is_error=True, content=[TextContent(type="text", text="Bad URL ?key=secret-token")]),
                "^Bad URL \\?key=\\$\\{GENE_TOKEN\\}$",
            ),
        ],
    )
    def test_a_refused_or_failed_call_shows_a_variable_in_place_of_its_value(self, answer: Any, message: str) -> None:
        class Answering:
            async def call_tool(self, name: str, arguments: dict[str, Any], read_timeout_seconds: float | None) -> Any:
                if isinstance(answer, Exception):
                    raise answer
                return answer

        loop = mcp_tools_module.EventLoop()
        connection = unconnected(hidden={"secret-token": "${GENE_TOKEN}"})
        connection.loop, connection.timeout = loop, 1.0
        connection.client, connection.holding, connection.generation = Answering(), mcp_tools_module.Holding(), 1
        connection.holding.task = concurrent.futures.Future()
        try:
            with pytest.raises(MCPToolError, match=message):
                connection.call("lookup", symbol="TP53")
        finally:
            loop.stop()

    def test_a_connection_the_sdk_found_to_have_ended_is_made_again(self) -> None:
        def ending(closed: Any) -> Any:
            return SimpleNamespace(session=SimpleNamespace(_dispatcher=SimpleNamespace(_closed=closed)))

        class Unentered:
            @property
            def session(self) -> Any:
                raise RuntimeError("Client must be used within an async context manager")

        assert mcp_tools_module.connection_ended(ending(True)) is True
        assert mcp_tools_module.connection_ended(ending(False)) is False
        # Where the SDK no longer says, the connection is taken to be open, as before
        assert mcp_tools_module.connection_ended(SimpleNamespace(session=SimpleNamespace())) is False
        assert mcp_tools_module.connection_ended(SimpleNamespace(session=SimpleNamespace(_dispatcher=object()))) is False
        assert mcp_tools_module.connection_ended(Unentered()) is False


class TestConnecting:
    def test_without_the_sdk_the_error_says_how_to_install_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "mcp", None)

        with pytest.raises(ImportError, match=r'pip install "virtual-lab\[mcp\]"'):
            connect_mcp(stdio_config())

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"timeout": 0}, "timeout must be a number of seconds above zero, or None"),
            ({"timeout": float("inf")}, "timeout must be"),
            ({"timeout": True}, "timeout must be"),
            ({"timeout": "10"}, "timeout must be"),
            ({"start_timeout": None}, "start_timeout must be a number of seconds above zero, not None"),
            ({"start_timeout": -1}, "start_timeout must be"),
        ],
    )
    def test_time_limits_must_be_positive_numbers(self, options: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            connect_mcp(stdio_config(), **options)

    def test_servers_that_are_not_in_the_config_are_refused(self) -> None:
        with pytest.raises(ValueError, match="has no server search, other: it has genes"):
            connect_mcp(stdio_config(), servers=["genes", "search", "other"])

    def test_a_server_asked_for_by_name_must_be_enabled(self) -> None:
        with pytest.raises(ValueError, match="genes is disabled in the config"):
            connect_mcp(stdio_config(enabled=False), servers="genes")

    def test_servers_whose_tools_would_share_names_are_refused(self) -> None:
        config = {"gene-tools": {"command": "x"}, "gene_tools": {"command": "y"}}

        with pytest.raises(ValueError, match="gene-tools and gene_tools would both name their tools gene_tools_"):
            connect_mcp(config)

    def test_a_config_enabling_nothing_has_no_tools(self) -> None:
        with pytest.warns(UserWarning, match="enables no servers"):
            tools = connect_mcp(stdio_config(disabled=True))

        assert (tools.tools, tools.servers) == ((), {})
        tools.close()

    def test_only_the_servers_asked_for_are_started(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"
        config = {
            "genes": {"command": [sys.executable, SERVER]},
            "other": {"command": [sys.executable, SERVER, "--pid-file", str(pid_file)]},
        }

        with connect_mcp(config, servers=["genes"]) as tools:
            assert list(tools.servers) == ["genes"]
        assert not pid_file.exists()


class TestCallingAServerStartedHere:
    def test_its_tools_are_named_described_and_take_what_the_server_declares(self, genes: MCPTools) -> None:
        tools = by_name(genes)

        assert list(genes.servers) == ["genes"]
        assert genes.tools == genes.servers["genes"]
        assert sorted(tools) == sorted(
            [
                "genes_lookup",
                "genes_echo",
                "genes_total",
                "genes_lengths",
                "genes_picture",
                "genes_variable",
                "genes_directory",
                "genes_process",
                "genes_wait",
                "genes_crash",
                "genes_authorization",
                "genes_find_genes_v2",
                "genes_outcome",
            ]
        )
        assert tools["genes_lookup"].description == "Looks a gene up by its symbol.\n\nOnly TP53 is known."
        assert tools["genes_lookup"].parameters == {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        }
        assert tools["genes_echo"].parameters["properties"]["times"] == {"type": "integer", "default": 1}

    def test_a_structured_result_is_given_back_as_it_is(self, genes: MCPTools) -> None:
        tools = by_name(genes)

        assert tools["genes_lookup"].function(symbol="TP53") == {"symbol": "TP53", "length": 393}
        assert tools["genes_lengths"].function(symbols=["TP53", "BRCA1"]) == {"TP53": 4, "BRCA1": 5}

    def test_a_value_the_server_wrapped_is_unwrapped(self, genes: MCPTools) -> None:
        tools = by_name(genes)

        assert tools["genes_echo"].function(text="ab", times=2) == "abab"
        assert tools["genes_total"].function(values=[1, 2, 3]) == 6
        assert tools["genes_find_genes_v2"].function(query="TP") == ["TP53", "TP63", "TP73"]

    def test_an_object_whose_one_field_is_called_result_is_not_unwrapped(self, genes: MCPTools) -> None:
        assert by_name(genes)["genes_outcome"].function() == {"result": "positive"}

    def test_the_servers_defaults_apply_to_what_is_left_out(self, genes: MCPTools) -> None:
        assert by_name(genes)["genes_echo"].function(text="ab") == "ab"

    def test_what_cannot_be_text_is_described(self, genes: MCPTools) -> None:
        assert by_name(genes)["genes_picture"].function() == (
            "[An image (image/png, 18 bytes), which cannot be shown as text]"
        )

    def test_a_tool_that_fails_raises_what_the_server_said(self, genes: MCPTools) -> None:
        with pytest.raises(MCPToolError, match="^Error executing tool lookup: There is no gene BRCA9$"):
            by_name(genes)["genes_lookup"].function(symbol="BRCA9")

    def test_arguments_the_server_refuses_raise_its_reason(self, genes: MCPTools) -> None:
        with pytest.raises(MCPToolError, match="Input should be a valid integer"):
            by_name(genes)["genes_total"].function(values=["many"])

    def test_an_agent_is_told_of_the_result_and_of_a_failure(self, genes: MCPTools) -> None:
        outputs, messages = run_tool_calls(
            [call_of("genes_lookup", {"symbol": "TP53"}), call_of("genes_lookup", {"symbol": "XYZ"})], genes.tools
        )

        assert outputs[0] == '{"symbol": "TP53", "length": 393}'
        assert outputs[1] == 'Error running tool "genes_lookup": MCPToolError: Error executing tool lookup: There is no gene XYZ'
        assert messages[0]["content"] == outputs[0]

    def test_an_argument_called_like_the_calls_own_parameter_is_passed_on(self, genes: MCPTools) -> None:
        # The server's tool is named positionally, so an argument cannot take its place
        assert by_name(genes)["genes_variable"].function(name="HOME") == os.environ.get("HOME", "<unset>")

    def test_calls_run_at_the_same_time(self, genes: MCPTools) -> None:
        import concurrent.futures

        wait = by_name(genes)["genes_wait"]
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(lambda _: wait.function(seconds=0.75), range(4)))

        assert answers == ["waited 0.75"] * 4
        assert time.monotonic() - started < 2.5

    def test_the_server_is_kept_running_from_one_call_to_the_next(self, genes: MCPTools) -> None:
        process = by_name(genes)["genes_process"]

        assert process.function() == process.function()

    def test_a_tool_of_a_project_step_is_described_by_what_the_agents_are_told(self, genes: MCPTools) -> None:
        described = describe_input("tools", by_name(genes)["genes_lookup"])

        assert described == {
            "name": "genes_lookup",
            "description": "Looks a gene up by its symbol.\n\nOnly TP53 is known.",
            "parameters": {"properties": {"symbol": {"type": "string"}}, "required": ["symbol"], "type": "object"},
            "function": "virtual_lab.mcp_tools.MCPConnection.call",
        }

    def test_code_in_a_session_calls_the_tools_by_name(self, genes: MCPTools, tmp_path: Path) -> None:
        with LocalSession(tmp_path, warn=False, tools=genes.tools) as session:
            found = session.run("gene = genes_lookup('TP53')\nprint(gene['length'], genes_find_genes_v2(query='TP7'))")
            failed = session.run("genes_lookup(symbol='XYZ')")

        assert found.error is None
        assert found.output == "393 ['TP73']\n"
        assert failed.error == (
            "HostToolError: genes_lookup failed: MCPToolError: Error executing tool lookup: There is no gene XYZ"
        )


class TestConfiguringAServerStartedHere:
    def test_the_server_gets_its_env_and_only_a_few_of_this_processs_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")
        monkeypatch.setenv("GENE_PRIVATE", "not for the server")

        with connect_mcp(stdio_config(env={"TOKEN": "${GENE_TOKEN}"}, tools=["variable"])) as tools:
            variable = tools.tools[0]

            assert variable.function(name="TOKEN") == "secret-token"
            assert variable.function(name="GENE_PRIVATE") == "<unset>"
            assert variable.function(name="PATH") == os.environ["PATH"]

    def test_the_server_starts_in_its_cwd(self, tmp_path: Path) -> None:
        with connect_mcp(stdio_config(cwd=str(tmp_path), tools=["directory"])) as tools:
            assert Path(tools.tools[0].function()).resolve() == tmp_path.resolve()

    def test_only_the_tools_chosen_are_offered_with_the_descriptions_given(self) -> None:
        with connect_mcp(
            stdio_config(tools=[{"biomni_name": "lookup", "description": "Find a human gene."}, "echo"])
        ) as tools:
            assert [(tool.name, tool.description) for tool in tools.tools] == [
                ("genes_lookup", "Find a human gene."),
                ("genes_echo", "Says the text back."),
            ]

    def test_a_tool_the_server_does_not_have_is_refused_and_the_server_stopped(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"

        with pytest.raises(ValueError, match="The MCP server genes has no tool lookups: it has lookup, echo"):
            connect_mcp(stdio_config("--pid-file", str(pid_file), tools=["lookups"]))

        assert wait_until_stopped(int(pid_file.read_text()))

    def test_tools_whose_names_would_be_the_same_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mcp_tools_module, "tool_name", lambda prefix, name: f"{prefix}_same")

        with pytest.raises(ValueError, match="The tools lookup and echo of the MCP server genes would both be called genes_same"):
            connect_mcp(stdio_config(tools=["lookup", "echo"]))


class TestAServerThatStopsOrCannotStart:
    def test_a_server_that_stops_is_said_to_have_and_is_started_again_with_a_warning(self, tmp_path: Path) -> None:
        with connect_mcp(stdio_config(), timeout=10) as tools:
            calls = by_name(tools)
            first = calls["genes_process"].function()

            with pytest.raises(MCPServerError, match="stopped before it answered. It is started again") as stopped:
                calls["genes_crash"].function()
            assert "the server is crashing" in str(stopped.value)
            assert wait_until_stopped(first)
            stopped_holding = tools.connections[0].holding

            with pytest.warns(UserWarning, match="genes stopped, so it is started again: whatever it held"):
                second = calls["genes_process"].function()
            assert second != first
            # The connection to the server that stopped is closed, rather than left waiting
            assert stopped_holding is not None and stopped_holding.task is not None and stopped_holding.task.done()

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert calls["genes_process"].function() == second

    def test_a_connection_that_ended_by_itself_is_made_again_before_the_call(self) -> None:
        with connect_mcp(stdio_config(tools=["process"])) as tools:
            connection = tools.connections[0]
            process = tools.tools[0]
            first = process.function()
            assert connection.holding is not None and connection.holding.task is not None
            connection.loop.loop.call_soon_threadsafe(connection.holding.stop.set)
            connection.holding.task.result(15)

            with pytest.warns(UserWarning, match="genes stopped, so it is started again"):
                assert process.function() != first

    def test_a_server_that_stops_between_calls_is_started_again_at_the_next_without_it_failing(self) -> None:
        with connect_mcp(stdio_config(tools=["process"]), timeout=10) as tools:
            connection = tools.connections[0]
            process = tools.tools[0]
            first = process.function()
            assert mcp_tools_module.connection_ended(connection.client) is False

            os.kill(first, signal.SIGKILL)
            assert wait_until_stopped(first)
            deadline = time.monotonic() + 15
            while not mcp_tools_module.connection_ended(connection.client) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert mcp_tools_module.connection_ended(connection.client) is True

            with pytest.warns(UserWarning, match="genes stopped, so it is started again"):
                second = process.function()
            assert second != first

    def test_a_server_whose_tools_cannot_be_listed_is_stopped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid_file = tmp_path / "pid"

        async def unlistable(self: MCPConnection, client: Any) -> list[Any]:
            raise RuntimeError("the listing broke")

        monkeypatch.setattr(MCPConnection, "list_tools", unlistable)

        with pytest.raises(MCPServerError, match="Could not start the MCP server genes .*: RuntimeError: the listing broke"):
            connect_mcp(stdio_config("--pid-file", str(pid_file)))

        assert wait_until_stopped(int(pid_file.read_text()))

    def test_a_tool_that_takes_too_long_is_given_up_and_the_server_kept(self) -> None:
        with connect_mcp(stdio_config(tools=["wait", "process"]), timeout=1) as tools:
            wait, process = tools.tools
            before = process.function()
            started = time.monotonic()

            with pytest.raises(TimeoutError, match="The MCP server genes did not answer within 1 seconds"):
                wait.function(seconds=5)

            assert time.monotonic() - started < 4
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert process.function() == before

    def test_a_server_that_exits_says_why_from_its_stderr(self) -> None:
        with pytest.raises(MCPServerError) as failed:
            connect_mcp(stdio_config("die"))

        message = str(failed.value)
        assert message.startswith(f"Could not start the MCP server genes (started with {sys.executable} {SERVER} die): ")
        assert "the test server has no config, so it stops" in message

    def test_what_a_server_that_exits_wrote_shows_a_variable_in_place_of_its_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")

        with pytest.raises(MCPServerError) as failed:
            connect_mcp(stdio_config("--say", "my key is ${GENE_TOKEN}", "die"))

        assert "secret-token" not in str(failed.value)
        assert "What it wrote to stderr ends:\nthe test server is starting\nmy key is ${GENE_TOKEN}\n" in str(failed.value)

    def test_a_tool_that_fails_shows_a_variable_in_place_of_its_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")

        with connect_mcp(stdio_config(env={"TOKEN": "${GENE_TOKEN}"}, tools=["lookup"])) as tools:
            with pytest.raises(MCPToolError, match=r"^Error executing tool lookup: There is no gene \$\{GENE_TOKEN\}$"):
                tools.tools[0].function(symbol="secret-token")

    def test_a_command_that_is_not_there_says_so(self) -> None:
        with pytest.raises(MCPServerError, match="Could not start the MCP server genes .*FileNotFoundError.*It wrote nothing to stderr"):
            connect_mcp({"genes": {"command": "virtual-lab-no-such-server"}})

    def test_a_server_too_slow_to_start_is_stopped(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"
        started = time.monotonic()

        with pytest.raises(MCPServerError, match="did not answer within 1 seconds. What it wrote to stderr ends:\nthe test server is starting"):
            connect_mcp(stdio_config("--pid-file", str(pid_file), "--sleep", "30"), start_timeout=1)

        assert time.monotonic() - started < 15
        assert wait_until_stopped(int(pid_file.read_text()))

    def test_when_one_server_fails_the_others_are_stopped(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"
        config = {
            "genes": {"command": [sys.executable, SERVER, "--pid-file", str(pid_file)]},
            "broken": {"command": [sys.executable, SERVER, "die"]},
        }

        with pytest.raises(MCPServerError, match="Could not start the MCP server broken"):
            connect_mcp(config)

        assert wait_until_stopped(int(pid_file.read_text()))

    def test_every_server_that_fails_is_said_to(self) -> None:
        config = {
            "first": {"command": [sys.executable, SERVER, "die"]},
            "second": {"command": "virtual-lab-no-such-server"},
        }

        with pytest.raises(MCPServerError) as failed:
            connect_mcp(config)

        assert "Could not start the MCP server first" in str(failed.value)
        assert "Could not start the MCP server second" in str(failed.value)


class TestClosing:
    def test_closing_stops_the_server_and_ends_its_tools(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"
        tools = connect_mcp(stdio_config("--pid-file", str(pid_file), tools=["echo"]))
        log_dir = tools.log_dir
        assert log_dir is not None and log_dir.is_dir()

        tools.close()
        tools.close()

        assert wait_until_stopped(int(pid_file.read_text()))
        assert not log_dir.exists()
        assert tools.loop is not None and not tools.loop.thread.is_alive()
        assert repr(tools) == "MCPTools(genes: 1 tools, closed)"
        with pytest.raises(RuntimeError, match="^The MCP server genes was closed: connect to it again with connect_mcp$"):
            tools.tools[0].function(text="x")

    def test_a_with_statement_closes_them(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"

        with connect_mcp(stdio_config("--pid-file", str(pid_file))) as tools:
            assert repr(tools) == "MCPTools(genes: 13 tools)"

        assert tools.closed
        assert wait_until_stopped(int(pid_file.read_text()))

    def test_tools_left_open_are_closed_when_python_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registered: list[Any] = []
        monkeypatch.setattr(atexit, "register", registered.append)
        monkeypatch.setattr(atexit, "unregister", registered.remove)

        tools = connect_mcp(stdio_config(tools=["echo"]))
        assert registered == [tools.close]

        tools.close()
        assert registered == []

    def test_the_tools_keep_working_when_only_they_are_kept(self) -> None:
        import gc

        echo = connect_mcp(stdio_config(tools=["echo"])).tools[0]
        gc.collect()

        assert echo.function(text="still here") == "still here"


class TestServersReachedAtAURL:
    def test_tools_are_called_over_streamable_http_with_the_headers_given(
        self, http_server: HTTPServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret-token")
        config = {"mcpServers": {"remote": {"url": http_server.url, "headers": {"Authorization": "Bearer ${GENE_TOKEN}"}}}}

        with connect_mcp(config, timeout=None) as tools:
            calls = by_name(tools)

            assert calls["remote_lookup"].function(symbol="TP53") == {"symbol": "TP53", "length": 393}
            assert calls["remote_authorization"].function() == "Bearer secret-token"

    def test_tools_are_called_over_sse_with_the_headers_given(self) -> None:
        server = HTTPServer("sse")
        server.start()
        try:
            config = {"remote": {"url": server.url, "headers": {"Authorization": "Bearer sse-token"}}}
            with connect_mcp(config, timeout=None) as tools:
                calls = by_name(tools)

                assert calls["remote_echo"].function(text="over sse") == "over sse"
                assert calls["remote_authorization"].function() == "Bearer sse-token"
        finally:
            server.stop()

    def test_a_lost_connection_is_said_to_be_and_made_again_with_a_warning(self, http_server: HTTPServer) -> None:
        with connect_mcp({"remote": {"url": http_server.url}}, timeout=10) as tools:
            process = by_name(tools)["remote_process"]
            first = process.function()

            http_server.stop()
            # Whether the loss is found by the call, or by the connection before the call is made,
            # depends on which is first; either way the call fails, saying why
            with warnings.catch_warnings(), pytest.raises(MCPServerError) as lost:
                warnings.simplefilter("ignore")
                process.function()
            assert str(lost.value).startswith(
                ("The MCP server remote stopped before it answered. It is connected to again at the next call",
                 f"Could not connect to the MCP server remote (at {http_server.url}): ")
            )

            http_server.start()
            with pytest.warns(UserWarning, match="connection to the MCP server remote was lost, so it is made again"):
                second = process.function()
            assert second != first

    def test_a_server_that_cannot_be_reached_says_so(self) -> None:
        url = f"http://127.0.0.1:{free_port()}/mcp"

        with pytest.raises(MCPServerError, match=f"Could not connect to the MCP server remote \\(at {url}\\): ConnectError"):
            connect_mcp({"remote": {"url": url}})

    def test_a_server_that_refuses_the_connection_shows_a_variable_in_place_of_its_value(
        self, http_server: HTTPServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENE_TOKEN", "secret/token+1")
        # The server speaks streamable HTTP, so it answers 404, with the URL, to a client of SSE
        url = http_server.url.removesuffix("/mcp") + "/sse?key=${GENE_TOKEN}"

        with pytest.raises(MCPServerError) as failed:
            connect_mcp({"remote": {"url": url}})

        message = str(failed.value)
        assert "404" in message
        assert "secret" not in message
        assert message.count("key=${GENE_TOKEN}") == 2
