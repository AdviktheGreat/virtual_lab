"""Tests for asking a person before a tool's call, passing on what a server asks, the presets of
the services' MCP servers, and what the agents are told of using a server's tools."""

import io
import json
import math
import sys
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeClient, text_response

from virtual_lab import approval
from virtual_lab.agent import Agent
from virtual_lab.approval import (
    ApprovalDeclined,
    ApprovalRequest,
    ServerQuestion,
    answer_in_terminal,
    approve_in_terminal,
    choices,
    read_value,
)
from virtual_lab.mcp_presets import MCP_PRESETS, preset_entry
from virtual_lab.project import describe_input
from virtual_lab.run_meeting import hold_meeting, run_meeting
from virtual_lab.tools import Tool, run_tool_calls, tool_instructions_prompt

SERVER = str(Path(__file__).with_name("mcp_test_server.py"))


class Terminal(io.StringIO):
    """What a person types at a terminal, given in advance."""

    def isatty(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return True


class NotATerminal(io.StringIO):
    def isatty(self) -> bool:
        return False


def typing(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    monkeypatch.setattr(sys, "stdin", Terminal("".join(f"{line}\n" for line in lines)))


def request(**fields: Any) -> ApprovalRequest:
    values: dict[str, Any] = {
        "tool": "lab_submit",
        "server": "lab",
        "server_tool": "submit",
        "arguments": {"experiment": "E-1"},
        "description": "Submits an experiment.\n\nIt costs money.",
    }
    return ApprovalRequest(**{**values, **fields})


class TestApprovingAtTheTerminal:
    @pytest.mark.parametrize("reply", ["y", "Y", "yes", " YES "])
    def test_yes_approves(self, monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
        typing(monkeypatch, reply)

        assert approve_in_terminal(request()) is True

    @pytest.mark.parametrize("reply", ["", "n", "no", "sure", "yess"])
    def test_anything_else_declines(self, monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
        typing(monkeypatch, reply)

        assert approve_in_terminal(request()) is False

    def test_input_that_has_ended_declines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdin", Terminal(""))

        assert approve_in_terminal(request()) is False

    def test_the_person_is_shown_the_tool_its_server_what_it_does_and_the_arguments(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        typing(monkeypatch, "n")

        approve_in_terminal(request(arguments={"experiment": "E-1", "copies": 2}))

        shown = capsys.readouterr()
        assert shown.out == ""
        assert "lab_submit, a tool of the MCP server lab, waits for your approval of each call." in shown.err
        assert "Submits an experiment." in shown.err
        assert "It costs money." not in shown.err
        assert json.dumps({"experiment": "E-1", "copies": 2}, indent=2) in shown.err
        assert shown.err.endswith("Allow this call? [y/N] ")

    def test_a_tool_without_a_description_is_shown_without_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        typing(monkeypatch, "n")

        approve_in_terminal(request(description=""))

        assert "approval of each call.\nIt is to be called with:" in capsys.readouterr().err

    def test_long_arguments_are_cut_short(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(approval, "APPROVAL_MAX_ARGUMENT_CHARS", 40)
        typing(monkeypatch, "n")

        approve_in_terminal(request(arguments={"sequence": "M" * 100}))

        shown = capsys.readouterr().err
        assert "M" * 100 not in shown
        full = json.dumps({"sequence": "M" * 100}, indent=2)
        assert f"{full[:40]}\n... and {len(full) - 40:,} more characters" in shown

    def test_without_a_terminal_the_call_is_declined_and_says_why(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdin", NotATerminal("y\n"))

        with pytest.raises(ApprovalDeclined, match="there is no terminal to ask at, so it was not made"):
            approve_in_terminal(request())

    def test_without_stdin_the_call_is_declined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdin", None)

        with pytest.raises(ApprovalDeclined, match="no terminal"):
            approve_in_terminal(request())

    def test_a_closed_stdin_is_no_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        closed = Terminal("y\n")
        closed.close()
        monkeypatch.setattr(sys, "stdin", closed)

        assert approval.terminal_available() is False


def question(**fields: Any) -> ServerQuestion:
    return ServerQuestion(**{"server": "lab", "message": "Go ahead?", **fields})


GO = {"approve": {"type": "boolean", "title": "Approve"}}

ORDER = {
    "name": {"type": "string", "title": "Name", "description": "What to call it"},
    "copies": {"type": "integer", "default": 1},
    "kind": {"type": "string", "enum": ["protein", "dna"]},
    "scale": {"type": "number"},
    "rush": {"type": "boolean"},
}


class TestAnsweringAtTheTerminal:
    def test_without_a_terminal_a_question_is_declined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdin", NotATerminal("y\n"))

        assert answer_in_terminal(question(fields=GO)) is None

    def test_the_server_and_its_question_are_shown(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        typing(monkeypatch, "y")

        answer_in_terminal(question(fields=GO, message="  Deploy it?\n"))

        assert "The MCP server lab asks:\nDeploy it?\n" in capsys.readouterr().err

    @pytest.mark.parametrize(("reply", "approved"), [("y", True), ("yes", True), ("n", False), ("", False)])
    def test_a_yes_or_no_question_is_asked_as_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], reply: str, approved: bool
    ) -> None:
        typing(monkeypatch, reply)

        assert answer_in_terminal(question(fields=GO)) == {"approve": approved}
        assert capsys.readouterr().err.endswith("Approve? [y/N] ")

    def test_a_yes_or_no_question_without_a_title_is_asked_by_its_name(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        typing(monkeypatch, "y")

        assert answer_in_terminal(question(fields={"go": {"type": "boolean"}})) == {"go": True}
        assert capsys.readouterr().err.endswith("go? [y/N] ")

    def test_a_yes_or_no_question_with_input_ended_is_declined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdin", Terminal(""))

        assert answer_in_terminal(question(fields=GO)) is None

    @pytest.mark.parametrize(("reply", "answer"), [("y", {}), ("n", None), ("", None)])
    def test_a_question_without_fields_is_agreed_to_or_not(
        self, monkeypatch: pytest.MonkeyPatch, reply: str, answer: dict[str, Any] | None
    ) -> None:
        typing(monkeypatch, reply)

        assert answer_in_terminal(question()) == answer

    def test_a_question_at_a_web_page_opens_it_once_agreed_to(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opened: list[str] = []
        monkeypatch.setattr(approval.webbrowser, "open", opened.append)
        typing(monkeypatch, "y")

        assert answer_in_terminal(question(url="https://pay.example.org/1")) == {}
        assert opened == ["https://pay.example.org/1"]
        assert "It asks you to go to https://pay.example.org/1\n" in capsys.readouterr().err

    @pytest.mark.parametrize("lines", [["n"], []])
    def test_a_question_at_a_web_page_not_agreed_to_is_declined(
        self, monkeypatch: pytest.MonkeyPatch, lines: list[str]
    ) -> None:
        opened: list[str] = []
        monkeypatch.setattr(approval.webbrowser, "open", opened.append)
        typing(monkeypatch, *lines)

        assert answer_in_terminal(question(url="https://pay.example.org/1")) is None
        assert opened == []

    def test_a_web_page_that_cannot_be_opened_is_left_to_be_opened_by_hand(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(url: str) -> bool:
            raise OSError("no browser")

        monkeypatch.setattr(approval.webbrowser, "open", fail)
        typing(monkeypatch, "y")

        assert answer_in_terminal(question(url="https://pay.example.org/1")) == {}

    def test_a_form_is_filled_in_field_by_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        typing(monkeypatch, "y", "PD-L1 binder", "3", "2", "1.5", "no")

        answer = answer_in_terminal(question(fields=ORDER, required=("name", "kind")))

        assert answer == {"name": "PD-L1 binder", "copies": 3, "kind": "dna", "scale": 1.5, "rush": False}

    def test_a_form_not_agreed_to_is_declined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        typing(monkeypatch, "n", "PD-L1 binder")

        assert answer_in_terminal(question(fields=ORDER)) is None

    def test_a_form_whose_input_ends_is_declined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        typing(monkeypatch, "y", "PD-L1 binder")

        assert answer_in_terminal(question(fields=ORDER)) is None

    def test_a_form_with_input_ended_before_it_is_agreed_to_is_declined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdin", Terminal(""))

        assert answer_in_terminal(question(fields=ORDER)) is None

    def test_an_optional_field_left_empty_takes_its_default_or_is_left_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        typing(monkeypatch, "y", "binder", "", "protein", "", "")

        answer = answer_in_terminal(question(fields=ORDER, required=("name", "kind")))

        assert answer == {"name": "binder", "copies": 1, "kind": "protein"}

    def test_a_required_field_left_empty_or_answered_wrongly_is_asked_again(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fields = {"copies": {"type": "integer"}}
        typing(monkeypatch, "y", "", "three", "3")

        assert answer_in_terminal(question(fields=fields, required=("copies",))) == {"copies": 3}
        assert capsys.readouterr().err.count("That is not an answer this takes.") == 2

    def test_each_field_is_shown_with_what_it_takes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        typing(monkeypatch, "y", "binder", "", "1", "", "")

        answer_in_terminal(question(fields=ORDER, required=("name", "kind")))

        shown = capsys.readouterr().err
        assert "Name\n  What to call it\n  (text): " in shown
        assert "copies\n  (a whole number, or Enter for 1): " in shown
        assert "kind\n  (one of 1. protein, 2. dna): " in shown
        assert "scale\n  (a number, or Enter to leave it out): " in shown
        assert "rush\n  (yes or no, or Enter to leave it out): " in shown

    def test_a_list_of_choices_is_shown_as_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fields = {"assays": {"type": "array", "items": {"enum": ["bli", "spr"]}}}
        typing(monkeypatch, "y", "bli, spr")

        assert answer_in_terminal(question(fields=fields, required=("assays",))) == {"assays": ["bli", "spr"]}
        assert "(a list, separated by commas, of bli, spr): " in capsys.readouterr().err


class TestReadingWhatIsTyped:
    @pytest.mark.parametrize(
        ("text", "schema", "read"),
        [
            ("yes", {"type": "boolean"}, (True, True)),
            ("TRUE", {"type": "boolean"}, (True, True)),
            ("1", {"type": "boolean"}, (True, True)),
            ("no", {"type": "boolean"}, (True, False)),
            ("0", {"type": "boolean"}, (True, False)),
            ("maybe", {"type": "boolean"}, (False, None)),
            ("12", {"type": "integer"}, (True, 12)),
            ("1.5", {"type": "integer"}, (False, None)),
            ("1.5", {"type": "number"}, (True, 1.5)),
            ("nan", {"type": "number"}, (False, float("nan"))),
            ("inf", {"type": "number"}, (False, float("inf"))),
            ("much", {"type": "number"}, (False, None)),
            ("dna", {"type": "string", "enum": ["protein", "dna"]}, (True, "dna")),
            ("1", {"type": "string", "enum": ["protein", "dna"]}, (True, "protein")),
            ("2", {"type": "string", "enum": ["protein", "dna"]}, (True, "dna")),
            ("3", {"type": "string", "enum": ["protein", "dna"]}, (False, None)),
            ("0", {"type": "string", "enum": ["protein", "dna"]}, (False, None)),
            ("rna", {"type": "string", "enum": ["protein", "dna"]}, (False, None)),
            ("b", {"oneOf": [{"const": "a", "title": "A"}, {"const": "b", "title": "B"}]}, (True, "b")),
            ("2", {"enum": [1, 2]}, (True, 2)),
            ("a, ,b", {"type": "array", "items": {"enum": ["a", "b"]}}, (True, ["a", "b"])),
            ("a,c", {"type": "array", "items": {"enum": ["a", "b"]}}, (False, None)),
            ("x, y", {"type": "array"}, (True, ["x", "y"])),
            ("x", {"type": "array", "items": "not a schema"}, (True, ["x"])),
            ("anything", {}, (True, "anything")),
            ("anything", {"type": "string"}, (True, "anything")),
        ],
    )
    def test_a_value_is_read_as_its_field_takes_it(self, text: str, schema: dict[str, Any], read: tuple) -> None:
        ok, value = read_value(text, schema)

        assert ok is read[0]
        if isinstance(read[1], float) and math.isnan(read[1]):
            assert math.isnan(value)
        else:
            assert value == read[1]

    def test_choices_are_listed_from_enum_or_from_options_with_values(self) -> None:
        assert choices({"enum": ["a", "b"]}) == ["a", "b"]
        assert choices({"oneOf": [{"const": 1}, {"const": 2}]}) == [1, 2]
        assert choices({"anyOf": [{"const": "x"}]}) == ["x"]
        assert choices({"oneOf": [{"const": 1}, {"type": "string"}]}) is None
        assert choices({"oneOf": []}) is None
        assert choices({"type": "string"}) is None


def stdio_entry(*arguments: str, **entry: Any) -> dict[str, Any]:
    return {"command": [sys.executable, SERVER, *arguments], **entry}


def connect(entry: dict[str, Any], **options: Any) -> Any:
    from virtual_lab.mcp_tools import connect_mcp

    return connect_mcp({"mcp_servers": {"genes": entry}}, timeout=20, **options)


def by_name(tools: Any) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools.tools}


@pytest.fixture
def mcp_tools() -> Any:
    pytest.importorskip("mcp")
    return import_module("virtual_lab.mcp_tools")


class TestReadingApproval:
    def test_no_approval_has_no_tool_wait(self, mcp_tools: Any) -> None:
        server = mcp_tools.parse_server("genes", stdio_entry())

        assert (server.ask, server.allow) == ((), ())
        assert server.needs_approval("anything") is False

    def test_a_list_names_the_tools_that_wait(self, mcp_tools: Any) -> None:
        server = mcp_tools.parse_server("genes", stdio_entry(approval=["submit", " pay_* "]))

        assert (server.ask, server.allow) == (("submit", "pay_*"), ())
        assert server.needs_approval("submit") is True
        assert server.needs_approval("pay_invoice") is True
        assert server.needs_approval("submitted") is False
        assert server.needs_approval("list") is False

    def test_allow_takes_tools_out_of_those_that_ask(self, mcp_tools: Any) -> None:
        server = mcp_tools.parse_server("genes", stdio_entry(approval={"ask": ["*"], "allow": ["list_*", "whoami"]}))

        assert server.needs_approval("list_experiments") is False
        assert server.needs_approval("whoami") is False
        assert server.needs_approval("whoami_now") is True
        assert server.needs_approval("create") is True

    def test_allow_alone_has_no_tool_wait(self, mcp_tools: Any) -> None:
        server = mcp_tools.parse_server("genes", stdio_entry(approval={"allow": ["*"]}))

        assert server.needs_approval("create") is False

    def test_names_are_matched_with_their_case(self, mcp_tools: Any) -> None:
        server = mcp_tools.parse_server("genes", stdio_entry(approval=["Submit"]))

        assert server.needs_approval("submit") is False
        assert server.needs_approval("Submit") is True

    @pytest.mark.parametrize(
        ("value", "error", "match"),
        [
            ("submit", TypeError, "is a list of the tools that wait for a person's approval, or a mapping"),
            ({"ask": "submit"}, TypeError, "The approval's ask of the MCP server genes lists the names of tools"),
            ({"allow": {"a": 1}}, TypeError, "The approval's allow of the MCP server genes lists"),
            (["submit", ""], ValueError, "lists tools by name, not ''"),
            ([" "], ValueError, "lists tools by name"),
            ([3], ValueError, "lists tools by name, not 3"),
            ({"ask": ["*"], "deny": ["x"]}, ValueError, "says deny, which it does not take: it takes ask and allow"),
        ],
    )
    def test_an_approval_that_is_not_one_is_refused(
        self, mcp_tools: Any, value: Any, error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            mcp_tools.parse_server("genes", stdio_entry(approval=value))

    def test_instructions_are_text(self, mcp_tools: Any) -> None:
        assert mcp_tools.parse_server("genes", stdio_entry(instructions="  Use it.\n")).instructions == "Use it."
        assert mcp_tools.parse_server("genes", stdio_entry(instructions="  ")).instructions is None
        assert mcp_tools.parse_server("genes", stdio_entry()).instructions is None
        with pytest.raises(TypeError, match="The instructions of the MCP server genes are text, not list"):
            mcp_tools.parse_server("genes", stdio_entry(instructions=["Use it."]))

    def test_a_server_at_a_url_takes_approval_and_instructions(self, mcp_tools: Any) -> None:
        server = mcp_tools.parse_server(
            "genes", {"url": "https://genes.example.org/mcp", "approval": ["submit"], "instructions": "Use it."}
        )

        assert server.needs_approval("submit") is True
        assert server.instructions == "Use it."

    def test_approval_and_instructions_are_settings_that_are_used(self, mcp_tools: Any) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mcp_tools.parse_server("genes", stdio_entry(approval=["x"], instructions="Use it."))


class TestPresets:
    def test_paperclip_is_reached_with_its_api_key(self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PAPERCLIP_API_KEY", "gxl_secret_key")

        server = mcp_tools.parse_server("paperclip", {"preset": "paperclip"})

        assert server.url == "https://paperclip.gxl.ai/mcp"
        assert server.transport == "http"
        assert server.headers == {"X-API-Key": "gxl_secret_key"}
        assert server.needs_approval("search") is False
        assert "Paperclip searches and reads" in server.instructions
        assert server.setup == MCP_PRESETS["paperclip"].setup

    @pytest.mark.parametrize(
        ("preset", "url", "variable"),
        [
            ("adaptyv", "https://mcp.adaptyvbio.com/mcp/", "FOUNDRY_API_TOKEN"),
            ("adaptyv_testing", "https://mcp.testing.adaptyvbio.com/mcp/", "FOUNDRY_TESTING_TOKEN"),
        ],
    )
    def test_adaptyv_is_reached_with_its_token_at_the_address_with_its_slash(
        self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch, preset: str, url: str, variable: str
    ) -> None:
        monkeypatch.setenv(variable, "foundry-token")

        server = mcp_tools.parse_server("lab", {"preset": preset})

        assert server.url == url
        assert server.headers == {"Authorization": "Bearer foundry-token"}

    @pytest.mark.parametrize("preset", ["adaptyv", "adaptyv_testing"])
    @pytest.mark.parametrize(
        ("tool", "waits"),
        [
            ("list_experiments", False),
            ("list_results", False),
            ("get_experiment", False),
            ("get_results", False),
            ("get_quote_pdf", False),
            ("cost_estimate", False),
            ("whoami", False),
            ("health", False),
            ("health_db", False),
            ("create_exp", True),
            ("create_experiment", True),
            ("modify_exp", True),
            ("add_sequences", True),
            ("submit_experiment", True),
            ("confirm_experiment_quote", True),
            ("confirm_quote", True),
            ("reject_quote", True),
            ("pay_invoice", True),
            ("attenuate_token", True),
            ("revoke_token", True),
            ("create_webhook", True),
            ("update_webhook", True),
            ("delete_webhook", True),
            ("create_custom_target_request", True),
            ("submit_feedback", True),
            ("a_tool_added_later", True),
        ],
    )
    def test_every_tool_of_adaptyv_waits_for_approval_but_those_that_read(
        self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch, preset: str, tool: str, waits: bool
    ) -> None:
        monkeypatch.setenv("FOUNDRY_API_TOKEN", "token")
        monkeypatch.setenv("FOUNDRY_TESTING_TOKEN", "token")

        assert mcp_tools.parse_server("lab", {"preset": preset}).needs_approval(tool) is waits

    def test_adaptyvs_testing_sandbox_says_it_is_one(self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOUNDRY_API_TOKEN", "token")
        monkeypatch.setenv("FOUNDRY_TESTING_TOKEN", "token")

        production = mcp_tools.parse_server("lab", {"preset": "adaptyv"}).instructions
        testing = mcp_tools.parse_server("lab", {"preset": "adaptyv_testing"}).instructions

        assert "testing sandbox" not in production
        assert testing.startswith(production)
        assert "testing sandbox, where nothing is made or charged for" in testing

    @pytest.mark.parametrize(
        ("tool", "waits"),
        [
            ("workspace_info", False),
            ("list_tools", False),
            ("search_tools", False),
            ("get_tool_schema", False),
            ("get_tool_example", False),
            ("get_tool_info", False),
            ("run_tool", False),
            ("get_deploy_status", False),
            ("list_runs", False),
            ("get_run_status", False),
            ("get_asset", False),
            ("deploy_tool", True),
            ("a_tool_added_later", True),
        ],
    )
    def test_proto_deploys_only_with_approval(self, mcp_tools: Any, tool: str, waits: bool) -> None:
        server = mcp_tools.parse_server("proto", {"preset": "proto"})

        assert server.transport == "stdio"
        assert (server.command, server.args) == ("proto-tools-mcp", ())
        assert server.needs_approval(tool) is waits

    def test_what_the_entry_gives_is_used_in_place_of_the_presets(self, mcp_tools: Any) -> None:
        server = mcp_tools.parse_server(
            "proto", {"preset": "proto", "command": [sys.executable, SERVER], "approval": [], "instructions": "Mine."}
        )

        assert (server.command, server.args) == (sys.executable, (SERVER,))
        assert server.needs_approval("deploy_tool") is False
        assert server.instructions == "Mine."
        assert server.setup == MCP_PRESETS["proto"].setup

    def test_a_preset_can_be_disabled(self, mcp_tools: Any) -> None:
        assert mcp_tools.parse_server("proto", {"preset": "proto", "enabled": False}) is None

    def test_a_preset_that_is_not_one_is_refused_with_those_that_are(self, mcp_tools: Any) -> None:
        with pytest.raises(ValueError, match="uses the preset 'paperclips', which is not one. The presets are: paperclip"):
            mcp_tools.parse_server("papers", {"preset": "paperclips"})
        with pytest.raises(ValueError, match="uses the preset 3"):
            mcp_tools.parse_server("papers", {"preset": 3})

    def test_a_preset_without_its_key_says_how_to_get_one(
        self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PAPERCLIP_API_KEY", raising=False)

        with pytest.raises(mcp_tools.UnsetVariableError) as raised:
            mcp_tools.parse_server("papers", {"preset": "paperclip"})

        assert "uses ${PAPERCLIP_API_KEY}, which is not set" in str(raised.value)
        assert str(raised.value).endswith(MCP_PRESETS["paperclip"].setup)
        assert "https://paperclip.gxl.ai/keys" in str(raised.value)

    def test_a_server_of_ones_own_without_its_variable_says_only_that(
        self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GENE_TOKEN", raising=False)

        with pytest.raises(mcp_tools.UnsetVariableError) as raised:
            mcp_tools.parse_server("genes", {"url": "https://genes.example.org/mcp", "headers": {"X": "${GENE_TOKEN}"}})

        assert str(raised.value).endswith("give a default as ${GENE_TOKEN:-default}")

    def test_other_errors_in_a_preset_are_not_given_its_setup(self, mcp_tools: Any) -> None:
        with pytest.raises(ValueError) as raised:
            mcp_tools.parse_server("proto", {"preset": "proto", "url": "https://proto.example.org/mcp"})

        assert "gives both a command and a URL" in str(raised.value)
        assert MCP_PRESETS["proto"].setup not in str(raised.value)

    def test_a_server_of_ones_own_has_no_setup(self, mcp_tools: Any) -> None:
        assert mcp_tools.parse_server("genes", stdio_entry()).setup is None

    def test_using_a_preset_does_not_change_it(self) -> None:
        entry, preset = preset_entry("lab", {"preset": "adaptyv"})
        entry["approval"]["allow"].append("*")
        entry["headers"]["Authorization"] = "changed"

        assert "*" not in MCP_PRESETS["adaptyv"].entry["approval"]["allow"]
        assert MCP_PRESETS["adaptyv"].entry["headers"]["Authorization"] == "Bearer ${FOUNDRY_API_TOKEN}"
        assert preset is MCP_PRESETS["adaptyv"]

    def test_an_entry_from_a_preset_is_the_presets_with_the_entrys_own_keys_in_place_of_its(self) -> None:
        entry, _ = preset_entry("lab", {"preset": "proto", "cwd": "/work", "approval": ["*"]})

        assert entry == {**MCP_PRESETS["proto"].entry, "cwd": "/work", "approval": ["*"]}

    def test_an_entry_without_a_preset_is_left_as_it_is(self) -> None:
        entry = {"command": "server"}

        assert preset_entry("genes", entry) == (entry, None)

    def test_the_presets_cannot_be_changed(self) -> None:
        with pytest.raises(TypeError):
            MCP_PRESETS["mine"] = MCP_PRESETS["proto"]  # type: ignore[index]


class TestConnectingToPresets:
    def test_neither_a_config_nor_presets_is_refused(self, mcp_tools: Any) -> None:
        with pytest.raises(ValueError, match="connects to the servers of a config, or to presets, or both"):
            mcp_tools.connect_mcp()

    def test_a_preset_that_is_not_one_is_refused(self, mcp_tools: Any) -> None:
        with pytest.raises(ValueError, match="There is no MCP preset 'protos': the presets are paperclip, adaptyv"):
            mcp_tools.connect_mcp(presets=["protos"])

    def test_a_preset_named_like_a_server_of_the_config_is_refused(self, mcp_tools: Any) -> None:
        with pytest.raises(ValueError, match="already has a server proto: give it preset: proto there"):
            mcp_tools.connect_mcp({"mcp_servers": {"proto": stdio_entry()}}, presets=["proto"])

    @pytest.mark.parametrize("presets", ["proto", ["proto"], ("proto",)])
    def test_presets_connect_as_servers_named_after_them(
        self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch, presets: Any
    ) -> None:
        connecting: list[str] = []

        def start(connection: Any) -> None:
            connecting.append(connection.server.name)
            raise mcp_tools.MCPServerError("not started in a test")

        monkeypatch.setattr(mcp_tools.MCPConnection, "start", start)
        with pytest.raises(mcp_tools.MCPServerError):
            mcp_tools.connect_mcp({"mcp_servers": {"genes": stdio_entry()}}, presets=presets)

        assert sorted(connecting) == ["genes", "proto"]

    def test_servers_chooses_among_the_presets_too(self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        connecting: list[str] = []

        def start(connection: Any) -> None:
            connecting.append(connection.server.name)
            raise mcp_tools.MCPServerError("not started in a test")

        monkeypatch.setattr(mcp_tools.MCPConnection, "start", start)
        with pytest.raises(mcp_tools.MCPServerError):
            mcp_tools.connect_mcp({"mcp_servers": {"genes": stdio_entry()}}, servers=["proto"], presets=["proto"])

        assert connecting == ["proto"]

    def test_a_preset_that_cannot_start_says_how_to_set_it_up(self, mcp_tools: Any) -> None:
        entry = {"preset": "proto", "command": [sys.executable, SERVER, "die"]}

        with pytest.raises(mcp_tools.MCPServerError) as raised:
            mcp_tools.connect_mcp({"mcp_servers": {"proto": entry}}, start_timeout=30)

        assert "the test server has no config, so it stops" in str(raised.value)
        assert str(raised.value).endswith(f"\n{MCP_PRESETS['proto'].setup}")

    def test_a_preset_that_does_not_answer_in_time_says_how_to_set_it_up(self, mcp_tools: Any) -> None:
        entry = {"preset": "proto", "command": [sys.executable, SERVER, "--sleep", "30"]}

        with pytest.raises(mcp_tools.MCPServerError) as raised:
            mcp_tools.connect_mcp({"mcp_servers": {"proto": entry}}, start_timeout=1)

        assert "did not answer within 1 seconds" in str(raised.value)
        assert str(raised.value).endswith(f"\n{MCP_PRESETS['proto'].setup}")

    def test_a_server_of_ones_own_that_cannot_start_says_nothing_of_setup(self, mcp_tools: Any) -> None:
        with pytest.raises(mcp_tools.MCPServerError) as raised:
            connect(stdio_entry("die"))

        assert str(raised.value).endswith("the test server has no config, so it stops")

    @pytest.mark.parametrize("option", ["approve", "answer"])
    def test_approve_and_answer_are_functions(self, mcp_tools: Any, option: str) -> None:
        with pytest.raises(TypeError, match=f"{option} is a function, or None to ask at the terminal, not str"):
            mcp_tools.connect_mcp({"mcp_servers": {"genes": stdio_entry()}}, **{option: "yes"})


class Approver:
    """Approves, or not, as told, and keeps what it was asked."""

    def __init__(self, answer: Any = True) -> None:
        self.answer = answer
        self.asked: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> Any:
        self.asked.append(request)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


@pytest.fixture
def approver() -> Approver:
    return Approver()


class TestApprovingCalls:
    @pytest.fixture(autouse=True)
    def needs_mcp(self) -> None:
        pytest.importorskip("mcp")

    def test_an_approved_call_is_made_and_the_approver_is_told_of_it(self, approver: Approver) -> None:
        with connect(stdio_entry(approval=["echo"]), approve=approver) as tools:
            assert by_name(tools)["genes_echo"].function(text="ab", times=2) == "abab"

        assert approver.asked == [
            ApprovalRequest(
                tool="genes_echo",
                server="genes",
                server_tool="echo",
                arguments={"text": "ab", "times": 2},
                description="Says the text back.",
            )
        ]

    def test_a_tool_named_differently_here_is_approved_by_its_name_on_the_server(self, approver: Approver) -> None:
        with connect(stdio_entry(approval=["find-genes.v2"]), approve=approver) as tools:
            assert by_name(tools)["genes_find_genes_v2"].function(query="TP5") == ["TP53"]

        assert [(asked.tool, asked.server_tool) for asked in approver.asked] == [("genes_find_genes_v2", "find-genes.v2")]

    def test_a_tool_that_does_not_wait_is_called_without_asking(self, approver: Approver) -> None:
        with connect(stdio_entry(approval=["echo"]), approve=approver) as tools:
            assert by_name(tools)["genes_total"].function(values=[1, 2]) == 3

        assert approver.asked == []

    @pytest.mark.parametrize("answer", [False, None, "yes", 1])
    def test_a_call_not_approved_is_not_made(self, answer: Any) -> None:
        approver = Approver(answer)
        with connect(stdio_entry(approval=["count"]), approve=approver) as tools:
            count = by_name(tools)["genes_count"]
            with pytest.raises(ApprovalDeclined, match="A person declined this call to genes_count, so it was not made"):
                count.function()
            approver.answer = True

            assert count.function() == 1

    def test_an_approver_can_decline_with_a_reason_of_its_own(self) -> None:
        approver = Approver(ApprovalDeclined("Over budget."))
        with connect(stdio_entry(approval=["count"]), approve=approver) as tools:
            with pytest.raises(ApprovalDeclined, match="^Over budget.$"):
                by_name(tools)["genes_count"].function()

    def test_an_approver_that_fails_makes_the_call_fail_unmade(self) -> None:
        approver = Approver(RuntimeError("the approver broke"))
        with connect(stdio_entry(approval=["count"]), approve=approver) as tools:
            count = by_name(tools)["genes_count"]
            with pytest.raises(RuntimeError, match="the approver broke"):
                count.function()
            approver.answer = True

            assert count.function() == 1

    def test_an_agent_is_told_a_declined_call_failed(self) -> None:
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )

        call_of = ChatCompletionMessageToolCall(
            id="call_1", type="function", function=Function(name="genes_echo", arguments=json.dumps({"text": "hi"}))
        )
        with connect(stdio_entry(approval=["echo"]), approve=Approver(False)) as tools:
            outputs, _ = run_tool_calls([call_of], tools.tools)

        assert outputs == [
            (
                'Error running tool "genes_echo": ApprovalDeclined: A person declined this call to genes_echo, so it '
                "was not made. Do not make it again unchanged."
            )
        ]

    def test_code_in_a_session_is_told_a_declined_call_failed(self, tmp_path: Path) -> None:
        from virtual_lab import LocalSession

        approver = Approver(False)
        with connect(stdio_entry(approval=["echo"]), approve=approver) as tools:
            with LocalSession(tmp_path, warn=False, tools=tools.tools) as session:
                declined = session.run("genes_echo(text='hi')")

        assert declined.error == (
            "HostToolError: genes_echo failed: ApprovalDeclined: A person declined this call to genes_echo, so it "
            "was not made. Do not make it again unchanged."
        )
        assert [asked.arguments for asked in approver.asked] == [{"text": "hi"}]

    def test_without_an_approver_and_without_a_terminal_a_call_is_declined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", NotATerminal(""))
        with connect(stdio_entry(approval=["count"])) as tools:
            count = by_name(tools)["genes_count"]
            with pytest.raises(ApprovalDeclined, match="there is no terminal to ask at"):
                count.function()

    def test_without_an_approver_a_person_at_the_terminal_is_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        typing(monkeypatch, "y")
        with connect(stdio_entry(approval=["count"])) as tools:
            assert by_name(tools)["genes_count"].function() == 1

    def test_a_tool_that_waits_says_so_in_its_description(self) -> None:
        with connect(stdio_entry(approval={"ask": ["*"], "allow": ["total"]}), approve=Approver()) as tools:
            echo, total = by_name(tools)["genes_echo"], by_name(tools)["genes_total"]

        assert echo.description == (
            "Says the text back.\n\nEach call waits for a person to approve it, and fails if they do not."
        )
        assert total.description == "Adds the values up."

    def test_a_tool_given_a_description_in_the_config_says_so_too(self) -> None:
        entry = stdio_entry(tools=[{"name": "echo", "description": "Repeats."}], approval=["echo"])
        with connect(entry, approve=Approver()) as tools:
            assert tools.tools[0].description == (
                "Repeats.\n\nEach call waits for a person to approve it, and fails if they do not."
            )

    def test_an_approval_naming_a_tool_the_server_lacks_is_warned_of(self) -> None:
        with pytest.warns(UserWarning, match=r"names submit, which it has no tool called, so no call waits for approval on its account"):
            with connect(stdio_entry(approval=["submit", "echo", "pay_*", "x?", "[ab]"]), approve=Approver()):
                pass

    def test_several_tools_the_server_lacks_are_warned_of_together(self) -> None:
        with pytest.warns(UserWarning, match=r"names submit, pay, which it has no tool called, so no call waits for approval on their account"):
            with connect(stdio_entry(approval=["submit", "pay"]), approve=Approver()):
                pass

    def test_an_approval_of_a_tool_left_out_of_those_offered_is_not_warned_of(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with connect(stdio_entry(tools=["total"], approval=["echo"]), approve=Approver()) as tools:
                assert [tool.name for tool in tools.tools] == ["genes_total"]


class Answerer:
    """Answers a server's questions as told, and keeps them."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.asked: list[ServerQuestion] = []

    def __call__(self, question: ServerQuestion) -> Any:
        self.asked.append(question)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


@pytest.fixture(params=["auto", "legacy"])
def protocol(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Each version of the protocol: the one from 2026, under which a server asks its questions in
    a tool's result, and the one before, under which it sends a request of its own, as Proto's
    server does."""
    mcp_tools = import_module("virtual_lab.mcp_tools")
    monkeypatch.setattr(mcp_tools.MCPConnection, "protocol_mode", request.param)
    return request.param


@pytest.fixture
def legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_tools = import_module("virtual_lab.mcp_tools")
    monkeypatch.setattr(mcp_tools.MCPConnection, "protocol_mode", "legacy")


@pytest.mark.usefixtures("protocol")
class TestAnsweringAServer:
    @pytest.fixture(autouse=True)
    def needs_mcp(self) -> None:
        pytest.importorskip("mcp")

    def test_a_yes_or_no_question_is_passed_on_and_answered(self) -> None:
        answerer = Answerer({"approve": True})
        with connect(stdio_entry(), answer=answerer) as tools:
            assert by_name(tools)["genes_confirm"].function() == "accept True"

        (asked,) = answerer.asked
        assert asked.server == "genes"
        assert asked.message == "Go ahead with the test?"
        assert asked.url is None
        assert list(asked.fields) == ["approve"]
        assert asked.fields["approve"]["type"] == "boolean"
        assert asked.required == ("approve",)

    def test_a_question_declined_is_declined_to_the_server(self) -> None:
        with connect(stdio_entry(), answer=Answerer(None)) as tools:
            assert by_name(tools)["genes_confirm"].function() == "decline"

    def test_a_form_is_answered_with_its_fields(self) -> None:
        answerer = Answerer({"name": "binder", "kind": "dna", "copies": 2})
        with connect(stdio_entry(), answer=answerer) as tools:
            answered = json.loads(by_name(tools)["genes_order"].function())

        assert answered == {"action": "accept", "order": {"name": "binder", "kind": "dna", "copies": 2}}
        (asked,) = answerer.asked
        assert set(asked.fields) == {"name", "kind", "copies"}
        assert asked.fields["kind"]["enum"] == ["protein", "dna"]
        assert set(asked.required) == {"name", "kind"}

    def test_an_answerer_that_fails_declines_with_a_warning(self) -> None:
        with connect(stdio_entry(), answer=Answerer(RuntimeError("the answerer broke"))) as tools:
            with pytest.warns(UserWarning, match="could not be answered, so it was declined: RuntimeError: the answerer broke"):
                assert by_name(tools)["genes_confirm"].function() == "decline"

    def test_without_an_answerer_and_without_a_terminal_a_question_is_declined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", NotATerminal(""))
        with connect(stdio_entry()) as tools:
            assert by_name(tools)["genes_confirm"].function() == "decline"

    def test_without_an_answerer_a_person_at_the_terminal_is_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        typing(monkeypatch, "y")
        with connect(stdio_entry()) as tools:
            assert by_name(tools)["genes_confirm"].function() == "accept True"

    def test_a_call_can_wait_for_approval_and_ask_a_question_too(self, approver: Approver) -> None:
        with connect(stdio_entry(approval=["confirm"]), approve=approver, answer=Answerer({"approve": False})) as tools:
            assert by_name(tools)["genes_confirm"].function() == "accept False"

        assert [asked.tool for asked in approver.asked] == ["genes_confirm"]


@pytest.mark.usefixtures("legacy")
class TestAnsweringAtAWebPage:
    """Under the protocol from before 2026, the only one under which a server can ask a person to
    go to a web page while one of its tools runs."""

    @pytest.fixture(autouse=True)
    def needs_mcp(self) -> None:
        pytest.importorskip("mcp")

    def test_a_question_at_a_web_page_is_passed_on_with_its_address(self) -> None:
        answerer = Answerer({})
        with connect(stdio_entry(), answer=answerer) as tools:
            assert by_name(tools)["genes_visit"].function() == "accept"

        (asked,) = answerer.asked
        assert (asked.message, asked.url, asked.fields) == ("Pay for the test at this page.", "https://pay.example.org/test", {})

    def test_a_question_at_a_web_page_declined_is_declined(self) -> None:
        with connect(stdio_entry(), answer=Answerer(None)) as tools:
            assert by_name(tools)["genes_visit"].function() == "decline"


class TestInstructions:
    @pytest.fixture(autouse=True)
    def needs_mcp(self) -> None:
        pytest.importorskip("mcp")

    def test_what_the_config_and_the_server_say_are_both_given(self) -> None:
        with connect(stdio_entry(instructions="Ask for TP53 first.")) as tools:
            instructions = {tool.instructions for tool in tools.tools}

        assert instructions == {
            (
                "The tools of the MCP server genes, named genes_ and then their names on the server:\nAsk for TP53 first.\n\n"
                "The server says of them:\nLook genes up by symbol."
            )
        }

    def test_a_server_that_says_nothing_is_given_what_the_config_says(self) -> None:
        with connect(stdio_entry("--instructions", "", instructions="Ask for TP53 first.")) as tools:
            assert tools.tools[0].instructions == "The tools of the MCP server genes, named genes_ and then their names on the server:\nAsk for TP53 first."

    def test_a_server_whose_tools_no_one_says_how_to_use_has_none(self) -> None:
        with connect(stdio_entry("--instructions", "  ")) as tools:
            assert {tool.instructions for tool in tools.tools} == {None}

    def test_what_a_server_says_is_cut_short(self, mcp_tools: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mcp_tools, "MCP_MAX_INSTRUCTIONS_CHARS", 30)

        with connect(stdio_entry("--instructions", "x" * 200)) as tools:
            said = tools.tools[0].instructions.split("The server says of them:\n", 1)[1]

        assert said.startswith("x" * 30)
        assert "x" * 31 not in said
        assert said != "x" * 30

    def test_what_a_server_says_does_not_show_a_secret_its_config_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GENE_SECRET", "very-secret-value")

        with connect(stdio_entry("--instructions", "Your key is ${GENE_SECRET}.")) as tools:
            said = tools.tools[0].instructions

        assert "very-secret-value" not in said
        assert "Your key is ${GENE_SECRET}." in said

    def test_a_tool_with_instructions_is_described_with_them(self) -> None:
        tool = Tool(name="t", description="d", parameters={}, function=print, instructions="Use it.")

        assert describe_input("tools", tool)["instructions"] == "Use it."
        assert "instructions" not in describe_input("tools", Tool(name="t", description="d", parameters={}, function=print))


def noting(name: str, instructions: str | None) -> Tool:
    return Tool(name=name, description=f"The tool {name}.", parameters={"type": "object", "properties": {}}, function=lambda: name, instructions=instructions)


class TestToldToTheAgents:
    def test_instructions_tools_share_are_given_once(self) -> None:
        prompt = tool_instructions_prompt([noting("a", "Use a and b."), noting("b", " Use a and b.\n"), noting("c", "Use c."), noting("d", None), noting("e", "  ")])

        assert prompt == "- How to use some of the tools, as those who provide them say:\n----\nUse a and b.\n\nUse c.\n----"

    def test_tools_without_instructions_add_nothing(self) -> None:
        assert tool_instructions_prompt([noting("a", None), noting("b", "")]) == ""
        assert tool_instructions_prompt([]) == ""

    def opening(self, fake_client: FakeClient) -> str:
        first = fake_client.completions.calls[0]["messages"]
        return next(message["content"] for message in first if message["role"] == "user")

    def test_an_individual_meeting_is_told_how_to_use_its_tools(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Done.")]

        run_meeting(
            meeting_type="individual",
            agenda="Order the binders.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(noting("lab_a", "Estimate the cost first."), noting("lab_b", "Estimate the cost first.")),
            num_rounds=0,
        )

        opening = self.opening(fake_client)
        assert opening.count("Estimate the cost first.") == 1
        assert opening.endswith("- How to use some of the tools, as those who provide them say:\n----\nEstimate the cost first.\n----")

    def test_a_team_meeting_is_told_how_to_use_its_tools(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        lead = Agent(title="Principal Investigator", expertise="biology", goal="lead", role="lead", model=team_member.model)
        fake_client.completions.responses = [text_response("Done.")]

        run_meeting(
            meeting_type="team",
            agenda="Order the binders.",
            save_dir=tmp_path,
            team_lead=lead,
            team_members=(team_member,),
            tools=(noting("lab_a", "Estimate the cost first."),),
            num_rounds=0,
        )

        assert "----\nEstimate the cost first.\n----" in self.opening(fake_client)

    def test_a_meeting_without_instructions_is_told_nothing_of_them(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Done.")]

        run_meeting(
            meeting_type="individual",
            agenda="Order the binders.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(noting("lab_a", None),),
            num_rounds=0,
        )

        assert "How to use some of the tools" not in self.opening(fake_client)

    def test_a_meeting_is_told_how_to_use_its_sessions_tools_after_the_session(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        from virtual_lab import LocalSession

        fake_client.completions.responses = [text_response("Done.")]
        with LocalSession(tmp_path / "work", warn=False, tools=[noting("lab_a", "Run small batches first.")]) as session:
            hold_meeting(
                meeting_type="individual",
                agenda="Order the binders.",
                save_dir=tmp_path,
                team_member=team_member,
                session=session,
                resources="none",
                tools=(noting("lab_b", "Estimate the cost first."),),
                num_rounds=0,
            )

        opening = self.opening(fake_client)
        assert "Functions added for this work" in opening
        assert opening.endswith("----\nEstimate the cost first.\n\nRun small batches first.\n----")
        assert opening.index("Functions added for this work") < opening.index("How to use some of the tools")


@pytest.fixture(autouse=True)
def no_terminal_by_default(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keeps a test that forgets to say what is typed from waiting on the real terminal."""
    if sys.stdin is not None and sys.stdin.isatty():
        monkeypatch.setattr(sys, "stdin", NotATerminal(""))
    yield
