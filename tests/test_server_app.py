"""Tests of the server's routes, as a page uses them."""

import asyncio
import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from virtual_lab.__about__ import __version__
from virtual_lab.server import app as app_module
from virtual_lab.server.app import MAX_MESSAGE_CHARS, RequestBodyReader, create_app, last_event
from virtual_lab.server.errors import ApiError
from virtual_lab.server.security import TOKEN_COOKIE
from virtual_lab.ui.workspace import Settings, Workspace

from conftest import TEST_MODEL, FakeClient, text_response
from test_server_conversations import hold_the_model, wait_until

TOKEN = "the-token-of-the-person-who-started-it"
LOCAL = "http://localhost"
BEARER = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def model_client(fake_client: FakeClient) -> FakeClient:
    return fake_client


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    store = Workspace(tmp_path / "workspace")
    store.save_settings(Settings(model=TEST_MODEL, max_cost=None))

    return store


@pytest.fixture
def app(workspace: Workspace) -> FastAPI:
    return create_app(workspace.root, token=TOKEN, check_keys=False)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, base_url=LOCAL, headers=BEARER) as opened:
        yield opened


def new_chat(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/api/chats", json={"team": [], **body})
    assert response.status_code == 201, response.text

    return response.json()


def wait_idle(client: TestClient, chat_id: str, turns: int = 1) -> dict[str, Any]:
    info: dict[str, Any] = {}

    def done() -> bool:
        nonlocal info
        info = client.get(f"/api/chats/{chat_id}").json()
        return info["state"] == "idle" and info["turns"] >= turns

    wait_until(done)

    return info


def handle_of(app: FastAPI, chat_id: str) -> Any:
    return app.state.conversations.get(chat_id)


def fail(message: str) -> Any:
    raise AssertionError(message)


async def call_app(
    app: Any, method: str, path: str, query: str, headers: dict[str, str], receive_next: Any
) -> list[dict[str, Any]]:
    """What the application sends for a request made to it directly, whose body is whatever receive_next gives."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return receive_next()

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": query.encode(),
        "headers": [(b"host", b"localhost"), (b"authorization", f"Bearer {TOKEN}".encode())]
        + [(name.encode(), value.encode()) for name, value in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("localhost", 80),
    }
    await app(scope, receive, send)

    return sent


def error_of(response: httpx.Response) -> dict[str, Any]:
    body = response.json()
    assert list(body) == ["error"], body

    return body["error"]


class TestAccess:
    def test_only_the_health_check_is_answered_without_the_token(self, app: FastAPI) -> None:
        with TestClient(app, base_url=LOCAL) as anonymous:
            health = anonymous.get("/api/health")
            assert health.json() == {"status": "ok", "version": __version__}

            for method, path in [
                ("GET", "/api/settings"),
                ("GET", "/api/chats"),
                ("POST", "/api/chats"),
                ("GET", "/api/keys"),
                ("PUT", "/api/keys/OPENAI_API_KEY"),
                ("GET", "/api/chats/anything/events"),
            ]:
                response = anonymous.request(method, path, json={})
                assert response.status_code == 401, path
                assert error_of(response)["code"] == "unauthorized"

    def test_a_page_of_another_address_may_not_change_anything(self, client: TestClient) -> None:
        response = client.post("/api/chats", json={"team": []}, headers={"Origin": "http://evil.example"})

        assert response.status_code == 403
        assert error_of(response)["code"] == "forbidden_origin"
        assert client.get("/api/chats").json() == {"chats": []}

    def test_a_name_that_is_not_this_machines_is_turned_away(self, app: FastAPI) -> None:
        with TestClient(app, base_url="http://evil.example", headers=BEARER) as elsewhere:
            assert elsewhere.get("/api/health").status_code == 403

    def test_an_address_there_is_none_of_is_answered_in_the_servers_shape(self, client: TestClient) -> None:
        for path in ("/api/nothing", "/api", "/api/chats/x/nothing"):
            response = client.get(path)
            assert response.status_code == 404, path
            assert error_of(response)["code"] == "not_found"

    def test_a_kind_of_request_an_address_does_not_take_is_answered_in_the_servers_shape(
        self, client: TestClient
    ) -> None:
        response = client.put("/api/chats")

        assert response.status_code == 405
        assert error_of(response)["code"] == "method_not_allowed"
        assert "GET" in response.headers["allow"]

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/api/docs", "/api/openapi.json"])
    def test_the_server_does_not_describe_itself_to_a_page(self, client: TestClient, path: str) -> None:
        response = client.get(path)

        assert response.status_code == 404
        assert error_of(response)["code"] == "not_found"

    def test_a_request_that_is_not_valid_says_which_part(self, client: TestClient) -> None:
        response = client.post("/api/chats", json={"team": "everyone", "max_cost": "plenty", "colour": "red"})

        error = error_of(response)
        assert response.status_code == 422
        assert error["code"] == "invalid"
        assert {item["field"] for item in error["fields"]} == {"team", "max_cost", "colour"}
        assert all(item["message"] for item in error["fields"])
        assert all(f"{item['field']}: {item['message']}" in error["message"] for item in error["fields"])
        # What was sent is not sent back
        assert "everyone" not in response.text and "plenty" not in response.text

    def test_a_body_that_is_not_json_is_not_valid(self, client: TestClient) -> None:
        response = client.post("/api/chats", content=b"{not json", headers={"Content-Type": "application/json"})

        assert response.status_code == 422
        assert error_of(response)["code"] == "invalid"

    def test_a_failure_in_the_server_is_not_told_to_the_page(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken() -> None:
            raise RuntimeError("secret detail of the machine")

        monkeypatch.setattr(app.state.conversations, "summaries", broken)

        with TestClient(app, base_url=LOCAL, headers=BEARER, raise_server_exceptions=False) as quiet:
            response = quiet.get("/api/chats")

        assert response.status_code == 500
        assert error_of(response)["code"] == "internal"
        assert "secret detail" not in response.text


class TestSettings:
    def test_the_settings_are_shown_with_where_everything_is_kept(
        self, client: TestClient, workspace: Workspace
    ) -> None:
        shown = client.get("/api/settings").json()

        assert shown == {
            "model": TEST_MODEL,
            "stream": True,
            "max_cost": None,
            "code": "none",
            "sandbox": "python",
            "network": True,
            "python": "",
            "workspace": str(workspace.root),
        }

    def test_a_change_is_kept_for_the_next_time(self, client: TestClient, workspace: Workspace) -> None:
        response = client.patch("/api/settings", json={"code": "docker", "max_cost": 4, "stream": False})

        assert response.status_code == 200
        assert response.json()["code"] == "docker"
        saved = workspace.load_settings()
        assert (saved.code, saved.max_cost, saved.stream, saved.model) == ("docker", 4.0, False, TEST_MODEL)
        assert client.get("/api/settings").json()["max_cost"] == 4.0

    def test_no_limit_is_set_by_null_and_a_setting_left_out_is_left_alone(
        self, client: TestClient, workspace: Workspace
    ) -> None:
        workspace.save_settings(Settings(model=TEST_MODEL, max_cost=2.0, network=False))

        client.patch("/api/settings", json={"max_cost": None})
        assert workspace.load_settings().max_cost is None
        assert workspace.load_settings().network is False

        client.patch("/api/settings", json={})
        assert workspace.load_settings().network is False

    def test_the_interfaces_other_settings_survive_a_change(self, client: TestClient, workspace: Workspace) -> None:
        workspace.save_settings(Settings(model=TEST_MODEL, mcp_config="/my/servers.json"))

        client.patch("/api/settings", json={"stream": False})

        assert workspace.load_settings().mcp_config == "/my/servers.json"

    @pytest.mark.parametrize(
        "body",
        [{"model": "no-such-model"}, {"code": "cloud"}, {"max_cost": -1}, {"mcp_config": "/x"}, {"stream": "maybe"}],
    )
    def test_a_change_that_cannot_be_used_is_refused_and_keeps_nothing(
        self, client: TestClient, workspace: Workspace, body: dict[str, Any]
    ) -> None:
        before = workspace.load_settings()

        response = client.patch("/api/settings", json=body)

        assert response.status_code == 422
        assert error_of(response)["code"] == "invalid"
        assert workspace.load_settings() == before

    def test_the_models_are_listed_with_whether_they_can_be_reached(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "set")

        shown = client.get("/api/models").json()

        assert shown["default"] == TEST_MODEL
        assert {"id": "gpt-5.2", "provider": "OpenAI", "ready": True} in shown["models"]


class TestKeys:
    def test_which_providers_have_their_keys_is_shown_without_any_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-never-shown")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        response = client.get("/api/keys")

        shown = response.json()
        providers = {item["provider"]: item for item in shown["providers"]}
        assert providers["OpenAI"] == {
            "provider": "OpenAI",
            "variables": ["OPENAI_API_KEY"],
            "missing": [],
            "ready": True,
        }
        assert providers["Groq"]["missing"] == ["GROQ_API_KEY"] and providers["Groq"]["ready"] is False
        assert shown["settable"] == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"]
        assert "sk-never-shown" not in response.text

    def test_a_key_is_set_until_the_server_stops_and_is_never_told_to_anyone(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
        monkeypatch.delenv("GEMINI_API_KEY")

        response = client.put("/api/keys/GEMINI_API_KEY", json={"value": "  gem-secret-value \n"})

        assert response.status_code == 200
        assert os.environ["GEMINI_API_KEY"] == "gem-secret-value"
        assert "gem-secret-value" not in response.text
        assert {item["provider"]: item["ready"] for item in response.json()["providers"]}["Gemini"] is True
        # Nothing of it is written down
        assert not any(
            "gem-secret-value" in path.read_text(errors="ignore")
            for path in workspace.root.rglob("*")
            if path.is_file()
        )

    def test_a_key_is_taken_away(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "to-remove")

        response = client.delete("/api/keys/GROQ_API_KEY")

        assert "GROQ_API_KEY" not in os.environ
        assert {item["provider"]: item["ready"] for item in response.json()["providers"]}["Groq"] is False
        assert client.delete("/api/keys/GROQ_API_KEY").status_code == 200

    @pytest.mark.parametrize("value", ["", "   ", "two\nlines", "tab\tinside", "nul\x00byte", "del\x7fchar"])
    def test_a_key_that_is_not_on_one_line_is_refused_and_not_told_back(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "kept")

        response = client.put("/api/keys/GROQ_API_KEY", json={"value": value})

        assert response.status_code == 422
        assert os.environ["GROQ_API_KEY"] == "kept"
        assert value.strip() == "" or value not in response.text

    def test_only_the_keys_the_interface_sets_can_be_set(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", os.environ["PATH"])

        for name in ("PATH", "AWS_REGION", "openai_api_key"):
            put = client.put(f"/api/keys/{name}", json={"value": "x"})
            assert put.status_code == 422, name
            assert error_of(put)["code"] == "invalid"
            assert client.delete(f"/api/keys/{name}").status_code == 422
        assert os.environ["PATH"] != "x"

    def test_a_key_that_is_too_long_is_refused(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "kept")

        response = client.put("/api/keys/GROQ_API_KEY", json={"value": "k" * 5000})

        assert response.status_code == 422
        assert os.environ["GROQ_API_KEY"] == "kept"


class TestScientists:
    EXPERT = {"title": "Nanobody Engineer", "expertise": "nanobodies", "goal": "design binders", "role": "propose"}

    def test_the_library_is_shown_with_those_the_person_described(self, client: TestClient) -> None:
        before = client.get("/api/scientists").json()["scientists"]
        assert all(item["builtin"] for item in before)

        after = client.put("/api/scientists", json=self.EXPERT).json()["scientists"]

        assert len(after) == len(before) + 1
        assert {**self.EXPERT, "builtin": False} in after
        assert client.get("/api/scientists").json()["scientists"] == after

    def test_a_scientist_that_is_described_again_is_replaced(self, client: TestClient) -> None:
        client.put("/api/scientists", json=self.EXPERT)

        after = client.put("/api/scientists", json={**self.EXPERT, "title": "nanobody ENGINEER", "role": "new"}).json()

        mine = [item for item in after["scientists"] if not item["builtin"]]
        assert [item["role"] for item in mine] == ["new"]

    def test_a_scientist_with_a_part_missing_is_refused(self, client: TestClient) -> None:
        for body in (
            {**self.EXPERT, "goal": ""},
            {**self.EXPERT, "goal": "  "},
            {"title": "Only a title"},
            {**self.EXPERT, "extra": 1},
        ):
            response = client.put("/api/scientists", json=body)
            assert response.status_code == 422, body
            assert error_of(response)["code"] == "invalid"

    def test_a_scientist_the_person_described_is_removed_and_the_librarys_cannot_be(self, client: TestClient) -> None:
        client.put("/api/scientists", json=self.EXPERT)

        removed = client.delete("/api/scientists/nanobody engineer")

        assert removed.status_code == 200
        assert all(item["builtin"] for item in removed.json()["scientists"])
        for title in ("Nanobody Engineer", "Principal Investigator", "Immunologist"):
            response = client.delete(f"/api/scientists/{title}")
            assert response.status_code == 404, title
        assert any(item["title"] == "Immunologist" for item in client.get("/api/scientists").json()["scientists"])

    def test_one_the_person_described_that_stands_in_place_of_the_librarys_is_given_up_for_it(
        self, client: TestClient
    ) -> None:
        client.put("/api/scientists", json={**self.EXPERT, "title": "Immunologist", "role": "mine"})

        client.delete("/api/scientists/Immunologist")

        immunologist = next(
            item for item in client.get("/api/scientists").json()["scientists"] if item["title"] == "Immunologist"
        )
        assert immunologist["builtin"] is True
        assert immunologist["role"] != "mine"


class TestChats:
    def test_a_conversation_is_made_from_what_the_request_gives_and_the_settings_for_the_rest(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/chats",
            json={
                "title": "Nanobodies",
                "team": ["Scientific Critic", "Immunologist"],
                "lead": "Principal Investigator",
                "code": {"resources": "all"},
                "max_cost": 1.25,
                "commercial_mode": True,
            },
        )

        assert response.status_code == 201
        made = response.json()
        assert made["title"] == "Nanobodies"
        assert made["state"] == "idle"
        assert made["team"] == ["Scientific Critic", "Immunologist"]
        assert made["lead"] == "Principal Investigator"
        assert made["max_cost"] == 1.25
        assert made["turns"] == 0 and made["last_event_id"] == 0
        assert made["config"]["model"] == TEST_MODEL
        assert made["config"]["code"]["resources"] == "all"
        assert made["config"]["commercial_mode"] is True
        assert made["config"]["stream"] is True

    def test_what_a_new_conversation_leaves_out_is_what_the_settings_say_and_null_is_no_limit(
        self, client: TestClient
    ) -> None:
        client.patch("/api/settings", json={"max_cost": 2.0, "stream": False})

        left_out = client.post("/api/chats", json={"team": []}).json()
        unlimited = client.post("/api/chats", json={"team": [], "max_cost": None}).json()

        assert (left_out["max_cost"], left_out["config"]["stream"]) == (2.0, False)
        assert unlimited["max_cost"] is None

    def test_a_conversation_made_with_nothing_has_the_leader_and_the_default_team(self, client: TestClient) -> None:
        made = client.post("/api/chats", json={}).json()

        assert made["lead"] == "Principal Investigator"
        assert made["team"][0] == "Scientific Critic"
        assert len(made["team"]) > 3

    @pytest.mark.parametrize(
        "body",
        [
            {"team": ["Alchemist"]},
            {"code": {"where": "cloud"}},
            {"model": "no-such-model"},
            {"max_cost": -5},
            {"lead": "Principal Investigator", "team": ["Principal Investigator"]},
            {"lead": {"title": "T", "expertise": "e", "goal": "g", "role": "r"}, "team": [{"title": "t"}]},
        ],
    )
    def test_a_conversation_that_cannot_be_made_is_refused_and_leaves_nothing(
        self, client: TestClient, workspace: Workspace, body: dict[str, Any]
    ) -> None:
        response = client.post("/api/chats", json=body)

        assert response.status_code == 422, response.text
        assert error_of(response)["code"] == "invalid"
        assert list(workspace.chats_dir.iterdir()) == []

    def test_a_lead_and_a_team_may_be_described_instead_of_chosen(self, client: TestClient) -> None:
        described = {"expertise": "e", "goal": "g", "role": "r"}

        made = client.post(
            "/api/chats",
            json={"lead": {"title": "Lead", **described}, "team": [{"title": "Other", **described}, "Immunologist"]},
        )

        assert made.status_code == 201
        assert made.json()["lead"] == "Lead"
        assert made.json()["team"] == ["Other", "Immunologist"]

    def test_conversations_are_listed_latest_first(self, client: TestClient, fake_client: FakeClient) -> None:
        first = new_chat(client, title="First")
        second = new_chat(client, title="Second")
        client.post(f"/api/chats/{first['id']}/messages", json={"text": "Hello"})
        wait_idle(client, first["id"])

        listed = client.get("/api/chats").json()["chats"]

        assert [item["id"] for item in listed] == [first["id"], second["id"]]
        assert listed[0]["turns"] == 1 and listed[1]["turns"] == 0
        assert listed[0]["state"] == "idle"

    def test_a_conversation_is_shown_by_its_id(self, client: TestClient) -> None:
        made = new_chat(client, title="Shown")

        shown = client.get(f"/api/chats/{made['id']}")

        assert shown.status_code == 200
        assert shown.json() == made

    @pytest.mark.parametrize("chat_id", ["nothing", "..", "a%2fb", ".hidden", "x" * 300])
    def test_there_is_none_with_an_id_that_is_not_one(self, client: TestClient, chat_id: str) -> None:
        for method, suffix, body in [
            ("GET", "", None),
            ("PATCH", "", {"title": "x"}),
            ("DELETE", "", None),
            ("POST", "/messages", {"text": "x"}),
            ("POST", "/note", {"text": "x"}),
            ("POST", "/stop", None),
            ("GET", "/files", None),
            ("GET", "/files/x.txt", None),
            ("GET", "/events", None),
            ("POST", "/uploads?name=x", None),
        ]:
            response = client.request(method, f"/api/chats/{chat_id}{suffix}", json=body)
            assert response.status_code == 404, (method, suffix)
            assert error_of(response)["code"] == "not_found"

    def test_a_conversation_is_renamed(self, client: TestClient, app: FastAPI) -> None:
        made = new_chat(client, title="Before")

        response = client.patch(f"/api/chats/{made['id']}", json={"title": "  After \n all "})

        assert response.status_code == 200
        assert response.json()["title"] == "After all"
        assert client.get("/api/chats").json()["chats"][0]["title"] == "After all"
        events = handle_of(app, made["id"]).chat.events_since(0)
        assert [event.text for event in events if event.kind == "title"] == ["After all"]

    def test_a_title_is_kept_short_and_may_not_be_empty(self, client: TestClient) -> None:
        made = new_chat(client)

        assert len(client.patch(f"/api/chats/{made['id']}", json={"title": "word " * 100}).json()["title"]) <= 80
        for body in ({"title": ""}, {"title": "   \n "}, {}, {"title": 5}, {"title": "x", "other": 1}):
            response = client.patch(f"/api/chats/{made['id']}", json=body)
            assert response.status_code == 422, body

    def test_a_conversation_is_removed_with_everything_it_saved(self, client: TestClient, workspace: Workspace) -> None:
        made = new_chat(client)

        response = client.delete(f"/api/chats/{made['id']}")

        assert response.status_code == 204
        assert response.content == b""
        assert client.get(f"/api/chats/{made['id']}").status_code == 404
        assert client.get("/api/chats").json() == {"chats": []}
        assert list(workspace.chats_dir.iterdir()) == []
        assert client.delete(f"/api/chats/{made['id']}").status_code == 404


class TestMessages:
    def test_a_message_is_answered_in_the_background(self, client: TestClient, fake_client: FakeClient) -> None:
        made = new_chat(client)
        fake_client.completions.responses.append(text_response("Nanobodies are small antibodies."))

        response = client.post(f"/api/chats/{made['id']}/messages", json={"text": "What is a nanobody?"})

        assert response.status_code == 202
        assert response.json()["turn"] == 1
        info = wait_idle(client, made["id"])
        assert info["title"] == "What is a nanobody?"
        assert info["spent"] is not None and info["spent"] > 0
        assert info["usage"]["cost"] is not None or info["usage"]
        assert info["last_event_id"] > 2

    def test_a_message_while_the_last_is_being_answered_is_turned_away(
        self, client: TestClient, fake_client: FakeClient
    ) -> None:
        made = new_chat(client)
        gate = hold_the_model(fake_client)
        client.post(f"/api/chats/{made['id']}/messages", json={"text": "First"})
        wait_until(gate.reached.is_set)

        response = client.post(f"/api/chats/{made['id']}/messages", json={"text": "Second"})

        assert response.status_code == 409
        assert error_of(response)["code"] == "busy"
        gate.release()
        assert wait_idle(client, made["id"])["turns"] == 1

    @pytest.mark.parametrize(
        "body",
        [{}, {"text": ""}, {"text": "  \n "}, {"text": "x", "attachments": ["missing.csv"]}, {"attachments": ["a"]}],
    )
    def test_a_message_with_nothing_in_it_or_a_file_that_was_not_attached_is_refused(
        self, client: TestClient, body: dict[str, Any]
    ) -> None:
        made = new_chat(client)

        response = client.post(f"/api/chats/{made['id']}/messages", json=body)

        assert response.status_code == 422
        assert error_of(response)["code"] == "invalid"
        assert client.get(f"/api/chats/{made['id']}").json()["turns"] == 0

    def test_a_message_that_is_too_long_or_not_as_asked_is_refused(self, client: TestClient) -> None:
        made = new_chat(client)

        for body in (
            {"text": "x" * (MAX_MESSAGE_CHARS + 1)},
            {"text": 5},
            {"text": "x", "attachments": "a.csv"},
            {"text": "x", "attachments": ["a"] * 101},
            {"text": "x", "mood": "hurried"},
        ):
            assert client.post(f"/api/chats/{made['id']}/messages", json=body).status_code == 422

    def test_a_message_is_not_taken_when_the_key_its_model_needs_is_not_set(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        app = create_app(workspace.root, token=TOKEN)
        with TestClient(app, base_url=LOCAL, headers=BEARER) as client:
            made = new_chat(client)
            client.app.state.conversations.close()  # type: ignore[attr-defined]
            monkeypatch.delenv("OPENAI_API_KEY")

            response = client.post(f"/api/chats/{made['id']}/messages", json={"text": "Hello"})
            created = client.post("/api/chats", json={"team": []})
            read = client.get(f"/api/chats/{made['id']}")

        assert response.status_code == 422
        assert error_of(response)["code"] == "missing_keys"
        assert error_of(response)["missing"] == ["OPENAI_API_KEY"]
        assert created.status_code == 422 and error_of(created)["code"] == "missing_keys"
        # What was said before can still be read
        assert read.status_code == 200

    def test_a_conversation_closed_while_a_message_is_sent_is_told_so(self, client: TestClient, app: FastAPI) -> None:
        made = new_chat(client)
        handle_of(app, made["id"]).chat.close()

        response = client.post(f"/api/chats/{made['id']}/messages", json={"text": "Hello"})

        assert response.status_code == 409
        assert error_of(response)["code"] == "closed"


class TestSteering:
    def test_steering_a_conversation_that_is_not_working_is_not_accepted(self, client: TestClient) -> None:
        made = new_chat(client)

        for action, body in (("note", {"text": "Also this"}), ("pause", None), ("resume", None), ("stop", None)):
            response = client.post(f"/api/chats/{made['id']}/{action}", json=body)
            assert response.status_code == 200, action
            assert response.json() == {"accepted": False}, action

    def test_a_turn_is_paused_noted_resumed_and_stopped(self, client: TestClient, fake_client: FakeClient) -> None:
        made = new_chat(client)
        gate = hold_the_model(fake_client)
        client.post(f"/api/chats/{made['id']}/messages", json={"text": "Go"})
        wait_until(gate.reached.is_set)
        url = f"/api/chats/{made['id']}"

        assert client.post(f"{url}/pause").json() == {"accepted": True}
        assert client.get(url).json()["state"] == "pausing"
        assert client.post(f"{url}/note", json={"text": "Mind the length"}).json() == {"accepted": True}
        assert client.post(f"{url}/resume").json() == {"accepted": True}
        assert client.get(url).json()["state"] == "running"
        assert client.post(f"{url}/resume").json() == {"accepted": False}
        assert client.post(f"{url}/stop").json() == {"accepted": True}
        assert client.get(url).json()["state"] == "stopping"
        gate.release()

        assert wait_idle(client, made["id"])["state"] == "idle"

    def test_a_note_needs_something_in_it(self, client: TestClient) -> None:
        made = new_chat(client)

        for body in ({}, {"text": ""}, {"text": 5}, {"text": "x", "more": 1}):
            assert client.post(f"/api/chats/{made['id']}/note", json=body).status_code == 422

    def test_steering_a_conversation_there_is_none_of_is_not_found(self, client: TestClient) -> None:
        for action in ("note", "pause", "resume", "stop"):
            assert client.post(f"/api/chats/nothing/{action}", json={"text": "x"}).status_code == 404


class TestUploads:
    def upload(self, client: TestClient, chat_id: str, name: str, content: Any) -> httpx.Response:
        return client.post(f"/api/chats/{chat_id}/uploads", params={"name": name}, content=content)

    def test_a_file_is_kept_with_the_conversation_and_listed_and_read_back(self, client: TestClient) -> None:
        made = new_chat(client)

        response = self.upload(client, made["id"], "data.csv", b"a,b\n1,2\n")

        assert response.status_code == 201
        assert response.json() == {"name": "data.csv", "path": "uploads/data.csv", "size": 8}
        listed = client.get(f"/api/chats/{made['id']}/files").json()
        assert [item["path"] for item in listed["files"]] == ["uploads/data.csv"]
        assert listed["files"][0]["type"] == "text/csv" and listed["files"][0]["size"] == 8
        assert listed["truncated"] is False
        read = client.get(f"/api/chats/{made['id']}/files/uploads/data.csv")
        assert read.content == b"a,b\n1,2\n"
        assert read.headers["content-type"].startswith("text/csv")

    def test_a_file_is_never_replaced_and_the_same_file_is_not_kept_twice(self, client: TestClient) -> None:
        made = new_chat(client)

        names = [
            self.upload(client, made["id"], "data.csv", content).json()["name"]
            for content in (b"one", b"two", b"one", b"two", b"three")
        ]

        assert names == ["data.csv", "data (2).csv", "data.csv", "data (2).csv", "data (3).csv"]
        assert len(client.get(f"/api/chats/{made['id']}/files").json()["files"]) == 3

    @pytest.mark.parametrize(
        ("name", "saved"),
        [
            ("../../outside.txt", "outside.txt"),
            ("..\\..\\outside.txt", "outside.txt"),
            ("/etc/passwd", "passwd"),
            (".hidden", "hidden"),
            ("a\x00b\nc.txt", "abc.txt"),
            ("..", "upload"),
        ],
    )
    def test_a_name_that_would_leave_the_uploads_is_made_safe(
        self, client: TestClient, workspace: Workspace, name: str, saved: str
    ) -> None:
        made = new_chat(client)

        response = self.upload(client, made["id"], name, b"x")

        assert response.status_code == 201
        assert response.json()["name"] == saved
        assert not (workspace.root / "outside.txt").exists()
        assert not (workspace.chats_dir / "outside.txt").exists()

    def test_an_empty_file_is_kept_and_a_name_is_needed(self, client: TestClient) -> None:
        made = new_chat(client)

        assert self.upload(client, made["id"], "empty.txt", b"").json()["size"] == 0
        for params in ({}, {"name": ""}, {"name": "x" * 1001}):
            response = client.post(f"/api/chats/{made['id']}/uploads", params=params, content=b"x")
            assert response.status_code == 422

    def test_a_file_that_says_it_is_too_large_is_refused_without_a_byte_of_it_being_read(
        self, client: TestClient, app: FastAPI
    ) -> None:
        made = new_chat(client)
        handle_of(app, made["id"]).chat.max_upload_bytes = 5

        sent = asyncio.run(
            call_app(
                app,
                "POST",
                f"/api/chats/{made['id']}/uploads",
                "name=big.bin",
                {"content-length": "6"},
                lambda: fail("The body was read"),
            )
        )

        assert sent[0]["status"] == 413
        assert json.loads(sent[1]["body"])["error"]["code"] == "too_large"

    def test_a_file_exactly_as_large_as_may_be_attached_is_kept(self, client: TestClient, app: FastAPI) -> None:
        made = new_chat(client)
        handle_of(app, made["id"]).chat.max_upload_bytes = 5

        assert self.upload(client, made["id"], "fits.bin", b"01234").status_code == 201
        assert self.upload(client, made["id"], "over.bin", b"012345").status_code == 413

    def test_a_sender_that_goes_away_part_of_the_way_is_not_an_error_of_the_server(
        self, client: TestClient, app: FastAPI
    ) -> None:
        made = new_chat(client)
        messages = iter([{"type": "http.request", "body": b"abc", "more_body": True}, {"type": "http.disconnect"}])

        sent = asyncio.run(
            call_app(app, "POST", f"/api/chats/{made['id']}/uploads", "name=cut.bin", {}, lambda: next(messages))
        )

        assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [499]
        assert self.leftovers(handle_of(app, made["id"])) == []

    def test_a_file_that_is_too_large_is_refused_before_it_is_read_and_leaves_nothing(
        self, client: TestClient, app: FastAPI, workspace: Workspace
    ) -> None:
        made = new_chat(client)
        handle_of(app, made["id"]).chat.max_upload_bytes = 5

        response = self.upload(client, made["id"], "big.bin", b"0123456789")

        assert response.status_code == 413
        assert error_of(response)["code"] == "too_large"
        assert self.leftovers(handle_of(app, made["id"])) == []

    def test_a_file_that_proves_too_large_as_it_arrives_is_refused_and_leaves_nothing(
        self, client: TestClient, app: FastAPI
    ) -> None:
        made = new_chat(client)
        handle_of(app, made["id"]).chat.max_upload_bytes = 5

        # With no length announced, which is how a body of unknown length is sent
        response = self.upload(client, made["id"], "big.bin", iter([b"abc", b"defg"]))

        assert response.status_code == 413
        assert error_of(response)["code"] == "too_large"
        assert self.leftovers(handle_of(app, made["id"])) == []

    def leftovers(self, handle: Any) -> list[str]:
        directory = handle.chat.uploads_dir
        return sorted(path.name for path in directory.iterdir()) if directory.exists() else []

    def test_a_file_is_attached_to_a_message_and_told_to_the_lead(
        self, client: TestClient, app: FastAPI, fake_client: FakeClient
    ) -> None:
        made = new_chat(client)
        attached = self.upload(client, made["id"], "data.csv", b"a,b\n").json()
        fake_client.completions.responses.append(text_response("I cannot open it here."))

        client.post(f"/api/chats/{made['id']}/messages", json={"text": "Look", "attachments": [attached["name"]]})
        wait_idle(client, made["id"])

        user = next(event for event in handle_of(app, made["id"]).chat.events_since(0) if event.kind == "user")
        assert user.data["attachments"] == [attached]

    def test_where_code_runs_files_are_kept_where_it_reads_them_and_shown_with_what_it_made(
        self, client: TestClient, app: FastAPI
    ) -> None:
        made = new_chat(client, code={"where": "local"})
        handle = handle_of(app, made["id"])
        self.upload(client, made["id"], "data.csv", b"a,b\n")
        (handle.session.directory / "result.txt").write_text("made by code")
        (handle.session.directory / "figures").mkdir()
        (handle.session.directory / "figures" / "plot.png").write_bytes(b"\x89PNG")

        listed = client.get(f"/api/chats/{made['id']}/files").json()["files"]

        assert [item["path"] for item in listed] == ["figures/plot.png", "result.txt", "uploads/data.csv"]
        assert (handle.session.directory / "uploads" / "data.csv").is_file()

    def test_a_file_arrives_whole_whatever_its_size(self, app: FastAPI) -> None:
        content = os.urandom(3 * 1024 * 1024 + 17)

        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            response = http.post(f"/api/chats/{made['id']}/uploads", params={"name": "big.bin"}, content=content)
            read = http.get(f"/api/chats/{made['id']}/files/uploads/big.bin", params={"download": "true"})

        assert response.json()["size"] == len(content)
        assert hashlib.sha256(read.content).hexdigest() == hashlib.sha256(content).hexdigest()
        assert read.headers["content-disposition"].startswith("attachment")

    def test_the_server_goes_on_answering_while_a_file_is_still_arriving(self, app: FastAPI) -> None:
        arrived = threading.Event()
        carry_on = threading.Event()

        def slowly() -> Iterator[bytes]:
            yield b"first part, "
            arrived.set()
            assert carry_on.wait(20)
            yield b"last part"

        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            results: list[httpx.Response] = []
            thread = threading.Thread(
                target=lambda: results.append(
                    http.post(f"/api/chats/{made['id']}/uploads", params={"name": "slow.txt"}, content=slowly())
                )
            )
            thread.start()
            assert arrived.wait(10)

            others = [httpx.get(f"{base}/api/health", timeout=5).status_code for _ in range(3)]
            carry_on.set()
            thread.join(20)

        assert others == [200, 200, 200]
        assert results[0].json()["size"] == len(b"first part, last part")

    def test_a_file_whose_sender_goes_away_leaves_nothing(self, app: FastAPI) -> None:
        def breaks() -> Iterator[bytes]:
            yield b"some of it"
            raise RuntimeError("the connection broke")

        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            with pytest.raises(RuntimeError, match="the connection broke"):
                http.post(f"/api/chats/{made['id']}/uploads", params={"name": "cut.txt"}, content=breaks())
            handle = handle_of(app, made["id"])
            time.sleep(0.3)

            assert self.leftovers(handle) == []
            assert http.get("/api/health").status_code == 200

    def test_a_file_that_stops_arriving_is_given_up_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app_module, "UPLOAD_CHUNK_TIMEOUT", 0.2)

        class Stalled:
            async def _chunks(self) -> Any:
                yield b"abc"
                await asyncio.sleep(30)
                yield b"never"

            def stream(self) -> Any:
                return self._chunks()

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            reader = RequestBodyReader(Stalled(), loop)  # type: ignore[arg-type]
            with pytest.raises(ApiError) as caught:
                reader.read(10)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)

        assert (caught.value.status, caught.value.code) == (408, "timeout")

    def test_the_body_of_a_request_is_read_in_the_sizes_asked_for(self) -> None:
        class Pieces:
            async def _chunks(self) -> Any:
                for piece in (b"abc", b"", b"defgh", b"i"):
                    yield piece

            def stream(self) -> Any:
                return self._chunks()

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            reader = RequestBodyReader(Pieces(), loop)  # type: ignore[arg-type]
            read = [reader.read(2), reader.read(4), reader.read(100), reader.read(5), reader.read(-1)]
            whole = RequestBodyReader(Pieces(), loop).read(-1)
            empty = RequestBodyReader(Pieces(), loop)
            empty.read(-1)
            after = empty.read(3)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)

        assert read == [b"ab", b"cdef", b"ghi", b"", b""]
        assert whole == b"abcdefghi"
        assert after == b""


class TestFiles:
    def make(self, client: TestClient, app: FastAPI) -> tuple[str, Path]:
        made = new_chat(client, code={"where": "local"})
        root = handle_of(app, made["id"]).session.directory

        return made["id"], root

    def test_a_picture_is_shown_in_the_page_and_a_page_that_was_written_is_not_run(
        self, client: TestClient, app: FastAPI
    ) -> None:
        chat_id, root = self.make(client, app)
        (root / "plot.png").write_bytes(b"\x89PNG\r\n")
        (root / "report.html").write_text("<script>steal()</script>")
        (root / "plot.svg").write_text("<svg><script>steal()</script></svg>")

        png = client.get(f"/api/chats/{chat_id}/files/plot.png")
        html = client.get(f"/api/chats/{chat_id}/files/report.html")
        svg = client.get(f"/api/chats/{chat_id}/files/plot.svg")

        assert png.headers["content-type"] == "image/png"
        assert png.headers["content-disposition"].startswith("inline")
        assert html.headers["content-type"].startswith("text/plain")
        assert html.text == "<script>steal()</script>"
        for response in (png, html, svg):
            assert response.headers["x-content-type-options"] == "nosniff"
            assert "sandbox" in response.headers["content-security-policy"]
            assert "default-src 'none'" in response.headers["content-security-policy"]

    def test_a_file_can_be_asked_for_to_be_saved(self, client: TestClient, app: FastAPI) -> None:
        chat_id, root = self.make(client, app)
        (root / "plot.png").write_bytes(b"\x89PNG")

        response = client.get(f"/api/chats/{chat_id}/files/plot.png", params={"download": "true"})

        assert response.headers["content-disposition"].startswith("attachment")

    def test_a_file_in_a_folder_is_found_by_its_path(self, client: TestClient, app: FastAPI) -> None:
        chat_id, root = self.make(client, app)
        (root / "a" / "b").mkdir(parents=True)
        (root / "a" / "b" / "deep file.txt").write_text("deep")

        assert client.get(f"/api/chats/{chat_id}/files/a/b/deep%20file.txt").text == "deep"

    @pytest.mark.parametrize(
        "path",
        [
            "..%2f..%2fconfig.json",
            "%2e%2e/%2e%2e/config.json",
            "..%2fconfig.json",
            "a/../../config.json",
            "%2fetc%2fpasswd",
            "/etc/passwd",
            "link.txt",
            "folder/secret.txt",
            "nothing.txt",
            "figures",
            "figures/",
            "a%00b",
        ],
    )
    def test_nothing_outside_the_conversations_files_is_given(
        self, client: TestClient, app: FastAPI, tmp_path: Path, path: str
    ) -> None:
        chat_id, root = self.make(client, app)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("keys")
        (root / "link.txt").symlink_to(outside / "secret.txt")
        (root / "folder").symlink_to(outside, target_is_directory=True)
        (root / "figures").mkdir()

        response = client.get(f"/api/chats/{chat_id}/files/{path}")

        assert response.status_code == 404, response.text
        assert "keys" not in response.text and "model" not in response.text.lower()

    def test_a_conversation_without_a_session_shows_only_what_was_attached_to_it(self, client: TestClient) -> None:
        made = new_chat(client)
        client.post(f"/api/chats/{made['id']}/uploads", params={"name": "mine.txt"}, content=b"mine")

        for path in ("config.json", "chat.json", "messages.jsonl", "events.jsonl"):
            assert client.get(f"/api/chats/{made['id']}/files/{path}").status_code == 404, path
        assert client.get(f"/api/chats/{made['id']}/files/uploads/mine.txt").text == "mine"
        assert [item["path"] for item in client.get(f"/api/chats/{made['id']}/files").json()["files"]] == [
            "uploads/mine.txt"
        ]

    def test_a_listing_that_was_cut_short_says_so(
        self, client: TestClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat_id, root = self.make(client, app)
        for name in ("a.txt", "b.txt", "c.txt"):
            (root / name).write_text(name)
        real = app_module.list_files
        monkeypatch.setattr(app_module, "list_files", lambda *args, **kwargs: real(*args, limit=2, **kwargs))

        listed = client.get(f"/api/chats/{chat_id}/files").json()

        assert listed["truncated"] is True
        assert len(listed["files"]) == 2

    def test_another_conversations_files_are_not_given(self, client: TestClient, app: FastAPI) -> None:
        first, root = self.make(client, app)
        second, _ = self.make(client, app)
        (root / "private.txt").write_text("first only")

        assert client.get(f"/api/chats/{first}/files/private.txt").status_code == 200
        assert client.get(f"/api/chats/{second}/files/private.txt").status_code == 404
        assert client.get(f"/api/chats/{second}/files/..%2f{first}%2fwork%2fprivate.txt").status_code == 404


class TestPages:
    @pytest.fixture
    def pages(self, tmp_path: Path) -> Path:
        root = tmp_path / "pages"
        (root / "_next" / "static").mkdir(parents=True)
        (root / "index.html").write_text("<html>home</html>")
        (root / "settings.html").write_text("<html>settings</html>")
        (root / "chat").mkdir()
        (root / "chat" / "index.html").write_text("<html>chat</html>")
        (root / "_next" / "static" / "app.js").write_text("console.log(1)")
        (tmp_path / "secret.txt").write_text("not a page")

        return root

    @pytest.fixture
    def site(self, workspace: Workspace, pages: Path) -> Iterator[TestClient]:
        app = create_app(workspace.root, token=TOKEN, static_dir=pages, check_keys=False)
        with TestClient(app, base_url=LOCAL, headers=BEARER) as opened:
            yield opened

    def test_the_pages_of_the_interface_are_served_by_their_address(self, site: TestClient) -> None:
        assert site.get("/").text == "<html>home</html>"
        assert site.get("/settings").text == "<html>settings</html>"
        assert site.get("/chat").text == "<html>chat</html>"
        assert site.get("/chat/").text == "<html>chat</html>"
        assert site.get("/index.html").text == "<html>home</html>"
        assert site.get("/_next/static/app.js").text == "console.log(1)"

    def test_an_address_of_no_page_shows_the_first_page_so_that_the_page_can_route_it(self, site: TestClient) -> None:
        assert site.get("/chat/2026-10-07_1500_what-is-a-nanobody").text == "<html>home</html>"
        assert site.get("/history").text == "<html>home</html>"

    def test_a_file_there_is_none_of_is_not_found(self, site: TestClient) -> None:
        response = site.get("/missing.js")

        assert response.status_code == 404
        assert error_of(response)["code"] == "not_found"

    def test_the_api_is_never_answered_with_a_page(self, site: TestClient) -> None:
        for path in ("/api/missing", "/api/chats/x/missing", "/api"):
            response = site.get(path)
            assert response.status_code == 404, path
            assert error_of(response)["code"] == "not_found"

    @pytest.mark.parametrize(
        "path", ["/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "//etc/passwd", "/a%00"]
    )
    def test_nothing_outside_the_pages_is_served(self, site: TestClient, path: str) -> None:
        response = site.get(path)

        assert "not a page" not in response.text
        assert "root:" not in response.text

    def test_a_link_in_the_pages_to_a_file_outside_them_is_not_followed(
        self, site: TestClient, pages: Path, tmp_path: Path
    ) -> None:
        (pages / "linked.txt").symlink_to(tmp_path / "secret.txt")
        (pages / "linked").symlink_to(tmp_path / "secret.txt")

        assert site.get("/linked.txt").status_code == 404
        assert "not a page" not in site.get("/linked").text

    def test_what_has_a_hash_in_its_name_is_kept_and_the_rest_is_looked_at_again(self, site: TestClient) -> None:
        assert site.get("/_next/static/app.js").headers["cache-control"] == "public, max-age=31536000, immutable"
        assert site.get("/").headers["cache-control"] == "no-cache"
        assert site.get("/chat").headers["cache-control"] == "no-cache"

    def test_the_link_with_the_token_opens_the_first_page_signed_in(self, workspace: Workspace, pages: Path) -> None:
        app = create_app(workspace.root, token=TOKEN, static_dir=pages, check_keys=False)

        with TestClient(app, base_url=LOCAL) as anonymous:
            opened = anonymous.get(f"/?token={TOKEN}", follow_redirects=True)

            assert opened.text == "<html>home</html>"
            assert anonymous.cookies.get(TOKEN_COOKIE) == TOKEN
            assert anonymous.get("/api/settings").status_code == 200

    def test_without_pages_the_server_serves_only_its_api(self, client: TestClient) -> None:
        assert client.get("/").status_code == 404
        assert client.get("/api/health").status_code == 200


class TestLifecycle:
    def test_the_conversations_are_told_where_the_loop_is_and_closed_when_the_server_stops(
        self, app: FastAPI, fake_client: FakeClient
    ) -> None:
        conversations = app.state.conversations
        assert conversations.loop is None

        with TestClient(app, base_url=LOCAL, headers=BEARER) as opened:
            assert conversations.loop is not None
            made = new_chat(opened)
            gate = hold_the_model(fake_client)
            opened.post(f"/api/chats/{made['id']}/messages", json={"text": "Go"})
            wait_until(gate.reached.is_set)
            handle = conversations.get(made["id"])
            threading.Timer(0.2, gate.release).start()

        assert handle.closed
        assert handle.chat.state == "idle"
        assert (handle.directory / "config.json").is_file()


@contextmanager
def serve(app: FastAPI, token: str | None = TOKEN) -> Iterator[tuple[str, httpx.Client]]:
    """The application served by a real server on a port of its own, and a client that is signed in to it."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=2)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_until(lambda: server.started)
    base = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=httpx.Timeout(15, read=15)) as http:
            yield base, http
    finally:
        server.should_exit = True
        thread.join(20)


def read_frames(
    http: httpx.Client,
    url: str,
    until: Any = None,
    quiet: float | None = None,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """What a page following a conversation is told, as its frames, until one is the last wanted, or, if quiet
    is given, until nothing has come for that many seconds."""
    frames: list[dict[str, Any]] = []
    fields: dict[str, Any] = {}
    timeout = httpx.Timeout(15, read=quiet) if quiet else None
    try:
        with http.stream("GET", url, headers=headers, timeout=timeout) as response:
            assert response.status_code == 200, response.read()
            assert response.headers["content-type"].startswith("text/event-stream")
            for line in response.iter_lines():
                if line == "":
                    if fields:
                        if "data" in fields:
                            fields["data"] = json.loads(fields["data"])
                        frames.append(fields)
                        fields = {}
                        if until is not None and until(frames[-1]):
                            break
                    continue
                name, _, value = line.partition(": ")
                fields[name] = value
    except httpx.ReadTimeout:
        assert quiet, "The stream stopped sending"

    return frames


def wait_until_answered(http: httpx.Client, chat_id: str) -> None:
    wait_until(lambda: http.get(f"/api/chats/{chat_id}").json()["state"] == "idle")


def is_answer(frame: dict[str, Any]) -> bool:
    return frame.get("data", {}).get("kind") == "answer"


def is_idle(frame: dict[str, Any]) -> bool:
    """The conversation is told to be idle, which is the last thing said of a turn."""
    data = frame.get("data", {})

    return data.get("kind") == "status" and data["data"]["state"] == "idle" and data["turn"] >= 1


class TestEvents:
    def test_the_events_of_a_conversation_are_sent_as_they_happen_and_a_page_that_reconnects_misses_none(
        self, app: FastAPI, fake_client: FakeClient
    ) -> None:
        fake_client.completions.responses.append(text_response("The answer is here"))
        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            url = f"/api/chats/{made['id']}/events"
            http.post(f"/api/chats/{made['id']}/messages", json={"text": "Question"})
            wait_until_answered(http, made["id"])

            whole = read_frames(http, url, until=is_idle)
            ids = [int(frame["id"]) for frame in whole if "id" in frame]
            after_two = read_frames(http, f"{url}?after={ids[1]}", until=is_idle)
            by_header = read_frames(http, url, until=is_idle, headers={"Last-Event-ID": str(ids[2])})
            nothing_new = read_frames(http, f"{url}?after={ids[-1]}", quiet=0.5)

        assert whole[0] == {"retry": "2000"}
        assert [frame["event"] for frame in whole[1:]] == ["chat"] * (len(whole) - 1)
        assert ids == sorted(set(ids)) and len(ids) >= 3
        assert [frame["data"]["id"] for frame in whole[1:]] == ids
        assert [frame["data"]["kind"] for frame in whole[1:] if frame["data"]["kind"] in ("user", "answer")] == [
            "user",
            "answer",
        ]
        assert [frame["data"]["text"] for frame in whole[1:] if frame["data"]["kind"] == "answer"] == [
            "The answer is here"
        ]
        assert [int(frame["id"]) for frame in after_two[1:]] == [i for i in ids if i > ids[1]]
        assert [int(frame["id"]) for frame in by_header[1:]] == [i for i in ids if i > ids[2]]
        assert [frame for frame in nothing_new if "id" in frame] == []

    def test_the_greater_of_the_address_and_the_header_is_where_a_page_carries_on_from(
        self, app: FastAPI, fake_client: FakeClient
    ) -> None:
        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            http.post(f"/api/chats/{made['id']}/messages", json={"text": "Question"})
            wait_until_answered(http, made["id"])
            url = f"/api/chats/{made['id']}/events"
            ids = [int(f["id"]) for f in read_frames(http, url, until=is_idle) if "id" in f]

            frames = read_frames(http, f"{url}?after={ids[1]}", until=is_idle, headers={"Last-Event-ID": str(ids[0])})
            garbage = read_frames(http, url, quiet=0.5, headers={"Last-Event-ID": "not a number"})

        assert [int(f["id"]) for f in frames if "id" in f] == [i for i in ids if i > ids[1]]
        assert [int(f["id"]) for f in garbage if "id" in f] == ids

    @pytest.mark.parametrize(
        ("header", "number"), [(None, 0), ("", 0), ("7", 7), (" 12 ", 12), ("-3", 0), ("not a number", 0), ("1.5", 0)]
    )
    def test_the_last_event_a_browser_names_is_a_number_or_nothing(self, header: str | None, number: int) -> None:
        assert last_event(header) == number

    def test_a_page_already_following_is_told_of_a_message_sent_afterwards(
        self, app: FastAPI, fake_client: FakeClient
    ) -> None:
        fake_client.completions.responses.append(text_response("Told as it happens"))
        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            sender = threading.Timer(
                0.5, lambda: http.post(f"/api/chats/{made['id']}/messages", json={"text": "Later"})
            )
            sender.start()

            frames = read_frames(http, f"/api/chats/{made['id']}/events", until=is_answer)
            sender.join(10)

        assert [f["data"]["kind"] for f in frames if "data" in f and f["data"]["kind"] in ("user", "answer")] == [
            "user",
            "answer",
        ]

    def test_several_pages_follow_one_conversation(self, app: FastAPI, fake_client: FakeClient) -> None:
        fake_client.completions.responses.append(text_response("For everyone"))
        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            results: dict[int, list[dict[str, Any]]] = {}

            def follow(number: int) -> None:
                with httpx.Client(base_url=base, headers=BEARER, timeout=15) as own:
                    results[number] = read_frames(own, f"/api/chats/{made['id']}/events", until=is_answer)

            threads = [threading.Thread(target=follow, args=(number,)) for number in range(3)]
            for thread in threads:
                thread.start()
            wait_until(lambda: len(handle_of(app, made["id"])._waiters) == 3)
            http.post(f"/api/chats/{made['id']}/messages", json={"text": "Hello all"})
            for thread in threads:
                thread.join(20)

        assert len(results) == 3
        assert results[0] == results[1] == results[2]
        assert results[0][-1]["data"]["text"] == "For everyone"

    def test_a_page_that_goes_away_is_forgotten(self, app: FastAPI) -> None:
        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            handle = handle_of(app, made["id"])

            read_frames(http, f"/api/chats/{made['id']}/events", until=lambda frame: "retry" in frame)

            wait_until(lambda: not handle._waiters)

    def test_the_stream_ends_when_the_conversation_is_removed(self, app: FastAPI) -> None:
        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            remover = threading.Timer(0.5, lambda: http.delete(f"/api/chats/{made['id']}"))
            remover.start()

            frames = read_frames(http, f"/api/chats/{made['id']}/events")
            remover.join(10)

        assert frames[0] == {"retry": "2000"}

    def test_the_stream_needs_the_token_and_a_conversation_there_is(self, app: FastAPI) -> None:
        with serve(app, token=None) as (base, http):
            unsigned = http.get("/api/chats/anything/events")
            with httpx.Client(base_url=base, headers=BEARER) as signed:
                missing = signed.get("/api/chats/nothing/events")

        assert unsigned.status_code == 401
        assert unsigned.headers["content-type"].startswith("application/json")
        assert missing.status_code == 404
        assert error_of(missing)["code"] == "not_found"

    def test_the_stream_is_not_kept_or_buffered_on_the_way(self, app: FastAPI) -> None:
        with serve(app) as (base, http):
            made = http.post("/api/chats", json={"team": []}).json()
            with http.stream("GET", f"/api/chats/{made['id']}/events") as response:
                headers = dict(response.headers)

        assert headers["cache-control"] == "no-store"
        assert headers["x-accel-buffering"] == "no"
        assert headers["content-type"].startswith("text/event-stream")
