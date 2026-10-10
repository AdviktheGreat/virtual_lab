"""Tests of the conversations the server keeps, opens, and removes."""

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from virtual_lab import llm as llm_module
from virtual_lab.constants import CHAT_FILE_NAME
from virtual_lab.server.config import CONFIG_FILE_NAME
from virtual_lab.server.conversations import (
    WORK_DIR_NAME,
    ChatHandle,
    Conversations,
    KeysNotSetError,
    MissingKeysModel,
)
from virtual_lab.server.errors import ApiError
from virtual_lab.session import LocalSession
from virtual_lab.ui.workspace import Settings, Workspace

from conftest import TEST_MODEL, FakeClient, fake_llm, text_response

TEAM = ["Scientific Critic", "Immunologist"]


@pytest.fixture(autouse=True)
def model_client(fake_client: FakeClient) -> FakeClient:
    """A conversation builds its models when it is opened, so the fake must be there before that."""
    return fake_client


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    store = Workspace(tmp_path / "workspace")
    store.save_settings(Settings(model=TEST_MODEL, max_cost=None))

    return store


@pytest.fixture
def conversations(workspace: Workspace):
    kept = Conversations(workspace, check_keys=False)
    yield kept
    kept.close()


def say(handle: ChatHandle, message: str = "Hello") -> None:
    handle.chat.start(message).join(20)
    assert handle.chat.state == "idle"


def directories(workspace: Workspace) -> list[str]:
    return sorted(path.name for path in workspace.chats_dir.iterdir())


class TestCreating:
    def test_a_conversation_is_made_in_a_directory_of_its_own_with_what_it_was_set_up_with(
        self, conversations: Conversations, workspace: Workspace
    ) -> None:
        handle = conversations.create({"team": TEAM, "max_cost": 1.5}, "What is a nanobody?")

        assert handle.directory.parent == workspace.chats_dir
        assert handle.directory.name.endswith("_what-is-a-nanobody")
        assert handle.id == handle.directory.name
        assert handle.chat.title == "What is a nanobody?"
        assert (handle.directory / CHAT_FILE_NAME).is_file()
        saved = json.loads((handle.directory / CONFIG_FILE_NAME).read_text())
        assert saved["model"] == TEST_MODEL
        assert [member["title"] for member in saved["team"]] == TEAM
        assert saved["max_cost"] == 1.5
        info = handle.info()
        assert info["id"] == handle.id
        assert info["config"] == saved
        assert info["team"] == TEAM
        assert info["state"] == "idle"
        assert info["can_run_code"] is False

    def test_a_conversation_without_a_title_is_given_one_by_its_first_message(
        self, conversations: Conversations
    ) -> None:
        handle = conversations.create({"team": []})

        assert handle.chat.title == ""
        assert "new-chat" in handle.directory.name
        say(handle, "How do I fold a protein?")
        assert handle.chat.title == "How do I fold a protein?"

    def test_a_title_is_made_one_line_and_short(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []}, "  A   very\nlong " + "title " * 40)

        assert "\n" not in handle.chat.title
        assert len(handle.chat.title) <= 80

    def test_two_conversations_of_one_name_in_one_minute_have_directories_of_their_own(
        self, conversations: Conversations
    ) -> None:
        first = conversations.create({"team": []}, "Same")
        second = conversations.create({"team": []}, "Same")

        assert first.id != second.id
        assert first.chat is not second.chat

    def test_where_code_runs_there_is_a_session_in_the_directory_of_the_conversation(
        self, conversations: Conversations
    ) -> None:
        handle = conversations.create({"team": [], "code": {"where": "local"}})

        assert isinstance(handle.session, LocalSession)
        assert handle.session.directory == (handle.directory / WORK_DIR_NAME).resolve()
        assert handle.chat.session is handle.session
        assert handle.info()["can_run_code"] is True
        assert handle.files_root == handle.session.directory
        assert handle.files_shown is None

    def test_where_code_does_not_run_only_the_files_attached_are_shown(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        assert handle.session is None
        assert handle.files_root == handle.directory
        assert handle.files_shown == ("uploads",)

    @pytest.mark.parametrize(
        "request_", [{"team": ["Alchemist"]}, {"model": "nothing-serves-this"}, {"code": {"where": "cloud"}}]
    )
    def test_something_that_cannot_be_used_is_refused_and_leaves_nothing_behind(
        self, conversations: Conversations, workspace: Workspace, request_: dict[str, Any]
    ) -> None:
        with pytest.raises(ApiError) as caught:
            conversations.create(request_)

        assert (caught.value.status, caught.value.code) == (422, "invalid")
        assert directories(workspace) == []

    def test_a_conversation_that_cannot_be_started_is_refused_and_leaves_nothing_behind(
        self, workspace: Workspace
    ) -> None:
        kept = Conversations(workspace, check_keys=False)

        with pytest.raises(ApiError) as caught:
            # A model with no price cannot be held to a limit on what is spent
            kept.create({"model": "o1-priced-by-nobody", "team": [], "max_cost": 1.0})

        assert (caught.value.status, caught.value.code) == (422, "unavailable")
        assert "max_cost cannot be enforced" in caught.value.message
        assert directories(workspace) == []

    def test_a_failed_start_closes_the_session_it_made(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[bool] = []
        monkeypatch.setattr(LocalSession, "close", lambda self: closed.append(True))
        kept = Conversations(workspace, check_keys=False)

        with pytest.raises(ApiError):
            kept.create({"model": "o1-priced-by-nobody", "team": [], "max_cost": 1.0, "code": {"where": "local"}})

        assert closed == [True]

    def test_what_a_conversation_is_set_up_with_is_what_its_chat_is_given(self, conversations: Conversations) -> None:
        handle = conversations.create(
            {
                "team": [],
                "stream": True,
                "max_cost": 2.5,
                "commercial_mode": True,
                "code": {"where": "local", "resources": "retrieve"},
            },
            "Set up",
        )

        chat = handle.chat
        assert (chat.stream, chat.max_cost, chat.commercial_mode, chat.resources) == (True, 2.5, True, "retrieve")
        assert chat.title == "Set up"
        assert chat.session is handle.session

        plain = conversations.create({"team": [], "stream": False, "max_cost": None, "commercial_mode": False})

        assert (plain.chat.stream, plain.chat.max_cost, plain.chat.commercial_mode) == (False, None, False)

    def test_where_no_code_runs_nothing_is_said_of_resources_to_use(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": [], "code": {"where": "none", "resources": "retrieve"}})

        assert handle.chat.resources == "none"

    def test_models_given_to_the_conversations_are_used_and_need_no_key(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        kept = Conversations(workspace, client=fake_client)
        fake_client.completions.responses.append(text_response("From the client I was given"))

        handle = kept.create({"team": []})
        say(handle)

        assert handle.chat.last_reply is not None
        assert handle.chat.last_reply.text == "From the client I was given"
        kept.close()

    def test_chat_models_given_to_the_conversations_need_no_key_either(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        kept = Conversations(workspace, chat_models={TEST_MODEL: fake_llm(fake_client)})
        fake_client.completions.responses.append(text_response("From the model I was given"))

        handle = kept.create({"team": []})
        say(handle)

        assert kept.missing_keys(handle.config) == []
        assert handle.chat.last_reply is not None
        assert handle.chat.last_reply.text == "From the model I was given"
        kept.close()


class TestOpening:
    def test_a_conversation_that_is_open_is_the_same_every_time(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        assert conversations.get(handle.id) is handle

    def test_a_conversation_is_opened_from_its_directory_as_it_was_left(
        self, workspace: Workspace, fake_client: FakeClient
    ) -> None:
        first = Conversations(workspace, check_keys=False)
        handle = first.create({"team": TEAM}, "Kept")
        fake_client.completions.responses.append(text_response("Kept answer"))
        say(handle, "Remember this")
        events = [event.to_dict() for event in handle.chat.events_since(0)]
        first.close()

        reopened = Conversations(workspace, check_keys=False).get(handle.id)

        assert reopened is not handle
        assert reopened.config == handle.config
        assert reopened.chat.title == "Kept"
        assert reopened.chat.turns == 1
        assert reopened.chat.state == "idle"
        assert [event.to_dict() for event in reopened.chat.events_since(0) if event.kind != "status"] == [
            event for event in events if event["kind"] != "status"
        ]
        fake_client.completions.responses.append(text_response("Second answer"))
        say(reopened, "And this")
        assert reopened.chat.last_reply is not None and reopened.chat.last_reply.text == "Second answer"
        assert reopened.chat.turns == 2

    @pytest.mark.parametrize("chat_id", ["nothing", "../workspace", "..", ".hidden", "a/b", "a b", "x" * 200, ""])
    def test_there_is_no_conversation_under_a_name_that_is_not_the_name_of_one(
        self, conversations: Conversations, chat_id: str
    ) -> None:
        for call in (conversations.get, conversations.delete, conversations.ready):
            with pytest.raises(ApiError) as caught:
                call(chat_id)
            assert (caught.value.status, caught.value.code) == (404, "not_found")

    def test_a_directory_that_is_not_a_conversation_is_not_one(
        self, conversations: Conversations, workspace: Workspace, tmp_path: Path
    ) -> None:
        (workspace.chats_dir / "empty").mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / CHAT_FILE_NAME).write_text("{}")
        (workspace.chats_dir / "linked").symlink_to(outside, target_is_directory=True)

        for name in ("empty", "linked"):
            with pytest.raises(ApiError) as caught:
                conversations.get(name)
            assert caught.value.status == 404

    def test_a_conversation_whose_setup_cannot_be_read_is_not_opened(
        self, conversations: Conversations, workspace: Workspace
    ) -> None:
        handle = conversations.create({"team": []})
        conversations.close()
        (handle.directory / CONFIG_FILE_NAME).write_text("not json")

        with pytest.raises(ApiError) as caught:
            Conversations(workspace, check_keys=False).get(handle.id)

        assert (caught.value.status, caught.value.code) == (409, "unreadable")


class TestListing:
    def test_conversations_are_listed_latest_first_with_what_a_list_shows(
        self, conversations: Conversations, fake_client: FakeClient
    ) -> None:
        older = conversations.create({"team": []}, "Older")
        newer = conversations.create({"team": []}, "Newer")
        fake_client.completions.responses.append(text_response("Done"))
        say(newer, "Go")
        time.sleep(0.01)
        fake_client.completions.responses.append(text_response("Done"))
        say(older, "Go")

        listed = conversations.summaries()

        assert [item["id"] for item in listed] == [older.id, newer.id]
        item = listed[0]
        assert item["title"] == "Older"
        assert item["state"] == "idle"
        assert item["turns"] == 1
        assert item["model"] == TEST_MODEL
        assert item["spent"] is not None and item["spent"] > 0
        assert item["max_cost"] is None
        assert item["created_at"] and item["updated_at"]

    def test_conversations_changed_at_the_same_moment_are_listed_in_the_same_order_every_time(
        self, conversations: Conversations
    ) -> None:
        made = [conversations.create({"team": []}, name) for name in ("A", "B", "C")]
        for handle in made:
            path = handle.directory / CHAT_FILE_NAME
            path.write_text(json.dumps({**json.loads(path.read_text()), "updated_at": "2026-01-01T00:00:00+00:00"}))

        listed = [item["id"] for item in conversations.summaries()]

        assert listed == sorted((handle.id for handle in made), reverse=True)

    def test_an_open_conversation_says_what_it_is_doing(
        self, conversations: Conversations, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": []})
        gate = hold_the_model(fake_client)
        handle.chat.start("Go")
        wait_until(lambda: gate.reached.is_set())

        assert [item["state"] for item in conversations.summaries()] == ["running"]

        gate.release()
        handle.chat.close(10)

    def test_a_conversation_saved_as_running_that_is_not_open_was_cut_off(
        self, workspace: Workspace, conversations: Conversations
    ) -> None:
        handle = conversations.create({"team": []})
        conversations.close()
        info = json.loads((handle.directory / CHAT_FILE_NAME).read_text())
        (handle.directory / CHAT_FILE_NAME).write_text(json.dumps({**info, "running": True}))

        assert [item["state"] for item in Conversations(workspace, check_keys=False).summaries()] == ["interrupted"]

    def test_what_is_not_a_conversation_is_not_listed(self, conversations: Conversations, workspace: Workspace) -> None:
        kept = conversations.create({"team": []})
        (workspace.chats_dir / "empty").mkdir()
        (workspace.chats_dir / "garbage").mkdir()
        (workspace.chats_dir / "garbage" / CHAT_FILE_NAME).write_text("not json")
        (workspace.chats_dir / "a-file.txt").write_text("x")

        assert [item["id"] for item in conversations.summaries()] == [kept.id]

    def test_a_conversation_whose_setup_is_lost_is_listed_without_a_model(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})
        (handle.directory / CONFIG_FILE_NAME).unlink()

        assert conversations.summaries()[0]["model"] is None


class TestClosing:
    def test_once_every_conversation_is_closed_none_is_kept_open(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        conversations.close()

        assert handle.closed
        reopened = conversations.get(handle.id)
        assert reopened is not handle
        assert not reopened.closed


class TestRemoving:
    def test_a_conversation_is_removed_with_everything_it_saved(
        self, conversations: Conversations, workspace: Workspace, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": []})
        say(handle)

        conversations.delete(handle.id)

        assert directories(workspace) == []
        assert handle.closed
        with pytest.raises(ApiError):
            conversations.get(handle.id)

    def test_a_conversation_that_is_not_open_is_removed_too(
        self, workspace: Workspace, conversations: Conversations
    ) -> None:
        handle = conversations.create({"team": []})
        conversations.close()

        Conversations(workspace, check_keys=False).delete(handle.id)

        assert directories(workspace) == []

    def test_a_conversation_that_is_working_is_stopped_before_it_is_removed(
        self, conversations: Conversations, workspace: Workspace, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": []})
        gate = hold_the_model(fake_client)
        handle.chat.start("Go")
        wait_until(lambda: gate.reached.is_set())
        # The model answers a little after the conversation is told to stop
        threading.Timer(0.2, gate.release).start()

        conversations.delete(handle.id)

        assert handle.closed
        assert handle.chat.state == "idle"
        assert directories(workspace) == []

    def test_closing_closes_every_open_conversation_and_leaves_the_files(
        self, conversations: Conversations, workspace: Workspace
    ) -> None:
        handles = [conversations.create({"team": []}) for _ in range(2)]

        conversations.close()
        conversations.close()

        assert all(handle.closed for handle in handles)
        assert len(directories(workspace)) == 2

    def test_the_session_is_closed_with_the_conversation(
        self, conversations: Conversations, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[bool] = []
        monkeypatch.setattr(LocalSession, "close", lambda self: closed.append(True))
        handle = conversations.create({"team": [], "code": {"where": "local"}})

        handle.close()
        handle.close()

        assert closed == [True]


class TestKeys:
    @pytest.fixture(autouse=True)
    def without_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_a_conversation_is_not_made_for_a_model_whose_key_is_not_set(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kept = Conversations(workspace)

        with pytest.raises(ApiError) as caught:
            kept.create({"team": []})

        error = caught.value
        assert (error.status, error.code) == (422, "missing_keys")
        assert error.details == {"missing": ["OPENAI_API_KEY"]}
        assert "OPENAI_API_KEY" in error.message
        assert directories(workspace) == []

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert kept.create({"team": []}).chat.state == "idle"
        kept.close()

    def test_a_conversation_is_opened_to_be_read_without_its_key_and_not_asked_anything(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        first = Conversations(workspace)
        handle = first.create({"team": []}, "Old")
        fake_client.completions.responses.append(text_response("It was answered"))
        say(handle, "A question from before")
        first.close()
        monkeypatch.delenv("OPENAI_API_KEY")
        # The real models are not built, which would fail for want of the key
        monkeypatch.setattr(llm_module, "get_llm", fail_if_built)

        kept = Conversations(workspace)
        opened = kept.get(handle.id)

        assert opened.keyless
        assert isinstance(opened.chat._llms[TEST_MODEL], MissingKeysModel)
        assert [event.text for event in opened.chat.events_since(0) if event.kind in ("user", "answer")] == [
            "A question from before",
            "It was answered",
        ]
        assert kept.get(handle.id) is opened
        with pytest.raises(ApiError) as caught:
            kept.ready(handle.id)
        assert (caught.value.status, caught.value.code) == (422, "missing_keys")
        kept.close()

    def test_a_conversation_opened_without_its_key_is_opened_again_when_the_key_is_set(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        first = Conversations(workspace)
        handle = first.create({"team": []}, "Old")
        say(handle, "Before")
        first.close()
        monkeypatch.delenv("OPENAI_API_KEY")
        kept = Conversations(workspace)
        keyless = kept.get(handle.id)
        assert keyless.keyless
        before = [event.id for event in keyless.chat.events_since(0)]

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        again = kept.ready(handle.id)

        assert again is not keyless
        assert not again.keyless
        assert kept.get(handle.id) is again
        # Those following the one that was open are told it is over, so they follow the one that is
        assert keyless.closed
        assert again.chat.turns == 1
        fake_client.completions.responses.append(text_response("Now it answers"))
        say(again, "After")
        assert again.chat.last_reply is not None and again.chat.last_reply.text == "Now it answers"
        # The new one carries on from the same events, so that a page that follows it misses none
        after = [event.id for event in again.chat.events_since(0)]
        assert after[: len(before)] == before
        assert len(after) > len(before)
        assert after == sorted(set(after))
        kept.close()

    def test_keys_set_between_looking_at_a_conversation_and_asking_it_something_open_it_again(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        first = Conversations(workspace)
        handle = first.create({"team": []}, "Old")
        first.close()
        monkeypatch.delenv("OPENAI_API_KEY")
        kept = Conversations(workspace)
        assert kept.get(handle.id).keyless
        require = kept.require_keys

        def keys_arrive_then_require(config: Any) -> None:
            monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
            require(config)

        monkeypatch.setattr(kept, "require_keys", keys_arrive_then_require)

        ready = kept.ready(handle.id)

        assert not ready.keyless
        assert not isinstance(ready.chat._llms[TEST_MODEL], MissingKeysModel)
        kept.close()

    def test_the_stand_in_for_a_model_says_what_is_missing_if_it_is_asked(self) -> None:
        model = MissingKeysModel(needs=("OPENAI_API_KEY",))

        assert model.bind_tools([]) is model
        with pytest.raises(KeysNotSetError, match="OPENAI_API_KEY is not set"):
            model.invoke("Hello")

    def test_keys_are_not_looked_for_when_asked_not_to(self, workspace: Workspace) -> None:
        kept = Conversations(workspace, check_keys=False)

        assert kept.missing_keys(kept.create({"team": []}).config) == []
        kept.close()


def fail_if_built(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("A model was built for a conversation that is only read")


class Gate:
    """Holds the model's answer until it is let go, and says when it was asked."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self._release = threading.Event()

    def release(self) -> None:
        self._release.set()

    def wait(self) -> None:
        self.reached.set()
        assert self._release.wait(20), "The model was never let go"


def hold_the_model(fake_client: FakeClient) -> Gate:
    gate = Gate()
    create = fake_client.completions.create

    def create_when_let_go(**kwargs: Any) -> Any:
        gate.wait()
        return create(**kwargs)

    fake_client.completions.create = create_when_let_go  # type: ignore[method-assign]

    return gate


def wait_until(condition: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "Waited for something that did not happen"
        time.sleep(0.005)


class TestFollowers:
    def test_those_following_a_conversation_are_woken_from_any_thread(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        async def main() -> list[bool]:
            handle._loop = asyncio.get_running_loop()
            first, second = handle.subscribe(), handle.subscribe()
            third = handle.subscribe()
            handle.unsubscribe(third)
            woken = [first.is_set(), second.is_set()]
            await asyncio.get_running_loop().run_in_executor(None, handle.notify)
            await asyncio.wait_for(first.wait(), 5)
            await asyncio.wait_for(second.wait(), 5)

            return [*woken, first.is_set(), second.is_set(), third.is_set()]

        assert asyncio.run(main()) == [False, False, True, True, False]

    def test_an_event_wakes_them_as_the_chats_callback(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        async def main() -> bool:
            handle._loop = asyncio.get_running_loop()
            waiter = handle.subscribe()
            await asyncio.get_running_loop().run_in_executor(None, lambda: handle.chat.add_note("x"))
            handle.chat.rename("Renamed")
            await asyncio.wait_for(waiter.wait(), 5)

            return waiter.is_set()

        assert asyncio.run(main())

    def test_nobody_is_woken_when_there_is_no_loop_or_it_is_gone(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})
        handle.notify()
        loop = asyncio.new_event_loop()
        handle._loop = loop
        loop.close()

        handle.notify()

    def test_a_loop_that_is_stopping_is_not_woken_either(
        self, conversations: Conversations, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handle = conversations.create({"team": []})
        loop = asyncio.new_event_loop()
        handle._loop = loop

        def refuse(*args: Any) -> None:
            raise RuntimeError("Event loop is closed")

        monkeypatch.setattr(loop, "call_soon_threadsafe", refuse)
        handle.notify()
        loop.close()

    def test_closing_wakes_them_and_ends_the_conversation(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        async def main() -> bool:
            handle._loop = asyncio.get_running_loop()
            waiter = handle.subscribe()
            await asyncio.get_running_loop().run_in_executor(None, handle.close)
            await asyncio.wait_for(waiter.wait(), 5)

            return handle.closed

        assert asyncio.run(main())
