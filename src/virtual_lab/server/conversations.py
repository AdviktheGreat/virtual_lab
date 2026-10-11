"""The conversations the server keeps: each in a directory of the workspace, opened when it is wanted
and kept open, with the session its code runs in, until the server stops or it is deleted.

    chats/2026-10-07_1500_what-is-a-nanobody/
        config.json     what it was set up with
        work/           where its code runs, which the files attached to it are saved in
        chat.json, messages.jsonl, events.jsonl, lab/    as Chat keeps them
"""

import asyncio
import re
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from openai import OpenAI, OpenAIError

from virtual_lab.chat import Chat
from virtual_lab.constants import CHAT_FILE_NAME, CHAT_UPLOADS_DIR_NAME, MAX_CHAT_TITLE_CHARS
from virtual_lab.events import ChatEvent
from virtual_lab.llm import ModelSource
from virtual_lab.server.config import ChatConfig, load_config, resolve_config, save_config
from virtual_lab.server.errors import ApiError, not_found
from virtual_lab.session import Session
from virtual_lab.ui.library import missing_keys
from virtual_lab.ui.workspace import Workspace, read_json
from virtual_lab.utils import CostUnknownError

# What a conversation's code runs in, under its directory
WORK_DIR_NAME = "work"

# What a conversation's id may be, which is the name of its directory
CHAT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Seconds to wait for a turn to end when a conversation is closed
CLOSE_TIMEOUT = 15.0


class KeysNotSetError(RuntimeError):
    """Raised by the stand-in for a model whose provider's key is not set."""


class MissingKeysModel(BaseChatModel):
    """The model of a conversation that was opened to be read while the key of its model's provider is not set.

    Building the real model fails without the key, which would leave a conversation that can be read
    unreadable until it is set. A message is not sent to one that has this: the server says which keys are
    missing first, and opens the conversation again once they are set.
    """

    needs: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "missing-keys"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "MissingKeysModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise KeysNotSetError(f"{', '.join(self.needs)} is not set")


class ChatHandle:
    """A conversation that is open: the Chat, the session its code runs in, and those following it.

    :param directory: The conversation's directory.
    :param config: What it was set up with.
    :param loop: The event loop those following it are on, which is told when anything happens.
    :param keyless: Whether it was opened with the key of its model's provider not set, so that it can be read
        but not asked anything.
    """

    def __init__(
        self, directory: Path, config: ChatConfig, loop: asyncio.AbstractEventLoop | None, keyless: bool = False
    ) -> None:
        self.directory = directory
        self.config = config
        self.id = directory.name
        self.chat: Chat
        self.session: Session | None = None
        self.keyless = keyless
        self.closed = False
        self._loop = loop
        self._waiters: set[asyncio.Event] = set()

    # Those following the conversation, who are on the event loop, are woken from whichever thread
    # something happens in, and look at the conversation for themselves

    def subscribe(self) -> asyncio.Event:
        """Registers someone following the conversation. Set when something has happened. Call it on the loop."""
        waiter = asyncio.Event()
        self._waiters.add(waiter)

        return waiter

    def unsubscribe(self, waiter: asyncio.Event) -> None:
        self._waiters.discard(waiter)

    def notify(self, event: ChatEvent | None = None) -> None:
        """Wakes everyone following the conversation. Safe from any thread, and the Chat's on_event."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._wake)
        except RuntimeError:
            # The loop has closed, and no one is left to tell
            pass

    def _wake(self) -> None:
        for waiter in list(self._waiters):
            waiter.set()

    @property
    def files_root(self) -> Path:
        """Where the conversation's files are, from which their paths are given: its session's directory, or,
        without a session, the conversation's own."""
        return self.chat.uploads_dir.parent

    @property
    def files_shown(self) -> tuple[str, ...] | None:
        """The parts of files_root that are shown, or None for all of it. Without a session, that is the files
        attached to the conversation, and not what it saves of itself."""
        return None if self.session is not None else (CHAT_UPLOADS_DIR_NAME,)

    def info(self) -> dict[str, Any]:
        return {**self.chat.describe(), "config": self.config.to_dict()}

    def close(self) -> None:
        """Ends the turn that is running, if one is, and closes the session, keeping the files."""
        if self.closed:
            return
        self.closed = True
        try:
            self.chat.close(timeout=CLOSE_TIMEOUT)
        finally:
            if self.session is not None:
                self.session.close()
            self.notify()


class Conversations:
    """Every conversation in a workspace, and those of them that are open.

    :param workspace: Where they are kept.
    :param check_keys: Whether to say, before a conversation is made or a message sent, that the key its
        model needs is not set, rather than after something has gone wrong.
    :param client: An OpenAI client for every conversation, as Chat takes one.
    :param chat_models: Chat models for every conversation, as Chat takes them.
    """

    def __init__(
        self,
        workspace: Workspace,
        check_keys: bool = True,
        client: OpenAI | None = None,
        chat_models: ModelSource | None = None,
    ) -> None:
        self.workspace = workspace
        self.check_keys = check_keys
        self.client = client
        self.chat_models = chat_models
        self.loop: asyncio.AbstractEventLoop | None = None
        self._open: dict[str, ChatHandle] = {}
        self._deleting: set[str] = set()
        self._lock = threading.RLock()

    def directory(self, chat_id: str) -> Path:
        """A conversation's directory, if there is one with this id."""
        with self._lock:
            deleting = chat_id in self._deleting
        if deleting or not CHAT_ID_PATTERN.fullmatch(chat_id):
            raise not_found("conversation with that id")
        directory = self.workspace.chats_dir / chat_id
        if directory.resolve().parent != self.workspace.chats_dir or not (directory / CHAT_FILE_NAME).is_file():
            raise not_found("conversation with that id")

        return directory

    def summaries(self) -> list[dict[str, Any]]:
        """Every conversation, the latest first, as a list shows them. Those that are open say what they are doing."""
        found: list[dict[str, Any]] = []
        for directory in self.workspace.chats_dir.iterdir():
            info = read_json(directory / CHAT_FILE_NAME)
            if not isinstance(info, dict):
                continue
            with self._lock:
                handle = self._open.get(directory.name)
            if handle is not None:
                state = handle.chat.state
            else:
                # A conversation not open is not running, so one saved as running was cut off
                state = "interrupted" if info.get("running") else "idle"
            config = load_config(directory)
            found.append(
                {
                    "id": directory.name,
                    "title": info.get("title") or "",
                    "created_at": info.get("created_at"),
                    "updated_at": info.get("updated_at"),
                    "state": state,
                    "turns": info.get("turns", 0),
                    "spent": info.get("spent"),
                    "max_cost": info.get("max_cost"),
                    "model": config.model if config is not None else None,
                }
            )

        return sorted(found, key=lambda item: (item["updated_at"] or "", item["id"]), reverse=True)

    def create(self, request: Mapping[str, Any], title: str | None = None) -> ChatHandle:
        """Makes a conversation and opens it.

        :param request: What it is set up with, as resolve_config takes it.
        :param title: What to call it; by default its first message.
        :raises ApiError: If something chosen cannot be used, or a key is not set.
        """
        title = " ".join((title or "").split())[:MAX_CHAT_TITLE_CHARS] or None
        try:
            config = resolve_config(request, self.workspace.load_settings(), self.workspace)
        except (ValueError, TypeError) as error:
            raise ApiError(422, "invalid", str(error)) from error
        self.require_keys(config)

        directory = self.workspace.new_directory("chat", title or "New chat")
        try:
            save_config(directory, config)
            handle = self._start(directory, config, title)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        with self._lock:
            self._open[handle.id] = handle

        return handle

    def get(self, chat_id: str) -> ChatHandle:
        """A conversation, opened if it is not.

        :raises ApiError: If there is none with this id, or it cannot be opened.
        """
        with self._lock:
            handle = self._open.get(chat_id)
            if handle is not None and not (handle.keyless and not self.missing_keys(handle.config)):
                return handle

            directory = self.directory(chat_id)
            config = load_config(directory)
            if config is None:
                raise ApiError(409, "unreadable", "This conversation's setup cannot be read, so it cannot be opened")
            reopened = self._start(directory, config, None, keyless=bool(self.missing_keys(config)))
            self._open[chat_id] = reopened

        # Its keys have been set since it was opened without them, so those following it are told it is
        # over and open it again, with no event missed, since the new one carries on from the same events
        if handle is not None:
            handle.close()

        return reopened

    def ready(self, chat_id: str) -> ChatHandle:
        """A conversation, opened to be asked something.

        :raises ApiError: If there is none with this id, it cannot be opened, or a key its model needs is not set.
        """
        handle = self.get(chat_id)
        self.require_keys(handle.config)
        if handle.keyless:
            # The keys were set between opening it and looking, so it is opened again with them
            handle = self.get(chat_id)

        return handle

    def delete(self, chat_id: str) -> None:
        """Stops a conversation, if it is running, and removes it and everything it saved."""
        with self._lock:
            directory = self.directory(chat_id)
            handle = self._open.pop(chat_id, None)
            # Until it is gone, it cannot be opened again, which would leave a conversation open on nothing
            self._deleting.add(chat_id)
        try:
            if handle is not None:
                handle.close()
            shutil.rmtree(directory)
        finally:
            with self._lock:
                self._deleting.discard(chat_id)

    def close(self) -> None:
        """Closes every conversation that is open, which the server does as it stops."""
        with self._lock:
            handles = list(self._open.values())
            self._open.clear()
        for handle in handles:
            handle.close()

    def missing_keys(self, config: ChatConfig) -> list[str]:
        """The variables its model's provider needs that are not set, which are none when keys are not checked
        or the conversation's models are given."""
        if not self.check_keys or self.client is not None or self.chat_models is not None:
            return []

        return missing_keys([config.model])

    def require_keys(self, config: ChatConfig) -> None:
        """Says what is missing before something that spends is done, rather than after it has gone wrong.

        :raises ApiError: If a key the model needs is not set.
        """
        if missing := self.missing_keys(config):
            raise ApiError(
                422,
                "missing_keys",
                f"{config.model} needs {', '.join(missing)}, which is not set. Add it in the settings.",
                missing=missing,
            )

    def _start(self, directory: Path, config: ChatConfig, title: str | None, keyless: bool = False) -> ChatHandle:
        handle = ChatHandle(directory, config, self.loop, keyless)
        session = config.code.session(directory / WORK_DIR_NAME)
        chat_models = MissingKeysModel(needs=tuple(self.missing_keys(config))) if keyless else self.chat_models
        try:
            handle.chat = Chat(
                directory,
                config.lead_agent(),
                config.team_agents(),
                session=session,
                client=self.client,
                chat_models=chat_models,
                title=title,
                max_cost=config.max_cost,
                stream=config.stream,
                resources=config.code.resources if session is not None else "none",
                commercial_mode=config.commercial_mode,
                on_event=handle.notify,
            )
        except (ValueError, CostUnknownError, OpenAIError, ImportError) as error:
            if session is not None:
                session.close()
            raise ApiError(422, "unavailable", str(error)) from error
        handle.session = session

        return handle
