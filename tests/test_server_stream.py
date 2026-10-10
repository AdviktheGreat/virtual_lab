"""Tests of following a conversation as server-sent events."""

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from virtual_lab.events import ChatEvent
from virtual_lab.server.conversations import Conversations
from virtual_lab.server.stream import RETRY_MILLISECONDS, follow, format_event
from virtual_lab.ui.workspace import Settings, Workspace

from conftest import TEST_MODEL, FakeClient, text_response


@pytest.fixture(autouse=True)
def model_client(fake_client: FakeClient) -> FakeClient:
    return fake_client


@pytest.fixture
def conversations(tmp_path: Path):
    workspace = Workspace(tmp_path / "workspace")
    workspace.save_settings(Settings(model=TEST_MODEL, max_cost=None))
    kept = Conversations(workspace, check_keys=False)
    yield kept
    kept.close()


def parse(frame: str) -> dict[str, Any]:
    """A frame of the stream as its fields: id, event, and data, which is parsed."""
    fields: dict[str, Any] = {}
    for line in frame.strip("\n").split("\n"):
        name, _, value = line.partition(": ")
        fields[name] = value
    if "data" in fields:
        fields["data"] = json.loads(fields["data"])

    return fields


async def take(stream: AsyncIterator[str], count: int, timeout: float = 10.0) -> list[str]:
    frames: list[str] = []
    async with asyncio.timeout(timeout):
        async for frame in stream:
            frames.append(frame)
            if len(frames) == count:
                break

    return frames


def events_of(frames: list[str]) -> list[dict[str, Any]]:
    return [parse(frame) for frame in frames if frame.startswith("id:")]


class TestFormat:
    def test_an_event_is_sent_with_its_number_as_the_id_a_page_names_when_it_reconnects(self) -> None:
        event = ChatEvent(kind="answer", id=7, turn=2, speaker="Lead", text="Naïve — “quoted”\nline", time=1.5)

        frame = format_event(event)

        assert frame.endswith("\n\n")
        assert frame.startswith("id: 7\nevent: chat\ndata: ")
        # The text is in one line of data, since a newline in it would end the field
        assert len(frame.strip("\n").split("\n")) == 3
        assert "Naïve — “quoted”" in frame
        assert parse(frame)["data"] == event.to_dict()

    def test_a_line_separator_in_what_was_said_does_not_break_the_frame(self) -> None:
        event = ChatEvent(kind="answer", id=1, text="a\u2028b\u2029c\x85d\rpart")

        assert parse(format_event(event))["data"]["text"] == "a\u2028b\u2029c\x85d\rpart"
        assert len(format_event(event).strip("\n").split("\n")) == 3


class TestFollowing:
    def test_a_page_is_told_to_retry_and_then_of_everything_that_happened_in_order(
        self, conversations: Conversations, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": []})
        fake_client.completions.responses.append(text_response("The answer"))
        handle.chat.start("The question").join(20)
        known = handle.chat.events_since(0)

        async def main() -> list[str]:
            handle._loop = asyncio.get_running_loop()
            stream = follow(handle, 0, batch=0)
            frames = await take(stream, 1 + len(known))
            await stream.aclose()

            return frames

        frames = asyncio.run(main())

        assert frames[0] == f"retry: {RETRY_MILLISECONDS}\n\n"
        told = events_of(frames[1:])
        assert [item["data"]["id"] for item in told] == [event.id for event in known]
        assert [item["id"] for item in told] == [str(event.id) for event in known]
        assert [item["data"]["kind"] for item in told if item["data"]["kind"] in ("user", "answer")] == [
            "user",
            "answer",
        ]
        assert all(item["event"] == "chat" for item in told)

    def test_a_page_that_reconnects_is_told_only_of_what_came_after_the_last_it_saw(
        self, conversations: Conversations, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": []})
        handle.chat.start("Question").join(20)
        known = handle.chat.events_since(0)
        saw = known[1].id

        async def main() -> list[str]:
            handle._loop = asyncio.get_running_loop()
            stream = follow(handle, saw, batch=0)
            frames = await take(stream, 1 + len(known) - 2)
            await stream.aclose()

            return frames

        told = events_of(asyncio.run(main()))

        assert [item["data"]["id"] for item in told] == [event.id for event in known if event.id > saw]

    def test_a_page_with_everything_is_told_of_the_next_event_as_it_happens(
        self, conversations: Conversations, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": []})
        last = handle.chat.events_since(0)[-1].id if handle.chat.events_since(0) else 0
        fake_client.completions.responses.append(text_response("Late answer"))

        async def main() -> list[str]:
            loop = asyncio.get_running_loop()
            handle._loop = loop
            stream = follow(handle, last, batch=0)
            assert await take(stream, 1) == [f"retry: {RETRY_MILLISECONDS}\n\n"]
            # Nothing has happened, so the stream is quiet until something does
            quiet = asyncio.ensure_future(take(stream, 1, timeout=60))
            await asyncio.sleep(0.1)
            assert not quiet.done()
            loop.run_in_executor(None, handle.chat.start, "Now")
            frames = await quiet
            await stream.aclose()

            return frames

        told = events_of(asyncio.run(main()))

        assert told and told[0]["data"]["id"] > last

    def test_a_quiet_stream_sends_a_comment_now_and_then_so_that_it_is_not_taken_for_dead(
        self, conversations: Conversations
    ) -> None:
        handle = conversations.create({"team": []})
        last = max((event.id for event in handle.chat.events_since(0)), default=0)

        async def main() -> list[str]:
            handle._loop = asyncio.get_running_loop()
            stream = follow(handle, last, keepalive=0.05, batch=0)
            frames = await take(stream, 3)
            await stream.aclose()

            return frames

        assert asyncio.run(main())[1:] == [": keepalive\n\n", ": keepalive\n\n"]

    def test_a_stream_that_is_woken_with_nothing_new_goes_back_to_waiting_and_still_sends_its_comments(
        self, conversations: Conversations
    ) -> None:
        handle = conversations.create({"team": []})
        last = max((event.id for event in handle.chat.events_since(0)), default=0)

        async def main() -> list[str]:
            handle._loop = asyncio.get_running_loop()
            stream = follow(handle, last, keepalive=0.05, batch=0)
            await take(stream, 1)
            waiting = asyncio.ensure_future(take(stream, 2, timeout=5))
            await asyncio.sleep(0.01)
            handle.notify()
            frames = await waiting
            await stream.aclose()

            return frames

        assert asyncio.run(main()) == [": keepalive\n\n", ": keepalive\n\n"]

    def test_a_stream_ends_when_the_conversation_is_closed_having_told_of_everything(
        self, conversations: Conversations
    ) -> None:
        handle = conversations.create({"team": []})
        handle.chat.start("Question").join(20)
        known = handle.chat.events_since(0)
        threading.Timer(0.2, handle.close).start()

        async def main() -> list[str]:
            handle._loop = asyncio.get_running_loop()
            async with asyncio.timeout(10):
                return [frame async for frame in follow(handle, 0, batch=0)]

        frames = asyncio.run(main())

        assert len(events_of(frames)) == len(known)

    def test_what_happens_while_a_batch_is_being_waited_out_is_told_even_if_the_conversation_closes_then(
        self, conversations: Conversations, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": []})
        fake_client.completions.responses.append(text_response("First"))
        fake_client.completions.responses.append(text_response("Second"))
        handle.chat.start("One").join(20)

        def again_and_close() -> None:
            handle.chat.start("Two").join(20)
            handle.close()

        async def main() -> list[str]:
            loop = asyncio.get_running_loop()
            handle._loop = loop
            async with asyncio.timeout(20):
                frames = []
                stream = follow(handle, 0, batch=1.0)
                async for frame in stream:
                    frames.append(frame)
                    if len(frames) == 2:
                        # The first batch has been sent, and the stream is waiting out its pause
                        loop.run_in_executor(None, again_and_close)

                return frames

        told = events_of(asyncio.run(main()))

        assert [item["data"]["text"] for item in told if item["data"]["kind"] == "answer"] == ["First", "Second"]

    def test_a_page_that_goes_away_leaves_nothing_behind(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        async def main() -> tuple[int, int]:
            handle._loop = asyncio.get_running_loop()
            streams = [follow(handle, 0, batch=0) for _ in range(3)]
            for stream in streams:
                await take(stream, 1)
            following = len(handle._waiters)
            for stream in streams:
                await stream.aclose()

            return following, len(handle._waiters)

        assert asyncio.run(main()) == (3, 0)

    def test_a_page_that_is_cancelled_while_it_waits_leaves_nothing_behind(self, conversations: Conversations) -> None:
        handle = conversations.create({"team": []})

        async def main() -> int:
            handle._loop = asyncio.get_running_loop()
            last = max((event.id for event in handle.chat.events_since(0)), default=0)
            task = asyncio.ensure_future(take(follow(handle, last, batch=0), 2, timeout=60))
            await asyncio.sleep(0.1)
            assert len(handle._waiters) == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)

            return len(handle._waiters)

        assert asyncio.run(main()) == 0

    def test_a_reply_being_written_is_sent_as_it_is_and_then_whole(
        self, conversations: Conversations, fake_client: FakeClient
    ) -> None:
        handle = conversations.create({"team": [], "stream": True})
        fake_client.completions.responses.append(text_response("One two three four five"))

        async def main() -> list[dict[str, Any]]:
            loop = asyncio.get_running_loop()
            handle._loop = loop
            seen: list[dict[str, Any]] = []
            stream = follow(handle, 0)
            async with asyncio.timeout(20):
                await take(stream, 1)
                loop.run_in_executor(None, handle.chat.start, "Say it")
                async for frame in stream:
                    if frame.startswith("id:"):
                        seen.append(parse(frame))
                        if seen[-1]["data"]["kind"] == "answer":
                            break
            await stream.aclose()

            return seen

        seen = asyncio.run(main())

        assert seen[-1]["data"]["text"] == "One two three four five"
        ids = [int(item["id"]) for item in seen]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)
