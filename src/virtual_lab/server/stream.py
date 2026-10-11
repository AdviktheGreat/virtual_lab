"""Following a conversation as server-sent events.

A page opens one stream per conversation it is showing, naming the last event it has, and is told of
every event after it, then of each as it happens. The server keeps nothing for it: what is sent is read
from the conversation, so a page that is slow is told of the latest of a reply being written rather than
of every word, one that reconnects with the number of the last event it saw misses nothing, and a page
that goes away costs nothing.
"""

import asyncio
import json
from collections.abc import AsyncIterator

from virtual_lab.events import ChatEvent
from virtual_lab.server.conversations import ChatHandle

# Seconds of quiet after which a comment is sent, so that proxies and the browser do not take the
# stream for dead
KEEPALIVE_SECONDS = 15.0

# Seconds a stream waits after sending what it had, so that a reply written word by word is sent in
# batches, not word by word
BATCH_SECONDS = 0.03

# Seconds a page waits before it opens the stream again, if it was cut off
RETRY_MILLISECONDS = 2000


def format_event(event: ChatEvent) -> str:
    """An event as the stream sends it: its number is the id the page names when it reconnects."""
    data = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))

    return f"id: {event.id}\nevent: chat\ndata: {data}\n\n"


async def follow(
    handle: ChatHandle, after: int = 0, keepalive: float = KEEPALIVE_SECONDS, batch: float = BATCH_SECONDS
) -> AsyncIterator[str]:
    """What to send a page following a conversation, until the conversation is closed or the page goes.

    :param handle: The conversation.
    :param after: The number of the last event the page has.
    :param keepalive: Seconds of quiet before a comment is sent.
    :param batch: Seconds to wait after sending events, to send what comes next together.
    """
    waiter = handle.subscribe()
    try:
        yield f"retry: {RETRY_MILLISECONDS}\n\n"
        while True:
            # Cleared before looking, so that nothing that happens after the look is missed
            waiter.clear()
            events = handle.chat.events_since(after)
            for event in events:
                yield format_event(event)
            if events:
                after = events[-1].id
                if batch:
                    await asyncio.sleep(batch)
                continue
            if handle.closed:
                return
            try:
                await asyncio.wait_for(waiter.wait(), keepalive)
            except TimeoutError:
                yield ": keepalive\n\n"
    finally:
        handle.unsubscribe(waiter)
