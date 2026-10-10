"""The server's application: what a page asks of the lab, and what the lab tells it.

Every route is under /api and answers in JSON, and errors do too, in one shape:
{"error": {"code": ..., "message": ...}}. The conversation's events come as server-sent events,
and everything else is a request and an answer.

    GET    /api/health                  that the server is there, which needs no token
    GET    /api/settings, PATCH         what a new conversation starts with
    GET    /api/models                  the models to choose from, and whether they can be reached
    GET    /api/keys                    which providers have their keys
    PUT    /api/keys/{name}, DELETE     a key, set until the server stops, or taken away
    GET    /api/scientists, PUT         the library and the scientists the person described
    DELETE /api/scientists/{title}      one of those the person described
    GET    /api/chats, POST             the conversations, and a new one
    GET    /api/chats/{id}              one conversation, which PATCH renames and DELETE removes
    POST   /api/chats/{id}/messages     a message for the lead, which is answered in the background
    POST   /api/chats/{id}/note         a note for the lead or the team, which they read as they work
    POST   /api/chats/{id}/pause        also resume and stop, steering the turn that is running
    GET    /api/chats/{id}/events       the events of the conversation, as they happen
    POST   /api/chats/{id}/uploads      a file for the conversation, as the body, named by ?name=
    GET    /api/chats/{id}/files        what the conversation made or was given, and /files/{path} one of them
"""

import asyncio
import os
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast

import anyio.to_thread
from fastapi import APIRouter, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from virtual_lab.__about__ import __version__
from virtual_lab.chat import ChatBusyError, ChatClosedError, UploadTooLargeError
from virtual_lab.constants import MAX_CHAT_TITLE_CHARS
from virtual_lab.server.config import describe_models, library, public_settings, update_settings
from virtual_lab.server.conversations import Conversations
from virtual_lab.server.errors import ApiError, not_found
from virtual_lab.server.files import file_response, list_files, resolve_inside
from virtual_lab.server.security import LOCAL_HOSTS, Guard
from virtual_lab.server.stream import follow
from virtual_lab.ui.library import SETTABLE_KEYS, provider_statuses
from virtual_lab.ui.workspace import DEFAULT_WORKSPACE, Workspace

# The most characters of a message, which no one writes by hand
MAX_MESSAGE_CHARS = 200_000

# Seconds the server waits for the next part of a file that is being uploaded
UPLOAD_CHUNK_TIMEOUT = 120.0

# What a key may be at most, and the characters it may not hold, which an environment variable cannot
MAX_KEY_CHARS = 4096

# What a page of the interface is given to keep: what has a hash in its name never changes
IMMUTABLE_PREFIX = "_next/static/"


class RequestModel(BaseModel):
    """What the body of a request is: nothing may be in it that is not asked for."""

    model_config = ConfigDict(extra="forbid")


class ScientistBody(RequestModel):
    title: str = Field(min_length=1, max_length=200)
    expertise: str = Field(min_length=1, max_length=4000)
    goal: str = Field(min_length=1, max_length=4000)
    role: str = Field(min_length=1, max_length=4000)


class CodeBody(RequestModel):
    where: str = "none"
    sandbox: str = "python"
    network: bool = True
    python: str = ""
    resources: str = "retrieve"


class ChatBody(RequestModel):
    """A new conversation. What is left out is what the settings say, or the library's lead and team."""

    title: str | None = Field(None, max_length=500)
    model: str = ""
    lead: str | ScientistBody | None = None
    team: list[str | ScientistBody] | None = None
    code: CodeBody | None = None
    max_cost: float | None = None
    stream: bool | None = None
    commercial_mode: bool = False


class SettingsBody(RequestModel):
    model: str = ""
    stream: bool = True
    max_cost: float | None = None
    code: str = "none"
    sandbox: str = "python"
    network: bool = True
    python: str = ""


class RenameBody(RequestModel):
    title: str = Field(min_length=1, max_length=500)


class MessageBody(RequestModel):
    text: str = Field("", max_length=MAX_MESSAGE_CHARS)
    attachments: list[str] = Field(default_factory=list, max_length=100)


class NoteBody(RequestModel):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class KeyBody(RequestModel):
    value: str = Field(max_length=MAX_KEY_CHARS)


class RequestBodyReader:
    """The body of a request, read from a worker thread as it arrives, so that a file is never held whole
    and the server goes on answering others while it comes.

    :param request: The request.
    :param loop: The event loop the request is being received on.
    """

    def __init__(self, request: Request, loop: asyncio.AbstractEventLoop) -> None:
        self._chunks = request.stream().__aiter__()
        self._loop = loop
        self._pending = b""
        self._done = False

    async def _next(self) -> bytes | None:
        try:
            return await self._chunks.__anext__()
        except StopAsyncIteration:
            return None

    def read(self, size: int = -1) -> bytes:
        """Up to size bytes, or all that is left if size is negative."""
        while (size < 0 or len(self._pending) < size) and not self._done:
            try:
                chunk = asyncio.run_coroutine_threadsafe(self._next(), self._loop).result(UPLOAD_CHUNK_TIMEOUT)
            except TimeoutError as error:
                raise ApiError(408, "timeout", "The file stopped arriving") from error
            if chunk is None:
                self._done = True
            else:
                self._pending += chunk
        if size < 0:
            size = len(self._pending)
        data, self._pending = self._pending[:size], self._pending[size:]

        return data


def validation_error(error: RequestValidationError) -> ApiError:
    fields = [
        {"field": ".".join(str(part) for part in item["loc"] if part != "body"), "message": item["msg"]}
        for item in error.errors()
    ]
    message = "; ".join(f"{item['field']}: {item['message']}" if item["field"] else item["message"] for item in fields)

    return ApiError(422, "invalid", message, fields=fields)


def last_event(header: str | None) -> int:
    """The number of the last event a page saw, as the browser sends it when it opens a stream again."""
    try:
        return max(int(header or 0), 0)
    except ValueError:
        return 0


def static_file(root: Path, path: str) -> Path | None:
    """The page or file of the interface that a path is: itself, or a page by its name, if there is one."""
    if "\x00" in path or ".." in Path(path).parts:
        return None
    resolved = root.resolve()
    candidates = (path, f"{path}.html", f"{path}/index.html") if path else ("index.html",)
    for candidate in candidates:
        found = (resolved / candidate).resolve()
        if found.is_relative_to(resolved) and found.is_file():
            return found

    return None


def create_app(
    workspace: Path | str = DEFAULT_WORKSPACE,
    token: str | None = None,
    allowed_hosts: Collection[str] = LOCAL_HOSTS,
    static_dir: Path | str | None = None,
    check_keys: bool = True,
    client: Any = None,
    chat_models: Any = None,
) -> FastAPI:
    """The server's application.

    Anyone who can reach it can spend on the keys of whoever started it and, where code runs, run code on
    their machine, so it is for one person: see virtual_lab.server.security for what it turns away.

    :param workspace: Where conversations and settings are kept.
    :param token: What the API wants from every request but the health check, or None to want none, which is
        for tests.
    :param allowed_hosts: The names the server may be reached by.
    :param static_dir: The built pages of the interface, to serve, with any address that is not a file or the
        API's showing its first page, or None to serve none.
    :param check_keys: Whether to say a key is not set before something is paid for, as Conversations does.
    :param client: An OpenAI client for every conversation, as Chat takes one.
    :param chat_models: Chat models for every conversation, as Chat takes them.
    """
    store = Workspace(Path(workspace))
    conversations = Conversations(store, check_keys=check_keys, client=client, chat_models=chat_models)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        conversations.loop = asyncio.get_running_loop()
        try:
            yield
        finally:
            await anyio.to_thread.run_sync(conversations.close)

    app = FastAPI(
        title="Virtual Lab", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.conversations = conversations
    app.state.workspace = store

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(error.body(), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return await api_error(request, validation_error(error))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        messages = {404: "There is no such address", 405: "That address does not take this kind of request"}
        message = messages.get(error.status_code, str(error.detail))
        body = ApiError(error.status_code, codes.get(error.status_code, "error"), message).body()

        return JSONResponse(body, status_code=error.status_code, headers=error.headers)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> JSONResponse:
        # What went wrong is in the server's log, where Starlette puts it, and not in what a page is told
        body = ApiError(500, "internal", "Something went wrong in the server. Its log says what.").body()

        return JSONResponse(body, status_code=500)

    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @api.get("/settings")
    def get_settings() -> dict[str, Any]:
        return {**public_settings(store.load_settings()), "workspace": str(store.root)}

    @api.patch("/settings")
    def change_settings(body: SettingsBody) -> dict[str, Any]:
        try:
            changed = update_settings(store.load_settings(), body.model_dump(exclude_unset=True))
        except (ValueError, TypeError) as error:
            raise ApiError(422, "invalid", str(error)) from error
        store.save_settings(changed)

        return {**public_settings(changed), "workspace": str(store.root)}

    @api.get("/models")
    def get_models() -> dict[str, Any]:
        return {"default": store.load_settings().model, "models": describe_models()}

    def key_statuses() -> dict[str, Any]:
        return {
            "settable": list(SETTABLE_KEYS),
            "providers": [
                {
                    "provider": status.provider,
                    "variables": list(status.variables),
                    "missing": list(status.missing),
                    "ready": status.ready,
                }
                for status in provider_statuses()
            ],
        }

    @api.get("/keys")
    def get_keys() -> dict[str, Any]:
        return key_statuses()

    @api.put("/keys/{name}")
    def set_key(name: str, body: KeyBody) -> dict[str, Any]:
        if name not in SETTABLE_KEYS:
            raise ApiError(422, "invalid", f"A key is one of {', '.join(SETTABLE_KEYS)}")
        value = body.value.strip()
        if not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ApiError(422, "invalid", "A key is on one line, with nothing else in it")
        # It is kept for the rest of this process, and no further: it is never written down or told to a page
        os.environ[name] = value

        return key_statuses()

    @api.delete("/keys/{name}")
    def remove_key(name: str) -> dict[str, Any]:
        if name not in SETTABLE_KEYS:
            raise ApiError(422, "invalid", f"A key is one of {', '.join(SETTABLE_KEYS)}")
        os.environ.pop(name, None)

        return key_statuses()

    @api.get("/scientists")
    def get_scientists() -> dict[str, Any]:
        return {"scientists": library(store)}

    @api.put("/scientists")
    def save_scientist(body: ScientistBody) -> dict[str, Any]:
        try:
            store.save_scientist(body.model_dump())
        except ValueError as error:
            raise ApiError(422, "invalid", str(error)) from error

        return {"scientists": library(store)}

    @api.delete("/scientists/{title:path}")
    def remove_scientist(title: str) -> dict[str, Any]:
        if not any(item["title"].casefold() == title.casefold() for item in store.load_scientists()):
            raise not_found("scientist of yours with that title")
        store.remove_scientist(title)

        return {"scientists": library(store)}

    @api.get("/chats")
    def list_chats() -> dict[str, Any]:
        return {"chats": conversations.summaries()}

    @api.post("/chats", status_code=201)
    def create_chat(body: ChatBody) -> dict[str, Any]:
        return conversations.create(body.model_dump(exclude_unset=True), body.title).info()

    @api.get("/chats/{chat_id}")
    def get_chat(chat_id: str) -> dict[str, Any]:
        return conversations.get(chat_id).info()

    @api.patch("/chats/{chat_id}")
    def rename_chat(chat_id: str, body: RenameBody) -> dict[str, Any]:
        title = " ".join(body.title.split())[:MAX_CHAT_TITLE_CHARS]
        if not title:
            raise ApiError(422, "invalid", "A conversation needs a title with something in it")
        handle = conversations.get(chat_id)
        handle.chat.rename(title)

        return handle.info()

    @api.delete("/chats/{chat_id}")
    def delete_chat(chat_id: str) -> Response:
        conversations.delete(chat_id)

        return Response(status_code=204)

    @api.post("/chats/{chat_id}/messages", status_code=202)
    def send_message(chat_id: str, body: MessageBody) -> dict[str, Any]:
        handle = conversations.ready(chat_id)
        try:
            handle.chat.start(body.text, body.attachments)
        except ChatBusyError as error:
            raise ApiError(409, "busy", str(error)) from error
        except ChatClosedError as error:
            raise ApiError(409, "closed", str(error)) from error
        except ValueError as error:
            raise ApiError(422, "invalid", str(error)) from error

        return {"turn": handle.chat.turns, "state": handle.chat.state}

    @api.post("/chats/{chat_id}/note")
    def add_note(chat_id: str, body: NoteBody) -> dict[str, bool]:
        return {"accepted": conversations.get(chat_id).chat.add_note(body.text)}

    @api.post("/chats/{chat_id}/pause")
    def pause(chat_id: str) -> dict[str, bool]:
        return {"accepted": conversations.get(chat_id).chat.pause()}

    @api.post("/chats/{chat_id}/resume")
    def resume(chat_id: str) -> dict[str, bool]:
        return {"accepted": conversations.get(chat_id).chat.resume()}

    @api.post("/chats/{chat_id}/stop")
    def stop(chat_id: str) -> dict[str, bool]:
        return {"accepted": conversations.get(chat_id).chat.stop()}

    @api.get("/chats/{chat_id}/events")
    async def follow_chat(
        chat_id: str,
        after: int = Query(0, ge=0),
        last_event_id: str | None = Header(None),
    ) -> StreamingResponse:
        handle = await run_in_threadpool(conversations.get, chat_id)

        return StreamingResponse(
            follow(handle, max(after, last_event(last_event_id))),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @api.post("/chats/{chat_id}/uploads", status_code=201)
    async def upload(chat_id: str, request: Request, name: str = Query(min_length=1, max_length=1000)) -> Any:
        handle = await run_in_threadpool(conversations.get, chat_id)
        limit = handle.chat.max_upload_bytes
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise ApiError(413, "too_large", f"The file is larger than the {limit} bytes that can be attached")
        reader = RequestBodyReader(request, asyncio.get_running_loop())
        try:
            attachment = await run_in_threadpool(handle.chat.save_upload, name, cast(BinaryIO, reader))
        except UploadTooLargeError as error:
            raise ApiError(413, "too_large", str(error)) from error
        except ClientDisconnect:
            return Response(status_code=499)

        return attachment.to_dict()

    @api.get("/chats/{chat_id}/files")
    def chat_files(chat_id: str) -> dict[str, Any]:
        handle = conversations.get(chat_id)
        found, truncated = list_files(handle.files_root, shown=handle.files_shown)

        return {"files": found, "truncated": truncated}

    @api.get("/chats/{chat_id}/files/{path:path}")
    def chat_file(chat_id: str, path: str, download: bool = False) -> FileResponse:
        handle = conversations.get(chat_id)

        return file_response(resolve_inside(handle.files_root, path, handle.files_shown), download=download)

    app.include_router(api)

    if static_dir is not None:
        pages = Path(static_dir)

        @app.get("/{path:path}", include_in_schema=False)
        def page(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404)
            found = static_file(pages, path)
            if found is None and not Path(path).suffix:
                found = static_file(pages, "")
            if found is None:
                raise not_found("page at this address")
            cache = "public, max-age=31536000, immutable" if path.startswith(IMMUTABLE_PREFIX) else "no-cache"

            return FileResponse(found, headers={"Cache-Control": cache})

    app.add_middleware(Guard, token=token, allowed_hosts=allowed_hosts)

    return app
