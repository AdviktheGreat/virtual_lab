"""Shared fixtures for building fake OpenAI and HTTP responses without hitting either."""

import json
import re
from importlib import import_module
from typing import Any

import pytest
from requests.structures import CaseInsensitiveDict
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat import ParsedChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage, ParsedChoice
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import BaseModel

from virtual_lab import paper_batch
from virtual_lab.agent import Agent
from virtual_lab.literature import Article, ArticleText
from virtual_lab.records import RecordNotFoundError

TEST_MODEL = "gpt-4o-2024-08-06"


def make_usage(
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> CompletionUsage:
    """Builds a usage object of the shape the API returns."""
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def text_response(
    content: str = "A response.",
    model: str = TEST_MODEL,
    usage: CompletionUsage | None = None,
) -> ChatCompletion:
    """Builds a completion in which the model answers with text."""
    return ChatCompletion(
        id="test",
        model=model,
        object="chat.completion",
        created=0,
        usage=usage if usage is not None else make_usage(),
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
    )


def tool_call_response(
    name: str,
    arguments: dict[str, Any] | None = None,
    call_id: str = "call_1",
    model: str = TEST_MODEL,
) -> ChatCompletion:
    """Builds a completion in which the model requests a tool call."""
    return ChatCompletion(
        id="test",
        model=model,
        object="chat.completion",
        created=0,
        usage=make_usage(),
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=call_id,
                            type="function",
                            function=Function(name=name, arguments=json.dumps(arguments or {})),
                        )
                    ],
                ),
            )
        ],
    )


def parsed_response(
    parsed: BaseModel | None,
    model: str = TEST_MODEL,
    refusal: str | None = None,
    content: str | None = None,
    usage: CompletionUsage | None = None,
) -> ParsedChatCompletion:
    """Builds the completion the parse helper returns for a structured request."""
    return ParsedChatCompletion(
        id="test",
        model=model,
        object="chat.completion",
        created=0,
        usage=usage if usage is not None else make_usage(),
        choices=[
            ParsedChoice(
                finish_reason="stop",
                index=0,
                message=ParsedChatCompletionMessage(
                    role="assistant",
                    content=content if content is not None else (parsed.model_dump_json() if parsed else None),
                    refusal=refusal,
                    parsed=parsed,
                ),
            )
        ],
    )


def stream_chunks(response: ChatCompletion, include_usage: bool) -> list[ChatCompletionChunk]:
    """Splits a completion into the chunks the API streams it as: the text a word at a time, each
    tool call in two halves, then why it stopped, and its usage last if it was asked for."""
    choice = response.choices[0]
    message = choice.message
    deltas: list[ChoiceDelta] = [ChoiceDelta(role="assistant", content="")]
    deltas += [ChoiceDelta(content=word) for word in re.findall(r"\S+\s*|\s+", message.content or "")]
    for index, call in enumerate(message.tool_calls or []):
        arguments = call.function.arguments
        half = len(arguments) // 2
        deltas.append(
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=index,
                        id=call.id,
                        type="function",
                        function=ChoiceDeltaToolCallFunction(name=call.function.name, arguments=arguments[:half]),
                    )
                ]
            )
        )
        rest = ChoiceDeltaToolCallFunction(arguments=arguments[half:])
        deltas.append(ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(index=index, function=rest)]))

    def chunk(choices: list[ChunkChoice], usage: CompletionUsage | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=response.id,
            model=response.model,
            object="chat.completion.chunk",
            created=0,
            choices=choices,
            usage=usage,
            system_fingerprint=response.system_fingerprint,
        )

    chunks = [chunk([ChunkChoice(index=0, delta=delta)]) for delta in deltas]
    chunks.append(chunk([ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason=choice.finish_reason)]))
    if include_usage:
        chunks.append(chunk([], usage=response.usage))

    return chunks


class FakeStream:
    """Stands in for the stream the SDK returns for a request made with stream=True."""

    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self.chunks = chunks

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __iter__(self) -> Any:
        return iter(self.chunks)


class FakeCompletions:
    """Stands in for client.chat.completions, replaying queued responses, streamed when asked."""

    def __init__(self) -> None:
        self.responses: list[ChatCompletion | BaseException] = []
        self.parsed_responses: list[ParsedChatCompletion | BaseException] = []
        self.calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        if not self.responses:
            response: ChatCompletion | BaseException = text_response()
        else:
            response = self.responses.pop(0)

        if isinstance(response, BaseException):
            raise response

        if kwargs.get("stream"):
            include_usage = bool((kwargs.get("stream_options") or {}).get("include_usage"))
            return FakeStream(stream_chunks(response, include_usage))

        return response

    def parse(self, **kwargs: Any) -> ParsedChatCompletion:
        self.parse_calls.append(kwargs)
        self.calls.append(kwargs)

        if not self.parsed_responses:
            # A generic instance cannot be built without inventing values for required fields,
            # and model_construct leaves them unset, so an unqueued response surfaces later as
            # a confusing AttributeError on the first field a caller touches. Failing here names
            # the schema instead.
            schema = kwargs["response_format"]

            raise AssertionError(
                f"No parsed response was queued for {schema.__name__}. Append one to "
                f"fake_client.completions.parsed_responses."
            )

        response = self.parsed_responses.pop(0)

        if isinstance(response, BaseException):
            raise response

        return response


    @property
    def with_raw_response(self) -> "RawResponses":
        # The shape LangChain calls, since it reads the response's headers as well as its body
        return RawResponses(self)


class RawResponse:
    """Stands in for the SDK's wrapper around a response and its HTTP headers."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.headers: dict[str, str] = {}
        self.http_response = None

    def parse(self) -> Any:
        return self.response


class RawResponses:
    """Stands in for client.chat.completions.with_raw_response."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions

    def create(self, **kwargs: Any) -> RawResponse:
        return RawResponse(self.completions.create(**kwargs))

    def parse(self, **kwargs: Any) -> RawResponse:
        return RawResponse(self.completions.parse(**kwargs))


class FakeClient:
    """Stands in for openai.OpenAI."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    @property
    def completions(self) -> FakeCompletions:
        return self.chat.completions


@pytest.fixture(autouse=True)
def forget_models_without_temperature() -> Any:
    """Keeps one test's temperature refusals from changing the requests another test sees."""
    from virtual_lab.completions import MODELS_WITHOUT_TEMPERATURE

    saved = set(MODELS_WITHOUT_TEMPERATURE)
    MODELS_WITHOUT_TEMPERATURE.clear()
    yield
    MODELS_WITHOUT_TEMPERATURE.clear()
    MODELS_WITHOUT_TEMPERATURE.update(saved)


def fake_llm(client: FakeClient, model: str = TEST_MODEL) -> Any:
    """A LangChain OpenAI chat model that sends its requests to a fake client."""
    from virtual_lab.llm import openai_chat_model

    return openai_chat_model(model, client)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Makes every chat model a meeting builds for itself send its requests to a fake.

    The same client answers for every model so that a test can queue responses before calling
    run_meeting and inspect the requests afterwards. What the model was built with is kept in
    init_kwargs.
    """
    client = FakeClient()

    def build_llm(model: str, **kwargs: Any) -> Any:
        client.init_kwargs = {"model": model, **kwargs}
        return fake_llm(client, model)

    monkeypatch.setattr(import_module("virtual_lab.llm"), "get_llm", build_llm)

    return client


class FakeResponse:
    """Stands in for a streaming requests.Response."""

    def __init__(
        self,
        status_code: int = 200,
        body: bytes = b"{}",
        headers: dict[str, str] | None = None,
        url: str = "https://rest.uniprot.org/test",
        json_body: Any = None,
    ) -> None:
        self.status_code = status_code
        self.body = json.dumps(json_body).encode() if json_body is not None else body
        # Case-insensitive, as real requests is. A plain dict would let production code read a
        # header under the wrong case and still pass here.
        self.headers = CaseInsensitiveDict(headers or {})
        self.url = url

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def is_redirect(self) -> bool:
        return self.status_code in {301, 302, 303, 307, 308} and "Location" in self.headers

    @property
    def is_permanent_redirect(self) -> bool:
        return self.status_code in {301, 308} and "Location" in self.headers

    def iter_content(self, chunk_size: int = 8192):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeTransport:
    """Records outbound requests and replays queued responses."""

    def __init__(self) -> None:
        self.responses: list[FakeResponse] = []
        self.post_responses: list[FakeResponse] = []
        self.requests: list[dict[str, Any]] = []
        self.post_requests: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})

        # An unexpected request is a test result, not something to answer. Replying 200 with an
        # empty body here would let a test that makes one more request than it queued pass, and
        # pass with the wrong body, which is how a test comes to assert nothing
        if not self.responses:
            raise AssertionError(
                f"Unexpected request to {url}: the queue is empty after "
                f"{len(self.requests) - 1} request(s)"
            )

        return self.responses.pop(0)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.post_requests.append({"url": url, **kwargs})

        if not self.post_responses:
            raise AssertionError(
                f"Unexpected POST to {url}: the queue is empty after "
                f"{len(self.post_requests) - 1} request(s)"
            )

        return self.post_responses.pop(0)

    def queue(self, *bodies: Any) -> None:
        """Queues JSON bodies to be returned in order."""
        self.responses.extend(FakeResponse(json_body=body) for body in bodies)

    @property
    def urls(self) -> list[str]:
        return [request["url"] for request in self.requests]

    @property
    def params(self) -> list[Any]:
        return [request.get("params") for request in self.requests]


@pytest.fixture
def web_transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    """Replaces the HTTP layer, and removes the cache and the deliberate delays with it.

    Anything reaching the real network from a test would make the suite slow, flaky, and
    dependent on someone else's uptime, so this is how every test that touches a database runs.
    """
    web = import_module("virtual_lab.web")
    transport = FakeTransport()

    # Patched at http_get rather than at requests.get, because a host configured with its own
    # session would otherwise slip past the fake and reach the real service from a test
    monkeypatch.setattr(web, "http_get", transport.get)
    monkeypatch.setattr(web.requests, "post", transport.post)
    monkeypatch.setattr(web.RATE_LIMITER, "min_interval", 0.0)
    monkeypatch.setattr(web.time, "sleep", lambda seconds: None)
    web.RESPONSE_CACHE.clear()

    yield transport

    web.RESPONSE_CACHE.clear()


@pytest.fixture
def team_lead() -> Agent:
    return Agent(
        title="Principal Investigator",
        expertise="running a lab",
        goal="do good science",
        role="lead the team",
        model=TEST_MODEL,
    )


@pytest.fixture
def team_member() -> Agent:
    return Agent(
        title="Immunologist",
        expertise="antibody engineering",
        goal="design nanobodies",
        role="advise on immunogenicity",
        model=TEST_MODEL,
    )


@pytest.fixture
def second_team_member() -> Agent:
    return Agent(
        title="Computational Biologist",
        expertise="protein structure prediction",
        goal="model complexes",
        role="run simulations",
        model=TEST_MODEL,
    )


def biorxiv_record(doi: str, published: str = "NA", **fields: Any) -> dict[str, Any]:
    return {
        "title": f"Title of {doi}",
        "authors": "A. Author",
        "doi": doi,
        "date": "2024-01-02",
        "version": "1",
        "license": "cc_by",
        "category": "neuroscience",
        "abstract": "An abstract.",
        "published": published,
        **fields,
    }


class FakeListing:
    """Stands in for bioRxiv's listing, which answers a page of records at a cursor."""

    def __init__(self, records: list[dict[str, Any]], page_size: int = 2) -> None:
        self.records = records
        self.page_size = page_size
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        self.calls.append((url, params or {}))
        cursor = int(url.split("/")[-2])
        page = self.records[cursor : cursor + self.page_size]

        return {"messages": [{"status": "ok", "total": str(len(self.records))}], "collection": page}


@pytest.fixture
def listing(monkeypatch: pytest.MonkeyPatch) -> FakeListing:
    fake = FakeListing([biorxiv_record(f"10.1101/{number}", published=f"10.9/{number}") for number in range(7)])
    monkeypatch.setattr(paper_batch, "request_json", fake)

    return fake


@pytest.fixture
def europe_pmc(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stands in for Europe PMC: the articles it holds by DOI, and the text of those by PMCID."""
    held: dict[str, Any] = {"articles": {}, "texts": {}, "asked": []}

    def get_article(identifier: str) -> Article:
        held["asked"].append(("article", identifier))
        if identifier not in held["articles"]:
            raise RecordNotFoundError(f"Europe PMC has no article for {identifier}")

        return held["articles"][identifier]

    def get_article_text(pmcid: str) -> ArticleText:
        held["asked"].append(("text", pmcid))
        if pmcid not in held["texts"]:
            raise RecordNotFoundError(f"Europe PMC holds no open full text for {pmcid}")

        return held["texts"][pmcid]

    monkeypatch.setattr(paper_batch, "get_article", get_article)
    monkeypatch.setattr(paper_batch, "get_article_text", get_article_text)

    return held


def open_article(pmcid: str = "PMC1", title: str = "The article") -> Article:
    return Article(article_id=pmcid, source="PMC", pmcid=pmcid, title=title, open_access=True, in_europe_pmc=True)
