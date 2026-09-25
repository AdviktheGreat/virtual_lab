"""Shared fixtures for building fake OpenAI responses without hitting the API."""

import json
from importlib import import_module
from typing import Any

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat import ParsedChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage, ParsedChoice
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import BaseModel

from virtual_lab.agent import Agent

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


class FakeCompletions:
    """Stands in for client.chat.completions, replaying queued responses."""

    def __init__(self) -> None:
        self.responses: list[ChatCompletion | Exception] = []
        self.parsed_responses: list[ParsedChatCompletion | Exception] = []
        self.calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> ChatCompletion:
        self.calls.append(kwargs)

        if not self.responses:
            return text_response()

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response

    def parse(self, **kwargs: Any) -> ParsedChatCompletion:
        self.parse_calls.append(kwargs)
        self.calls.append(kwargs)

        if not self.parsed_responses:
            # Default to a valid instance of whatever schema was asked for, built from its
            # example values, so tests that do not care about the payload need not queue one
            schema = kwargs["response_format"]

            return parsed_response(parsed=schema.model_construct())

        response = self.parsed_responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


class FakeClient:
    """Stands in for openai.OpenAI."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    @property
    def completions(self) -> FakeCompletions:
        return self.chat.completions


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Replaces the OpenAI client used by run_meeting with a fake.

    The same instance is returned for every construction so a test can queue responses before
    calling run_meeting and inspect the requests afterwards.
    """
    client = FakeClient()

    def build_client(**kwargs: Any) -> FakeClient:
        client.init_kwargs = kwargs
        return client

    # The module must be patched through the module object, since virtual_lab.run_meeting
    # resolves to the re-exported function rather than the module it lives in.
    monkeypatch.setattr(import_module("virtual_lab.run_meeting"), "OpenAI", build_client)

    return client


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
