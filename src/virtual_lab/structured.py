"""Typed meeting outputs.

A meeting produces prose. Anything downstream that needs to act on its conclusions, rather than
show them to a person, needs those conclusions as data. These helpers ask the agent who closed
the meeting to restate its conclusions against a schema, so the result can be validated and
handed to the next step without a human transcribing it.
"""

import json
from pathlib import Path
from typing import TypeVar

from openai import NOT_GIVEN, ContentFilterFinishReasonError, LengthFinishReasonError, OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ParsedChatCompletion
from pydantic import BaseModel, ValidationError

from virtual_lab.completions import send_request
from virtual_lab.constants import OUTPUT_DIR_NAME

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when a model does not return a usable instance of the requested schema.

    The request was still made and still paid for, so the response is kept, when there is one,
    for the caller to count what it used. A schema's own validators fail inside the SDK before
    the response is returned, so for those there is none, and what they used is unknown.
    """

    def __init__(self, message: str, response: ChatCompletion | None = None) -> None:
        super().__init__(message)
        self.response = response

    @property
    def usage(self) -> CompletionUsage | None:
        """The tokens the failed request used, or None if they are unknown."""
        return self.response.usage if self.response is not None else None


def request_structured_output(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    schema: type[SchemaT],
    temperature: float | None,
    max_completion_tokens: int | None = None,
) -> tuple[SchemaT, ParsedChatCompletion[SchemaT]]:
    """Asks a model to answer as an instance of a schema.

    :param client: The OpenAI client to use.
    :param model: The model to ask.
    :param messages: The messages to send, including the agent's system prompt.
    :param schema: The pydantic model the answer must conform to.
    :param temperature: The sampling temperature, or None for the model's default.
    :param max_completion_tokens: The most tokens the answer may use, or None for no limit.
    :raises StructuredOutputError: If the model refuses, runs out of tokens, or returns something
        that does not validate against the schema.
    :return: The validated instance, and the full response so its usage can be recorded.
    """
    # The SDK validates the answer itself, before returning it, so a schema's own validators
    # and a cut-off answer both surface here as exceptions rather than as a missing parse
    try:
        response = send_request(
            client.chat.completions.parse,
            model=model,
            temperature=temperature,
            messages=messages,
            response_format=schema,
            max_completion_tokens=(
                max_completion_tokens if max_completion_tokens is not None else NOT_GIVEN
            ),
        )
    except LengthFinishReasonError as error:
        raise StructuredOutputError(
            f"The model ran out of tokens before finishing {schema.__name__}. Allow it more "
            f"with max_completion_tokens.",
            response=error.completion,
        ) from error
    except ContentFilterFinishReasonError as error:
        raise StructuredOutputError(
            f"The model's answer for {schema.__name__} was stopped by the content filter",
            response=error.completion,
        ) from error
    except ValidationError as error:
        raise StructuredOutputError(
            f"The model's answer did not validate as {schema.__name__}: {error}"
        ) from error

    message = response.choices[0].message

    # A refusal is a deliberate decision by the model, not a transport error, so retrying it
    # would just spend money to get the same answer
    if message.refusal:
        raise StructuredOutputError(
            f"The model refused to produce {schema.__name__}: {message.refusal}",
            response=response,
        )

    if message.parsed is None:
        raise StructuredOutputError(
            f"The model did not return a parseable {schema.__name__}. "
            f"Raw content: {message.content!r}",
            response=response,
        )

    return message.parsed, response


def save_output(save_dir: Path, save_name: str, output: BaseModel) -> Path:
    """Writes a meeting's structured output into the outputs subdirectory.

    :param save_dir: The directory the transcript was saved in.
    :param save_name: The name the transcript was saved under.
    :param output: The structured output to write.
    :return: The path written.
    """
    output_dir = save_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{save_name}.json"

    with open(path, "w") as f:
        json.dump(output.model_dump(mode="json"), f, indent=4)

    return path
