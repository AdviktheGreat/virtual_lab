"""Typed meeting outputs.

A meeting produces prose. Anything downstream that needs to act on its conclusions, rather than
show them to a person, needs those conclusions as data. These helpers ask the agent who closed
the meeting to restate its conclusions against a schema, so the result can be validated and
handed to the next step without a human transcribing it.
"""

import json
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ParsedChatCompletion
from pydantic import BaseModel

from virtual_lab.constants import OUTPUT_DIR_NAME

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when a model does not return a usable instance of the requested schema."""


def request_structured_output(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    schema: type[SchemaT],
    temperature: float,
) -> tuple[SchemaT, ParsedChatCompletion[SchemaT]]:
    """Asks a model to answer as an instance of a schema.

    :param client: The OpenAI client to use.
    :param model: The model to ask.
    :param messages: The messages to send, including the agent's system prompt.
    :param schema: The pydantic model the answer must conform to.
    :param temperature: The sampling temperature.
    :raises StructuredOutputError: If the model refuses, or returns something unparseable.
    :return: The validated instance, and the full response so its usage can be recorded.
    """
    response = client.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=schema,
    )
    message = response.choices[0].message

    # A refusal is a deliberate decision by the model, not a transport error, so retrying it
    # would just spend money to get the same answer
    if message.refusal:
        raise StructuredOutputError(
            f"The model refused to produce {schema.__name__}: {message.refusal}"
        )

    if message.parsed is None:
        raise StructuredOutputError(
            f"The model did not return a parseable {schema.__name__}. "
            f"Raw content: {message.content!r}"
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
