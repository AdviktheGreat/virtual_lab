"""Typed meeting outputs.

A meeting produces prose. Anything downstream that needs to act on its conclusions, rather than
show them to a person, needs those conclusions as data. These helpers ask the agent who closed
the meeting to restate its conclusions against a schema, so the result can be validated and
handed to the next step without a human transcribing it.
"""

import json
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from virtual_lab.completions import send_request
from virtual_lab.constants import OUTPUT_DIR_NAME
from virtual_lab.llm import ModelReply, configure, reads_speaker_names, reply_from, to_langchain_messages

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when a model does not return a usable instance of the requested schema.

    The request was still made and still paid for, so what it used is kept, when it is known,
    for the caller to count. A schema's own validators can fail inside the OpenAI SDK before the
    response is returned, so for those there is no response, and what they used is unknown.
    """

    def __init__(self, message: str, usage: CompletionUsage | None = None, response: Any = None) -> None:
        super().__init__(message)
        self.usage = usage
        self.response = response


def request_structured_output(
    llm: BaseChatModel,
    model: str,
    messages: list[ChatCompletionMessageParam],
    schema: type[SchemaT],
    temperature: float | None,
    max_completion_tokens: int | None = None,
    reader: str | None = None,
) -> tuple[SchemaT, ModelReply]:
    """Asks a model to answer as an instance of a schema.

    :param llm: The chat model to ask.
    :param model: The model's name, which is remembered if it refuses a temperature.
    :param messages: The messages to send, including the agent's system prompt.
    :param schema: The pydantic model the answer must conform to.
    :param temperature: The sampling temperature, or None for the model's default.
    :param max_completion_tokens: The most tokens the answer may use, or None for no limit.
    :param reader: The name of the agent being asked, for models that cannot read names.
    :raises StructuredOutputError: If the model refuses, runs out of tokens, or returns something
        that does not validate against the schema.
    :return: The validated instance, and the reply so its usage can be recorded.
    """
    converted = to_langchain_messages(messages, reader, reads_speaker_names(llm))

    def send(sent_temperature: float | None) -> dict[str, Any]:
        configured = configure(llm, sent_temperature, max_completion_tokens)
        # include_raw keeps the response when parsing fails, so that what it used is counted
        return configured.with_structured_output(schema, include_raw=True).invoke(converted)  # type: ignore[return-value]

    # The OpenAI SDK validates the answer itself, before returning it, so for OpenAI models a
    # schema's own validators and a cut-off answer both surface here as exceptions
    try:
        result = send_request(send, model=model, temperature=temperature)
    except LengthFinishReasonError as error:
        raise StructuredOutputError(
            f"The model ran out of tokens before finishing {schema.__name__}. Allow it more "
            f"with max_completion_tokens.",
            usage=error.completion.usage,
            response=error.completion,
        ) from error
    except ContentFilterFinishReasonError as error:
        raise StructuredOutputError(
            f"The model's answer for {schema.__name__} was stopped by the content filter",
            usage=error.completion.usage,
            response=error.completion,
        ) from error
    except ValidationError as error:
        raise StructuredOutputError(
            f"The model's answer did not validate as {schema.__name__}: {error}"
        ) from error

    reply = reply_from(result["raw"])
    parsed = result.get("parsed")
    parsing_error = result.get("parsing_error")
    refusal = reply.message.additional_kwargs.get("refusal")

    def fail(message: str) -> StructuredOutputError:
        return StructuredOutputError(message, usage=reply.usage, response=reply.message)

    # A refusal is a deliberate decision by the model, not a transport error, so retrying it
    # would just spend money to get the same answer
    if refusal:
        raise fail(f"The model refused to produce {schema.__name__}: {refusal}")

    if parsed is None and reply.finish_reason == "length":
        raise fail(
            f"The model ran out of tokens before finishing {schema.__name__}. Allow it more "
            f"with max_completion_tokens."
        )

    if parsed is None and reply.finish_reason == "content_filter":
        raise fail(f"The model's answer for {schema.__name__} was stopped by the content filter")

    if parsed is None:
        detail = f" ({parsing_error})" if parsing_error is not None else ""
        raise fail(
            f"The model did not return a parseable {schema.__name__}{detail}. "
            f"Raw content: {reply.content!r}"
        )

    # Some providers hand back the fields rather than the instance
    if not isinstance(parsed, schema):
        try:
            parsed = schema.model_validate(parsed)
        except ValidationError as error:
            raise fail(f"The model's answer did not validate as {schema.__name__}: {error}") from error

    return parsed, reply


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
