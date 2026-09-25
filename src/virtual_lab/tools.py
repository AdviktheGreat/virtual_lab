"""Tools that agents can call during a meeting."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from virtual_lab.utils import run_pubmed_search


@dataclass(frozen=True)
class Tool:
    """A function that an agent can call during a meeting.

    :param name: The name the model uses to call the tool.
    :param description: What the tool does, used by the model to decide when to call it.
    :param parameters: A JSON Schema object describing the tool's arguments.
    :param function: The callable that runs the tool. Its return value is sent to the model,
        so it should be a string or convertible to one.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    @property
    def definition(self) -> ChatCompletionToolParam:
        """Returns the tool in OpenAI API form."""
        return ChatCompletionToolParam(  # type: ignore[misc]
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }
        )


PUBMED_TOOL = Tool(
    name="pubmed_search",
    description="Get abstracts or the full text of biomedical and life sciences articles from PubMed Central.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to use to search PubMed Central for scientific articles.",
            },
            "num_articles": {
                "type": "integer",
                "description": "The number of articles to return from the search query.",
            },
            "abstract_only": {
                "type": "boolean",
                "description": "Whether to return only the abstract of the articles.",
            },
        },
        "required": ["query", "num_articles"],
    },
    function=run_pubmed_search,
)


def run_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    tools: tuple[Tool, ...],
) -> tuple[list[str], list[ChatCompletionMessageParam]]:
    """Runs the tool calls requested by a model.

    A tool that fails reports the error back to the model as its output instead of ending the
    meeting, so the agent can correct the call, try a different query, or continue without it.
    A network hiccup on a literature search should not discard an entire meeting.

    :param tool_calls: The tool calls from the chat completion response.
    :param tools: The tools available in this meeting.
    :return: The tool outputs as strings, and the corresponding tool response messages.
    """
    name_to_tool = {tool.name: tool for tool in tools}

    tool_outputs: list[str] = []
    tool_messages: list[ChatCompletionMessageParam] = []

    for tool_call in tool_calls:
        name = tool_call.function.name
        tool = name_to_tool.get(name)

        if tool is None:
            available = ", ".join(sorted(name_to_tool)) or "none"
            output = f'Error: unknown tool "{name}". Available tools: {available}.'
            print(output)
        else:
            try:
                arguments = json.loads(tool_call.function.arguments)
                output = str(tool.function(**arguments))
            except Exception as e:
                output = f'Error running tool "{name}": {type(e).__name__}: {e}'
                print(output)

        tool_outputs.append(output)
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            }
        )

    return tool_outputs, tool_messages
