"""Chat models from any provider, and one way for a meeting to talk to all of them.

get_llm chooses and builds a LangChain chat model the way Biomni does (snap-stanford/Biomni,
biomni/llm.py, Apache License 2.0): the same eight sources, the same rules for telling them
apart by model name, and the same environment variables for each. Three things differ, each on
purpose:

- No temperature is fixed when the model is built. A meeting sets it on every request, so that
  a model which refuses one can be asked again without it (see virtual_lab.completions), where
  Biomni instead drops it for every model whose name starts with gpt-5.
- No shell is started to read a key out of ~/.bash_profile. Keys come from the environment.
- max_retries and timeout are passed to every model that takes them, since a meeting is many
  requests long and one dropped connection should not end it.

The rest of this module is the boundary between a meeting, which keeps its messages in the
OpenAI chat format it has always saved, and the providers, which each want something slightly
different. What comes back is normalised to one shape, ModelReply, whatever answered.
"""

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails

from virtual_lab.constants import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_RETRIES

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"]
ALLOWED_SOURCES: frozenset[str] = frozenset(SourceType.__args__)  # type: ignore[attr-defined]

# Names Biomni sends to a local Ollama server when nothing else claims them
OLLAMA_MODEL_FAMILIES = ("llama", "mistral", "qwen", "gemma", "phi", "dolphin", "orca", "vicuna", "deepseek")
BEDROCK_PREFIXES = ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")

AZURE_API_VERSION = "2024-12-01-preview"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def detect_source(model: str, base_url: str | None = None) -> SourceType:
    """Works out which provider serves a model, by Biomni's rules.

    The LLM_SOURCE environment variable, if it names a source, overrides the rules.

    :param model: The model name.
    :param base_url: The address of a self-hosted server, if there is one.
    :raises ValueError: If the name matches no rule.
    :return: The source.
    """
    env_source = os.getenv("LLM_SOURCE")
    if env_source in ALLOWED_SOURCES:
        return env_source  # type: ignore[return-value]

    if model.startswith("claude-"):
        return "Anthropic"
    if model.startswith("gpt-oss"):
        return "Ollama"
    if model.startswith("gpt-"):
        return "OpenAI"
    if model.startswith("azure-"):
        return "AzureOpenAI"
    if model.startswith("gemini-"):
        return "Gemini"
    if "groq" in model.lower():
        return "Groq"
    if base_url is not None:
        return "Custom"
    if "/" in model or any(name in model.lower() for name in OLLAMA_MODEL_FAMILIES):
        return "Ollama"
    if model.startswith(BEDROCK_PREFIXES):
        return "Bedrock"

    # Biomni stops here. The o-series reasoning models and fine-tuned models are OpenAI's too,
    # and this repository's price tables have always included them.
    if model.startswith(("o1", "o3", "o4", "ft:")):
        return "OpenAI"

    raise ValueError(f'Unable to determine the source of "{model}". Please specify source.')


def supported_arguments(model_class: type, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keeps only the arguments a chat model class declares, so one call can build any of them."""
    fields = getattr(model_class, "model_fields", {})
    aliases = {field.alias for field in fields.values() if getattr(field, "alias", None)}

    return {key: value for key, value in arguments.items() if key in fields or key in aliases}


def get_llm(
    model: str,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float | None = None,
) -> BaseChatModel:
    """Builds a chat model for any of the sources Biomni supports.

    :param model: The model name, e.g. "gpt-5.2", "claude-sonnet-4-5", or "gemini-2.5-pro".
    :param temperature: A fixed sampling temperature, or None to leave it to each request.
    :param stop_sequences: Sequences that end a response.
    :param source: The provider, or None to work it out from the model name.
    :param base_url: The address of an OpenAI-compatible server, for a Custom source.
    :param api_key: The key for a Custom source.
    :param max_tokens: The most tokens a response may use, or None for the provider default,
        except that Anthropic and Custom sources, which need a figure, get Biomni's 8,192.
    :param max_retries: How many times to retry a failed request, with backoff.
    :param timeout: Seconds to wait for a response, or None for the provider default.
    :raises ValueError: If the source cannot be determined or is not one of ALLOWED_SOURCES.
    :raises ImportError: If the package for the source is not installed.
    :return: The chat model.
    """
    if source is None:
        source = detect_source(model, base_url)

    if source not in ALLOWED_SOURCES:
        raise ValueError(f"Invalid source: {source}. Valid options are {', '.join(sorted(ALLOWED_SOURCES))}")

    common: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
        "max_retries": max_retries,
        "timeout": timeout,
    }
    if max_tokens is not None:
        common["max_tokens"] = max_tokens

    def build(model_class: type, **arguments: Any) -> BaseChatModel:
        # Anything left as None is left to the class's own default, which for a temperature is
        # to send none
        merged = {key: value for key, value in {**common, **arguments}.items() if value is not None}
        return model_class(**supported_arguments(model_class, merged))

    if source in {"OpenAI", "Gemini", "Groq", "Custom"}:
        chat_openai = import_chat_model("langchain_openai", "ChatOpenAI", "langchain-openai")

        if source == "OpenAI":
            return build(chat_openai)
        if source == "Gemini":
            return build(chat_openai, api_key=os.getenv("GEMINI_API_KEY"), base_url=GEMINI_BASE_URL)
        if source == "Groq":
            return build(chat_openai, api_key=os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL)

        # Custom serving, such as SGLang for Biomni-R0, must expose an OpenAI-compatible API
        if base_url is None:
            raise ValueError("base_url must be provided for a Custom source")
        return build(
            chat_openai,
            base_url=base_url,
            api_key=api_key or "EMPTY",
            max_tokens=max_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        )

    if source == "AzureOpenAI":
        azure = import_chat_model("langchain_openai", "AzureChatOpenAI", "langchain-openai")
        return build(
            azure,
            model=None,
            azure_deployment=model.removeprefix("azure-"),
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            api_version=AZURE_API_VERSION,
        )

    if source == "Anthropic":
        anthropic = import_chat_model("langchain_anthropic", "ChatAnthropic", "langchain-anthropic")
        return build(anthropic, max_tokens=max_tokens or DEFAULT_MAX_OUTPUT_TOKENS)

    if source == "Ollama":
        ollama = import_chat_model("langchain_ollama", "ChatOllama", "langchain-ollama")
        return build(ollama, num_predict=max_tokens, stop=stop_sequences)

    bedrock = import_chat_model("langchain_aws", "ChatBedrock", "langchain-aws")
    # ChatBedrock takes the model under its alias, "model", which common already sets
    return build(bedrock, region_name=os.getenv("AWS_REGION", "us-east-1"))


def import_chat_model(module: str, name: str, package: str) -> type:
    """Imports a provider's chat model class, saying what to install if it is missing."""
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except ImportError as error:
        raise ImportError(f"{package} is required for this model. Install it with: pip install {package}") from error


def openai_chat_model(model: str, client: Any, max_retries: int = DEFAULT_MAX_RETRIES) -> BaseChatModel:
    """Builds an OpenAI chat model around an existing OpenAI client.

    :param model: The model name.
    :param client: An openai.OpenAI instance, or anything shaped like one.
    :param max_retries: Recorded on the model; the client's own retry setting still applies.
    :return: The chat model.
    """
    chat_openai = import_chat_model("langchain_openai", "ChatOpenAI", "langchain-openai")

    return chat_openai(
        model=model,
        client=client.chat.completions,
        root_client=client,
        api_key=getattr(client, "api_key", None) or "unused",
        temperature=None,
        max_retries=max_retries,
    )


ModelSource = Mapping[str, BaseChatModel] | Callable[[str], BaseChatModel] | BaseChatModel


def resolve_chat_models(
    models: list[str],
    chat_models: ModelSource | None = None,
    client: Any = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, BaseChatModel]:
    """Finds a chat model for every model a meeting uses.

    :param models: The model names the agents use.
    :param chat_models: A model to use for everyone, a mapping from name to model, or a
        function from name to model. A name the mapping lacks is built with get_llm.
    :param client: An OpenAI client to build OpenAI models around, as before chat_models existed.
    :param max_retries: Retries for models built here.
    :return: A chat model per model name.
    """
    resolved: dict[str, BaseChatModel] = {}

    for model in dict.fromkeys(models):
        if isinstance(chat_models, BaseChatModel):
            resolved[model] = chat_models
        elif isinstance(chat_models, Mapping) and model in chat_models:
            resolved[model] = chat_models[model]
        elif callable(chat_models) and not isinstance(chat_models, Mapping):
            resolved[model] = chat_models(model)
        elif client is not None:
            resolved[model] = openai_chat_model(model, client, max_retries=max_retries)
        else:
            resolved[model] = get_llm(model, source=meeting_source(model), max_retries=max_retries)

    return resolved


def meeting_source(model: str) -> SourceType:
    """The provider for a model a meeting builds itself, keeping what worked before providers.

    Meetings sent every model to OpenAI's client, which reads OPENAI_BASE_URL, before they
    could reach any other provider. Biomni's rules would refuse a name they do not recognise,
    such as chatgpt-4o-latest, and send names that look like open models, such as
    meta-llama/Llama-3-70b, to a local Ollama server. A name the rules cannot place still goes
    to OpenAI, and while OPENAI_BASE_URL points at a server of your own, so do names that look
    like open models. LLM_SOURCE overrides both.
    """
    try:
        source = detect_source(model)
    except ValueError:
        return "OpenAI"

    if source == "Ollama" and os.getenv("OPENAI_BASE_URL") and os.getenv("LLM_SOURCE") not in ALLOWED_SOURCES:
        return "OpenAI"

    return source


@dataclass(frozen=True)
class FunctionCall:
    """The function a tool call names, shaped as the OpenAI SDK shapes it."""

    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCall:
    """A tool call from any provider, shaped as the OpenAI SDK shapes one."""

    id: str
    function: FunctionCall
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        """The call in the OpenAI message format a meeting keeps."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


@dataclass(frozen=True)
class ModelReply:
    """What a model said, whichever provider it came from.

    :param content: The text of the answer, empty if there was none.
    :param tool_calls: The tools it asked to call.
    :param finish_reason: Why it stopped: "stop", "length", "tool_calls", "content_filter", or
        whatever else the provider said, or None if it did not say.
    :param usage: The tokens it used, or None if the provider did not report them.
    :param system_fingerprint: The backend configuration, when the provider reports one.
    :param message: The response as LangChain returned it.
    """

    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    usage: CompletionUsage | None
    system_fingerprint: str | None
    message: AIMessage


# How each provider says why it stopped, in the terms a meeting checks for
FINISH_REASONS = {
    "stop": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "STOP": "stop",
    "length": "length",
    "max_tokens": "length",
    "max_output_tokens": "length",
    "MAX_TOKENS": "length",
    "tool_calls": "tool_calls",
    "tool_use": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
    "refusal": "content_filter",
    "SAFETY": "content_filter",
}


def finish_reason_of(message: AIMessage) -> str | None:
    """Why a model stopped, in the terms OpenAI uses, whichever provider answered."""
    metadata = message.response_metadata or {}
    reason = (
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("done_reason")
        or (metadata.get("incomplete_details") or {}).get("reason")
    )

    if reason is None:
        return "tool_calls" if message.tool_calls else None

    return FINISH_REASONS.get(str(reason), str(reason))


def usage_of(message: AIMessage) -> CompletionUsage | None:
    """The tokens a response used, in the shape the usage counters read.

    LangChain reports every provider's usage the same way, with cached input tokens included in
    the input count. Anthropic bills writing to its cache at a quarter more than ordinary input,
    which this does not see, so a cost computed from it can be slightly low for a model that
    writes to its cache.

    :param message: The response.
    :return: The usage, or None if the provider reported none.
    """
    metadata = message.usage_metadata
    if not metadata:
        return None

    input_details = metadata.get("input_token_details") or {}
    output_details = metadata.get("output_token_details") or {}
    input_tokens = int(metadata.get("input_tokens") or 0)
    output_tokens = int(metadata.get("output_tokens") or 0)

    return CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=int(metadata.get("total_tokens") or input_tokens + output_tokens),
        prompt_tokens_details=PromptTokensDetails(cached_tokens=int(input_details.get("cache_read") or 0)),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=int(output_details.get("reasoning") or 0)),
    )


def reads_speaker_names(llm: BaseChatModel) -> bool:
    """Whether a model is told who wrote each earlier turn by the message's name field.

    OpenAI's own API reads the name on a message. Anthropic's has no such field, and
    OpenAI-compatible servers generally ignore it, so for those the speaker goes into the text.
    """
    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
    except ImportError:
        return False

    if isinstance(llm, AzureChatOpenAI):
        return True

    if isinstance(llm, ChatOpenAI):
        base_url = getattr(llm, "openai_api_base", None)
        return base_url is None or "api.openai.com" in base_url

    return False


def tool_call_from_dict(call: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """A tool call in a meeting's saved format, as LangChain wants it.

    :return: The call, and whether its arguments were valid. A call whose arguments were not a
        JSON object keeps them as the model wrote them, so that it is shown what it actually sent
        alongside the error that came back.
    """
    function = call.get("function") or {}
    raw = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        arguments = None

    if isinstance(arguments, dict):
        return {"id": call.get("id"), "name": function.get("name", ""), "args": arguments}, True

    return {"id": call.get("id"), "name": function.get("name", ""), "args": raw, "error": None}, False


def describe_call(call: dict[str, Any]) -> str:
    """A tool call's arguments as text, however they were written."""
    arguments = call["args"]

    return arguments if isinstance(arguments, str) else json.dumps(arguments)


def to_langchain_messages(
    messages: list[ChatCompletionMessageParam],
    reader: str | None = None,
    speaker_names: bool = True,
    tool_blocks: bool = True,
) -> list[BaseMessage]:
    """Turns a meeting's messages into what a LangChain model takes.

    A meeting is several agents talking, but a chat API has one assistant. OpenAI's API tells
    them apart by the name on each message. For a model that cannot, every turn written by
    someone other than the reader is given to it as a user message that says who wrote it,
    along with any tools that agent called and what came back. Read as its own words, the
    other agents' turns push the reader towards agreeing with them.

    :param messages: The messages, in OpenAI's format.
    :param reader: The name of the agent the messages are for.
    :param speaker_names: Whether the model reads the name field.
    :param tool_blocks: Whether the reader's own tool calls can be sent as tool calls. Anthropic
        refuses a request that holds tool calls but offers no tools, which is what the last,
        forced attempt of a turn is, so for that request they are written out as text too.
    :return: The messages as LangChain messages.
    """
    converted: list[BaseMessage] = []
    # Each is removed once its result is read, so that a provider reusing an id in a later
    # response cannot have that result attributed to the earlier call
    others_calls: dict[str, tuple[str, str]] = {}

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        name = message.get("name")

        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "tool":
            call_id = str(message.get("tool_call_id"))
            if call_id in others_calls:
                speaker, tool = others_calls.pop(call_id)
                recipient = "you" if speaker == "" else speaker
                converted.append(HumanMessage(content=f"[What {tool} returned to {recipient}]\n{content}"))
            else:
                converted.append(ToolMessage(content=content, tool_call_id=call_id))
        elif role == "assistant":
            calls = [tool_call_from_dict(call) for call in message.get("tool_calls") or []]
            is_other = not speaker_names and name is not None and reader is not None and name != reader

            if is_other or (calls and not tool_blocks):
                # The empty speaker marks the reader's own calls, written out for a request
                # that offers no tools
                speaker = str(name) if is_other else ""
                caller = str(name) if is_other else "You"
                lines = [f"{name}: {content}" if is_other else content] if content else []
                for call, _ in calls:
                    others_calls[str(call["id"])] = (speaker, call["name"])
                    lines.append(f"[{caller} called {call['name']} with {describe_call(call)}]")
                text = "\n".join(lines) or f"{name}: (no answer)"
                converted.append(HumanMessage(content=text) if is_other else AIMessage(content=text))
            else:
                converted.append(
                    AIMessage(
                        content=content,
                        name=name if speaker_names else None,
                        tool_calls=[call for call, valid in calls if valid],
                        invalid_tool_calls=[call for call, valid in calls if not valid],
                    )
                )
        else:
            raise ValueError(f"Unknown message role: {role}")

    return converted


def configure(llm: BaseChatModel, temperature: float | None, max_tokens: int | None) -> BaseChatModel:
    """A copy of a model that sends this temperature and output limit.

    Each provider names the output limit differently, and a temperature of None leaves it out of
    the request altogether, which is what a model that refuses one needs.
    """
    fields = getattr(type(llm), "model_fields", {})
    update: dict[str, Any] = {}

    if "temperature" in fields:
        update["temperature"] = temperature

    if max_tokens is not None:
        for field in ("max_tokens", "num_predict", "max_output_tokens"):
            if field in fields:
                update[field] = max_tokens
                break

    return llm.model_copy(update=update) if update else llm


def text_of(message: AIMessage) -> str:
    """The text of a response, which some providers return as a list of blocks."""
    if isinstance(message.content, str):
        return message.content

    return str(message.text)


def reply_from(message: AIMessage) -> ModelReply:
    """Normalises a LangChain response into a ModelReply."""
    # A provider that leaves out ids gets fresh ones, unique across the meeting, since a result
    # is matched to its call by id alone
    calls = [
        ToolCall(id=str(call.get("id") or f"call_{uuid4().hex}"), function=FunctionCall(call["name"], json.dumps(call.get("args") or {})))
        for call in message.tool_calls
    ]
    # A call whose arguments were not valid JSON is still passed on, so that the tool's error
    # reaches the model and it can correct the call
    calls += [
        ToolCall(
            id=str(call.get("id") or f"call_{uuid4().hex}"),
            function=FunctionCall(str(call.get("name") or ""), str(call.get("args") or "")),
        )
        for call in message.invalid_tool_calls
    ]

    return ModelReply(
        content=text_of(message),
        tool_calls=tuple(calls),
        finish_reason=finish_reason_of(message),
        usage=usage_of(message),
        system_fingerprint=(message.response_metadata or {}).get("system_fingerprint") or None,
        message=message,
    )


def streamed(runnable: Any, messages: list[BaseMessage], on_text: Callable[[str], None]) -> AIMessage:
    """Asks a model for its response as it is written, telling on_text of the text so far each
    time more arrives, and returns the whole response once it has."""
    gathered: AIMessageChunk | None = None
    told = ""

    for chunk in runnable.stream(messages):
        # A model that cannot stream, or is told not to, sends its whole response as one message
        if isinstance(chunk, AIMessage) and not isinstance(chunk, AIMessageChunk) and gathered is None:
            if text := text_of(chunk):
                on_text(text)
            return chunk
        if not isinstance(chunk, AIMessageChunk):
            raise TypeError(f"Expected an AIMessageChunk from the model, got {type(chunk).__name__}")
        gathered = chunk if gathered is None else gathered + chunk
        if (text := text_of(gathered)) != told:
            told = text
            on_text(text)

    if gathered is None:
        raise RuntimeError("The model sent nothing back")

    message = message_chunk_to_message(gathered)
    assert isinstance(message, AIMessage)

    return message


def ask(
    llm: BaseChatModel,
    messages: list[ChatCompletionMessageParam],
    temperature: float | None,
    tools: list[ChatCompletionToolParam] | None = None,
    max_tokens: int | None = None,
    reader: str | None = None,
    on_text: Callable[[str], None] | None = None,
) -> ModelReply:
    """Sends a meeting's messages to a model and normalises what comes back.

    :param llm: The chat model.
    :param messages: The messages, in OpenAI's format.
    :param temperature: The sampling temperature, or None for the model's default.
    :param tools: Tool definitions in OpenAI's format, or None to offer none.
    :param max_tokens: The most tokens the response may use, or None for the model's own limit.
    :param reader: The name of the agent being asked, for models that cannot read names.
    :param on_text: Called with the text of the reply so far each time more of it arrives, which
        has the reply streamed. A model that cannot stream sends it all at once.
    :return: The reply.
    """
    model = configure(llm, temperature, max_tokens)
    # LangChain asks OpenAI's own API for the usage of a streamed response, but no other server
    # that speaks its protocol, and a reply without its usage has an unknown cost. One told
    # not to send it is left as it was told.
    streams_usage = "stream_usage" in getattr(type(model), "model_fields", {})
    if on_text is not None and streams_usage and model.stream_usage is None:  # type: ignore[attr-defined]
        model = model.model_copy(update={"stream_usage": True})
    runnable = model.bind_tools(tools) if tools else model
    speaker_names = reads_speaker_names(llm)
    # OpenAI accepts earlier tool calls in a request that offers no tools; others may not
    converted = to_langchain_messages(messages, reader, speaker_names, tool_blocks=bool(tools) or speaker_names)
    message = runnable.invoke(converted) if on_text is None else streamed(runnable, converted, on_text)

    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AIMessage from the model, got {type(message).__name__}")

    return reply_from(message)
