"""Tests for reaching models from any provider through LangChain."""

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from conftest import FakeClient, fake_llm
from virtual_lab.agent import Agent
from virtual_lab.completions import MODELS_WITHOUT_TEMPERATURE, rejects_temperature
from virtual_lab.constants import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_RETRIES, MAX_TOOL_ITERATIONS
from virtual_lab.llm import (
    FINISH_REASONS,
    GEMINI_BASE_URL,
    GROQ_BASE_URL,
    ask,
    configure,
    detect_source,
    finish_reason_of,
    get_llm,
    meeting_source,
    reads_speaker_names,
    reply_from,
    resolve_chat_models,
    to_langchain_messages,
    usage_of,
)
from virtual_lab.run_meeting import describe_chat_model, hold_meeting
from virtual_lab.structured import StructuredOutputError, request_structured_output
from virtual_lab.tools import Tool
from virtual_lab.utils import compute_token_cost

CLAUDE = "claude-sonnet-4-5"


class ScriptedChatModel(BaseChatModel):
    """A chat model from no provider in particular, replaying scripted answers.

    It stands in for Anthropic, Gemini, a local server, or anything else that is not OpenAI's
    own API: it reads no names on messages, and it supports tools the generic LangChain way.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Any, since pydantic would otherwise try to read a queued exception as a message
    replies: list[Any] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)
    settings: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        self.settings.append({"temperature": self.temperature, "max_tokens": self.max_tokens, **kwargs})

        reply = self.replies.pop(0) if self.replies else answer("A response.")
        if isinstance(reply, BaseException):
            raise reply

        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(tools=tools, **kwargs)


def answer(content: Any = "A response.", reason: str = "end_turn", **kwargs: Any) -> AIMessage:
    """An answer as a provider other than OpenAI reports one."""
    return AIMessage(
        content=content,
        response_metadata={"stop_reason": reason},
        usage_metadata={"input_tokens": 300, "output_tokens": 40, "total_tokens": 340},
        **kwargs,
    )


def provider_error(status: int, message: str) -> Exception:
    """An error shaped as Anthropic's SDK and most others shape a refused request."""
    error = Exception(message)
    error.status_code = status  # type: ignore[attr-defined]
    error.message = message  # type: ignore[attr-defined]
    return error


class Decision(BaseModel):
    choice: str
    confidence: float


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(name, f"test-{name.lower()}")
    monkeypatch.setenv("OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.delenv("LLM_SOURCE", raising=False)


class TestDetectSource:
    @pytest.mark.parametrize(
        ("model", "source"),
        [
            ("claude-sonnet-4-5", "Anthropic"),
            ("gpt-oss-20b", "Ollama"),
            ("gpt-5.2", "OpenAI"),
            ("azure-my-deployment", "AzureOpenAI"),
            ("gemini-2.5-pro", "Gemini"),
            ("llama-3.3-70b-groq", "Groq"),
            ("qwen2.5:14b", "Ollama"),
            ("meta-llama/Llama-3-8B", "Ollama"),
            ("anthropic.claude-3-sonnet", "Bedrock"),
            ("us.amazon.nova-pro-v1:0", "Bedrock"),
            ("o3-mini", "OpenAI"),
            ("o4-mini", "OpenAI"),
            ("ft:gpt-4o-2024-08-06:org::abc", "OpenAI"),
        ],
    )
    def test_names_follow_biomni_rules(self, model: str, source: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        assert detect_source(model) == source

    def test_a_server_address_makes_an_unknown_name_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        assert detect_source("biomni-r0", base_url="http://localhost:30000/v1") == "Custom"

    def test_the_environment_overrides_the_rules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_SOURCE", "Groq")
        assert detect_source("gpt-5.2") == "Groq"

    def test_an_unknown_environment_value_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_SOURCE", "Nonsense")
        assert detect_source("gpt-5.2") == "OpenAI"

    def test_an_unknown_name_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        with pytest.raises(ValueError, match="Unable to determine the source"):
            detect_source("mystery-model")


class TestGetLlm:
    def test_openai_models_send_no_fixed_temperature(self, keys: None) -> None:
        llm = get_llm("gpt-5.2")

        assert isinstance(llm, ChatOpenAI)
        assert llm.temperature is None
        assert llm.max_retries == DEFAULT_MAX_RETRIES

    def test_anthropic_models_get_an_output_limit(self, keys: None) -> None:
        llm = get_llm(CLAUDE)

        assert isinstance(llm, ChatAnthropic)
        assert llm.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS
        assert llm.temperature is None

    def test_an_explicit_output_limit_is_used(self, keys: None) -> None:
        assert get_llm(CLAUDE, max_tokens=123).max_tokens == 123

    def test_gemini_goes_to_googles_openai_compatible_endpoint(self, keys: None) -> None:
        llm = get_llm("gemini-2.5-pro")

        assert isinstance(llm, ChatOpenAI)
        assert llm.openai_api_base == GEMINI_BASE_URL
        assert llm.openai_api_key.get_secret_value() == "test-gemini_api_key"

    def test_groq_goes_to_groqs_endpoint(self, keys: None) -> None:
        llm = get_llm("llama-3.3-70b-groq")

        assert llm.openai_api_base == GROQ_BASE_URL
        assert llm.openai_api_key.get_secret_value() == "test-groq_api_key"

    def test_a_custom_server_needs_an_address(self, keys: None) -> None:
        with pytest.raises(ValueError, match="base_url"):
            get_llm("biomni-r0", source="Custom")

    def test_a_custom_server_is_reached_at_its_address(self, keys: None) -> None:
        llm = get_llm("biomni-r0", base_url="http://localhost:30000/v1")

        assert llm.openai_api_base == "http://localhost:30000/v1"
        assert llm.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS

    def test_azure_names_a_deployment(self, keys: None) -> None:
        llm = get_llm("azure-my-deployment")

        assert isinstance(llm, AzureChatOpenAI)
        assert llm.deployment_name == "my-deployment"

    def test_ollama_names_its_output_limit_differently(self, keys: None) -> None:
        llm = get_llm("qwen2.5:14b", max_tokens=321)

        assert type(llm).__name__ == "ChatOllama"
        assert llm.num_predict == 321

    def test_bedrock_uses_the_configured_region(self, keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")

        llm = get_llm("anthropic.claude-3-sonnet")

        assert type(llm).__name__ == "ChatBedrock"
        assert llm.model_id == "anthropic.claude-3-sonnet"
        assert llm.region_name == "eu-west-1"

    def test_an_invalid_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Invalid source"):
            get_llm("gpt-5.2", source="Nowhere")  # type: ignore[arg-type]

    def test_a_missing_package_says_what_to_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langchain_anthropic":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)

        with pytest.raises(ImportError, match="pip install langchain-anthropic"):
            get_llm(CLAUDE)


class TestMeetingSource:
    def test_a_name_the_rules_cannot_place_goes_to_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        assert meeting_source("chatgpt-4o-latest") == "OpenAI"

    def test_open_model_names_go_to_a_configured_openai_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

        assert meeting_source("meta-llama/Llama-3-70b") == "OpenAI"
        assert meeting_source(CLAUDE) == "Anthropic"

    def test_open_model_names_go_to_ollama_otherwise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_SOURCE", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        assert meeting_source("meta-llama/Llama-3-70b") == "Ollama"

    def test_the_environment_still_decides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("LLM_SOURCE", "Ollama")

        assert meeting_source("meta-llama/Llama-3-70b") == "Ollama"


class TestResolveChatModels:
    def test_one_model_answers_for_everyone(self) -> None:
        shared = ScriptedChatModel()

        assert resolve_chat_models(["a", "b"], chat_models=shared) == {"a": shared, "b": shared}

    def test_a_mapping_is_used_where_it_has_the_name(self, fake_client: FakeClient) -> None:
        mine = ScriptedChatModel()

        resolved = resolve_chat_models(["mine", "gpt-5.2"], chat_models={"mine": mine})

        assert resolved["mine"] is mine
        assert isinstance(resolved["gpt-5.2"], ChatOpenAI)
        assert fake_client.init_kwargs["model"] == "gpt-5.2"

    def test_a_function_is_asked_once_per_name(self) -> None:
        asked = []

        def build(model: str) -> BaseChatModel:
            asked.append(model)
            return ScriptedChatModel()

        resolve_chat_models(["a", "b", "a"], chat_models=build)

        assert asked == ["a", "b"]

    def test_a_client_answers_for_names_nothing_else_covers(self) -> None:
        client = FakeClient()

        llm = resolve_chat_models(["any-name-at-all"], client=client)["any-name-at-all"]

        assert llm.root_client is client


class TestMessages:
    MEETING: list[dict[str, Any]] = [
        {"role": "system", "content": "You are the critic."},
        {"role": "user", "content": "Discuss."},
        {"role": "assistant", "name": "Immunologist", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "pubmed_search", "arguments": '{"query": "VHH"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Three papers."},
        {"role": "assistant", "name": "Immunologist", "content": "Use VHH."},
        {"role": "user", "content": "Critic, your turn."},
        {"role": "assistant", "name": "Scientific_Critic", "content": "Earlier I said no."},
    ]

    def test_names_are_kept_for_a_model_that_reads_them(self) -> None:
        converted = to_langchain_messages(self.MEETING, reader="Scientific_Critic", speaker_names=True)

        assert [type(message) for message in converted] == [
            SystemMessage, HumanMessage, AIMessage, ToolMessage, AIMessage, HumanMessage, AIMessage
        ]
        assert converted[4].name == "Immunologist"
        assert converted[2].tool_calls[0]["args"] == {"query": "VHH"}

    def test_other_agents_speak_as_named_users_to_a_model_that_cannot_read_names(self) -> None:
        converted = to_langchain_messages(self.MEETING, reader="Scientific_Critic", speaker_names=False)

        assert [type(message) for message in converted] == [
            SystemMessage, HumanMessage, HumanMessage, HumanMessage, HumanMessage, HumanMessage, AIMessage
        ]
        assert converted[2].content == '[Immunologist called pubmed_search with {"query": "VHH"}]'
        assert converted[3].content == "[What pubmed_search returned to Immunologist]\nThree papers."
        assert converted[4].content == "Immunologist: Use VHH."
        # The reader's own turn stays its own, and carries no name the provider would reject
        assert converted[6].content == "Earlier I said no."
        assert converted[6].name is None

    def test_the_readers_own_tool_calls_stay_tool_calls(self) -> None:
        converted = to_langchain_messages(self.MEETING, reader="Immunologist", speaker_names=False)

        assert isinstance(converted[2], AIMessage)
        assert isinstance(converted[3], ToolMessage)
        assert converted[3].tool_call_id == "call_1"
        assert converted[6].content == "Scientific_Critic: Earlier I said no."

    def test_unreadable_arguments_are_sent_back_as_written(self) -> None:
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
        ]}]

        converted = to_langchain_messages(messages)[0]

        assert converted.tool_calls == []
        assert converted.invalid_tool_calls[0]["args"] == "{not json"

    def test_unreadable_arguments_reach_openai_as_written(self) -> None:
        client = FakeClient()
        messages = [
            {"role": "user", "content": "Go."},
            {"role": "assistant", "name": "A", "content": None, "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
            ]},
            {"role": "tool", "tool_call_id": "c", "content": "Invalid JSON"},
        ]

        ask(fake_llm(client), messages, temperature=None)

        sent = client.completions.calls[0]["messages"][1]["tool_calls"][0]["function"]
        assert sent == {"name": "f", "arguments": "{not json"}

    def test_a_reused_id_is_matched_to_the_latest_call(self) -> None:
        call = {"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        messages = [
            {"role": "assistant", "name": "A", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_0", "content": "resA"},
            {"role": "assistant", "name": "B", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_0", "content": "resB"},
        ]

        converted = to_langchain_messages(messages, reader="B", speaker_names=False)

        assert converted[1].content == "[What search returned to A]\nresA"
        assert isinstance(converted[3], ToolMessage)
        assert converted[3].content == "resB"

    def test_own_tool_calls_become_text_when_no_tools_can_be_offered(self) -> None:
        converted = to_langchain_messages(self.MEETING, reader="Immunologist", speaker_names=False, tool_blocks=False)

        assert not any(isinstance(message, ToolMessage) for message in converted)
        assert not any(getattr(message, "tool_calls", None) for message in converted)
        assert converted[2].content == '[You called pubmed_search with {"query": "VHH"}]'
        assert isinstance(converted[2], AIMessage)
        assert converted[3].content == "[What pubmed_search returned to you]\nThree papers."

    def test_an_unknown_role_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown message role"):
            to_langchain_messages([{"role": "developer", "content": "x"}])


class TestReplies:
    @pytest.mark.parametrize(
        ("metadata", "reason"),
        [
            ({"finish_reason": "stop"}, "stop"),
            ({"stop_reason": "end_turn"}, "stop"),
            ({"stop_reason": "max_tokens"}, "length"),
            ({"stop_reason": "tool_use"}, "tool_calls"),
            ({"stop_reason": "refusal"}, "content_filter"),
            ({"done_reason": "length"}, "length"),
            ({"finish_reason": "MAX_TOKENS"}, "length"),
            ({"finish_reason": "SAFETY"}, "content_filter"),
            ({"incomplete_details": {"reason": "max_output_tokens"}}, "length"),
            ({"finish_reason": "something_new"}, "something_new"),
        ],
    )
    def test_every_providers_reason_for_stopping_is_read(self, metadata: dict[str, Any], reason: str) -> None:
        assert finish_reason_of(AIMessage(content="", response_metadata=metadata)) == reason

    def test_an_unstated_reason_is_inferred_from_tool_calls(self) -> None:
        with_calls = AIMessage(content="", tool_calls=[{"id": "c", "name": "f", "args": {}}])

        assert finish_reason_of(with_calls) == "tool_calls"
        assert finish_reason_of(AIMessage(content="done")) is None

    def test_every_reason_maps_to_one_a_meeting_checks_for(self) -> None:
        assert set(FINISH_REASONS.values()) == {"stop", "length", "tool_calls", "content_filter"}

    def test_usage_is_read_with_cached_and_reasoning_tokens(self) -> None:
        message = AIMessage(content="", usage_metadata={
            "input_tokens": 500, "output_tokens": 70, "total_tokens": 570,
            "input_token_details": {"cache_read": 200}, "output_token_details": {"reasoning": 30},
        })

        usage = usage_of(message)

        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (500, 70, 570)
        assert usage.prompt_tokens_details.cached_tokens == 200
        assert usage.completion_tokens_details.reasoning_tokens == 30

    def test_unreported_usage_stays_unknown(self) -> None:
        assert usage_of(AIMessage(content="")) is None

    def test_text_is_read_from_content_blocks(self) -> None:
        message = AIMessage(content=[{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Answer."}])

        assert reply_from(message).content == "Answer."

    def test_a_call_with_malformed_arguments_is_passed_on(self) -> None:
        message = AIMessage(content="", invalid_tool_calls=[{"id": "bad", "name": "f", "args": "{oops", "error": None}])

        call = reply_from(message).tool_calls[0]

        assert (call.id, call.function.name, call.function.arguments) == ("bad", "f", "{oops")

    def test_calls_without_ids_get_ids_unique_across_replies(self) -> None:
        message = AIMessage(content="", tool_calls=[{"id": None, "name": "f", "args": {}}])

        first, second = reply_from(message).tool_calls[0].id, reply_from(message).tool_calls[0].id

        assert first != second
        assert first.startswith("call_")

    def test_tool_calls_are_shaped_as_openais(self) -> None:
        message = AIMessage(content="", tool_calls=[{"id": "c1", "name": "f", "args": {"x": 1}}])

        assert reply_from(message).tool_calls[0].model_dump() == {
            "id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}
        }


class TestSettings:
    def test_openais_own_api_reads_names(self) -> None:
        assert reads_speaker_names(fake_llm(FakeClient()))

    def test_other_servers_do_not(self, keys: None) -> None:
        assert not reads_speaker_names(get_llm("gemini-2.5-pro"))
        assert not reads_speaker_names(get_llm(CLAUDE))
        assert not reads_speaker_names(ScriptedChatModel())

    def test_no_temperature_is_sent_when_none_is_given(self) -> None:
        client = FakeClient()

        ask(configure(fake_llm(client), 0.7, None), [{"role": "user", "content": "Hi."}], temperature=None)

        assert "temperature" not in client.completions.calls[0]

    def test_the_output_limit_is_set_under_each_providers_name(self, keys: None) -> None:
        assert configure(get_llm(CLAUDE), None, 50).max_tokens == 50
        assert configure(get_llm("qwen2.5:14b"), None, 60).num_predict == 60

    def test_configuring_leaves_the_original_alone(self, keys: None) -> None:
        original = get_llm(CLAUDE)

        configure(original, 0.3, 50)

        assert (original.temperature, original.max_tokens) == (None, DEFAULT_MAX_OUTPUT_TOKENS)


def critic_meeting(tmp_path, llm: BaseChatModel, **kwargs: Any):  # type: ignore[no-untyped-def]
    member = Agent(title="Immunologist", expertise="VHH", goal="design", role="advise", model=CLAUDE)
    return hold_meeting(
        meeting_type="individual",
        agenda="Design a nanobody.",
        save_dir=tmp_path,
        team_member=member,
        num_rounds=1,
        chat_models=llm,
        **kwargs,
    )


class TestMeetingsWithOtherProviders:
    def test_a_meeting_runs_on_a_model_that_is_not_openais(self, tmp_path) -> None:
        llm = ScriptedChatModel(replies=[answer("Use VHH."), answer("Too vague."), answer("Revised.")])

        result = critic_meeting(tmp_path, llm)

        assert result.summary == "Revised."
        assert result.usage.input_tokens == 900
        assert result.cost == pytest.approx(3 * compute_token_cost(CLAUDE, 300, 40))
        assert result.record.chat_models[CLAUDE]["class"].endswith("ScriptedChatModel")

    def test_the_critic_reads_the_other_agent_by_name(self, tmp_path) -> None:
        llm = ScriptedChatModel(replies=[answer("Use VHH."), answer("Too vague."), answer("Revised.")])

        critic_meeting(tmp_path, llm)

        critic_view = llm.received[1]
        assert any(message.content == "Immunologist: Use VHH." for message in critic_view)
        assert not any(isinstance(message, AIMessage) for message in critic_view)
        # And the member reads its own earlier turn as its own
        member_view = llm.received[2]
        assert [message.content for message in member_view if isinstance(message, AIMessage)] == ["Use VHH."]
        assert any(message.content == "Scientific_Critic: Too vague." for message in member_view)

    def test_a_cut_off_answer_is_reported(self, tmp_path) -> None:
        llm = ScriptedChatModel(replies=[answer("", reason="max_tokens")])

        with pytest.raises(Exception, match="ran out of tokens"):
            critic_meeting(tmp_path, llm)

    def test_a_refused_temperature_is_dropped_for_any_provider(self, tmp_path) -> None:
        error = provider_error(400, "temperature is not supported for this model")
        llm = ScriptedChatModel(replies=[error, answer("A."), answer("B."), answer("C.")])

        result = critic_meeting(tmp_path, llm, temperature=0.4)

        assert [setting["temperature"] for setting in llm.settings] == [0.4, None, None, None]
        assert result.record.models_at_default_temperature == [CLAUDE]
        assert CLAUDE in MODELS_WITHOUT_TEMPERATURE

    def test_the_completion_limit_reaches_the_provider(self, tmp_path) -> None:
        llm = ScriptedChatModel()

        critic_meeting(tmp_path, llm, max_completion_tokens=77)

        assert {setting["max_tokens"] for setting in llm.settings} == {77}

    def test_the_forced_last_attempt_carries_no_tool_calls(self, tmp_path) -> None:
        tool = Tool(
            name="search",
            description="Search.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            function=lambda query: f"results for {query}",
        )
        searching = [
            answer("", reason="tool_use", tool_calls=[{"id": f"s{index}", "name": "search", "args": {"query": "x"}}])
            for index in range(MAX_TOOL_ITERATIONS)
        ]
        llm = ScriptedChatModel(replies=[*searching, answer("Done.")])

        critic_meeting(tmp_path, llm, tools=(tool,))

        # Anthropic refuses a request that holds tool calls but offers no tools
        last = llm.received[MAX_TOOL_ITERATIONS]
        assert "tools" not in llm.settings[MAX_TOOL_ITERATIONS]
        assert not any(isinstance(message, ToolMessage) or getattr(message, "tool_calls", None) for message in last)
        assert any("[What search returned to you]" in str(message.content) for message in last)
        # Earlier attempts, which offer the tool, keep the calls as calls
        assert any(isinstance(message, ToolMessage) for message in llm.received[MAX_TOOL_ITERATIONS - 1])

    def test_structured_output_works_by_tool_calling(self, tmp_path) -> None:
        decided = answer(tool_calls=[{"id": "d", "name": "Decision", "args": {"choice": "VHH", "confidence": 0.8}}], content="")
        llm = ScriptedChatModel(replies=[answer("A."), answer("B."), answer("C."), decided])

        result = critic_meeting(tmp_path, llm, output_schema=Decision)

        assert result.output == Decision(choice="VHH", confidence=0.8)
        assert result.usage.num_calls == 4
        # The closing agent restates the meeting knowing which turns were the critic's
        assert any(message.content == "Scientific_Critic: B." for message in llm.received[3])


class TestStructuredOutputFromOtherProviders:
    def ask_for(self, reply: AIMessage) -> tuple[Any, Any]:
        return request_structured_output(
            llm=ScriptedChatModel(replies=[reply]),
            model=CLAUDE,
            messages=[{"role": "user", "content": "Decide."}],
            schema=Decision,
            temperature=0.2,
        )

    def test_arguments_that_do_not_validate_are_reported_with_their_usage(self) -> None:
        wrong = answer(tool_calls=[{"id": "d", "name": "Decision", "args": {"choice": "VHH"}}], content="")

        with pytest.raises(StructuredOutputError, match="parseable Decision") as caught:
            self.ask_for(wrong)

        assert caught.value.usage.prompt_tokens == 300

    def test_no_call_at_all_is_reported(self) -> None:
        with pytest.raises(StructuredOutputError, match="Raw content: 'I would rather chat.'"):
            self.ask_for(answer("I would rather chat."))

    def test_running_out_of_tokens_is_reported(self) -> None:
        with pytest.raises(StructuredOutputError, match="ran out of tokens"):
            self.ask_for(answer("", reason="max_tokens"))

    def test_a_refusal_is_reported(self) -> None:
        with pytest.raises(StructuredOutputError, match="stopped by the content filter"):
            self.ask_for(answer("", reason="refusal"))


class TestRejectsTemperature:
    error = staticmethod(lambda status, message: provider_error(status, message))

    def test_a_400_calling_the_temperature_unsupported_is_a_refusal(self) -> None:
        assert rejects_temperature(self.error(400, "`temperature` is deprecated for this model"))
        assert rejects_temperature(self.error(400, "temperature and top_p cannot both be specified"))

    def test_an_out_of_range_value_is_the_callers_mistake(self) -> None:
        assert not rejects_temperature(self.error(400, "temperature: must be less than or equal to 1"))

    def test_other_failures_are_not_refusals(self) -> None:
        assert not rejects_temperature(self.error(500, "temperature not supported"))
        assert not rejects_temperature(self.error(400, "max_tokens not supported"))


class TestRecord:
    def test_the_server_is_recorded_without_its_password(self) -> None:
        llm = ChatOpenAI(model="x", api_key="k", base_url="https://user:secret@example.com:8443/v1")

        described = describe_chat_model(llm)

        assert described["base_url"] == "https://example.com:8443/v1"
        assert described["class"] == "langchain_openai.chat_models.base.ChatOpenAI"

    def test_openais_default_server_is_recorded_as_none(self) -> None:
        assert describe_chat_model(fake_llm(FakeClient()))["base_url"] is None


class TestPrices:
    def test_anthropic_snapshots_are_priced_as_their_model(self) -> None:
        assert compute_token_cost("claude-sonnet-4-5-20250929", 10**6, 10**6) == pytest.approx(18)

    def test_gemini_is_priced(self) -> None:
        assert compute_token_cost("gemini-2.5-flash", 10**6, 10**6) == pytest.approx(2.8)

