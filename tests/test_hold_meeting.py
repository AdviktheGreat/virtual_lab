"""Tests for hold_meeting: what it returns, what it may spend, and how it fails."""

import json
import math
from importlib import import_module
from types import SimpleNamespace

import openai
import pytest
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError, field_validator

from virtual_lab.agent import Agent
from virtual_lab.constants import (
    MAX_RECORDED_ARGUMENT_CHARS,
    MAX_TOOL_ITERATIONS,
    METADATA_DIR_NAME,
    OUTPUT_DIR_NAME,
    PARTIAL_MEETING_DIR_NAME,
)
from virtual_lab.run_meeting import MeetingResult, TruncatedResponseError, hold_meeting, run_meeting
from virtual_lab.structured import StructuredOutputError
from virtual_lab.tools import Tool
from virtual_lab.repair import request_repair
from virtual_lab.utils import BudgetExceededError, CostUnknownError, MeetingUsage, compute_token_cost

from conftest import (
    TEST_MODEL,
    FakeClient,
    fake_llm,
    make_usage,
    parsed_response,
    text_response,
    tool_call_response,
)

UNPRICED_MODEL = "an-unreleased-model"

# What one default fake response costs, so that limits can be set in whole responses
ONE_RESPONSE = compute_token_cost(TEST_MODEL, 100, 20)


class Verdict(BaseModel):
    decision: str


class StrictVerdict(BaseModel):
    decision: str

    @field_validator("decision")
    @classmethod
    def must_be_known(cls, value: str) -> str:
        if value not in {"go", "stop"}:
            raise ValueError("decision must be go or stop")
        return value


def individual(team_member: Agent, save_dir, **kwargs) -> MeetingResult:
    return hold_meeting(
        meeting_type="individual",
        agenda="Design a nanobody.",
        save_dir=save_dir,
        team_member=team_member,
        **kwargs,
    )


def partial_record(save_dir) -> dict:
    return json.loads(
        (save_dir / PARTIAL_MEETING_DIR_NAME / METADATA_DIR_NAME / "discussion.json").read_text()
    )


def partial_transcript(save_dir) -> list[dict[str, str]]:
    return json.loads((save_dir / PARTIAL_MEETING_DIR_NAME / "discussion.json").read_text())


def bad_request(
    param: str | None, message: str = "Unsupported value", code: str = "unsupported_value"
) -> openai.BadRequestError:
    """Builds the error the API returns when it refuses a request parameter."""
    # Only what the error reads from the response; the HTTP library under the SDK has changed
    # name between releases, so it is not constructed here
    response = SimpleNamespace(status_code=400, request=None, headers={})
    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}

    return openai.BadRequestError(message, response=response, body=body)  # type: ignore[arg-type]


def truncated_response(content: str) -> openai.types.chat.ChatCompletion:
    response = text_response(content)
    response.choices[0].finish_reason = "length"

    return response


class TestResult:
    def test_everything_the_meeting_produced_is_returned(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [text_response("The answer.")]

        result = individual(team_member, tmp_path)

        assert result.summary == "The answer."
        assert result.output is None
        assert result.output_path is None
        assert result.usage.num_calls == 1
        assert result.cost == pytest.approx(ONE_RESPONSE)
        assert result.record.status == "completed"
        assert result.discussion[-1] == {"agent": team_member.title, "message": "The answer."}
        assert json.loads(result.transcript_path.read_text()) == list(result.discussion)
        assert json.loads(result.record_path.read_text())["status"] == "completed"

    def test_summary_is_kept_alongside_a_structured_output(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The structured output is the last turn of the transcript, so the summary cannot be
        # recovered from the transcript once there is one
        fake_client.completions.responses = [text_response("Go ahead.")]
        fake_client.completions.parsed_responses = [parsed_response(Verdict(decision="go"))]

        result = individual(team_member, tmp_path, output_schema=Verdict)

        assert result.summary == "Go ahead."
        assert result.output == Verdict(decision="go")
        assert result.output_path == tmp_path / OUTPUT_DIR_NAME / "discussion.json"
        assert json.loads(result.output_path.read_text()) == {"decision": "go"}

    def test_result_cannot_be_changed_by_whoever_receives_it(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        result = individual(team_member, tmp_path)

        with pytest.raises(AttributeError):
            result.summary = "Something else."  # type: ignore[misc]

    def test_cost_is_unknown_for_an_unpriced_model_rather_than_zero(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [text_response(model=UNPRICED_MODEL)]

        result = individual(team_member.with_model(UNPRICED_MODEL), tmp_path)

        assert result.cost is None


class TestRunMeetingKeepsItsInterface:
    def test_nothing_is_returned_by_default(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        assert run_meeting(
            meeting_type="individual", agenda="A.", save_dir=tmp_path, team_member=team_member
        ) is None

    def test_summary_is_returned_when_asked_for(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [text_response("The answer.")]

        summary = run_meeting(
            meeting_type="individual",
            agenda="A.",
            save_dir=tmp_path,
            team_member=team_member,
            return_summary=True,
        )

        assert summary == "The answer."

    def test_structured_output_is_returned_when_a_schema_is_given(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(Verdict(decision="go"))]

        output = run_meeting(
            meeting_type="individual",
            agenda="A.",
            save_dir=tmp_path,
            team_member=team_member,
            output_schema=Verdict,
        )

        assert output == Verdict(decision="go")

    def test_asking_for_both_a_summary_and_a_schema_is_refused(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="either output_schema or return_summary"):
            run_meeting(
                meeting_type="individual",
                agenda="A.",
                save_dir=tmp_path,
                team_member=team_member,
                output_schema=Verdict,
                return_summary=True,
            )

        assert fake_client.completions.calls == []


class TestArguments:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"num_rounds": -1}, "num_rounds"),
            ({"max_cost": -0.01}, "max_cost"),
            ({"max_cost": math.nan}, "max_cost"),
            ({"max_cost": math.inf}, "max_cost"),
            ({"temperature": 2.5}, "temperature"),
            ({"temperature": -0.1}, "temperature"),
            ({"temperature": math.nan}, "temperature"),
            ({"max_completion_tokens": 0}, "max_completion_tokens"),
        ],
    )
    def test_impossible_settings_are_refused_before_any_request(
        self, fake_client: FakeClient, team_member: Agent, tmp_path, kwargs, message
    ) -> None:
        with pytest.raises(ValueError, match=message):
            individual(team_member, tmp_path, **kwargs)

        assert fake_client.completions.calls == []
        assert not (tmp_path / PARTIAL_MEETING_DIR_NAME).exists()

    @pytest.mark.parametrize(
        "meeting, message",
        [
            (lambda lead, a, b: {"meeting_type": "team", "team_lead": lead}, "requires team lead"),
            (
                lambda lead, a, b: {"meeting_type": "team", "team_lead": lead, "team_members": ()},
                "requires team lead",
            ),
            (
                lambda lead, a, b: {
                    "meeting_type": "team",
                    "team_lead": lead,
                    "team_members": (a,),
                    "team_member": b,
                },
                "does not require individual",
            ),
            (
                lambda lead, a, b: {"meeting_type": "team", "team_lead": lead, "team_members": (lead, a)},
                "separate from team members",
            ),
            (
                lambda lead, a, b: {"meeting_type": "team", "team_lead": lead, "team_members": (a, a)},
                "unique",
            ),
            (
                lambda lead, a, b: {
                    "meeting_type": "team",
                    "team_lead": lead,
                    "team_members": (a,),
                    "critic": b,
                },
                "does not use a separate critic",
            ),
            (lambda lead, a, b: {"meeting_type": "individual"}, "requires individual"),
            (
                lambda lead, a, b: {"meeting_type": "individual", "team_member": a, "team_lead": lead},
                "does not require team lead",
            ),
            (
                lambda lead, a, b: {"meeting_type": "individual", "team_member": a, "critic": a},
                "Critic must be separate",
            ),
            (lambda lead, a, b: {"meeting_type": "panel", "team_member": a}, "Invalid meeting type"),
        ],
    )
    def test_malformed_meetings_are_refused_before_any_request(
        self,
        fake_client: FakeClient,
        team_lead: Agent,
        team_member: Agent,
        second_team_member: Agent,
        tmp_path,
        meeting,
        message,
    ) -> None:
        kwargs = meeting(team_lead, team_member, second_team_member)

        with pytest.raises(ValueError, match=message):
            hold_meeting(agenda="A.", save_dir=tmp_path, **kwargs)

        assert fake_client.completions.calls == []

    def test_given_client_is_used_instead_of_building_one(
        self, monkeypatch: pytest.MonkeyPatch, team_member: Agent, tmp_path
    ) -> None:
        def refuse(*args, **kwargs) -> None:
            raise AssertionError("a chat model was built although a client was given")

        monkeypatch.setattr(import_module("virtual_lab.llm"), "get_llm", refuse)
        client = FakeClient()

        individual(team_member, tmp_path, client=client)

        assert len(client.completions.calls) == 1

    def test_completion_limit_is_sent_with_every_request(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(Verdict(decision="go"))]

        individual(
            team_member, tmp_path, num_rounds=1, max_completion_tokens=500, output_schema=Verdict
        )

        assert len(fake_client.completions.calls) == 4
        assert all(call["max_completion_tokens"] == 500 for call in fake_client.completions.calls)

    def test_no_completion_limit_is_sent_unless_asked_for(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        individual(team_member, tmp_path)

        assert "max_completion_tokens" not in fake_client.completions.calls[0]


class TestBudget:
    def test_meeting_stops_before_the_request_that_would_pass_the_limit(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(BudgetExceededError) as caught:
            individual(team_member, tmp_path, num_rounds=2, max_cost=1.5 * ONE_RESPONSE)

        assert len(fake_client.completions.calls) == 2
        assert caught.value.spent == pytest.approx(2 * ONE_RESPONSE)
        assert caught.value.limit == pytest.approx(1.5 * ONE_RESPONSE)

    def test_what_was_bought_before_the_limit_is_kept(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [text_response("First."), text_response("Second.")]

        with pytest.raises(BudgetExceededError):
            individual(team_member, tmp_path, num_rounds=2, max_cost=1.5 * ONE_RESPONSE)

        record = partial_record(tmp_path)
        messages = [turn["message"] for turn in partial_transcript(tmp_path)]

        assert "First." in messages and "Second." in messages
        assert record["status"] == "failed"
        assert record["error"]["type"] == "BudgetExceededError"
        assert record["usage"]["num_calls"] == 2
        assert record["max_cost"] == pytest.approx(1.5 * ONE_RESPONSE)

    def test_zero_limit_makes_no_request(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(BudgetExceededError):
            individual(team_member, tmp_path, max_cost=0)

        assert fake_client.completions.calls == []

    def test_limit_covers_the_structured_output_request(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # Two discussion turns fit under the limit; the extraction after them must not
        with pytest.raises(BudgetExceededError):
            individual(
                team_member,
                tmp_path,
                num_rounds=1,
                max_cost=2.5 * ONE_RESPONSE,
                output_schema=Verdict,
            )

        assert fake_client.completions.parse_calls == []

    def test_limit_covers_each_tool_iteration_within_a_turn(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        lookup = Tool(
            name="lookup",
            description="Looks something up.",
            parameters={"type": "object", "properties": {}},
            function=lambda: "found",
        )
        fake_client.completions.responses = [tool_call_response("lookup")] * MAX_TOOL_ITERATIONS

        with pytest.raises(BudgetExceededError):
            individual(team_member, tmp_path, tools=(lookup,), max_cost=1.5 * ONE_RESPONSE)

        assert len(fake_client.completions.calls) == 2

    def test_limit_is_refused_for_a_model_with_no_price(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        with pytest.raises(CostUnknownError, match=UNPRICED_MODEL):
            individual(team_member.with_model(UNPRICED_MODEL), tmp_path, max_cost=1.0)

        assert fake_client.completions.calls == []

    def test_limit_is_refused_for_an_unpriced_critic(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        critic = Agent(
            title="Critic",
            expertise="criticism",
            goal="find flaws",
            role="critique",
            model=UNPRICED_MODEL,
        )

        with pytest.raises(CostUnknownError):
            individual(team_member, tmp_path, critic=critic, max_cost=1.0)

    def test_response_without_usage_stops_a_limited_meeting(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # A response that reports no usage cannot be counted, and counting it as free would
        # let a meeting spend past its limit without ever noticing
        response = text_response()
        response.usage = None
        fake_client.completions.responses = [response]

        with pytest.raises(CostUnknownError):
            individual(team_member, tmp_path, num_rounds=1, max_cost=1.0)

        assert len(fake_client.completions.calls) == 1
        assert partial_record(tmp_path)["usage"]["unreported_calls"] == 1

    def test_response_without_usage_leaves_an_unlimited_meeting_with_unknown_cost(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        response = text_response()
        response.usage = None
        fake_client.completions.responses = [response]

        result = individual(team_member, tmp_path, num_rounds=1)

        assert result.cost is None
        assert result.usage.unreported_calls == 1
        assert result.usage.num_calls == 3


class TestUsageCallback:
    def test_caller_hears_about_every_response_as_it_arrives(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        seen: list[int] = []

        individual(team_member, tmp_path, num_rounds=1, on_usage=lambda u: seen.append(u.num_calls))

        assert seen == [1, 2, 3]

    def test_caller_can_stop_the_meeting_and_keep_what_it_did(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        class Enough(Exception):
            pass

        def stop_after_two(usage) -> None:
            if usage.num_calls == 2:
                raise Enough

        with pytest.raises(Enough):
            individual(team_member, tmp_path, num_rounds=2, on_usage=stop_after_two)

        assert len(fake_client.completions.calls) == 2
        assert partial_record(tmp_path)["usage"]["num_calls"] == 2


class TestTeamMeetingWithoutDiscussion:
    def test_team_lead_answers_the_agenda_alone(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path
    ) -> None:
        result = hold_meeting(
            meeting_type="team",
            agenda="Pick a target.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=(team_member,),
            num_rounds=0,
        )

        speakers = [turn["agent"] for turn in result.discussion if turn["agent"] != "User"]
        final_prompt = result.discussion[-2]["message"]
        start_prompt = result.discussion[0]["message"]

        assert speakers == [team_lead.title]
        assert "summarize the meeting" in final_prompt
        assert "Pick a target." in final_prompt
        assert "no discussion" in start_prompt
        assert "rounds" not in start_prompt


class TestTeamMeetingPrompts:
    def test_team_lead_opens_synthesises_and_closes(
        self,
        fake_client: FakeClient,
        team_lead: Agent,
        team_member: Agent,
        tmp_path,
    ) -> None:
        result = hold_meeting(
            meeting_type="team",
            agenda="Pick a target.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=(team_member,),
            num_rounds=2,
        )

        # Each prompt is followed by the reply of whoever it was addressed to
        lead_prompts = [
            result.discussion[index - 1]["message"]
            for index, turn in enumerate(result.discussion)
            if turn["agent"] == team_lead.title
        ]

        assert len(lead_prompts) == 3
        assert "initial thoughts" in lead_prompts[0]
        assert "This concludes round 1 of 2" in lead_prompts[1]
        assert "summarize the meeting" in lead_prompts[2]
        assert "This will continue for 2 rounds" in result.discussion[0]["message"]


class TestTemperature:
    def test_model_that_refuses_a_temperature_runs_at_its_default(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [bad_request("temperature")]

        result = individual(team_member, tmp_path, num_rounds=1)
        calls = fake_client.completions.calls

        assert calls[0]["temperature"] == 0.2
        assert all("temperature" not in call for call in calls[1:])
        assert result.usage.num_calls == 3
        assert result.record.models_at_default_temperature == [TEST_MODEL]

    def test_refusal_is_remembered_rather_than_paid_for_again(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [bad_request("temperature")]

        individual(team_member, tmp_path, save_name="first")
        individual(team_member, tmp_path, save_name="second")

        assert len(fake_client.completions.calls) == 3

    def test_structured_output_request_also_falls_back(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            bad_request("temperature"),
            parsed_response(Verdict(decision="go")),
        ]

        result = individual(team_member, tmp_path, output_schema=Verdict)

        assert result.output == Verdict(decision="go")
        assert "temperature" not in fake_client.completions.parse_calls[-1]

    def test_model_that_takes_no_temperature_parameter_at_all_runs_at_its_default(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [
            bad_request("temperature", "Unsupported parameter", code="unsupported_parameter")
        ]

        result = individual(team_member, tmp_path)

        assert "temperature" not in fake_client.completions.calls[-1]
        assert result.record.models_at_default_temperature == [TEST_MODEL]

    def test_temperature_out_of_the_model_range_is_an_error_not_a_fallback(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # Reported against "temperature" too, but it is the caller's mistake, and falling back
        # would also stop every later meeting on this model from sending its temperature
        fake_client.completions.responses = [
            bad_request("temperature", "Expected a value <= 1", code="decimal_above_max_value")
        ]

        with pytest.raises(openai.BadRequestError):
            individual(team_member, tmp_path, temperature=1.5)

        individual(team_member, tmp_path, save_name="next")

        assert len(fake_client.completions.calls) == 2
        assert fake_client.completions.calls[-1]["temperature"] == 0.2

    def test_other_refusals_are_not_retried(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [bad_request("messages", "Invalid messages")]

        with pytest.raises(openai.BadRequestError):
            individual(team_member, tmp_path)

        assert len(fake_client.completions.calls) == 1

    def test_model_that_accepts_a_temperature_is_recorded_as_using_it(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        result = individual(team_member, tmp_path, temperature=0.7)

        assert fake_client.completions.calls[0]["temperature"] == 0.7
        assert result.record.models_at_default_temperature == []


class TestTruncatedResponses:
    def test_response_cut_off_before_any_answer_stops_the_meeting(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # Recorded as an empty turn, it would be read by the critic and the summary as a reply
        fake_client.completions.responses = [truncated_response("")]

        with pytest.raises(TruncatedResponseError, match=team_member.title):
            individual(team_member, tmp_path, num_rounds=1)

        assert partial_record(tmp_path)["usage"]["num_calls"] == 1

    def test_response_cut_off_partway_is_kept_and_flagged(
        self, fake_client: FakeClient, team_member: Agent, tmp_path, capsys
    ) -> None:
        fake_client.completions.responses = [truncated_response("The first half")]

        result = individual(team_member, tmp_path)
        response_turn = next(turn for turn in result.record.turns if turn.kind == "response")

        assert result.summary == "The first half"
        assert response_turn.finish_reason == "length"
        assert "cut short" in capsys.readouterr().out

    def test_complete_response_records_why_it_stopped(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        result = individual(team_member, tmp_path)
        response_turn = next(turn for turn in result.record.turns if turn.kind == "response")

        assert response_turn.finish_reason == "stop"


class TestStructuredOutputFailures:
    def test_answer_cut_off_by_the_token_limit_is_a_structured_output_error(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        cut_off = text_response("{", usage=make_usage(prompt_tokens=300, completion_tokens=50))
        fake_client.completions.parsed_responses = [LengthFinishReasonError(completion=cut_off)]

        with pytest.raises(StructuredOutputError, match="ran out of tokens"):
            individual(team_member, tmp_path, output_schema=Verdict)

        # The cut-off request was still paid for
        usage = partial_record(tmp_path)["usage"]
        assert usage["num_calls"] == 2
        assert usage["input_tokens"] == 100 + 300

    def test_answer_failing_the_schema_validators_is_a_structured_output_error(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        try:
            StrictVerdict(decision="maybe")
        except ValidationError as error:
            invalid = error

        fake_client.completions.parsed_responses = [invalid]

        with pytest.raises(StructuredOutputError, match="StrictVerdict"):
            individual(team_member, tmp_path, output_schema=StrictVerdict)

        # The SDK raises before handing back the response, so what it used is unknown
        usage = partial_record(tmp_path)["usage"]
        assert usage["num_calls"] == 2
        assert usage["unreported_calls"] == 1

    def test_answer_stopped_by_the_content_filter_is_a_structured_output_error(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            ContentFilterFinishReasonError(completion=text_response(""))
        ]

        with pytest.raises(StructuredOutputError, match="content filter"):
            individual(team_member, tmp_path, output_schema=Verdict)

        assert partial_record(tmp_path)["usage"]["num_calls"] == 2

    def test_failed_repair_request_is_counted(self, team_member: Agent) -> None:
        client = FakeClient()
        client.completions.parsed_responses = [
            parsed_response(None, refusal="No.", usage=make_usage(prompt_tokens=700))
        ]
        usage = MeetingUsage()

        with pytest.raises(StructuredOutputError):
            request_repair(
                llm=fake_llm(client),
                author=team_member,
                model=TEST_MODEL,
                files=[],
                filename="analysis.py",
                report="It failed.",
                attempt=1,
                max_attempts=2,
                temperature=0.2,
                usage=usage,
            )

        assert usage.num_calls == 1
        assert usage.input_tokens == 700

    def test_refusal_is_counted_towards_the_meeting_cost(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [
            parsed_response(None, refusal="I cannot help with that.")
        ]

        with pytest.raises(StructuredOutputError, match="refused"):
            individual(team_member, tmp_path, output_schema=Verdict)

        usage = partial_record(tmp_path)["usage"]
        assert usage["num_calls"] == 2
        assert usage["unreported_calls"] == 0


class TestInterruption:
    def test_interrupted_meeting_keeps_what_it_had_done(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [text_response("First."), KeyboardInterrupt()]

        with pytest.raises(KeyboardInterrupt):
            individual(team_member, tmp_path, num_rounds=1)

        record = partial_record(tmp_path)

        assert record["status"] == "failed"
        assert record["error"]["type"] == "KeyboardInterrupt"
        assert record["usage"]["num_calls"] == 1
        assert "First." in [turn["message"] for turn in partial_transcript(tmp_path)]


class TestRecord:
    def test_what_the_meeting_was_asked_is_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(Verdict(decision="go"))]

        result = individual(
            team_member,
            tmp_path,
            agenda_questions=("Which target?",),
            agenda_rules=("Cite sources.",),
            summaries=("An earlier meeting.",),
            contexts=("Some context.", "More context."),
            output_schema=Verdict,
            max_cost=1.0,
            max_completion_tokens=800,
        )
        record = json.loads(result.record_path.read_text())

        assert record["agenda"] == "Design a nanobody."
        assert record["agenda_questions"] == ["Which target?"]
        assert record["agenda_rules"] == ["Cite sources."]
        assert len(record["summaries_sha256"]) == 1
        assert len(record["contexts_sha256"]) == 2
        assert record["contexts_sha256"][0] != record["contexts_sha256"][1]
        assert "An earlier meeting." not in json.dumps(record)
        assert record["output_schema"] == "Verdict"
        assert record["max_cost"] == 1.0
        assert record["max_completion_tokens"] == 800
        assert record["max_tool_iterations"] == MAX_TOOL_ITERATIONS

    def test_prices_the_cost_was_computed_at_are_recorded(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        result = individual(team_member, tmp_path)
        prices = json.loads(result.record_path.read_text())["prices"]

        assert set(prices) == {TEST_MODEL}
        assert prices[TEST_MODEL]["input"] > 0
        assert prices[TEST_MODEL]["output"] > prices[TEST_MODEL]["input"]

    def test_unpriced_model_is_recorded_without_a_price(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.responses = [text_response(model=UNPRICED_MODEL)]

        result = individual(team_member.with_model(UNPRICED_MODEL), tmp_path)

        assert json.loads(result.record_path.read_text())["prices"] == {UNPRICED_MODEL: None}

    def test_long_tool_arguments_are_shortened_in_the_record(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        lookup = Tool(
            name="lookup",
            description="Looks something up.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            function=lambda query: "found",
        )
        fake_client.completions.responses = [
            tool_call_response("lookup", {"query": "x" * 10_000}),
            text_response("Done."),
        ]

        result = individual(team_member, tmp_path, tools=(lookup,))
        response_turn = next(turn for turn in result.record.turns if turn.kind == "response")
        (call,) = response_turn.tool_calls

        assert call["name"] == "lookup"
        assert len(call["arguments"]) <= MAX_RECORDED_ARGUMENT_CHARS + 100
        assert call["arguments"].startswith('{"query": "xxx')
