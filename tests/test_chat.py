"""Tests for a conversation with the head of a lab: answering, running code, bringing in the team,
steering, and keeping the conversation so that it can be carried on."""

import io
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from virtual_lab import chat as chat_module
from virtual_lab.agent import Agent
from virtual_lab.chat import (
    STOPPED_REPLY,
    Attachment,
    Chat,
    ChatBusyError,
    ChatClosedError,
    ChatReply,
    UploadTooLargeError,
    describe_size,
    safe_filename,
    title_of,
)
from virtual_lab.constants import HUMAN_SPEAKER
from virtual_lab.events import ChatEvent
from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC
from virtual_lab.session import LocalSession
from virtual_lab.utils import ContextLengthExceededError, CostUnknownError, compute_token_cost

from conftest import TEST_MODEL, FakeClient, make_usage, text_response, tool_call_response
from test_events import PlottingSession
from test_resources import small_resources

LEAD = PRINCIPAL_INVESTIGATOR.with_model(TEST_MODEL)
CRITIC = SCIENTIFIC_CRITIC.with_model(TEST_MODEL)
IMMUNOLOGIST = Agent(
    title="Immunologist",
    expertise="antibody engineering",
    goal="design nanobodies",
    role="advise on immunogenicity",
    model=TEST_MODEL,
)
BIOLOGIST = Agent(
    title="Computational Biologist",
    expertise="protein structure prediction",
    goal="model complexes",
    role="run simulations",
    model=TEST_MODEL,
)
TEAM = (IMMUNOLOGIST, BIOLOGIST, CRITIC)

# What one of the fake model's responses costs, which every response here is
PER_RESPONSE = compute_token_cost(TEST_MODEL, 100, 20)


def make_chat(tmp_path: Path, **options: Any) -> Chat:
    return Chat(tmp_path / "chat", LEAD, TEAM, **{"resources": "none", "max_retries": 0, **options})


def kinds(events: list[ChatEvent], leave_out: tuple[str, ...] = ("status",)) -> list[str]:
    return [event.kind for event in events if event.kind not in leave_out]


def of(events: list[ChatEvent], kind: str) -> list[ChatEvent]:
    return [event for event in events if event.kind == kind]


def lab_events(events: list[ChatEvent], kind: str | None = None) -> list[ChatEvent]:
    return [event for event in of(events, "lab") if kind is None or event.data["event"] == kind]


def tool_calls_response(*calls: tuple[str, dict[str, Any]]) -> ChatCompletion:
    """A completion in which the model asks for several tool calls at once."""
    return ChatCompletion(
        id="test",
        model=TEST_MODEL,
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
                            id=f"call_{number}",
                            type="function",
                            function=Function(name=name, arguments=json.dumps(arguments)),
                        )
                        for number, (name, arguments) in enumerate(calls, start=1)
                    ],
                ),
            )
        ],
    )


def team_meeting(agenda: str = "Design a nanobody.", members: tuple[str, ...] = ("Immunologist",), **extra: Any) -> Any:
    return tool_call_response("convene_team", {"agenda": agenda, "members": list(members), **extra})


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def request_messages(fake_client: FakeClient, call: int) -> list[dict[str, Any]]:
    return fake_client.completions.calls[call]["messages"]


@pytest.fixture(autouse=True)
def model_client(fake_client: FakeClient) -> FakeClient:
    """A conversation builds its models when it is made, so every test needs the fake before that."""
    return fake_client


@pytest.fixture
def session(tmp_path: Path):
    with PlottingSession(tmp_path / "work", warn=False, timeout=20) as opened:
        yield opened


class TestAnswering:
    def test_the_lead_answers_a_message_and_the_conversation_is_kept(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Nanobodies are small antibodies.")]
        chat = make_chat(tmp_path)

        reply = chat.send("What is a nanobody?")

        assert reply == chat.last_reply
        assert (reply.status, reply.text, reply.error) == ("answered", "Nanobodies are small antibodies.", None)
        assert reply.usage["num_calls"] == 1
        assert reply.usage["cost"] == pytest.approx(PER_RESPONSE)
        assert chat.state == "idle"
        assert chat.turns == 1

        events = chat.events_since()
        assert kinds(events) == ["user", "title", "usage", "answer"]
        user, title, usage, answer = (event for event in events if event.kind != "status")
        assert (user.speaker, user.text, user.data["attachments"]) == (HUMAN_SPEAKER, "What is a nanobody?", [])
        assert title.text == "What is a nanobody?"
        assert usage.data["num_calls"] == 1 and usage.data["max_cost"] is None
        assert (answer.text, answer.data["usage"]["num_calls"]) == ("Nanobodies are small antibodies.", 1)
        assert {event.turn for event in events} == {1}
        assert [event.id for event in events] == sorted(event.id for event in events)

        # What the lead is sent: its own prompt, as the system's, and the researcher's message
        sent = request_messages(fake_client, 0)
        assert sent[0]["role"] == "system" and "You are a Principal Investigator" in sent[0]["content"]
        assert "convene_team" in sent[0]["content"] and "Immunologist: antibody engineering" in sent[0]["content"]
        assert sent[1] == {"role": "user", "content": "What is a nanobody?"}
        assert [tool["function"]["name"] for tool in fake_client.completions.calls[0]["tools"]] == [
            "convene_team",
            "consult",
        ]

        assert chat.messages == [
            {"role": "user", "content": "What is a nanobody?"},
            {"role": "assistant", "name": "Principal_Investigator", "content": "Nanobodies are small antibodies."},
        ]
        assert lines(tmp_path / "chat" / "messages.jsonl") == chat.messages
        saved = [ChatEvent.from_dict(item) for item in lines(tmp_path / "chat" / "events.jsonl")]
        assert saved == [event for event in events if event.kind != "status"]

        info = json.loads((tmp_path / "chat" / "chat.json").read_text())
        assert info["title"] == "What is a nanobody?" and info["turns"] == 1 and info["running"] is False
        assert info["lead"]["title"] == "Principal Investigator"
        assert [member["title"] for member in info["team"]] == [
            "Immunologist",
            "Computational Biologist",
            "Scientific Critic",
        ]
        assert info["spent"] == pytest.approx(PER_RESPONSE) and info["usage"]["num_calls"] == 1

    def test_a_lead_with_no_team_works_alone(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [text_response("An answer.")]
        chat = Chat(tmp_path / "chat", LEAD, resources="none", max_retries=0)

        chat.send("Hello")

        assert "tools" not in fake_client.completions.calls[0]
        system = request_messages(fake_client, 0)[0]["content"]
        assert "convene_team" not in system and "Your team" not in system

    @pytest.mark.parametrize("options", [{"max_delegations": 0}, {"team": (CRITIC,)}])
    def test_the_team_cannot_be_brought_in_if_there_is_nobody_to_bring_or_no_way_to(
        self, fake_client: FakeClient, tmp_path: Path, options: dict[str, Any]
    ) -> None:
        fake_client.completions.responses = [text_response("An answer.")]
        team = options.pop("team", TEAM)
        chat = Chat(tmp_path / "chat", LEAD, team, resources="none", max_retries=0, **options)

        chat.send("Hello")

        assert "tools" not in fake_client.completions.calls[0]

    def test_a_reply_is_streamed_as_it_is_written_and_not_kept(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [text_response("Nanobodies are small.")]
        told: list[ChatEvent] = []
        chat = make_chat(tmp_path, stream=True, on_event=told.append)

        chat.send("What is a nanobody?")

        writing = of(told, "writing")
        assert [event.text for event in writing] == ["Nanobodies ", "Nanobodies are ", "Nanobodies are small."]
        assert {(event.speaker, event.data["request"], event.data["model"]) for event in writing} == {
            ("Principal Investigator", 1, TEST_MODEL)
        }
        # The answer says it whole, so what was being written is not what a page that comes later is shown
        assert of(chat.events_since(), "writing") == []
        assert "writing" not in {item["kind"] for item in lines(tmp_path / "chat" / "events.jsonl")}

    def test_a_page_that_comes_while_a_reply_is_written_is_shown_what_is_written(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Nanobodies are small.")]
        seen: list[list[ChatEvent]] = []
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "writing" and event.text == "Nanobodies are ":
                seen.append(chat.events_since())

        chat = make_chat(tmp_path, stream=True, on_event=on_event)
        chat.send("What is a nanobody?")

        (during,) = seen
        assert kinds(during) == ["user", "title", "writing"]
        assert of(during, "writing")[0].text == "Nanobodies are "

    def test_a_message_needs_something_in_it(self, fake_client: FakeClient, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)

        with pytest.raises(ValueError, match="something in it"):
            chat.send("   ")

        assert chat.state == "idle" and chat.turns == 0 and fake_client.completions.calls == []

    def test_a_closed_conversation_takes_no_messages(self, fake_client: FakeClient, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)
        chat.close()

        with pytest.raises(ChatClosedError):
            chat.send("Hello")

    def test_the_title_is_the_first_message_unless_it_was_given(self, fake_client: FakeClient, tmp_path: Path) -> None:
        chat = make_chat(tmp_path, title="Nanobody design")
        chat.send("What is a nanobody?")
        chat.rename("  Better   title ")

        assert chat.title == "Better title"
        assert [event.text for event in of(chat.events_since(), "title")] == ["Better title"]
        assert json.loads((tmp_path / "chat" / "chat.json").read_text())["title"] == "Better title"

    def test_the_first_message_gives_the_title_when_none_was_given(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path)
        chat.send("What is a nanobody?\nAnd how is it made?")

        assert chat.title == "What is a nanobody?"
        assert [event.text for event in of(chat.events_since(), "title")] == ["What is a nanobody?"]

    def test_titles_are_made_from_the_first_line(self) -> None:
        assert title_of("\n  First   line here\nSecond", []) == "First line here"
        assert title_of("x" * 100, []) == "x" * 79 + "…"
        assert title_of("", [Attachment("counts.csv", "uploads/counts.csv", 3)]) == "counts.csv"


class TestCode:
    def test_the_lead_runs_code_and_the_figure_is_told_of(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        session.run("earlier = 1")
        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "print(6 * 7)"}),
            text_response("It is 42."),
        ]
        chat = make_chat(tmp_path, session=session)

        reply = chat.send("What is 6 times 7?")

        assert reply.text == "It is 42."
        events = chat.events_since()
        assert kinds(events) == ["user", "title", "usage", "tool_calls", "cell", "tool_output", "usage", "answer"]
        (calls,) = of(events, "tool_calls")
        assert calls.speaker == "Principal Investigator"
        assert calls.data["calls"] == [{"id": "call_1", "name": "run_code", "arguments": '{"code": "print(6 * 7)"}'}]

        # Only what the conversation ran, not what was run before it
        (cell,) = of(events, "cell")
        assert (cell.data["code"], cell.data["status"], cell.data["output"].strip()) == ("print(6 * 7)", "ok", "42")
        assert cell.data["plots"] == ["plots/figure_1.png"]
        assert cell.data["plot_paths"] == [str(session.directory / "plots" / "figure_1.png")]
        assert cell.data["call"] == "call_1"
        (output,) = of(events, "tool_output")
        assert (output.data["call"], output.data["name"]) == ("call_1", "run_code")
        assert "42" in output.text

        # The lead is told how to run code, and the second request has the result in it
        first = fake_client.completions.calls[0]
        assert "run_code" in [tool["function"]["name"] for tool in first["tools"]]
        assert f"The code runs in {session.directory}" in first["messages"][0]["content"]
        assert request_messages(fake_client, 1)[-1]["role"] == "tool"
        assert [message["role"] for message in chat.messages] == ["user", "assistant", "tool", "assistant"]
        assert chat.messages[1]["tool_calls"][0]["id"] == "call_1"

    def test_a_lead_without_a_session_is_told_it_cannot_run_code(self, fake_client: FakeClient, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)

        chat.send("Hello")

        assert "There is no session to run code in" in request_messages(fake_client, 0)[0]["content"]

    def test_tools_are_withheld_when_the_lead_has_used_all_its_calls(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "print(1)"}),
            text_response("Done."),
        ]
        chat = make_chat(tmp_path, session=session, max_tool_iterations=1)

        chat.send("Go")

        assert "tools" in fake_client.completions.calls[0]
        assert "tools" not in fake_client.completions.calls[1]

    def test_a_tool_the_lead_makes_up_is_answered_with_an_error(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [tool_call_response("run_code", {"code": "1"}), text_response("Sorry.")]
        chat = make_chat(tmp_path)

        chat.send("Go")

        (output,) = of(chat.events_since(), "tool_output")
        assert output.text.startswith('Error: unknown tool "run_code"')

    def test_tools_given_to_the_conversation_are_the_leads_to_call(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        from virtual_lab.tools import Tool

        look_up = Tool(
            name="look_up",
            description="Looks something up.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            function=lambda query: f"Found {query}.",
            instructions="Look things up sparingly.",
        )
        fake_client.completions.responses = [tool_call_response("look_up", {"query": "KP.3"}), text_response("Done.")]
        chat = make_chat(tmp_path, tools=(look_up,))

        chat.send("Go")

        assert [tool["function"]["name"] for tool in fake_client.completions.calls[0]["tools"]][0] == "look_up"
        assert "Look things up sparingly." in request_messages(fake_client, 0)[0]["content"]
        assert of(chat.events_since(), "tool_output")[0].text == "Found KP.3."


class TestFiles:
    def test_a_file_is_saved_where_the_session_finds_it_under_a_name_that_is_safe(
        self, session: LocalSession, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path, session=session)

        saved = chat.save_upload("..\\..//data/.counts.csv", b"a,b\n1,2\n")

        assert saved == Attachment(name="counts.csv", path="uploads/counts.csv", size=8)
        assert (session.directory / "uploads" / "counts.csv").read_bytes() == b"a,b\n1,2\n"
        assert chat.uploads_dir == session.directory / "uploads"
        assert session.run("print(open('uploads/counts.csv').read().strip().splitlines())").output.strip() == (
            "['a,b', '1,2']"
        )

    def test_without_a_session_files_are_kept_with_the_conversation(self, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)

        chat.save_upload("notes.txt", b"x")

        assert (tmp_path / "chat" / "uploads" / "notes.txt").read_bytes() == b"x"

    def test_a_file_is_never_replaced_but_the_same_file_is_not_kept_twice(self, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)

        first = chat.save_upload("counts.csv", b"one")
        same = chat.save_upload("counts.csv", io.BytesIO(b"one"))
        other = chat.save_upload("counts.csv", b"two")
        third = chat.save_upload("counts.csv", b"three")

        assert [first.name, same.name, other.name, third.name] == [
            "counts.csv",
            "counts.csv",
            "counts (2).csv",
            "counts (3).csv",
        ]
        assert (tmp_path / "chat" / "uploads" / "counts.csv").read_bytes() == b"one"
        assert sorted(path.name for path in (tmp_path / "chat" / "uploads").iterdir()) == [
            "counts (2).csv",
            "counts (3).csv",
            "counts.csv",
        ]

    def test_a_file_that_is_too_large_is_refused_and_leaves_nothing(self, tmp_path: Path) -> None:
        chat = make_chat(tmp_path, max_upload_bytes=10)

        with pytest.raises(UploadTooLargeError, match="larger than the 10 bytes"):
            chat.save_upload("big.bin", io.BytesIO(b"x" * 11))

        assert list((tmp_path / "chat" / "uploads").iterdir()) == []
        assert chat.save_upload("ok.bin", b"x" * 10).size == 10

    def test_names_are_made_safe(self) -> None:
        assert safe_filename("/etc/passwd") == "passwd"
        assert safe_filename("C:\\Users\\me\\data.csv") == "data.csv"
        assert safe_filename("..") == "upload"
        assert safe_filename("a\x00b\n.txt") == "ab.txt"
        assert safe_filename("") == "upload"
        long = safe_filename("n" * 400 + ".csv")
        assert len(long) == 150 and long.endswith(".csv")
        assert describe_size(8) == "8 bytes" and describe_size(1536) == "1.5 KB"
        assert describe_size(5 * 1024**2) == "5.0 MB" and describe_size(3 * 1024**3) == "3.0 GB"

    def test_the_lead_is_told_of_the_files_attached_to_a_message(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path, session=session)
        counts = chat.save_upload("counts.csv", b"a,b\n1,2\n")
        chat.save_upload("other.csv", b"x")

        chat.send("Look at this.", attachments=[counts, "other.csv"])

        content = request_messages(fake_client, 0)[-1]["content"]
        assert content == (
            "Look at this.\n\nThe researcher attached these files, which are in the session's working directory:\n"
            "- uploads/counts.csv (8 bytes)\n- uploads/other.csv (1 bytes)"
        )
        (user,) = of(chat.events_since(), "user")
        assert user.data["attachments"] == [counts.to_dict(), Attachment("other.csv", "uploads/other.csv", 1).to_dict()]

    def test_a_message_can_be_only_a_file_and_without_a_session_the_lead_is_told_it_cannot_open_it(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path)
        chat.save_upload("counts.csv", b"abc")

        chat.send("", attachments=["counts.csv"])

        assert request_messages(fake_client, 0)[-1]["content"] == (
            "The researcher attached these files, but there is no session to open them in:\n"
            "- uploads/counts.csv (3 bytes)"
        )
        assert chat.title == "counts.csv"

    @pytest.mark.parametrize("name", ["missing.csv", "../outside.csv", "uploads/counts.csv"])
    def test_a_file_that_was_not_attached_is_refused(self, name: str, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)
        chat.save_upload("counts.csv", b"abc")
        (tmp_path / "outside.csv").write_text("secret")

        with pytest.raises(ValueError, match="was attached to this conversation"):
            chat.send("Look.", attachments=[name])

        assert chat.turns == 0


class TestTeam:
    def test_the_lead_convenes_the_team_and_is_given_the_summary(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [
            team_meeting(members=("Immunologist", "Scientific Critic"), rounds=1),
            text_response("Opening thoughts."),
            text_response("An immunology view."),
            text_response("A critique."),
            text_response("The summary of the team."),
            text_response("The lab found a summary."),
        ]
        chat = make_chat(tmp_path)

        reply = chat.send("Design a nanobody.")

        assert reply.text == "The lab found a summary."
        events = chat.events_since()
        assert kinds(events)[:5] == ["user", "title", "usage", "tool_calls", "lab"]
        assert kinds(events)[-3:] == ["tool_output", "usage", "answer"]

        # The meeting's events, as the conversation's, told of with the call that started it
        assert {event.data["call"] for event in of(events, "lab")} == {"call_1"}
        assert {event.data["meeting"] for event in of(events, "lab")} == {"lab_001"}
        assert [event.data["event"] for event in of(events, "lab")][:2] == ["started", "turn"]
        assert [event.data["event"] for event in of(events, "lab")][-1] == "finished"
        assert [(event.speaker, event.data["round"]) for event in lab_events(events, "turn")] == [
            ("Principal Investigator", 1),
            ("Immunologist", 1),
            ("Scientific Critic", 1),
            ("Principal Investigator", 2),
        ]
        said = [(event.speaker, event.text) for event in lab_events(events, "message")]
        assert said == [
            ("Principal Investigator", "Opening thoughts."),
            ("Immunologist", "An immunology view."),
            ("Scientific Critic", "A critique."),
            ("Principal Investigator", "The summary of the team."),
        ]
        assert lab_events(events, "finished")[0].text == "The summary of the team."
        started = lab_events(events, "started")[0]
        assert started.data["details"]["meeting_type"] == "team"
        assert started.data["details"]["agenda"] == "Design a nanobody."

        # The lead is given the summary, and who took part
        (output,) = of(events, "tool_output")
        assert output.data == {"call": "call_1", "name": "convene_team"}
        assert output.text == (
            "Immunologist, Scientific Critic took part in a discussion that you led, for 1 round of discussion. "
            "The summary you reached with them:\n\nThe summary of the team."
        )
        assert request_messages(fake_client, 5)[-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": output.text,
        }

        # It was held as a project holds a meeting, so it is saved, and counted
        (step,) = chat.project.steps
        assert (step.name, step.status) == ("lab_001", "completed")
        assert json.loads((tmp_path / "chat" / "lab" / "meetings" / "lab_001.json").read_text())[-1]["message"] == (
            "The summary of the team."
        )
        assert chat.usage.num_calls == 6 and chat.spent == pytest.approx(6 * PER_RESPONSE)
        assert [event.data["num_calls"] for event in of(events, "usage")] == [1, 2, 3, 4, 5, 6]

    def test_code_the_team_runs_is_the_meetings_to_tell_of_not_the_leads(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            team_meeting(members=("Immunologist",), rounds=1),
            text_response("Opening."),
            tool_call_response("run_code", {"code": "print(6 * 7)"}),
            text_response("It is 42."),
            text_response("Summary."),
            text_response("Done."),
        ]
        chat = make_chat(tmp_path, session=session)

        chat.send("Go")

        events = chat.events_since()
        assert of(events, "cell") == []
        (cell,) = lab_events(events, "cell")
        assert (cell.data["call"], cell.data["details"]["code"], cell.data["details"]["output"]) == (
            "call_1",
            "print(6 * 7)",
            "42\n",
        )
        assert cell.data["details"]["plot_paths"] == [str(session.directory / "plots" / "figure_1.png")]

    def test_a_meetings_prompts_and_usage_are_not_passed_on(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [team_meeting(rounds=0), text_response("Alone."), text_response("Done.")]
        chat = make_chat(tmp_path)

        chat.send("Design a nanobody.")

        events = chat.events_since()
        assert "usage" not in {event.data["event"] for event in of(events, "lab")}
        assert "prompt" not in {event.data["details"].get("kind") for event in lab_events(events, "message")}
        # The transcript has them
        transcript = json.loads((tmp_path / "chat" / "lab" / "meetings" / "lab_001.json").read_text())
        assert transcript[0]["agent"] == "User" and "Design a nanobody." in transcript[0]["message"]

    def test_the_lead_consults_one_scientist_who_is_reviewed_by_the_critic(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("consult", {"member": "Immunologist", "agenda": "Is it immunogenic?", "rounds": 1}),
            text_response("A first answer."),
            text_response("A critique."),
            text_response("A better answer."),
            text_response("Not immunogenic."),
        ]
        chat = make_chat(tmp_path)

        chat.send("Is it immunogenic?")

        events = chat.events_since()
        assert [(event.speaker, event.data["round"]) for event in lab_events(events, "turn")] == [
            ("Immunologist", 1),
            ("Scientific Critic", 1),
            ("Immunologist", 2),
        ]
        assert lab_events(events, "started")[0].data["details"]["meeting_type"] == "individual"
        (output,) = of(events, "tool_output")
        assert output.text == (
            "Immunologist worked on it, with the Scientific Critic reviewing, for 1 round of review. "
            "Their answer:\n\nA better answer."
        )

    def test_the_agenda_and_questions_are_the_meetings(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [
            tool_call_response(
                "consult",
                {
                    "member": "immunologist",
                    "agenda": "Is it immunogenic?",
                    "questions": ["Where?", "How much?"],
                    "rounds": 0,
                },
            ),
            text_response("Yes."),
            text_response("Done."),
        ]
        chat = make_chat(tmp_path)

        chat.send("Go")

        prompt = request_messages(fake_client, 1)[-1]["content"]
        assert "Is it immunogenic?" in prompt and "1. Where?" in prompt and "2. How much?" in prompt

    def test_the_meetings_options_are_the_conversations(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [team_meeting(rounds=0), text_response("Alone."), text_response("Done.")]
        chat = make_chat(tmp_path, temperature=0.7, max_completion_tokens=500)

        chat.send("Go")

        assert [call["temperature"] for call in fake_client.completions.calls] == [0.7, 0.7, 0.7]
        assert [call["max_completion_tokens"] for call in fake_client.completions.calls] == [500, 500, 500]

    @pytest.mark.parametrize(
        ("call", "arguments", "refusal"),
        [
            (
                "convene_team",
                {"agenda": "Go", "members": ["Astrologer"]},
                'Not held: there is no "Astrologer" on your team',
            ),
            ("convene_team", {"agenda": "Go", "members": []}, "Not held: name at least one member"),
            ("convene_team", {"agenda": "Go", "members": ["Principal Investigator"]}, "you lead the discussion"),
            ("convene_team", {"agenda": "  ", "members": ["Immunologist"]}, "Not held: the agenda is empty"),
            (
                "consult",
                {"member": "Scientific Critic", "agenda": "Go"},
                'Not held: "Scientific Critic" cannot be consulted',
            ),
            ("consult", {"member": "Nobody", "agenda": "Go"}, 'Not held: "Nobody" cannot be consulted'),
        ],
    )
    def test_a_request_that_cannot_be_held_is_answered_and_costs_nothing(
        self, fake_client: FakeClient, tmp_path: Path, call: str, arguments: dict[str, Any], refusal: str
    ) -> None:
        fake_client.completions.responses = [tool_call_response(call, arguments), text_response("I see.")]
        chat = make_chat(tmp_path)

        reply = chat.send("Go")

        assert reply.text == "I see."
        assert refusal in of(chat.events_since(), "tool_output")[0].text
        assert chat.project.steps == () and len(fake_client.completions.calls) == 2

    def test_a_request_the_lead_got_wrong_is_answered_with_the_error(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("convene_team", {"agenda": "Go"}),
            text_response("I see."),
        ]
        chat = make_chat(tmp_path)

        chat.send("Go")

        assert of(chat.events_since(), "tool_output")[0].text.startswith('Error running tool "convene_team": TypeError')

    def test_the_lead_may_start_only_so_many_sessions_for_one_message(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            team_meeting(rounds=0),
            text_response("Alone."),
            team_meeting(rounds=0),
            text_response("Done."),
        ]
        chat = make_chat(tmp_path, max_delegations=1)

        chat.send("Go")

        outputs = of(chat.events_since(), "tool_output")
        assert "which is the most allowed" in outputs[1].text and "worked 1 time on this message" in outputs[1].text
        assert [step.name for step in chat.project.steps] == ["lab_001"]

        # The next message has the team as it had at first
        fake_client.completions.responses = [team_meeting(rounds=0), text_response("Alone."), text_response("Done.")]
        chat.send("Again")

        assert [step.name for step in chat.project.steps] == ["lab_001", "lab_002"]

    def test_the_rounds_the_lead_asks_for_are_limited(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [
            tool_call_response("consult", {"member": "Immunologist", "agenda": "Go", "rounds": 9}),
            text_response("Alone."),
            text_response("Done."),
        ]
        chat = make_chat(tmp_path, max_rounds=0)

        chat.send("Go")

        # One answer, and no review
        assert len(fake_client.completions.calls) == 3
        assert "round" in of(chat.events_since(), "tool_output")[0].text

    def test_a_scientist_with_a_critic_of_the_rosters_own_is_reviewed_by_it(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        critic = Agent("Scientific Critic", "being a stickler", "find faults", "review", model=TEST_MODEL)
        fake_client.completions.responses = [
            tool_call_response("consult", {"member": "Immunologist", "agenda": "Go", "rounds": 1}),
            text_response("An answer."),
            text_response("A critique."),
            text_response("A better answer."),
            text_response("Done."),
        ]
        chat = Chat(tmp_path / "chat", LEAD, (IMMUNOLOGIST, critic), resources="none", max_retries=0)

        chat.send("Go")

        assert "being a stickler" in request_messages(fake_client, 2)[0]["content"]

    def test_a_failed_meeting_is_answered_with_the_error_and_the_lead_goes_on(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            team_meeting(rounds=0),
            RuntimeError("The server went away"),
            text_response("I could not ask the team."),
        ]
        chat = make_chat(tmp_path)

        reply = chat.send("Go")

        assert reply.status == "answered"
        (output,) = of(chat.events_since(), "tool_output")
        assert output.text == 'Error running tool "convene_team": RuntimeError: The server went away'
        assert lab_events(chat.events_since(), "failed")[0].text == "The server went away"
        assert [step.status for step in chat.project.steps] == ["failed"]

    def test_an_agent_cannot_have_the_title_of_the_researchers_notes(self, tmp_path: Path) -> None:
        impostor = Agent(HUMAN_SPEAKER, "x", "y", "z", model=TEST_MODEL)

        with pytest.raises(ValueError, match="no agent may have it"):
            Chat(tmp_path / "chat", LEAD, (impostor,))


class TestBudget:
    def test_the_leads_requests_and_the_teams_count_towards_one_limit(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            team_meeting(members=("Immunologist",), rounds=1),
            text_response("Opening."),
            text_response("A view."),
            text_response("Summary."),
            text_response("Done."),
        ]
        # Room for two requests, and the one that begins before the limit is reached
        chat = make_chat(tmp_path, max_cost=PER_RESPONSE * 2.5)

        reply = chat.send("Go")

        assert reply.status == "failed"
        assert reply.error is not None and reply.error.startswith("BudgetExceededError: The conversation has cost")
        assert len(fake_client.completions.calls) == 3
        events = chat.events_since()
        assert kinds(events)[-1] == "failed" and of(events, "failed")[0].data["type"] == "BudgetExceededError"
        assert of(events, "usage")[-1].data["max_cost"] == PER_RESPONSE * 2.5
        assert chat.spent == pytest.approx(3 * PER_RESPONSE)

        # The meeting was stopped by it, which the lead was told, and what it did is kept
        (output,) = of(events, "tool_output")
        assert output.text.startswith('Error running tool "convene_team": BudgetExceededError')
        assert [step.status for step in chat.project.steps] == ["failed"]
        assert chat.messages[-1]["role"] == "tool"

    def test_a_conversation_that_has_spent_its_limit_asks_for_nothing_more(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path, max_cost=PER_RESPONSE)
        assert chat.send("One").status == "answered"

        reply = chat.send("Two")

        assert reply.status == "failed" and reply.error is not None and "BudgetExceededError" in reply.error
        assert len(fake_client.completions.calls) == 1

    def test_a_limit_is_for_the_whole_conversation_and_every_run_of_it(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path, max_cost=PER_RESPONSE * 1.5)
        assert chat.send("One").status == "answered"
        chat.close()

        again = make_chat(tmp_path, max_cost=PER_RESPONSE * 1.5)
        assert again.send("Two").status == "answered"
        assert again.send("Three").status == "failed"

        # The second run began with what the first had spent, and the third was refused before it asked
        assert again.spent == pytest.approx(2 * PER_RESPONSE)
        assert len(fake_client.completions.calls) == 2

    def test_a_model_without_a_price_cannot_be_limited(self, fake_client: FakeClient, tmp_path: Path) -> None:
        unpriced = Agent("Immunologist", "x", "y", "z", model="not-a-priced-model")

        with pytest.raises(CostUnknownError, match="max_cost cannot be enforced"):
            Chat(tmp_path / "chat", LEAD, (unpriced,), max_cost=1.0)

        Chat(tmp_path / "chat", LEAD, (unpriced,))

    @pytest.mark.parametrize("limit", [-1.0, float("inf"), float("nan")])
    def test_a_limit_must_be_a_finite_amount(self, tmp_path: Path, limit: float) -> None:
        with pytest.raises(ValueError, match="max_cost must be a finite amount"):
            make_chat(tmp_path, max_cost=limit)


class TestSteering:
    def test_a_stopped_turn_ends_before_the_next_call_and_leaves_a_conversation_that_goes_on(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            tool_calls_response(("run_code", {"code": "print(1)"}), ("run_code", {"code": "print(2)"})),
        ]
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "tool_output":
                chat.stop()

        chat = make_chat(tmp_path, session=session, on_event=on_event)

        reply = chat.send("Go")

        assert reply.status == "stopped" and reply.text == "" and reply.error is None
        events = chat.events_since()
        assert kinds(events) == [
            "user",
            "title",
            "usage",
            "tool_calls",
            "cell",
            "tool_output",
            "tool_output",
            "stopped",
        ]
        outputs = of(events, "tool_output")
        assert [output.data["call"] for output in outputs] == ["call_1", "call_2"]
        assert outputs[1].text == "Not run: the researcher stopped the turn."
        assert of(events, "stopped")[0].text == "Stopped by you. What was done is kept."
        assert [cell.data["code"] for cell in of(events, "cell")] == ["print(1)"]
        assert chat.state == "idle"

        # Every call has its answer, so what is sent next is valid, and says what happened
        assert [message["role"] for message in chat.messages] == ["user", "assistant", "tool", "tool", "assistant"]
        assert chat.messages[-1]["content"] == STOPPED_REPLY

        fake_client.completions.responses = [text_response("Back again.")]
        chat.on_event = None
        assert chat.send("Hello again").text == "Back again."
        assert request_messages(fake_client, 1)[-2]["content"] == STOPPED_REPLY
        assert request_messages(fake_client, 1)[-1] == {"role": "user", "content": "Hello again"}

    def test_a_stop_before_anything_is_sent_ends_the_turn_without_a_request(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "user":
                chat.stop()

        chat = make_chat(tmp_path, on_event=on_event)

        assert chat.send("Go").status == "stopped"
        assert fake_client.completions.calls == []

    def test_a_stop_ends_a_reply_as_it_is_written(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [text_response("Nanobodies are small antibodies.")]
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "writing" and event.text.startswith("Nanobodies are"):
                chat.stop()

        chat = make_chat(tmp_path, stream=True, on_event=on_event)

        reply = chat.send("What is a nanobody?")

        assert reply.status == "stopped"
        assert chat.messages[-1]["content"] == STOPPED_REPLY
        assert chat.usage.num_calls == 0

    def test_a_stop_ends_a_team_session_where_it_is_and_keeps_what_was_said(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [team_meeting(rounds=1), text_response("Opening.")]
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "lab" and event.data["event"] == "message" and event.text == "Opening.":
                chat.stop()

        chat = make_chat(tmp_path, on_event=on_event)

        reply = chat.send("Go")

        assert reply.status == "stopped"
        assert len(fake_client.completions.calls) == 2
        events = chat.events_since()
        assert lab_events(events, "failed")[0].data["details"]["partial_path"] is not None
        assert (tmp_path / "chat" / "lab" / "meetings" / "partial" / "lab_001.json").is_file()
        assert of(events, "tool_output")[0].text == (
            "The researcher stopped the team before it finished, so there is no summary."
        )
        assert [step.status for step in chat.project.steps] == ["failed"]
        assert [message["role"] for message in chat.messages] == ["user", "assistant", "tool", "assistant"]

    def test_a_note_is_read_by_the_lead_before_its_next_request(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "print(1)"}),
            text_response("Done."),
        ]
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "tool_calls":
                assert chat.add_note("  Use the other file.  ")

        chat = make_chat(tmp_path, session=session, on_event=on_event)

        chat.send("Go")

        assert request_messages(fake_client, 1)[-1] == {
            "role": "user",
            "content": "The human researcher added a note while you were working:\n\nUse the other file.",
        }
        assert [message["role"] for message in chat.messages] == ["user", "assistant", "tool", "user", "assistant"]
        (note,) = of(chat.events_since(), "note")
        assert (note.speaker, note.text, note.data) == (
            HUMAN_SPEAKER,
            "Use the other file.",
            {"to": "lead", "read": True},
        )

    def test_a_note_is_read_by_the_team_when_it_is_working_and_the_lead_is_told_so(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            team_meeting(rounds=1),
            text_response("Opening."),
            text_response("A view."),
            text_response("Summary."),
            text_response("Done."),
        ]
        chat: Chat
        added: list[bool] = []

        def on_event(event: ChatEvent) -> None:
            if event.kind == "lab" and event.data["event"] == "turn" and not added:
                added.append(chat.add_note("Mind the glycosylation."))

        chat = make_chat(tmp_path, on_event=on_event)

        chat.send("Go")

        assert added == [True]
        (note,) = of(chat.events_since(), "note")
        assert note.data == {"to": "lab", "read": True}
        transcript = json.loads((tmp_path / "chat" / "lab" / "meetings" / "lab_001.json").read_text())
        assert {"agent": HUMAN_SPEAKER, "message": "Mind the glycosylation."} in transcript
        assert of(chat.events_since(), "tool_output")[0].text.endswith(
            "While the team worked, the researcher told it:\n\nMind the glycosylation."
        )
        # The lead was not given it twice
        assert [m["role"] for m in chat.messages if "glycosylation" in str(m.get("content"))] == ["tool"]

    def test_a_note_that_comes_too_late_is_told_of_as_not_read(self, fake_client: FakeClient, tmp_path: Path) -> None:
        chat: Chat
        added: list[bool] = []

        def on_event(event: ChatEvent) -> None:
            if event.kind == "usage":
                added.append(chat.add_note("Also KP.3."))

        chat = make_chat(tmp_path, on_event=on_event)

        chat.send("Go")

        assert added == [True]
        (note,) = of(chat.events_since(), "note")
        assert (note.text, note.data) == ("Also KP.3.", {"to": None, "read": False})
        assert kinds(chat.events_since())[-2:] == ["note", "answer"]
        assert len(chat.messages) == 2

    def test_notes_are_only_taken_while_the_lead_is_working(self, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)

        assert chat.add_note("Hello?") is False
        assert (chat.pause(), chat.resume(), chat.stop()) == (False, False, False)

    def test_a_paused_turn_waits_before_its_next_request_until_it_is_resumed(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Hello.")]
        states: list[str] = []
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "status":
                states.append(event.data["state"])
            if event.kind == "user":
                assert chat.pause()

        chat = make_chat(tmp_path, on_event=on_event)

        thread = chat.start("Hi")

        last = 0
        while chat.state != "paused":
            events = chat.wait_for_events(last, timeout=5)
            assert events, "The turn never reached the pause"
            last = events[-1].id
        assert fake_client.completions.calls == []
        assert thread.is_alive()

        # One message at a time
        with pytest.raises(ChatBusyError):
            chat.send("Another")
        with pytest.raises(ChatBusyError):
            chat.start("Another")

        assert chat.resume()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert chat.last_reply is not None and chat.last_reply.text == "Hello."
        assert states == ["running", "pausing", "paused", "running", "idle"]

    def test_a_paused_turn_can_be_stopped(self, fake_client: FakeClient, tmp_path: Path) -> None:
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "user":
                chat.pause()

        chat = make_chat(tmp_path, on_event=on_event)
        thread = chat.start("Hi")
        while chat.state != "paused":
            chat.wait_for_events(0, timeout=0.01)

        assert chat.stop()
        thread.join(timeout=10)

        assert chat.last_reply is not None and chat.last_reply.status == "stopped"
        assert fake_client.completions.calls == []
        assert chat.state == "idle"

    def test_a_paused_team_session_waits_before_its_next_turn(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [team_meeting(rounds=0), text_response("Alone."), text_response("Done.")]
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "tool_calls":
                chat.pause()

        chat = make_chat(tmp_path, on_event=on_event)
        thread = chat.start("Go")
        while chat.state != "paused":
            chat.wait_for_events(0, timeout=0.01)

        # The meeting has been started, but not its first request
        assert len(fake_client.completions.calls) == 1
        assert lab_events(chat.events_since(), "turn") == []

        chat.resume()
        thread.join(timeout=10)

        assert chat.last_reply is not None and chat.last_reply.text == "Done."

    def test_a_page_waits_for_what_happens_next(self, fake_client: FakeClient, tmp_path: Path) -> None:
        chat = make_chat(tmp_path)

        assert chat.wait_for_events(0, timeout=0.01) == []

        thread = chat.start("Hi")
        thread.join(timeout=10)
        events = chat.wait_for_events(0, timeout=5)

        assert kinds(events, leave_out=()) == ["user", "title", "usage", "answer", "status"]
        later = chat.wait_for_events(events[-1].id, timeout=0.01)
        assert later == []

    def test_closing_the_conversation_stops_a_turn_and_wakes_those_waiting(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        chat: Chat

        def on_event(event: ChatEvent) -> None:
            if event.kind == "user":
                chat.pause()

        chat = make_chat(tmp_path, on_event=on_event)
        chat.start("Hi")
        while chat.state != "paused":
            chat.wait_for_events(0, timeout=0.01)
        woken: list[Any] = []
        waiting = threading.Thread(target=lambda: woken.append(chat.wait_for_events(10**6, timeout=10)))
        waiting.start()

        chat.close(timeout=10)
        waiting.join(timeout=10)

        assert woken == [[]]
        assert chat.last_reply is not None and chat.last_reply.status == "stopped"


class TestResources:
    def test_resources_given_are_listed_to_the_lead(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path, session=session, resources=small_resources())

        chat.send("Hello")

        system = request_messages(fake_client, 0)[0]["content"]
        assert "These are the resources of Biomni's environment available in the session." in system
        assert "Method: find_gene" in system and "a.parquet: Table A." in system
        assert "find_resources" not in system
        (told,) = of(chat.events_since(), "resources")
        assert told.data["mode"] == "given"
        assert told.data["available"] == {"tools": 3, "data_lake": 2, "libraries": 2, "know_how": 1}
        assert told.data["selected"]["data_lake"] == ["a.parquet", "b.csv"]

    def test_the_lead_asks_which_resources_suit_a_task(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(chat_module, "available_resources", lambda session, commercial_mode: small_resources())
        fake_client.completions.responses = [
            tool_call_response("find_resources", {"task": "Find a gene linked to fever."}),
            text_response("TOOLS: [0]\nDATA_LAKE: []\nLIBRARIES: [1]\nKNOW_HOW: []"),
            text_response("Use find_gene."),
        ]
        chat = make_chat(tmp_path, session=session, resources="retrieve")

        reply = chat.send("Which gene?")

        assert reply.text == "Use find_gene."
        first = fake_client.completions.calls[0]
        assert "find_resources" in [tool["function"]["name"] for tool in first["tools"]]
        assert (
            "3 tool functions, 2 data lake files, 2 software libraries, 1 know-how documents"
            in (first["messages"][0]["content"])
        )
        assert "Method: find_gene" not in first["messages"][0]["content"]

        # The question of which resources suit it is asked on its own, as a meeting asks it
        retrieval = fake_client.completions.calls[1]
        assert len(retrieval["messages"]) == 1 and "tools" not in retrieval
        assert "USER QUERY: Find a gene linked to fever." in retrieval["messages"][0]["content"]

        events = chat.events_since()
        initial, chosen = of(events, "resources")
        assert initial.data["selected"] == {"tools": [], "data_lake": [], "libraries": [], "know_how": []}
        assert chosen.data["selected"]["tools"] == ["biomni.tool.genetics.find_gene"]
        assert chosen.data["selected"]["libraries"] == ["samtools"]
        assert chosen.data["retrieval"]["task"] == "Find a gene linked to fever."
        (output,) = of(events, "tool_output")
        assert "Method: find_gene" in output.text and "samtools: Alignments." in output.text
        assert "blast" not in output.text and "scanpy" not in output.text
        assert chat.usage.num_calls == 3

    def test_an_answer_that_names_nothing_is_not_a_choice(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(chat_module, "available_resources", lambda session, commercial_mode: small_resources())
        fake_client.completions.responses = [
            tool_call_response("find_resources", {"task": "Anything"}),
            text_response("I do not know."),
            text_response("Done."),
        ]
        chat = make_chat(tmp_path, session=session, resources="retrieve")

        chat.send("Go")

        assert "could not be worked out" in of(chat.events_since(), "tool_output")[0].text

    def test_nothing_is_listed_with_none_and_there_is_no_tool_to_ask(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path, session=session)

        chat.send("Hello")

        assert [tool["function"]["name"] for tool in fake_client.completions.calls[0]["tools"]] == [
            "run_code",
            "convene_team",
            "consult",
        ]
        assert of(chat.events_since(), "resources") == []

    def test_the_resources_are_the_teams_too(
        self, fake_client: FakeClient, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [team_meeting(rounds=0), text_response("Alone."), text_response("Done.")]
        chat = make_chat(tmp_path, session=session, resources=small_resources())

        chat.send("Go")

        # In what the team is first told of, not only what the lead is
        assert "Method: find_gene" in request_messages(fake_client, 1)[1]["content"]

    def test_resources_need_a_session(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs a session"):
            make_chat(tmp_path, resources=small_resources())

        with pytest.raises(ValueError, match="resources must be"):
            make_chat(tmp_path, resources="some")


class TestKeeping:
    def test_a_conversation_is_carried_on_from_where_it_was_left(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [team_meeting(rounds=0), text_response("Alone."), text_response("First.")]
        chat = make_chat(tmp_path)
        chat.send("One")
        before = chat.events_since()
        chat.close()

        fake_client.completions.responses = [
            team_meeting(rounds=0),
            text_response("Alone again."),
            text_response("Second."),
        ]
        again = make_chat(tmp_path)

        assert (again.title, again.turns, again.state) == ("One", 1, "idle")
        # Everything but the state it was in, which is not kept
        assert before[-1].kind == "status"
        assert again.events_since() == before[:-1]
        assert again.messages == chat.messages
        assert again.usage.num_calls == 3

        seen: list[ChatEvent] = []
        again.on_event = seen.append
        reply = again.send("Two")

        assert reply.text == "Second."
        # The lead reads what was said before, in order, with the new message last
        sent = request_messages(fake_client, 3)
        assert [message["role"] for message in sent] == ["system", "user", "assistant", "tool", "assistant", "user"]
        assert sent[-1]["content"] == "Two"
        events = again.events_since()
        assert [event.turn for event in events if event.kind == "user"] == [1, 2]
        assert len({event.id for event in events}) == len(events)
        assert [event.id for event in events] == sorted(event.id for event in events)
        # Numbered after the state the conversation was left in, which a page may have been told of
        assert (seen[0].kind, seen[0].id > before[-1].id) == ("status", True)
        # What it held is not held under the same name again
        assert [step.name for step in again.project.steps] == ["lab_001", "lab_002"]
        assert again.usage.num_calls == 6

    def test_a_turn_that_a_process_did_not_end_is_ended_when_the_conversation_is_opened(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        chat = make_chat(tmp_path)
        chat.send("One")
        directory = tmp_path / "chat"
        info = json.loads((directory / "chat.json").read_text())
        (directory / "chat.json").write_text(json.dumps({**info, "running": True, "turns": 2}))
        with (directory / "messages.jsonl").open("a") as file:
            file.write(json.dumps({"role": "user", "content": "Two"}) + "\n")
            file.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "name": "Principal_Investigator",
                        "content": None,
                        "tool_calls": [
                            {"id": "a", "type": "function", "function": {"name": "run_code", "arguments": "{}"}},
                            {"id": "b", "type": "function", "function": {"name": "run_code", "arguments": "{}"}},
                        ],
                    }
                )
                + "\n"
            )
            file.write(json.dumps({"role": "tool", "tool_call_id": "a", "content": "Done."}) + "\n")

        again = make_chat(tmp_path)

        reason = "The process running this conversation stopped before the turn ended."
        assert again.messages[-1] == {"role": "tool", "tool_call_id": "b", "content": f"Not run. {reason}"}
        last = again.events_since()[-1]
        assert (last.kind, last.text, last.data) == ("failed", reason, {"type": "Interrupted"})
        assert json.loads((directory / "chat.json").read_text())["running"] is False

        # It can be carried on, and is not ended again
        assert again.send("Three").status == "answered"
        assert kinds(make_chat(tmp_path).events_since()).count("failed") == 1

    def test_a_line_cut_short_by_a_crash_is_dropped_and_any_other_is_an_error(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        make_chat(tmp_path).send("One")
        directory = tmp_path / "chat"
        with (directory / "events.jsonl").open("a") as file:
            file.write('{"kind": "usage", "id"')

        again = make_chat(tmp_path)

        assert "cut short" in capsys.readouterr().out
        assert kinds(again.events_since()) == ["user", "title", "usage", "answer"]
        assert lines(directory / "events.jsonl")[-1]["kind"] == "answer"

        text = (directory / "messages.jsonl").read_text()
        (directory / "messages.jsonl").write_text("not json\n" + text)
        with pytest.raises(ValueError, match="Line 1 of .*messages.jsonl is not JSON"):
            make_chat(tmp_path)

    def test_the_project_the_conversation_keeps_its_meetings_in_is_not_another(self, tmp_path: Path) -> None:
        make_chat(tmp_path)

        assert json.loads((tmp_path / "chat" / "lab" / "project.json").read_text())["goal"] == (
            "Conversation with the head of a lab"
        )


class TestFailures:
    def test_a_failed_turn_is_told_of_and_the_conversation_goes_on(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [RuntimeError("The server went away"), text_response("Back.")]
        chat = make_chat(tmp_path)

        reply = chat.send("Hello")

        assert reply == ChatReply("failed", "", "RuntimeError: The server went away", reply.usage, reply.elapsed_time)
        events = chat.events_since()
        assert kinds(events) == ["user", "title", "failed"]
        (failed,) = of(events, "failed")
        assert (failed.text, failed.data["type"]) == ("RuntimeError: The server went away", "RuntimeError")
        assert chat.state == "idle"

        assert chat.send("Hello again").text == "Back."

    def test_a_reply_with_nothing_in_it_is_a_failure(self, fake_client: FakeClient, tmp_path: Path) -> None:
        fake_client.completions.responses = [text_response("   ")]
        chat = make_chat(tmp_path)

        reply = chat.send("Hello")

        assert reply.error == "RuntimeError: Principal Investigator sent no answer"

    def test_a_conversation_that_has_outgrown_the_model_says_so(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def too_long(**kwargs: Any) -> None:
            raise ContextLengthExceededError("Reduce num_rounds")

        monkeypatch.setattr(chat_module, "check_context_length", too_long)
        chat = make_chat(tmp_path)

        reply = chat.send("Hello")

        assert reply.error is not None and "Start a new conversation" in reply.error and "num_rounds" not in reply.error
        assert fake_client.completions.calls == []

    def test_a_session_that_will_not_start_fails_the_turn_before_anything_is_spent(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        class Broken(LocalSession):
            def start(self) -> None:
                raise RuntimeError("The image was never built")

        chat = make_chat(tmp_path, session=Broken(tmp_path / "work", warn=False))

        reply = chat.send("Hello")

        assert reply.error == "RuntimeError: The image was never built"
        assert fake_client.completions.calls == []
        assert chat.messages == []

    def test_an_exception_from_on_event_ends_the_turn(self, fake_client: FakeClient, tmp_path: Path) -> None:
        def on_event(event: ChatEvent) -> None:
            if event.kind == "usage":
                raise ValueError("The page closed")

        chat = make_chat(tmp_path, on_event=on_event)

        reply = chat.send("Hello")

        assert reply.error == "ValueError: The page closed"

    def test_an_exception_when_told_of_the_end_does_not_hide_how_it_ended(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def on_event(event: ChatEvent) -> None:
            if event.kind == "answer":
                raise ValueError("The page closed")

        chat = make_chat(tmp_path, on_event=on_event)

        reply = chat.send("Hello")

        assert reply.status == "answered"
        assert (
            "on_event failed when told of the end of the turn: ValueError('The page closed')" in capsys.readouterr().out
        )
        assert chat.state == "idle"

    def test_an_interruption_ends_the_turn_properly_and_is_raised(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [KeyboardInterrupt()]
        chat = make_chat(tmp_path)

        with pytest.raises(KeyboardInterrupt):
            chat.send("Hello")

        assert chat.state == "idle" and kinds(chat.events_since())[-1] == "failed"
        assert json.loads((tmp_path / "chat" / "chat.json").read_text())["running"] is False

    def test_a_turn_that_cannot_start_a_thread_gives_the_conversation_back(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(self: threading.Thread) -> None:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", refuse)
        chat = make_chat(tmp_path)

        with pytest.raises(RuntimeError, match="can't start new thread"):
            chat.start("Hello")

        assert chat.state == "idle"

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"temperature": 3.0}, "temperature must be between 0 and 2"),
            ({"max_completion_tokens": 0}, "max_completion_tokens must be at least 1"),
            ({"max_tool_iterations": 0}, "max_tool_iterations must be at least 1"),
            ({"max_delegations": -1}, "max_delegations must be zero or more"),
            ({"max_rounds": -1}, "max_rounds must be zero or more"),
            ({"max_upload_bytes": 0}, "max_upload_bytes must be at least 1"),
            ({"tools": ("run_code",)}, "A conversation's tools are Tools, not str"),
        ],
    )
    def test_arguments_that_cannot_be_used_are_refused(
        self, tmp_path: Path, options: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError if "Tools" not in message else TypeError, match=message):
            make_chat(tmp_path, **options)

    def test_titles_must_be_their_own(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must have different titles"):
            Chat(tmp_path / "chat", LEAD, (LEAD,))
        with pytest.raises(ValueError, match="must have different titles"):
            Chat(tmp_path / "chat", LEAD, (IMMUNOLOGIST, IMMUNOLOGIST))
        with pytest.raises(ValueError, match="needs a title"):
            Chat(tmp_path / "chat", LEAD, (Agent(" ", "x", "y", "z", TEST_MODEL),))

    def test_the_tools_that_are_the_conversations_own_cannot_be_taken(self, tmp_path: Path) -> None:
        from virtual_lab.tools import Tool

        tool = Tool(
            name="consult", description="x", parameters={"type": "object", "properties": {}}, function=lambda: ""
        )

        with pytest.raises(ValueError, match="has tools of its own named consult"):
            make_chat(tmp_path, tools=(tool,))
