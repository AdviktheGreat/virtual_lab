"""Tests for code as an agent's action: reading it from replies, and running it in meetings."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from virtual_lab.actions import (
    describe_code_run,
    describe_tool_output,
    find_code_action,
    observation,
)
from virtual_lab.agent import Agent
from virtual_lab.constants import (
    MAX_TOOL_ITERATIONS,
    METADATA_DIR_NAME,
    PARTIAL_MEETING_DIR_NAME,
    SANDBOX_DATA_LAKE_DIR,
    SESSION_LOG_DIR_NAME,
    SESSION_MAX_TOOL_ITERATIONS,
)
from virtual_lab.prompts import code_session_prompt
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.session import CellResult, LocalSession, Session, SessionError

from conftest import FakeClient, text_response, tool_call_response


def cell(status: str = "ok", output: str = "42\n", error: str | None = None) -> CellResult:
    return CellResult(
        language="python", code="print(42)", status=status, output=output, error=error, duration=0.5
    )


class TestFindCodeAction:
    def test_reply_without_a_block_has_no_action(self) -> None:
        assert find_code_action("I think the answer is 42.") is None

    def test_python_block_is_found(self) -> None:
        action = find_code_action("Let me check.\n<execute>\nx = 1\nprint(x)\n</execute>")

        assert action is not None
        assert action.language == "python"
        assert action.code == "x = 1\nprint(x)"
        assert action.said == "Let me check.\n<execute>\nx = 1\nprint(x)\n</execute>"

    def test_text_after_the_block_is_dropped(self) -> None:
        action = find_code_action("<execute>print(1)</execute>\n<observation>1</observation> So 1.")

        assert action is not None
        assert action.said == "<execute>print(1)</execute>"

    def test_only_the_first_block_runs(self) -> None:
        action = find_code_action("<execute>a = 1</execute> then <execute>b = 2</execute>")

        assert action is not None
        assert action.code == "a = 1"

    def test_missing_close_runs_to_the_end(self) -> None:
        action = find_code_action("Here:\n<execute>\nprint('cut off')")

        assert action is not None
        assert action.code == "print('cut off')"
        assert action.said.endswith("</execute>")

    @pytest.mark.parametrize(
        ("marker", "language"),
        [
            ("#!R", "r"),
            ("# R code", "r"),
            ("#!BASH", "bash"),
            ("#!CLI", "bash"),
            ("# Bash script", "bash"),
            ("#!bash", "bash"),
        ],
    )
    def test_language_markers(self, marker: str, language: str) -> None:
        action = find_code_action(f"<execute>\n{marker}\necho hi\n</execute>")

        assert action is not None
        assert action.language == language
        assert action.code == "echo hi"

    def test_a_comment_is_not_a_marker(self) -> None:
        action = find_code_action("<execute>\n# Run the model\nprint(1)\n</execute>")

        assert action is not None
        assert action.language == "python"
        assert action.code == "# Run the model\nprint(1)"

    def test_fence_inside_the_block_is_removed(self) -> None:
        action = find_code_action("<execute>\n```python\nprint(1)\n```\n</execute>")

        assert action is not None
        assert action.code == "print(1)"

    def test_marker_inside_a_fence_is_read(self) -> None:
        action = find_code_action("<execute>\n```\n#!R\nprint(1)\n```\n</execute>")

        assert action is not None
        assert action.language == "r"
        assert action.code == "print(1)"


class TestObservation:
    def test_wraps_the_report(self) -> None:
        text = observation(cell(), runs_left=3)

        assert text.startswith("<observation>\n")
        assert text.endswith("</observation>")
        assert "42" in text

    def test_last_run_says_so(self) -> None:
        assert "last code you can run" in observation(cell(), runs_left=0)
        assert "last code you can run" not in observation(cell(), runs_left=1)


class TestDescribe:
    def test_code_run_shows_code_then_report(self) -> None:
        text = describe_code_run("r", "print(1)", "Ran.")

        assert text == "```r\nprint(1)\n```\n\nRan."

    def test_code_containing_a_fence_gets_a_longer_one(self) -> None:
        text = describe_code_run("python", "s = '```'", "Ran.")

        assert text.startswith("````python\n")

    def test_tool_output_of_run_code_shows_the_code(self) -> None:
        call = SimpleNamespace(
            function=SimpleNamespace(name="run_code", arguments='{"code": "1 + 1", "language": "python"}')
        )

        assert describe_tool_output(call, "2", "run_code") == "```python\n1 + 1\n```\n\n2"

    @pytest.mark.parametrize(
        "call",
        [
            SimpleNamespace(function=SimpleNamespace(name="pubmed_search", arguments='{"code": "x"}')),
            SimpleNamespace(function=SimpleNamespace(name="run_code", arguments="not json")),
            SimpleNamespace(function=SimpleNamespace(name="run_code", arguments='{"code": 5}')),
            SimpleNamespace(function=SimpleNamespace(name="run_code", arguments="[1]")),
            SimpleNamespace(),
        ],
    )
    def test_other_output_is_left_alone(self, call) -> None:  # type: ignore[no-untyped-def]
        assert describe_tool_output(call, "output", "run_code") == "output"


class TestCodeSessionPrompt:
    def test_tool_mode(self) -> None:
        text = code_session_prompt("tool", "/workspace", network=True, data_lake=None)

        assert "run_code" in text
        assert "<execute>" not in text
        assert "/workspace" in text
        assert "can reach the internet" in text
        assert "data lake" not in text

    def test_tags_mode_without_network_with_data_lake(self) -> None:
        text = code_session_prompt("tags", "/workspace", network=False, data_lake="/data_lake")

        assert "<execute>" in text and "#!R" in text and "#!BASH" in text
        assert "cannot reach the internet" in text
        assert "/data_lake" in text


class TestSessionEnvironment:
    def test_local_session_reaches_the_network_without_a_lake(self, tmp_path: Path) -> None:
        opened = LocalSession(tmp_path, warn=False)

        assert opened.can_reach_network() is True
        assert opened.data_lake_path() is None

    def test_docker_session_reports_its_executor(self, tmp_path: Path) -> None:
        from virtual_lab.execution import DockerExecutor
        from virtual_lab.session import DockerSession

        lake = tmp_path / "lake"
        lake.mkdir()
        closed = DockerSession(tmp_path / "work", executor=DockerExecutor(allow_network=False))
        mounted = DockerSession(tmp_path / "work", executor=DockerExecutor(data_lake=lake))

        assert closed.can_reach_network() is False
        assert closed.data_lake_path() is None
        assert mounted.data_lake_path() == SANDBOX_DATA_LAKE_DIR


@pytest.fixture
def session(tmp_path: Path):
    with LocalSession(tmp_path / "work", warn=False, timeout=20) as opened:
        yield opened


def individual(team_member: Agent, save_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
    return hold_meeting(
        meeting_type="individual",
        agenda="Compute a number.",
        save_dir=save_dir,
        team_member=team_member,
        # Resources are tested in test_resources; here they would add a request to every meeting
        **{"resources": "none", **kwargs},
    )


def saved_record(save_dir: Path) -> dict:
    return json.loads((save_dir / METADATA_DIR_NAME / "discussion.json").read_text())


def sent_messages(fake_client: FakeClient, call: int) -> list[dict]:
    return fake_client.completions.calls[call]["messages"]


def sent_tools(fake_client: FakeClient, call: int) -> list[str]:
    return [tool["function"]["name"] for tool in fake_client.completions.calls[call].get("tools", [])]


class TestMeetingValidation:
    def test_tags_need_a_session(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs a session"):
            individual(team_member, tmp_path, code_actions="tags")

        assert fake_client.completions.calls == []

    def test_unknown_code_actions(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="code_actions"):
            individual(team_member, tmp_path, session=session, code_actions="xml")  # type: ignore[arg-type]

    def test_iterations_must_be_positive(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_tool_iterations"):
            individual(team_member, tmp_path, max_tool_iterations=0)

    def test_run_code_name_cannot_be_taken(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        from virtual_lab.session import session_tool

        with pytest.raises(ValueError, match="unique"):
            individual(team_member, tmp_path, session=session, tools=(session_tool(session),))

    def test_session_that_cannot_start_fails_before_any_request(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        broken = LocalSession(tmp_path / "work", python="false", warn=False)

        with pytest.raises(SessionError):
            individual(team_member, tmp_path, session=broken)

        assert fake_client.completions.calls == []
        assert not (tmp_path / PARTIAL_MEETING_DIR_NAME).exists()


class TestMeetingWithoutSession:
    def test_default_iterations_and_no_session_in_record(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        result = individual(team_member, tmp_path)

        assert result.record.max_tool_iterations == MAX_TOOL_ITERATIONS
        assert result.record.session is None
        assert saved_record(tmp_path)["session"] is None
        assert all(turn["code_runs"] == [] for turn in saved_record(tmp_path)["turns"])
        assert not (tmp_path / SESSION_LOG_DIR_NAME).exists()

    def test_tags_are_not_run_without_a_session(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("<execute>print(1)</execute>")]

        result = individual(team_member, tmp_path)

        assert len(fake_client.completions.calls) == 1
        assert result.summary == "<execute>print(1)</execute>"


class TestToolMode:
    def test_agent_runs_code_through_the_tool(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "answer = 6 * 7\nanswer"}),
            text_response("The answer is 42."),
        ]

        result = individual(team_member, tmp_path, session=session)

        assert result.summary == "The answer is 42."
        assert "run_code" in sent_tools(fake_client, 0)
        assert result.record.tools == ["run_code"]
        assert result.record.max_tool_iterations == SESSION_MAX_TOOL_ITERATIONS

        # The start prompt tells the agent about the session
        first_prompt = sent_messages(fake_client, 0)[-1]["content"]
        assert "running Python session" in first_prompt
        assert "call the run_code tool" in first_prompt

        # The model gets the output; the transcript shows the code with it
        tool_message = sent_messages(fake_client, 1)[-1]
        assert tool_message["role"] == "tool"
        assert "42" in tool_message["content"]
        assert "answer = 6 * 7" not in tool_message["content"]
        tool_turn = next(turn for turn in result.discussion if turn["agent"] == "Tool")
        assert tool_turn["message"].startswith("```python\nanswer = 6 * 7\nanswer\n```")
        assert "42" in tool_turn["message"]

        # The variable is still there for whoever runs code next
        assert session.run("answer").output.strip() == "42"

    def test_record_and_log(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        session.run("earlier = 1")
        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "print('first')"}),
            tool_call_response("run_code", {"code": "raise ValueError('bad')"}),
            text_response("Done."),
        ]

        individual(team_member, tmp_path, session=session)

        record = saved_record(tmp_path)
        assert record["session"]["type"] == "LocalSession"
        assert record["session"]["code_actions"] == "tool"
        assert record["session"]["cells_run"] == 2
        assert record["session"]["log"] == f"{SESSION_LOG_DIR_NAME}/discussion.json"

        response = next(turn for turn in record["turns"] if turn["kind"] == "response")
        assert [(run["cell"], run["status"]) for run in response["code_runs"]] == [(0, "ok"), (1, "error")]

        # Only this meeting's code is in its log
        log = json.loads((tmp_path / record["session"]["log"]).read_text())
        assert [entry["code"] for entry in log["cells"]] == ["print('first')", "raise ValueError('bad')"]
        assert log["cells"][0]["output"] == "first\n"
        assert log["session"]["cells"] == 3

    def test_code_runs_are_split_by_turn(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "x = 1"}),
            text_response("Set x."),
            text_response("Critique."),
            tool_call_response("run_code", {"code": "x + 1"}),
            text_response("x + 1 is 2."),
        ]

        individual(team_member, tmp_path, session=session, num_rounds=1)

        responses = [turn for turn in saved_record(tmp_path)["turns"] if turn["kind"] == "response"]
        assert [[run["cell"] for run in turn["code_runs"]] for turn in responses] == [[0], [], [1]]


class TestTagsMode:
    def test_agent_runs_code_written_in_its_reply(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("Let me compute.\n<execute>\nvalue = 6 * 7\nprint(value)\n</execute>\n<observation>99"),
            text_response("<execute>\n#!BASH\necho from-bash\n</execute>"),
            text_response("The value is 42."),
        ]

        result = individual(team_member, tmp_path, session=session, code_actions="tags")

        assert result.summary == "The value is 42."
        assert "run_code" not in sent_tools(fake_client, 0)
        assert result.record.tools == []
        assert "<execute>" in sent_messages(fake_client, 0)[-1]["content"]

        # The agent's own reply is cut after the block, and the observation follows it
        second = sent_messages(fake_client, 1)
        assert second[-2]["role"] == "assistant"
        assert second[-2]["content"].endswith("</execute>")
        assert "99" not in second[-2]["content"]
        assert second[-1]["role"] == "user"
        assert second[-1]["content"].startswith("<observation>")
        assert "42" in second[-1]["content"]

        third = sent_messages(fake_client, 2)
        assert "from-bash" in third[-1]["content"]

        agents = [turn["agent"] for turn in result.discussion]
        assert agents[-5:] == [team_member.title, "Session", team_member.title, "Session", team_member.title]

        record = saved_record(tmp_path)
        kinds = [turn["kind"] for turn in record["turns"]]
        assert kinds[-5:] == ["code_action", "code_output", "code_action", "code_output", "response"]
        assert len(record["turns"]) == len(result.discussion)
        response = record["turns"][-1]
        assert [run["language"] for run in response["code_runs"]] == ["python", "bash"]
        assert response["num_api_calls"] == 3
        assert record["session"]["code_actions"] == "tags"

    def test_last_run_is_announced_and_further_code_is_not_run(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("<execute>print('one')</execute>"),
            text_response("<execute>print('two')</execute>"),
            text_response("One last look.\n<execute>print('three')</execute>\n<observation>3</observation>"),
        ]

        result = individual(
            team_member, tmp_path, session=session, code_actions="tags", max_tool_iterations=2
        )

        assert len(fake_client.completions.calls) == 3
        assert "last code you can run" in sent_messages(fake_client, 2)[-1]["content"]
        assert "last code you can run" not in sent_messages(fake_client, 1)[-1]["content"]
        assert [entry.code for entry in session.history] == ["print('one')", "print('two')"]
        # Code too late to run is not left in the answer to look as though it ran
        assert result.summary.startswith("One last look.\n\n(The code this reply went on to write was not run")
        assert "print('three')" not in result.summary
        assert "<observation>" not in result.summary

    def test_tool_calls_still_work_alongside_tags(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        from virtual_lab.tools import Tool

        echo = Tool(
            name="echo",
            description="Echoes.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=lambda text: f"echo: {text}",
        )
        fake_client.completions.responses = [
            tool_call_response("echo", {"text": "hi"}),
            text_response("<execute>print('ran')</execute>"),
            text_response("Both worked."),
        ]

        result = individual(team_member, tmp_path, session=session, code_actions="tags", tools=(echo,))

        assert result.summary == "Both worked."
        assert sent_tools(fake_client, 0) == ["echo"]
        # Nothing was skipped, so the agent was not told anything was
        assert not any("was not run" in str(message["content"]) for message in sent_messages(fake_client, 2))
        assert [entry.code for entry in session.history] == ["print('ran')"]

    def test_code_beside_a_tool_call_is_not_run_and_the_agent_is_told(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        from virtual_lab.tools import Tool

        echo = Tool(
            name="echo",
            description="Echoes.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=lambda text: f"echo: {text}",
        )
        both = tool_call_response("echo", {"text": "hi"})
        both.choices[0].message.content = "<execute>print('skipped')</execute>"
        fake_client.completions.responses = [both, text_response("Understood.")]

        result = individual(team_member, tmp_path, session=session, code_actions="tags", tools=(echo,))

        assert session.history == []
        assert "was not run, because the reply also called a tool" in sent_messages(fake_client, 1)[-1]["content"]
        record = saved_record(tmp_path)
        assert len(record["turns"]) == len(result.discussion)

    @pytest.mark.parametrize("failure", [SessionError("This session was closed."), OSError("docker vanished")])
    def test_session_error_is_told_to_the_agent(
        self,
        fake_client: FakeClient,
        team_member: Agent,
        session: Session,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise failure

        fake_client.completions.responses = [
            text_response("<execute>print(1)</execute>"),
            text_response("Could not run it."),
        ]
        monkeypatch.setattr(session, "run", refuse)

        result = individual(team_member, tmp_path, session=session, code_actions="tags")

        assert result.summary == "Could not run it."
        assert f"could not be run: {failure}" in sent_messages(fake_client, 1)[-1]["content"]

    def test_team_members_share_the_session(
        self,
        fake_client: FakeClient,
        team_lead: Agent,
        team_member: Agent,
        session: Session,
        tmp_path: Path,
    ) -> None:
        fake_client.completions.responses = [
            text_response("<execute>shared = [1, 2, 3]</execute>"),
            text_response("I loaded the data."),
            text_response("<execute>print(sum(shared))</execute>"),
            text_response("The sum is 6."),
            text_response("Summary: 6."),
        ]

        result = hold_meeting(
            meeting_type="team",
            agenda="Add numbers.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=(team_member,),
            num_rounds=1,
            session=session,
            code_actions="tags",
            resources="none",
        )

        assert result.summary == "Summary: 6."
        assert session.history[-1].output.strip() == "6"
        # The team's opening prompt describes the session
        assert "<execute>" in sent_messages(fake_client, 0)[1]["content"]
        # The member sees the lead's code and what it printed
        member_messages = sent_messages(fake_client, 2)
        assert any("shared = [1, 2, 3]" in str(message.get("content")) for message in member_messages)

    def test_failed_meeting_saves_its_session_log(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("<execute>print('before the failure')</execute>"),
            RuntimeError("the API went away"),
        ]

        with pytest.raises(RuntimeError, match="went away"):
            individual(team_member, tmp_path, session=session, code_actions="tags", max_retries=0)

        partial = tmp_path / PARTIAL_MEETING_DIR_NAME
        record = json.loads((partial / METADATA_DIR_NAME / "discussion.json").read_text())
        assert record["session"]["cells_run"] == 1
        log = json.loads((partial / record["session"]["log"]).read_text())
        assert log["cells"][0]["output"] == "before the failure\n"
        assert not (tmp_path / SESSION_LOG_DIR_NAME).exists()

    @pytest.mark.parametrize("fails", [False, True])
    def test_log_that_cannot_be_saved_keeps_the_transcript(
        self,
        fake_client: FakeClient,
        team_member: Agent,
        session: Session,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fails: bool,
    ) -> None:
        from importlib import import_module

        run_meeting_module = import_module("virtual_lab.run_meeting")

        def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("disk full")

        monkeypatch.setattr(run_meeting_module, "save_session_log", broken)
        fake_client.completions.responses = [text_response("<execute>print(1)</execute>")]
        fake_client.completions.responses.append(RuntimeError("the API went away") if fails else text_response("Done."))

        if fails:
            # The error that ended the meeting is the one raised, not the log's
            with pytest.raises(RuntimeError, match="went away"):
                individual(team_member, tmp_path, session=session, code_actions="tags", max_retries=0)
            saved = tmp_path / PARTIAL_MEETING_DIR_NAME
        else:
            individual(team_member, tmp_path, session=session, code_actions="tags")
            saved = tmp_path

        assert (saved / "discussion.json").exists()
        assert json.loads((saved / METADATA_DIR_NAME / "discussion.json").read_text())["session"] is None


def gene_length(gene: str) -> int:
    """How long a gene's protein is, in residues.

    :param gene: A gene symbol, such as TP53.
    """
    return {"TP53": 393}[gene]


class TestSessionTools:
    def test_the_agents_are_told_of_the_sessions_tools_and_their_code_calls_them(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        from virtual_lab.custom_tools import tool_from_function

        fake_client.completions.responses = [
            tool_call_response("run_code", {"code": "gene_length('TP53')"}),
            text_response("It is 393 residues long."),
        ]

        with LocalSession(tmp_path / "work", warn=False, tools=(tool_from_function(gene_length),)) as opened:
            individual(team_member, tmp_path, session=opened)

        first_prompt = sent_messages(fake_client, 0)[-1]["content"]
        assert "Functions added for this work, already defined in the session" in first_prompt
        assert "gene_length(gene)\n  How long a gene's protein is, in residues." in first_prompt
        # Called from code, not offered to the model as a tool of its own
        assert sent_tools(fake_client, 0) == ["run_code"]
        assert "393" in sent_messages(fake_client, 1)[-1]["content"]

        record = saved_record(tmp_path)
        assert record["session"]["tools"] == ["gene_length"]
        log = json.loads((tmp_path / record["session"]["log"]).read_text())
        assert log["cells"][0]["tool_calls"][0]["tool"] == "gene_length"

    def test_a_session_without_tools_says_nothing_of_them(
        self, fake_client: FakeClient, team_member: Agent, session: Session, tmp_path: Path
    ) -> None:
        individual(team_member, tmp_path, session=session)

        assert "Functions added for this work" not in sent_messages(fake_client, 0)[-1]["content"]

    def test_a_meeting_is_given_tools_not_functions(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="tool_from_function"):
            individual(team_member, tmp_path, tools=(gene_length,))

        assert fake_client.completions.calls == []
