"""Tests for saving meetings and projects as documents, in HTML and as PDFs."""

import base64
import builtins
import dataclasses
import http.server
import io
import json
import re
import sys
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import virtual_lab.export as export
from virtual_lab.agent import Agent
from virtual_lab.export import (
    MAX_EXPORTED_OUTPUT_CHARS,
    PDFExportError,
    image_uri,
    load_meeting,
    meeting_html,
    project_html,
    project_meetings,
    save_meeting_html,
    save_meeting_pdf,
    save_project_html,
    save_project_pdf,
    usage_table,
)
from virtual_lab.planning import run_project
from virtual_lab.project import Project
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.session import LocalSession
from virtual_lab.tools import Tool
from virtual_lab.utils import MeetingUsage, price_per_million

from conftest import TEST_MODEL, FakeClient, make_usage, parsed_response, text_response, tool_call_response
from test_events import CRITIC, GOAL, LEAD
from test_planning import decide, found, plan, review, roster

# The smallest PNG there is: one transparent pixel
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def weasyprint_works() -> bool:
    try:
        export.import_weasyprint()
    except PDFExportError:
        return False
    return True


needs_weasyprint = pytest.mark.skipif(not weasyprint_works(), reason="WeasyPrint or the Pango library is not installed")


class Verdict(BaseModel):
    decision: str


def team(team_lead: Agent, team_member: Agent, save_dir: Path, **options: Any) -> Any:
    return hold_meeting(
        meeting_type="team",
        agenda="Design a nanobody.",
        save_dir=save_dir,
        team_lead=team_lead,
        team_members=(team_member,),
        **options,
    )


def individual(team_member: Agent, save_dir: Path, **options: Any) -> Any:
    return hold_meeting(
        meeting_type="individual",
        agenda="Compute a number.",
        save_dir=save_dir,
        team_member=team_member,
        **{"resources": "none", **options},
    )


def text_of(document: str) -> str:
    """The text of a document, without its style or tags, with its spacing collapsed."""
    body = document.split("</style>", 1)[-1]
    return " ".join(re.sub(r"<[^>]+>", " ", body).split())


def section(document: str, heading: str, level: int = 2) -> str:
    """The text of a document from a heading to the next heading of its level."""
    start = document.index(f"<h{level}>{heading}</h{level}>")
    end = document.find(f"<h{level}>", start + 1)
    return text_of(document[start : end if end != -1 else None])


def in_order(text: str, *parts: str) -> bool:
    positions = [text.index(part) for part in parts]
    return positions == sorted(positions)


class DrawingSession(LocalSession):
    """A session whose code is taken to have drawn a figure, since matplotlib is not installed here."""

    def run(self, code: str, language: str = "python", timeout: float | None = None) -> Any:
        result = super().run(code, language=language, timeout=timeout)
        (self.directory / "plots").mkdir(exist_ok=True)
        (self.directory / "plots" / "figure_1.png").write_bytes(PNG)
        drawn = dataclasses.replace(result, plots=("plots/figure_1.png",))
        self.history[-1] = drawn
        return drawn


@pytest.fixture
def session(tmp_path: Path):
    with DrawingSession(tmp_path / "work", warn=False, timeout=20) as opened:
        yield opened


class TestMeetings:
    def test_a_team_meeting_is_laid_out_as_a_meeting(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("Let us begin with **binding**."),
            text_response("I suggest the KP.3 epitope."),
            text_response("We will design against KP.3. Summary done."),
        ]
        result = team(team_lead, team_member, tmp_path, num_rounds=1, save_name="kickoff_meeting")

        document = meeting_html(result.transcript_path)
        text = text_of(document)

        assert "<title>Kickoff meeting</title>" in document
        assert "Virtual Lab · Team meeting" in text
        assert '<span class="badge ok">completed</span>' in document
        assert "Agenda Design a nanobody." in text
        # The team, its lead marked, each in its own colour
        assert f"{team_lead.title} (team lead)" in text
        assert team_member.expertise in text
        assert f"--colour: {export.AGENT_COLOURS[0]}" in document and f"--colour: {export.AGENT_COLOURS[1]}" in document
        # The summary is the lead's last word, and comes before the discussion
        assert section(document, "Summary") == f"Summary {team_lead.title} We will design against KP.3. Summary done."
        assert in_order(document, "<h2>Summary</h2>", "<h2>Discussion</h2>")
        assert "<strong>binding</strong>" in document
        # The discussion round by round, the first round being everyone's and the last only the lead's
        assert in_order(
            section(document, "Discussion"),
            "Round 1 of 1",
            "Let us begin",
            "I suggest the KP.3 epitope",
            "Final round",
            "We will design against KP.3",
        )
        # The prompts are left out, the agenda standing for them
        assert "Prompt" not in text
        assert "This is the beginning of a team meeting" not in text

    def test_prompts_are_shown_when_asked_for(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        result = team(team_lead, team_member, tmp_path, num_rounds=1)
        transcript = json.loads(result.transcript_path.read_text())

        text = text_of(meeting_html(result.transcript_path, include_prompts=True))

        prompts = [turn for turn in transcript if turn["agent"] == "User"]
        assert text.count("Prompt") == len(prompts) == 4
        assert "team meeting" in text.lower()

    def test_what_the_meeting_cost_is_shown_per_model(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        result = team(team_lead, team_member, tmp_path, num_rounds=1)

        document = meeting_html(result.transcript_path)

        cost = export.money(result.usage.compute_cost())
        assert result.usage.compute_cost() > 0
        assert f"cost {cost}" in text_of(document)
        table = document[document.index("What it used") :]
        row = f"<td>{TEST_MODEL}</td><td class='number'>3</td><td class='number'>300</td><td class='number'>60</td>"
        assert row in table
        # The model's row and the total agree, priced as the meeting priced them
        assert table.count(f"<td class='number'>{cost}</td>") == 1
        assert f"<th class='number'>{cost}</th>" in table

    def test_an_individual_meeting_marks_its_critic_and_its_rounds(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("First answer."),
            text_response("A criticism."),
            text_response("Better answer."),
        ]
        result = individual(team_member, tmp_path, num_rounds=1)

        document = meeting_html(result.transcript_path)
        text = text_of(document)

        assert "Virtual Lab · Individual meeting" in text
        assert "Scientific Critic (critic)" in text
        assert "(team lead)" not in text
        discussion = section(document, "Discussion")
        assert in_order(discussion, "Round 1 of 1", "First answer.", "A criticism.", "Final round", "Better answer.")
        assert section(document, "Summary") == f"Summary {team_member.title} Better answer."

    def test_a_meeting_without_rounds_of_discussion_has_only_its_final_round(
        self, fake_client: FakeClient, team_lead: Agent, team_member: Agent, tmp_path: Path
    ) -> None:
        result = team(team_lead, team_member, tmp_path, num_rounds=0)

        text = text_of(meeting_html(result.transcript_path))

        assert "Final round" in text
        assert "Round 1 of 0" not in text

    def test_code_its_output_and_its_figures_are_shown_with_the_turn_that_ran_them(
        self, fake_client: FakeClient, team_member: Agent, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("I will compute it.\n<execute>\nprint(6 * 7)\n</execute>"),
            text_response("The answer is 42, as the figure shows."),
        ]
        result = individual(team_member, tmp_path, session=session, code_actions="tags", num_rounds=0)

        document = meeting_html(result.transcript_path)
        text = text_of(document)

        assert "<p>I will compute it.</p>" in document
        code = '<pre class="code"><code class="language-python">print(6 * 7)</code></pre>'
        assert f'<div class="label">Code (python)</div>{code}' in document
        assert "&lt;execute&gt;" not in document
        # What the code printed, without the tags the agent was shown it in
        printed = document[document.index("What the code printed") :]
        assert re.search(r'<pre class="output">[^<]*42', printed)
        assert "&lt;observation&gt;" not in document
        # The figure, carried in the document, before what the agent made of it
        uri = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
        assert f'<img src="{uri}" alt="Figure 1">' in document
        assert "Figure 1. Drawn by code Immunologist ran (plots/figure_1.png)." in text
        assert document.index(uri) < document.rindex("The answer is 42")

        saved = load_meeting(result.transcript_path)
        assert saved.name == "discussion"
        assert saved.record is not None and saved.record["session"]["directory"] == str(session.directory)
        assert saved.session_log is not None
        assert [cell["code"] for cell in saved.session_log["cells"]] == ["print(6 * 7)"]

    def test_code_an_agent_ran_does_not_move_the_turns_after_it_to_another_round(
        self, fake_client: FakeClient, team_member: Agent, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("<execute>\nprint(1)\n</execute>"),
            text_response("First answer."),
            text_response("A criticism."),
            text_response("Better answer."),
        ]
        result = individual(team_member, tmp_path, session=session, code_actions="tags", num_rounds=1)

        discussion = section(meeting_html(result.transcript_path), "Discussion")

        assert in_order(discussion, "Round 1 of 1", "First answer.", "A criticism.", "Final round", "Better answer.")

    def test_a_figure_that_is_gone_is_said_to_be_missing(
        self, fake_client: FakeClient, team_member: Agent, session: LocalSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response("<execute>\nprint(1)\n</execute>"),
            text_response("Done."),
        ]
        result = individual(team_member, tmp_path, session=session, code_actions="tags", num_rounds=0)
        (session.directory / "plots" / "figure_1.png").unlink()

        document = meeting_html(result.transcript_path)

        assert "<em>plots/figure_1.png could not be found.</em>" in document
        assert "<img" not in document

    def test_only_figures_in_the_session_directory_are_read(self, tmp_path: Path) -> None:
        directory = tmp_path / "work"
        (directory / "plots").mkdir(parents=True)
        (directory / "plots" / "figure.png").write_bytes(PNG)
        (tmp_path / "outside.png").write_bytes(PNG)
        (directory / "notes.txt").write_text("not a figure")

        uri = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
        assert image_uri(directory, "plots/figure.png") == uri
        assert image_uri(directory, "../outside.png") is None
        assert image_uri(directory, str(tmp_path / "outside.png")) is None
        assert image_uri(directory, "notes.txt") is None
        assert image_uri(directory, "plots/missing.png") is None

    def test_tool_output_is_shown_and_cut_short_when_long(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        long = "A" * 4_000 + "MIDDLE" + "Z" * 4_000
        tool = Tool(
            name="look_up",
            description="Looks something up.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: long,
        )
        fake_client.completions.responses = [tool_call_response("look_up"), text_response("Done.")]
        result = individual(team_member, tmp_path, tools=(tool,), num_rounds=0)

        text = text_of(meeting_html(result.transcript_path))

        assert "What the tools returned" in text
        assert "AAAA" in text and "ZZZZ" in text
        assert "MIDDLE" not in text
        transcript = json.loads(result.transcript_path.read_text())
        left_out = len(transcript[1]["message"]) - 2 * (MAX_EXPORTED_OUTPUT_CHARS // 2)
        assert f"[... {left_out:,} characters left out ...]" in text

        whole = text_of(meeting_html(result.transcript_path, max_output_chars=100_000))
        assert "MIDDLE" in whole and "characters left out" not in whole

    def test_a_structured_output_is_shown_as_the_meeting_conclusions(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Go ahead.")]
        fake_client.completions.parsed_responses = [parsed_response(Verdict(decision="go"))]
        result = individual(team_member, tmp_path, output_schema=Verdict, num_rounds=0)

        document = meeting_html(result.transcript_path)
        text = text_of(document)

        assert "Conclusions, as asked for" in text
        assert '<code class="language-json">{\n    &quot;decision&quot;: &quot;go&quot;\n}</code>' in document
        # The summary is the last thing said, not the conclusions restated
        assert section(document, "Summary") == f"Summary {team_member.title} Go ahead."

    def test_html_the_agents_write_is_shown_rather_than_rendered(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [
            text_response('<script>alert("x")</script> and <img src=x onerror=alert(1)> [link](javascript:alert(1))'),
        ]
        member = Agent(
            title='Chemist <b onmouseover="x">',
            expertise="<i>chemistry</i>",
            goal="make molecules",
            role="advise",
            model=TEST_MODEL,
        )
        result = hold_meeting(
            meeting_type="individual",
            agenda="<iframe src=//evil></iframe>",
            agenda_questions=("<style>body{display:none}</style>?",),
            save_dir=tmp_path,
            team_member=member,
            num_rounds=0,
            resources="none",
        )

        document = meeting_html(result.transcript_path)
        body = document.split("</style>", 1)[1]

        for tag in ("<script", "<img", "<iframe", "<b ", "<style", "<i>"):
            assert tag not in body
        assert "&lt;script&gt;" in body and "&lt;iframe" in body and "&lt;i&gt;chemistry" in body
        assert 'href="javascript:' not in body
        assert "Chemist &lt;b onmouseover=&quot;x&quot;&gt;" in body
        assert "<title>Discussion</title>" in document

    def test_a_failed_meeting_is_shown_with_what_stopped_it(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("A first answer.")]
        responses = 0

        def on_event(event: Any) -> None:
            nonlocal responses
            if event.kind == "usage":
                responses += 1
                if responses == 2:
                    raise RuntimeError("the lab lost power")

        with pytest.raises(RuntimeError):
            individual(team_member, tmp_path, num_rounds=1, on_event=on_event)

        text = text_of(meeting_html(tmp_path / "partial" / "discussion.json"))

        assert "failed" in text
        assert "The meeting stopped with RuntimeError: the lab lost power" in text
        assert "A first answer." in text

    def test_a_transcript_without_a_record_is_shown_from_the_transcript_alone(self, tmp_path: Path) -> None:
        transcript = [
            {"agent": "User", "message": "This is the beginning of a meeting about **antibodies**."},
            {"agent": "Principal Investigator", "message": "Let us start."},
            {"agent": "Tool", "message": "Searched PubMed."},
            {"agent": "Immunologist", "message": "Nanobodies are small."},
        ]
        path = tmp_path / "old_meeting.json"
        path.write_text(json.dumps(transcript))

        document = meeting_html(path)
        text = text_of(document)

        assert "Virtual Lab · Meeting" in text
        assert section(document, "Agenda") == "Agenda This is the beginning of a meeting about antibodies ."
        assert "<strong>antibodies</strong>" in document
        assert "Round" not in text and "badge" not in document.split("</style>", 1)[1]
        assert "What the tools returned Searched PubMed." in text
        # The last to answer sums up, and every speaker has a colour of their own
        assert section(document, "Summary") == "Summary Immunologist Nanobodies are small."
        assert document.count(f"--colour: {export.AGENT_COLOURS[0]}") == 1
        assert document.count(f"--colour: {export.AGENT_COLOURS[1]}") == 2

    def test_a_transcript_without_a_record_shows_its_first_prompt_once(self, tmp_path: Path) -> None:
        path = tmp_path / "old_meeting.json"
        path.write_text(json.dumps([{"agent": "User", "message": "The agenda."}, {"agent": "Lead", "message": "Hi."}]))

        document = meeting_html(path, include_prompts=True)

        assert "<h2>Agenda</h2>" not in document
        assert text_of(document).count("The agenda.") == 1
        assert "Prompt The agenda." in text_of(document)

    def test_a_record_that_does_not_match_its_transcript_is_not_used_for_its_turns(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("An answer.")]
        result = individual(team_member, tmp_path, num_rounds=0)
        transcript = json.loads(result.transcript_path.read_text())
        transcript.append({"agent": "Tool", "message": "Added later."})
        result.transcript_path.write_text(json.dumps(transcript))

        text = text_of(meeting_html(result.transcript_path))

        assert "What the tools returned Added later." in text
        assert "Agenda Compute a number." in text

    def test_output_at_the_limit_is_shown_whole(self) -> None:
        assert export.shorten("x" * 100, 100) == "x" * 100
        text = "".join(chr(ord("a") + index % 26) for index in range(101))
        assert export.shorten(text, 100) == f"{text[:50]}\n\n[... 1 characters left out ...]\n\n{text[51:]}"
        assert export.shorten(text, 51) == f"{text[:25]}\n\n[... 51 characters left out ...]\n\n{text[76:]}"

    def test_a_file_that_is_not_a_transcript_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        for content in ({"goal": "x"}, [{"agent": "User"}], [{"agent": 1, "message": "x"}]):
            path.write_text(json.dumps(content))
            with pytest.raises(ValueError, match="is not a meeting's transcript"):
                load_meeting(path)

        with pytest.raises(FileNotFoundError):
            load_meeting(tmp_path / "missing.json")

    def test_a_meeting_is_saved_beside_its_transcript_unless_told_where(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        result = individual(team_member, tmp_path, num_rounds=0)

        saved = save_meeting_html(result.transcript_path)
        elsewhere = save_meeting_html(result.transcript_path, tmp_path / "out" / "meeting.html", include_prompts=True)

        assert saved == tmp_path / "discussion.html"
        assert saved.read_text(encoding="utf-8") == meeting_html(result.transcript_path)
        assert elsewhere == tmp_path / "out" / "meeting.html"
        assert elsewhere.read_text(encoding="utf-8") == meeting_html(result.transcript_path, include_prompts=True)


class TestUsageTable:
    def usage(self, **counts: Any) -> dict[str, Any]:
        usage = MeetingUsage()
        usage.add(TEST_MODEL, make_usage(prompt_tokens=1_000_000, completion_tokens=100_000, cached_tokens=500_000))
        data = usage.to_dict()
        data["per_model"][TEST_MODEL].update(counts)
        return data

    def test_a_model_is_priced_as_the_meeting_priced_it(self) -> None:
        usage = self.usage()

        table = usage_table(usage, {TEST_MODEL: price_per_million(TEST_MODEL)})

        cost = export.money(usage["cost"])
        assert f"<td class='number'>{cost}</td>" in table
        assert f"<th class='number'>{cost}</th>" in table

    def test_a_cost_worked_out_two_ways_reads_the_same(self) -> None:
        # 500 input and 100 output tokens of gpt-4o, as a model's cost and as a meeting's
        assert (500 * 2.5 + 100 * 10) / 1_000_000 != 500 * 2.5e-6 + 100 * 1e-5
        assert export.money((500 * 2.5 + 100 * 10) / 1_000_000) == export.money(500 * 2.5e-6 + 100 * 1e-5) == "$0.0023"
        # 428 input and 198 output tokens come to just under $0.00305 as a float, for $0.00305
        assert 428 * 2.5e-6 + 198 * 1e-5 < 0.00305
        assert export.money(428 * 2.5e-6 + 198 * 1e-5) == "$0.0031"
        assert export.money(1234.56789) == "$1,234.5679"
        assert export.money(0.00004) == "$0.0000"
        assert export.money(0.00005) == "$0.0001"
        assert export.money(None) == "unknown"

    def test_an_unpriced_model_and_unreported_usage_are_said_to_be(self) -> None:
        assert "<td class='number'>unpriced</td>" in usage_table(self.usage(), {TEST_MODEL: None})
        assert "<td class='number'>unpriced</td>" in usage_table(self.usage(), {})

        unreported = usage_table(self.usage(unreported_calls=1), {TEST_MODEL: price_per_million(TEST_MODEL)})
        assert "<td class='number'>unknown</td>" in unreported


def run_a_project(fake_client: FakeClient, tmp_path: Path) -> Project:
    fake_client.completions.parsed_responses.extend(
        [
            roster("Immunologist"),
            plan("Task 1", "Task 2"),
            decide("team_meeting", participants=["Immunologist"], agenda="Compare the binding data."),
            found("A binds better"),
            decide("finish", done=2, answer="Nanobody **A** binds better.", findings=["F1"]),
            review(True),
        ]
    )
    project = Project(tmp_path, GOAL)
    run_project(project, team_lead=LEAD, critic=CRITIC)

    return project


class TestProjects:
    def test_a_project_is_shown_as_its_report(self, fake_client: FakeClient, tmp_path: Path) -> None:
        run_a_project(fake_client, tmp_path)
        report = json.loads((tmp_path / "report.json").read_text())

        document = project_html(tmp_path)
        text = text_of(document)

        assert f"<title>{GOAL}</title>" in document
        assert "Virtual Lab · Project report" in text
        assert '<span class="badge ok">finished</span>' in document
        assert f"{len(report['rounds'])} rounds" in text
        assert report["reason"] in text
        assert "Answer Nanobody A binds better." in text and "<strong>A</strong>" in document
        assert f"{LEAD.title} (team lead)" in text and f"{CRITIC.title} (critic)" in text and "Immunologist" in text
        assert '<li class="done">Task 1</li><li class="done">Task 2</li>' in document
        assert "Findings Id Claim Evidence F1 A binds better Shown for A binds better" in text
        assert "team meeting (Immunologist): Compare the binding data." in text
        assert "finish" in text
        # The meetings are left out unless asked for
        assert '<section class="meeting">' not in document

    def test_a_project_can_be_shown_with_every_meeting_it_finished(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        run_a_project(fake_client, tmp_path)
        meetings = project_meetings(tmp_path)

        document = project_html(tmp_path, meetings=True)

        assert len(meetings) >= 3 and all(path.is_file() for path in meetings)
        assert document.count('<section class="meeting">') == len(meetings)
        sections = document.split('<section class="meeting">')[1:]
        for path, section in zip(meetings, sections):
            assert f"<h2>{export.humanize(path.stem)}</h2>" in section
        # The critic reviewing the answer on its own is both the member and the critic, and is shown once
        (review,) = [section for path, section in zip(meetings, sections) if path.stem.endswith("_review")]
        assert text_of(review).count(f"{CRITIC.title} (critic)") == 1
        assert text_of(review).count(CRITIC.expertise) == 1
        # The report alone has a top heading, so each page's footer names the project
        assert document.count("<h1>") == 1

    def test_only_meetings_that_completed_are_added(self, fake_client: FakeClient, tmp_path: Path) -> None:
        run_a_project(fake_client, tmp_path)
        ledger_path = tmp_path / "project.json"
        ledger = json.loads(ledger_path.read_text())
        meetings = [step for step in ledger["steps"] if step["kind"] == "meeting"]
        meetings[0]["status"] = "failed"
        meetings[1]["status"] = "interrupted"
        meetings[2]["status"] = "running"
        ledger_path.write_text(json.dumps(ledger))

        assert project_meetings(tmp_path) == [tmp_path / step["files"]["transcript"] for step in meetings[3:]]

    def test_a_project_that_did_not_finish_shows_the_answer_the_critic_refused(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        fake_client.completions.parsed_responses.extend(
            [
                roster("Immunologist"),
                plan("Task 1", "Task 2"),
                decide("finish", done=1, answer="Nanobody A, *probably*."),
                review(False, "Task 2 is not done.", "No **binding** data."),
            ]
        )
        run_project(Project(tmp_path, GOAL), team_lead=LEAD, critic=CRITIC, max_rounds=1)

        document = project_html(tmp_path)
        text = text_of(document)

        assert '<span class="badge bad">out of rounds</span>' in document
        assert "Last answer proposed, which the critic did not accept Nanobody A, probably ." in text
        assert "<em>probably</em>" in document
        assert "The critic's objections Task 2 is not done. No binding data." in text
        assert "<strong>binding</strong>" in document
        assert '<li class="done">Task 1</li><li class="todo">Task 2</li>' in document
        assert "<h2>Answer</h2>" not in document

    def test_a_project_without_a_report_cannot_be_shown(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="has no report.json"):
            project_html(tmp_path)

    def test_a_project_is_saved_in_its_directory_unless_told_where(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        run_a_project(fake_client, tmp_path)

        saved = save_project_html(tmp_path)
        elsewhere = save_project_html(tmp_path, tmp_path / "out" / "everything.html", meetings=True)

        assert saved == tmp_path / "report.html"
        assert elsewhere == tmp_path / "out" / "everything.html"
        assert saved.read_text(encoding="utf-8") == project_html(tmp_path)
        assert elsewhere.read_text(encoding="utf-8") == project_html(tmp_path, meetings=True)


class TestPDF:
    def test_a_pdf_without_weasyprint_says_how_to_install_it(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = individual(team_member, tmp_path, num_rounds=0)
        monkeypatch.setitem(sys.modules, "weasyprint", None)

        with pytest.raises(PDFExportError, match=r"pip install 'virtual-lab\[pdf\]'.*brew install pango"):
            save_meeting_pdf(result.transcript_path)

        assert not (tmp_path / "discussion.pdf").exists()

    def test_a_pdf_without_pango_says_how_to_install_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def without_pango(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "weasyprint":
                print("WeasyPrint could not import some external libraries.")
                raise OSError("cannot load library 'libgobject-2.0-0'")
            return real_import(name, *args, **kwargs)

        real_import = builtins.__import__
        monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
        monkeypatch.setattr(builtins, "__import__", without_pango)
        printed = io.StringIO()
        monkeypatch.setattr(sys, "stdout", printed)

        with pytest.raises(PDFExportError, match=r"apt install libpango.*OSError: cannot load library"):
            export.render_pdf("<p>x</p>", tmp_path / "x.pdf")

        assert printed.getvalue() == ""

    def test_what_weasyprint_prints_and_warns_of_as_it_loads_is_kept_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        made: list[tuple[str, Any, Path]] = []

        class HTML:
            def __init__(self, string: str, url_fetcher: Any) -> None:
                self.string, self.url_fetcher = string, url_fetcher

            def write_pdf(self, path: Path) -> None:
                assert path.parent.is_dir()
                made.append((self.string, self.url_fetcher, path))

        def noisy(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "weasyprint":
                print("Some advice.")
                warnings.warn("HarfBuzz-Subset will be required by future versions", DeprecationWarning)
                return SimpleNamespace(HTML=HTML)
            return real_import(name, *args, **kwargs)

        real_import = builtins.__import__
        monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
        monkeypatch.setattr(builtins, "__import__", noisy)
        printed = io.StringIO()
        monkeypatch.setattr(sys, "stdout", printed)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            path = export.render_pdf("<p>x</p>", tmp_path / "new" / "x.pdf")

        assert path == tmp_path / "new" / "x.pdf"
        assert [(string, written) for string, _, written in made] == [("<p>x</p>", path)]
        assert printed.getvalue() == ""

    def test_weasyprint_from_68_fetches_only_data_uris(self) -> None:
        made: list[dict[str, Any]] = []

        def URLFetcher(**options: Any) -> str:
            made.append(options)
            return "fetcher"

        assert export.url_fetcher(SimpleNamespace(URLFetcher=URLFetcher, default_url_fetcher=None)) == "fetcher"
        assert made == [{"allowed_protocols": ("data",)}]

    def test_weasyprint_before_68_fetches_only_data_uris(self) -> None:
        fetched: list[tuple[str, dict[str, Any]]] = []

        def default_url_fetcher(url: str, **options: Any) -> dict[str, Any]:
            fetched.append((url, options))
            return {"string": b"png"}

        fetch = export.url_fetcher(SimpleNamespace(default_url_fetcher=default_url_fetcher))

        assert fetch("data:image/png;base64,AAAA", timeout=5) == {"string": b"png"}
        assert fetch("DATA:image/png;base64,AAAA") == {"string": b"png"}
        for url in ("http://127.0.0.1/plot.png", "https://example.org/a.png", "file:///etc/hosts", "plots/a.png"):
            with pytest.raises(ValueError, match="is not fetched"):
                fetch(url)
        assert fetched == [("data:image/png;base64,AAAA", {"timeout": 5}), ("DATA:image/png;base64,AAAA", {})]

    @needs_weasyprint
    def test_a_meeting_is_saved_as_a_pdf_with_its_figures(
        self, fake_client: FakeClient, team_member: Agent, session: LocalSession, tmp_path: Path
    ) -> None:
        from pypdf import PdfReader

        fake_client.completions.responses = [
            text_response("I will compute it.\n<execute>\nprint(6 * 7)\n</execute>"),
            text_response("The answer is 42, as the figure shows."),
        ]
        result = individual(team_member, tmp_path, session=session, code_actions="tags", num_rounds=0)

        path = save_meeting_pdf(result.transcript_path)

        assert path == tmp_path / "discussion.pdf"
        reader = PdfReader(path)
        text = " ".join(page.extract_text() for page in reader.pages)
        assert "VIRTUAL LAB · INDIVIDUAL MEETING" in text
        assert "The answer is 42" in text
        assert "print(6 * 7)" in text
        assert re.search(r"Page 1 of \d+", text)
        # Counted by content, since pages can share the resources an image is listed in
        assert len({image.data for page in reader.pages for image in page.images}) == 1

    @needs_weasyprint
    def test_a_pdf_fetches_nothing(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        from pypdf import PdfReader

        asked: list[str] = []

        class Figures(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                asked.append(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(PNG)

            def log_message(self, *args: Any) -> None:
                pass

        (tmp_path / "local.png").write_bytes(PNG)
        server = http.server.HTTPServer(("127.0.0.1", 0), Figures)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            fake_client.completions.responses = [
                text_response(
                    f"See ![a plot](http://127.0.0.1:{server.server_port}/plot.png) and "
                    f"![local]({(tmp_path / 'local.png').as_uri()})."
                ),
            ]
            result = individual(team_member, tmp_path, num_rounds=0)

            path = save_meeting_pdf(result.transcript_path, tmp_path / "pdf" / "meeting.pdf")
        finally:
            server.shutdown()
            server.server_close()

        assert path.is_file()
        assert asked == []
        assert sum(len(page.images) for page in PdfReader(path).pages) == 0

    @needs_weasyprint
    def test_a_project_is_saved_as_a_pdf_with_its_meetings(self, fake_client: FakeClient, tmp_path: Path) -> None:
        from pypdf import PdfReader

        run_a_project(fake_client, tmp_path)

        report = save_project_pdf(tmp_path)
        everything = save_project_pdf(tmp_path, tmp_path / "everything.pdf", meetings=True)

        assert report == tmp_path / "report.pdf"
        short, long = PdfReader(report), PdfReader(everything)
        assert "Nanobody A binds better." in short.pages[0].extract_text()
        # Each meeting starts a page of its own
        assert len(long.pages) >= len(short.pages) + len(project_meetings(tmp_path))
