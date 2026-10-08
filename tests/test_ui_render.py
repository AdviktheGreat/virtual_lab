"""Tests for what the web interface shows of a run: that nothing a model, a tool, or a server said
can run in the page, and that each state of a run is shown as it is."""

import re
from pathlib import Path
from typing import Any

from virtual_lab.approval import ApprovalRequest, ServerQuestion
from virtual_lab.events import MeetingEvent, NextTurn, ProjectEvent
from virtual_lab.ui import render
from virtual_lab.ui.runs import RunView, Waiting
from virtual_lab.ui.workspace import Entry

from test_planning import step

MEETING = "meeting-1"
TEAM = [
    {"title": "Immunologist", "model": "gpt-5.2"},
    {"title": "Scientific Critic", "model": "gpt-5.2"},
]
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000001e221bc330000000049454e44ae426082"
)


def event(
    name: str, round_: int | None = None, speaker: str | None = None, text: str = "", /, **data: Any
) -> MeetingEvent:
    return MeetingEvent(name, MEETING, round_, speaker, text, data)  # type: ignore[arg-type]


def started(**data: Any) -> MeetingEvent:
    data = {
        "meeting_type": "team",
        "agenda": "Choose the epitopes.",
        "agenda_questions": ["Which one?"],
        "num_rounds": 2,
        "team": TEAM,
        **data,
    }
    return event("started", **data)


def message(kind: str, text: str, speaker: str = "Immunologist", round_: int | None = 1) -> MeetingEvent:
    return event("message", round_, speaker, text, kind=kind)


def view(events: tuple[Any, ...] = (), **fields: Any) -> RunView:
    values: dict[str, Any] = {
        "id": "run-1",
        "kind": "meeting",
        "title": "Choose the epitopes.",
        "directory": Path("/work/run"),
        "status": "running",
        "error": None,
        "events": events,
        "notes": (),
        "paused": False,
        "holding": False,
        "next_turn": None,
        "stopping": None,
        "autonomous": False,
        "waiting": (),
        "spent": 0.5,
        "max_cost": 2.0,
        "started_at": 1000.0,
        "ended_at": None,
        "version": 1,
        "result": None,
    }
    return RunView(**{**values, **fields})


class TestNothingSaidRunsInThePage:
    def test_what_agents_and_tools_said_is_shown_as_text(self) -> None:
        attack = "<script>alert(1)</script><img src=x onerror=alert(2)>"
        events = [
            started(agenda=attack, agenda_questions=[attack], team=[{"title": attack, "model": attack}]),
            event("turn", 1, attack, model=attack),
            message("response", attack, speaker=attack),
            message("note", attack),
            message("tool_output", attack),
            message("code_output", attack),
            message("structured_output", attack),
            event("tool_calls", 1, attack, attack, calls=[{"name": attack, "arguments": attack}]),
            event("cell", None, None, "", status=attack, error=attack, duration=1, plot_paths=[]),
            event("finished", None, None, attack, usage={}, elapsed_time=1),
            event("failed", None, None, attack, type=attack),
            event("read_back", None, None, attack),
        ]

        page = render.meeting_feed(events, live=False)

        assert "<script" not in page and "<img" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page

    def test_what_a_server_asks_and_what_a_tool_would_be_called_with_is_shown_as_text(self) -> None:
        attack = "<script>alert(1)</script>"
        request = ApprovalRequest(
            tool="t", server=attack, server_tool=attack, arguments={"q": attack}, description=attack
        )
        question = ServerQuestion(
            server=attack,
            message=attack,
            fields={attack: {"type": "string", "description": attack, "enum": [attack]}},
            required=(attack,),
        )

        for subject in (request, question):
            page = render.waiting_html(Waiting(1, "approval", subject))

            assert "<script" not in page

    def test_a_web_page_a_server_asks_for_is_linked_only_if_it_is_a_web_page(self) -> None:
        def page(url: str) -> str:
            question = ServerQuestion(server="proto", message="Sign in.", url=url)
            return render.waiting_html(Waiting(1, "question", question))

        linked = page("https://example.org/sign-in?a=1&b=2")
        assert 'href="https://example.org/sign-in?a=1&amp;b=2"' in linked
        assert 'target="_blank"' in linked and 'rel="noopener noreferrer"' in linked

        for url in ("javascript:alert(1)", " JavaScript:alert(1)", "data:text/html,<b>", "file:///etc/passwd"):
            shown = page(url)
            assert "href=" not in shown
            assert "not a web page" in shown

    def test_links_in_what_was_said_open_in_a_new_tab_and_never_run_script(self) -> None:
        text = "[paper](https://example.org/a) and [bad](javascript:alert(1)) and <a href='https://x'>raw</a>"

        page = render.markdown(text)

        assert page.count("<a ") == 1
        assert 'target="_blank" rel="noopener noreferrer" href="https://example.org/a"' in page
        assert "javascript:alert(1)" in page and 'href="javascript' not in page
        assert "&lt;a href=" in page

    def test_a_document_is_framed_with_every_permission_off(self) -> None:
        framed = render.framed('<html onload="x()"><body>"quoted"</body></html>')

        assert 'sandbox=""' in framed
        assert "srcdoc=" in framed
        assert "<html" not in framed
        assert "&quot;quoted&quot;" in framed


class TestMeetingFeed:
    def test_the_agenda_team_and_rounds_are_laid_out(self) -> None:
        events = [
            started(),
            event("turn", 1, "Immunologist", model="gpt-5.2"),
            message("response", "First.", round_=1),
            message("response", "Second.", "Scientific Critic", round_=2),
            message("response", "Third.", round_=3),
            event("finished", None, None, "The summary.", usage={"cost": 0.0123}, elapsed_time=65),
        ]

        page = render.meeting_feed(events, live=False)

        assert "Team meeting · 2 rounds of discussion" in page
        assert "Choose the epitopes." in page and "Which one?" in page
        assert page.count('class="vl-chip"') == 2
        assert "Round 1 of 2" in page and "Round 2 of 2" in page and "Final round" in page
        assert page.index("First.") < page.index("Second.") < page.index("Third.")
        assert "Summary · cost $0.0123 · 1 min 5 s" in page
        assert "The summary." in page

    def test_an_individual_meeting_with_no_rounds_ends_in_the_answer(self) -> None:
        page = render.meeting_feed(
            [started(meeting_type="individual", num_rounds=0, team=TEAM[:1]), message("response", "Done.", round_=1)],
            live=False,
        )

        assert "Individual meeting · one answer" in page
        assert "The answer" in page and "Final round" not in page

    def test_only_the_latest_event_of_a_live_meeting_shows_what_is_under_way(self) -> None:
        thinking = [started(), event("turn", 1, "Immunologist", model="gpt-5.2")]
        writing = [*thinking, event("writing", 1, "Immunologist", "Half a rep", model="gpt-5.2", request=1)]

        assert "vl-thinking" in render.meeting_feed(thinking, live=True)
        assert "vl-thinking" not in render.meeting_feed(thinking, live=False)
        assert "vl-caret" in render.meeting_feed(writing, live=True)
        assert "Half a rep" in render.meeting_feed(writing, live=True)
        assert "Half a rep" not in render.meeting_feed(writing, live=False)
        assert "vl-thinking" not in render.meeting_feed([*writing, message("response", "Whole.")], live=True)

    def test_each_kind_of_message_is_shown_as_what_it_is(self) -> None:
        code = "Let me look.\n<execute>\nprint(1)\n</execute>"
        events = [
            started(),
            message("prompt", "Asked this."),
            message("note", "A note from the person."),
            message("tool_output", "Tool said so."),
            message("code_action", code),
            message("code_output", "<observation>1</observation>"),
            message("structured_output", '{"a": 1}'),
        ]

        page = render.meeting_feed(events, live=False)

        assert 'class="vl-fold vl-prompt"' in page and "What the next speaker was asked" in page
        assert "You</span>" in page and "a note to the meeting" in page
        assert "What the tools returned" in page and "What the code printed" in page
        assert "Code (python)" in page and "print(1)" in page and "<execute>" not in page
        assert "<observation>" not in page
        assert "conclusions, as asked for" in page

    def test_a_figure_is_carried_in_the_page_and_one_that_cannot_be_read_is_said_so(self, tmp_path: Path) -> None:
        drawn = tmp_path / "plot.png"
        drawn.write_bytes(PNG)
        (tmp_path / "notes.txt").write_text("not a figure")

        def feed(*paths: Path) -> str:
            return render.meeting_feed(
                [event("cell", None, None, "", status="ok", plot_paths=[str(p) for p in paths])], False
            )

        assert 'src="data:image/png;base64,' in feed(drawn)
        assert "Figure 1" in feed(drawn)
        assert "cannot be shown here" in feed(tmp_path / "missing.png")
        assert "cannot be shown here" in feed(tmp_path / "notes.txt")
        assert "data:" not in feed(tmp_path / "notes.txt")

    def test_a_cell_that_failed_without_figures_is_folded_with_its_error(self) -> None:
        page = render.meeting_feed([event("cell", None, None, "", status="error", error="NameError: x")], live=False)

        assert "<details" in page and "NameError: x" in page and "vl-badge bad" in page

    def test_how_a_meeting_ended_is_said(self) -> None:
        stopped = render.meeting_feed([event("failed", None, None, "stopped", type="RunStopped")], live=False)
        failed = render.meeting_feed([event("failed", None, None, "the key is wrong", type="AuthError")], live=False)

        assert "Stopped by you" in stopped
        assert "AuthError" in failed and "the key is wrong" in failed

    def test_a_fold_keeps_its_key_as_the_meeting_grows(self) -> None:
        before = [started(), message("prompt", "Asked this.")]
        after = [*before, message("response", "Said that.")]

        def keys(events: list[MeetingEvent]) -> list[str]:
            return re.findall(r'data-key="([^"]+)"', render.meeting_feed(events, live=True))

        assert keys(before) and keys(after)[: len(keys(before))] == keys(before)


class TestProjectFeed:
    def project_events(self) -> tuple[Any, ...]:
        proposed = step("team_meeting", participants=["Immunologist"], agenda="Compare.").model_dump()
        done = {"round": {"outcome": "done", "note": ""}, "plan": [{"task": "A", "status": "done"}]}
        return (
            ProjectEvent("team", None, "", {"team": TEAM, "change": {"why": "Chosen by the team lead."}}),
            ProjectEvent("plan", None, "", {"plan": [{"task": "A", "status": "to do"}]}),
            ProjectEvent("decided", 1, "", {"proposed": proposed}),
            event("started", meeting_type="team", agenda="Compare.", num_rounds=1, team=TEAM),
            event("finished", None, None, "Summary.", usage={}),
            ProjectEvent("round", 1, "", done),
        )

    def test_each_meeting_is_a_fold_and_the_one_going_on_is_open(self) -> None:
        second = MeetingEvent(
            "started", "round-02", None, None, "", {"meeting_type": "team", "num_rounds": 1, "team": TEAM}
        )
        events = (*self.project_events(), second)

        page = render.project_feed(view(events, kind="project"))

        assert page.count('class="vl-fold vl-meeting"') == 2
        assert re.search(r'<details class="vl-fold vl-meeting" data-key="meeting:meeting-1">', page)
        assert re.search(r'<details class="vl-fold vl-meeting" data-key="meeting:round-02" open>', page)
        assert "The team: <strong>Immunologist, Scientific Critic</strong>. Chosen by the team lead." in page
        assert "The team made a plan of 1 tasks." in page
        assert "Team meeting</strong> with Immunologist: Compare." in page

    def test_a_project_that_has_done_nothing_yet_says_it_is_starting(self) -> None:
        assert "The project is starting" in render.project_feed(view((), kind="project"))
        assert "starting" not in render.project_feed(view((), kind="project", status="completed"))

    def test_the_board_shows_the_goal_team_plan_rounds_and_report(self) -> None:
        events = self.project_events()[:6]
        report = {
            "status": "finished",
            "reason": "The critic agreed.",
            "answer": "Nanobody A.",
            "objections": ["Small n."],
        }
        finished = ProjectEvent(
            "finished", 2, "done", {"report": {**report, "plan": [{"task": "A", "status": "done"}]}}
        )

        running = render.project_board(view(events, kind="project", title="Find a nanobody."))
        ended = render.project_board(view((*events, finished), kind="project", title="Find a nanobody."))

        assert "Find a nanobody." in running and "Immunologist" in running and "Plan · 1 of 1 done" in running
        assert "Report" not in running
        assert "Nanobody A." in ended and "The critic agreed." in ended and "Small n." in ended

    def test_a_board_with_no_plan_says_so(self) -> None:
        assert "Not made yet." in render.project_board(view((), kind="project"))


class TestStatus:
    def test_what_a_run_is_doing_is_said_in_a_few_words(self) -> None:
        turn = [started(), event("turn", 1, "Immunologist")]
        cases = [
            (view(turn), "Immunologist is thinking."),
            (view([*turn, event("writing", 1, "Immunologist", "x")]), "Immunologist is writing."),
            (view([*turn, event("tool_calls", 1, "Immunologist", calls=[])]), "Immunologist is calling tools."),
            (view([*turn, event("code", 1, "Immunologist")]), "Immunologist's code is running."),
            (view(turn, paused=True), "Pausing once the reply being written is done."),
            (
                view(turn, paused=True, holding=True, next_turn=NextTurn(MEETING, 1, "Scientific Critic")),
                "Paused before Scientific Critic's turn.",
            ),
            (view(turn, paused=True, holding=True), "Paused before the next decision."),
            (view(turn, stopping="now"), "Stopping…"),
            (view((), kind="project"), "Choosing the team."),
            (view(turn, status="completed"), "Finished."),
            (view(turn, status="stopped"), "Stopped by you."),
            (view(turn, status="failed", error="boom"), "Stopped with an error."),
        ]

        for run, said in cases:
            assert render.doing(run) == said

    def test_a_run_waiting_for_the_person_says_so_before_anything_else(self) -> None:
        waiting = (Waiting(1, "approval", ApprovalRequest("t", "s", "st", {})),)

        page = render.status_html(view([started()], waiting=waiting, paused=True))

        assert "Waiting for you." in page
        assert 'class="vl-badge ask">Waiting' in page

    def test_the_budget_meter_turns_amber_and_red_as_it_runs_out(self) -> None:
        def tone(spent: float) -> str:
            return re.search(
                r'class="vl-meter (\w+)"><span style="width:([\d.]+)%', render.status_html(view(spent=spent))
            )[1]  # type: ignore[index]

        assert [tone(0.5), tone(1.5), tone(1.9)] == ["ok", "ask", "bad"]
        assert "Spent $0.5000 of $2.0000" in render.status_html(view())
        assert "with no budget" in render.status_html(view(max_cost=None))

    def test_the_round_of_a_meeting_is_shown_with_its_final_round(self) -> None:
        events = [started(), event("turn", 1, "Immunologist")]
        final = [started(), event("turn", 3, "Immunologist")]

        assert "Round 1 of 2" in render.status_html(view(events))
        assert "Final round" in render.status_html(view(final))
        assert "Round 1" not in render.status_html(view(events, kind="project"))

    def test_notes_not_yet_read_and_a_stop_after_the_step_are_listed(self) -> None:
        page = render.status_html(view([started()], notes=("Look at <b>B</b>.",), stopping="after_step"))

        assert "Notes waiting for the next turn" in page and "Look at &lt;b&gt;B&lt;/b&gt;." in page
        assert "Stopping after the step under way." in page

    def test_a_finished_run_shows_how_long_it_took_and_its_error(self) -> None:
        page = render.status_html(view(status="failed", error="<bad>", started_at=1000, ended_at=1075), now=5000)

        assert "1 min 15 s" in page and "&lt;bad&gt;" in page

    def test_the_meter_is_empty_for_no_limit_and_full_past_it(self) -> None:
        assert "width:0.0%" in render.meter(1, 0)
        assert "width:100.0%" in render.meter(5, 2)
        assert "vl-meter ok" in render.meter(1.9, 2)
        assert "vl-meter bad" in render.meter(1.9, 2, warn=True)


class TestWaiting:
    def test_a_decision_shows_the_step_and_what_it_proposes(self) -> None:
        proposed = step("finish", answer="Nanobody A.", agenda_questions=["Why?"], rationale="Enough is known.")

        page = render.waiting_html(Waiting(3, "decision", proposed, round=4))

        assert "Round 4 · The team lead's decision waits for you" in page
        assert "Finish with an answer" in page and "Enough is known." in page
        assert "Nanobody A." in page and "Why?" in page

    def test_an_approval_shows_the_call_and_a_question_its_fields(self) -> None:
        request = ApprovalRequest("order", "lab", "place_order", {"n": 2}, "Orders a thing.")
        question = ServerQuestion(
            "lab",
            "Which plan?",
            {"plan": {"enum": ["a", "b"], "description": "The plan."}, "n": {"type": "integer"}},
            ("plan",),
        )

        approval = render.waiting_html(Waiting(1, "approval", request))
        asked = render.waiting_html(Waiting(2, "question", question))

        assert "place_order" in approval and "Orders a thing." in approval and "&quot;n&quot;: 2" in approval
        assert "Which plan?" in asked and "one of a, b (required)" in asked and "The plan." in asked

    def test_a_decision_that_changes_the_team_says_who_comes_and_goes(self) -> None:
        proposed = step("change_team", remove_members=["Geneticist"])

        page = render.waiting_html(Waiting(1, "decision", proposed, round=2))

        assert "Joining: no one. Leaving: Geneticist." in page


class TestHistory:
    def entry(self, tmp_path: Path, status: str, cost: float | None, title: str = "About  it") -> Entry:
        return Entry("meeting", tmp_path / status, title, "2026-10-07T15:30:12", status, cost)

    def test_each_entry_is_a_row_and_a_run_no_longer_going_is_interrupted(self, tmp_path: Path) -> None:
        going = self.entry(tmp_path, "running", 0.5)
        left = Entry("project", tmp_path / "left", "Goal", "2026-10-07T15:31:00", "running", None)
        done = self.entry(tmp_path, "completed", 0.0123, title="x " * 100)

        carried = Entry("project", tmp_path / "carried", "Again", "2026-10-07T15:32:00", "finished", 1.0)

        rows = render.history_rows([going, left, done, carried], running=[going.path, carried.path])

        assert rows[0] == ["2026-10-07 15:30", "Meeting", "About it", "Running", "$0.5000"]
        assert rows[1] == ["2026-10-07 15:31", "Project", "Goal", "Interrupted", "unknown"]
        assert rows[2][3] == "Completed" and rows[2][4] == "$0.0123"
        assert len(rows[2][2]) <= 150 and "characters left out" in rows[2][2]
        assert rows[3][3] == "Running"

    def test_a_project_that_has_no_report_yet_is_shown_as_such(self, tmp_path: Path) -> None:
        directory = tmp_path / "project"
        directory.mkdir()
        entry = Entry("project", directory, "A <goal>", "", "not finished", None)

        document = render.entry_document(entry)

        assert "no report yet" in document and "&lt;goal&gt;" in document
