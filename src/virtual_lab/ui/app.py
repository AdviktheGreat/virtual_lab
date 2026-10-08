"""The web interface, as Biomni's Gradio demo is, but for a lab: a meeting room to hold a meeting
in and follow it as it happens, a project room to run a project to its goal and approve each
step, the history of everything run, and settings.

The page asks for what is going on every half second, and shows only what changed. A run goes on
whether or not a page is open, and a page opened later shows the run going on.
"""

import argparse
import ipaddress
import os
import secrets
from pathlib import Path
from typing import Any

import gradio as gr

from virtual_lab.constants import CONSISTENT_TEMPERATURE
from virtual_lab.export import PDFExportError, save_meeting_html, save_meeting_pdf, save_project_html, save_project_pdf
from virtual_lab.planning import NextStep
from virtual_lab.ui import library, render
from virtual_lab.ui.runs import (
    CODE_PLACES,
    RESOURCE_MODES,
    SANDBOXES,
    CodeSetup,
    Connections,
    Lab,
    MeetingSetup,
    ProjectSetup,
    Run,
    agent_data,
    changed_step,
    form_answer,
    read_setup,
)
from virtual_lab.ui.theme import CSS, JS, gradio_theme
from virtual_lab.ui.workspace import DEFAULT_WORKSPACE, Settings, Workspace

TICK_SECONDS = 0.5

CODE_CHOICES = [
    ("Nowhere: the agents only talk", "none"),
    ("In a Docker container", "docker"),
    ("On this machine, with no isolation", "local"),
]
SANDBOX_CHOICES = [
    ("Plain Python (python:3.12-slim)", "python"),
    ("Biomni's base environment", "base"),
    ("Biomni's bioinformatics environment", "bio"),
    ("Biomni's full environment", "full"),
]
RESOURCE_CHOICES = [
    ("The lead picks what the agenda needs", "retrieve"),
    ("Everything", "all"),
    ("Nothing", "none"),
]
RESOURCES_LABEL = "What the agents are told of Biomni's tools, data, and know-how"
RESOURCES_INFO = "Used only where the agents' code runs, and most useful in one of Biomni's environments."
MEMORY_CHOICES = [
    ("The team lead picks the findings each step needs", "pick"),
    ("Each step is given the findings that match it best", "bm25"),
    ("Each step is given every summary before it", "summaries"),
]
DEFAULT_MEMBERS = ["Immunologist", "Machine Learning Specialist", "Computational Biologist"]

MEETING_HERO = (
    '<div class="vl-hero"><h2>The meeting room</h2>'
    "<p>Set up a meeting on the left and start it. The discussion appears here as it happens, reply by reply.</p>"
    "<p>While it runs, add a note for the next agent to speak, pause it, or stop it, on the right.</p></div>"
)
PROJECT_HERO = (
    '<div class="vl-hero"><h2>The project room</h2>'
    "<p>Give the lab a goal. The team lead chooses a team, makes a plan, and decides each step, "
    "which waits for your approval unless you let it run on its own.</p></div>"
)

assert {value for _, value in CODE_CHOICES} == set(CODE_PLACES)
assert {value for _, value in SANDBOX_CHOICES} == set(SANDBOXES)
assert {value for _, value in RESOURCE_CHOICES} == set(RESOURCE_MODES)


class Interface:
    """The page's handlers, over one workspace and its runs."""

    def __init__(self, lab: Lab) -> None:
        self.lab = lab
        self.workspace = lab.workspace

    # What there is to choose from

    def settings(self) -> Settings:
        return self.workspace.load_settings()

    def scientist_titles(self) -> list[str]:
        titles = [agent.title for agent in library.SCIENTISTS]
        return titles + [item["title"] for item in self.workspace.load_scientists() if item["title"] not in titles]

    def person(self, title: str, model: str) -> dict[str, str]:
        """A scientist, of the library or the person's own, given the model chosen."""
        for item in self.workspace.load_scientists():
            if item["title"] == title:
                return {**item, "model": model}
        try:
            return agent_data(library.scientist(title).with_model(model))
        except KeyError:
            raise gr.Error(f"There is no scientist titled {title!r}") from None

    def summary_choices(self) -> list[tuple[str, str]]:
        """Meetings that finished, to give a new meeting their summaries."""
        return [
            (f"{entry.started_at[:10]} · {render.shorten(' '.join(entry.title.split()), 80)}", str(entry.transcript))
            for entry in self.workspace.entries()
            if entry.kind == "meeting" and entry.status == "completed" and entry.transcript is not None
        ]

    def mcp_choices(self) -> list[tuple[str, str]]:
        return library.mcp_choices(self.settings().mcp_config)[0]

    def check_keys(self, model: str) -> None:
        if self.lab.client is not None or self.lab.chat_models is not None:
            return
        if missing := library.missing_keys([model]):
            raise gr.Error(f"{model} needs {', '.join(missing)}, which is not set. Set it in Settings.")

    def code_setup(self, where: str, resources: str) -> CodeSetup:
        settings = self.settings()
        return CodeSetup(
            where=where,
            sandbox=settings.sandbox,
            network=settings.network,
            python=settings.python,
            resources=resources,
        )

    # Starting a run

    def start_meeting(
        self,
        meeting_type: str,
        agenda: str,
        questions: str,
        rules: str,
        lead: str,
        members: list[str],
        critic: str,
        rounds: float,
        model: str,
        budget: float | None,
        temperature: float,
        tools: list[str],
        servers: list[str],
        code: str,
        resources: str,
        summaries: list[str],
        stream: bool,
    ) -> None:
        running = self.lab.latest("meeting")
        if running is not None and running.status == "running":
            raise gr.Error("A meeting is going on. Stop it, or wait for it to end, before starting another.")
        model = (model or "").strip()
        if not model:
            raise gr.Error("Choose a model")
        if not lead:
            raise gr.Error("Choose who leads the meeting" if meeting_type == "team" else "Choose who works on it")
        self.check_keys(model)

        try:
            setup = MeetingSetup(
                meeting_type=meeting_type,
                agenda=agenda.strip(),
                lead=self.person(lead, model),
                members=[self.person(title, model) for title in members if title != lead]
                if meeting_type == "team"
                else [],
                critic=self.person(critic, model) if meeting_type == "individual" and critic else None,
                agenda_questions=lines(questions),
                agenda_rules=lines(rules),
                summaries=list(summaries or []),
                num_rounds=int(rounds),
                temperature=float(temperature),
                max_cost=budget_of(budget),
                stream=bool(stream),
                code=self.code_setup(code, resources),
                connections=Connections(
                    tools=list(tools or []), mcp_config=self.settings().mcp_config, mcp_servers=list(servers or [])
                ),
            )
            self.lab.start_meeting(setup)
        except (TypeError, ValueError) as error:
            raise gr.Error(str(error)) from None

    def start_project(
        self,
        goal: str,
        autonomous: bool,
        choose_team: bool,
        team: list[str],
        team_size: float,
        rounds: float,
        stalled: float,
        meeting_rounds: float,
        memory: str,
        model: str,
        budget: float | None,
        tools: list[str],
        servers: list[str],
        code: str,
        resources: str,
        stream: bool,
    ) -> None:
        running = self.lab.latest("project")
        if running is not None and running.status == "running":
            raise gr.Error("A project is going on. Stop it, or wait for it to end, before starting another.")
        model = (model or "").strip()
        if not model:
            raise gr.Error("Choose a model")
        self.check_keys(model)

        try:
            setup = ProjectSetup(
                goal=goal.strip(),
                lead=self.person("Principal Investigator", model),
                critic=self.person("Scientific Critic", model),
                team=[self.person(title, model) for title in team or []] if choose_team else [],
                max_team_size=int(team_size),
                max_rounds=int(rounds),
                max_stalled_rounds=int(stalled),
                meeting_rounds=int(meeting_rounds),
                memory=memory,
                max_cost=budget_of(budget),
                autonomous=bool(autonomous),
                stream=bool(stream),
                code=self.code_setup(code, resources),
                connections=Connections(
                    tools=list(tools or []), mcp_config=self.settings().mcp_config, mcp_servers=list(servers or [])
                ),
            )
            if choose_team and not setup.team:
                raise ValueError("Choose the team, or let the team lead choose it")
            self.lab.start_project(setup)
        except (TypeError, ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from None

    def carry_on(self, path: str, budget: float | None, rounds: float) -> Any:
        """Carries on a project from the history, with its budget and rounds raised."""
        if not path:
            raise gr.Error("Choose a project in the history")
        running = self.lab.latest("project")
        if running is not None and running.status == "running":
            raise gr.Error("A project is going on. Stop it, or wait for it to end, before carrying on another.")
        setup = read_setup(Path(path))
        if not isinstance(setup, ProjectSetup):
            raise gr.Error("This project was not started here, so what it was set up with is not known")
        setup.max_cost = budget_of(budget)
        setup.max_rounds = int(rounds)
        self.check_keys(setup.lead["model"])
        try:
            self.lab.start_project(setup, directory=Path(path))
        except (TypeError, ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from None

        return gr.Tabs(selected="project")

    # Steering

    def run_of(self, kind: str) -> Run:
        run = self.lab.latest(kind)  # type: ignore[arg-type]
        if run is None:
            raise gr.Error(f"No {kind} has been started")
        return run

    def add_note(self, kind: str, note: str) -> str:
        run = self.run_of(kind)
        if not note.strip():
            raise gr.Error("Write the note first")
        if not run.add_note(note):
            raise gr.Error(f"The {kind} has ended, so there is no one to read the note")
        gr.Info("The next agent to speak will read your note first.")
        return ""

    def toggle_pause(self, kind: str) -> None:
        run = self.run_of(kind)
        if run.view().paused:
            run.resume()
        else:
            run.pause()

    def stop(self, kind: str, now: bool) -> None:
        self.run_of(kind).stop(now=now)

    def set_autonomous(self, autonomous: bool) -> None:
        run = self.lab.latest("project")
        if run is not None:
            run.set_autonomous(bool(autonomous))

    def answer_ask(self, kind: str, seen: dict[str, Any], yes: bool, form: str) -> str:
        """Answers the approval or question the page showed, as the person did."""
        run = self.run_of(kind)
        shown = (seen or {}).get(f"{kind}_ask")
        waiting = next((item for item in run.view().waiting if item.id == shown), None)
        if waiting is None:
            raise gr.Error("This is no longer waiting for an answer")
        if waiting.kind == "approval":
            run.respond(waiting.id, yes)
        elif not yes:
            run.respond(waiting.id, None)
        elif waiting.subject.fields:  # type: ignore[union-attr]
            values = dict(
                (name.strip(), value.strip())
                for name, _, value in (line.partition(":") for line in form.splitlines())
                if name.strip()
            )
            answer, problem = form_answer(waiting.subject, values)  # type: ignore[arg-type]
            if problem is not None:
                raise gr.Error(problem)
            run.respond(waiting.id, answer)
        else:
            run.respond(waiting.id, {})

        return ""

    def decide(
        self, seen: dict[str, Any], choice: str, agenda: str, questions: str, participants: str, answer: str
    ) -> None:
        run = self.run_of("project")
        shown = (seen or {}).get("project_decision")
        waiting = next((item for item in run.view().waiting if item.id == shown and item.kind == "decision"), None)
        if waiting is None:
            raise gr.Error("This decision is no longer waiting for you")
        step = waiting.subject
        assert isinstance(step, NextStep)
        if choice == "stop":
            run.respond(waiting.id, None)
        elif choice == "change":
            try:
                run.respond(
                    waiting.id,
                    changed_step(
                        step, agenda=agenda, agenda_questions=questions, participants=participants, answer=answer
                    ),
                )
            except ValueError as error:
                raise gr.Error(str(error)) from None
        else:
            run.respond(waiting.id, step)

    # Exporting

    def export(self, kind: str, path: str | None, pdf: bool) -> Any:
        """Saves a meeting or project as a document, for the page to offer to download."""
        if not path:
            raise gr.Error(f"There is no {kind} to save yet")
        try:
            if kind == "meeting":
                transcript = Path(path)
                return str(save_meeting_pdf(transcript) if pdf else save_meeting_html(transcript))
            directory = Path(path)
            if not (directory / "report.json").is_file():
                raise gr.Error("The project has no report yet: it has one once a run of it ends")
            return str(
                save_project_pdf(directory, meetings=True) if pdf else save_project_html(directory, meetings=True)
            )
        except PDFExportError as error:
            raise gr.Error(str(error)) from None

    # The history

    def history(self, kind: str) -> tuple[list[list[str]], list[str]]:
        entries = [entry for entry in self.workspace.entries() if kind == "all" or entry.kind == kind]
        running = [run.directory for run in self.lab.runs.values() if run.status == "running"]
        return render.history_rows(entries, running), [str(entry.path) for entry in entries]

    # Settings

    def save_settings(
        self,
        model: str,
        stream: bool,
        budget: float | None,
        code: str,
        sandbox: str,
        network: bool,
        python: str,
        mcp_config: str,
    ) -> None:
        settings = Settings(
            model=(model or "").strip() or Settings().model,
            stream=bool(stream),
            max_cost=budget_of(budget),
            code=code,
            sandbox=sandbox,
            network=bool(network),
            python=(python or "").strip(),
            mcp_config=(mcp_config or "").strip(),
        )
        self.workspace.save_settings(settings)
        if problem := library.mcp_choices(settings.mcp_config)[1]:
            gr.Warning(problem)
        gr.Info("Saved. New meetings and projects start with these.")


def lines(text: str | None) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def budget_of(budget: float | None) -> float | None:
    if budget is None or budget == "":
        return None
    budget = float(budget)
    if budget < 0:
        raise gr.Error("A budget cannot be less than nothing")
    return budget


def keys_html() -> str:
    rows = []
    for status in library.provider_statuses():
        state = (
            '<span class="vl-badge ok">ready</span>'
            if status.ready
            else f'<span class="vl-badge neutral">needs {render.escape(", ".join(status.missing))}</span>'
        )
        rows.append(f"<div>{render.escape(status.provider)}</div><div>{state}</div>")
    return f'<div class="vl-keys">{"".join(rows)}</div>'


def shown_path(path: Path) -> str:
    """A path as a person would write it, with their home directory as ~."""
    try:
        return f"~/{path.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return str(path)


def header_html(workspace: Workspace) -> str:
    where = render.escape(str(workspace.root))
    return (
        '<div id="vl-header"><div class="vl-mark">VL</div><div><div class="vl-name">Virtual Lab</div>'
        f'<div class="vl-tagline" title="{where}">A lab of AI scientists, working in '
        f"{render.escape(shown_path(workspace.root))}</div></div></div>"
    )


class RunPanel:
    """What the page shows of the latest run of one kind, kept up to date by the timer."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.shown = 0

    def build_feed(self) -> None:
        hero = MEETING_HERO if self.kind == "meeting" else PROJECT_HERO
        self.feed = gr.HTML(hero, elem_id=f"vl-{self.kind}-feed", elem_classes=["vl-panel"])

    def build_decision(self) -> None:
        """A project's next step as the team lead decided it, to approve, change, or stop at."""
        with gr.Group(visible=False) as self.decision_group:
            self.decision_html = gr.HTML()
            self.agenda = gr.Textbox(label="Agenda", lines=4)
            self.questions = gr.Textbox(label="Questions, one a line", lines=3)
            self.participants = gr.Textbox(label="Who takes part, separated by commas")
            self.answer = gr.Textbox(label="Answer", lines=5)
            with gr.Row():
                self.approve = gr.Button("Approve", variant="primary", size="sm")
                self.change = gr.Button("Approve my changes", size="sm")
                self.halt = gr.Button("Stop the project", variant="stop", size="sm")
        self.autonomous = gr.Checkbox(value=False, label="Let it run on its own from here")

    def build_board(self) -> None:
        self.board = gr.HTML(elem_classes=["vl-panel"])

    def build_rail(self) -> None:
        kind = self.kind
        self.status = gr.HTML(f'<div class="vl-empty">No {self.kind} has been started yet.</div>')
        with gr.Group(visible=False) as self.ask_group:
            self.ask_html = gr.HTML()
            self.ask_form = gr.Textbox(label="Your answer, a field a line, as name: value", lines=3, visible=False)
            with gr.Row():
                self.ask_yes = gr.Button("Approve", variant="primary", size="sm")
                self.ask_no = gr.Button("Decline", variant="stop", size="sm")
        self.note = gr.Textbox(
            label="A note for the next agent to speak",
            lines=3,
            placeholder="Point the discussion somewhere, correct a mistake, or add what you know.",
            interactive=False,
        )
        self.send = gr.Button("Add the note", size="sm", interactive=False)
        with gr.Row():
            self.pause = gr.Button("Pause", size="sm", interactive=False)
            if kind == "project":
                self.stop_after = gr.Button("Stop after this step", size="sm", interactive=False)
            self.stop = gr.Button(
                "Stop now" if kind == "project" else "Stop", variant="stop", size="sm", interactive=False
            )
        with gr.Group(visible=False) as self.done_group:
            with gr.Row():
                self.save_html = gr.Button("Save as HTML", size="sm")
                self.save_pdf = gr.Button("Save as PDF", size="sm")
            self.file = gr.File(label="Saved", visible=False, interactive=False)

    def outputs(self) -> list[Any]:
        outputs = [
            self.feed,
            self.status,
            self.ask_group,
            self.ask_html,
            self.ask_form,
            self.ask_yes,
            self.ask_no,
            self.note,
            self.send,
            self.pause,
            self.stop,
            self.done_group,
        ]
        if self.kind == "project":
            outputs += [
                self.stop_after,
                self.board,
                self.decision_group,
                self.decision_html,
                self.agenda,
                self.questions,
                self.participants,
                self.answer,
                self.autonomous,
            ]
        return outputs

    def show(self) -> None:
        """Called when the tab of this panel is opened, so its parts are laid out again."""
        self.shown += 1

    def update(self, lab: Lab, seen: dict[str, Any]) -> dict[Any, Any]:
        """What changed since the page was last updated, and what the page has now seen."""
        run = lab.latest(self.kind)  # type: ignore[arg-type]
        changes: dict[Any, Any] = {}
        if run is None:
            return changes
        view = run.view()
        kind = self.kind
        fresh = seen.get(f"{kind}_run") != view.id
        # A tab that has not been opened yet drops the updates that hide its parts, so what each part
        # should look like is sent again when the tab is shown
        restyle = seen.get(f"{kind}_shown") != self.shown
        running = view.status == "running"

        # The rail's clock moves while the run goes on, even when nothing else does
        if running or fresh or restyle or seen.get(f"{kind}_version") != view.version:
            changes[self.status] = render.status_html(view)
        if not (fresh or restyle) and seen.get(f"{kind}_version") == view.version:
            return changes
        seen[f"{kind}_run"] = view.id
        seen[f"{kind}_version"] = view.version
        seen[f"{kind}_shown"] = self.shown

        if kind == "meeting":
            feed = render.meeting_feed(view.events, live=running)  # type: ignore[arg-type]
        else:
            feed = render.project_feed(view)
            changes[self.board] = render.project_board(view)
        changes[self.feed] = feed or '<div class="vl-feed-empty">Starting…</div>'

        if fresh or restyle or seen.get(f"{kind}_status") != view.status:
            seen[f"{kind}_status"] = view.status
            changes[self.note] = gr.update(interactive=running)
            changes[self.send] = gr.update(interactive=running)
            changes[self.stop] = gr.update(interactive=running)
            changes[self.done_group] = gr.update(visible=not running and view.result is not None)
            if kind == "project":
                changes[self.stop_after] = gr.update(interactive=running)
        if fresh or restyle or seen.get(f"{kind}_paused") != (view.paused, running):
            seen[f"{kind}_paused"] = (view.paused, running)
            changes[self.pause] = gr.update(value="Resume" if view.paused else "Pause", interactive=running)

        asks = [item for item in view.waiting if item.kind != "decision"]
        ask = asks[0] if asks else None
        ask_id = ask.id if ask else None
        asked = fresh or seen.get(f"{kind}_ask") != ask_id
        if asked or restyle:
            seen[f"{kind}_ask"] = ask_id
            changes[self.ask_group] = gr.update(visible=ask is not None)
            if ask is not None:
                changes[self.ask_html] = render.waiting_html(ask)
                fields = getattr(ask.subject, "fields", None) or {}
                url = getattr(ask.subject, "url", None)
                # What the person has typed in is kept when only the layout is sent again
                typed = {"value": "\n".join(f"{name}: " for name in fields)} if asked else {}
                changes[self.ask_form] = gr.update(visible=bool(fields), **typed)
                label = "Approve" if ask.kind == "approval" else "I have done it" if url else "Send"
                changes[self.ask_yes] = gr.update(value=label)

        if kind == "project":
            decisions = [item for item in view.waiting if item.kind == "decision"]
            decision = decisions[0] if decisions else None
            decision_id = decision.id if decision else None
            decided = fresh or seen.get("project_decision") != decision_id
            if decided or restyle:
                seen["project_decision"] = decision_id
                changes[self.decision_group] = gr.update(visible=decision is not None)
                if decision is not None:
                    step = decision.subject
                    assert isinstance(step, NextStep)
                    changes[self.decision_html] = render.waiting_html(decision)
                    meeting = step.action in ("team_meeting", "individual_meeting", "write_code")
                    form = {
                        self.agenda: (step.agenda, meeting),
                        self.questions: ("\n".join(step.agenda_questions), meeting),
                        self.participants: (", ".join(step.participants), meeting),
                        self.answer: (step.answer, step.action == "finish"),
                    }
                    for field, (value, visible) in form.items():
                        # The fields hold what the person has edited, so only a new decision fills them
                        changes[field] = (
                            gr.update(value=value, visible=visible) if decided else gr.update(visible=visible)
                        )
            if fresh:
                changes[self.autonomous] = gr.update(value=view.autonomous)

        return changes


def build_app(lab: Lab) -> gr.Blocks:
    """The page, over the runs of lab and its workspace."""
    ui = Interface(lab)
    settings = ui.settings()
    titles = ui.scientist_titles()
    models = library.models()
    if settings.model not in models:
        models.insert(0, settings.model)
    tools = library.tool_choices()
    meeting = RunPanel("meeting")
    project = RunPanel("project")

    with gr.Blocks(title="Virtual Lab") as app:
        seen = gr.State({})
        with gr.Row(equal_height=False, elem_id="vl-header-row"):
            gr.HTML(header_html(ui.workspace))
            theme_button = gr.Button("Light / dark", size="sm", elem_id="vl-theme-toggle", scale=0)

        with gr.Tabs() as tabs:
            # The meeting room
            with gr.Tab("Meeting room", id="meeting") as meeting_tab:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        m_type = gr.Radio(
                            [("Team meeting", "team"), ("Individual meeting", "individual")],
                            value="team",
                            label="Kind of meeting",
                        )
                        m_agenda = gr.Textbox(
                            label="Agenda",
                            lines=5,
                            placeholder="What should the meeting work out? Give the context the agents need.",
                        )
                        m_questions = gr.Textbox(label="Questions to answer, one a line", lines=3)
                        m_lead = gr.Dropdown(titles, value="Principal Investigator", label="Team lead")
                        m_members = gr.Dropdown(titles, value=DEFAULT_MEMBERS, multiselect=True, label="Team members")
                        m_critic = gr.Dropdown(titles, value="Scientific Critic", label="Critic", visible=False)
                        m_rounds = gr.Slider(0, 6, value=2, step=1, label="Rounds of discussion")
                        with gr.Accordion("Model, budget, and tools", open=False):
                            m_model = gr.Dropdown(
                                models, value=settings.model, allow_custom_value=True, label="Model for every agent"
                            )
                            m_budget = gr.Number(
                                value=settings.max_cost, label="Budget in USD (empty for none)", minimum=0
                            )
                            m_temperature = gr.Slider(
                                0, 1, value=CONSISTENT_TEMPERATURE, step=0.05, label="Temperature"
                            )
                            m_code = gr.Radio(CODE_CHOICES, value=settings.code, label="Where the agents' code runs")
                            m_resources = gr.Radio(
                                RESOURCE_CHOICES,
                                value="retrieve",
                                label=RESOURCES_LABEL,
                                info=RESOURCES_INFO,
                            )
                            m_summaries = gr.Dropdown(
                                ui.summary_choices(), multiselect=True, label="Build on earlier meetings"
                            )
                            m_rules = gr.Textbox(label="Rules, one a line", lines=2)
                            m_stream = gr.Checkbox(value=settings.stream, label="Show each reply word by word")
                            m_tools = gr.CheckboxGroup(tools, label="Databases the agents can search")
                            m_servers = gr.CheckboxGroup(ui.mcp_choices(), label="MCP servers the agents can use")
                        with gr.Accordion("Add a scientist of your own", open=False):
                            s_title = gr.Textbox(label="Title", placeholder="Virologist")
                            s_expertise = gr.Textbox(label="Expertise", placeholder="the evolution of SARS-CoV-2")
                            s_goal = gr.Textbox(label="Goal", placeholder="predict which variants will escape")
                            s_role = gr.Textbox(label="Role", placeholder="judge designs against new variants")
                            s_save = gr.Button("Add to the lab", size="sm")
                        m_start = gr.Button("Start the meeting", variant="primary")
                    with gr.Column(scale=7, min_width=420):
                        meeting.build_feed()
                    with gr.Column(scale=3, min_width=280):
                        meeting.build_rail()
                        m_follow = gr.Button("Follow up in a new meeting", size="sm", visible=True)

            # The project room
            with gr.Tab("Project", id="project") as project_tab:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        p_goal = gr.Textbox(
                            label="Goal",
                            lines=5,
                            placeholder="What should the project achieve? Say what an answer has to show.",
                        )
                        p_autonomous = gr.Radio(
                            [("Ask me before each step", False), ("Let it run on its own", True)],
                            value=False,
                            label="Each step the team lead decides",
                        )
                        p_choose = gr.Checkbox(value=False, label="Choose the team myself")
                        p_team = gr.Dropdown(titles, value=[], multiselect=True, label="Team", visible=False)
                        p_size = gr.Slider(1, 8, value=3, step=1, label="Largest team the team lead may choose")
                        p_rounds = gr.Slider(1, 40, value=10, step=1, label="Most rounds, each one step")
                        with gr.Accordion("Model, budget, and tools", open=False):
                            p_model = gr.Dropdown(
                                models, value=settings.model, allow_custom_value=True, label="Model for every agent"
                            )
                            p_budget = gr.Number(
                                value=settings.max_cost, label="Budget in USD (empty for none)", minimum=0
                            )
                            p_meeting_rounds = gr.Slider(
                                0, 4, value=1, step=1, label="Rounds of discussion in each meeting"
                            )
                            p_stalled = gr.Slider(
                                1, 10, value=3, step=1, label="Rounds without progress before it stops"
                            )
                            p_memory = gr.Radio(
                                MEMORY_CHOICES, value="pick", label="What each step is given of the work before it"
                            )
                            p_code = gr.Radio(CODE_CHOICES, value=settings.code, label="Where the agents' code runs")
                            p_resources = gr.Radio(
                                RESOURCE_CHOICES,
                                value="retrieve",
                                label=RESOURCES_LABEL,
                                info=RESOURCES_INFO,
                            )
                            p_stream = gr.Checkbox(value=settings.stream, label="Show each reply word by word")
                            p_tools = gr.CheckboxGroup(tools, label="Databases the agents can search")
                            p_servers = gr.CheckboxGroup(ui.mcp_choices(), label="MCP servers the agents can use")
                        p_start = gr.Button("Start the project", variant="primary")
                    with gr.Column(scale=6, min_width=420):
                        project.build_feed()
                    with gr.Column(scale=4, min_width=300):
                        project.build_decision()
                        project.build_rail()
                        project.build_board()

            # The history
            with gr.Tab("History", id="history") as history_tab:
                h_paths = gr.State([])
                h_chosen = gr.State("")
                with gr.Row():
                    h_kind = gr.Radio(
                        [("Everything", "all"), ("Meetings", "meeting"), ("Projects", "project")],
                        value="all",
                        show_label=False,
                        scale=4,
                    )
                    h_refresh = gr.Button("Refresh", size="sm", scale=0)
                h_table = gr.Dataframe(
                    headers=["Started", "Kind", "About", "Status", "Cost"],
                    interactive=False,
                    wrap=True,
                    column_widths=["14%", "9%", "55%", "12%", "10%"],
                    max_height=320,
                    elem_id="vl-history-table",
                )
                with gr.Row(visible=False) as h_actions:
                    h_html = gr.Button("Save as HTML", size="sm")
                    h_pdf = gr.Button("Save as PDF", size="sm")
                    h_follow = gr.Button("Follow up in a new meeting", size="sm")
                    h_budget = gr.Number(value=settings.max_cost, label="New budget (USD)", minimum=0, visible=False)
                    h_rounds = gr.Slider(1, 60, value=20, step=1, label="New most rounds", visible=False)
                    h_carry = gr.Button("Carry the project on", variant="primary", size="sm", visible=False)
                h_file = gr.File(label="Saved", visible=False, interactive=False)
                h_document = gr.HTML()

            # Settings
            with gr.Tab("Settings", id="settings"):
                with gr.Row(equal_height=False):
                    with gr.Column():
                        gr.Markdown("### Look")
                        t_theme = gr.Radio(
                            [
                                ("As the system is", "system"),
                                ("Light: the lab notebook", "light"),
                                ("Dark: the control room", "dark"),
                            ],
                            value="system",
                            label="Theme",
                        )
                        gr.Markdown("### What new meetings and projects start with")
                        g_model = gr.Dropdown(models, value=settings.model, allow_custom_value=True, label="Model")
                        g_budget = gr.Number(value=settings.max_cost, label="Budget in USD (empty for none)", minimum=0)
                        g_stream = gr.Checkbox(value=settings.stream, label="Show each reply word by word")
                        g_code = gr.Radio(CODE_CHOICES, value=settings.code, label="Where the agents' code runs")
                        g_sandbox = gr.Radio(
                            SANDBOX_CHOICES, value=settings.sandbox, label="The container code runs in, with Docker"
                        )
                        g_network = gr.Checkbox(
                            value=settings.network, label="Code in a container can reach the network"
                        )
                        g_python = gr.Textbox(
                            value=settings.python,
                            label="The Python code runs with on this machine (empty for this one)",
                        )
                        g_mcp = gr.Textbox(
                            value=settings.mcp_config, label="A file of MCP servers, as connect_mcp reads it"
                        )
                        g_save = gr.Button("Save", variant="primary")
                    with gr.Column():
                        gr.Markdown("### Model providers")
                        k_status = gr.HTML(keys_html())
                        gr.Markdown(
                            "A key set here is used until the interface stops, and is not saved or shown. "
                            "To keep one, put it in a .env file or the environment."
                        )
                        k_name = gr.Dropdown(list(library.SETTABLE_KEYS), value=library.SETTABLE_KEYS[0], label="Key")
                        k_value = gr.Textbox(label="Value", type="password")
                        k_set = gr.Button("Use this key", size="sm")
                        gr.Markdown(f"### Workspace\n\nEverything run here is saved in `{ui.workspace.root}`.")

        timer = gr.Timer(TICK_SECONDS)

        # The theme
        theme_button.click(None, js="() => window.vlToggleTheme()", outputs=t_theme)
        t_theme.input(None, inputs=t_theme, js="(mode) => window.vlSetTheme(mode)")
        app.load(None, js="() => window.vlTheme ? window.vlTheme() : 'system'", outputs=t_theme)

        # The meeting room
        m_type.change(
            lambda kind: (
                gr.update(
                    label="Team lead" if kind == "team" else "Who works on it",
                    value="Principal Investigator" if kind == "team" else "Immunologist",
                ),
                gr.update(visible=kind == "team"),
                gr.update(visible=kind == "individual"),
                gr.update(label="Rounds of discussion" if kind == "team" else "Rounds of critique"),
            ),
            inputs=m_type,
            outputs=[m_lead, m_members, m_critic, m_rounds],
        )
        meeting_inputs = [
            m_type,
            m_agenda,
            m_questions,
            m_rules,
            m_lead,
            m_members,
            m_critic,
            m_rounds,
            m_model,
            m_budget,
            m_temperature,
            m_tools,
            m_servers,
            m_code,
            m_resources,
            m_summaries,
            m_stream,
        ]
        m_start.click(ui.start_meeting, inputs=meeting_inputs, outputs=[])

        def add_scientist(title: str, expertise: str, goal: str, role: str) -> Any:
            try:
                ui.workspace.save_scientist({"title": title, "expertise": expertise, "goal": goal, "role": role})
            except ValueError as error:
                raise gr.Error(str(error)) from None
            gr.Info(f"{title.strip()} joined the lab.")
            choices = ui.scientist_titles()
            return [gr.update(choices=choices)] * 4 + [""] * 4

        s_save.click(
            add_scientist,
            inputs=[s_title, s_expertise, s_goal, s_role],
            outputs=[m_lead, m_members, m_critic, p_team, s_title, s_expertise, s_goal, s_role],
        )

        def follow_up() -> Any:
            run = lab.latest("meeting")
            if run is None or run.view().result is None:
                raise gr.Error("Follow up on a meeting once it has finished")
            return gr.update(choices=ui.summary_choices(), value=[str(run.view().result)]), ""

        m_follow.click(follow_up, outputs=[m_summaries, m_agenda])

        # The project room
        p_choose.change(
            lambda choose: (gr.update(visible=choose), gr.update(visible=not choose)),
            inputs=p_choose,
            outputs=[p_team, p_size],
        )
        project_inputs = [
            p_goal,
            p_autonomous,
            p_choose,
            p_team,
            p_size,
            p_rounds,
            p_stalled,
            p_meeting_rounds,
            p_memory,
            p_model,
            p_budget,
            p_tools,
            p_servers,
            p_code,
            p_resources,
            p_stream,
        ]
        p_start.click(ui.start_project, inputs=project_inputs, outputs=[])
        project.autonomous.input(ui.set_autonomous, inputs=project.autonomous)
        project.approve.click(lambda seen: ui.decide(seen, "approve", "", "", "", ""), inputs=seen)
        project.change.click(
            lambda seen, *fields: ui.decide(seen, "change", *fields),
            inputs=[seen, project.agenda, project.questions, project.participants, project.answer],
        )
        project.halt.click(lambda seen: ui.decide(seen, "stop", "", "", "", ""), inputs=seen)
        project.stop_after.click(lambda: ui.stop("project", now=False))

        # Steering either
        for panel in (meeting, project):
            kind = panel.kind
            panel.send.click(lambda note, kind=kind: ui.add_note(kind, note), inputs=panel.note, outputs=panel.note)
            panel.note.submit(lambda note, kind=kind: ui.add_note(kind, note), inputs=panel.note, outputs=panel.note)
            panel.pause.click(lambda kind=kind: ui.toggle_pause(kind))
            panel.stop.click(lambda kind=kind: ui.stop(kind, now=True))
            panel.ask_yes.click(
                lambda seen, form, kind=kind: ui.answer_ask(kind, seen, True, form),
                inputs=[seen, panel.ask_form],
                outputs=panel.ask_form,
            )
            panel.ask_no.click(
                lambda seen, form, kind=kind: ui.answer_ask(kind, seen, False, form),
                inputs=[seen, panel.ask_form],
                outputs=panel.ask_form,
            )

            def saved(pdf: bool, kind: str = kind) -> Any:
                run = lab.latest(kind)  # type: ignore[arg-type]
                result = run.view().result if run is not None else None
                return gr.update(value=ui.export(kind, str(result) if result else None, pdf), visible=True)

            panel.save_html.click(lambda kind=kind, saved=saved: saved(False), outputs=panel.file)
            panel.save_pdf.click(lambda kind=kind, saved=saved: saved(True), outputs=panel.file)

        # Keeping the page up to date
        outputs = [*meeting.outputs(), *project.outputs(), seen]

        def tick(state: dict[str, Any]) -> dict[Any, Any]:
            state = dict(state or {})
            changes = {**meeting.update(lab, state), **project.update(lab, state)}
            changes[seen] = state
            return changes

        timer.tick(tick, inputs=seen, outputs=outputs, show_progress="hidden")
        for panel, tab in ((meeting, meeting_tab), (project, project_tab)):
            tab.select(panel.show, show_progress="hidden")

        # The history
        def show_history(kind: str) -> Any:
            rows, paths = ui.history(kind)
            return rows, paths

        def choose(paths: list[str], event: gr.SelectData) -> Any:
            row = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
            if row is None or row >= len(paths):
                raise gr.Error("Choose a row")
            path = paths[row]
            entry = ui.workspace.find(path)
            if entry is None:
                raise gr.Error("It is no longer in the workspace")
            is_project = entry.kind == "project"
            carry = is_project and isinstance(read_setup(entry.path), ProjectSetup) and entry.status != "finished"
            return (
                path,
                render.framed(render.entry_document(entry)),
                gr.update(visible=True),
                gr.update(visible=not is_project and entry.status == "completed"),
                gr.update(visible=carry),
                gr.update(visible=carry),
                gr.update(visible=carry),
                gr.update(visible=False),
            )

        for trigger in (h_refresh.click, h_kind.change, history_tab.select):
            trigger(show_history, inputs=h_kind, outputs=[h_table, h_paths])
        h_table.select(
            choose,
            inputs=h_paths,
            outputs=[h_chosen, h_document, h_actions, h_follow, h_budget, h_rounds, h_carry, h_file],
        )

        def export_chosen(path: str, pdf: bool) -> Any:
            entry = ui.workspace.find(path) if path else None
            if entry is None:
                raise gr.Error("Choose a meeting or project first")
            target = entry.transcript if entry.kind == "meeting" else entry.path
            return gr.update(value=ui.export(entry.kind, str(target) if target else None, pdf), visible=True)

        h_html.click(lambda path: export_chosen(path, False), inputs=h_chosen, outputs=h_file)
        h_pdf.click(lambda path: export_chosen(path, True), inputs=h_chosen, outputs=h_file)

        def follow_from_history(path: str) -> Any:
            entry = ui.workspace.find(path) if path else None
            if entry is None or entry.transcript is None:
                raise gr.Error("Choose a meeting that finished")
            return gr.update(choices=ui.summary_choices(), value=[str(entry.transcript)]), gr.Tabs(selected="meeting")

        h_follow.click(follow_from_history, inputs=h_chosen, outputs=[m_summaries, tabs])
        h_carry.click(ui.carry_on, inputs=[h_chosen, h_budget, h_rounds], outputs=tabs)

        # Settings
        def save_settings(*values: Any) -> Any:
            ui.save_settings(*values)
            settings = ui.settings()
            servers = ui.mcp_choices()
            return (
                gr.update(value=settings.model),
                gr.update(value=settings.model),
                gr.update(value=settings.max_cost),
                gr.update(value=settings.max_cost),
                gr.update(value=settings.code),
                gr.update(value=settings.code),
                gr.update(value=settings.stream),
                gr.update(value=settings.stream),
                gr.update(choices=servers, value=[]),
                gr.update(choices=servers, value=[]),
            )

        g_save.click(
            save_settings,
            inputs=[g_model, g_stream, g_budget, g_code, g_sandbox, g_network, g_python, g_mcp],
            outputs=[m_model, p_model, m_budget, p_budget, m_code, p_code, m_stream, p_stream, m_servers, p_servers],
        )

        def set_key(name: str, value: str) -> Any:
            if name not in library.SETTABLE_KEYS:
                raise gr.Error("Choose one of the keys listed")
            if not (value or "").strip():
                raise gr.Error("Paste the key first")
            os.environ[name] = value.strip()
            gr.Info(f"{name} is set until the interface stops.")
            return "", keys_html()

        k_set.click(set_key, inputs=[k_name, k_value], outputs=[k_value, k_status])

    return app


def launch_ui(
    workspace: Path | str = DEFAULT_WORKSPACE,
    host: str = "127.0.0.1",
    port: int | None = None,
    password: str | None = None,
    share: bool = False,
    open_browser: bool = True,
    client: Any = None,
    chat_models: Any = None,
    block: bool = True,
) -> gr.Blocks:
    """Starts the web interface.

    Anyone who can reach it can spend on your API keys and, if code runs, run code on your
    machine or in its containers, so a password is required to share it or to listen anywhere
    but this machine.

    :param workspace: Where meetings, projects, and settings are kept.
    :param host: The address to listen on; 127.0.0.1 for this machine alone.
    :param port: The port, or None for Gradio's choice, from 7860.
    :param password: The password to sign in with, with any user name, or None for none.
    :param share: Whether to make a public link through Gradio's share servers, for 72 hours.
    :param open_browser: Whether to open the page in a browser.
    :param client: An OpenAI client for every run, as hold_meeting takes one.
    :param chat_models: Chat models for every run, as hold_meeting takes them.
    :param block: Whether to wait until the interface is stopped.
    :raises ValueError: If it would be shared, or listen beyond this machine, with no password.
    :return: The interface.
    """
    if (share or not is_loopback(host)) and not password:
        raise ValueError(
            "Sharing the interface, or listening beyond this machine, lets whoever reaches it spend on your keys and "
            "run code: give a password"
        )
    lab = Lab(Workspace(Path(workspace)), client=client, chat_models=chat_models)
    app = build_app(lab)
    auth = None
    if password:
        expected = password

        def auth(username: str, given: str) -> bool:
            return secrets.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))

    app.queue(default_concurrency_limit=None)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser,
        auth=auth,
        auth_message="Sign in with the interface's password, and any name." if password else None,
        prevent_thread_lock=not block,
        allowed_paths=[str(lab.workspace.root)],
        footer_links=[],
        theme=gradio_theme(),
        css=CSS,
        js=JS,
    )

    return app


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> None:
    """virtual-lab-ui: starts the web interface."""
    parser = argparse.ArgumentParser(prog="virtual-lab-ui", description="The Virtual Lab's web interface.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="where everything run is kept")
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on (default: this machine alone)")
    parser.add_argument("--port", type=int, default=None, help="the port (default: the first free from 7860)")
    parser.add_argument(
        "--password",
        default=os.environ.get("VIRTUAL_LAB_UI_PASSWORD"),
        help="a password to sign in with (default: $VIRTUAL_LAB_UI_PASSWORD); needed with --share or another host",
    )
    parser.add_argument("--share", action="store_true", help="make a public link through Gradio, for 72 hours")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in a browser")
    options = parser.parse_args(argv)

    try:
        launch_ui(
            workspace=options.workspace,
            host=options.host,
            port=options.port,
            password=options.password,
            share=options.share,
            open_browser=not options.no_browser,
        )
    except ValueError as error:
        parser.error(str(error))
