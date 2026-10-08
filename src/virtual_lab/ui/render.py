"""What the web interface shows of a run, as HTML: the discussion as it happens, a project's
plan and rounds, where the run stands, what it waits on the person for, and the history.

Everything an agent, a tool, or a server said is escaped, or rendered from Markdown with any
HTML in it shown as text, as the exported documents are, so nothing said in a meeting can run
in the page. Figures are read from the session's directory and carried in the page itself.
"""

import base64
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from virtual_lab.actions import find_code_action
from virtual_lab.approval import ApprovalRequest, ServerQuestion, choices, field_kind
from virtual_lab.constants import HUMAN_SPEAKER
from virtual_lab.events import MeetingEvent, ProjectEvent
from virtual_lab.export import (
    AGENT_COLOURS,
    IMAGE_TYPES,
    duration,
    escape,
    humanize,
    markdown as export_markdown,
    meeting_html,
    money,
    project_html,
    shorten,
    without_observation,
)
from virtual_lab.planning import NextStep
from virtual_lab.ui.runs import RunView, Waiting
from virtual_lab.ui.workspace import Entry

# The most of a tool's or the session's output shown for one message, half from each end
MAX_SHOWN_OUTPUT_CHARS = 4_000

# The largest figure carried in the page
MAX_FIGURE_BYTES = 8_000_000

PLAN_MARKS = {"done": "done", "dropped": "dropped", "in progress": "progress", "to do": "todo"}

ACTION_NAMES = {
    "team_meeting": "Team meeting",
    "individual_meeting": "Individual meeting",
    "write_code": "Write and run code",
    "change_team": "Change the team",
    "finish": "Finish with an answer",
}

STATUS_NAMES = {
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "stopped": "Stopped",
    "finished": "Finished",
    "out_of_budget": "Out of budget",
    "out_of_rounds": "Out of rounds",
    "stalled": "Stalled",
    "not finished": "Not finished",
    "interrupted": "Interrupted",
}


def markdown(text: str) -> str:
    """Markdown as HTML, with any HTML in it shown as text, and its links opening in a tab of their
    own so that following one does not take the person from the run."""
    return export_markdown(text).replace("<a href=", '<a target="_blank" rel="noopener noreferrer" href=')


def badge(status: str) -> str:
    tone = {
        "completed": "ok",
        "finished": "ok",
        "running": "live",
        "waiting": "ask",
        "paused": "ask",
        "failed": "bad",
        "interrupted": "bad",
    }.get(status, "neutral")
    name = STATUS_NAMES.get(status, status.replace("_", " ").capitalize())
    return f'<span class="vl-badge {tone}">{escape(name)}</span>'


def figure(path: str, number: int, speaker: str | None) -> str:
    """A figure the session drew, carried in the page, or a note that it cannot be shown."""
    file = Path(path)
    kind = IMAGE_TYPES.get(file.suffix.lower())
    try:
        data = file.read_bytes() if kind is not None and file.stat().st_size <= MAX_FIGURE_BYTES else None
    except OSError:
        data = None
    caption = f"Figure {number}" + (f", drawn by code {escape(speaker)} ran" if speaker else "")
    if data is None:
        return f'<figure class="vl-figure"><em>{escape(file.name)} cannot be shown here.</em></figure>'
    uri = f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}"

    return f'<figure class="vl-figure"><img src="{uri}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


class Colours:
    """A colour for every speaker, the team's first, in the order they take part."""

    def __init__(self) -> None:
        self.colours: dict[str, str] = {}

    def __call__(self, title: str | None) -> str:
        if not title:
            return "var(--vl-muted)"
        if title not in self.colours:
            self.colours[title] = AGENT_COLOURS[len(self.colours) % len(AGENT_COLOURS)]
        return self.colours[title]


def speaker_line(title: str, colour: str, model: str | None = None, extra: str = "") -> str:
    model_html = f'<span class="vl-model">{escape(model)}</span>' if model else ""
    return (
        f'<div class="vl-speaker"><span class="vl-dot" style="background:{colour}"></span>'
        f"<span>{escape(title)}</span>{model_html}{extra}</div>"
    )


def collapsible(summary: str, body: str, open_: bool = False, kind: str = "", key: str | None = None) -> str:
    """A fold. With a key, the page keeps it as the person left it, open or shut, when the
    discussion around it is shown again with more in it."""
    key_html = f' data-key="{escape(key)}"' if key else ""
    return (
        f'<details class="vl-fold {kind}"{key_html}{" open" if open_ else ""}><summary>{summary}</summary>'
        f'<div class="vl-fold-body">{body}</div></details>'
    )


def code_html(code: str, language: str) -> str:
    return f'<pre class="vl-code"><code class="language-{escape(language)}">{escape(code)}</code></pre>'


def meeting_feed(events: Sequence[MeetingEvent], live: bool, colours: Colours | None = None) -> str:
    """One meeting's discussion as it happens: the agenda and team, each round, every reply,
    note, call to a tool, and run of code, the reply being written, and how it ended.

    :param events: The meeting's events, in order.
    :param live: Whether the meeting is going on, so that what is under way is shown as such.
    :param colours: The speakers' colours, to share across a project's meetings.
    """
    colours = colours if colours is not None else Colours()
    models: dict[str, str] = {}
    parts: list[str] = []
    num_rounds: int | None = None
    shown_round: int | None = None
    figures = 0
    last = len(events) - 1

    for position, event in enumerate(events):
        latest = live and position == last
        key = f"{event.meeting}:{position}"

        if event.kind == "started":
            data = event.data
            num_rounds = data.get("num_rounds")
            team = list(data.get("team") or [])
            for agent in team:
                colours(agent.get("title"))
                models[agent.get("title", "")] = agent.get("model", "")
            chips = "".join(
                f'<span class="vl-chip" style="--colour:{colours(agent.get("title"))}">'
                f'<span class="vl-dot" style="background:{colours(agent.get("title"))}"></span>'
                f'{escape(agent.get("title", ""))}<span class="vl-model">{escape(agent.get("model", ""))}</span></span>'
                for agent in team
            )
            questions = "".join(f"<li>{markdown(question)}</li>" for question in data.get("agenda_questions") or [])
            kind = "Team meeting" if data.get("meeting_type") == "team" else "Individual meeting"
            rounds = f"{num_rounds} round{'s' if num_rounds != 1 else ''} of discussion" if num_rounds else "one answer"
            tools = data.get("tools") or []
            extras = [rounds]
            if tools:
                extras.append(f"{len(tools)} tool{'s' if len(tools) != 1 else ''}")
            if data.get("session"):
                extras.append("runs code")
            parts.append(
                f'<div class="vl-agenda"><div class="vl-eyebrow">{kind} · {escape(" · ".join(extras))}</div>'
                f'<div class="vl-agenda-text">{markdown(data.get("agenda") or "")}</div>'
                + (f'<ol class="vl-questions">{questions}</ol>' if questions else "")
                + f'<div class="vl-chips">{chips}</div></div>'
            )
            continue

        if event.kind == "resources":
            selected = event.data.get("selected") or {}
            count = (
                sum(len(items) for items in selected.values() if isinstance(items, list))
                if isinstance(selected, dict)
                else 0
            )
            body = code_html(json.dumps(dict(event.data), indent=2, default=str)[:MAX_SHOWN_OUTPUT_CHARS], "json")
            parts.append(
                collapsible(f"Biomni's resources chosen for the agents ({count})", body, kind="vl-quiet", key=key)
            )
            continue

        if event.round is not None and event.round != shown_round and event.kind in ("turn", "message"):
            shown_round = event.round
            if num_rounds is not None and event.round > num_rounds:
                label = "Final round" if num_rounds else "The answer"
            else:
                label = f"Round {event.round}" + (f" of {num_rounds}" if num_rounds else "")
            parts.append(f'<div class="vl-round">{label}</div>')

        if event.kind == "turn":
            if event.speaker:
                models[event.speaker] = str(event.data.get("model") or "")
            if latest:
                parts.append(
                    f'<div class="vl-turn vl-pending" style="--colour:{colours(event.speaker)}">'
                    f"{speaker_line(event.speaker or '', colours(event.speaker), models.get(event.speaker or ''))}"
                    '<div class="vl-thinking"><span></span><span></span><span></span></div></div>'
                )
            continue

        if event.kind == "writing":
            if latest:
                parts.append(
                    f'<div class="vl-turn vl-writing" style="--colour:{colours(event.speaker)}">'
                    f"{speaker_line(event.speaker or '', colours(event.speaker), event.data.get('model'))}"
                    f'<div class="vl-body">{markdown(event.text)}<span class="vl-caret"></span></div></div>'
                )
            continue

        if event.kind == "tool_calls":
            calls = "".join(
                f'<span class="vl-call"><code>{escape(call.get("name"))}</code>'
                f'<span class="vl-args">{escape(shorten(str(call.get("arguments") or ""), 160))}</span></span>'
                for call in event.data.get("calls") or []
            )
            said = f'<div class="vl-body">{markdown(event.text)}</div>' if event.text else ""
            running = '<span class="vl-spinner"></span>' if latest else ""
            parts.append(
                f'<div class="vl-turn vl-tools" style="--colour:{colours(event.speaker)}">'
                + speaker_line(event.speaker or "", colours(event.speaker), extra=' <span class="vl-what">calls</span>')
                + f'{said}<div class="vl-calls">{running}{calls}</div></div>'
            )
            continue

        if event.kind == "code":
            if latest:
                parts.append(
                    f'<div class="vl-turn vl-tools" style="--colour:{colours(event.speaker)}">'
                    f"{speaker_line(event.speaker or '', colours(event.speaker))}"
                    f'<div class="vl-calls"><span class="vl-spinner"></span>Running '
                    f"{escape(event.data.get('language') or 'code')}…</div>"
                    f"{code_html(str(event.data.get('code') or ''), str(event.data.get('language') or ''))}</div>"
                )
            continue

        if event.kind == "cell":
            data = event.data
            status = str(data.get("status") or "")
            took = duration(data.get("duration"))
            summary = (
                f'<span class="vl-badge {"ok" if status == "ok" else "bad"}">{escape(status or "ran")}</span> '
                f"Code ran{f' in {escape(took)}' if took else ''}"
            )
            body = ""
            if data.get("error"):
                body += f'<div class="vl-error-text">{escape(data["error"])}</div>'
            for path in data.get("plot_paths") or []:
                figures += 1
                body += figure(str(path), figures, None)
            if data.get("plot_paths"):
                parts.append(f'<div class="vl-cell">{summary}{body}</div>')
            elif body:
                parts.append(collapsible(summary, body, kind="vl-quiet", key=key))
            continue

        if event.kind == "message":
            parts.append(message_html(event, colours, models, key))
            continue

        if event.kind == "finished":
            usage = event.data.get("usage") or {}
            took = duration(event.data.get("elapsed_time"))
            meta = " · ".join(part for part in (f"cost {money(usage.get('cost'))}", took) if part)
            parts.append(
                f'<div class="vl-summary"><div class="vl-eyebrow">Summary · {escape(meta)}</div>'
                f'<div class="vl-body">{markdown(event.text)}</div></div>'
            )
            continue

        if event.kind == "failed":
            if event.data.get("type") == "RunStopped":
                parts.append('<div class="vl-ended">Stopped by you. What was said until then is saved.</div>')
            else:
                parts.append(
                    '<div class="vl-failed"><strong>The meeting stopped with '
                    f"{escape(event.data.get('type'))}:</strong> "
                    f"{escape(event.text)}</div>"
                )
            continue

        if event.kind == "read_back":
            parts.append(
                collapsible(
                    "Read back from an earlier run of the project, for nothing",
                    f'<div class="vl-body">{markdown(event.text)}</div>',
                    kind="vl-quiet",
                    key=key,
                )
            )

    return "".join(parts)


def message_html(event: MeetingEvent, colours: Colours, models: dict[str, str], key: str | None = None) -> str:
    kind = event.data.get("kind")
    speaker = event.speaker or ""
    text = event.text

    if kind == "prompt":
        return collapsible(
            "What the next speaker was asked", f'<div class="vl-body">{markdown(text)}</div>', kind="vl-prompt", key=key
        )
    if kind == "note" or speaker == HUMAN_SPEAKER:
        return (
            '<div class="vl-turn vl-human"><div class="vl-speaker"><span class="vl-dot vl-you"></span>'
            f'<span>You</span><span class="vl-model">a note to the meeting</span></div>'
            f'<div class="vl-body">{markdown(text)}</div></div>'
        )
    if kind in ("tool_output", "code_output"):
        label = "What the tools returned" if kind == "tool_output" else "What the code printed"
        shown = shorten(without_observation(text) if kind == "code_output" else text, MAX_SHOWN_OUTPUT_CHARS)
        body = markdown(shown) if kind == "tool_output" else f'<pre class="vl-output">{escape(shown)}</pre>'
        return collapsible(label, body, kind="vl-quiet", key=key)
    if kind == "structured_output":
        return collapsible(
            f"{escape(speaker)}'s conclusions, as asked for",
            code_html(text, "json"),
            open_=True,
            kind="vl-quiet",
            key=key,
        )

    colour = colours(speaker)
    body = ""
    if kind == "code_action" and (action := find_code_action(text)) is not None:
        prose = text[: text.find("<execute>")].strip() if "<execute>" in text else ""
        body = (markdown(prose) if prose else "") + collapsible(
            f"Code ({escape(action.language)})", code_html(action.code, action.language), open_=True, key=key
        )
    else:
        body = markdown(text)

    return (
        f'<div class="vl-turn" style="--colour:{colour}">{speaker_line(speaker, colour, models.get(speaker))}'
        f'<div class="vl-body">{body}</div></div>'
    )


# Projects


def grouped(events: Iterable[MeetingEvent | ProjectEvent]) -> list[ProjectEvent | list[MeetingEvent]]:
    """A project's events, with each meeting's run of events together."""
    groups: list[ProjectEvent | list[MeetingEvent]] = []
    for event in events:
        if isinstance(event, ProjectEvent):
            groups.append(event)
        elif groups and isinstance(groups[-1], list) and groups[-1][0].meeting == event.meeting:
            groups[-1].append(event)
        else:
            groups.append([event])

    return groups


def meeting_state(events: Sequence[MeetingEvent]) -> str:
    kinds = {event.kind for event in events}
    if "finished" in kinds or "read_back" in kinds:
        return "completed"
    if "failed" in kinds:
        return "failed"
    return "running"


def project_feed(view: RunView) -> str:
    """A project as it happens: each decision, and each meeting, the one going on open."""
    colours = Colours()
    groups = grouped(view.events)
    live = view.status == "running"
    parts: list[str] = []
    for position, group in enumerate(groups):
        latest = position == len(groups) - 1
        if isinstance(group, ProjectEvent):
            parts.append(project_event_html(group, key=f"project:{position}"))
            continue
        name = group[0].meeting
        state = meeting_state(group)
        summary = f"{badge(state)} <span>{escape(humanize(name))}</span>"
        parts.append(
            collapsible(
                summary,
                meeting_feed(group, live=live and latest, colours=colours),
                open_=latest and state == "running",
                kind="vl-meeting",
                key=f"meeting:{name}",
            )
        )
    if live and not groups:
        parts.append('<div class="vl-empty">The project is starting.</div>')

    return "".join(parts)


def project_event_html(event: ProjectEvent, key: str | None = None) -> str:
    data = event.data
    prefix = f"Round {event.round} · " if event.round else ""
    if event.kind == "team":
        titles = ", ".join(agent.get("title", "") for agent in data.get("team") or [])
        change = data.get("change") or {}
        why = f" {escape(change.get('why'))}" if change.get("why") else ""
        team = f"<strong>{escape(titles or 'none yet')}</strong>"
        return f'<div class="vl-marker">{escape(prefix)}The team: {team}.{why}</div>'
    if event.kind == "plan":
        return f'<div class="vl-marker">The team made a plan of {len(data.get("plan") or [])} tasks.</div>'
    if event.kind == "decided":
        proposed = data.get("proposed") or {}
        return f'<div class="vl-marker vl-decided">{escape(prefix)}The team lead decided: {step_line(proposed)}</div>'
    if event.kind == "code":
        tone = "ok" if data.get("succeeded") else "bad"
        return collapsible(
            f'<span class="vl-badge {tone}">code</span> {escape(prefix)}What came of the code',
            f'<pre class="vl-output">{escape(shorten(event.text, MAX_SHOWN_OUTPUT_CHARS))}</pre>',
            kind="vl-quiet",
            key=key,
        )
    if event.kind == "round":
        round_ = data.get("round") or {}
        note = f": {escape(round_.get('note'))}" if round_.get("note") else ""
        outcome = escape(str(round_.get("outcome") or "").replace("_", " "))
        return f'<div class="vl-marker vl-quiet">{escape(prefix)}ended: {outcome}{note}</div>'
    if event.kind == "finished":
        return f'<div class="vl-ended">{escape(event.text)}</div>'

    return ""


def step_line(step: dict[str, Any]) -> str:
    what = f"<strong>{escape(ACTION_NAMES.get(step.get('action', ''), step.get('action', '')))}</strong>"
    if step.get("participants"):
        what += f" with {escape(', '.join(step['participants']))}"
    if step.get("agenda"):
        what += f": {escape(shorten(step['agenda'], 240))}"

    return what


def latest_plan(events: Sequence[MeetingEvent | ProjectEvent]) -> list[dict[str, str]]:
    for event in reversed(events):
        if isinstance(event, ProjectEvent):
            if event.kind in ("plan", "round") and event.data.get("plan") is not None:
                return list(event.data["plan"])
            if event.kind == "decided":
                return list((event.data.get("proposed") or {}).get("plan") or [])
            if event.kind == "finished":
                return list((event.data.get("report") or {}).get("plan") or [])
    return []


def project_board(view: RunView) -> str:
    """Where a project stands: its team, its plan, its rounds, and its answer once it has one."""
    events = view.events
    team = next(
        (
            list(event.data.get("team") or [])
            for event in reversed(events)
            if isinstance(event, ProjectEvent) and event.kind == "team"
        ),
        [],
    )
    plan = latest_plan(events)
    report = next(
        (
            event.data.get("report")
            for event in reversed(events)
            if isinstance(event, ProjectEvent) and event.kind == "finished"
        ),
        None,
    )
    parts = [f'<div class="vl-eyebrow">Goal</div><div class="vl-goal">{markdown(view.title)}</div>']

    if report:
        status = str(report.get("status"))
        answer = report.get("answer") or report.get("proposed_answer")
        heading = "Answer" if report.get("answer") else "Last answer proposed" if answer else None
        parts.append(
            f'<div class="vl-report"><div class="vl-eyebrow">Report {badge(status)}</div>'
            f"<p>{escape(report.get('reason') or '')}</p>"
            + (f'<div class="vl-label">{heading}</div><div class="vl-body">{markdown(answer)}</div>' if heading else "")
            + (
                '<div class="vl-label">The critic objected</div><ul>'
                + "".join(f"<li>{markdown(item)}</li>" for item in report.get("objections") or [])
                + "</ul>"
                if report.get("objections")
                else ""
            )
            + "</div>"
        )

    if team:
        chips = "".join(f'<span class="vl-chip">{escape(agent.get("title", ""))}</span>' for agent in team)
        parts.append(f'<div class="vl-label">Team</div><div class="vl-chips">{chips}</div>')

    if plan:
        done = sum(task.get("status") == "done" for task in plan)
        items = "".join(
            f'<li class="{PLAN_MARKS.get(task.get("status", ""), "todo")}">{escape(task.get("task", ""))}</li>'
            for task in plan
        )
        parts.append(
            f'<div class="vl-label">Plan · {done} of {len(plan)} done</div>'
            f'{meter(done, len(plan))}<ul class="vl-plan">{items}</ul>'
        )
    else:
        parts.append('<div class="vl-label">Plan</div><div class="vl-empty">Not made yet.</div>')

    rounds = [event for event in events if isinstance(event, ProjectEvent) and event.kind in ("decided", "round")]
    if rounds:
        items = []
        for event in rounds:
            if event.kind == "decided":
                items.append(
                    f'<li><span class="vl-round-number">{event.round}</span>'
                    f"{step_line(event.data.get('proposed') or {})}</li>"
                )
            else:
                outcome = str((event.data.get("round") or {}).get("outcome") or "").replace("_", " ")
                items.append(f'<li class="vl-outcome">→ {escape(outcome)}</li>')
        parts.append(f'<div class="vl-label">Rounds</div><ol class="vl-rounds">{"".join(items)}</ol>')

    return "".join(parts)


def meter(amount: float, limit: float, warn: bool = False) -> str:
    """A bar of how much of a limit is used: green, then amber and red as a budget runs out when warn."""
    share = 0.0 if limit <= 0 else max(0.0, min(1.0, amount / limit))
    tone = ("bad" if share >= 0.9 else "ask" if share >= 0.7 else "ok") if warn else "ok"
    return f'<div class="vl-meter {tone}"><span style="width:{share * 100:.1f}%"></span></div>'


# Where a run stands


def current_meeting(view: RunView) -> list[MeetingEvent]:
    meeting_events = [event for event in view.events if isinstance(event, MeetingEvent)]
    if not meeting_events:
        return []
    name = meeting_events[-1].meeting
    return [event for event in meeting_events if event.meeting == name]


def doing(view: RunView) -> str:
    """What the run is doing now, in a few words."""
    if view.status != "running":
        return {
            "completed": "Finished.",
            "stopped": "Stopped by you.",
            "failed": "Stopped with an error.",
        }.get(view.status, "")
    if view.waiting:
        return "Waiting for you."
    if view.holding and view.next_turn is not None:
        return f"Paused before {view.next_turn.speaker}'s turn."
    if view.holding:
        return "Paused before the next decision."
    if view.paused:
        return "Pausing once the reply being written is done."
    if view.stopping == "now":
        return "Stopping…"

    events = current_meeting(view)
    last = events[-1] if events else None
    if last is not None and meeting_state(events) == "running":
        if last.kind == "writing":
            return f"{last.speaker} is writing."
        if last.kind == "turn":
            return f"{last.speaker} is thinking."
        if last.kind == "tool_calls":
            return f"{last.speaker} is calling tools."
        if last.kind == "code":
            return f"{last.speaker}'s code is running."
        return "The meeting is going on."
    if view.kind == "project":
        return "The team lead is deciding the next step." if view.events else "Choosing the team."

    return "Starting."


def round_progress(view: RunView) -> tuple[int, int] | None:
    """The round of the meeting going on, and how many there are, final round included."""
    events = current_meeting(view)
    started = next((event for event in events if event.kind == "started"), None)
    if started is None:
        return None
    total = int(started.data.get("num_rounds") or 0) + 1
    current = max((event.round or 0 for event in events), default=0)

    return min(current, total), total


def status_html(view: RunView, now: float | None = None) -> str:
    """The rail beside a run: what it is doing, its round, what it has spent, and its notes."""
    now = time.time() if now is None else now
    state = "waiting" if view.waiting else "paused" if view.paused and view.status == "running" else view.status
    elapsed = duration((view.ended_at or now) - view.started_at)
    parts = [
        f'<div class="vl-state">{badge(state)}<span class="vl-elapsed">{escape(elapsed or "")}</span></div>',
        f'<div class="vl-doing">{escape(doing(view))}</div>',
    ]
    if view.error and view.status != "running":
        parts.append(f'<div class="vl-error-text">{escape(view.error)}</div>')

    progress = round_progress(view)
    if progress is not None and view.kind == "meeting":
        current, total = progress
        label = "Final round" if current == total else f"Round {current} of {total - 1}"
        parts.append(f'<div class="vl-label">{label}</div>{meter(current, total)}')

    if view.max_cost is not None:
        spent = view.spent or 0.0
        parts.append(
            f'<div class="vl-label">Spent {money(view.spent)} of {money(view.max_cost)}</div>'
            + meter(spent, view.max_cost, warn=True)
        )
    else:
        parts.append(f'<div class="vl-label">Spent {money(view.spent)}, with no budget</div>')

    if view.notes:
        items = "".join(f"<li>{escape(note)}</li>" for note in view.notes)
        parts.append(f'<div class="vl-label">Notes waiting for the next turn</div><ul class="vl-notes">{items}</ul>')
    if view.stopping == "after_step" and view.status == "running":
        parts.append('<div class="vl-marker">Stopping after the step under way.</div>')

    return "".join(parts)


# What a run waits on the person for


def waiting_html(waiting: Waiting) -> str:
    subject = waiting.subject
    if isinstance(subject, NextStep):
        return decision_html(subject, waiting.round)
    if isinstance(subject, ApprovalRequest):
        arguments = json.dumps(subject.arguments, indent=2, default=str)
        about = f'<div class="vl-body">{markdown(subject.description)}</div>' if subject.description else ""
        return (
            f'<div class="vl-ask"><div class="vl-eyebrow">A tool waits for your approval</div>'
            f"<h3><code>{escape(subject.server_tool)}</code> on {escape(subject.server)}</h3>{about}"
            f'<div class="vl-label">It would be called with</div>{code_html(arguments, "json")}</div>'
        )
    if isinstance(subject, ServerQuestion):
        fields = []
        for name, schema in subject.fields.items():
            kind = field_kind(schema)
            options = choices(schema)
            hint = f"one of {', '.join(map(str, options))}" if options else kind
            required = " (required)" if name in subject.required else ""
            description = f" — {escape(schema.get('description'))}" if schema.get("description") else ""
            fields.append(f"<li><code>{escape(name)}</code>: {escape(hint)}{required}{description}</li>")
        page = f"<p>Answer at {link(subject.url)}, then say it is done.</p>" if subject.url else ""
        form = f'<div class="vl-label">Fields</div><ul>{"".join(fields)}</ul>' if fields else ""
        return (
            f'<div class="vl-ask"><div class="vl-eyebrow">{escape(subject.server)} asks</div>'
            f'<div class="vl-body">{markdown(subject.message)}</div>{page}{form}</div>'
        )

    return ""


def link(url: str) -> str:
    """A link to a page a server sent, which is a link only if it is a web page: a server could send
    one that runs script in this page when it is followed."""
    if url.strip().lower().startswith(("http://", "https://")):
        return f'<a href="{escape(url.strip())}" target="_blank" rel="noopener noreferrer">{escape(url)}</a>'

    return f"<code>{escape(url)}</code> (not a web page, so not linked)"


def decision_html(step: NextStep, round_: int | None) -> str:
    parts = [
        f'<div class="vl-ask"><div class="vl-eyebrow">Round {round_} · The team lead\'s decision waits for you</div>',
        f"<h3>{step_line(step.model_dump())}</h3>",
        f'<div class="vl-label">Why</div><div class="vl-body">{markdown(step.rationale)}</div>',
    ]
    if step.agenda_questions:
        items = "".join(f"<li>{markdown(question)}</li>" for question in step.agenda_questions)
        parts.append(f'<div class="vl-label">Questions</div><ol>{items}</ol>')
    if step.answer:
        parts.append(f'<div class="vl-label">Answer proposed</div><div class="vl-body">{markdown(step.answer)}</div>')
    if step.add_members or step.remove_members:
        joining = ", ".join(member.title for member in step.add_members) or "no one"
        leaving = ", ".join(step.remove_members) or "no one"
        parts.append(f"<p>Joining: {escape(joining)}. Leaving: {escape(leaving)}.</p>")
    parts.append(
        collapsible(
            "What the work has established so far",
            f'<div class="vl-body">{markdown(step.progress)}</div>',
            kind="vl-quiet",
        )
    )
    parts.append("</div>")

    return "".join(parts)


# History


def history_rows(entries: Sequence[Entry], running: Iterable[Path] = ()) -> list[list[str]]:
    """The history as a table's rows: when, what, its title, how it stands, and its cost.

    :param running: The directories of runs going on now, which show as running whatever their
        ledgers say; any other a ledger says is running was interrupted.
    """
    live = {Path(path).resolve() for path in running}
    rows = []
    for entry in entries:
        status = entry.status
        if entry.path.resolve() in live:
            status = "running"
        elif status == "running":
            status = "interrupted"
        rows.append(
            [
                entry.started_at[:16].replace("T", " "),
                entry.kind.capitalize(),
                shorten(" ".join(entry.title.split()), 110),
                STATUS_NAMES.get(status, status.replace("_", " ")),
                money(entry.cost) if entry.cost is not None else "unknown",
            ]
        )

    return rows


def entry_document(entry: Entry) -> str:
    """A meeting or project as a document, as it would be saved, to read in the page."""
    if entry.kind == "meeting" and entry.transcript is not None:
        return meeting_html(entry.transcript)
    try:
        return project_html(entry.path, meetings=True)
    except FileNotFoundError:
        return (
            '<!DOCTYPE html><html><body style="font-family: sans-serif; padding: 24px">'
            f"<h2>{escape(entry.title)}</h2><p>This project has no report yet: it has not finished a run. "
            "Carry it on to finish it.</p></body></html>"
        )


def framed(document: str, height: int = 760) -> str:
    """A document shown in the page in a frame of its own, so that its style stays its own."""
    return (
        f'<iframe class="vl-document" sandbox="" srcdoc="{escape(document)}" '
        f'style="width:100%;height:{height}px;border:0"></iframe>'
    )
