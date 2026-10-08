"""Meetings and projects as documents to read, print, or share: HTML, and PDF through WeasyPrint.

Biomni saves its agent's conversation as a PDF, turning the Markdown it writes into HTML and that
into a PDF with WeasyPrint, with its figures embedded. A meeting is exported the same way, laid
out as a meeting: its agenda and team first, then its summary, then the discussion round by round,
each agent in a colour of its own, with the code that was run, what it printed, and the figures
it drew next to the turn that drew them, and what the meeting cost last. A project is exported
as its report, with its meetings after it if asked for.

Everything is read from what a meeting or project saved, so a document can be made of any meeting
at any time, including one that failed, from what it left under partial/. A transcript saved
before meetings kept records, as the Virtual Lab's original ones were, is exported from the
transcript alone.

Markdown is rendered without the HTML it may hold, and neither the HTML nor the PDF fetches anything:
figures are embedded in the document, and an image an agent linked to on the web is shown as a link.
"""

import base64
import contextlib
import functools
import html
import io
import json
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from virtual_lab.actions import find_code_action
from virtual_lab.constants import (
    METADATA_DIR_NAME,
    PROJECT_FILE_NAME,
    REPORT_FILE_NAME,
    SESSION_LOG_DIR_NAME,
)

# The most of a tool's or a session's output shown for one message, half from each end
MAX_EXPORTED_OUTPUT_CHARS = 6_000

# One colour per agent, in the order they take part, each dark enough to read on paper
AGENT_COLOURS = ("#2f6f8f", "#a5522b", "#4f7d3a", "#7b4f9e", "#b0413e", "#2d7a74", "#8a6d1f", "#5a5f9c")

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}

PDF_INSTALL_HELP = (
    "Making a PDF needs WeasyPrint and the Pango library it draws text with. Install WeasyPrint with "
    "pip install 'virtual-lab[pdf]', and Pango with brew install pango on macOS, or apt install "
    "libpango-1.0-0 libpangoft2-1.0-0 on Debian and Ubuntu; for Windows and the rest, see "
    "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html. The HTML the PDF is made from can be "
    "saved without either, with save_meeting_html or save_project_html."
)


class PDFExportError(RuntimeError):
    """Raised when a PDF cannot be made, as when WeasyPrint or Pango is not installed."""


@dataclass(frozen=True)
class SavedMeeting:
    """A meeting as it was saved.

    :param name: The name it was saved under.
    :param transcript: The transcript, turn by turn.
    :param record: Its record, as saved beside it, or None if it has none.
    :param session_log: The log of the code it ran in its session, or None if it ran none.
    """

    name: str
    transcript: list[dict[str, str]]
    record: dict[str, Any] | None
    session_log: dict[str, Any] | None


def load_meeting(transcript_path: Path) -> SavedMeeting:
    """Reads a meeting back from where it was saved.

    :param transcript_path: The transcript, as hold_meeting saved it, such as results/discussion.json.
    :raises FileNotFoundError: If there is no transcript there.
    :raises ValueError: If the file is not a transcript.
    :return: The meeting.
    """
    transcript_path = Path(transcript_path)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(transcript, list) or not all(
        isinstance(turn, dict) and isinstance(turn.get("agent"), str) and isinstance(turn.get("message"), str)
        for turn in transcript
    ):
        raise ValueError(f"{transcript_path} is not a meeting's transcript")

    name = transcript_path.stem
    record_path = transcript_path.parent / METADATA_DIR_NAME / f"{name}.json"
    log_path = transcript_path.parent / SESSION_LOG_DIR_NAME / f"{name}.json"

    return SavedMeeting(
        name=name,
        transcript=transcript,
        record=json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else None,
        session_log=json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else None,
    )


# Rendering pieces

@functools.cache
def markdown_renderer() -> Any:
    from markdown_it import MarkdownIt

    renderer = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable(["table", "strikethrough"])
    renderer.add_render_rule("image", render_image)
    return renderer


def render_image(self: Any, tokens: list[Any], idx: int, options: Any, env: Any) -> str:
    """An image embedded in the Markdown is shown, and one elsewhere is linked to, so opening the
    document never fetches anything."""
    token = tokens[idx]
    source = str(token.attrGet("src") or "")
    alt = self.renderInlineAsText(token.children or [], options, env)
    if source.startswith("data:"):
        return f'<img src="{escape(source)}" alt="{escape(alt)}">'
    return f'<a href="{escape(source)}">{escape(alt or source)}</a>'


def markdown(text: str) -> str:
    """Markdown as HTML, with any HTML in it shown as text rather than rendered."""
    return markdown_renderer().render(text)


def escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def shorten(text: str, max_chars: int) -> str:
    """Text cut to at most about max_chars, from the middle, saying how much was left out."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    left_out = len(text) - 2 * half

    return f"{text[:half]}\n\n[... {left_out:,} characters left out ...]\n\n{text[-half:]}"


def humanize(name: str) -> str:
    return re.sub(r"[_-]+", " ", name).strip().capitalize() or name


def when(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp

    return moment.strftime("%d %B %Y, %H:%M UTC")


def duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    return f"{hours} h {minutes} min" if hours else f"{minutes} min {rest} s" if minutes else f"{rest} s"


def money(cost: float | None) -> str:
    """A cost in USD to a hundredth of a cent, rounded half up once float noise is trimmed, so that
    a cost worked out two ways, as a model's and as the meeting's, reads the same."""
    if cost is None:
        return "unknown"
    rounded = Decimal(repr(round(cost, 10))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    return f"${rounded:,.4f}"


def image_uri(directory: Path, name: str) -> str | None:
    """A figure saved in a session's directory as a data URI, so that the document carries it, or
    None if it cannot be read or is not in the directory."""
    path = (directory / name).resolve()
    kind = IMAGE_TYPES.get(path.suffix.lower())
    if kind is None or not path.is_relative_to(directory.resolve()) or not path.is_file():
        return None

    return f"data:{kind};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def without_observation(text: str) -> str:
    match = re.fullmatch(r"\s*<observation>\n?(.*?)\n?</observation>(.*)", text, re.DOTALL)
    if match is None:
        return text

    return match.group(1) + (f"\n\n{match.group(2).strip()}" if match.group(2).strip() else "")


def code_block(code: str, language: str) -> str:
    return f'<pre class="code"><code class="language-{escape(language)}">{escape(code)}</code></pre>'


def css() -> str:
    """The document's style: a lab notebook, in print and on screen."""
    return """
@page {
  size: A4;
  margin: 18mm 17mm 20mm 17mm;
  @bottom-left { content: string(title); color: #8b8578;
    font: 8pt "Inter", "Helvetica Neue", "DejaVu Sans", sans-serif; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); color: #8b8578;
    font: 8pt "Inter", "Helvetica Neue", "DejaVu Sans", sans-serif; }
}
html { font-size: 10pt; }
body {
  margin: 0; color: #24221f; background: #fdfcf8;
  font-family: "Charter", "Iowan Old Style", "Georgia", "DejaVu Serif", serif; line-height: 1.5;
}
main { max-width: 820px; margin: 0 auto; padding: 24px; }
@media print { body { background: none; } main { padding: 0; max-width: none; } }
h1, h2, h3, h4, .eyebrow, .speaker, .badge, table, .meta, figcaption, .label {
  font-family: "Inter", "Helvetica Neue", "Helvetica", "DejaVu Sans", sans-serif;
}
h1 { string-set: title content(); font-size: 22pt; line-height: 1.2; margin: 4px 0 10px; letter-spacing: -0.01em; }
h2 { font-size: 13pt; margin: 26px 0 10px; padding-bottom: 4px; border-bottom: 1.5px solid #d9d3c4; }
h3 { font-size: 11pt; margin: 18px 0 6px; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 8pt; color: #8b8578; font-weight: 600; }
.meta { color: #6b665c; font-size: 8.5pt; margin-bottom: 14px; }
.meta span + span::before { content: "  ·  "; color: #b9b2a3; }
.badge { display: inline-block; border-radius: 9px; padding: 1px 8px; font-size: 7.5pt; font-weight: 600; }
.badge.ok { background: #e3efe0; color: #2f5d27; }
.badge.bad { background: #f6e1df; color: #8e2d27; }
.badge.neutral { background: #ece8de; color: #5b564c; }
.callout { border: 1px solid #e3ddcf; border-radius: 6px; background: #f7f4ec; padding: 10px 14px; margin: 10px 0; }
.callout.error { border-color: #e7c3bf; background: #fbefed; }
.team { display: flex; flex-wrap: wrap; gap: 8px; }
.member { flex: 1 1 230px; border: 1px solid #e3ddcf; border-left: 4px solid var(--colour); border-radius: 5px;
  padding: 6px 10px; background: #fffefa; }
.member .name { font-family: "Inter", "Helvetica Neue", "DejaVu Sans", sans-serif; font-weight: 600;
  color: var(--colour); }
.member .about { font-size: 8.5pt; color: #5b564c; }
.round { margin: 26px 0 8px; font-family: "Inter", "Helvetica Neue", "DejaVu Sans", sans-serif; font-size: 8.5pt;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #8b8578; display: flex;
  align-items: center; gap: 10px; }
.round::after { content: ""; flex: 1; border-top: 1px solid #d9d3c4; }
.turn { border-left: 3px solid var(--colour); padding: 2px 0 2px 12px; margin: 12px 0; break-inside: auto; }
.turn .speaker { font-weight: 700; color: var(--colour); font-size: 9.5pt; }
.turn .speaker, .label, .round, h2, h3, figure { break-after: avoid; }
.body h1, .body h2, .body h3, .body h4, .body h5, .body h6 { font-size: 10.5pt; margin: 12px 0 4px; padding: 0;
  border: none; string-set: none; }
.turn .speaker .model { font-weight: 400; color: #8b8578; font-size: 8pt; margin-left: 6px; }
.turn.note { --colour: #b9b2a3; }
.turn.note .body { font-size: 8.5pt; color: #4a463f; }
.label { font-size: 7.5pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #8b8578;
  margin: 6px 0 2px; }
pre, code { font-family: "JetBrains Mono", "SF Mono", "Menlo", "DejaVu Sans Mono", monospace; font-size: 8pt; }
pre { background: #f3efe5; border: 1px solid #e3ddcf; border-radius: 5px; padding: 8px 10px; white-space: pre-wrap;
  word-wrap: break-word; overflow-wrap: anywhere; }
pre.output { background: #fbfaf6; color: #3d3a34; }
:not(pre) > code { background: #f3efe5; border-radius: 3px; padding: 0 3px; }
blockquote { margin: 8px 0; padding: 2px 12px; border-left: 3px solid #d9d3c4; color: #4a463f; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 8px 0; width: 100%; }
th, td { border-bottom: 1px solid #e3ddcf; padding: 4px 8px; text-align: left; vertical-align: top; }
td > :first-child { margin-top: 0; }
td > :last-child { margin-bottom: 0; }
tr, table.usage { break-inside: avoid; }
th { color: #6b665c; font-weight: 600; }
td.number, th.number { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 10px 0; break-inside: avoid; }
figure img { max-width: 100%; max-height: 120mm; border: 1px solid #e3ddcf; border-radius: 4px; background: white; }
.body img { max-width: 100%; max-height: 120mm; }
figcaption { font-size: 8pt; color: #6b665c; margin-top: 3px; }
ul.plan { list-style: none; padding-left: 0; }
ul.plan li::before { display: inline-block; width: 1.6em; font-family: "DejaVu Sans", sans-serif; }
ul.plan li.done::before { content: "\\2611"; color: #2f5d27; }
ul.plan li.dropped::before { content: "\\2612"; color: #8e2d27; }
ul.plan li.dropped { color: #8b8578; text-decoration: line-through; }
ul.plan li.progress::before { content: "\\25D1"; color: #8a6d1f; }
ul.plan li.todo::before { content: "\\2610"; color: #8b8578; }
section.meeting { break-before: page; }
a { color: #2f6f8f; }
"""


def document(title: str, body: str) -> str:
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title>'
        f"<style>{css()}</style></head><body><main>{body}</main></body></html>\n"
    )


def meta_line(parts: Iterable[str | None]) -> str:
    shown = "".join(f"<span>{part}</span>" for part in parts if part)
    return f'<div class="meta">{shown}</div>' if shown else ""


def team_cards(members: list[tuple[dict[str, str], str, str | None]]) -> str:
    cards = []
    for agent, colour, part in members:
        about = escape(agent.get("expertise") or "")
        model = f' <span class="badge neutral">{escape(agent["model"])}</span>' if agent.get("model") else ""
        role = f" ({escape(part)})" if part else ""
        cards.append(
            f'<div class="member" style="--colour: {colour}"><div class="name">{escape(agent.get("title", ""))}{role}'
            f'{model}</div><div class="about">{about}</div></div>'
        )

    return f'<div class="team">{"".join(cards)}</div>'


# Meetings

def kinds_of(meeting: SavedMeeting) -> list[str]:
    """Each turn's kind: the record's, or for a transcript without one, worked out from who spoke."""
    turns = (meeting.record or {}).get("turns") or []
    if len(turns) == len(meeting.transcript):
        return [turn["kind"] for turn in turns]

    by_speaker = {"User": "prompt", "Tool": "tool_output", "Session": "code_output"}
    return [by_speaker.get(turn["agent"], "response") for turn in meeting.transcript]


def meeting_body(meeting: SavedMeeting, include_prompts: bool, max_output_chars: int, heading_level: int = 1) -> str:
    """A meeting as the body of a document, or as a section of a project's."""
    record = meeting.record or {}
    kinds = kinds_of(meeting)
    turns = record.get("turns") if len(record.get("turns") or []) == len(meeting.transcript) else None
    team: list[dict[str, str]] = list(record.get("team") or [])
    critic = record.get("critic")

    # Every agent who spoke gets a colour, those on the team first, in the team's order
    speakers = [agent["title"] for agent in team]
    speakers += [
        turn["agent"]
        for turn, kind in zip(meeting.transcript, kinds)
        if kind in ("response", "code_action", "structured_output") and turn["agent"] not in speakers
    ]
    colours = {title: AGENT_COLOURS[index % len(AGENT_COLOURS)] for index, title in enumerate(dict.fromkeys(speakers))}

    meeting_type = record.get("meeting_type")
    kind_name = {"team": "Team meeting", "individual": "Individual meeting"}.get(str(meeting_type), "Meeting")
    status = record.get("status")
    usage = record.get("usage") or {}
    badge = (
        '<span class="badge ok">completed</span>'
        if status == "completed"
        else f'<span class="badge bad">{escape(status)}</span>'
        if status
        else None
    )
    h = heading_level
    parts = [
        f'<div class="eyebrow">Virtual Lab · {kind_name}</div>',
        f"<h{h}>{escape(humanize(meeting.name))}</h{h}>",
        meta_line(
            [
                badge,
                escape(when(record.get("started_at")) or "") or None,
                escape(duration(record.get("elapsed_seconds")) or "") or None,
                f"cost {money(usage.get('cost'))}" if usage else None,
            ]
        ),
    ]

    if record.get("error"):
        error = record["error"]
        parts.append(
            f'<div class="callout error"><strong>The meeting stopped with {escape(error.get("type"))}:</strong> '
            f"{escape(error.get('message'))}</div>"
        )

    # The agenda, from the record, or from the first prompt of a transcript without one
    sub = min(h + 1, 6)
    if record.get("agenda"):
        parts.append(f"<h{sub}>Agenda</h{sub}>{markdown(record['agenda'])}")
        if record.get("agenda_questions"):
            items = "".join(f"<li>{markdown(question)}</li>" for question in record["agenda_questions"])
            parts.append(f'<div class="label">Questions</div><ol>{items}</ol>')
        if record.get("agenda_rules"):
            items = "".join(f"<li>{markdown(rule)}</li>" for rule in record["agenda_rules"])
            parts.append(f'<div class="label">Rules</div><ul>{items}</ul>')
    elif not include_prompts and meeting.transcript and kinds[0] == "prompt":
        parts.append(f"<h{sub}>Agenda</h{sub}>{markdown(meeting.transcript[0]['message'])}")

    if team:
        lead = team[0]["title"] if meeting_type == "team" else None
        # A critic reviewing on its own, as a project's critic does, is both the team member and
        # the critic, and is shown once
        team = [agent for index, agent in enumerate(team) if agent not in team[:index]]
        critic_title = critic.get("title") if critic else None
        members = [
            (
                agent,
                colours[agent["title"]],
                "team lead" if agent["title"] == lead else "critic" if agent["title"] == critic_title else None,
            )
            for agent in team
        ]
        parts.append(f"<h{sub}>Team</h{sub}>{team_cards(members)}")

    responses = [index for index, kind in enumerate(kinds) if kind == "response"]
    if responses:
        summary = meeting.transcript[responses[-1]]
        colour = colours.get(summary["agent"], AGENT_COLOURS[0])
        # A meeting that stopped never reached its summary; a transcript without a record did
        title = "Summary" if status in (None, "completed") else "Last said before it stopped"
        parts.append(
            f'<h{sub}>{title}</h{sub}><div class="turn" style="--colour: {colour}">'
            f'<div class="speaker">{escape(summary["agent"])}</div>'
            f'<div class="body">{markdown(summary["message"])}</div></div>'
        )

    parts.append(f"<h{sub}>Discussion</h{sub}>")
    parts.append(discussion_html(meeting, kinds, turns, colours, include_prompts, max_output_chars))

    if usage.get("per_model"):
        parts.append(f"<h{sub}>What it used</h{sub}>{usage_table(usage, record.get('prices') or {})}")

    return "\n".join(parts)


def discussion_html(
    meeting: SavedMeeting,
    kinds: list[str],
    turns: list[dict[str, Any]] | None,
    colours: dict[str, str],
    include_prompts: bool,
    max_output_chars: int,
) -> str:
    record = meeting.record or {}
    team_size = len(record.get("team") or [])
    num_rounds = record.get("num_rounds")
    session = record.get("session") or {}
    directory = Path(session["directory"]) if session.get("directory") else None
    figures = 0

    parts: list[str] = []
    responses_so_far = 0
    shown_round: int | None = None

    for index, (turn, kind) in enumerate(zip(meeting.transcript, kinds)):
        speaker, message = turn["agent"], turn["message"]

        if kind == "prompt":
            if include_prompts:
                parts.append(
                    '<div class="turn note"><div class="label">Prompt</div>'
                    f'<div class="body">{markdown(message)}</div></div>'
                )
            continue

        # Each round is every agent's turn, the last only the first agent's, so a turn's round is
        # how many answers came before it
        if team_size and num_rounds is not None and kind != "structured_output":
            round_number = responses_so_far // team_size + 1
            if round_number != shown_round:
                shown_round = round_number
                label = f"Round {round_number} of {num_rounds}" if round_number <= num_rounds else "Final round"
                parts.append(f'<div class="round">{label}</div>')

        colour = colours.get(speaker, "#8b8578")
        model = (turns[index].get("model") if turns else None) or ""
        model_html = f'<span class="model">{escape(model)}</span>' if model else ""

        if kind in ("tool_output", "code_output"):
            label = "What the tools returned" if kind == "tool_output" else "What the code printed"
            text = shorten(without_observation(message) if kind == "code_output" else message, max_output_chars)
            body = markdown(text) if kind == "tool_output" else f'<pre class="output">{escape(text)}</pre>'
            parts.append(f'<div class="turn note"><div class="label">{label}</div><div class="body">{body}</div></div>')
            continue

        if kind == "structured_output":
            parts.append(
                f'<div class="turn" style="--colour: {colour}">'
                f'<div class="speaker">{escape(speaker)} {model_html}</div>'
                '<div class="label">Conclusions, as asked for</div>'
                f'<div class="body">{code_block(message, "json")}</div></div>'
            )
            continue

        body = ""
        if kind == "code_action" and (action := find_code_action(message)) is not None:
            prose = message[: message.find("<execute>")].strip()
            body = (markdown(prose) if prose else "") + f'<div class="label">Code ({escape(action.language)})</div>'
            body += code_block(action.code, action.language)
        else:
            # The figures the turn's code drew, shown before what the agent made of them
            for run in (turns[index].get("code_runs") or []) if turns and kind == "response" else []:
                for plot in run.get("plots") or []:
                    figures += 1
                    uri = image_uri(directory, plot) if directory is not None else None
                    picture = (
                        f'<img src="{uri}" alt="Figure {figures}">'
                        if uri
                        else f"<em>{escape(plot)} could not be found.</em>"
                    )
                    body += (
                        f"<figure>{picture}<figcaption>Figure {figures}. Drawn by code {escape(speaker)} ran"
                        f" ({escape(plot)}).</figcaption></figure>"
                    )
            body += markdown(message)

        parts.append(
            f'<div class="turn" style="--colour: {colour}"><div class="speaker">{escape(speaker)}{model_html}</div>'
            f'<div class="body">{body}</div></div>'
        )
        if kind == "response":
            responses_so_far += 1

    return "\n".join(parts)


def usage_table(usage: dict[str, Any], prices: dict[str, Any]) -> str:
    rows = []
    for model, counts in usage["per_model"].items():
        price = prices.get(model)
        # Priced as MeetingUsage.compute_cost prices it, cached input at the full input rate
        if not price:
            cost = "unpriced"
        elif counts.get("unreported_calls"):
            cost = "unknown"
        else:
            cost = money(
                (counts.get("input_tokens", 0) * price["input"] + counts.get("output_tokens", 0) * price["output"])
                / 1_000_000
            )
        rows.append(
            f"<tr><td>{escape(model)}</td><td class='number'>{counts.get('num_calls', 0):,}</td>"
            f"<td class='number'>{counts.get('input_tokens', 0):,}</td>"
            f"<td class='number'>{counts.get('output_tokens', 0):,}</td>"
            f"<td class='number'>{cost}</td></tr>"
        )
    rows.append(
        f"<tr><th>Total</th><th class='number'>{usage.get('num_calls', 0):,}</th>"
        f"<th class='number'>{usage.get('input_tokens', 0):,}</th>"
        f"<th class='number'>{usage.get('output_tokens', 0):,}</th>"
        f"<th class='number'>{money(usage.get('cost'))}</th></tr>"
    )

    return (
        "<table class='usage'><thead><tr><th>Model</th><th class='number'>Requests</th>"
        "<th class='number'>Input tokens</th><th class='number'>Output tokens</th><th class='number'>Cost</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def meeting_html(
    transcript_path: Path,
    include_prompts: bool = False,
    max_output_chars: int = MAX_EXPORTED_OUTPUT_CHARS,
) -> str:
    """A meeting as an HTML document: its agenda, team, and summary, then the discussion round by
    round, with the code it ran, what that printed, and its figures, then what it cost.

    :param transcript_path: The meeting's transcript, as hold_meeting saved it.
    :param include_prompts: Whether to show the prompts each agent was given, which the agenda,
        questions, and rules at the top otherwise stand for.
    :param max_output_chars: The most of a tool's or the session's output to show for one message.
    :return: The document.
    """
    meeting = load_meeting(transcript_path)

    return document(humanize(meeting.name), meeting_body(meeting, include_prompts, max_output_chars))


def save_meeting_html(transcript_path: Path, html_path: Path | None = None, **options: Any) -> Path:
    """Saves a meeting as an HTML document, beside its transcript unless told where.

    :param transcript_path: The meeting's transcript, as hold_meeting saved it.
    :param html_path: Where to save it, or None for the transcript's path with .html.
    :param options: What meeting_html takes.
    :return: Where it was saved.
    """
    transcript_path = Path(transcript_path)
    path = Path(html_path) if html_path is not None else transcript_path.with_suffix(".html")
    document = meeting_html(transcript_path, **options)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")

    return path


def save_meeting_pdf(transcript_path: Path, pdf_path: Path | None = None, **options: Any) -> Path:
    """Saves a meeting as a PDF, beside its transcript unless told where, as Biomni saves its
    agent's conversation.

    :param transcript_path: The meeting's transcript, as hold_meeting saved it.
    :param pdf_path: Where to save it, or None for the transcript's path with .pdf.
    :param options: What meeting_html takes.
    :raises PDFExportError: If WeasyPrint or Pango is not installed.
    :return: Where it was saved.
    """
    transcript_path = Path(transcript_path)
    path = Path(pdf_path) if pdf_path is not None else transcript_path.with_suffix(".pdf")

    return render_pdf(meeting_html(transcript_path, **options), path)


# Projects

def project_body(project_dir: Path) -> tuple[str, str]:
    """A project's report as the body of a document, and its title."""
    report_path = project_dir / REPORT_FILE_NAME
    if not report_path.is_file():
        raise FileNotFoundError(f"{project_dir} has no {REPORT_FILE_NAME}: run_project writes one when a run ends")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    status = report["status"]
    badge = f'<span class="badge {"ok" if status == "finished" else "bad"}">{escape(status.replace("_", " "))}</span>'
    limit = f" of a limit of {money(report['max_cost'])}" if report.get("max_cost") is not None else ""
    parts = [
        '<div class="eyebrow">Virtual Lab · Project report</div>',
        f"<h1>{escape(first_line(report['goal']))}</h1>",
        meta_line([badge, f"{len(report['rounds'])} rounds", f"spent {money(report.get('spent'))}{limit}"]),
        f'<div class="callout"><div class="label">Goal</div>{markdown(report["goal"])}'
        f"<p>{escape(report['reason'])}</p></div>",
    ]

    if report.get("answer") is not None:
        parts.append(f"<h2>Answer</h2>{markdown(report['answer'])}")
    elif report.get("proposed_answer") is not None:
        parts.append(
            f"<h2>Last answer proposed, which the critic did not accept</h2>{markdown(report['proposed_answer'])}"
        )
    if report.get("objections"):
        items = "".join(f"<li>{markdown(objection)}</li>" for objection in report["objections"])
        parts.append(f"<h2>The critic's objections</h2><ul>{items}</ul>")

    agents = [
        (report["team_lead"], "team lead"),
        (report["critic"], "critic"),
        *((member, None) for member in report["team"]),
    ]
    members = [(agent, AGENT_COLOURS[index % len(AGENT_COLOURS)], part) for index, (agent, part) in enumerate(agents)]
    parts.append(f"<h2>Team</h2>{team_cards(members)}")

    if report.get("team_changes"):
        rows = "".join(
            f"<tr><td class='number'>{change['round'] or 'start'}</td>"
            f"<td>{escape(', '.join(agent['title'] for agent in change['added']) or '-')}</td>"
            f"<td>{escape(', '.join(change['removed']) or '-')}</td><td>{escape(change['why'])}</td></tr>"
            for change in report["team_changes"]
        )
        parts.append(
            "<h3>Changes to the team</h3><table><thead><tr><th class='number'>Round</th><th>Joined</th><th>Left</th>"
            f"<th>Why</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    marks = {"done": "done", "dropped": "dropped", "in progress": "progress", "to do": "todo"}
    items = "".join(
        f'<li class="{marks.get(task["status"], "todo")}">{escape(task["task"])}</li>' for task in report["plan"]
    )
    parts.append(f'<h2>Plan</h2><ul class="plan">{items}</ul>')

    if report.get("findings"):
        rows = "".join(
            f"<tr><td>{escape(finding['id'])}</td><td>{markdown(finding['claim'])}</td>"
            f"<td>{markdown(finding.get('evidence') or '')}</td></tr>"
            for finding in report["findings"]
        )
        parts.append(
            "<h2>Findings</h2><table><thead><tr><th>Id</th><th>Claim</th><th>Evidence</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    rows = []
    for round_ in report["rounds"]:
        decision = round_.get("approved") or round_["proposed"]
        what = decision["action"].replace("_", " ")
        if decision.get("participants"):
            what += f" ({', '.join(decision['participants'])})"
        if decision.get("agenda"):
            what += f": {decision['agenda']}"
        outcome = round_["outcome"].replace("_", " ") + (f": {round_['note']}" if round_.get("note") else "")
        rows.append(
            f"<tr><td class='number'>{round_['number']}</td><td>{escape(what)}</td><td>{escape(outcome)}</td></tr>"
        )
    parts.append(
        "<h2>Rounds</h2><table><thead><tr><th class='number'>Round</th><th>Decision</th><th>Outcome</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    return "\n".join(parts), first_line(report["goal"])


def first_line(text: str, max_chars: int = 120) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= max_chars else line[: max_chars - 1].rstrip() + "…"


def project_meetings(project_dir: Path) -> list[Path]:
    """The transcripts of a project's finished meetings, in the order they were held."""
    ledger = json.loads((project_dir / PROJECT_FILE_NAME).read_text(encoding="utf-8"))

    return [
        project_dir / step["files"]["transcript"]
        for step in ledger["steps"]
        if step["kind"] == "meeting" and step["status"] == "completed" and "transcript" in step.get("files", {})
    ]


def project_html(
    project_dir: Path,
    meetings: bool = False,
    include_prompts: bool = False,
    max_output_chars: int = MAX_EXPORTED_OUTPUT_CHARS,
) -> str:
    """A project run with run_project as an HTML document: its report, and its meetings if asked for.

    :param project_dir: The project's directory, where run_project saved its report.
    :param meetings: Whether to add every meeting the project finished, each from a new page.
    :param include_prompts: For the meetings, whether to show each agent's prompts.
    :param max_output_chars: For the meetings, the most output to show for one message.
    :raises FileNotFoundError: If the project has no report, as before any run of it has ended.
    :return: The document.
    """
    project_dir = Path(project_dir)
    body, title = project_body(project_dir)
    if meetings:
        for path in project_meetings(project_dir):
            section = meeting_body(load_meeting(path), include_prompts, max_output_chars, heading_level=2)
            body += f'\n<section class="meeting">{section}</section>'

    return document(title, body)


def save_project_html(project_dir: Path, html_path: Path | None = None, **options: Any) -> Path:
    """Saves a project's report as an HTML document, as report.html in its directory unless told where.

    :param project_dir: The project's directory.
    :param html_path: Where to save it, or None for report.html in the project's directory.
    :param options: What project_html takes.
    :return: Where it was saved.
    """
    project_dir = Path(project_dir)
    path = Path(html_path) if html_path is not None else project_dir / "report.html"
    document = project_html(project_dir, **options)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")

    return path


def save_project_pdf(project_dir: Path, pdf_path: Path | None = None, **options: Any) -> Path:
    """Saves a project's report as a PDF, as report.pdf in its directory unless told where.

    :param project_dir: The project's directory.
    :param pdf_path: Where to save it, or None for report.pdf in the project's directory.
    :param options: What project_html takes.
    :raises PDFExportError: If WeasyPrint or Pango is not installed.
    :return: Where it was saved.
    """
    project_dir = Path(project_dir)
    path = Path(pdf_path) if pdf_path is not None else project_dir / "report.pdf"

    return render_pdf(project_html(project_dir, **options), path)


# PDF

def url_fetcher(weasyprint: Any) -> Any:
    """What WeasyPrint fetches with: only data: URIs, which the document carries its figures in,
    so that nothing else, such as an image an agent linked to, is fetched from the web or the
    disk.

    From WeasyPrint 68 a fetcher is a URLFetcher, and before that a function.
    """
    if hasattr(weasyprint, "URLFetcher"):
        return weasyprint.URLFetcher(allowed_protocols=("data",))

    def fetch(url: str, *args: Any, **kwargs: Any) -> Any:
        if not url.lower().startswith("data:"):
            raise ValueError(f"{url} is not fetched for a document made here")
        return weasyprint.default_url_fetcher(url, *args, **kwargs)

    return fetch


def import_weasyprint() -> Any:
    """WeasyPrint, imported without the advice it prints and warns of as it loads.

    It prints how to install Pango when Pango is missing, which the error raised here replaces,
    and warns at every import of the libraries its future versions will need, which a PDF made
    now does not.

    :raises PDFExportError: If WeasyPrint or Pango is not installed.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import weasyprint
    except (ImportError, OSError) as error:
        raise PDFExportError(f"{PDF_INSTALL_HELP} ({type(error).__name__}: {error})") from error

    return weasyprint


def render_pdf(document_html: str, pdf_path: Path) -> Path:
    """Makes a PDF of an HTML document with WeasyPrint.

    :raises PDFExportError: If WeasyPrint or Pango is not installed.
    :return: Where the PDF was saved.
    """
    weasyprint = import_weasyprint()

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    weasyprint.HTML(string=document_html, url_fetcher=url_fetcher(weasyprint)).write_pdf(pdf_path)

    return pdf_path
