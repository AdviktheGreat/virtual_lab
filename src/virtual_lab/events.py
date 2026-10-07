"""What a meeting or a project tells a caller as it goes, so that it can be followed live.

Biomni's agent yields each step of its work once the step is done, for its web interface to show.
A meeting here does the same through a function given as on_event, which is called with a
MeetingEvent at every turn, every message added to the transcript, every tool call and run of
code, and every response's usage. With stream, it is also called as each reply is written, with
the reply so far, so that a person can read it word by word. A project run with run_project is
followed the same way, with a ProjectEvent at each decision and round besides the events of the
meetings it holds.

An event is told of once it has happened, apart from "tool_calls" and "code", which are told of
before the tools or code run, so that what is running can be shown while it does.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

MeetingEventKind = Literal[
    "started",
    "resources",
    "turn",
    "writing",
    "message",
    "tool_calls",
    "code",
    "cell",
    "usage",
    "finished",
    "failed",
    "read_back",
]

ProjectEventKind = Literal["team", "plan", "decided", "code", "round", "finished"]


@dataclass(frozen=True)
class MeetingEvent:
    """Something that happened in a meeting.

    :param kind: What happened:

        - "started": the meeting began. data holds its meeting_type, agenda, agenda_questions,
          num_rounds, team (each agent as describe_agent gives it, the one who leads or answers
          first), tools, max_cost, and session (as Session.describe gives it, or None).
        - "resources": the resources of Biomni's environment the agents are told of were chosen.
          data holds them as the meeting's record does.
        - "turn": an agent's turn began. speaker is its title, and data holds its name and model.
        - "writing": part of a reply has been written. text is the reply so far, which replaces
          what an earlier "writing" event with the same data["request"] said: a request that is
          sent again, as one is without its temperature for a model that refuses one, starts
          again from nothing. Only with stream.
        - "message": a message was added to the transcript. speaker and text are as the
          transcript has them, and data holds its kind, as the record has it ("prompt",
          "response", "code_action", "code_output", "tool_output", or "structured_output"), and
          its index in the transcript.
        - "tool_calls": an agent called tools, which are about to run. data["calls"] holds each
          one's name and arguments.
        - "code": an agent wrote code, which is about to run in the session. data holds its
          code and language.
        - "cell": code ran in the session, whether an agent wrote it or called a tool to run it.
          data holds the run, as CellResult.to_dict gives it, and plot_paths, where each of its
          figures is on this machine.
        - "usage": a response came back. data is the meeting's usage so far, as
          MeetingUsage.to_dict gives it, with its cost, or None if that is not known.
        - "finished": the meeting ended and was saved. text is its summary, and data holds the
          transcript_path, record_path, and output_path, output as JSON, usage, and elapsed_time.
        - "failed": the meeting stopped with an error. text is the error, and data holds its
          type, the partial_path where what was done was saved, or None, and usage.
        - "read_back": a project read a meeting it held before back from disk, rather than
          holding it. text is its summary, and data holds its transcript_path, record_path,
          output_path, and usage.

    :param meeting: The name the meeting is saved under.
    :param round: The round, from 1, where the last is num_rounds + 1, or None outside the rounds.
    :param speaker: Who spoke, by title, where someone did.
    :param text: What was said, written, or found, as kind describes.
    :param data: Everything else, as kind describes, in JSON's types.
    """

    kind: MeetingEventKind
    meeting: str
    round: int | None = None
    speaker: str | None = None
    text: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectEvent:
    """Something that happened in a project run with run_project.

    :param kind: What happened:

        - "team": the team was chosen, given, or changed. data holds the team, each member as
          describe_agent gives it, and the change, as the research log records it.
        - "plan": the team made the plan. data["plan"] holds its tasks.
        - "decided": the team lead decided the round's step. data holds the decision as
          proposed, as NextStep.model_dump gives it, which the approve hook is asked about next.
        - "code": code written for the round was run, and repaired if it failed. text is what
          came of it, as its author is told.
        - "round": the round ended. data holds the round as the research log records it, and the
          plan as it stands.
        - "finished": the project ended. text is why, and data holds the report, as
          ProjectReport.to_dict gives it.

    :param round: The round, from 1, or None before the first.
    :param text: What was found or decided, as kind describes.
    :param data: Everything else, as kind describes, in JSON's types.
    """

    kind: ProjectEventKind
    round: int | None = None
    text: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


OnMeetingEvent = Callable[[MeetingEvent], None]

# What run_project calls, with its own events and those of every meeting it holds
OnProjectEvent = Callable[[MeetingEvent | ProjectEvent], None]
