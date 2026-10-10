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

A conversation with the head of a lab, which has the lab's meetings held for it, is followed the same
way, with a ChatEvent that wraps the events of the meetings it starts.

A meeting can also be steered as it goes, through a function given as steer, which is asked
before every turn for a note from the person following it. A note is added to the discussion,
for the agent about to speak and every one after it to read, and the meeting waits while the
function does, which is how it is paused.
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

ChatEventKind = Literal[
    "user",
    "title",
    "resources",
    "writing",
    "tool_calls",
    "tool_output",
    "cell",
    "lab",
    "note",
    "usage",
    "answer",
    "stopped",
    "failed",
    "status",
]


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
          transcript has them, and data holds its kind, as the record has it ("prompt", "note",
          "response", "code_action", "code_output", "tool_output", or "structured_output"), and
          its index in the transcript. A "note" is one a person gave through steer.
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


@dataclass(frozen=True)
class ChatEvent:
    """Something that happened in a conversation with the head of a lab.

    A conversation is a series of turns. A turn begins with the researcher's "user" message and
    ends with one of "answer", "stopped", or "failed".

    :param kind: What happened:

        - "user": the researcher said something, which begins a turn. text is what they wrote, and
          data["attachments"] holds the files they attached, each with its name, its path in the
          session's directory, and its size in bytes.
        - "title": the conversation was given a title, as text.
        - "resources": the resources of Biomni's environment the lead is told of were worked out, or
          it asked for those that suit a task. data holds them as a meeting's record does: its mode,
          commercial_mode, the available counts, those selected by name, and how they were chosen.
        - "writing": part of a reply has been written, by the lead, or, with data["meeting"], by a
          scientist in a team session. text is the reply so far, which replaces what an earlier
          "writing" event with the same data["request"] and data["meeting"] said. Only with
          stream, and never kept in the saved log.
        - "tool_calls": the lead called tools, which are about to run. text is what it said
          beside them, and data["calls"] holds each one's id, name, and arguments.
        - "tool_output": a tool the lead called finished. text is what it returned, and data holds
          the call's id and the tool's name.
        - "cell": code the lead ran in the session ran. data is the run, as CellResult.to_dict gives
          it, with plot_paths, where each of its figures is on this machine, and the call's id.
        - "lab": something happened in a team session the lead started, with data["call"] the id of
          the call that started it. speaker and text are those of the meeting event, data["event"]
          is its kind, data["meeting"] its name, data["round"] its round, and data["details"] the
          rest of its data, as MeetingEvent describes. A meeting's prompts and usage are left out,
          since they are saved with its transcript and told of as the conversation's usage.
        - "note": the researcher's note was read, or was not. text is the note, and data holds to,
          "lead" or "lab" for who read it, and read, which is False, with to None, for a note that
          arrived too late for anyone to.
        - "usage": a response came back. data is the usage of the whole conversation, as
          MeetingUsage.to_dict gives it, with its cost, or None if that is not known, and
          max_cost.
        - "answer": the lead answered, which ends the turn. text is the answer, and data holds the
          conversation's usage and elapsed_time.
        - "stopped": the researcher stopped the turn, which ends it. text says so.
        - "failed": the turn ended with an error. text is the error, and data holds its type, which
          is "Interrupted" for a turn that the process running it did not live to end.
        - "status": the conversation's state changed. data["state"] is "idle", "running",
          "pausing", "paused", or "stopping". Never kept in the saved log.

    :param id: Its number in the conversation, from 1, in the order told of, across turns and
        across runs of the same conversation. Every event has a number of its own, including a
        "writing" event that replaces an earlier one, so that a page that follows by number is told
        of the change.
    :param turn: The turn it belongs to, from 1, or 0 for what happens outside one.
    :param speaker: Who spoke, by title, where someone did.
    :param text: What was said, written, or found, as kind describes.
    :param data: Everything else, as kind describes, in JSON's types.
    :param time: When it happened, in seconds since the epoch.
    """

    kind: ChatEventKind
    id: int = 0
    turn: int = 0
    speaker: str | None = None
    text: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """The event in JSON's types, as it is saved and sent."""
        return {
            "kind": self.kind,
            "id": self.id,
            "turn": self.turn,
            "speaker": self.speaker,
            "text": self.text,
            "data": dict(self.data),
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, saved: Mapping[str, Any]) -> "ChatEvent":
        """Rebuilds the event to_dict described."""
        return cls(
            kind=saved["kind"],
            id=saved["id"],
            turn=saved["turn"],
            speaker=saved["speaker"],
            text=saved["text"],
            data=saved["data"],
            time=saved["time"],
        )


@dataclass(frozen=True)
class NextTurn:
    """The turn a meeting is about to take, as the function given as steer is told of it.

    :param meeting: The name the meeting is saved under.
    :param round: The round, from 1, where the last is num_rounds + 1.
    :param speaker: The title of the agent about to speak.
    """

    meeting: str
    round: int
    speaker: str


OnMeetingEvent = Callable[[MeetingEvent], None]

# What a conversation calls, with each ChatEvent
OnChatEvent = Callable[[ChatEvent], None]

# What run_project calls, with its own events and those of every meeting it holds
OnProjectEvent = Callable[[MeetingEvent | ProjectEvent], None]

# What a meeting asks before every turn, for a note from the person following it, or None
Steer = Callable[[NextTurn], str | None]
