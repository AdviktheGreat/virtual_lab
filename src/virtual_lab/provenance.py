"""Provenance records describing how a meeting was produced.

The transcript says what was said. This says what produced it: which model spoke each turn, what
it cost, which tools it called, and which versions of everything were involved. It is written to
a sidecar file so that the transcript itself stays in the format existing readers expect.
"""

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.constants import METADATA_DIR_NAME
from virtual_lab.utils import MeetingUsage


def utc_timestamp() -> str:
    """Returns the current time as an ISO 8601 string in UTC."""
    return datetime.now(timezone.utc).isoformat()


def describe_agent(agent: Agent) -> dict[str, str]:
    """Describes an agent completely enough to reconstruct it.

    :param agent: The agent to describe.
    :return: The agent's persona, author name, and model.
    """
    return {
        "title": agent.title,
        "name": agent.name,
        "model": agent.model,
        "expertise": agent.expertise,
        "goal": agent.goal,
        "role": agent.role,
    }


@dataclass
class TurnRecord:
    """What produced a single turn of a meeting.

    A turn can span several API calls when an agent uses tools, so the token counts are summed
    across the calls that produced it.

    :param index: The turn's position in the transcript, matching the transcript's own ordering.
    :param speaker: The title of whoever produced the turn, or "User" or "Tool".
    :param kind: Whether the turn is a prompt, an agent's response, or tool output.
    :param timestamp: When the turn was recorded.
    :param name: The agent's author name, or None for prompts and tool output.
    :param model: The model that produced the turn, or None for prompts and tool output.
    :param num_api_calls: The number of API calls the turn took, above one if tools were used.
    :param tool_calls: Each tool called while producing the turn, by name, with its arguments.
    :param system_fingerprint: The backend configuration the API reported, when it reports one.
    :param finish_reason: Why the model stopped, which is "length" if it ran out of tokens.
    """

    index: int
    speaker: str
    kind: str
    timestamp: str
    name: str | None = None
    model: str | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    num_api_calls: int = 0
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    system_fingerprint: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns the record as a JSON-serializable dictionary."""
        return {key: value for key, value in self.__dict__.items()}


@dataclass
class MeetingRecord:
    """What produced a meeting.

    :param meeting_type: Either "team" or "individual".
    :param save_name: The name the transcript was saved under.
    :param status: "completed" if the meeting finished, "failed" if it raised partway through.
    :param summaries_sha256: A hash of each summary passed in, so that a meeting can be matched
        to the meetings it was given without copying their text into every record.
    :param models_at_default_temperature: Models that refused the temperature asked for and ran
        at their own default instead.
    :param prices: What each model used was priced at, in USD per million tokens, since the
        price tables change and a cost is only comparable at the prices it was computed with.
    """

    meeting_type: str
    save_name: str
    num_rounds: int
    temperature: float
    max_retries: int
    team: list[dict[str, str]]
    critic: dict[str, str] | None = None
    tools: list[str] = field(default_factory=list)
    agenda: str = ""
    agenda_questions: list[str] = field(default_factory=list)
    agenda_rules: list[str] = field(default_factory=list)
    summaries_sha256: list[str] = field(default_factory=list)
    contexts_sha256: list[str] = field(default_factory=list)
    output_schema: str | None = None
    max_tool_iterations: int | None = None
    max_completion_tokens: int | None = None
    max_cost: float | None = None
    models_at_default_temperature: list[str] = field(default_factory=list)
    prices: dict[str, dict[str, float] | None] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_timestamp)
    ended_at: str | None = None
    elapsed_seconds: float | None = None
    status: str = "running"
    error: dict[str, str] | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def record_turn(self, **kwargs: Any) -> TurnRecord:
        """Appends a turn record, numbering it by its position.

        :param kwargs: The fields of the turn record, other than index and timestamp.
        :return: The appended record, so that token counts can be added as calls complete.
        """
        turn = TurnRecord(index=len(self.turns), timestamp=utc_timestamp(), **kwargs)
        self.turns.append(turn)

        return turn

    def finish(self, usage: MeetingUsage, elapsed_time: float, error: BaseException | None) -> None:
        """Closes the record once the meeting has ended.

        :param usage: The usage accumulated over the meeting.
        :param elapsed_time: How long the meeting took, in seconds.
        :param error: The exception that ended the meeting, or None if it completed.
        """
        self.ended_at = utc_timestamp()
        self.elapsed_seconds = elapsed_time
        self.status = "completed" if error is None else "failed"
        self.usage = usage.to_dict()

        if error is not None:
            self.error = {"type": type(error).__name__, "message": str(error)}

    def to_dict(self) -> dict[str, Any]:
        """Returns the record as a JSON-serializable dictionary."""
        return {
            "virtual_lab_version": __version__,
            "openai_version": openai.__version__,
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "meeting_type": self.meeting_type,
            "save_name": self.save_name,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": self.elapsed_seconds,
            "num_rounds": self.num_rounds,
            "temperature": self.temperature,
            "models_at_default_temperature": self.models_at_default_temperature,
            "max_retries": self.max_retries,
            "max_tool_iterations": self.max_tool_iterations,
            "max_completion_tokens": self.max_completion_tokens,
            "max_cost": self.max_cost,
            "agenda": self.agenda,
            "agenda_questions": self.agenda_questions,
            "agenda_rules": self.agenda_rules,
            "summaries_sha256": self.summaries_sha256,
            "contexts_sha256": self.contexts_sha256,
            "output_schema": self.output_schema,
            "team": self.team,
            "critic": self.critic,
            "tools": self.tools,
            "usage": self.usage,
            "prices": self.prices,
            "turns": [turn.to_dict() for turn in self.turns],
        }


def save_record(save_dir: Path, save_name: str, record: MeetingRecord) -> Path:
    """Writes a meeting record into the metadata subdirectory of the transcript's directory.

    :param save_dir: The directory the transcript was saved in.
    :param save_name: The name the transcript was saved under.
    :param record: The record to write.
    :return: The path written.
    """
    metadata_dir = save_dir / METADATA_DIR_NAME
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / f"{save_name}.json"

    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=4)

    return path
