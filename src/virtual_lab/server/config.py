"""What a conversation is set up with, which is kept with it so that it is carried on as it was set up.

A new conversation is given only what differs from the defaults in the workspace's settings. What
it is set up with is then written out in full, with each scientist as they were described then, so
that changing the library or the settings later does not change a conversation that has begun.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from virtual_lab.agent import Agent
from virtual_lab.llm import detect_source
from virtual_lab.prompts import PRINCIPAL_INVESTIGATOR, SCIENTIFIC_CRITIC
from virtual_lab.schemas import normalize_title
from virtual_lab.scientists import SCIENTISTS
from virtual_lab.ui.library import missing_keys, models, provider_of
from virtual_lab.ui.runs import CodeSetup, agent_from
from virtual_lab.ui.workspace import SCIENTIST_FIELDS, Settings, Workspace, read_json
from virtual_lab.utils import write_atomically

CONFIG_FILE_NAME = "config.json"

# Who is on the team when none is chosen: the Scientific Critic, who reviews the others, and every
# other scientist of the library, so that the lead can bring in whoever a question needs
DEFAULT_TEAM = (
    SCIENTIFIC_CRITIC.title,
    *(
        agent.title
        for agent in SCIENTISTS
        if agent.title not in (PRINCIPAL_INVESTIGATOR.title, SCIENTIFIC_CRITIC.title)
    ),
)

# The settings a page shows and changes: those of a conversation, and not the MCP config the interface keeps
PUBLIC_SETTINGS = ("model", "stream", "max_cost", "code", "sandbox", "network", "python")

# What a team's scientist is described with
Spec = Mapping[str, Any]


@dataclass(frozen=True)
class ChatConfig:
    """What a conversation is set up with.

    :param model: The model the lead and the team use.
    :param lead: The head of the lab, as its title, expertise, goal, and role.
    :param team: The scientists the lead may bring in, each described the same way.
    :param code: Where code runs, and with what of Biomni's resources the agents are told of.
    :param max_cost: The most the conversation may spend in USD, or None for no limit.
    :param stream: Whether replies are shown as they are written.
    :param commercial_mode: Whether to leave out data and know-how that may not be used commercially.
    """

    model: str
    lead: dict[str, str]
    team: tuple[dict[str, str], ...]
    code: CodeSetup
    max_cost: float | None
    stream: bool
    commercial_mode: bool

    def lead_agent(self) -> Agent:
        return agent_from({**self.lead, "model": self.model})

    def team_agents(self) -> tuple[Agent, ...]:
        return tuple(agent_from({**member, "model": self.model}) for member in self.team)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "lead": dict(self.lead),
            "team": [dict(member) for member in self.team],
            "code": {
                "where": self.code.where,
                "sandbox": self.code.sandbox,
                "network": self.code.network,
                "python": self.code.python,
                "resources": self.code.resources,
            },
            "max_cost": self.max_cost,
            "stream": self.stream,
            "commercial_mode": self.commercial_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChatConfig":
        return cls(
            model=str(data["model"]),
            lead={key: str(data["lead"][key]) for key in SCIENTIST_FIELDS},
            team=tuple({key: str(member[key]) for key in SCIENTIST_FIELDS} for member in data["team"]),
            code=CodeSetup(**data["code"]),
            max_cost=data["max_cost"],
            stream=bool(data["stream"]),
            commercial_mode=bool(data["commercial_mode"]),
        )


def public_settings(settings: Settings) -> dict[str, Any]:
    return {key: getattr(settings, key) for key in PUBLIC_SETTINGS}


def update_settings(settings: Settings, changes: Mapping[str, Any]) -> Settings:
    """The settings with these changed, which are not kept if one cannot be used.

    :param changes: Any of the public settings, with the value to give it.
    :raises ValueError: If a setting is not one that can be changed, or its value cannot be used.
    """
    if unknown := sorted(set(changes) - set(PUBLIC_SETTINGS)):
        raise ValueError(f"There is no setting {', '.join(unknown)}")
    changed = replace(settings, **changes)
    detect_source(changed.model)
    CodeSetup(where=changed.code, sandbox=changed.sandbox)
    if changed.max_cost is not None:
        changed.max_cost = float(changed.max_cost)
        if not (math.isfinite(changed.max_cost) and changed.max_cost >= 0):
            raise ValueError(f"max_cost must be a finite amount, zero or more, not {changed.max_cost}")

    return changed


def describe_models() -> list[dict[str, Any]]:
    """The models to choose from, each with its provider and whether the keys that provider needs are set."""
    return [{"id": model, "provider": provider_of(model), "ready": not missing_keys([model])} for model in models()]


def library(workspace: Workspace) -> list[dict[str, Any]]:
    """The scientists to choose from: the library's, then those the person described, which stand in
    place of any with the same title. Each has its title, expertise, goal, and role, and whether it is
    built in."""
    chosen: dict[str, dict[str, Any]] = {
        agent.title.casefold(): {**{key: getattr(agent, key) for key in SCIENTIST_FIELDS}, "builtin": True}
        for agent in SCIENTISTS
    }
    for item in workspace.load_scientists():
        chosen[item["title"].casefold()] = {**item, "builtin": False}

    return list(chosen.values())


def describe_scientist(choice: str | Spec, scientists: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """A scientist as the request names them, by title, or as it describes them.

    :raises ValueError: If a title is not in the library or a description lacks a part.
    """
    if isinstance(choice, str):
        found = next((item for item in scientists if str(item["title"]).casefold() == choice.strip().casefold()), None)
        if found is None:
            raise ValueError(f'There is no scientist titled "{choice}"')
        choice = found
    spec = {key: " ".join(str(choice.get(key) or "").split()) for key in SCIENTIST_FIELDS}
    if empty := [key for key, value in spec.items() if not value]:
        raise ValueError(f"A scientist needs {', '.join(empty)}")

    return spec


def resolve_config(request: Mapping[str, Any], settings: Settings, workspace: Workspace) -> ChatConfig:
    """What a new conversation is set up with: what the request gives, and the settings for the rest.

    :param request: What the person chose. A key left out is taken from the settings, or the library: "model",
        "lead" and each of "team", by title or described, "code" with any of where, sandbox, network, python,
        and resources, "max_cost" (null for no limit), "stream", and "commercial_mode".
    :raises ValueError: If something chosen cannot be used.
    """
    scientists = library(workspace)
    model = str(request.get("model") or settings.model).strip()
    detect_source(model)

    lead = describe_scientist(request.get("lead") or PRINCIPAL_INVESTIGATOR.title, scientists)
    team_choice = request["team"] if request.get("team") is not None else DEFAULT_TEAM
    team = tuple(describe_scientist(choice, scientists) for choice in team_choice)
    titles = [normalize_title(member["title"]) for member in (lead, *team)]
    if len(set(titles)) != len(titles):
        raise ValueError("The lead and the team must have different titles")

    code = CodeSetup(
        **{
            "where": settings.code,
            "sandbox": settings.sandbox,
            "network": settings.network,
            "python": settings.python,
            **(request.get("code") or {}),
        }
    )

    max_cost = request["max_cost"] if "max_cost" in request else settings.max_cost
    if max_cost is not None:
        max_cost = float(max_cost)
        if not (math.isfinite(max_cost) and max_cost >= 0):
            raise ValueError(f"max_cost must be a finite amount, zero or more, not {max_cost}")

    stream = request["stream"] if request.get("stream") is not None else settings.stream

    return ChatConfig(
        model=model,
        lead=lead,
        team=team,
        code=code,
        max_cost=max_cost,
        stream=bool(stream),
        commercial_mode=bool(request.get("commercial_mode") or False),
    )


def save_config(directory: Path, config: ChatConfig) -> None:
    write_atomically(directory / CONFIG_FILE_NAME, json.dumps(config.to_dict(), indent=4).encode("utf-8"))


def load_config(directory: Path) -> ChatConfig | None:
    """The configuration saved in a conversation's directory, or None if there is none that can be read."""
    try:
        return ChatConfig.from_dict(read_json(directory / CONFIG_FILE_NAME))
    except (KeyError, TypeError, ValueError):
        return None
