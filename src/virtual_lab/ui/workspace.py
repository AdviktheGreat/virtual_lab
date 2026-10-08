"""Where the web interface keeps what it runs: a meeting or a project in a directory of its own,
and its settings, all under one workspace directory.

    virtual_lab_workspace/
        settings.json                                    what every new form starts with
        scientists.json                                  the scientists the person described
        meetings/2026-10-07_1530_choose-the-epitopes/    a meeting, saved as hold_meeting saves it
        projects/2026-10-07_1600_design-nanobodies/      a project, as Project saves it

A run's directory also holds interface_setup.json, what its form was filled in with, so that a
project can be carried on as it was set up.

Everything shown in the history is read from those files, so a meeting or project run from a
script into the workspace is shown too.
"""

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from virtual_lab.constants import (
    DEFAULT_MODEL,
    METADATA_DIR_NAME,
    PARTIAL_MEETING_DIR_NAME,
    PROJECT_FILE_NAME,
    REPORT_FILE_NAME,
)
from virtual_lab.utils import write_atomically

DEFAULT_WORKSPACE = Path.home() / "virtual_lab_workspace"

# What every meeting the interface holds is saved under, in its own directory
MEETING_NAME = "meeting"

# What a project run from the interface was set up with, so that it can be carried on as it was
PROJECT_SETUP_FILE_NAME = "interface_setup.json"

SETTINGS_FILE_NAME = "settings.json"

# The scientists the person described, to put on a team beside the library's
SCIENTISTS_FILE_NAME = "scientists.json"
SCIENTIST_FIELDS = ("title", "expertise", "goal", "role")

# The error a meeting's record names when the person following it stopped it: runs.RunStopped
STOPPED_ERROR_TYPE = "RunStopped"

# The most characters of a title a directory is named after
MAX_SLUG_CHARS = 48


@dataclass
class Settings:
    """What the interface starts every form with, kept in the workspace's settings.json.

    :param model: The model every new agent is given.
    :param stream: Whether replies are shown word by word as they are written.
    :param max_cost: The budget a meeting or project starts with, in USD, or None for none.
    :param code: Where agents' code runs: "none", "docker", or "local".
    :param sandbox: The sandbox image a session runs in with Docker: "python", or a stage of
        Biomni's environment, "base", "bio", or "full".
    :param network: Whether code in a session can reach the network.
    :param python: The interpreter code runs with on this machine, or "" for this one.
    :param mcp_config: A file of MCP servers to offer, as connect_mcp reads it, or "".
    """

    model: str = DEFAULT_MODEL
    stream: bool = True
    max_cost: float | None = 2.0
    code: str = "none"
    sandbox: str = "python"
    network: bool = True
    python: str = ""
    mcp_config: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True)
class Entry:
    """A meeting or project in the workspace, as the history lists it.

    :param kind: "meeting" or "project".
    :param path: Its directory.
    :param title: What it was about: a meeting's agenda, or a project's goal.
    :param started_at: When it began, as an ISO timestamp, or "" if not known.
    :param status: How it ended, or "running".
    :param cost: What it cost in USD, or None if not known.
    :param transcript: For a meeting, the transcript to show, finished or partial; else None.
    """

    kind: str
    path: Path
    title: str
    started_at: str
    status: str
    cost: float | None
    transcript: Path | None = field(default=None)


def slug(text: str) -> str:
    """Text made into a short name for a directory: lower case words joined by "-"."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    name = ""
    for word in words:
        if len(name) + len(word) + 1 > MAX_SLUG_CHARS:
            break
        name = f"{name}-{word}" if name else word

    return name or "untitled"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Workspace:
    """The directory the interface saves meetings and projects in.

    :param root: The directory, created if missing.
    """

    def __init__(self, root: Path = DEFAULT_WORKSPACE) -> None:
        self.root = Path(root).expanduser().resolve()
        self.meetings_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    @property
    def meetings_dir(self) -> Path:
        return self.root / "meetings"

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    @property
    def settings_path(self) -> Path:
        return self.root / SETTINGS_FILE_NAME

    def load_settings(self) -> Settings:
        data = read_json(self.settings_path)
        return Settings.from_dict(data) if isinstance(data, dict) else Settings()

    def save_settings(self, settings: Settings) -> None:
        write_atomically(self.settings_path, json.dumps(asdict(settings), indent=4).encode("utf-8"))

    @property
    def scientists_path(self) -> Path:
        return self.root / SCIENTISTS_FILE_NAME

    def load_scientists(self) -> list[dict[str, str]]:
        """The scientists the person described, each with a title, expertise, goal, and role."""
        data = read_json(self.scientists_path)
        if not isinstance(data, list):
            return []

        return [
            {key: str(item[key]) for key in SCIENTIST_FIELDS}
            for item in data
            if isinstance(item, dict) and all(str(item.get(key) or "").strip() for key in SCIENTIST_FIELDS)
        ]

    def save_scientist(self, scientist: dict[str, str]) -> None:
        """Keeps a scientist the person described, in place of any with the same title.

        :raises ValueError: If a field is empty.
        """
        described = {key: " ".join(str(scientist.get(key) or "").split()) for key in SCIENTIST_FIELDS}
        if empty := [key for key, value in described.items() if not value]:
            raise ValueError(f"A scientist needs {', '.join(empty)}")
        kept = [item for item in self.load_scientists() if item["title"].casefold() != described["title"].casefold()]
        write_atomically(self.scientists_path, json.dumps([*kept, described], indent=4).encode("utf-8"))

    def remove_scientist(self, title: str) -> None:
        kept = [item for item in self.load_scientists() if item["title"].casefold() != title.casefold()]
        write_atomically(self.scientists_path, json.dumps(kept, indent=4).encode("utf-8"))

    def new_directory(self, kind: str, title: str, now: datetime | None = None) -> Path:
        """A new directory for a meeting or project, named after when it began and what it is about.

        :param kind: "meeting" or "project".
        :param title: Its agenda or goal.
        :param now: When it begins, defaulting to now.
        """
        parent = self.meetings_dir if kind == "meeting" else self.projects_dir
        base = f"{(now or datetime.now()).strftime('%Y-%m-%d_%H%M')}_{slug(title)}"
        path = parent / base
        number = 2
        while path.exists():
            path = parent / f"{base}-{number}"
            number += 1
        path.mkdir(parents=True)

        return path

    def entries(self) -> list[Entry]:
        """Every meeting and project in the workspace, the latest first."""
        found = [entry for path in self.meetings_dir.iterdir() if (entry := meeting_entry(path))]
        found += [entry for path in self.projects_dir.iterdir() if (entry := project_entry(path))]

        return sorted(found, key=lambda entry: (entry.started_at, entry.path.name), reverse=True)

    def find(self, path: Path | str) -> Entry | None:
        """The meeting or project in this directory, if it is one of the workspace's."""
        path = Path(path).resolve()
        if path.parent == self.meetings_dir:
            return meeting_entry(path)
        if path.parent == self.projects_dir:
            return project_entry(path)

        return None


def meeting_entry(path: Path) -> Entry | None:
    """A meeting directory as the history lists it: finished if it was saved, else as it stopped."""
    if not path.is_dir():
        return None
    for where in (path, path / PARTIAL_MEETING_DIR_NAME):
        record = read_json(where / METADATA_DIR_NAME / f"{MEETING_NAME}.json")
        if isinstance(record, dict) and (where / f"{MEETING_NAME}.json").is_file():
            status = str(record.get("status") or "unknown")
            if status == "failed" and (record.get("error") or {}).get("type") == STOPPED_ERROR_TYPE:
                status = "stopped"
            return Entry(
                kind="meeting",
                path=path,
                title=str(record.get("agenda") or path.name),
                started_at=str(record.get("started_at") or ""),
                status=status,
                cost=(record.get("usage") or {}).get("cost"),
                transcript=where / f"{MEETING_NAME}.json",
            )

    # A meeting still running, or one stopped before anything was said, has saved no record yet
    return None


def project_entry(path: Path) -> Entry | None:
    """A project directory as the history lists it, with its status from its report, if it has one."""
    ledger = read_json(path / PROJECT_FILE_NAME)
    if not isinstance(ledger, dict):
        return None

    report = read_json(path / REPORT_FILE_NAME)
    steps = ledger.get("steps") or []
    costs = [(step.get("usage") or {}).get("cost") for step in steps]
    if isinstance(report, dict):
        status = str(report.get("status"))
    elif any(step.get("status") == "running" for step in steps):
        status = "running"
    else:
        status = "not finished"

    return Entry(
        kind="project",
        path=path,
        title=str(ledger.get("goal") or path.name),
        started_at=str(ledger.get("created_at") or ""),
        status=status,
        cost=None if any(cost is None for cost in costs) else sum(costs),
    )
