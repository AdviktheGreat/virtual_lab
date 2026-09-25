"""Virtual Lab package."""

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting
from virtual_lab.schemas import AgentSpec, TeamRoster
from virtual_lab.structured import StructuredOutputError
from virtual_lab.tools import PUBMED_TOOL, Tool


__all__ = [
    "__version__",
    "Agent",
    "AgentSpec",
    "PUBMED_TOOL",
    "StructuredOutputError",
    "TeamRoster",
    "Tool",
    "run_meeting",
]
