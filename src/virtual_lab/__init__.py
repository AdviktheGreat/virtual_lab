"""Virtual Lab package."""

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.artifacts import CodeArtifacts, CodeFile, UnsafeFilenameError, save_artifacts
from virtual_lab.execution import (
    DockerExecutor,
    DockerUnavailableError,
    ExecutionError,
    ExecutionResult,
    LocalExecutor,
    UnsupportedLanguageError,
    run_files,
)
from virtual_lab.repair import (
    RepairAttempt,
    RepairOutcome,
    run_with_repair,
    save_execution_record,
)
from virtual_lab.run_meeting import run_meeting
from virtual_lab.schemas import AgentSpec, ComponentAssignment, ImplementationPlan, TeamRoster
from virtual_lab.structured import StructuredOutputError
from virtual_lab.tools import PUBMED_TOOL, Tool
from virtual_lab.web import (
    ALLOWED_HOSTS,
    DisallowedHostError,
    ResponseTooLargeError,
    WebRequestError,
    request_json,
    request_text,
)


__all__ = [
    "__version__",
    "ALLOWED_HOSTS",
    "Agent",
    "AgentSpec",
    "CodeArtifacts",
    "CodeFile",
    "ComponentAssignment",
    "DisallowedHostError",
    "DockerExecutor",
    "DockerUnavailableError",
    "ExecutionError",
    "ExecutionResult",
    "ImplementationPlan",
    "LocalExecutor",
    "PUBMED_TOOL",
    "RepairAttempt",
    "RepairOutcome",
    "ResponseTooLargeError",
    "StructuredOutputError",
    "TeamRoster",
    "Tool",
    "UnsafeFilenameError",
    "UnsupportedLanguageError",
    "WebRequestError",
    "request_json",
    "request_text",
    "run_files",
    "run_meeting",
    "run_with_repair",
    "save_artifacts",
    "save_execution_record",
]
