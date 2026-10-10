"""What the web interface offers to choose from: scientists to put on a team, models, the
providers' keys, and the tools agents can be given."""

import os
from dataclasses import dataclass
from pathlib import Path

from virtual_lab.constants import DEFAULT_MODEL, MODEL_TO_INPUT_PRICE_PER_TOKEN
from virtual_lab.llm import detect_source
from virtual_lab.mcp_presets import MCP_PRESETS
from virtual_lab.scientists import SCIENTISTS as SCIENTISTS, scientist as scientist
from virtual_lab.tools import DATABASE_TOOLS

# The environment variables each provider's models are reached with, as the README lists them
PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "OpenAI": ("OPENAI_API_KEY",),
    "Anthropic": ("ANTHROPIC_API_KEY",),
    "Gemini": ("GEMINI_API_KEY",),
    "Groq": ("GROQ_API_KEY",),
    "AzureOpenAI": ("OPENAI_API_KEY", "OPENAI_ENDPOINT"),
    "Bedrock": ("AWS_REGION",),
}

# The keys a person may give the interface for the rest of the process
SETTABLE_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")


@dataclass(frozen=True)
class ProviderStatus:
    """Whether a provider's models can be reached, by whether its variables are set.

    :param provider: The provider, as detect_source names it.
    :param variables: The variables it needs.
    :param missing: Those of them not set.
    """

    provider: str
    variables: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


def models() -> list[str]:
    """The models to choose from: the default, then every priced one, newest families first."""
    return list(dict.fromkeys([DEFAULT_MODEL, *sorted(MODEL_TO_INPUT_PRICE_PER_TOKEN, key=model_order)]))


MODEL_FAMILIES = ("gpt-5", "claude-", "gemini-", "gpt-4", "o")


def model_order(model: str) -> tuple[int, str]:
    family = next(
        (index for index, prefix in enumerate(MODEL_FAMILIES) if model.startswith(prefix)), len(MODEL_FAMILIES)
    )
    return family, model


def provider_of(model: str) -> str | None:
    try:
        return detect_source(model)
    except ValueError:
        return None


def provider_statuses() -> list[ProviderStatus]:
    """Every provider, with the variables it needs that are not set. No value is read out."""
    return [
        ProviderStatus(provider, variables, tuple(name for name in variables if not os.environ.get(name)))
        for provider, variables in PROVIDER_KEYS.items()
    ]


def missing_keys(models_used: list[str]) -> list[str]:
    """The variables the providers of these models need that are not set, to say so before a run."""
    statuses = {status.provider: status for status in provider_statuses()}
    missing: list[str] = []
    for model in models_used:
        status = statuses.get(provider_of(model) or "")
        if status is not None:
            missing += [name for name in status.missing if name not in missing]

    return missing


def mcp_choices(config: str = "") -> tuple[list[tuple[str, str]], str | None]:
    """The MCP servers to offer, as (label, name): those of the config, then the presets it does
    not name; and what is wrong with the config, if it cannot be read.

    :param config: A file of MCP servers, as connect_mcp reads it, or "" for the presets alone.
    """
    from virtual_lab.mcp_tools import read_config

    servers: dict[str, object] = {}
    problem = None
    if config.strip():
        try:
            servers = read_config(Path(config.strip()).expanduser())
        except Exception as error:
            problem = f"The MCP config {config.strip()} cannot be read: {error}"

    offered = [(f"{name} (from your config)", str(name)) for name in servers]
    offered += [(preset.about, name) for name, preset in MCP_PRESETS.items() if name not in servers]

    return offered, problem


def tool_choices() -> list[tuple[str, str]]:
    """The database tools, as (label, name): the label is the first sentence of what each does."""
    return [(f"{tool.name}: {tool.description.split('. ')[0].rstrip('.')}", tool.name) for tool in DATABASE_TOOLS]
