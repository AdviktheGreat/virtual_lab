"""Tools that agents can call during a meeting.

A tool's description is the whole of what a model knows about it when deciding whether to call it,
so each one says what the tool is for and what it needs, and the awkward cases say what the tool
is not for. A lookup that takes an accession is useless to an agent holding only a protein name,
and will be called with the name anyway unless the description rules it out.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from virtual_lab.constants import MAX_SEARCH_RESULTS, STRUCTURE_DIR_NAME
from virtual_lab.databases import (
    download_structure,
    get_predicted_structure,
    get_protein,
    get_structure,
    search_proteins,
)
from virtual_lab.utils import run_pubmed_search


@dataclass(frozen=True)
class Tool:
    """A function that an agent can call during a meeting.

    :param name: The name the model uses to call the tool.
    :param description: What the tool does, used by the model to decide when to call it.
    :param parameters: A JSON Schema object describing the tool's arguments.
    :param function: The callable that runs the tool. Its return value is sent to the model,
        so it should be a string or convertible to one.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    @property
    def definition(self) -> ChatCompletionToolParam:
        """Returns the tool in OpenAI API form."""
        return ChatCompletionToolParam(  # type: ignore[misc]
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }
        )


PUBMED_TOOL = Tool(
    name="pubmed_search",
    description="Get abstracts or the full text of biomedical and life sciences articles from PubMed Central.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to use to search PubMed Central for scientific articles.",
            },
            "num_articles": {
                "type": "integer",
                "description": "The number of articles to return from the search query.",
            },
            "abstract_only": {
                "type": "boolean",
                "description": "Whether to return only the abstract of the articles.",
            },
        },
        "required": ["query", "num_articles"],
    },
    function=run_pubmed_search,
)


def look_up_protein(accession: str) -> str:
    """Runs a UniProt lookup and renders it for a model."""
    return get_protein(accession=accession).report()


def find_proteins(query: str, limit: int = 10, reviewed_only: bool = False) -> str:
    """Runs a UniProt search and renders the results for a model."""
    return search_proteins(query=query, limit=limit, reviewed_only=reviewed_only).report()


def look_up_structure(pdb_id: str, include_sequences: bool = True) -> str:
    """Runs a PDB lookup and renders it for a model."""
    return get_structure(pdb_id=pdb_id, include_sequences=include_sequences).report()


def look_up_predicted_structure(accession: str) -> str:
    """Runs an AlphaFold lookup and renders it for a model."""
    return get_predicted_structure(accession=accession).report()


def fetch_structure_file(
    save_dir: Path,
    identifier: str,
    source: str = "pdb",
    file_format: str = "cif",
) -> str:
    """Downloads a structure file and renders where it landed.

    The directory is the first argument and is bound before the tool is offered, so it is not in
    the schema a model sees. Where files are written is the lab's decision, not a meeting's.
    """
    return download_structure(
        identifier=identifier,
        save_dir=save_dir,
        source=source,
        file_format=file_format,
    ).report()


UNIPROT_LOOKUP_TOOL = Tool(
    name="uniprot_lookup",
    description=(
        "Get a protein's sequence, function, domains, disulfide bonds, and known structures from "
        "UniProt. Requires a UniProt accession such as P01308; use uniprot_search first if you "
        "only have a name or a description."
    ),
    parameters={
        "type": "object",
        "properties": {
            "accession": {
                "type": "string",
                "description": "A UniProt accession, such as P01308 or A0A0F6YEF6.",
            },
        },
        "required": ["accession"],
    },
    function=look_up_protein,
)

UNIPROT_SEARCH_TOOL = Tool(
    name="uniprot_search",
    description=(
        "Find proteins in UniProt by name, gene, organism, or description, and get their "
        "accessions to look up. Accepts free text or UniProt query syntax such as "
        '"gene:INS AND organism_id:9606". Free text also matches an entry\'s references, so a '
        "protein merely studied using a technique can rank above an example of it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, as free text or UniProt query syntax.",
            },
            "limit": {
                "type": "integer",
                "description": f"How many results to return, at most {MAX_SEARCH_RESULTS}.",
            },
            "reviewed_only": {
                "type": "boolean",
                "description": (
                    "Restrict to curated Swiss-Prot entries. Leave this off unless you want only "
                    "well characterised proteins: curation excludes most sequences, including "
                    "every camelid nanobody."
                ),
            },
        },
        "required": ["query"],
    },
    function=find_proteins,
)

PDB_LOOKUP_TOOL = Tool(
    name="pdb_lookup",
    description=(
        "Get an experimentally determined structure's title, method, resolution, chains, bound "
        "ligands, and chain sequences from the Protein Data Bank. Requires a four character PDB "
        "identifier such as 4HHB, which uniprot_lookup will list for a given protein. This "
        "returns the metadata and sequences, not the coordinates; use fetch_structure_file for "
        "coordinates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pdb_id": {
                "type": "string",
                "description": "A four character PDB identifier, such as 4HHB.",
            },
            "include_sequences": {
                "type": "boolean",
                "description": "Whether to include the sequence of each chain. Defaults to true.",
            },
        },
        "required": ["pdb_id"],
    },
    function=look_up_structure,
)

ALPHAFOLD_LOOKUP_TOOL = Tool(
    name="alphafold_lookup",
    description=(
        "Get a predicted structure from the AlphaFold database for a UniProt accession, with its "
        "pLDDT confidence. Use this when no experimental structure exists. A prediction is not a "
        "measurement, and a low pLDDT means the region has no single structure rather than that "
        "the prediction is merely imprecise."
    ),
    parameters={
        "type": "object",
        "properties": {
            "accession": {
                "type": "string",
                "description": "A UniProt accession, such as P01308.",
            },
        },
        "required": ["accession"],
    },
    function=look_up_predicted_structure,
)


def structure_file_tool(save_dir: Path) -> Tool:
    """Builds the structure download tool, bound to a directory.

    This one is a function rather than a constant because it needs somewhere to write, and that
    somewhere must not be a parameter a model fills in.

    :param save_dir: The directory to write structure files into.
    :return: The tool.
    """
    return Tool(
        name="fetch_structure_file",
        description=(
            "Download the atomic coordinates of a structure so that code can read them. The file "
            "is written next to the code that will be run, and its name is reported back. Use "
            "this rather than asking for coordinates directly: a structure file is hundreds of "
            "kilobytes and is for code to parse, not to read in a discussion. Code written in "
            "this project runs without network access, so anything it needs must be fetched here "
            "first."
        ),
        parameters={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": (
                        "A PDB identifier such as 4HHB, or a UniProt accession such as P01308 "
                        "when the source is alphafold."
                    ),
                },
                "source": {
                    "type": "string",
                    "enum": ["pdb", "alphafold"],
                    "description": "Whether to fetch an experimental structure or a prediction.",
                },
                "file_format": {
                    "type": "string",
                    "enum": ["cif", "pdb"],
                    "description": "Which format to download. Defaults to cif.",
                },
            },
            "required": ["identifier"],
        },
        function=partial(fetch_structure_file, save_dir),
    )


# The tools that need no configuration, which is all of them but the download
DATABASE_TOOLS: tuple[Tool, ...] = (
    PUBMED_TOOL,
    UNIPROT_LOOKUP_TOOL,
    UNIPROT_SEARCH_TOOL,
    PDB_LOOKUP_TOOL,
    ALPHAFOLD_LOOKUP_TOOL,
)

TOOL_REGISTRY: dict[str, Tool] = {tool.name: tool for tool in DATABASE_TOOLS}


def tools_for(*names: str) -> tuple[Tool, ...]:
    """Looks tools up by name.

    :param names: The tool names wanted.
    :raises KeyError: If a name is not registered, naming what is.
    :return: The tools, in the order asked for.
    """
    missing = [name for name in names if name not in TOOL_REGISTRY]

    if missing:
        raise KeyError(
            f"Unknown tool(s): {', '.join(missing)}. "
            f"Registered: {', '.join(sorted(TOOL_REGISTRY))}."
        )

    return tuple(TOOL_REGISTRY[name] for name in names)


def all_tools(save_dir: Path | None = None) -> tuple[Tool, ...]:
    """Returns every tool an agent can be given.

    :param save_dir: Where a downloaded structure file should be written. Without it the download
        tool is left out rather than given a default, since a library should not decide to write
        into a caller's working directory.
    :return: The tools.
    """
    if save_dir is None:
        return DATABASE_TOOLS

    return DATABASE_TOOLS + (structure_file_tool(save_dir / STRUCTURE_DIR_NAME),)


def run_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    tools: tuple[Tool, ...],
) -> tuple[list[str], list[ChatCompletionMessageParam]]:
    """Runs the tool calls requested by a model.

    A tool that fails reports the error back to the model as its output instead of ending the
    meeting, so the agent can correct the call, try a different query, or continue without it.
    A network hiccup on a literature search should not discard an entire meeting.

    :param tool_calls: The tool calls from the chat completion response.
    :param tools: The tools available in this meeting.
    :return: The tool outputs as strings, and the corresponding tool response messages.
    """
    name_to_tool = {tool.name: tool for tool in tools}

    tool_outputs: list[str] = []
    tool_messages: list[ChatCompletionMessageParam] = []

    for tool_call in tool_calls:
        name = tool_call.function.name
        tool = name_to_tool.get(name)

        if tool is None:
            available = ", ".join(sorted(name_to_tool)) or "none"
            output = f'Error: unknown tool "{name}". Available tools: {available}.'
            print(output)
        else:
            try:
                arguments = json.loads(tool_call.function.arguments)
                output = str(tool.function(**arguments))
            except Exception as e:
                output = f'Error running tool "{name}": {type(e).__name__}: {e}'
                print(output)

        tool_outputs.append(output)
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            }
        )

    return tool_outputs, tool_messages
