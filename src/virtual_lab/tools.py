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

from virtual_lab.chemistry import (
    get_activities,
    get_compound,
    get_drug,
    search_drugs,
    search_targets,
)
from virtual_lab.constants import (
    ARTIFACT_DIR_NAME,
    MAX_ACTIVITIES_REPORTED,
    MAX_PUBMED_ARTICLES,
    MAX_SEARCH_RESULTS,
    MAX_TOOL_ERROR_CHARACTERS,
    MAX_TOOL_NAME_CHARACTERS,
    MAX_TOOL_OUTPUT_CHARS,
    STRUCTURE_DIR_NAME,
)
from virtual_lab.databases import (
    download_structure,
    get_predicted_structure,
    get_protein,
    get_structure,
    search_proteins,
)
from virtual_lab.literature import (
    get_article,
    get_article_text,
    search_articles,
    search_preprints,
)
from virtual_lab.records import truncate_text
from virtual_lab.tables import describe_table, list_data_files
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
                "description": (
                    "The number of articles to return from the search query, "
                    f"at most {MAX_PUBMED_ARTICLES}."
                ),
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
    work_dir: Path,
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
        save_dir=work_dir / STRUCTURE_DIR_NAME,
        source=source,
        file_format=file_format,
        working_dir=work_dir,
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


def look_up_compound(identifier: str, namespace: str = "name") -> str:
    """Runs a PubChem lookup and renders it for a model."""
    return get_compound(identifier=identifier, namespace=namespace).report()


def look_up_drug(chembl_id: str) -> str:
    """Runs a ChEMBL molecule lookup and renders it for a model."""
    return get_drug(chembl_id=chembl_id).report()


def find_drugs(query: str, limit: int = 10) -> str:
    """Runs a ChEMBL molecule search and renders the results for a model."""
    return search_drugs(query=query, limit=limit).report()


def find_targets(
    query: str,
    organism: str = "",
    single_proteins_only: bool = True,
    limit: int = 10,
) -> str:
    """Runs a ChEMBL target search and renders the results for a model."""
    return search_targets(
        query=query,
        organism=organism,
        single_proteins_only=single_proteins_only,
        limit=limit,
    ).report()


def find_activities(
    target_chembl_id: str = "",
    molecule_chembl_id: str = "",
    activity_type: str = "",
    limit: int = MAX_ACTIVITIES_REPORTED,
) -> str:
    """Runs a ChEMBL bioactivity lookup and renders the results for a model."""
    return get_activities(
        target_chembl_id=target_chembl_id,
        molecule_chembl_id=molecule_chembl_id,
        activity_type=activity_type,
        limit=limit,
    ).report()


PUBCHEM_LOOKUP_TOOL = Tool(
    name="pubchem_lookup",
    description=(
        "Get a small molecule's formula, weight, SMILES, InChIKey, and computed descriptors from "
        "PubChem. Accepts a common name, a PubChem CID, a SMILES string, or an InChIKey; say "
        "which in the namespace argument. This is for chemical structure and properties. For "
        "whether a compound is a drug, what it acts on, or how potent it is, use chembl_lookup "
        "or chembl_activities instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "description": "The name, CID, SMILES, or InChIKey of the compound.",
            },
            "namespace": {
                "type": "string",
                "enum": ["name", "cid", "smiles", "inchikey"],
                "description": "Which kind of identifier was given. Defaults to name.",
            },
        },
        "required": ["identifier"],
    },
    function=look_up_compound,
)

CHEMBL_LOOKUP_TOOL = Tool(
    name="chembl_lookup",
    description=(
        "Get a drug's clinical status, route of administration, mechanism of action, and "
        "drug-likeness properties from ChEMBL. Requires a ChEMBL identifier such as CHEMBL25; "
        "use chembl_search first if you only have a name. Reports whether a molecule was "
        "approved, how far it got, and whether it was withdrawn, which a structure database "
        "cannot tell you."
    ),
    parameters={
        "type": "object",
        "properties": {
            "chembl_id": {
                "type": "string",
                "description": "A ChEMBL molecule identifier, such as CHEMBL25.",
            },
        },
        "required": ["chembl_id"],
    },
    function=look_up_drug,
)

CHEMBL_SEARCH_TOOL = Tool(
    name="chembl_search",
    description=(
        "Find drugs and compounds in ChEMBL by name, and get the identifiers to look up. Use "
        "this to turn a drug name into a ChEMBL identifier."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A drug or compound name."},
            "limit": {
                "type": "integer",
                "description": f"How many results to return, at most {MAX_SEARCH_RESULTS}.",
            },
        },
        "required": ["query"],
    },
    function=find_drugs,
)

CHEMBL_TARGET_SEARCH_TOOL = Tool(
    name="chembl_target_search",
    description=(
        "Find a protein target in ChEMBL by name or gene symbol, and get the identifier needed "
        "by chembl_activities. Searching a gene symbol unfiltered ranks protein complexes and "
        "other species above the protein itself, so give the organism and leave "
        "single_proteins_only on unless you specifically want complexes or cell lines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A target name or gene symbol, such as EGFR.",
            },
            "organism": {
                "type": "string",
                "description": 'The organism, such as "Homo sapiens". Strongly recommended.',
            },
            "single_proteins_only": {
                "type": "boolean",
                "description": (
                    "Exclude complexes, cell lines, and protein-protein interactions. "
                    "Defaults to true."
                ),
            },
            "limit": {
                "type": "integer",
                "description": f"How many results to return, at most {MAX_SEARCH_RESULTS}.",
            },
        },
        "required": ["query"],
    },
    function=find_targets,
)

CHEMBL_ACTIVITIES_TOOL = Tool(
    name="chembl_activities",
    description=(
        "Get the most potent measured interactions for a target or for a molecule from ChEMBL: "
        "what binds it, how tightly, and in what assay. Give a target identifier to find what "
        "acts on it, or a molecule identifier to find what it acts on; identifiers come from "
        "chembl_target_search and chembl_search. Results are ordered by pChEMBL, the negative "
        "log of the molar potency, and measurements from different assays are not directly "
        "comparable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_chembl_id": {
                "type": "string",
                "description": "A ChEMBL target identifier, such as CHEMBL203.",
            },
            "molecule_chembl_id": {
                "type": "string",
                "description": "A ChEMBL molecule identifier, such as CHEMBL25.",
            },
            "activity_type": {
                "type": "string",
                "description": 'Restrict to one kind of measurement, such as "IC50" or "Ki".',
            },
            "limit": {
                "type": "integer",
                "description": (
                    f"How many measurements to return, at most {MAX_ACTIVITIES_REPORTED}."
                ),
            },
        },
        "required": [],
    },
    function=find_activities,
)


def find_articles(
    query: str,
    limit: int = 10,
    sort: str = "relevance",
    open_access_only: bool = False,
    include_preprints: bool = True,
) -> str:
    """Runs a Europe PMC search and renders the results for a model."""
    return search_articles(
        query=query,
        limit=limit,
        sort=sort,
        open_access_only=open_access_only,
        include_preprints=include_preprints,
    ).report()


def look_up_article(identifier: str) -> str:
    """Runs a Europe PMC article lookup and renders it for a model."""
    return get_article(identifier=identifier).report()


def read_article(pmcid: str) -> str:
    """Fetches an open access full text and renders it for a model."""
    return get_article_text(pmcid=pmcid).report()


def find_preprints(
    query: str,
    limit: int = 10,
    category: str = "",
    sort: str = "relevance",
) -> str:
    """Runs an arXiv search and renders the results for a model."""
    return search_preprints(query=query, limit=limit, category=category, sort=sort).report()


EUROPE_PMC_SEARCH_TOOL = Tool(
    name="europepmc_search",
    description=(
        "Search the published biomedical literature through Europe PMC, which covers PubMed, "
        "PubMed Central, and the biology preprint servers. Returns titles, authors, journals, "
        "citation counts, and which results have a full text you can then read. Unquoted words "
        'must all appear, so quote a phrase to require the words together. Sort by "cited" when '
        'you want what the field is built on and by "recent" when you want what is new; the '
        "default ranking favours new papers, which have not been cited yet either way. Europe "
        "PMC does not index arXiv, so use arxiv_search as well for anything computational."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "limit": {
                "type": "integer",
                "description": f"How many results to return, at most {MAX_SEARCH_RESULTS}.",
            },
            "sort": {
                "type": "string",
                "enum": ["relevance", "cited", "recent"],
                "description": "How to order the results. Defaults to relevance.",
            },
            "open_access_only": {
                "type": "boolean",
                "description": (
                    "Only return articles whose full text can then be read with "
                    "europepmc_fulltext. Defaults to false."
                ),
            },
            "include_preprints": {
                "type": "boolean",
                "description": "Whether to include preprints. Defaults to true.",
            },
        },
        "required": ["query"],
    },
    function=find_articles,
)

EUROPE_PMC_LOOKUP_TOOL = Tool(
    name="europepmc_lookup",
    description=(
        "Get one article's abstract and details from Europe PMC by PMID, PMCID, or DOI. Use this "
        "when you have an identifier and want to know what the paper actually says, rather than "
        "inferring it from the title in a search result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "description": (
                    "A PubMed identifier such as 21937511, a PMCID such as PMC3258128, or a DOI."
                ),
            },
        },
        "required": ["identifier"],
    },
    function=look_up_article,
)

EUROPE_PMC_FULLTEXT_TOOL = Tool(
    name="europepmc_fulltext",
    description=(
        "Read the full text of an open access article held by Europe PMC. Needs a PMCID, which "
        "europepmc_search reports for the results that have one. References, funding, and "
        "author contribution sections are left out, and long sections are cut. Only some "
        "articles are open access; for the rest the abstract from europepmc_lookup is all there "
        "is. Ask for this when the abstract is not enough to settle a question, not by default: "
        "one article's text is a substantial part of what this meeting can hold."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pmcid": {
                "type": "string",
                "description": "A PubMed Central identifier such as PMC3258128.",
            },
        },
        "required": ["pmcid"],
    },
    function=read_article,
)

ARXIV_SEARCH_TOOL = Tool(
    name="arxiv_search",
    description=(
        "Search arXiv for preprints, which is where most machine learning and computational "
        "method work is published and much of it stays. Europe PMC does not index arXiv, so "
        "this is the only way to reach it here. Returns titles, authors, abstracts' subject "
        "categories, and any note saying the work was accepted somewhere. Everything returned "
        "is a preprint: unless a journal reference says otherwise, it has not been peer "
        "reviewed. Restrict with a category such as q-bio.BM or cs.LG when a term means "
        "different things in different fields."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to search for. Quote a phrase to require the words together."
                ),
            },
            "limit": {
                "type": "integer",
                "description": f"How many results to return, at most {MAX_SEARCH_RESULTS}.",
            },
            "category": {
                "type": "string",
                "description": (
                    "Restrict to one arXiv category, such as q-bio.BM, q-bio.QM, or cs.LG."
                ),
            },
            "sort": {
                "type": "string",
                "enum": ["relevance", "recent", "updated"],
                "description": "How to order the results. Defaults to relevance.",
            },
        },
        "required": ["query"],
    },
    function=find_preprints,
)


def show_data_files(work_dir: Path) -> str:
    """Lists the working directory and renders it for a model."""
    return list_data_files(work_dir).report()


def inspect_data_file(work_dir: Path, filename: str, sheet: str | None = None) -> str:
    """Describes one data file and renders it for a model."""
    return describe_table(work_dir=work_dir, filename=filename, sheet=sheet).report()


def data_file_tools(work_dir: Path) -> tuple[Tool, ...]:
    """Builds the tools that read the project's own data, bound to a directory.

    Bound rather than parameterised for the same reason the download is: which directory an
    agent may read is the lab's decision, and a directory a model can name is a directory a
    model can name /etc in.

    :param work_dir: The directory the sandboxed code will run in, which is the only one these
        tools can see.
    :return: The tools.
    """
    listing = Tool(
        name="data_files",
        description=(
            "List the data files in the project directory, with their sizes. Use this before "
            "assuming a file exists or guessing what it is called. It shows the same directory "
            "the code you write will run in, so the names it gives are the names to open."
        ),
        parameters={"type": "object", "properties": {}},
        function=partial(show_data_files, work_dir),
    )

    inspection = Tool(
        name="inspect_data_file",
        description=(
            "Look inside a CSV, TSV, or Excel file: how many rows and columns it has, what each "
            "column holds, how much is missing, and the ways the file will be read wrongly. Use "
            "this before writing any code that reads a data file, so the code is written "
            "against the columns that are there rather than the ones you would expect. It "
            "reports problems that raise nothing and change the answer, such as a column of "
            "measurements holding one '<0.001', identifiers whose leading zeros a numeric read "
            "would drop, and gene symbols a spreadsheet has turned into dates."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "The file, as data_files names it. It must be inside the project "
                        "directory; a path leading out of it is refused."
                    ),
                },
                "sheet": {
                    "type": "string",
                    "description": (
                        "Which sheet of a workbook to read, by name. The first sheet is used "
                        "when this is left out, and the report names the others."
                    ),
                },
            },
            "required": ["filename"],
        },
        function=partial(inspect_data_file, work_dir),
    )

    return (listing, inspection)


def structure_file_tool(work_dir: Path) -> Tool:
    """Builds the structure download tool, bound to a directory.

    This one is a function rather than a constant because it needs somewhere to write, and that
    somewhere must not be a parameter a model fills in.

    :param work_dir: The directory the sandboxed code will run in. Files are written to a
        subdirectory of it, so that code can open them by the path it is told.
    :return: The tool.
    """
    return Tool(
        name="fetch_structure_file",
        description=(
            "Download the atomic coordinates of a structure so that code can read them. The file "
            "is written where the code that will be run can open it, and the path to use is "
            "reported back. Use this rather than asking for coordinates directly: a structure "
            "file is hundreds of kilobytes and is for code to parse, not to read in a "
            "discussion. Code written in this project runs without network access, so anything "
            "it needs must be fetched here first."
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
        function=partial(fetch_structure_file, work_dir),
    )


# The tools that need no configuration, which is all of them but the download
DATABASE_TOOLS: tuple[Tool, ...] = (
    PUBMED_TOOL,
    UNIPROT_LOOKUP_TOOL,
    UNIPROT_SEARCH_TOOL,
    PDB_LOOKUP_TOOL,
    ALPHAFOLD_LOOKUP_TOOL,
    PUBCHEM_LOOKUP_TOOL,
    CHEMBL_LOOKUP_TOOL,
    CHEMBL_SEARCH_TOOL,
    CHEMBL_TARGET_SEARCH_TOOL,
    CHEMBL_ACTIVITIES_TOOL,
    EUROPE_PMC_SEARCH_TOOL,
    EUROPE_PMC_LOOKUP_TOOL,
    EUROPE_PMC_FULLTEXT_TOOL,
    ARXIV_SEARCH_TOOL,
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


def all_tools(save_dir: Path | None = None, save_name: str = "discussion") -> tuple[Tool, ...]:
    """Returns every tool an agent can be given.

    A downloaded file is only useful if the code that reads it can see it, and the only host
    directory the sandbox mounts is the meeting's artifact directory. So the download is bound
    inside that directory rather than beside it, and takes the same two arguments that decide
    where a meeting's code is written.

    :param save_dir: Where the meeting is being saved. Without it the tools that need a
        directory are left out rather than given a default, since a library should not decide to
        read or write in a caller's working directory.
    :param save_name: The name the meeting is saved under, which is the artifact subdirectory
        the code runs in.
    :return: The tools.
    """
    if save_dir is None:
        return DATABASE_TOOLS

    work_dir = save_dir / ARTIFACT_DIR_NAME / save_name

    return DATABASE_TOOLS + (structure_file_tool(work_dir),) + data_file_tools(work_dir)


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
        function = getattr(tool_call, "function", None)
        # Every tool here is a function, but the API also defines custom tool calls, which carry
        # their input in a different field and have no function to read a name from
        name = str(getattr(function, "name", ""))
        shown = truncate_text(name, MAX_TOOL_NAME_CHARACTERS)
        tool = name_to_tool.get(name)

        if function is None:
            kind = truncate_text(
                str(getattr(tool_call, "type", "unknown")), MAX_TOOL_NAME_CHARACTERS
            )
            output = f'Error: a tool call of type "{kind}" cannot be run; only function calls can.'
            print(output)
        elif tool is None:
            available = ", ".join(sorted(name_to_tool)) or "none"
            output = f'Error: unknown tool "{shown}". Available tools: {available}.'
            print(output)
        else:
            try:
                arguments = json.loads(function.arguments)
                output = str(tool.function(**arguments))
            except Exception as e:
                problem = truncate_text(f"{type(e).__name__}: {e}", MAX_TOOL_ERROR_CHARACTERS)
                output = f'Error running tool "{shown}": {problem}'
                print(output)

        output = truncate_text(output, MAX_TOOL_OUTPUT_CHARS)

        tool_outputs.append(output)
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            }
        )

    return tool_outputs, tool_messages
