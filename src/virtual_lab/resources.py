"""What a meeting's agents can use in their session: Biomni's tools, data, software, and know-how.

Biomni's agent is told in its system prompt which of Biomni's tool functions it can import,
which files of the data lake it can read, which software is installed, and what know-how it has.
Before each task, it asks its model which of these are relevant, and is told only of those. A
meeting with a session does the same before its first turn; see hold_meeting's resources.

The retrieval prompt, the parser for its answer, the listing of tool functions, and the wording
of the resource listing are adapted from Biomni (https://github.com/snap-stanford/Biomni, commit
400c1f366b96a35ca253e13c9b06c5076af41d65: biomni/model/retriever.py, biomni/agent/a1.py, and
textify_api_dict in biomni/utils.py), Copyright the Biomni authors, used under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Where they depart from Biomni, a comment
says how.

Nothing here imports Biomni's package. Its descriptions are read from their files as data, so
that listing the tools needs none of the libraries the tools themselves do, and its know-how
loader, which needs only the standard library, is loaded from its own file.
"""

import ast
import contextlib
import copy
import importlib.util
import inspect
import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from virtual_lab.constants import SOFTWARE_CHECK_TIMEOUT
from virtual_lab.execution import biomni_package_directory

# The modules of biomni.tool, in the order Biomni's read_module2api lists them
BIOMNI_TOOL_MODULES = (
    "literature",
    "biochemistry",
    "bioimaging",
    "bioengineering",
    "biophysics",
    "glycoengineering",
    "cancer_biology",
    "cell_biology",
    "molecular_biology",
    "genetics",
    "genomics",
    "immunology",
    "microbiology",
    "pathology",
    "pharmacology",
    "physiology",
    "synthetic_biology",
    "systems_biology",
    "support_tools",
    "database",
    "lab_automation",
    "protocols",
)

# Biomni's agent leaves its own interpreter out of the functions it lists, and the session is
# that interpreter
EXCLUDED_TOOLS = frozenset({"run_python_repl"})

RESOURCE_CATEGORIES = ("tools", "data_lake", "libraries", "know_how")


def biomni_source() -> Path:
    """The copy of Biomni's package that ships with virtual_lab."""
    return biomni_package_directory() / "biomni"


def read_literal(path: Path, name: str) -> Any:
    """Reads a value a Python file assigns to a name, without running the file.

    :param path: The file.
    :param name: The name the value is assigned to at the top level.
    :raises ValueError: If the file assigns nothing to that name, or not a literal.
    :return: The value.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)

    raise ValueError(f"{path} does not assign {name}")


@cache
def _biomni_tools() -> tuple[dict[str, Any], ...]:
    tools = []
    for module in BIOMNI_TOOL_MODULES:
        path = biomni_source() / "tool" / "tool_description" / f"{module}.py"
        for api in read_literal(path, "description"):
            if api["name"] not in EXCLUDED_TOOLS:
                tools.append({**api, "module": f"biomni.tool.{module}"})

    return tuple(tools)


def biomni_tools() -> tuple[dict[str, Any], ...]:
    """Biomni's tool functions, each described as Biomni describes it to its agent.

    :return: For each function, its name, description, required_parameters, and
        optional_parameters, as in Biomni's tool descriptions, and the module to import it from.
    """
    return copy.deepcopy(_biomni_tools())


@cache
def environment_descriptions(commercial_mode: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    """Biomni's descriptions of its data lake files and its software.

    :param commercial_mode: Whether to use the descriptions of Biomni's commercial mode, which
        leave out data that may not be used commercially.
    :return: The descriptions of the data lake's files and of the software, each by name.
    """
    path = biomni_source() / ("env_desc_cm.py" if commercial_mode else "env_desc.py")

    return read_literal(path, "data_lake_dict"), read_literal(path, "library_content_dict")


def commercial_data_lake() -> tuple[str, ...]:
    """The data lake files Biomni's commercial mode uses, to pass as download_data_lake's names."""
    return tuple(environment_descriptions(commercial_mode=True)[0])


@dataclass(frozen=True)
class Resource:
    """A data lake file or a piece of software, with what it is."""

    name: str
    description: str


@dataclass(frozen=True)
class KnowHow:
    """A know-how document: best practice, a protocol, or troubleshooting advice.

    :param id: Its file name without the extension.
    :param name: Its title.
    :param description: A sentence or two on what it covers.
    :param content: The document, without its metadata section.
    :param metadata: Its authors, license, and so on, as its metadata section gives them.
    """

    id: str
    name: str
    description: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


def allows_commercial_use(metadata: dict[str, str]) -> bool:
    """Whether a know-how document's metadata permits commercial use, as Biomni judges it."""
    commercial_use = metadata.get("commercial_use", "")

    return not ("❌" in commercial_use or "Not Allowed" in commercial_use or "Non-Commercial" in commercial_use)


def load_know_how(directory: Path | None = None, commercial_mode: bool = False) -> tuple[KnowHow, ...]:
    """Loads know-how documents with Biomni's own loader.

    :param directory: A directory of Markdown documents, defaulting to Biomni's own.
    :param commercial_mode: Whether to leave out documents that may not be used commercially.
    :return: The documents, ordered by id.
    """
    # Loaded from its file, since importing it as part of Biomni's package would run the
    # package's __init__, which the host has no need of
    path = biomni_source() / "know_how" / "loader.py"
    spec = importlib.util.spec_from_file_location("virtual_lab._biomni_know_how_loader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loader = module.KnowHowLoader(str(directory if directory is not None else biomni_source() / "know_how"))
    documents = []
    for doc_id in sorted(loader.documents):
        document = loader.documents[doc_id]
        if commercial_mode and not allows_commercial_use(document["metadata"]):
            continue
        documents.append(
            KnowHow(
                id=document["id"],
                name=document["name"],
                description=document["description"],
                content=document["content_without_metadata"],
                metadata=dict(document["metadata"]),
            )
        )

    return tuple(documents)


@dataclass(frozen=True)
class Resources:
    """What a meeting's agents are told they can use in their session.

    :param tools: Tool functions, each described as biomni_tools describes them, with its module.
    :param data_lake: Files of the data lake.
    :param libraries: Software installed in the environment.
    :param know_how: Know-how documents, given to the agents in full.
    :param data_lake_path: Where the data lake is, as the code sees it.
    :param not_installed: What of Biomni's software and tool modules was left out because the
        session does not have it, each with why; None if that was not checked.
    """

    tools: tuple[dict[str, Any], ...] = ()
    data_lake: tuple[Resource, ...] = ()
    libraries: tuple[Resource, ...] = ()
    know_how: tuple[KnowHow, ...] = ()
    data_lake_path: str | None = None
    not_installed: dict[str, str] | None = None

    def is_empty(self) -> bool:
        """Whether there is nothing to tell the agents of."""
        return not (self.tools or self.data_lake or self.libraries or self.know_how)

    def counts(self) -> dict[str, int]:
        """How many of each kind of resource there are."""
        return {category: len(getattr(self, category)) for category in RESOURCE_CATEGORIES}

    def names(self) -> dict[str, list[str]]:
        """Each resource by name, for a record: tools as module and function, know-how by id."""
        return {
            "tools": [f"{tool['module']}.{tool['name']}" for tool in self.tools],
            "data_lake": [item.name for item in self.data_lake],
            "libraries": [item.name for item in self.libraries],
            "know_how": [document.id for document in self.know_how],
        }


# Software Biomni lists under a name that is none of its Python distribution, its R package, or
# its command: HOMER is a set of Perl scripts, and Open Babel's command is obabel
SOFTWARE_COMMANDS = {
    "Homer": ("findMotifs.pl", "findMotifsGenome.pl", "homer2"),
    "openbabel": ("obabel",),
}

INSTALLED_MARKER = "VIRTUAL_LAB_INSTALLED "


def report_installed(libraries: list[str], commands: dict[str, list[str]], modules: list[str], timeout: float) -> None:
    """Prints which of the software is installed and which modules fail to import, as JSON after
    INSTALLED_MARKER.

    Its source is run in a session's interpreter, which the analysis shares, so it defines
    nothing outside itself and imports only the standard library. The modules are imported in a
    process of their own: importing all of Biomni's holds most of a gigabyte that the analysis
    may never need. A part of the check that cannot be done is reported as None.

    A piece of software is installed if it is a Python distribution, a module Python can find,
    an R package, or a command, under its name or, for a command, an alias.
    """
    import importlib.util
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    from importlib import metadata

    def normalized(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    distributions = {normalized(d.metadata["Name"]) for d in metadata.distributions() if d.metadata["Name"]}

    def importable(name: str) -> bool:
        # Only a top-level name is looked for: finding a.b would import a, in the analysis's
        # interpreter
        if not name.isidentifier():
            return False
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return False
        # The working directory is on the path, so a directory there, such as the output folder
        # of the program being looked for, is found as a namespace package, and a file there as
        # a module. Neither is the software.
        if spec is None or spec.origin is None:
            return False
        if spec.origin in ("built-in", "frozen"):
            return True
        here = os.path.realpath(os.getcwd())
        return not os.path.realpath(spec.origin).startswith(here + os.sep)

    r_packages: set[str] | None = set()
    if shutil.which("Rscript"):
        try:
            listed = subprocess.run(
                ["Rscript", "-e", "cat(rownames(installed.packages()), sep = '\\n')"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            r_packages = set(listed.stdout.split()) if listed.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            r_packages = None

    installed = None
    if r_packages is not None:
        installed = [
            name
            for name in libraries
            if normalized(name) in distributions
            or importable(name)
            or name in r_packages
            or any(shutil.which(command) for command in [name, name.lower(), *commands.get(name, [])])
        ]

    marker = "VIRTUAL_LAB_MODULES "
    script = (
        "import importlib, json, sys\n"
        "failed = {}\n"
        "for module in sys.argv[1:]:\n"
        "    try:\n"
        "        importlib.import_module(module)\n"
        "    except BaseException as error:\n"
        "        failed[module] = f'{type(error).__name__}: {error}'\n"
        f"print({marker!r} + json.dumps(failed))\n"
    )
    failed = None
    if not modules:
        failed = {}
    else:
        try:
            imported = subprocess.run(
                [sys.executable, "-c", script, *modules], capture_output=True, text=True, timeout=timeout
            )
            lines = [line for line in imported.stdout.splitlines() if line.startswith(marker)]
            if lines:
                failed = json.loads(lines[-1][len(marker) :])
        except (OSError, subprocess.SubprocessError, ValueError):
            failed = None

    print("VIRTUAL_LAB_INSTALLED " + json.dumps({"libraries": installed, "failed_modules": failed}))


def check_installed(
    session: Any, libraries: list[str], modules: list[str], what: str = "Biomni's software and tools"
) -> tuple[set[str], dict[str, str]] | None:
    """Finds which of Biomni's software a session has, and which of its tool modules import.

    Biomni lists all of its software to its agent, in whose environment it is all installed. A
    session here may run in a stage of the sandbox image that has only some of it, or in an
    environment of the user's own, and code written for software that is not there fails.

    :param session: The session, which is started if it is not running.
    :param libraries: The software to look for, by the names Biomni gives it.
    :param modules: The tool modules to import, such as biomni.tool.genomics.
    :param what: What is being looked for, for the warning if it cannot be.
    :return: The software that is missing, and each module that fails to import with its error;
        or None, with a warning, if the check could not be done.
    """
    code = (
        f"{inspect.getsource(report_installed)}\n"
        "try:\n"
        f"    report_installed({libraries!r}, {dict(SOFTWARE_COMMANDS)!r}, {modules!r}, {SOFTWARE_CHECK_TIMEOUT!r})\n"
        "finally:\n"
        "    del report_installed\n"
    )
    # Longer than both parts together, so that the check is always abandoned from within: a
    # session stopped at its limit would lose everything the analysis had defined
    result = session.check(code, timeout=2 * SOFTWARE_CHECK_TIMEOUT + 120)

    report = None
    lines = [line for line in result.output.splitlines() if line.startswith(INSTALLED_MARKER)]
    if lines:
        with contextlib.suppress(ValueError):
            report = json.loads(lines[-1][len(INSTALLED_MARKER) :])

    installed = report.get("libraries") if isinstance(report, dict) else None
    failed = report.get("failed_modules") if isinstance(report, dict) else None
    if not (
        isinstance(installed, list)
        and all(isinstance(name, str) for name in installed)
        and isinstance(failed, dict)
        and all(isinstance(name, str) and isinstance(error, str) for name, error in failed.items())
    ):
        detail = result.error or "it did not report what it found"
        print(
            f"Warning: could not check which of {what} the session has, so all of them are listed: {detail}"
        )
        return None

    return set(libraries) - set(installed), failed


def available_resources(session: Any, commercial_mode: bool = False) -> Resources:
    """Everything a session offers of Biomni's resources.

    Tools and software are offered only where code can import Biomni's tools, since both come
    with Biomni's environment, and then only the software that is installed and the tools whose
    modules import. Data lake files are offered only where the data lake is mounted. Know-how
    is offered everywhere: it is advice, not something to run.

    :param session: The session, as in DockerSession or LocalSession, which is started to check
        what it has installed.
    :param commercial_mode: Whether to leave out data and know-how that may not be used
        commercially, as Biomni's commercial mode does.
    :return: The resources.
    """
    data_lake_descriptions, library_descriptions = environment_descriptions(commercial_mode)
    data_lake_path = session.data_lake_path()

    files = session.data_lake_files() if data_lake_path is not None else []
    if commercial_mode:
        # Biomni's commercial mode only downloads the files it may use, then lists what it finds.
        # A data lake here may have been fetched in full, so what may not be used is left out.
        files = [name for name in files if name in data_lake_descriptions]

    tools: tuple[dict[str, Any], ...] = ()
    libraries: tuple[Resource, ...] = ()
    not_installed = None
    if session.has_biomni_tools():
        tools = biomni_tools()
        libraries = tuple(Resource(name, description) for name, description in library_descriptions.items())
        modules = [f"biomni.tool.{module}" for module in BIOMNI_TOOL_MODULES]

        checked = check_installed(session, [library.name for library in libraries], modules)
        if checked is not None:
            missing, failed = checked
            tools = tuple(tool for tool in tools if tool["module"] not in failed)
            libraries = tuple(library for library in libraries if library.name not in missing)
            not_installed = {
                **{name: "not installed" for name in library_descriptions if name in missing},
                **{module: failed[module] for module in modules if module in failed},
            }

    return Resources(
        tools=tools,
        data_lake=tuple(
            Resource(name, data_lake_descriptions.get(name, f"Data lake item: {name}")) for name in files
        ),
        libraries=libraries,
        know_how=load_know_how(commercial_mode=commercial_mode),
        data_lake_path=data_lake_path,
        not_installed=not_installed,
    )


def format_for_retrieval(items: list[tuple[str, str]]) -> str:
    """Numbers resources by name and description, as Biomni's retriever lists them."""
    formatted = [f"{index}. {name}: {description}" for index, (name, description) in enumerate(items)]

    return "\n".join(formatted) if formatted else "None available"


def retrieval_prompt(query: str, resources: Resources) -> str:
    """Asks a model which resources are relevant to a task, in the words Biomni's retriever uses.

    :param query: The task.
    :param resources: What there is to choose from.
    :return: The prompt.
    """
    tools = format_for_retrieval([(tool["name"], tool.get("description", "")) for tool in resources.tools])
    data_lake = format_for_retrieval([(item.name, item.description) for item in resources.data_lake])
    libraries = format_for_retrieval([(item.name, item.description) for item in resources.libraries])

    prompt_sections = [
        f"""
You are an expert biomedical research assistant. Your task is to select the relevant resources to help answer a user's query.

USER QUERY: {query}

Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

AVAILABLE TOOLS:
{tools}

AVAILABLE DATA LAKE ITEMS:
{data_lake}

AVAILABLE SOFTWARE LIBRARIES:
{libraries}"""
    ]

    if resources.know_how:
        know_how = format_for_retrieval([(document.name, document.description) for document in resources.know_how])
        prompt_sections.append(f"""
AVAILABLE KNOW-HOW DOCUMENTS (Best Practices & Protocols):
{know_how}""")

    response_format = """
For each category, respond with ONLY the indices of the relevant items in the following format:
TOOLS: [list of indices]
DATA_LAKE: [list of indices]
LIBRARIES: [list of indices]"""

    if resources.know_how:
        response_format += "\nKNOW_HOW: [list of indices]"

    response_format += """

For example:
TOOLS: [0, 3, 5, 7, 9]
DATA_LAKE: [1, 2, 4]
LIBRARIES: [0, 2, 4, 5, 8]"""

    if resources.know_how:
        response_format += "\nKNOW_HOW: [0, 1]"

    response_format += """

If a category has no relevant items, use an empty list, e.g., DATA_LAKE: []

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. ALWAYS prioritize database tools for general queries - include as many database tools as possible
3. Include all literature search tools
4. For wet lab sequence type of queries, ALWAYS include molecular biology tools
5. For data lake items, include datasets that could provide useful information
6. For libraries, include those that provide functions needed for analysis
7. For know-how documents, include those that provide relevant protocols, best practices, or troubleshooting guidance
8. Don't exclude resources just because they're not explicitly mentioned in the query
9. When in doubt about a database tool or molecular biology tool, include it rather than exclude it
"""

    return "\n".join(prompt_sections) + response_format


RETRIEVAL_PATTERNS = {
    "tools": re.compile(r"TOOLS:\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "data_lake": re.compile(r"DATA_LAKE:\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "libraries": re.compile(r"LIBRARIES:\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "know_how": re.compile(r"KNOW[-_]HOW:\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
}


def parse_retrieval(response: str) -> dict[str, list[int]] | None:
    """Reads the indices a model chose from its answer to retrieval_prompt.

    As in Biomni, a category whose list cannot be read is taken as empty.

    :param response: The model's answer.
    :return: The indices chosen in each category, or None if the answer names no category at
        all, which means it is not an answer to the prompt.
    """
    chosen: dict[str, list[int]] = {category: [] for category in RESOURCE_CATEGORIES}
    found = False

    for category, pattern in RETRIEVAL_PATTERNS.items():
        match = pattern.search(response)
        if match is None:
            continue
        found = True
        if match.group(1).strip():
            with contextlib.suppress(ValueError):
                chosen[category] = [int(index.strip()) for index in match.group(1).split(",") if index.strip()]

    return chosen if found else None


def select_resources(resources: Resources, chosen: dict[str, list[int]]) -> Resources:
    """Keeps the resources a model chose.

    Unlike Biomni's retriever, which accepts a negative index as counting from the end, an index
    outside the list is ignored, and the chosen resources keep the order they were offered in,
    once each, however the model listed them.

    :param resources: What there was to choose from.
    :param chosen: The indices chosen in each category, as parse_retrieval reads them.
    :return: The chosen resources.
    """

    def keep(items: tuple, indices: list[int]) -> tuple:
        wanted = {index for index in indices if 0 <= index < len(items)}
        return tuple(item for index, item in enumerate(items) if index in wanted)

    return Resources(
        tools=keep(resources.tools, chosen.get("tools", [])),
        data_lake=keep(resources.data_lake, chosen.get("data_lake", [])),
        libraries=keep(resources.libraries, chosen.get("libraries", [])),
        know_how=keep(resources.know_how, chosen.get("know_how", [])),
        data_lake_path=resources.data_lake_path,
        not_installed=resources.not_installed,
    )


def textify_api_dict(api_dict: dict[str, list[dict[str, Any]]]) -> str:
    """Lists tool functions by module, with their parameters, as Biomni's textify_api_dict does."""
    lines = []
    for category, methods in api_dict.items():
        lines.append(f"Import file: {category}")
        lines.append("=" * (len("Import file: ") + len(category)))
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(f"  Description: {method.get('description', 'No description provided.')}")

            for title, key in (("Required Parameters", "required_parameters"), ("Optional Parameters", "optional_parameters")):
                parameters = method.get(key, [])
                if parameters:
                    lines.append(f"  {title}:")
                    for parameter in parameters:
                        name = parameter.get("name", "N/A")
                        kind = parameter.get("type", "N/A")
                        description = parameter.get("description", "No description")
                        default = parameter.get("default", "None")
                        lines.append(f"    - {name} ({kind}): {description} [Default: {default}]")

            lines.append("")
        lines.append("")

    return "\n".join(lines)


def format_item_with_description(name: str, description: str) -> str:
    """Lists a resource with its description, wrapped as Biomni wraps it."""
    if not description:
        description = f"Data lake item: {name}"

    if len(description) <= 80:
        return f"{name}: {description}"

    wrapped, line = [], ""
    for word in description.split():
        if len(line) + len(word) + 1 <= 80:
            line = f"{line} {word}" if line else word
        else:
            wrapped.append(line)
            line = word
    if line:
        wrapped.append(line)

    return f"{name}:\n  " + "\n  ".join(wrapped)


def resources_prompt(resources: Resources, code_actions: str, retrieved: bool) -> str:
    """Tells the meeting what it can use in its session, in the terms Biomni tells its agent.

    :param resources: The resources.
    :param code_actions: "tool" or "tags", as in hold_meeting, which decides how R and bash code
        is to be marked.
    :param retrieved: Whether the resources were chosen for this meeting, rather than all there are.
    :return: The prompt, or an empty string if there is nothing to tell.
    """
    if resources.is_empty():
        return ""

    chosen = "Based on the agenda, these are the most relevant of" if retrieved else "These are"
    sections = [f"{chosen} the resources of Biomni's environment available in the session."]

    if resources.know_how:
        documents = "\n\n".join(f"{document.name}:\n{document.content}" for document in resources.know_how)
        sections.append(
            "KNOW-HOW DOCUMENTS (BEST PRACTICES & PROTOCOLS - ALREADY LOADED):\n"
            f"{documents}\n\n"
            "These documents are already in your context: use them directly for experimental "
            "design, methodology, parameters, and troubleshooting, without a separate step to "
            "retrieve or review them."
        )

    if resources.tools:
        by_module: dict[str, list[dict[str, Any]]] = {}
        for tool in resources.tools:
            by_module.setdefault(tool["module"], []).append(tool)
        sections.append(
            "- Function Dictionary:\n"
            "Functions you can call in your code, grouped by the module to import them from:\n"
            f"---\n{textify_api_dict(by_module)}---\n\n"
            "IMPORTANT: When using any function, you MUST first import it from its module. For "
            "example:\nfrom [module_name] import [function_name]\n"
            "When calling these functions, save the output and print it, for example "
            "result = understand_scRNA(XXX); print(result). Otherwise nobody will know what was done."
        )

    if resources.data_lake:
        items = "\n".join(format_item_with_description(item.name, item.description) for item in resources.data_lake)
        sections.append(
            f"- Biological data lake\nYou can access a biological data lake at the following path: "
            f"{resources.data_lake_path}.\nEach item is listed with its description to help you "
            f"understand its contents.\n----\n{items}\n----"
        )

    if resources.libraries:
        items = "\n".join(format_item_with_description(item.name, item.description) for item in resources.libraries)
        how = (
            "start the block with #!R for R and #!BASH for command-line tools"
            if code_actions == "tags"
            else 'run R with language="r" and command-line tools with language="bash"'
        )
        sections.append(
            "- Software Library:\nThe environment supports a list of libraries that can be "
            "directly used. Do not forget the import statement. Each library is listed with its "
            f"description to help you understand its functionality.\n----\n{items}\n----\n"
            f"To use R packages or command-line software, {how}."
        )

    if any(tool["module"] == "biomni.tool.protocols" for tool in resources.tools):
        sections.append(
            "PROTOCOL GENERATION:\nIf an experimental protocol is needed, use search_protocols(), "
            "list_local_protocols(), and read_local_protocol() to generate an accurate protocol. "
            "Include details such as reagents (with catalog numbers if available), equipment "
            "specifications, replicate requirements, error handling, and troubleshooting - but ONLY "
            "include information found in these resources. Do not make up specifications, catalog "
            "numbers, or equipment details. Prioritize accuracy over completeness."
        )

    return "\n\n".join(sections)
