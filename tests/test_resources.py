"""Tests for Biomni's resources: its tools in the sandbox, and choosing what a meeting is told of."""

import inspect
import json
import sys
from pathlib import Path

import pytest

from virtual_lab.agent import Agent
from virtual_lab.constants import (
    METADATA_DIR_NAME,
    SANDBOX_BIOMNI_PACKAGE_DIR,
    SANDBOX_BIOMNI_PATH,
    SANDBOX_DATA_LAKE_DIR,
    SOFTWARE_CHECK_TIMEOUT,
)
from virtual_lab.execution import (
    DockerExecutor,
    ExecutionError,
    LocalExecutor,
    biomni_package_directory,
)
from virtual_lab.repair import describe_executor
from virtual_lab.resources import (
    BIOMNI_TOOL_MODULES,
    INSTALLED_MARKER,
    KnowHow,
    Resource,
    Resources,
    allows_commercial_use,
    available_resources,
    biomni_tools,
    check_installed,
    commercial_data_lake,
    environment_descriptions,
    format_item_with_description,
    load_know_how,
    parse_retrieval,
    read_literal,
    report_installed,
    resources_prompt,
    retrieval_prompt,
    select_resources,
    textify_api_dict,
)
from virtual_lab.run_meeting import hold_meeting
from virtual_lab.session import CellResult, DockerSession, LocalSession, list_data_lake, session_executor
from virtual_lab.utils import BudgetExceededError

from conftest import FakeClient, text_response


def env_arguments(arguments: tuple[str, ...]) -> list[str]:
    return [arguments[index + 1] for index, item in enumerate(arguments) if item == "--env"]


def mounts(arguments: tuple[str, ...]) -> list[str]:
    return [arguments[index + 1] for index, item in enumerate(arguments) if item == "--mount"]


def command(executor: DockerExecutor, directory: Path = Path("/meeting")) -> tuple[str, ...]:
    return executor.build_command(directory=directory, command=("true",), container_name="probe")


class TestBiomniPackage:
    def test_the_copy_is_complete_enough_to_import_its_tools(self) -> None:
        package = biomni_package_directory() / "biomni"

        for module in BIOMNI_TOOL_MODULES:
            assert (package / "tool" / f"{module}.py").is_file()
            assert (package / "tool" / "tool_description" / f"{module}.py").is_file()
        for name in ("__init__.py", "config.py", "llm.py", "utils.py", "env_desc.py", "env_desc_cm.py"):
            assert (package / name).is_file()
        assert list((package / "tool" / "protocols").rglob("*.txt"))
        assert list((package / "tool" / "schema_db").glob("*.pkl"))
        assert (biomni_package_directory() / "LICENSE").is_file()
        assert (biomni_package_directory() / "NOTICE").is_file()

    def test_nothing_else_is_on_the_path_beside_it(self) -> None:
        # Everything in the directory becomes importable in the sandbox
        entries = {path.name for path in biomni_package_directory().iterdir()}

        assert entries == {"biomni", "LICENSE", "NOTICE", "license_info.md"}

    def test_it_is_left_out_of_the_image_build(self) -> None:
        ignored = (biomni_package_directory().parent / ".dockerignore").read_text().split()

        assert "biomni_package" in ignored


class TestToolDescriptions:
    def test_every_tool_is_read_with_its_module(self) -> None:
        tools = biomni_tools()
        names = [tool["name"] for tool in tools]

        assert len(tools) == 223
        assert len(set(names)) == len(names)
        assert {tool["module"] for tool in tools} == {f"biomni.tool.{module}" for module in BIOMNI_TOOL_MODULES}
        assert all({"name", "description", "required_parameters", "module"} <= set(tool) for tool in tools)

    def test_the_interpreter_tool_is_left_out(self) -> None:
        # The session is the interpreter
        assert "run_python_repl" not in {tool["name"] for tool in biomni_tools()}
        assert "read_function_source_code" in {tool["name"] for tool in biomni_tools()}

    def test_modules_are_in_biomnis_order(self) -> None:
        modules = list(dict.fromkeys(tool["module"] for tool in biomni_tools()))

        assert modules == [f"biomni.tool.{module}" for module in BIOMNI_TOOL_MODULES]

    def test_a_caller_cannot_change_the_cached_descriptions(self) -> None:
        biomni_tools()[0]["name"] = "changed"
        biomni_tools()[0]["required_parameters"].append("changed")

        assert biomni_tools()[0]["name"] != "changed"
        assert "changed" not in biomni_tools()[0]["required_parameters"]

    def test_read_literal_does_not_run_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "module.py"
        path.write_text("import sys\nsys.exit(1)\nvalue = {'a': [1, None]}\n")

        assert read_literal(path, "value") == {"a": [1, None]}
        with pytest.raises(ValueError, match="does not assign"):
            read_literal(path, "other")

        path.write_text("value = open('x')\n")
        with pytest.raises(ValueError):
            read_literal(path, "value")


class TestEnvironmentDescriptions:
    def test_commercial_mode_leaves_out_restricted_data(self) -> None:
        data_lake, libraries = environment_descriptions()
        commercial, commercial_libraries = environment_descriptions(commercial_mode=True)

        assert len(data_lake) == 76 and len(commercial) == 41
        assert set(commercial) < set(data_lake)
        assert "BindingDB_All_202409.tsv" in data_lake
        assert "BindingDB_All_202409.tsv" not in commercial
        assert set(commercial_libraries) <= set(libraries)
        assert commercial_data_lake() == tuple(commercial)

    def test_matches_the_copy_the_environment_module_uses(self) -> None:
        from virtual_lab.biomni_env_desc import data_lake_dict, library_content_dict

        assert environment_descriptions() == (data_lake_dict, library_content_dict)


class TestKnowHow:
    def test_biomnis_documents_load_without_their_metadata(self) -> None:
        documents = load_know_how()

        assert [document.id for document in documents] == ["sgRNA_design_guide", "single_cell_annotation"]
        for document in documents:
            assert document.name and document.description
            assert "## Metadata" not in document.content
            assert document.content.startswith("# ")
            assert "commercial_use" in document.metadata

    def test_commercial_mode_leaves_out_documents_that_forbid_it(self, tmp_path: Path) -> None:
        for name, use in (("open", "✅ Allowed"), ("closed", "❌ Not Allowed"), ("academic", "Non-Commercial only")):
            (tmp_path / f"{name}.md").write_text(
                f"# {name.title()}\n\n## Metadata\n\n**Commercial Use**: {use}\n\n---\n\n## Overview\n\nAbout {name}.\n"
            )

        assert [document.id for document in load_know_how(tmp_path)] == ["academic", "closed", "open"]
        assert [document.id for document in load_know_how(tmp_path, commercial_mode=True)] == ["open"]

    @pytest.mark.parametrize(
        ("use", "allowed"),
        [("✅ Allowed", True), ("", True), ("❌ No", False), ("Not Allowed", False), ("Non-Commercial", False)],
    )
    def test_allows_commercial_use(self, use: str, allowed: bool) -> None:
        assert allows_commercial_use({"commercial_use": use}) is allowed


EVERY_LIBRARY = sorted({*environment_descriptions(False)[1], *environment_descriptions(True)[1]})


def installed_report(missing: tuple[str, ...] = (), failed: dict[str, str] | None = None) -> str:
    """What report_installed prints in an environment missing some software and modules."""
    libraries = [name for name in EVERY_LIBRARY if name not in missing]
    return INSTALLED_MARKER + json.dumps({"libraries": libraries, "failed_modules": failed or {}}) + "\n"


def reported(output: str, status: str = "ok", error: str | None = None) -> CellResult:
    return CellResult(language="python", code="", status=status, output=output, error=error, duration=0.1)


class FakeSession:
    def __init__(
        self,
        tools: bool = True,
        lake: str | None = SANDBOX_DATA_LAKE_DIR,
        files: tuple[str, ...] = (),
        report: CellResult | None = None,
    ) -> None:
        self.tools, self.lake, self.files = tools, lake, list(files)
        self.report = report if report is not None else reported(installed_report())
        self.checks: list[tuple[str, float | None]] = []

    def has_biomni_tools(self) -> bool:
        return self.tools

    def data_lake_path(self) -> str | None:
        return self.lake

    def data_lake_files(self) -> list[str]:
        return self.files

    def check(self, code: str, timeout: float | None = None) -> CellResult:
        self.checks.append((code, timeout))
        return self.report


class TestAvailableResources:
    def test_everything_in_biomnis_environment(self) -> None:
        resources = available_resources(FakeSession(files=("DisGeNET.parquet", "mine.csv")))

        assert len(resources.tools) == 223
        assert len(resources.libraries) == 113
        assert resources.data_lake == (
            Resource("DisGeNET.parquet", "Gene-disease associations from multiple sources."),
            Resource("mine.csv", "Data lake item: mine.csv"),
        )
        assert resources.data_lake_path == SANDBOX_DATA_LAKE_DIR
        assert len(resources.know_how) == 2

    def test_without_biomnis_tools_only_know_how_and_data(self) -> None:
        resources = available_resources(FakeSession(tools=False, files=("DisGeNET.parquet",)))

        assert resources.tools == () and resources.libraries == ()
        assert [item.name for item in resources.data_lake] == ["DisGeNET.parquet"]
        assert resources.know_how

    def test_no_data_without_a_mounted_lake(self) -> None:
        resources = available_resources(FakeSession(lake=None, files=("DisGeNET.parquet",)))

        assert resources.data_lake == () and resources.data_lake_path is None

    def test_commercial_mode(self) -> None:
        resources = available_resources(
            FakeSession(files=("BindingDB_All_202409.tsv", "affinity_capture-ms.parquet", "mine.csv")),
            commercial_mode=True,
        )

        assert [item.name for item in resources.data_lake] == ["affinity_capture-ms.parquet"]
        assert len(resources.libraries) == len(environment_descriptions(True)[1])

    def test_leaves_out_what_the_session_does_not_have(self) -> None:
        error = "ModuleNotFoundError: No module named 'esm'"
        session = FakeSession(
            report=reported(installed_report(missing=("DESeq2", "hyperopt"), failed={"biomni.tool.genomics": error}))
        )

        resources = available_resources(session)

        assert not any(tool["module"] == "biomni.tool.genomics" for tool in resources.tools)
        genomics = sum(tool["module"] == "biomni.tool.genomics" for tool in biomni_tools())
        assert genomics > 0 and len(resources.tools) == 223 - genomics
        names = [library.name for library in resources.libraries]
        assert "DESeq2" not in names and "hyperopt" not in names and len(names) == 111
        # In Biomni's order: software first, then modules
        assert resources.not_installed == {
            "hyperopt": "not installed",
            "DESeq2": "not installed",
            "biomni.tool.genomics": error,
        }
        code, timeout = session.checks[0]
        assert "'DESeq2'" in code and "'biomni.tool.genomics'" in code
        assert timeout == 2 * SOFTWARE_CHECK_TIMEOUT + 120

    def test_everything_installed_leaves_nothing_out(self) -> None:
        resources = available_resources(FakeSession())

        assert resources.not_installed == {}

    @pytest.mark.parametrize(
        "report",
        [
            reported("", status="error", error="NameError: boom"),
            reported("", status="lost", error="it was killed"),
            reported("nothing useful\n"),
            reported(INSTALLED_MARKER + "{not json\n"),
            reported(INSTALLED_MARKER + json.dumps({"libraries": None, "failed_modules": {}}) + "\n"),
            reported(INSTALLED_MARKER + json.dumps({"libraries": [{}], "failed_modules": {}}) + "\n"),
            reported(INSTALLED_MARKER + json.dumps([]) + "\n"),
            reported(INSTALLED_MARKER + json.dumps({"libraries": [], "failed_modules": None}) + "\n"),
            reported(INSTALLED_MARKER + json.dumps({"libraries": [], "failed_modules": {"m": 1}}) + "\n"),
        ],
    )
    def test_a_check_that_fails_lists_everything(self, report: CellResult, capsys: pytest.CaptureFixture) -> None:
        resources = available_resources(FakeSession(report=report))

        assert len(resources.tools) == 223 and len(resources.libraries) == 113
        assert resources.not_installed is None
        assert "could not check which of Biomni's software" in capsys.readouterr().out

    def test_nothing_is_checked_without_biomnis_tools(self) -> None:
        session = FakeSession(tools=False)

        resources = available_resources(session)

        assert session.checks == [] and resources.not_installed is None


def write_command(directory: Path, name: str, script: str) -> None:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)


class TestCheckInstalled:
    def test_finds_distributions_commands_and_modules(self, capsys: pytest.CaptureFixture) -> None:
        report_installed(
            ["PyTest", "typing-extensions", "sh", "Homer", "not-a-real-package-xyz"],
            {"Homer": ["sh"]},
            ["json", "not_a_real_module_xyz"],
            60,
        )

        line = capsys.readouterr().out.strip()
        assert line.startswith(INSTALLED_MARKER)
        found = json.loads(line[len(INSTALLED_MARKER) :])
        # Distribution names are matched however they are spelled, and commands by name or alias
        assert found["libraries"] == ["PyTest", "typing-extensions", "sh", "Homer"]
        assert list(found["failed_modules"]) == ["not_a_real_module_xyz"]
        assert found["failed_modules"]["not_a_real_module_xyz"].startswith("ModuleNotFoundError")

    def test_r_packages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        write_command(tmp_path, "Rscript", "printf 'DESeq2\\nlimma\\n'")
        monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")

        report_installed(["DESeq2", "edgeR", "limma"], {}, [], 60)

        found = json.loads(capsys.readouterr().out.strip()[len(INSTALLED_MARKER) :])
        assert found["libraries"] == ["DESeq2", "limma"]

    def test_r_that_fails_is_not_a_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        write_command(tmp_path, "Rscript", "exit 1")
        monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")

        report_installed(["DESeq2"], {}, [], 60)

        assert json.loads(capsys.readouterr().out.strip()[len(INSTALLED_MARKER) :])["libraries"] is None

    def test_runs_in_the_session_without_disturbing_it(self, tmp_path: Path) -> None:
        with LocalSession(tmp_path, warn=False) as session:
            session.run("kept = 42")

            checked = check_installed(session, ["pytest", "not-a-real-package-xyz"], ["json", "not_a_real_module_xyz"])

            assert checked is not None
            missing, failed = checked
            assert missing == {"not-a-real-package-xyz"}
            assert list(failed) == ["not_a_real_module_xyz"]
            # Not part of the analysis: not in the history, and nothing left behind
            assert len(session.history) == 1
            after = session.run("print(kept, 'report_installed' in globals())")
            assert after.output.strip() == "42 False"

    def test_finds_modules_by_name_without_importing_anything(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import types

        # A module whose spec cannot be read makes find_spec raise rather than answer
        monkeypatch.setitem(sys.modules, "spec_less_xyz", types.ModuleType("spec_less_xyz"))
        monkeypatch.delitem(sys.modules, "xml.dom", raising=False)

        report_installed(["_pytest", "xml.dom", "spec_less_xyz", "not_a_real_module_xyz"], {}, [], 60)

        found = json.loads(capsys.readouterr().out.strip()[len(INSTALLED_MARKER) :])
        # _pytest is no distribution's name, only a module's
        assert found["libraries"] == ["_pytest"]
        # Looking for xml.dom would have imported xml
        assert "xml.dom" not in sys.modules

    def test_what_is_in_the_working_directory_is_not_software(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        # As in a session, whose working directory is first on the path
        (tmp_path / "output_folder_xyz").mkdir()
        (tmp_path / "package_xyz").mkdir()
        (tmp_path / "package_xyz" / "__init__.py").write_text("")
        (tmp_path / "script_xyz.py").write_text("")
        monkeypatch.chdir(tmp_path)
        # Through a link, as the same directory can be reached by another path
        (tmp_path.parent / f"{tmp_path.name}-link").symlink_to(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path.parent / f"{tmp_path.name}-link"))

        report_installed(["output_folder_xyz", "package_xyz", "script_xyz", "_pytest", "sys"], {}, [], 60)

        found = json.loads(capsys.readouterr().out.strip()[len(INSTALLED_MARKER) :])
        assert found["libraries"] == ["_pytest", "sys"]

    def test_with_no_modules_none_are_imported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import subprocess

        ran: list[object] = []
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: ran.append(args))

        report_installed(["pytest"], {}, [], 60)

        found = json.loads(capsys.readouterr().out.strip()[len(INSTALLED_MARKER) :])
        assert found == {"libraries": ["pytest"], "failed_modules": {}}
        assert ran == []

    def test_a_check_that_fails_says_what_it_was_for(self, capsys: pytest.CaptureFixture) -> None:
        session = FakeSession(report=reported("nothing\n"))

        assert check_installed(session, ["mine"], [], what="the software added to it") is None
        assert "could not check which of the software added to it the session has" in capsys.readouterr().out

    def test_a_folder_the_code_made_is_not_software_in_the_session(self, tmp_path: Path) -> None:
        with LocalSession(tmp_path / "work", warn=False) as session:
            session.run("import os\nos.makedirs('fastqc_xyz')")

            checked = check_installed(session, ["fastqc_xyz", "pytest"], [])

        assert checked == ({"fastqc_xyz"}, {})

    def test_the_marker_is_the_one_printed(self) -> None:
        # The function's source runs in the session, where it cannot see this module's constant
        assert repr(INSTALLED_MARKER)[1:-1] in inspect.getsource(report_installed)


def small_resources() -> Resources:
    return Resources(
        tools=(
            {"name": "find_gene", "description": "Finds a gene.", "required_parameters": [], "module": "biomni.tool.genetics"},
            {"name": "search_protocols", "description": "Searches protocols.", "required_parameters": [], "module": "biomni.tool.protocols"},
            {"name": "blast", "description": "Runs BLAST.", "required_parameters": [], "module": "biomni.tool.genetics"},
        ),
        data_lake=(Resource("a.parquet", "Table A."), Resource("b.csv", "Table B.")),
        libraries=(Resource("scanpy", "Single-cell analysis."), Resource("samtools", "Alignments.")),
        know_how=(KnowHow("guide", "A Guide", "How to.", "# A Guide\n\nDo this."),),
        data_lake_path="/lake",
    )


class TestRetrievalPrompt:
    def test_lists_every_resource_by_index(self) -> None:
        prompt = retrieval_prompt("Find a gene for fever.", small_resources())

        assert "USER QUERY: Find a gene for fever." in prompt
        assert "AVAILABLE TOOLS:\n0. find_gene: Finds a gene.\n1. search_protocols: Searches protocols.\n2. blast: Runs BLAST." in prompt
        assert "AVAILABLE DATA LAKE ITEMS:\n0. a.parquet: Table A.\n1. b.csv: Table B." in prompt
        assert "AVAILABLE SOFTWARE LIBRARIES:\n0. scanpy: Single-cell analysis." in prompt
        assert "AVAILABLE KNOW-HOW DOCUMENTS (Best Practices & Protocols):\n0. A Guide: How to." in prompt
        assert "KNOW_HOW: [list of indices]" in prompt
        assert "ALWAYS prioritize database tools" in prompt

    def test_know_how_is_asked_for_only_when_there_is_some(self) -> None:
        prompt = retrieval_prompt("Q", Resources(tools=small_resources().tools))

        assert "KNOW" not in prompt
        assert "AVAILABLE DATA LAKE ITEMS:\nNone available" in prompt


class TestParseRetrieval:
    def test_reads_each_category(self) -> None:
        answer = "Here:\nTOOLS: [0, 2]\nDATA_LAKE: []\nLIBRARIES: [1]\nKNOW-HOW: [0]"

        assert parse_retrieval(answer) == {"tools": [0, 2], "data_lake": [], "libraries": [1], "know_how": [0]}

    def test_case_and_line_breaks_inside_a_list(self) -> None:
        assert parse_retrieval("tools: [0,\n 1]\nknow_how: [0]")["tools"] == [0, 1]

    def test_unreadable_list_is_empty(self) -> None:
        assert parse_retrieval("TOOLS: [all of them]\nLIBRARIES: [2]") == {
            "tools": [],
            "data_lake": [],
            "libraries": [2],
            "know_how": [],
        }

    def test_an_answer_naming_no_category_is_not_an_answer(self) -> None:
        assert parse_retrieval("I would use BLAST and scanpy.") is None


class TestSelectResources:
    def test_keeps_what_was_chosen_in_the_order_offered(self) -> None:
        selected = select_resources(
            small_resources(), {"tools": [2, 0, 2], "data_lake": [1], "libraries": [], "know_how": [0]}
        )

        assert [tool["name"] for tool in selected.tools] == ["find_gene", "blast"]
        assert [item.name for item in selected.data_lake] == ["b.csv"]
        assert selected.libraries == ()
        assert selected.know_how == small_resources().know_how
        assert selected.data_lake_path == "/lake"

    def test_indices_outside_the_list_are_ignored(self) -> None:
        # Biomni's retriever would take -1 as the last tool
        selected = select_resources(small_resources(), {"tools": [-1, 3, 99, 1]})

        assert [tool["name"] for tool in selected.tools] == ["search_protocols"]

    def test_names_and_counts(self) -> None:
        resources = small_resources()

        assert resources.counts() == {"tools": 3, "data_lake": 2, "libraries": 2, "know_how": 1}
        assert resources.names()["tools"][0] == "biomni.tool.genetics.find_gene"
        assert resources.names()["know_how"] == ["guide"]
        assert Resources().is_empty() and not resources.is_empty()


class TestResourcesPrompt:
    def test_textify_matches_biomnis_layout(self) -> None:
        text = textify_api_dict(
            {
                "biomni.tool.x": [
                    {
                        "name": "f",
                        "description": "Does f.",
                        "required_parameters": [{"name": "a", "type": "str", "description": "A.", "default": None}],
                        "optional_parameters": [{"name": "b", "type": "int", "description": "B.", "default": 2}],
                    }
                ]
            }
        )

        assert text == (
            "Import file: biomni.tool.x\n"
            "==========================\n"
            "Method: f\n"
            "  Description: Does f.\n"
            "  Required Parameters:\n"
            "    - a (str): A. [Default: None]\n"
            "  Optional Parameters:\n"
            "    - b (int): B. [Default: 2]\n"
            "\n"
        )

    def test_long_descriptions_are_wrapped(self) -> None:
        text = format_item_with_description("name", "word " * 30)

        assert text.startswith("name:\n  word")
        assert all(len(line) <= 82 for line in text.splitlines())
        assert format_item_with_description("short", "Brief.") == "short: Brief."
        assert format_item_with_description("x", "") == "x: Data lake item: x"

    def test_lists_everything_given(self) -> None:
        prompt = resources_prompt(small_resources(), "tool", retrieved=True)

        assert prompt.startswith("Based on the agenda, these are the most relevant of")
        assert "A Guide:\n# A Guide\n\nDo this." in prompt
        assert "Import file: biomni.tool.genetics" in prompt and "Method: blast" in prompt
        assert "from [module_name] import [function_name]" in prompt
        assert "following path: /lake." in prompt and "a.parquet: Table A." in prompt
        assert "scanpy: Single-cell analysis." in prompt
        assert 'language="r"' in prompt
        assert "PROTOCOL GENERATION" in prompt

    def test_tags_mode_and_sections_left_out(self) -> None:
        resources = Resources(libraries=small_resources().libraries)
        prompt = resources_prompt(resources, "tags", retrieved=False)

        assert prompt.startswith("These are the resources")
        assert "#!R" in prompt and "#!BASH" in prompt
        for absent in ("Function Dictionary", "data lake", "KNOW-HOW", "PROTOCOL GENERATION"):
            assert absent not in prompt

    def test_nothing_to_list(self) -> None:
        assert resources_prompt(Resources(), "tool", retrieved=True) == ""


class TestExecutorOptions:
    def test_biomni_tools_are_mounted_read_only_on_the_path(self) -> None:
        arguments = command(DockerExecutor(biomni_tools=True))

        assert f"type=bind,source={biomni_package_directory()},target={SANDBOX_BIOMNI_PACKAGE_DIR},readonly" in mounts(arguments)
        assert f"PYTHONPATH={SANDBOX_BIOMNI_PACKAGE_DIR}" in env_arguments(arguments)
        assert f"BIOMNI_PATH={SANDBOX_BIOMNI_PATH}" in env_arguments(arguments)

    def test_nothing_of_biomnis_without_asking(self) -> None:
        arguments = command(DockerExecutor())

        assert len(mounts(arguments)) == 1
        assert not any(value.startswith(("PYTHONPATH", "BIOMNI")) for value in env_arguments(arguments))

    def test_forwarded_variables_are_named_without_their_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_API_KEY", "secret-value")
        arguments = command(DockerExecutor(forward_env=("SOME_API_KEY",), environment={"BIOMNI_LLM": "gpt-5"}))

        assert "SOME_API_KEY" in env_arguments(arguments)
        assert "BIOMNI_LLM=gpt-5" in env_arguments(arguments)
        assert not any("secret-value" in argument for argument in arguments)

    def test_a_missing_variable_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ABSENT_KEY", raising=False)

        with pytest.raises(ExecutionError, match="ABSENT_KEY.*not set"):
            command(DockerExecutor(forward_env=("ABSENT_KEY",)))

    @pytest.mark.parametrize(
        ("executor", "match"),
        [
            (DockerExecutor(environment={"HOME": "/root"}), "HOME is set by the executor"),
            (DockerExecutor(biomni_tools=True, environment={"PYTHONPATH": "/x"}), "PYTHONPATH is set"),
            (DockerExecutor(data_lake=Path("/lake"), environment={"BIOMNI_DATA_LAKE": "/x"}), "BIOMNI_DATA_LAKE is set"),
            (DockerExecutor(environment={"BAD-NAME": "x"}), "not a valid"),
            (DockerExecutor(environment={"A=B": "x"}), "not a valid"),
            (DockerExecutor(forward_env=("PATH",), environment={"PATH": "x"}), "more than once"),
        ],
    )
    def test_variables_that_cannot_be_given(self, executor: DockerExecutor, match: str, tmp_path: Path) -> None:
        if executor.data_lake is not None:
            executor.data_lake = tmp_path / "lake"
            executor.data_lake.mkdir()

        with pytest.raises(ExecutionError, match=match):
            command(executor, directory=tmp_path / "work")

    def test_a_variable_the_executor_sets_only_when_asked_is_free_otherwise(self) -> None:
        arguments = command(DockerExecutor(environment={"PYTHONPATH": "/mine"}))

        assert "PYTHONPATH=/mine" in env_arguments(arguments)

    def test_records_name_variables_but_not_forwarded_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_API_KEY", "secret-value")
        described = describe_executor(
            DockerExecutor(biomni_tools=True, forward_env=("SOME_API_KEY",), environment={"BIOMNI_LLM": "x"})
        )

        assert described["biomni_tools"] is True
        assert described["forward_env"] == ["SOME_API_KEY"]
        assert described["environment"] == {"BIOMNI_LLM": "x"}
        assert "secret-value" not in json.dumps(described)

    def test_local_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_API_KEY", "secret-value")
        environment = LocalExecutor(
            warn=False, biomni_tools=True, forward_env=("SOME_API_KEY",), environment={"BIOMNI_LLM": "x"}
        ).build_environment()

        assert environment["PYTHONPATH"] == str(biomni_package_directory())
        assert environment["SOME_API_KEY"] == "secret-value"
        assert environment["BIOMNI_LLM"] == "x"

        plain = LocalExecutor(warn=False).build_environment()
        assert "PYTHONPATH" not in plain and "SOME_API_KEY" not in plain

        with pytest.raises(ExecutionError, match="PATH is set"):
            LocalExecutor(warn=False, environment={"PATH": "/bin"}).build_environment()


class TestSessions:
    def test_session_executor_gives_biomnis_tools_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_API_KEY", "x")
        executor = session_executor(forward_env=["SOME_API_KEY"], environment={"BIOMNI_LLM": "y"})  # type: ignore[arg-type]

        assert executor.biomni_tools is True
        assert executor.forward_env == ("SOME_API_KEY",)
        assert executor.environment == {"BIOMNI_LLM": "y"}
        assert session_executor(biomni_tools=False).biomni_tools is False

    def test_docker_session_reports_its_tools_and_lake(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        lake.mkdir()
        for name in ("b.csv", "a.parquet", ".c.csv.part"):
            (lake / name).write_text("x")
        (lake / "folder").mkdir()

        session = DockerSession(tmp_path / "work", executor=DockerExecutor(biomni_tools=True, data_lake=lake))

        assert session.has_biomni_tools() is True
        assert session.data_lake_files() == ["a.parquet", "b.csv"]
        assert DockerSession(tmp_path / "work", executor=DockerExecutor()).data_lake_files() == []
        assert list_data_lake(tmp_path / "missing") == []

    def test_local_session_imports_biomni_and_gets_forwarded_variables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOME_API_KEY", "passed")
        with LocalSession(
            tmp_path, warn=False, biomni_tools=True, forward_env=("SOME_API_KEY",), environment={"BIOMNI_LLM": "m"}
        ) as session:
            result = session.run(
                "import os, biomni, biomni.config\n"
                "print(biomni.__file__)\n"
                "print(os.environ['SOME_API_KEY'], biomni.config.default_config.llm)"
            )

            assert session.has_biomni_tools() is True
            assert result.status == "ok", result.error
            path, values = result.output.strip().splitlines()
            assert Path(path).is_relative_to(biomni_package_directory())
            assert values == "passed m"
            assert session.describe()["forward_env"] == ["SOME_API_KEY"]
            assert "passed" not in json.dumps(session.describe())

    def test_local_session_without_tools_cannot_import_biomni(self, tmp_path: Path) -> None:
        with LocalSession(tmp_path, warn=False) as session:
            assert session.has_biomni_tools() is False
            result = session.run("import importlib.util\nprint(importlib.util.find_spec('biomni') is None)")

        # The host may have its own Biomni installed; this one must not be what is found
        assert result.output.strip() == "True" or "biomni_package" not in result.output


class LakeSession(LocalSession):
    """A local session that reports a data lake, to test what a meeting lists of it."""

    def __init__(self, directory: Path, files: tuple[str, ...], **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(directory, warn=False, **kwargs)
        self.files = list(files)

    def data_lake_path(self) -> str | None:
        return SANDBOX_DATA_LAKE_DIR

    def data_lake_files(self) -> list[str]:
        return self.files

    def check(self, code: str, timeout: float | None = None) -> CellResult:
        # The host's own environment has only some of Biomni's software; this one has it all
        return reported(installed_report())


@pytest.fixture
def session(tmp_path: Path):
    files = ("affinity_capture-ms.parquet", "BindingDB_All_202409.tsv")
    with LakeSession(tmp_path / "work", files=files, biomni_tools=True) as opened:
        yield opened


def meeting(team_member: Agent, save_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
    return hold_meeting(
        meeting_type="individual",
        agenda="Find genes linked to fever.",
        agenda_questions=("Which gene is strongest?",),
        save_dir=save_dir,
        team_member=team_member,
        **kwargs,
    )


def first_prompt(fake_client: FakeClient, call: int) -> str:
    return fake_client.completions.calls[call]["messages"][-1]["content"]


def saved_record(save_dir: Path) -> dict:
    return json.loads((save_dir / METADATA_DIR_NAME / "discussion.json").read_text())


class TestMeetingResources:
    def test_retrieval_chooses_what_the_meeting_is_told_of(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        tools = biomni_tools()
        index = next(i for i, tool in enumerate(tools) if tool["name"] == "query_uniprot")
        fake_client.completions.responses = [
            text_response(f"TOOLS: [{index}]\nDATA_LAKE: [0]\nLIBRARIES: [0]\nKNOW_HOW: [1]"),
            text_response("An answer."),
        ]

        result = meeting(team_member, tmp_path, session=session)

        # The first request is the retrieval, alone, without tools or a system prompt
        retrieval = fake_client.completions.calls[0]
        assert len(retrieval["messages"]) == 1
        assert "tools" not in retrieval
        assert "USER QUERY: Find genes linked to fever.\n\n1. Which gene is strongest?" in retrieval["messages"][0]["content"]

        prompt = first_prompt(fake_client, 1)
        assert "Based on the agenda, these are the most relevant" in prompt
        assert "Import file: biomni.tool.database" in prompt and "Method: query_uniprot" in prompt
        assert "Method: query_pdb" not in prompt
        assert "affinity_capture-ms.parquet:\n  Protein-protein interactions" in prompt
        assert "BindingDB" not in prompt
        library = next(iter(environment_descriptions()[1]))
        assert f"{library}:" in prompt
        assert "Single Cell RNA-seq Cell Type Annotation" in prompt
        assert "sgRNA Design Guide" not in prompt

        record = saved_record(tmp_path)["resources"]
        assert record["mode"] == "retrieve"
        assert record["available"] == {"tools": 223, "data_lake": 2, "libraries": 113, "know_how": 2}
        assert record["not_installed"] == {}
        assert record["selected"]["tools"] == ["biomni.tool.database.query_uniprot"]
        assert record["selected"]["know_how"] == ["single_cell_annotation"]
        assert record["retrieval"]["understood"] is True
        assert record["retrieval"]["model"] == team_member.model
        assert record["retrieval"]["input_tokens"] == 100
        assert record["retrieval"]["reply"].startswith("TOOLS:")

        # Its tokens are the meeting's, but it is not a turn of the discussion
        assert result.usage.num_calls == 1 + 1
        assert len(saved_record(tmp_path)["turns"]) == len(result.discussion)

    def test_an_answer_that_is_not_a_choice_lists_everything(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path, capsys
    ) -> None:
        fake_client.completions.responses = [text_response("Use whatever you like."), text_response("Done.")]

        meeting(team_member, tmp_path, session=session)

        prompt = first_prompt(fake_client, 1)
        assert prompt.count("Method: ") == 223
        assert "These are the resources" in prompt
        assert saved_record(tmp_path)["resources"]["retrieval"]["understood"] is False
        assert "did not say which resources" in capsys.readouterr().out

    def test_all_lists_everything_without_asking(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("Done."), text_response("Critique.")]

        meeting(team_member, tmp_path, session=session, resources="all")

        prompt = first_prompt(fake_client, 0)
        assert prompt.count("Method: ") == 223
        record = saved_record(tmp_path)["resources"]
        assert record["mode"] == "all" and record["retrieval"] is None
        assert len(record["selected"]["tools"]) == 223

    def test_none_lists_nothing(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        meeting(team_member, tmp_path, session=session, resources="none")

        assert "Function Dictionary" not in first_prompt(fake_client, 0)
        assert "running Python session" in first_prompt(fake_client, 0)
        assert saved_record(tmp_path)["resources"] is None

    def test_given_resources_are_listed_as_they_are(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        meeting(team_member, tmp_path, session=session, resources=small_resources())

        prompt = first_prompt(fake_client, 0)
        assert "Method: find_gene" in prompt and "These are the resources" in prompt
        assert "following path: /lake." in prompt
        record = saved_record(tmp_path)["resources"]
        assert record["mode"] == "given" and record["not_installed"] is None

    def test_given_data_without_a_path_is_where_the_session_has_it(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        given = Resources(data_lake=(Resource("a.parquet", "Table A."),))

        meeting(team_member, tmp_path, session=session, resources=given)

        assert f"following path: {SANDBOX_DATA_LAKE_DIR}." in first_prompt(fake_client, 0)

    def test_commercial_mode(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        meeting(team_member, tmp_path, session=session, resources="all", commercial_mode=True)

        prompt = first_prompt(fake_client, 0)
        assert "affinity_capture-ms.parquet" in prompt and "BindingDB" not in prompt
        record = saved_record(tmp_path)["resources"]
        assert record["commercial_mode"] is True
        assert record["available"]["data_lake"] == 1

    def test_a_session_without_biomni_offers_only_know_how(
        self, fake_client: FakeClient, team_member: Agent, tmp_path: Path
    ) -> None:
        fake_client.completions.responses = [text_response("TOOLS: []\nKNOW_HOW: [0]"), text_response("Done.")]
        with LocalSession(tmp_path / "work", warn=False) as plain:
            meeting(team_member, tmp_path, session=plain)

        retrieval = fake_client.completions.calls[0]["messages"][0]["content"]
        assert "AVAILABLE TOOLS:\nNone available" in retrieval
        assert "sgRNA Design Guide" in first_prompt(fake_client, 1)

    def test_nothing_to_offer_asks_nothing(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        meeting(team_member, tmp_path, session=session, resources=Resources())

        assert "Function Dictionary" not in first_prompt(fake_client, 0)
        assert len(fake_client.completions.calls) == 1

    def test_team_meeting_asks_the_lead_and_tells_everyone(
        self,
        fake_client: FakeClient,
        team_lead: Agent,
        team_member: Agent,
        session: LakeSession,
        tmp_path: Path,
    ) -> None:
        fake_client.completions.responses = [text_response("TOOLS: [0]\nKNOW_HOW: []"), text_response("Summary.")]

        hold_meeting(
            meeting_type="team",
            agenda="Pick a target.",
            save_dir=tmp_path,
            team_lead=team_lead,
            team_members=(team_member,),
            session=session,
        )

        start = fake_client.completions.calls[1]["messages"][1]["content"]
        assert start.startswith("This is the beginning of a team meeting")
        assert f"Method: {biomni_tools()[0]['name']}" in start
        assert saved_record(tmp_path)["resources"]["retrieval"]["name"] == team_lead.name

    def test_the_budget_is_checked_before_retrieval(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        with pytest.raises(BudgetExceededError):
            meeting(team_member, tmp_path, session=session, max_cost=0.0)

        assert fake_client.completions.calls == []

    def test_the_retrieval_is_checked_against_the_models_input_limit(
        self,
        fake_client: FakeClient,
        team_member: Agent,
        session: LakeSession,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from virtual_lab.utils import ContextLengthExceededError

        # The package's run_meeting is the function, which hides the module of the same name
        module = sys.modules["virtual_lab.run_meeting"]

        def too_long(messages, model):  # type: ignore[no-untyped-def]
            raise ContextLengthExceededError(f"{len(messages[0]['content'])} characters")

        monkeypatch.setattr(module, "check_context_length", too_long)

        with pytest.raises(ContextLengthExceededError):
            meeting(team_member, tmp_path, session=session)

        assert fake_client.completions.calls == []

    def test_retrieval_counts_towards_on_usage(
        self, fake_client: FakeClient, team_member: Agent, session: LakeSession, tmp_path: Path
    ) -> None:
        seen: list[int] = []
        fake_client.completions.responses = [text_response("TOOLS: []"), text_response("Done.")]

        meeting(team_member, tmp_path, session=session, on_usage=lambda usage: seen.append(usage.num_calls))

        assert seen[0] == 1

    @pytest.mark.parametrize("value", ["some", 3])
    def test_invalid_resources(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path, value) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="resources must be"):
            meeting(team_member, tmp_path, resources=value)

    def test_given_resources_need_a_session(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs a session"):
            meeting(team_member, tmp_path, resources=small_resources())

    def test_no_session_means_no_retrieval(self, fake_client: FakeClient, team_member: Agent, tmp_path: Path) -> None:
        result = meeting(team_member, tmp_path)

        assert result.record.resources is None
        assert len(fake_client.completions.calls) == 1


def test_host_does_not_import_biomni() -> None:
    import virtual_lab  # noqa: F401

    available_resources(FakeSession())
    load_know_how()

    assert not any(name == "biomni" or name.startswith("biomni.") for name in sys.modules)
