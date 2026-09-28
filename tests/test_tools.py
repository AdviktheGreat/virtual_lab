"""Tests for the tool registry and the tool-calling loop."""

import json

import openai
import pytest

from virtual_lab.agent import Agent
from virtual_lab.constants import ARTIFACT_DIR_NAME, MAX_TOOL_ITERATIONS, STRUCTURE_DIR_NAME
from virtual_lab.run_meeting import run_meeting
from virtual_lab.tables import TableError, list_data_files
from virtual_lab.tools import (
    DATABASE_TOOLS,
    PUBMED_TOOL,
    TOOL_REGISTRY,
    Tool,
    all_tools,
    data_file_tools,
    run_tool_calls,
    structure_file_tool,
    tools_for,
)

from conftest import FakeClient, text_response, tool_call_response


@pytest.fixture
def lookup_calls() -> list[str]:
    return []


@pytest.fixture
def lookup_tool(lookup_calls: list[str]) -> Tool:
    def lookup(accession: str) -> str:
        lookup_calls.append(accession)
        return f"UniProt {accession}: spike glycoprotein"

    return Tool(
        name="uniprot_lookup",
        description="Look up a protein by accession.",
        parameters={
            "type": "object",
            "properties": {"accession": {"type": "string"}},
            "required": ["accession"],
        },
        function=lookup,
    )


@pytest.fixture
def failing_tool() -> Tool:
    def fail() -> str:
        raise ConnectionError("PubMed unreachable")

    return Tool(
        name="broken_tool",
        description="Always fails.",
        parameters={"type": "object", "properties": {}},
        function=fail,
    )


def tool_turns(discussion: list[dict[str, str]]) -> list[str]:
    return [turn["message"] for turn in discussion if turn["agent"] == "Tool"]


class TestToolDefinition:
    def test_definition_is_valid_api_json(self, lookup_tool: Tool) -> None:
        definition = lookup_tool.definition

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "uniprot_lookup"
        assert definition["function"]["parameters"] == lookup_tool.parameters
        json.dumps(definition)

    def test_pubmed_tool_is_an_ordinary_tool(self) -> None:
        assert isinstance(PUBMED_TOOL, Tool)
        assert PUBMED_TOOL.name == "pubmed_search"


class TestRunToolCalls:
    def test_runs_the_matching_tool(self, lookup_tool: Tool, lookup_calls: list[str]) -> None:
        call = tool_call_response("uniprot_lookup", {"accession": "P0DTC2"}).choices[0].message
        outputs, messages = run_tool_calls(call.tool_calls, (lookup_tool,))

        assert lookup_calls == ["P0DTC2"]
        assert outputs == ["UniProt P0DTC2: spike glycoprotein"]
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"

    def test_tool_failure_is_reported_to_the_model(self, failing_tool: Tool) -> None:
        call = tool_call_response("broken_tool").choices[0].message
        outputs, _ = run_tool_calls(call.tool_calls, (failing_tool,))

        assert "ConnectionError" in outputs[0]
        assert "PubMed unreachable" in outputs[0]

    def test_unknown_tool_lists_what_is_available(self, lookup_tool: Tool) -> None:
        call = tool_call_response("does_not_exist").choices[0].message
        outputs, _ = run_tool_calls(call.tool_calls, (lookup_tool,))

        assert "unknown tool" in outputs[0]
        assert "uniprot_lookup" in outputs[0]

    def test_bad_arguments_are_reported_to_the_model(self, lookup_tool: Tool) -> None:
        call = tool_call_response("uniprot_lookup", {"wrong_argument": "x"}).choices[0].message
        outputs, _ = run_tool_calls(call.tool_calls, (lookup_tool,))

        assert "Error running tool" in outputs[0]


class TestToolLoopInMeeting:
    def test_tools_are_offered_again_after_a_result(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, lookup_calls, tmp_path
    ) -> None:
        # The second search is only possible if the tools are re-offered with the first result
        fake_client.completions.responses = [
            tool_call_response("uniprot_lookup", {"accession": "P0DTC2"}, "call_1"),
            tool_call_response("uniprot_lookup", {"accession": "P59594"}, "call_2"),
            text_response("Both proteins considered."),
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Compare two spike proteins.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        assert lookup_calls == ["P0DTC2", "P59594"]
        assert all(call["tools"] is not openai.NOT_GIVEN for call in fake_client.completions.calls)

    def test_tool_output_is_recorded_in_the_transcript(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, tmp_path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("uniprot_lookup", {"accession": "P0DTC2"}),
            text_response("Answer."),
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Look it up.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())

        assert tool_turns(discussion) == ["UniProt P0DTC2: spike glycoprotein"]

    def test_a_failing_tool_does_not_end_the_meeting(
        self, fake_client: FakeClient, team_member: Agent, failing_tool: Tool, tmp_path
    ) -> None:
        fake_client.completions.responses = [
            tool_call_response("broken_tool"),
            text_response("Proceeding without the tool."),
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Try the tool.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(failing_tool,),
            num_rounds=0,
        )

        discussion = json.loads((tmp_path / "discussion.json").read_text())

        assert "ConnectionError" in tool_turns(discussion)[0]
        assert discussion[-1]["message"] == "Proceeding without the tool."

    def test_runaway_tool_use_is_capped(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, lookup_calls, tmp_path
    ) -> None:
        # An agent that never stops asking for tools
        fake_client.completions.responses = [
            tool_call_response("uniprot_lookup", {"accession": f"A{index}"}, f"call_{index}")
            for index in range(MAX_TOOL_ITERATIONS + 5)
        ]

        run_meeting(
            meeting_type="individual",
            agenda="Search forever.",
            save_dir=tmp_path,
            team_member=team_member,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        calls = fake_client.completions.calls

        assert len(lookup_calls) == MAX_TOOL_ITERATIONS
        assert len(calls) == MAX_TOOL_ITERATIONS + 1
        # Tools are withheld on the final attempt to force a text answer, and none are run
        assert calls[-1]["tools"] is openai.NOT_GIVEN
        assert all(call["tools"] is not openai.NOT_GIVEN for call in calls[:-1])

    def test_no_tools_means_none_are_offered(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="No tools here.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
        )

        assert fake_client.completions.calls[0]["tools"] is openai.NOT_GIVEN


class TestToolRegistration:
    def test_pubmed_search_shorthand_registers_the_tool(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Search the literature.",
            save_dir=tmp_path,
            team_member=team_member,
            pubmed_search=True,
            num_rounds=0,
        )

        offered = fake_client.completions.calls[0]["tools"]

        assert [tool["function"]["name"] for tool in offered] == ["pubmed_search"]

    def test_custom_tools_combine_with_pubmed_search(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Search and look up.",
            save_dir=tmp_path,
            team_member=team_member,
            pubmed_search=True,
            tools=(lookup_tool,),
            num_rounds=0,
        )

        offered = fake_client.completions.calls[0]["tools"]

        assert {tool["function"]["name"] for tool in offered} == {
            "pubmed_search",
            "uniprot_lookup",
        }

    def test_pubmed_tool_passed_twice_is_deduplicated(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        run_meeting(
            meeting_type="individual",
            agenda="Search the literature.",
            save_dir=tmp_path,
            team_member=team_member,
            pubmed_search=True,
            tools=(PUBMED_TOOL,),
            num_rounds=0,
        )

        offered = fake_client.completions.calls[0]["tools"]

        assert len(offered) == 1

    def test_duplicate_tool_names_are_rejected(
        self, fake_client: FakeClient, team_member: Agent, lookup_tool: Tool, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="unique"):
            run_meeting(
                meeting_type="individual",
                agenda="Ambiguous tools.",
                save_dir=tmp_path,
                team_member=team_member,
                tools=(lookup_tool, lookup_tool),
                num_rounds=0,
            )


class TestTheSuppliedTools:
    """The database tools, as a model sees them."""

    def test_every_registered_tool_has_a_valid_definition(self) -> None:
        for tool in DATABASE_TOOLS:
            definition = tool.definition

            assert definition["type"] == "function"
            assert definition["function"]["name"] == tool.name
            assert definition["function"]["parameters"]["type"] == "object"
            assert definition["function"]["description"]

    def test_names_are_unique(self) -> None:
        names = [tool.name for tool in DATABASE_TOOLS]

        assert len(names) == len(set(names))

    def test_the_registry_holds_every_tool(self) -> None:
        # Pinned to the names rather than compared to DATABASE_TOOLS, which the registry is built
        # from: that comparison is true however either one changes, including by losing a tool
        assert set(TOOL_REGISTRY) == {
            "pubmed_search",
            "uniprot_lookup",
            "uniprot_search",
            "pdb_lookup",
            "alphafold_lookup",
            "pubchem_lookup",
            "chembl_lookup",
            "chembl_search",
            "chembl_target_search",
            "chembl_activities",
            "europepmc_search",
            "europepmc_lookup",
            "europepmc_fulltext",
            "arxiv_search",
        }
        assert set(TOOL_REGISTRY) == {tool.name for tool in DATABASE_TOOLS}

    def test_tools_can_be_looked_up_in_the_order_asked_for(self) -> None:
        selected = tools_for("pdb_lookup", "uniprot_lookup")

        assert [tool.name for tool in selected] == ["pdb_lookup", "uniprot_lookup"]

    def test_an_unknown_name_lists_what_is_available(self) -> None:
        with pytest.raises(KeyError, match="uniprot_lookup"):
            tools_for("protein_lookup")

    def test_every_required_parameter_is_described(self) -> None:
        # A required parameter with no description is one a model has to guess at
        for tool in DATABASE_TOOLS:
            properties = tool.parameters["properties"]

            for name in tool.parameters.get("required", []):
                assert properties[name].get("description"), f"{tool.name}.{name}"

    def test_no_tool_requires_an_argument_it_did_not_declare(self) -> None:
        for tool in DATABASE_TOOLS:
            declared = set(tool.parameters["properties"])

            assert set(tool.parameters.get("required", [])) <= declared, tool.name

    def test_a_declared_parameter_is_one_the_function_accepts(self) -> None:
        # A schema and a signature that disagree produce a TypeError only once a model calls it
        from inspect import signature

        for tool in DATABASE_TOOLS:
            accepted = set(signature(tool.function).parameters)

            assert set(tool.parameters["properties"]) <= accepted, tool.name

    def test_each_tool_reaches_the_database_its_description_names(self, web_transport) -> None:
        # Comparing parameter names cannot tell two tools apart when their schemas agree, so
        # uniprot_lookup wired to the AlphaFold function would pass every other test here. This
        # calls each one and checks where the request went.
        from conftest import FakeResponse

        expected = {
            "uniprot_lookup": (
                "rest.uniprot.org/uniprotkb/P01308",
                {"accession": "P01308"},
                [{}],
            ),
            "uniprot_search": (
                "rest.uniprot.org/uniprotkb/search",
                {"query": "insulin"},
                [{"results": []}],
            ),
            "pdb_lookup": (
                "data.rcsb.org/rest/v1/core/entry/4HHB",
                {"pdb_id": "4HHB"},
                [{}],
            ),
            "alphafold_lookup": (
                "alphafold.ebi.ac.uk/api/prediction/P01308",
                {"accession": "P01308"},
                [[{"uniprotAccession": "P01308"}]],
            ),
            "pubchem_lookup": (
                "pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin",
                {"identifier": "aspirin"},
                [{"PropertyTable": {"Properties": [{"CID": 2244}]}},
                 {"InformationList": {"Information": [{"Title": "Aspirin"}]}}],
            ),
            "chembl_lookup": (
                "www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json",
                {"chembl_id": "CHEMBL25"},
                [{"molecule_chembl_id": "CHEMBL25"}, {"mechanisms": []}],
            ),
            "chembl_search": (
                "www.ebi.ac.uk/chembl/api/data/molecule/search",
                {"query": "aspirin"},
                [{"molecules": [], "page_meta": {"total_count": 0}}],
            ),
            "chembl_target_search": (
                "www.ebi.ac.uk/chembl/api/data/target/search",
                {"query": "EGFR"},
                [{"targets": [], "page_meta": {"total_count": 0}}],
            ),
            "chembl_activities": (
                "www.ebi.ac.uk/chembl/api/data/activity",
                {"target_chembl_id": "CHEMBL203"},
                [{"activities": [], "page_meta": {"total_count": 0}}],
            ),
        }

        for name, (fragment, arguments, bodies) in expected.items():
            web_transport.requests.clear()
            # A list, because a tool may make more than one request: the PubChem lookup fetches
            # the properties and then the description
            web_transport.responses = [FakeResponse(json_body=body) for body in bodies]

            TOOL_REGISTRY[name].function(**arguments)

            assert fragment in web_transport.urls[0], f"{name} went to {web_transport.urls[0]}"


class TestTheStructureDownloadTool:
    def test_the_save_directory_is_not_something_a_model_chooses(self, tmp_path) -> None:
        # Where files land is the lab's decision. Exposed in the schema, it would be an argument a
        # model fills in, and the filename checks would be the only thing standing in the way.
        tool = structure_file_tool(tmp_path)

        assert "save_dir" not in tool.parameters["properties"]
        assert "save_dir" not in json.dumps(tool.definition)

    def test_the_bound_directory_is_the_one_written_to(self, web_transport, tmp_path) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data_4HHB")]
        tool = structure_file_tool(tmp_path / "somewhere")

        output = tool.function(identifier="4HHB")

        written = tmp_path / "somewhere" / STRUCTURE_DIR_NAME / "4HHB.cif"

        assert str(tmp_path / "somewhere") in output
        assert written.exists()
        assert written.read_text() == "data_4HHB"

    def test_it_is_left_out_when_there_is_nowhere_to_write(self) -> None:
        # Rather than defaulting to the working directory, which is not a library's to write into
        assert "fetch_structure_file" not in {tool.name for tool in all_tools()}

    def test_it_is_included_when_a_directory_is_given(self, tmp_path) -> None:
        names = {tool.name for tool in all_tools(save_dir=tmp_path)}

        assert "fetch_structure_file" in names
        assert names > {tool.name for tool in all_tools()}

    def test_files_land_where_the_sandbox_can_read_them(self, web_transport, tmp_path) -> None:
        # The only host directory mounted into the container is the meeting's artifact
        # directory, so a file written beside it is invisible to the code that needs it
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data")]
        tool = [
            t
            for t in all_tools(save_dir=tmp_path, save_name="meeting")
            if t.name == "fetch_structure_file"
        ][0]

        output = tool.function(identifier="4HHB")

        work_dir = tmp_path / ARTIFACT_DIR_NAME / "meeting"
        written = work_dir / STRUCTURE_DIR_NAME / "4HHB.cif"

        assert written.exists()
        assert written.is_relative_to(work_dir)

    def test_the_agent_is_told_the_path_that_will_work_from_its_code(
        self, web_transport, tmp_path
    ) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data")]
        tool = [
            t
            for t in all_tools(save_dir=tmp_path, save_name="meeting")
            if t.name == "fetch_structure_file"
        ][0]

        output = tool.function(identifier="4HHB")

        work_dir = tmp_path / ARTIFACT_DIR_NAME / "meeting"

        assert f'Read it from code as "{STRUCTURE_DIR_NAME}/4HHB.cif"' in output
        # The path the model is given must resolve to the file, from where the code will run
        assert (work_dir / f"{STRUCTURE_DIR_NAME}/4HHB.cif").exists()

    def test_a_failed_download_reports_back_instead_of_ending_the_meeting(
        self, web_transport, tmp_path
    ) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=404)]
        tool = structure_file_tool(tmp_path)

        class Call:
            id = "call_1"
            function = type("F", (), {"name": "fetch_structure_file", "arguments": '{"identifier": "ZZZZ"}'})()

        outputs, messages = run_tool_calls([Call()], (tool,))

        assert "Error running tool" in outputs[0]
        assert messages[0]["role"] == "tool"


class TestTheDataFileTools:
    def test_the_directory_is_not_something_a_model_chooses(self, tmp_path) -> None:
        # A directory a model can name is a directory a model can name /etc in
        for tool in data_file_tools(tmp_path):
            assert "work_dir" not in tool.parameters["properties"]
            assert "work_dir" not in json.dumps(tool.definition)

    def test_the_listing_takes_no_arguments_at_all(self, tmp_path) -> None:
        listing = data_file_tools(tmp_path)[0]

        assert listing.name == "data_files"
        assert listing.parameters["properties"] == {}
        assert listing.function() == list_data_files(tmp_path).report()

    def test_only_the_filename_is_required(self, tmp_path) -> None:
        inspection = data_file_tools(tmp_path)[1]

        assert inspection.name == "inspect_data_file"
        assert inspection.parameters["required"] == ["filename"]
        assert set(inspection.parameters["properties"]) == {"filename", "sheet"}

    def test_the_bound_directory_is_the_one_read(self, tmp_path) -> None:
        inside = tmp_path / "inside"
        inside.mkdir()
        (inside / "data.csv").write_text("gene,n\nTP53,1\n")
        (tmp_path / "outside.csv").write_text("secret,n\nx,1\n")

        listing, inspection = data_file_tools(inside)

        assert "data.csv" in listing.function()
        assert "outside.csv" not in listing.function()
        assert "2 columns" in inspection.function(filename="data.csv")

    def test_a_path_out_of_the_directory_comes_back_as_an_error_not_a_crash(
        self, tmp_path
    ) -> None:
        inspection = data_file_tools(tmp_path)[1]

        class Call:
            id = "call_1"
            function = type(
                "F",
                (),
                {
                    "name": "inspect_data_file",
                    "arguments": '{"filename": "../../etc/passwd"}',
                },
            )()

        outputs, messages = run_tool_calls([Call()], (inspection,))

        assert "Error running tool" in outputs[0]
        assert "UnsafeFilenameError" in outputs[0]
        assert messages[0]["role"] == "tool"

    def test_they_are_left_out_when_there_is_nowhere_to_read(self) -> None:
        names = {tool.name for tool in all_tools()}

        assert "data_files" not in names
        assert "inspect_data_file" not in names

    def test_they_read_the_directory_the_sandbox_runs_in(self, tmp_path) -> None:
        work_dir = tmp_path / ARTIFACT_DIR_NAME / "meeting"
        work_dir.mkdir(parents=True)
        (work_dir / "results.csv").write_text("gene,n\nTP53,1\n")

        listing = [
            t for t in all_tools(save_dir=tmp_path, save_name="meeting") if t.name == "data_files"
        ][0]

        assert "results.csv" in listing.function()

    def test_a_sheet_can_be_named_and_reaches_the_reader(self, tmp_path) -> None:
        inspection = data_file_tools(tmp_path)[1]
        (tmp_path / "book.xlsx").write_bytes(b"not a spreadsheet")

        with pytest.raises(TableError, match="not a readable spreadsheet"):
            inspection.function(filename="book.xlsx", sheet="Results")
