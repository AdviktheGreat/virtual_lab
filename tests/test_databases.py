"""Tests for the database lookups agents use.

These run offline against fixtures shaped like the real responses, because a test that depends on
UniProt being up is a test that fails for reasons having nothing to do with the code. The fixtures
are the risk in that arrangement: they can drift from what the services actually send and the
suite would not notice. So the classes at the end query the real databases and assert that the
fields these fixtures claim exist really do, and they run when VIRTUAL_LAB_LIVE_TESTS=1 is set.
"""

import os

import pytest

from virtual_lab import databases
from virtual_lab.artifacts import UnsafeFilenameError
from virtual_lab.constants import (
    MAX_CHAINS_REPORTED,
    MAX_COMMENTS_REPORTED,
    MAX_FEATURES_REPORTED,
    MAX_FIELD_CHARACTERS,
    MAX_ITEMS_LISTED,
    MAX_SEARCH_RESULTS,
    MAX_SEQUENCE_RESIDUES_REPORTED,
)
from virtual_lab.records import (
    DatabaseError,
    RecordNotFoundError,
    as_int,
    as_text,
    listing,
)
from virtual_lab.databases import (
    Chain,
    PredictedStructure,
    Structure,
    bounded,
    download_structure,
    get_chain,
    get_predicted_structure,
    get_protein,
    get_structure,
    protein_from,
    search_proteins,
    truncate_sequence,
    truncate_text,
    wrap_sequence,
)
from virtual_lab.web import WebRequestError

live_only = pytest.mark.skipif(
    os.environ.get("VIRTUAL_LAB_LIVE_TESTS") != "1",
    reason="Set VIRTUAL_LAB_LIVE_TESTS=1 to query the real services",
)


# Trimmed from a real response to https://rest.uniprot.org/uniprotkb/P01308.json, keeping one of
# each kind of nested shape the parsing has to handle
INSULIN_ENTRY = {
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "primaryAccession": "P01308",
    "uniProtkbId": "INS_HUMAN",
    "organism": {"scientificName": "Homo sapiens", "commonName": "Human", "taxonId": 9606},
    "proteinDescription": {"recommendedName": {"fullName": {"value": "Insulin"}}},
    "genes": [{"geneName": {"value": "INS"}}],
    "sequence": {"value": "MALWMRLLPLLALLALWGPDPAAA", "length": 24, "molWeight": 11981},
    "comments": [
        {"commentType": "FUNCTION", "texts": [{"value": "Insulin decreases blood glucose."}]},
        {"commentType": "SUBUNIT", "texts": [{"value": "Heterodimer of a B chain and an A chain."}]},
        {"commentType": "SIMILARITY", "texts": [{"value": "Belongs to the insulin family."}]},
    ],
    "features": [
        {"type": "Signal", "location": {"start": {"value": 1}, "end": {"value": 24}}, "description": ""},
        {
            "type": "Disulfide bond",
            "location": {"start": {"value": 31}, "end": {"value": 96}},
            "description": "Interchain",
        },
        {"type": "Sequence conflict", "location": {"start": {"value": 5}, "end": {"value": 5}}, "description": "noise"},
    ],
    "uniProtKBCrossReferences": [
        {"database": "PDB", "id": "1A7F"},
        {"database": "PDB", "id": "1AI0"},
        {"database": "EMBL", "id": "AY899304"},
    ],
}

HAEMOGLOBIN_ENTRY = {
    "struct": {"title": "THE CRYSTAL STRUCTURE OF HUMAN DEOXYHAEMOGLOBIN"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "rcsb_entry_info": {
        "resolution_combined": [1.74],
        "nonpolymer_bound_components": ["HEM"],
        "polymer_entity_count": 2,
    },
    "rcsb_accession_info": {"initial_release_date": "1984-07-17T00:00:00.000+00:00"},
    "rcsb_entry_container_identifiers": {
        "entry_id": "4HHB",
        "polymer_entity_ids": ["1", "2"],
        "pubmed_id": 6726807,
    },
}

ALPHA_CHAIN = {
    "entity_poly": {
        "pdbx_seq_one_letter_code_can": "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMF",
        "pdbx_strand_id": "A,C",
    },
    "rcsb_polymer_entity": {
        "pdbx_description": "Hemoglobin subunit alpha",
        "pdbx_number_of_molecules": 2,
    },
    "rcsb_entity_source_organism": [{"scientific_name": "Homo sapiens"}],
}

INSULIN_PREDICTION = [
    {
        "uniprotAccession": "P01308",
        "uniprotDescription": "Insulin",
        "organismScientificName": "Homo sapiens",
        "uniprotSequence": "MALWMRLLPLLALLALWGPDPAAA",
        "globalMetricValue": 52.91,
        "latestVersion": 6,
        "fractionPlddtConfident": 0.09,
        "fractionPlddtVeryHigh": 0.04,
        "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P01308-F1-model_v6.pdb",
        "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P01308-F1-model_v6.cif",
    }
]


class TestReadingAProteinRecord:
    def test_the_fields_a_report_depends_on_are_read(self) -> None:
        protein = protein_from(INSULIN_ENTRY)

        assert protein.accession == "P01308"
        assert protein.entry_name == "INS_HUMAN"
        assert protein.name == "Insulin"
        assert protein.organism == "Homo sapiens"
        assert protein.taxon_id == 9606
        assert protein.genes == ("INS",)
        assert protein.length == 24
        assert protein.mass == 11981
        assert protein.reviewed is True

    def test_only_the_informative_comments_are_kept(self) -> None:
        # An entry carries comment types describing the evidence rather than the molecule, and
        # they would crowd out the ones an agent needs
        protein = protein_from(INSULIN_ENTRY)

        report = protein.report()

        assert [kind for kind, _ in protein.comments] == ["FUNCTION", "SUBUNIT"]
        # Both halves matter: the kept comments have to reach the report, or dropping the
        # uninformative one would be satisfied just as well by rendering none of them
        assert "Function: Insulin decreases blood glucose" in report
        assert "Subunit:" in report
        assert "insulin family" not in report

    def test_comments_come_out_in_a_fixed_order(self) -> None:
        # Not the order the service happened to use, so two reports can be compared
        reversed_entry = INSULIN_ENTRY | {"comments": list(reversed(INSULIN_ENTRY["comments"]))}

        assert [kind for kind, _ in protein_from(reversed_entry).comments] == [
            "FUNCTION",
            "SUBUNIT",
        ]

    def test_only_the_informative_features_are_kept(self) -> None:
        protein = protein_from(INSULIN_ENTRY)

        assert [kind for kind, _, _ in protein.features] == ["Signal", "Disulfide bond"]

    def test_a_feature_spanning_one_residue_is_not_written_as_a_range(self) -> None:
        entry = INSULIN_ENTRY | {
            "features": [
                {
                    "type": "Active site",
                    "location": {"start": {"value": 7}, "end": {"value": 7}},
                    "description": "",
                }
            ]
        }

        assert protein_from(entry).features[0][1] == "7"

    def test_only_pdb_cross_references_are_collected(self) -> None:
        assert protein_from(INSULIN_ENTRY).pdb_ids == ("1A7F", "1AI0")

    def test_an_unreviewed_entry_is_marked_as_such(self) -> None:
        # An agent about to build on an annotation needs to know it was generated automatically
        entry = INSULIN_ENTRY | {"entryType": "UniProtKB unreviewed (TrEMBL)"}
        protein = protein_from(entry)

        assert protein.reviewed is False
        assert "unreviewed" in protein.report()

    def test_a_submitted_name_is_used_when_there_is_no_recommended_one(self) -> None:
        # Most unreviewed entries have no recommended name, and reporting "not stated" for all of
        # them would discard the only name they have
        entry = INSULIN_ENTRY | {
            "proteinDescription": {"submissionNames": [{"fullName": {"value": "Nanobody VHH"}}]}
        }

        assert protein_from(entry).name == "Nanobody VHH"

    def test_a_record_with_almost_nothing_in_it_still_reports(self) -> None:
        protein = protein_from({"primaryAccession": "X00000"})

        assert protein.name == ""
        assert protein.length == 0
        assert "X00000" in protein.report()

    def test_a_length_missing_from_the_response_falls_back_to_the_sequence(self) -> None:
        entry = {"primaryAccession": "X1", "sequence": {"value": "MKV"}}

        assert protein_from(entry).length == 3

    def test_the_features_shown_are_capped(self) -> None:
        entry = INSULIN_ENTRY | {
            "features": [
                {"type": "Domain", "location": {"start": {"value": i}, "end": {"value": i}}, "description": ""}
                for i in range(200)
            ]
        }

        assert len(protein_from(entry).features) == MAX_FEATURES_REPORTED


class TestWhatAReportCosts:
    """Everything a report contains goes into a request and is paid for by the token."""

    def test_a_long_sequence_is_truncated_and_says_so(self) -> None:
        protein = protein_from(INSULIN_ENTRY | {"sequence": {"value": "M" * 40_000, "length": 40_000}})
        report = protein.report()

        assert "residues not shown" in report
        assert len(report) < 4_000
        # The record still holds all of it, since code may need the whole sequence
        assert len(protein.sequence) == 40_000

    def test_a_sequence_at_the_limit_is_shown_whole(self) -> None:
        sequence = "M" * MAX_SEQUENCE_RESIDUES_REPORTED

        assert truncate_sequence(sequence) == sequence
        assert "not shown" not in truncate_sequence(sequence)

    def test_the_count_left_out_is_accurate(self) -> None:
        truncated = truncate_sequence("M" * 1_500, limit=1_000)

        assert "500 of 1,500 residues not shown" in truncated

    def test_a_sequence_is_wrapped_rather_than_run_together(self) -> None:
        wrapped = wrap_sequence("M" * 130, width=60)

        assert [len(line) for line in wrapped.splitlines()] == [60, 60, 10]

    def test_the_default_width_is_the_one_the_reports_use(self) -> None:
        # Passing a width tests the argument, not the value every report relies on
        lines = wrap_sequence("M" * 130).splitlines()

        assert max(len(line) for line in lines) == 60
        assert len(lines) == 3

    def test_the_pdb_list_is_capped(self) -> None:
        # A well studied protein has hundreds of structures, and listing them all would cost more
        # than the rest of the record put together
        entry = INSULIN_ENTRY | {
            "uniProtKBCrossReferences": [
                {"database": "PDB", "id": f"{i:04d}"} for i in range(400)
            ]
        }
        report = protein_from(entry).report()

        assert "and 392 more" in report

    def test_a_structure_report_is_capped_however_the_record_was_built(self) -> None:
        # get_structure fetches no more chains than it reports, so this cap is reachable only for a
        # record assembled some other way. It is what keeps the report bounded in that case.
        structure = Structure(
            pdb_id="9XXX",
            title="A ribosome, say",
            method="ELECTRON MICROSCOPY",
            resolution=2.8,
            released="2024-01-01",
            chains=tuple(
                Chain(
                    entity_id=str(i),
                    description=f"protein {i}",
                    chain_ids=(chr(65 + i),),
                    organism="Escherichia coli",
                    sequence=f"{chr(65 + i)}" * 300,
                    copies=1,
                )
                for i in range(MAX_CHAINS_REPORTED + 5)
            ),
        )
        report = structure.report()

        assert f"{MAX_CHAINS_REPORTED + 5} polymer chain(s)" in report
        assert "and 5 more" in report
        assert "protein 0" in report
        assert f"protein {MAX_CHAINS_REPORTED + 4}" not in report

        # The chain summaries are one line each; the sequence blocks are the bulk of the report
        # and are capped separately, so a test that only reads the summary list misses the half
        # that actually costs anything
        assert report.count("Sequence of chain ") == MAX_CHAINS_REPORTED
        assert "Sequence of chain A" in report
        assert "A" * 60 in report
        assert f"Sequence of chain {chr(65 + MAX_CHAINS_REPORTED + 4)}" not in report
        assert f"{chr(65 + MAX_CHAINS_REPORTED + 4)}" * 60 not in report

    def test_every_protein_field_that_is_parsed_reaches_the_report(self) -> None:
        # The parsing tests assert on the record. A field can be read correctly and then be
        # missing from the one thing the model actually sees.
        report = protein_from(INSULIN_ENTRY).report()

        assert "P01308" in report
        assert "INS_HUMAN" in report
        assert "Insulin" in report
        assert "Homo sapiens" in report
        assert "taxon 9606" in report
        assert "Genes: INS" in report
        assert "24 residues" in report
        assert "11,981 Da" in report
        assert "1A7F" in report
        assert "Annotated positions" in report
        assert "Disulfide bond" in report
        assert "MALWMRLLPLLALLALWGPDPAAA" in report

    def test_every_structure_field_that_is_parsed_reaches_the_report(self) -> None:
        structure = Structure(
            pdb_id="4HHB",
            title="THE CRYSTAL STRUCTURE OF HUMAN DEOXYHAEMOGLOBIN",
            method="X-RAY DIFFRACTION",
            resolution=1.74,
            released="1984-07-17",
            chains=(
                Chain(
                    entity_id="1",
                    description="Hemoglobin subunit alpha",
                    chain_ids=("A", "C"),
                    organism="Homo sapiens",
                    sequence="VLSPADKTNV",
                    copies=2,
                ),
            ),
            entity_count=2,
            ligand_ids=("HEM",),
            pubmed_id=6726807,
        )
        report = structure.report()

        assert "4HHB" in report
        assert "DEOXYHAEMOGLOBIN" in report
        assert "X-RAY DIFFRACTION" in report
        assert "1.74 A resolution" in report
        assert "Released: 1984-07-17" in report
        assert "Chain A/C" in report
        assert "2 copies" in report
        assert "Homo sapiens" in report
        assert "Bound components: HEM" in report
        assert "Described in PubMed 6726807" in report
        assert "VLSPADKTNV" in report

    def test_a_structure_reports_the_chain_count_it_has_not_the_number_fetched(self) -> None:
        # get_structure retrieves only the first few entities of a large assembly. Counting the
        # tuple would tell an agent a 56 chain ribosome is an 8 chain complex.
        structure = Structure(
            pdb_id="7K00",
            title="A 70S ribosome",
            method="ELECTRON MICROSCOPY",
            resolution=2.0,
            released="2020-01-01",
            chains=tuple(
                Chain(
                    entity_id=str(i),
                    description=f"protein {i}",
                    chain_ids=(str(i),),
                    organism="Escherichia coli",
                    sequence="M" * 50,
                    copies=1,
                )
                for i in range(MAX_CHAINS_REPORTED)
            ),
            entity_count=56,
        )
        report = structure.report()

        assert "56 polymer chain(s)" in report
        assert f"and {56 - MAX_CHAINS_REPORTED} more, not fetched" in report


class TestWhatTheReviewFound:
    """Behaviour added because a review showed the report or the error was misleading."""

    def test_the_comments_are_capped_like_everything_else(self) -> None:
        # Comments were the one unbounded field: a well curated entry carries over 11,000
        # characters of them, which is more than the capped sequence and the rest put together
        entry = INSULIN_ENTRY | {
            "comments": [
                {"commentType": "FUNCTION", "texts": [{"value": "F" * 5_000}]},
                {"commentType": "SUBUNIT", "texts": [{"value": "S" * 5_000}]},
            ]
        }
        report = protein_from(entry).report()

        assert "characters not shown" in report
        assert "F" * 5_000 not in report
        assert len(report) < 3_000

    def test_a_long_annotation_says_how_much_it_left_out(self) -> None:
        assert "400 of 1,400 characters not shown" in truncate_text("x" * 1_400, 1_000)

    def test_the_annotations_shown_are_capped_by_count(self) -> None:
        # A curated entry carries many comments of one type rather than one of each, and there
        # are fewer reportable types than the cap, so only the former exercises it
        entry = INSULIN_ENTRY | {
            "comments": [
                {"commentType": "FUNCTION", "texts": [{"value": f"finding {i}"}]}
                for i in range(MAX_COMMENTS_REPORTED + 6)
            ]
        }
        protein = protein_from(entry)
        report = protein.report()

        assert len(protein.comments) == MAX_COMMENTS_REPORTED + 6
        assert report.count("finding ") == MAX_COMMENTS_REPORTED
        assert "and 6 more annotation(s) not shown" in report

    def test_features_are_kept_by_importance_not_by_position(self) -> None:
        # A protein carries far more eligible features than the cap. Taken in the order the
        # service returns them, the cap keeps whatever is near the N terminus and discards the
        # binding sites, which is the opposite of why the list was chosen.
        entry = INSULIN_ENTRY | {
            "features": [
                {"type": "Region", "location": {"start": {"value": i}, "end": {"value": i}}}
                for i in range(MAX_FEATURES_REPORTED + 10)
            ]
            + [{"type": "Binding site", "location": {"start": {"value": 900}, "end": {"value": 900}}}]
        }
        kinds = [kind for kind, _, _ in protein_from(entry).features]

        assert "Binding site" in kinds

    def test_a_half_known_range_is_not_rendered_as_none(self) -> None:
        # UniProt reports an undetermined terminus as a null, and writing "None-1465" reads as
        # a parsing failure rather than as the fact that one end is unknown
        entry = INSULIN_ENTRY | {
            "features": [
                {
                    "type": "Chain",
                    "location": {"start": {"value": None}, "end": {"value": 1465}},
                    "description": "WND",
                }
            ]
        }
        protein = protein_from(entry)

        assert protein.features[0][1] == "?-1465"
        assert "None" not in protein.report()


class TestWhenADatabaseCannotAnswer:
    """A service being down and an identifier naming nothing are different answers."""

    def test_a_service_outage_is_not_reported_as_a_missing_protein(self, web_transport) -> None:
        # Telling an agent the protein does not exist because the service returned 503 is worse
        # than telling it nothing, because it will stop looking
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=503) for _ in range(5)]

        with pytest.raises(WebRequestError) as raised:
            get_protein("P01308")

        assert not isinstance(raised.value, RecordNotFoundError)

    def test_a_rejected_accession_is_reported_as_a_missing_protein(self, web_transport) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=400)]

        with pytest.raises(RecordNotFoundError):
            get_protein("NOTANACCESSION")

    def test_an_outage_is_not_reported_as_a_missing_structure(self, web_transport) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=503) for _ in range(5)]

        with pytest.raises(WebRequestError) as raised:
            get_structure("4HHB")

        assert not isinstance(raised.value, RecordNotFoundError)

    def test_an_outage_is_not_reported_as_a_missing_prediction(self, web_transport) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=503) for _ in range(5)]

        with pytest.raises(WebRequestError) as raised:
            get_predicted_structure("P01308")

        assert not isinstance(raised.value, RecordNotFoundError)

    def test_a_withdrawn_accession_is_an_error_not_a_zero_length_protein(
        self, web_transport
    ) -> None:
        # UniProt answers a deleted accession with 200 and a stub, which parses into a plausible
        # protein of no residues. A model proposing an accession from an older paper hits this.
        web_transport.queue(
            {
                "entryType": "Inactive",
                "primaryAccession": "A0A008APQ8",
                "uniProtkbId": "A0A008APQ8_STAAU",
                "inactiveReason": {"inactiveReasonType": "DELETED"},
            }
        )

        with pytest.raises(RecordNotFoundError, match="no longer active"):
            get_protein("A0A008APQ8")

    def test_a_demerged_accession_names_what_it_became(self, web_transport) -> None:
        web_transport.queue(
            {
                "entryType": "Inactive",
                "primaryAccession": "Q00001",
                "inactiveReason": {
                    "inactiveReasonType": "DEMERGED",
                    "mergeDemergeTo": ["P11111", "P22222"],
                },
            }
        )

        with pytest.raises(RecordNotFoundError, match="P11111, P22222"):
            get_protein("Q00001")

    def test_a_prediction_endpoint_that_changes_shape_is_an_error(self, web_transport) -> None:
        # Indexing an object as if it were the documented list gives a KeyError the agent
        # cannot act on
        web_transport.queue({"predictions": []})

        with pytest.raises(RecordNotFoundError):
            get_predicted_structure("P01308")


class TestLookingUpAProtein:
    def test_the_accession_is_encoded_into_the_url(self, web_transport) -> None:
        web_transport.queue(INSULIN_ENTRY)

        get_protein("P01308")

        assert web_transport.urls == ["https://rest.uniprot.org/uniprotkb/P01308.json"]

    def test_a_hostile_accession_cannot_reshape_the_url(self, web_transport) -> None:
        web_transport.queue(INSULIN_ENTRY)

        get_protein("../../../etc/passwd")

        assert "../.." not in web_transport.urls[0]
        assert web_transport.urls[0].startswith("https://rest.uniprot.org/uniprotkb/")

    def test_an_unknown_accession_says_what_an_accession_looks_like(self, web_transport) -> None:
        # UniProt answers an unknown accession with 400, so the status cannot distinguish a bad
        # identifier from a bad request, and the message has to do the work instead
        web_transport.responses = [
            __import__("conftest").FakeResponse(status_code=400) for _ in range(3)
        ]

        with pytest.raises(RecordNotFoundError, match="P01308"):
            get_protein("NOTREAL999")

    def test_not_found_is_a_database_error(self) -> None:
        # So a caller can catch one kind and handle every database the same way
        assert issubclass(RecordNotFoundError, DatabaseError)


class TestSearchingForProteins:
    def test_the_query_travels_as_a_parameter(self, web_transport) -> None:
        web_transport.queue({"results": []})

        search_proteins("insulin receptor", limit=5)

        assert web_transport.params[0]["query"] == "insulin receptor"
        assert web_transport.params[0]["size"] == 5

    def test_the_search_goes_to_the_search_endpoint(self, web_transport) -> None:
        # Asserting the parameters says nothing about where they were sent
        web_transport.queue({"results": []})

        search_proteins("insulin")

        assert web_transport.urls == ["https://rest.uniprot.org/uniprotkb/search"]

    def test_only_the_fields_a_hit_needs_are_asked_for(self, web_transport) -> None:
        # Without this the service returns whole entries, and a 25 hit search becomes megabytes
        web_transport.queue({"results": []})

        search_proteins("insulin")

        requested = web_transport.params[0]["fields"].split(",")

        assert "accession" in requested
        assert "sequence" not in requested

    def test_curated_entries_are_not_the_default(self) -> None:
        # Learned the hard way: with this on, a search for a nanobody returns proteins whose
        # reference titles mention one, because every real camelid VHH entry is uncurated
        from inspect import signature

        assert signature(search_proteins).parameters["reviewed_only"].default is False

    def test_asking_for_curated_entries_adds_the_filter(self, web_transport) -> None:
        web_transport.queue({"results": []})

        search_proteins("insulin", reviewed_only=True)

        assert web_transport.params[0]["query"] == "(insulin) AND reviewed:true"

    def test_the_query_is_sent_unchanged_when_not_filtering(self, web_transport) -> None:
        web_transport.queue({"results": []})

        search_proteins("gene:INS AND organism_id:9606")

        assert web_transport.params[0]["query"] == "gene:INS AND organism_id:9606"

    def test_results_are_summarised_one_per_line(self, web_transport) -> None:
        web_transport.queue({"results": [INSULIN_ENTRY]})

        results = search_proteins("insulin")

        assert len(results.hits) == 1
        assert "P01308" in results.report()
        assert "Insulin" in results.report()
        assert "Homo sapiens" in results.report()

    def test_an_empty_result_suggests_what_to_do(self, web_transport) -> None:
        web_transport.queue({"results": []})

        report = search_proteins("zzzqqq").report()

        assert "No UniProt entries matched" in report
        assert "fewer terms" in report

    def test_the_report_warns_that_ranking_includes_references(self, web_transport) -> None:
        # Without this an agent reads the top hit as an example of what it searched for
        web_transport.queue({"results": [INSULIN_ENTRY]})

        assert "references" in search_proteins("nanobody").report()

    def test_a_request_for_too_many_results_is_reduced(self, web_transport) -> None:
        web_transport.queue({"results": []})

        search_proteins("insulin", limit=10_000)

        assert web_transport.params[0]["size"] == MAX_SEARCH_RESULTS

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_nonsensical_limit_is_refused(self, limit: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            search_proteins("insulin", limit=limit)

    def test_a_missing_results_key_does_not_raise(self, web_transport) -> None:
        # A service answering 200 with something unexpected should not end a meeting
        web_transport.queue({"messages": ["service is busy"]})

        assert search_proteins("insulin").hits == ()

    def test_bounded_reports_what_it_is_bounding(self) -> None:
        with pytest.raises(ValueError, match="limit must be at least 1"):
            bounded(0, 25, "limit")

        assert bounded(5, 25, "limit") == 5
        assert bounded(99, 25, "limit") == 25

    def test_a_whole_number_written_as_a_float_is_accepted(self) -> None:
        # JSON has one number type, and a model asking for 5.0 is asking for five
        assert bounded(5.0, 25, "limit") == 5
        assert type(bounded(5.0, 25, "limit")) is int

    @pytest.mark.parametrize(
        "limit", ["5", 5.5, float("nan"), float("inf"), True, None, [5]]
    )
    def test_a_limit_that_is_not_a_whole_number_is_refused_clearly(self, limit) -> None:
        # A string raised TypeError from the comparison, and NaN passed it, since every
        # comparison with NaN is false
        with pytest.raises(ValueError, match="limit must be a whole number"):
            bounded(limit, 25, "limit")

    def test_a_limit_that_is_not_a_whole_number_is_refused_before_any_request(
        self, web_transport
    ) -> None:
        with pytest.raises(ValueError, match="whole number"):
            search_proteins("insulin", limit="ten")

        assert web_transport.requests == []


class TestLookingUpAStructure:
    def test_the_entry_and_each_chain_are_fetched(self, web_transport) -> None:
        web_transport.queue(HAEMOGLOBIN_ENTRY, ALPHA_CHAIN, ALPHA_CHAIN)

        structure = get_structure("4hhb")

        assert web_transport.urls == [
            "https://data.rcsb.org/rest/v1/core/entry/4HHB",
            "https://data.rcsb.org/rest/v1/core/polymer_entity/4HHB/1",
            "https://data.rcsb.org/rest/v1/core/polymer_entity/4HHB/2",
        ]
        assert len(structure.chains) == 2

    def test_the_identifier_is_upper_cased(self, web_transport) -> None:
        # The PDB accepts either case but reports its own identifiers in upper case, and a report
        # that echoes what was typed makes two lookups of one structure look like two structures
        web_transport.queue(HAEMOGLOBIN_ENTRY, ALPHA_CHAIN, ALPHA_CHAIN)

        assert get_structure("4hhb").pdb_id == "4HHB"

    def test_the_fields_a_report_depends_on_are_read(self, web_transport) -> None:
        web_transport.queue(HAEMOGLOBIN_ENTRY, ALPHA_CHAIN, ALPHA_CHAIN)

        structure = get_structure("4HHB")

        assert structure.method == "X-RAY DIFFRACTION"
        assert structure.resolution == 1.74
        assert structure.released == "1984-07-17"
        assert structure.ligand_ids == ("HEM",)
        assert structure.pubmed_id == 6726807
        assert "DEOXYHAEMOGLOBIN" in structure.title

    def test_a_chain_carries_its_organism_copies_and_sequence(self, web_transport) -> None:
        web_transport.queue(HAEMOGLOBIN_ENTRY, ALPHA_CHAIN, ALPHA_CHAIN)

        chain = get_structure("4HHB").chains[0]

        assert chain.chain_ids == ("A", "C")
        assert chain.copies == 2
        assert chain.organism == "Homo sapiens"
        assert chain.sequence.startswith("VLSPADK")

    def test_sequences_can_be_skipped_to_save_requests(self, web_transport) -> None:
        web_transport.queue(HAEMOGLOBIN_ENTRY)

        structure = get_structure("4HHB", include_sequences=False)

        assert structure.chains == ()
        assert len(web_transport.requests) == 1

    def test_one_unreadable_chain_does_not_lose_the_structure(self, web_transport) -> None:
        from conftest import FakeResponse

        web_transport.responses = [
            FakeResponse(json_body=HAEMOGLOBIN_ENTRY),
            FakeResponse(json_body=ALPHA_CHAIN),
            *[FakeResponse(status_code=500) for _ in range(3)],
        ]

        structure = get_structure("4HHB")

        assert len(structure.chains) == 1
        assert "DEOXYHAEMOGLOBIN" in structure.report()

    def test_an_unreadable_chain_is_reported_as_none(self, web_transport) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=404)]

        assert get_chain("4HHB", "1") is None

    def test_the_chains_fetched_are_capped(self, web_transport) -> None:
        from conftest import FakeResponse

        entry = HAEMOGLOBIN_ENTRY | {
            "rcsb_entry_container_identifiers": {
                "polymer_entity_ids": [str(i) for i in range(50)]
            }
        }
        web_transport.responses = [
            FakeResponse(json_body=entry),
            *[FakeResponse(json_body=ALPHA_CHAIN) for _ in range(60)],
        ]

        get_structure("4HHB")

        # One for the entry, and no more than the cap for the chains, or a large complex would
        # cost fifty requests to describe
        assert len(web_transport.requests) == 1 + MAX_CHAINS_REPORTED

    def test_an_unknown_identifier_says_what_one_looks_like(self, web_transport) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=404)]

        with pytest.raises(RecordNotFoundError, match="4HHB"):
            get_structure("ZZZZ")

    def test_a_structure_with_no_chains_still_reports(self, web_transport) -> None:
        web_transport.queue({"struct": {"title": "A structure"}})

        assert "A structure" in get_structure("1ABC").report()


class TestLookingUpAPrediction:
    def test_the_fields_a_report_depends_on_are_read(self, web_transport) -> None:
        web_transport.queue(INSULIN_PREDICTION)

        prediction = get_predicted_structure("P01308")

        assert prediction.accession == "P01308"
        assert prediction.name == "Insulin"
        assert prediction.mean_plddt == 52.91
        assert prediction.version == 6
        assert prediction.cif_url.endswith("model_v6.cif")

    def test_the_request_goes_to_the_prediction_endpoint(self, web_transport) -> None:
        web_transport.queue(INSULIN_PREDICTION)

        get_predicted_structure("P01308")

        assert web_transport.urls == [
            "https://alphafold.ebi.ac.uk/api/prediction/P01308"
        ]

    def test_every_field_the_record_holds_reaches_the_report(self, web_transport) -> None:
        # Parsing a field and then dropping it on the way to the model is the same as not
        # having it, and the parsing tests above cannot see the difference
        web_transport.queue(INSULIN_PREDICTION)

        report = get_predicted_structure("P01308").report()

        assert "P01308" in report
        assert "Insulin" in report
        assert "version 6" in report
        assert "52.9" in report
        assert "pLDDT over 70" in report

    def test_a_prediction_with_nothing_confident_says_so(self, web_transport) -> None:
        # Both bands at zero is a fact about a disordered protein, not a missing field
        web_transport.queue(
            [
                INSULIN_PREDICTION[0]
                | {"fractionPlddtConfident": 0.0, "fractionPlddtVeryHigh": 0.0}
            ]
        )

        prediction = get_predicted_structure("P01308")

        assert prediction.fraction_confident == 0.0
        assert "Residues at pLDDT over 70: 0%" in prediction.report()

    def test_the_confident_fraction_combines_both_bands(self, web_transport) -> None:
        web_transport.queue(INSULIN_PREDICTION)

        assert get_predicted_structure("P01308").fraction_confident == pytest.approx(0.13)

    @pytest.mark.parametrize(
        "score, expected",
        [
            (95.0, "very high confidence"),
            (75.0, "confident backbone"),
            (55.0, "low confidence"),
            (30.0, "disordered"),
        ],
    )
    def test_the_score_is_explained_not_just_stated(self, score: float, expected: str) -> None:
        # A low pLDDT does not mean a blurrier picture of the same structure, and an agent that
        # reads it that way will build on a prediction that says there is nothing to build on
        prediction = PredictedStructure(
            accession="P01308",
            name="Insulin",
            organism="Homo sapiens",
            sequence="MKV",
            mean_plddt=score,
            version=6,
            fraction_confident=0.5,
            pdb_url="",
            cif_url="",
        )

        assert expected in prediction.confidence
        assert expected in prediction.report()

    def test_a_missing_score_is_not_reported_as_zero(self) -> None:
        prediction = PredictedStructure(
            accession="P01308", name="", organism="", sequence="MKV",
            mean_plddt=None, version=None, fraction_confident=None, pdb_url="", cif_url="",
        )

        assert "confidence not reported" in prediction.confidence
        assert "pLDDT" not in prediction.report()

    def test_the_report_says_it_is_a_prediction(self, web_transport) -> None:
        web_transport.queue(INSULIN_PREDICTION)

        report = get_predicted_structure("P01308").report()

        assert "not a measurement" in report
        assert "prefer it" in report

    def test_an_empty_list_is_not_found_rather_than_an_index_error(self, web_transport) -> None:
        web_transport.queue([])

        with pytest.raises(RecordNotFoundError):
            get_predicted_structure("P01308")

    def test_an_accession_with_no_prediction_explains_what_is_covered(self, web_transport) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=400) for _ in range(3)]

        with pytest.raises(RecordNotFoundError, match="UniProt"):
            get_predicted_structure("NOTREAL")


class TestDownloadingAStructureFile:
    """Structure files go to disk, because the sandbox that runs the code has no network."""

    def test_the_file_is_written_and_named_from_the_url(self, web_transport, tmp_path) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data_4HHB\n")]

        downloaded = download_structure("4hhb", save_dir=tmp_path / "structures")

        assert downloaded.path.name == "4HHB.cif"
        assert downloaded.path.read_text() == "data_4HHB\n"
        assert downloaded.characters == 10

    def test_the_report_tells_the_agent_how_to_read_it(self, web_transport, tmp_path) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data")]

        report = download_structure("4HHB", save_dir=tmp_path).report()

        assert "4HHB.cif" in report
        # With no working directory to relate it to, the only honest answer is the full path
        assert str(tmp_path / "4HHB.cif") in report

    def test_the_report_gives_the_path_the_running_code_will_use(
        self, web_transport, tmp_path
    ) -> None:
        # The file is written beside the code, not into it, so the bare name would be wrong
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data")]

        report = download_structure(
            "4HHB", save_dir=tmp_path / "structures", working_dir=tmp_path
        ).report()

        assert 'Read it from code as "structures/4HHB.cif"' in report

    def test_a_file_outside_the_working_directory_is_reported_absolutely(
        self, web_transport, tmp_path
    ) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data")]
        elsewhere = tmp_path / "elsewhere"

        report = download_structure(
            "4HHB", save_dir=elsewhere, working_dir=tmp_path / "code"
        ).report()

        assert f'Read it from code as "{elsewhere / "4HHB.cif"}"' in report

    def test_the_directory_is_created(self, web_transport, tmp_path) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data")]
        target = tmp_path / "does" / "not" / "exist"

        assert download_structure("4HHB", save_dir=target).path.parent == target.resolve()

    @pytest.mark.parametrize("file_format, expected", [("cif", ".cif"), ("pdb", ".pdb")])
    def test_both_formats_can_be_asked_for(
        self, web_transport, tmp_path, file_format: str, expected: str
    ) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"data")]

        downloaded = download_structure("4HHB", save_dir=tmp_path, file_format=file_format)

        assert downloaded.path.suffix == expected

    def test_an_unknown_format_is_refused_before_any_request(self, web_transport, tmp_path) -> None:
        with pytest.raises(ValueError, match="cif"):
            download_structure("4HHB", save_dir=tmp_path, file_format="exe")

        assert web_transport.requests == []

    def test_an_unknown_source_is_refused_before_any_request(self, web_transport, tmp_path) -> None:
        with pytest.raises(ValueError, match="source"):
            download_structure("4HHB", save_dir=tmp_path, source="http://evil.com")

        assert web_transport.requests == []

    def test_a_prediction_file_name_is_asked_for_not_guessed(self, web_transport, tmp_path) -> None:
        # The name carries a model version, so an assembled URL looks right and 404s
        from conftest import FakeResponse

        web_transport.responses = [
            FakeResponse(json_body=INSULIN_PREDICTION),
            FakeResponse(body=b"data_AF"),
        ]

        downloaded = download_structure("P01308", save_dir=tmp_path, source="alphafold")

        assert downloaded.path.name == "AF-P01308-F1-model_v6.cif"
        assert web_transport.urls[1] == INSULIN_PREDICTION[0]["cifUrl"]

    def test_a_prediction_without_the_format_asked_for_is_reported(
        self, web_transport, tmp_path
    ) -> None:
        web_transport.queue([INSULIN_PREDICTION[0] | {"pdbUrl": ""}])

        with pytest.raises(RecordNotFoundError, match="no pdb file"):
            download_structure("P01308", save_dir=tmp_path, source="alphafold", file_format="pdb")

    def test_a_file_name_the_service_chooses_is_not_used(self, web_transport, tmp_path) -> None:
        # The URL comes from AlphaFold's own response rather than from a template here. Naming the
        # file after its last segment let a response ending in /solve.py overwrite that file
        from conftest import FakeResponse

        (tmp_path / "solve.py").write_text("print('mine')")
        web_transport.responses = [
            FakeResponse(
                json_body=[
                    INSULIN_PREDICTION[0] | {"cifUrl": "https://alphafold.ebi.ac.uk/files/solve.py"}
                ]
            ),
            FakeResponse(body=b"data"),
        ]

        downloaded = download_structure("P01308", save_dir=tmp_path, source="alphafold")

        assert downloaded.path.name == "AF-P01308-F1-model_v6.cif"
        assert (tmp_path / "solve.py").read_text() == "print('mine')"

    def test_a_prediction_without_a_version_is_named_without_one(
        self, web_transport, tmp_path
    ) -> None:
        from conftest import FakeResponse

        web_transport.responses = [
            FakeResponse(json_body=[INSULIN_PREDICTION[0] | {"latestVersion": "six"}]),
            FakeResponse(body=b"data"),
        ]

        downloaded = download_structure(
            "p01308", save_dir=tmp_path, source="alphafold", file_format="pdb"
        )

        assert downloaded.path.name == "AF-P01308-F1.pdb"

    @pytest.mark.parametrize("identifier", ["../P01308", "P01308/x", "solve.py", "P0130"])
    def test_an_identifier_that_is_not_an_accession_is_refused_before_any_request(
        self, web_transport, tmp_path, identifier: str
    ) -> None:
        with pytest.raises(ValueError, match="not a UniProt accession"):
            download_structure(identifier, save_dir=tmp_path, source="alphafold")

        assert web_transport.requests == []

    def test_a_name_that_escapes_after_resolution_is_refused(
        self, web_transport, tmp_path, monkeypatch
    ) -> None:
        # check_filename runs on the name as written; the containment check runs on the path as
        # resolved. Only the second can see a name that becomes an escape once it is joined, so
        # it is tested by letting such a name through the first and asserting the second holds.
        from conftest import FakeResponse

        monkeypatch.setattr(databases, "check_filename", lambda name: "../outside.cif")
        web_transport.responses = [FakeResponse(body=b"data")]

        with pytest.raises(UnsafeFilenameError, match="outside"):
            download_structure("4HHB", save_dir=tmp_path / "inside")

        assert not (tmp_path / "outside.cif").exists()

    def test_the_name_check_refuses_a_traversing_segment_on_its_own(self) -> None:
        # The guard above is the second of two. This is the first, which is what actually stops
        # the case above in production, so neither is left resting on the other.
        with pytest.raises(UnsafeFilenameError):
            databases.check_filename("..")

    def test_a_download_that_fails_does_not_leave_a_file(self, web_transport, tmp_path) -> None:
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(status_code=404)]

        with pytest.raises(RecordNotFoundError):
            download_structure("ZZZZ", save_dir=tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_a_structure_file_is_not_cached(self, web_transport, tmp_path) -> None:
        # These are megabytes each, and the response cache is bounded, so caching one would evict
        # everything an agent had already looked up
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"first"), FakeResponse(body=b"second")]

        download_structure("4HHB", save_dir=tmp_path)
        second = download_structure("4HHB", save_dir=tmp_path)

        assert second.path.read_text() == "second"
        assert len(web_transport.requests) == 2

    def test_a_large_file_is_allowed_where_a_normal_response_is_not(
        self, web_transport, tmp_path
    ) -> None:
        # A response is capped at 5 MB because it would be shown to a model; this one is not
        from conftest import FakeResponse

        web_transport.responses = [FakeResponse(body=b"x" * 8_000_000)]

        assert download_structure("4HHB", save_dir=tmp_path).characters == 8_000_000


class TestAResponseOfTheWrongShape:
    """A field of the wrong type is read as absent, and a long one is shortened when it is read.

    Every one of these escaped as AttributeError, TypeError, or ValueError, which is none of the
    errors a caller handles, or made a report the size of the response.
    """

    @pytest.mark.parametrize(
        "entry",
        [
            {"comments": [{"commentType": "FUNCTION", "texts": ["not an object"]}]},
            {"comments": ["not an object"], "features": "not a list"},
            {"features": [{"type": "Chain", "location": "not an object"}]},
            {"features": [{"type": "Chain", "location": {"start": "5", "end": ["7"]}}]},
            {"features": [{"type": "Chain", "location": {"start": {"value": "\u00b2"}}}]},
            {"proteinDescription": ["not an object"], "organism": "Homo sapiens"},
            {"genes": ["INS"], "uniProtKBCrossReferences": [["PDB", "1A7F"]]},
            {"sequence": {"value": 7, "length": "long", "molWeight": "heavy"}},
            {"sequence": {"value": "MA", "length": True, "molWeight": float("inf")}},
            {"organism": {"taxonId": "9606x"}, "primaryAccession": {"id": "P01308"}},
        ],
    )
    def test_a_protein_entry_of_the_wrong_shape_still_reports(self, entry) -> None:
        protein = protein_from(INSULIN_ENTRY | entry)

        assert protein.report().startswith("UniProt")

    def test_numbers_sent_as_strings_are_read_as_numbers(self) -> None:
        protein = protein_from(
            INSULIN_ENTRY | {"sequence": {"value": "MA", "length": "110", "molWeight": "11981"}}
        )

        assert protein.length == 110
        assert protein.mass == 11981.0
        assert "11,981 Da" in protein.report()

    def test_a_comment_without_any_text_is_left_out(self) -> None:
        protein = protein_from(
            INSULIN_ENTRY | {"comments": [{"commentType": "FUNCTION"}, {"commentType": "DOMAIN"}]}
        )

        assert protein.comments == ()

    def test_a_feature_with_neither_end_known_is_placed_at_a_question_mark(self) -> None:
        protein = protein_from(
            INSULIN_ENTRY
            | {
                "features": [
                    {
                        "type": "Region",
                        "location": {"start": {"value": None}, "end": {"value": None}},
                        "description": "Disordered",
                    }
                ]
            }
        )

        assert protein.features == (("Region", "?", "Disordered"),)

    def test_every_list_and_name_in_a_protein_report_is_bounded(self) -> None:
        # One gene name repeated 10,000 times made a report of half a megabyte
        huge = "x" * 1_000_000
        protein = protein_from(
            INSULIN_ENTRY
            | {
                "genes": [{"geneName": {"value": f"GENE{index}"}} for index in range(10_000)],
                "proteinDescription": {"recommendedName": {"fullName": {"value": huge}}},
                "organism": {"scientificName": huge},
                "features": [
                    {"type": "Chain", "location": {"start": {"value": 1}}, "description": huge}
                ],
                "uniProtKBCrossReferences": [{"database": "PDB", "id": huge}] * 20,
            }
        )
        report = protein.report()

        assert len(report) < 20_000
        assert f"and {10_000 - MAX_ITEMS_LISTED:,} more" in report
        assert "GENE19" in report and "GENE20" not in report

    def test_every_list_and_name_in_a_structure_report_is_bounded(self, web_transport) -> None:
        huge = "x" * 1_000_000
        web_transport.queue(
            HAEMOGLOBIN_ENTRY
            | {
                "struct": {"title": huge},
                "exptl": [{"method": huge}] * 3,
                "rcsb_entry_info": {"nonpolymer_bound_components": ["HEM"] * 100_000},
            },
            ALPHA_CHAIN
            | {
                "entity_poly": {"pdbx_strand_id": ",".join(["A"] * 10_000)},
                "rcsb_polymer_entity": {"pdbx_description": huge},
            },
            ALPHA_CHAIN,
        )

        report = get_structure("4HHB").report()

        assert len(report) < 20_000
        assert f"and {100_000 - MAX_ITEMS_LISTED:,} more" in report
        assert f"and {10_000 - MAX_ITEMS_LISTED:,} more" in report

    @pytest.mark.parametrize(
        "entry",
        [
            {"struct": "a title", "exptl": "X-RAY", "rcsb_entry_info": []},
            {"rcsb_entry_info": {"resolution_combined": ["fine"]}},
            {"rcsb_entry_container_identifiers": {"pubmed_id": "PMID6726807"}},
        ],
    )
    def test_a_structure_entry_of_the_wrong_shape_still_reports(
        self, web_transport, entry
    ) -> None:
        web_transport.queue(HAEMOGLOBIN_ENTRY | entry, ["not an entity"], ALPHA_CHAIN)

        assert get_structure("4HHB").report().startswith("PDB 4HHB")

    def test_a_structure_response_that_is_not_an_object_is_read_as_empty(
        self, web_transport
    ) -> None:
        web_transport.queue(["not an entry"])

        structure = get_structure("4HHB")

        assert (structure.title, structure.chains, structure.entity_count) == ("", (), 0)

    def test_a_chain_of_the_wrong_shape_is_read_as_empty(self, web_transport) -> None:
        web_transport.queue(
            {
                "entity_poly": {"pdbx_strand_id": 7, "pdbx_seq_one_letter_code_can": ["M"]},
                "rcsb_polymer_entity": {"pdbx_number_of_molecules": "two"},
                "rcsb_entity_source_organism": ["Homo sapiens"],
            }
        )

        chain = get_chain("4HHB", "1")

        assert (chain.chain_ids, chain.sequence, chain.copies, chain.organism) == ((), "", 1, "")

    @pytest.mark.parametrize(
        "fields",
        [
            {"globalMetricValue": "52.9", "fractionPlddtConfident": "0.5"},
            {"globalMetricValue": "high", "fractionPlddtConfident": "x", "latestVersion": "six"},
            {"uniprotSequence": None, "uniprotDescription": {"text": "Insulin"}},
        ],
    )
    def test_a_prediction_of_the_wrong_shape_still_reports(self, web_transport, fields) -> None:
        web_transport.queue([INSULIN_PREDICTION[0] | fields])

        assert get_predicted_structure("P01308").report().startswith("AlphaFold")

    def test_a_confidence_sent_as_a_string_is_read_as_a_number(self, web_transport) -> None:
        web_transport.queue([INSULIN_PREDICTION[0] | {"globalMetricValue": "52.9"}])

        assert get_predicted_structure("P01308").mean_plddt == 52.9

    @pytest.mark.parametrize(
        "value,expected",
        [
            (7, 7),
            ("110", 110),
            (" -3 ", -3),
            (110.0, 110),
            (110.5, None),
            ("\u00b2", None),
            ("\u0663", None),
            ("9" * 5000, None),
            (10**20, None),
            (float("inf"), None),
            (float("nan"), None),
            (True, None),
            ("", None),
            ("-", None),
            (None, None),
            ([1], None),
        ],
    )
    def test_a_whole_number_is_read_only_when_it_is_one(self, value, expected) -> None:
        assert as_int(value) == expected

    def test_a_whole_number_that_is_not_one_takes_the_default(self) -> None:
        assert as_int("many", 0) == 0

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("INS", "INS"),
            (9606, "9606"),
            (None, ""),
            (False, ""),
            ({"value": "x"}, ""),
            (["x"], ""),
        ],
    )
    def test_a_text_field_is_read_only_when_it_is_text_or_a_number(self, value, expected) -> None:
        assert as_text(value) == expected

    def test_a_long_text_field_is_shortened_and_says_so(self) -> None:
        text = as_text("x" * 10_000)

        assert text.startswith("x" * MAX_FIELD_CHARACTERS)
        assert "not shown" in text
        assert as_text("x" * 10_000, limit=None) == "x" * 10_000

    def test_a_short_list_is_joined_without_a_count(self) -> None:
        assert listing(["a", "b"], 2) == "a, b"
        assert listing(["a", "b", "c"], 2, separator="/") == "a/b and 1 more"


class TestAgainstTheRealDatabases:
    """Confirms the fixtures above still match what the services send.

    Every other test here trusts a fixture. These are what notice when a service renames a field,
    which would otherwise show up as a report quietly missing half of itself.
    """

    @live_only
    def test_uniprot_still_returns_the_fields_the_fixtures_claim(self) -> None:
        protein = get_protein("P01308")

        assert protein.accession == "P01308"
        assert protein.entry_name == "INS_HUMAN"
        assert protein.name == "Insulin"
        assert protein.organism == "Homo sapiens"
        assert protein.length == 110
        assert protein.reviewed is True
        assert protein.genes == ("INS",)
        assert protein.sequence.startswith("MALWMRLLPLL")
        assert dict(protein.comments)["FUNCTION"]
        assert "1A7F" in protein.pdb_ids
        assert any(kind == "Disulfide bond" for kind, _, _ in protein.features)

    @live_only
    def test_a_uniprot_search_still_returns_hits_with_names(self) -> None:
        results = search_proteins("gene:INS AND organism_id:9606", limit=3)

        assert results.hits
        assert any(hit.accession == "P01308" for hit in results.hits)
        assert all(hit.length > 0 for hit in results.hits)

    @live_only
    def test_the_pdb_still_returns_the_fields_the_fixtures_claim(self) -> None:
        structure = get_structure("4HHB")

        assert structure.pdb_id == "4HHB"
        assert "HAEMOGLOBIN" in structure.title.upper()
        assert structure.resolution == 1.74
        assert structure.released == "1984-07-17"
        assert len(structure.chains) == 2
        assert structure.chains[0].sequence.startswith("VLSPADK")
        assert structure.chains[0].organism == "Homo sapiens"

    @live_only
    def test_alphafold_still_returns_the_fields_the_fixtures_claim(self) -> None:
        prediction = get_predicted_structure("P01308")

        assert prediction.accession == "P01308"
        assert prediction.organism == "Homo sapiens"
        assert prediction.mean_plddt is not None
        assert prediction.cif_url.startswith("https://alphafold.ebi.ac.uk/")
        assert len(prediction.sequence) == 110

    @live_only
    def test_a_structure_file_really_downloads(self, tmp_path) -> None:
        downloaded = download_structure("4HHB", save_dir=tmp_path, file_format="pdb")

        assert downloaded.path.exists()
        assert downloaded.characters > 100_000
        assert downloaded.path.read_text().startswith("HEADER")

    @live_only
    def test_an_unknown_accession_really_raises_not_found(self) -> None:
        with pytest.raises(RecordNotFoundError):
            get_protein("NOTANACCESSION123")
