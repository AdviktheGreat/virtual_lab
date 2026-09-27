"""Tests for the chemistry lookups agents use.

Offline against fixtures trimmed from real responses, for the same reason the protein tests are,
with live tests at the end that check the fixtures still describe reality.

Much of what is asserted here is not that a field is parsed but that a wrong answer is refused.
Both of these services will answer a question they cannot answer: PubChem returns CID 0 with a
200 for a structure it has never seen, and ChEMBL holds measurements whose units were lost in
standardisation and which are therefore wrong by eight orders of magnitude. Those are the cases
worth testing, because they are the ones that reach a model as fact.
"""

import os

import pytest

from conftest import FakeResponse
from virtual_lab import chemistry
from virtual_lab.chemistry import (
    Activity,
    ActivityResults,
    Compound,
    Drug,
    DrugHit,
    SearchHits,
    Target,
    activity_from,
    compound_from,
    get_activities,
    get_compound,
    get_drug,
    molecule_from,
    number_or_none,
    search_drugs,
    search_targets,
    tidy,
)
from virtual_lab.constants import (
    MAX_ACTIVITIES_REPORTED,
    MAX_ASSAY_DESCRIPTION_CHARACTERS,
    MAX_DESCRIPTION_CHARACTERS,
    MAX_SEARCH_RESULTS,
    MIN_PCHEMBL_REPORTED,
)
from virtual_lab.records import RecordNotFoundError
from virtual_lab.web import WebRequestError

live_only = pytest.mark.skipif(
    os.environ.get("VIRTUAL_LAB_LIVE_TESTS") != "1",
    reason="Set VIRTUAL_LAB_LIVE_TESTS=1 to query the real services",
)


# Trimmed from a real response for aspirin. The property names are the ones PubChem uses now:
# asking for CanonicalSMILES is answered with a key called ConnectivitySMILES.
ASPIRIN_PROPERTIES = {
    "CID": 2244,
    "MolecularFormula": "C9H8O4",
    "MolecularWeight": "180.16",
    "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ConnectivitySMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    "IUPACName": "2-acetyloxybenzoic acid",
    "XLogP": 1.2,
    "TPSA": 63.6,
    "Charge": 0,
    "HBondDonorCount": 1,
    "HBondAcceptorCount": 4,
    "RotatableBondCount": 3,
    "HeavyAtomCount": 13,
}

ASPIRIN_PROPERTY_RESPONSE = {"PropertyTable": {"Properties": [ASPIRIN_PROPERTIES]}}

# The real shape: a title with no description, then a one line regulatory notice, then the
# definition. The order is PubChem's own.
ASPIRIN_DESCRIPTION_RESPONSE = {
    "InformationList": {
        "Information": [
            {"CID": 2244, "Title": "Aspirin"},
            {
                "CID": 2244,
                "Description": "Aspirin can cause developmental toxicity.",
                "DescriptionSourceName": "California Office of Environmental Health Hazard",
            },
            {
                "CID": 2244,
                "Description": "Acetylsalicylic acid is a member of the class of benzoic acids "
                "that is salicylic acid in which the hydrogen attached to the phenolic hydroxy "
                "group has been replaced by an acetoxy group.",
                "DescriptionSourceName": "ChEBI",
            },
        ]
    }
}

ASPIRIN_MOLECULE = {
    "molecule_chembl_id": "CHEMBL25",
    "pref_name": "ASPIRIN",
    "max_phase": "4.0",
    "molecule_type": "Small molecule",
    "first_approval": 1950,
    "withdrawn_flag": False,
    "oral": True,
    "parenteral": False,
    "topical": False,
    "molecule_properties": {
        "full_mwt": "180.16",
        "alogp": "1.31",
        "hbd": 1,
        "hba": 3,
        "psa": "63.60",
        "num_ro5_violations": 0,
        "qed_weighted": "0.55",
    },
    "molecule_structures": {
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        "molfile": "\n     RDKit          2D\n\n 13 13  0  0",
    },
}

ASPIRIN_MECHANISM_RESPONSE = {
    "mechanisms": [
        {
            "action_type": "INHIBITOR",
            "mechanism_of_action": "Cyclooxygenase inhibitor",
            "target_chembl_id": "CHEMBL2094253",
        }
    ],
    "page_meta": {"total_count": 1},
}

EGFR_ACTIVITY = {
    "molecule_chembl_id": "CHEMBL176582",
    "target_chembl_id": "CHEMBL203",
    "target_pref_name": "Epidermal growth factor receptor",
    "target_organism": "Homo sapiens",
    "standard_type": "IC50",
    "standard_relation": "=",
    "standard_value": "0.01",
    "standard_units": "nM",
    "pchembl_value": "11.00",
    "assay_chembl_id": "CHEMBL683040",
    "assay_description": "Inhibition of tyrosine kinase activity",
    "document_year": 1996,
}


def property_response(**overrides):
    """Builds a PubChem property response with a field changed."""
    return {"PropertyTable": {"Properties": [ASPIRIN_PROPERTIES | overrides]}}


def queue(transport, *bodies):
    """Queues JSON responses in the order the code will ask for them."""
    transport.responses = [FakeResponse(json_body=body) for body in bodies]


class TestReadingACompoundRecord:
    def test_the_fields_of_a_compound_are_parsed(self) -> None:
        compound = compound_from(ASPIRIN_PROPERTIES, description="A drug.", title="Aspirin")

        assert compound.cid == 2244
        assert compound.title == "Aspirin"
        assert compound.formula == "C9H8O4"
        assert compound.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        assert compound.iupac_name == "2-acetyloxybenzoic acid"
        assert compound.description == "A drug."

    def test_a_weight_sent_as_a_string_becomes_a_number(self) -> None:
        # PubChem sends "180.16", not 180.16, and the report formats it as a number
        assert ASPIRIN_PROPERTIES["MolecularWeight"] == "180.16"
        assert compound_from(ASPIRIN_PROPERTIES).weight == pytest.approx(180.16)

    def test_the_report_names_the_compound_and_its_descriptors(self) -> None:
        report = compound_from(ASPIRIN_PROPERTIES, title="Aspirin").report()

        assert "CID 2244" in report
        assert "C9H8O4" in report
        assert "180.16" in report
        assert "XLogP: 1.2" in report
        assert "H-bond donors: 1" in report
        assert "pubchem.ncbi.nlm.nih.gov/compound/2244" in report

    def test_a_compound_without_stereochemistry_does_not_repeat_its_smiles(self) -> None:
        report = compound_from(ASPIRIN_PROPERTIES).report()

        assert report.count("CC(=O)OC1=CC=CC=C1C(=O)O") == 1
        assert "without stereochemistry" not in report

    def test_a_compound_with_stereochemistry_shows_both_smiles(self) -> None:
        compound = compound_from(
            ASPIRIN_PROPERTIES | {"SMILES": r"C/C=C\C(=O)O", "ConnectivitySMILES": "CC=CC(=O)O"}
        )
        report = compound.report()

        assert r"SMILES: C/C=C\C(=O)O" in report
        assert "SMILES without stereochemistry: CC=CC(=O)O" in report

    def test_a_missing_descriptor_is_left_out_rather_than_shown_as_none(self) -> None:
        bare = {"CID": 1, "MolecularFormula": "CH4"}
        report = compound_from(bare).report()

        assert "None" not in report
        assert "XLogP" not in report

    def test_a_long_description_is_capped(self) -> None:
        compound = compound_from(ASPIRIN_PROPERTIES, description="x" * 2_000)
        report = compound.report()

        assert "1,400 of 2,000 characters not shown" in report
        assert "x" * MAX_DESCRIPTION_CHARACTERS in report
        assert "x" * (MAX_DESCRIPTION_CHARACTERS + 1) not in report


class TestReadingADrugRecord:
    def test_the_fields_of_a_molecule_are_parsed(self) -> None:
        drug = molecule_from(ASPIRIN_MOLECULE, mechanisms=())

        assert drug.chembl_id == "CHEMBL25"
        assert drug.name == "ASPIRIN"
        assert drug.molecule_type == "Small molecule"
        assert drug.first_approval == 1950
        assert drug.smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert drug.routes == ("oral",)

    def test_a_phase_sent_as_a_string_becomes_a_number(self) -> None:
        assert ASPIRIN_MOLECULE["max_phase"] == "4.0"
        assert molecule_from(ASPIRIN_MOLECULE, mechanisms=()).max_phase == pytest.approx(4.0)

    def test_the_molfile_is_never_reported(self) -> None:
        # It is 1.4 kB of coordinates that ChEMBL sends whether asked for or not, and it is of no
        # use in a discussion
        report = molecule_from(ASPIRIN_MOLECULE, mechanisms=()).report()

        assert "RDKit" not in report
        assert "V2000" not in report

    def test_the_report_says_what_the_phase_number_means(self) -> None:
        report = molecule_from(ASPIRIN_MOLECULE, mechanisms=()).report()

        assert "approved (first approved 1950)" in report
        assert "max_phase" not in report

    def test_the_mechanisms_are_reported(self) -> None:
        drug = molecule_from(
            ASPIRIN_MOLECULE,
            mechanisms=(("INHIBITOR", "Cyclooxygenase inhibitor", "CHEMBL2094253"),),
        )

        assert "inhibitor Cyclooxygenase inhibitor (CHEMBL2094253)" in drug.report()

    def test_a_withdrawn_drug_says_so(self) -> None:
        drug = molecule_from(ASPIRIN_MOLECULE | {"withdrawn_flag": True}, mechanisms=())

        assert "Withdrawn" in drug.report()


class TestWhatAPhaseNumberMeans:
    """ChEMBL's max_phase is a number whose ordering does not match its meaning."""

    @pytest.mark.parametrize(
        "phase,expected",
        [
            ("4.0", "approved"),
            ("3.0", "reached clinical phase 3"),
            ("0.5", "reached clinical phase 0.5"),
            ("0", "no clinical development recorded"),
            (None, "clinical status not recorded"),
        ],
    )
    def test_each_phase_is_described_in_words(self, phase, expected) -> None:
        drug = molecule_from(
            ASPIRIN_MOLECULE | {"max_phase": phase, "first_approval": None}, mechanisms=()
        )

        assert drug.development_stage == expected

    def test_an_unknown_status_is_not_reported_as_a_negative_phase(self) -> None:
        # ChEMBL uses -1 for 2,499 molecules whose clinical status it does not know, including
        # camptothecin. "reached clinical phase -1" reads as a parsing failure.
        drug = molecule_from(ASPIRIN_MOLECULE | {"max_phase": "-1.0"}, mechanisms=())

        assert drug.development_stage == "clinical status not recorded"
        assert "-1" not in drug.report()

    def test_a_search_hit_describes_an_unknown_status_too(self) -> None:
        assert "clinical status unknown" in DrugHit("CHEMBL1", max_phase=-1.0).summary()
        assert "approved" in DrugHit("CHEMBL1", max_phase=4.0).summary()


class TestNumbersSentAsStrings:
    @pytest.mark.parametrize(
        "value,expected",
        [("180.16", 180.16), (1.5, 1.5), (3, 3.0), ("-1.0", -1.0), ("", None), (None, None)],
    )
    def test_a_number_is_read_whatever_type_it_arrived_as(self, value, expected) -> None:
        assert number_or_none(value) == expected

    def test_a_boolean_is_not_a_number(self) -> None:
        # float(True) is 1.0, which would turn a flag into a molecular weight
        assert number_or_none(True) is None
        assert number_or_none(False) is None

    def test_something_that_is_not_a_number_is_refused(self) -> None:
        assert number_or_none("about four") is None
        assert number_or_none({"value": 1}) is None

    def test_an_absent_field_is_described_rather_than_printed(self) -> None:
        assert tidy(None) == "not stated"
        assert tidy("") == "not stated"
        assert tidy("Small molecule") == "Small molecule"


class TestLookingUpACompound:
    def test_the_lookup_reaches_pubchem(self, web_transport) -> None:
        queue(web_transport, ASPIRIN_PROPERTY_RESPONSE, ASPIRIN_DESCRIPTION_RESPONSE)
        get_compound("aspirin")

        assert "pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin" in web_transport.urls[0]
        assert "MolecularFormula" in web_transport.urls[0]

    def test_the_renamed_properties_are_the_ones_asked_for(self, web_transport) -> None:
        # PubChem renamed these in 2025. A request for CanonicalSMILES is answered with a key
        # called ConnectivitySMILES, so code that asks for the old name and reads back what it
        # asked for raises KeyError on a perfectly good response.
        queue(web_transport, ASPIRIN_PROPERTY_RESPONSE, ASPIRIN_DESCRIPTION_RESPONSE)
        get_compound("aspirin")

        assert "CanonicalSMILES" not in web_transport.urls[0]
        assert "IsomericSMILES" not in web_transport.urls[0]
        assert "ConnectivitySMILES" in web_transport.urls[0]

    def test_a_smiles_is_sent_as_a_parameter_not_a_path_segment(self, web_transport) -> None:
        # A SMILES carries "/" and "\" for stereochemistry. Percent-encoded into a path segment
        # PubChem answers 400, which the shared rule would have reported as "no such compound"
        # for a compound that exists.
        queue(web_transport, ASPIRIN_PROPERTY_RESPONSE, ASPIRIN_DESCRIPTION_RESPONSE)
        get_compound(r"C/C=C\C(=O)O", namespace="smiles")

        assert "%2F" not in web_transport.urls[0]
        assert web_transport.params[0] == {"smiles": r"C/C=C\C(=O)O"}

    def test_an_identifier_still_cannot_add_path_segments(self, web_transport) -> None:
        queue(web_transport, ASPIRIN_PROPERTY_RESPONSE, ASPIRIN_DESCRIPTION_RESPONSE)
        get_compound("../../../etc/passwd")

        assert "etc/passwd" not in web_transport.urls[0]
        assert "%2F" in web_transport.urls[0]

    def test_an_unknown_namespace_is_refused_before_any_request(self, web_transport) -> None:
        with pytest.raises(ValueError, match="inchi"):
            get_compound("x", namespace="inchi")

        assert web_transport.requests == []

    def test_the_longest_description_is_the_one_shown(self, web_transport) -> None:
        # The first description PubChem returns for aspirin is a Proposition 65 notice; the one
        # that says what the molecule is comes after it
        queue(web_transport, ASPIRIN_PROPERTY_RESPONSE, ASPIRIN_DESCRIPTION_RESPONSE)
        compound = get_compound("aspirin")

        assert "member of the class of benzoic acids" in compound.description
        assert "developmental toxicity" not in compound.description

    def test_a_compound_without_a_description_is_not_a_failed_lookup(self, web_transport) -> None:
        queue(
            web_transport,
            ASPIRIN_PROPERTY_RESPONSE,
            {"InformationList": {"Information": [{"CID": 2244, "Title": "Aspirin"}]}},
        )
        compound = get_compound("aspirin")

        assert compound.title == "Aspirin"
        assert compound.description == ""

    def test_a_failed_description_does_not_lose_the_compound(self, web_transport) -> None:
        web_transport.responses = [
            FakeResponse(json_body=ASPIRIN_PROPERTY_RESPONSE),
            FakeResponse(status_code=500),
            FakeResponse(status_code=500),
            FakeResponse(status_code=500),
        ]
        compound = get_compound("aspirin")

        assert compound.cid == 2244
        assert compound.description == ""


class TestWhenPubChemCannotAnswer:
    def test_an_unknown_name_is_reported_as_a_missing_compound(self, web_transport) -> None:
        web_transport.responses = [FakeResponse(status_code=404)]

        with pytest.raises(RecordNotFoundError, match="notacompound"):
            get_compound("notacompound")

    def test_a_service_outage_is_not_reported_as_a_missing_compound(self, web_transport) -> None:
        web_transport.responses = [FakeResponse(status_code=503) for _ in range(5)]

        with pytest.raises(WebRequestError) as raised:
            get_compound("aspirin")

        assert not isinstance(raised.value, RecordNotFoundError)

    def test_a_rejected_request_is_not_reported_as_a_missing_compound(self, web_transport) -> None:
        # PubChem keeps 400 for a request it could not parse, unlike UniProt. Treating it as a
        # missing record would report a compound as nonexistent whenever a URL was built wrongly,
        # which is how the SMILES path encoding would have been hidden.
        web_transport.responses = [FakeResponse(status_code=400)]

        with pytest.raises(WebRequestError) as raised:
            get_compound("aspirin")

        assert not isinstance(raised.value, RecordNotFoundError)

    def test_a_structure_pubchem_does_not_hold_is_reported_as_missing(self, web_transport) -> None:
        # PubChem answers a valid but unknown structure with 200 and CID 0, so the status says
        # nothing and the identifier is the only sign that nothing was found
        queue(web_transport, {"PropertyTable": {"Properties": [{"CID": 0}]}})

        with pytest.raises(RecordNotFoundError, match="holds no such structure"):
            get_compound("CCCCCCCCCCCCCCCCCCCCCCN1C(=O)C2(CC2)Oc3ccccc31", namespace="smiles")

    def test_an_empty_property_table_is_reported_as_missing(self, web_transport) -> None:
        queue(web_transport, {"PropertyTable": {"Properties": []}})

        with pytest.raises(RecordNotFoundError):
            get_compound("aspirin")


class TestLookingUpADrug:
    def test_the_lookup_reaches_chembl(self, web_transport) -> None:
        queue(web_transport, ASPIRIN_MOLECULE, ASPIRIN_MECHANISM_RESPONSE)
        get_drug("CHEMBL25")

        assert "www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json" in web_transport.urls[0]
        assert "www.ebi.ac.uk/chembl/api/data/mechanism" in web_transport.urls[1]

    def test_the_request_narrows_the_fields(self, web_transport) -> None:
        # Without this ChEMBL sends the whole record, which is 11,500 characters for aspirin
        queue(web_transport, ASPIRIN_MOLECULE, ASPIRIN_MECHANISM_RESPONSE)
        get_drug("CHEMBL25")

        assert "molecule_chembl_id" in web_transport.params[0]["only"]
        assert "molecule_structures" in web_transport.params[0]["only"]

    def test_the_identifier_is_normalised(self, web_transport) -> None:
        queue(web_transport, ASPIRIN_MOLECULE, ASPIRIN_MECHANISM_RESPONSE)
        get_drug("  chembl25  ")

        assert "CHEMBL25.json" in web_transport.urls[0]

    def test_an_unknown_identifier_is_reported_as_missing(self, web_transport) -> None:
        web_transport.responses = [FakeResponse(status_code=404)]

        with pytest.raises(RecordNotFoundError, match="CHEMBL99999999"):
            get_drug("CHEMBL99999999")

    def test_an_outage_is_not_reported_as_a_missing_drug(self, web_transport) -> None:
        web_transport.responses = [FakeResponse(status_code=503) for _ in range(5)]

        with pytest.raises(WebRequestError) as raised:
            get_drug("CHEMBL25")

        assert not isinstance(raised.value, RecordNotFoundError)

    def test_a_response_that_is_not_a_molecule_is_reported_as_missing(self, web_transport) -> None:
        queue(web_transport, {"error_message": "not found"})

        with pytest.raises(RecordNotFoundError):
            get_drug("CHEMBL25")

    def test_a_failed_mechanism_lookup_does_not_lose_the_drug(self, web_transport) -> None:
        web_transport.responses = [
            FakeResponse(json_body=ASPIRIN_MOLECULE),
            FakeResponse(status_code=500),
            FakeResponse(status_code=500),
            FakeResponse(status_code=500),
        ]
        drug = get_drug("CHEMBL25")

        assert drug.name == "ASPIRIN"
        assert drug.mechanisms == ()


class TestSearchingChEMBL:
    def test_a_molecule_search_reaches_the_search_endpoint(self, web_transport) -> None:
        queue(web_transport, {"molecules": [], "page_meta": {"total_count": 0}})
        search_drugs("aspirin")

        assert "chembl/api/data/molecule/search" in web_transport.urls[0]
        assert web_transport.params[0]["q"] == "aspirin"

    def test_a_target_search_reaches_the_target_endpoint(self, web_transport) -> None:
        queue(web_transport, {"targets": [], "page_meta": {"total_count": 0}})
        search_targets("EGFR")

        assert "chembl/api/data/target/search" in web_transport.urls[0]

    def test_the_filters_that_make_a_target_search_useful_are_sent(self, web_transport) -> None:
        # Searching EGFR unfiltered puts two protein-protein interactions and mouse EGFR above
        # human EGFR, and the organism filter alone does not fix it
        queue(web_transport, {"targets": [], "page_meta": {"total_count": 0}})
        search_targets("EGFR", organism="Homo sapiens", single_proteins_only=True)

        assert web_transport.params[0]["organism__iexact"] == "Homo sapiens"
        assert web_transport.params[0]["target_type"] == "SINGLE PROTEIN"

    def test_the_filters_are_left_off_when_not_asked_for(self, web_transport) -> None:
        queue(web_transport, {"targets": [], "page_meta": {"total_count": 0}})
        search_targets("EGFR")

        assert "organism__iexact" not in web_transport.params[0]
        assert "target_type" not in web_transport.params[0]

    def test_a_search_is_held_to_what_is_worth_requesting(self, web_transport) -> None:
        queue(web_transport, {"molecules": [], "page_meta": {"total_count": 0}})
        search_drugs("aspirin", limit=500)

        assert web_transport.params[0]["limit"] == MAX_SEARCH_RESULTS

    def test_a_search_for_nothing_is_refused(self, web_transport) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            search_drugs("aspirin", limit=0)

        assert web_transport.requests == []

    def test_the_results_are_rendered(self, web_transport) -> None:
        queue(
            web_transport,
            {
                "molecules": [
                    {
                        "molecule_chembl_id": "CHEMBL25",
                        "pref_name": "ASPIRIN",
                        "max_phase": "4.0",
                        "molecule_type": "Small molecule",
                    }
                ],
                "page_meta": {"total_count": 52},
            },
        )
        report = search_drugs("aspirin", limit=1).report()

        assert "52 molecules" in report
        assert "CHEMBL25" in report
        assert "approved" in report

    def test_a_search_that_found_nothing_says_so(self) -> None:
        report = SearchHits(query="zzz", kind="molecules", total=0, hits=()).report()

        assert "No molecules in ChEMBL match" in report

    def test_the_organism_appears_in_the_report_without_stray_quotes(self) -> None:
        hits = SearchHits(
            query="EGFR in Homo sapiens",
            kind="targets",
            total=1,
            hits=(Target("CHEMBL203", "EGFR", "SINGLE PROTEIN", "Homo sapiens"),),
        )
        report = hits.report()

        assert '"EGFR in Homo sapiens"' in report
        assert '""' not in report


class TestLookingUpBioactivity:
    def test_a_target_is_required_or_a_molecule(self, web_transport) -> None:
        with pytest.raises(ValueError, match="target_chembl_id or a molecule_chembl_id"):
            get_activities()

        assert web_transport.requests == []

    def test_the_request_excludes_the_measurements_that_lost_their_units(
        self, web_transport
    ) -> None:
        # ChEMBL's standardisation sometimes drops a measurement's units, leaving a value wrong
        # by orders of magnitude: the most potent IC50 against EGFR by raw value reads
        # "5.012E-9 nM", whose original value was 17.3. Those rows have no pChEMBL value.
        queue(web_transport, {"activities": [], "page_meta": {"total_count": 0}})
        get_activities(target_chembl_id="CHEMBL203")

        assert web_transport.params[0]["pchembl_value__isnull"] == "false"
        assert web_transport.params[0]["order_by"] == "-pchembl_value"
        assert web_transport.params[0]["pchembl_value__gte"] == MIN_PCHEMBL_REPORTED

    def test_the_kind_of_measurement_is_passed_through(self, web_transport) -> None:
        queue(web_transport, {"activities": [], "page_meta": {"total_count": 0}})
        get_activities(target_chembl_id="CHEMBL203", activity_type="IC50")

        assert web_transport.params[0]["standard_type"] == "IC50"

    def test_the_count_is_held_to_what_is_worth_requesting(self, web_transport) -> None:
        queue(web_transport, {"activities": [], "page_meta": {"total_count": 0}})
        get_activities(target_chembl_id="CHEMBL203", limit=900)

        assert web_transport.params[0]["limit"] == MAX_ACTIVITIES_REPORTED

    def test_a_measurement_is_parsed(self) -> None:
        activity = activity_from(EGFR_ACTIVITY)

        assert activity.molecule_chembl_id == "CHEMBL176582"
        assert activity.value == pytest.approx(0.01)
        assert activity.pchembl == pytest.approx(11.0)
        assert activity.units == "nM"
        assert activity.year == 1996

    def test_html_entities_in_an_assay_description_are_decoded(self) -> None:
        # The assay descriptions ChEMBL imported from PubChem BioAssay carry HTML entities
        activity = activity_from(
            EGFR_ACTIVITY | {"assay_description": "Inhibitors of Bloom&apos;s syndrome helicase"}
        )

        assert activity.assay_description == "Inhibitors of Bloom's syndrome helicase"

    def test_the_report_explains_what_a_pchembl_value_is(self, web_transport) -> None:
        queue(
            web_transport,
            {"activities": [EGFR_ACTIVITY], "page_meta": {"total_count": 19378}},
        )
        report = get_activities(target_chembl_id="CHEMBL203").report()

        assert "19,378 measurement(s)" in report
        assert "9 is nanomolar" in report
        assert "not directly comparable" in report

    def test_a_long_assay_description_is_capped(self, web_transport) -> None:
        queue(
            web_transport,
            {
                "activities": [EGFR_ACTIVITY | {"assay_description": "y" * 900}],
                "page_meta": {"total_count": 1},
            },
        )
        report = get_activities(target_chembl_id="CHEMBL203").report()

        assert "700 of 900 characters not shown" in report
        assert "y" * MAX_ASSAY_DESCRIPTION_CHARACTERS in report
        assert "y" * (MAX_ASSAY_DESCRIPTION_CHARACTERS + 1) not in report

    def test_nothing_measured_says_so_rather_than_showing_an_empty_list(self) -> None:
        report = ActivityResults(subject="target CHEMBL1", total=0, activities=()).report()

        assert "no potency measurements" in report


class TestHowMeasurementsAreGrouped:
    """Asking about a target and grouping by that target would hide the answer."""

    def test_asking_about_a_target_groups_by_molecule(self, web_transport) -> None:
        queue(
            web_transport,
            {
                "activities": [
                    EGFR_ACTIVITY,
                    EGFR_ACTIVITY | {"molecule_chembl_id": "CHEMBL999", "pchembl_value": "10.0"},
                ],
                "page_meta": {"total_count": 2},
            },
        )
        results = get_activities(target_chembl_id="CHEMBL203")

        assert results.grouped_by == "molecule"
        assert [heading for heading, _ in results.groups()] == ["CHEMBL176582", "CHEMBL999"]

    def test_asking_about_a_molecule_groups_by_target(self, web_transport) -> None:
        queue(
            web_transport,
            {
                "activities": [
                    EGFR_ACTIVITY,
                    EGFR_ACTIVITY
                    | {"target_chembl_id": "CHEMBL999", "target_pref_name": "Other kinase"},
                ],
                "page_meta": {"total_count": 2},
            },
        )
        results = get_activities(molecule_chembl_id="CHEMBL25")

        assert results.grouped_by == "target"
        assert [heading for heading, _ in results.groups()] == [
            "Epidermal growth factor receptor (CHEMBL203) [Homo sapiens]",
            "Other kinase (CHEMBL999) [Homo sapiens]",
        ]

    def test_repeated_measurements_share_one_heading(self) -> None:
        # ChEMBL holds the same measurement from more than one document, and fifteen lines each
        # repeating a target name reads as fifteen findings when it is one
        twice = tuple(
            Activity(
                molecule_chembl_id="CHEMBL25",
                target_chembl_id="CHEMBL1293237",
                target_name="BLM helicase",
                measurement="Potency",
                value=2.8,
                pchembl=8.55,
            )
            for _ in range(2)
        )
        results = ActivityResults(subject="molecule CHEMBL25", total=2, activities=twice)

        assert len(results.groups()) == 1
        assert results.report().count("BLM helicase") == 1

    def test_the_strongest_result_still_leads(self) -> None:
        strong = Activity("CHEMBL1", target_name="A", target_chembl_id="CHEMBLA", pchembl=11.0)
        weak = Activity("CHEMBL2", target_name="B", target_chembl_id="CHEMBLB", pchembl=5.0)
        results = ActivityResults(
            subject="molecule X", total=2, activities=(strong, weak), grouped_by="target"
        )

        assert results.groups()[0][0].startswith("A ")


class TestWhatAReportCosts:
    def test_a_compound_report_stays_small(self) -> None:
        compound = compound_from(ASPIRIN_PROPERTIES, description="x" * 5_000, title="Aspirin")

        assert len(compound.report()) < 1_500

    def test_a_drug_report_stays_small(self) -> None:
        drug = molecule_from(
            ASPIRIN_MOLECULE,
            mechanisms=(("INHIBITOR", "Cyclooxygenase inhibitor", "CHEMBL2094253"),),
        )

        assert len(drug.report()) < 1_000

    def test_a_full_activity_report_stays_small(self) -> None:
        activities = tuple(
            activity_from(
                EGFR_ACTIVITY
                | {"molecule_chembl_id": f"CHEMBL{index}", "assay_description": "z" * 900}
            )
            for index in range(MAX_ACTIVITIES_REPORTED)
        )
        results = ActivityResults(
            subject="target CHEMBL203", total=19378, activities=activities, grouped_by="molecule"
        )

        assert len(results.report()) < 6_000


class TestAgainstTheRealDatabases:
    """Checks that the fixtures above still describe what the services send."""

    @live_only
    def test_pubchem_still_answers_a_name_lookup(self) -> None:
        compound = get_compound("aspirin")

        assert compound.cid == 2244
        assert compound.formula == "C9H8O4"
        assert compound.inchikey.startswith("BSYNRYMUTXBXSQ")

    @live_only
    def test_pubchem_still_uses_the_renamed_property_keys(self) -> None:
        # The whole point of the 2025 rename: if these come back empty the constant is stale
        compound = get_compound("morphine")

        assert compound.smiles
        assert compound.connectivity_smiles
        assert compound.smiles != compound.connectivity_smiles

    @live_only
    def test_pubchem_still_accepts_a_stereochemical_smiles(self) -> None:
        compound = get_compound(r"C/C=C\C(=O)O", namespace="smiles")

        assert compound.cid == 643792

    @live_only
    def test_pubchem_still_answers_an_unknown_structure_with_cid_zero(self) -> None:
        with pytest.raises(RecordNotFoundError):
            get_compound("CCCCCCCCCCCCCCCCCCCCCCCCCCN1C(=O)C2(CC2)Oc3ccccc31", namespace="smiles")

    @live_only
    def test_chembl_still_answers_a_molecule_lookup(self) -> None:
        drug = get_drug("CHEMBL25")

        assert drug.name == "ASPIRIN"
        assert drug.development_stage.startswith("approved")
        assert drug.mechanisms

    @live_only
    def test_chembl_still_ranks_the_obvious_target_first_when_filtered(self) -> None:
        hits = search_targets(
            "EGFR", organism="Homo sapiens", single_proteins_only=True, limit=5
        )

        assert hits.hits[0].chembl_id == "CHEMBL203"

    @live_only
    def test_chembl_still_returns_sane_potencies(self) -> None:
        # The guard against the rows whose units were lost: every value should be a real
        # affinity, not a number eight orders of magnitude below one
        results = get_activities(target_chembl_id="CHEMBL203", activity_type="IC50", limit=5)

        assert results.activities

        for activity in results.activities:
            assert activity.pchembl is not None
            assert MIN_PCHEMBL_REPORTED <= activity.pchembl < 15
            assert activity.value is None or activity.value > 1e-6

    @live_only
    def test_a_missing_compound_is_still_a_404(self) -> None:
        with pytest.raises(RecordNotFoundError):
            get_compound("notarealcompoundxyzzy")
