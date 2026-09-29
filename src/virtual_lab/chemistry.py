"""Lookups in the chemistry databases: compounds in PubChem, drugs and bioactivity in ChEMBL.

Structured the same way as the protein lookups: a function per question, a record holding what
the service returned, and a report() that renders only what is worth paying for in a request.

Both services have the habit of answering a question that was not asked. PubChem accepts a
structure it has never seen and returns CID 0 rather than an error, ChEMBL returns its numbers as
strings, and an activity whose units were lost in standardization keeps a value that is off by
eight orders of magnitude. Each of those arrives as a successful response, so each is checked
here rather than passed on to a model as fact.
"""

import html
import math
from dataclasses import dataclass, field
from typing import Any

from virtual_lab.constants import (
    MAX_ACTIVITIES_REPORTED,
    MAX_ASSAY_DESCRIPTION_CHARACTERS,
    MAX_DESCRIPTION_CHARACTERS,
    MAX_MECHANISMS_REPORTED,
    MAX_SEARCH_RESULTS,
    MAX_STRUCTURE_CHARACTERS,
    MIN_PCHEMBL_REPORTED,
)
from virtual_lab.records import (
    RecordNotFoundError,
    as_dict,
    as_int,
    as_list,
    as_text,
    bounded,
    number_or_none,
    truncate_text,
)
from virtual_lab.web import WebRequestError, build_url, request_json

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Deliberately narrower than the shared set, which includes 400 because UniProt answers an
# unknown accession with it. Both services here answer an identifier they do not hold with 404
# and keep 400 for a request they could not parse. Accepting 400 as "no such record" would report
# a compound as nonexistent whenever this library built a URL wrongly, which is a confident wrong
# answer rather than a visible failure, and the SMILES encoding below is exactly that case.
NOT_FOUND_STATUSES = frozenset({404})

# How a compound may be named. Anything else is a typo or an invented namespace, and passing it
# through would produce a URL that means something other than what the caller asked for.
COMPOUND_NAMESPACES = ("name", "cid", "smiles", "inchikey")

# Namespaces whose value cannot go in the path. A SMILES carries "/" and "\" for stereochemistry;
# percent-encoded in a path segment PubChem answers 400, and the same string as a query parameter
# it answers correctly, stereo bonds and all.
QUERY_NAMESPACES = frozenset({"smiles"})

# Properties to ask PubChem for. The names matter more than they look: PubChem renamed these in
# 2025, so a request for CanonicalSMILES is answered with a key called ConnectivitySMILES and
# code reading back what it asked for raises KeyError on a perfectly good response. SMILES is now
# the one with stereochemistry and ConnectivitySMILES the one without.
COMPOUND_PROPERTIES = (
    "MolecularFormula",
    "MolecularWeight",
    "SMILES",
    "ConnectivitySMILES",
    "InChIKey",
    "IUPACName",
    "XLogP",
    "TPSA",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
)

# Fields to ask ChEMBL for. Without this the service sends the whole record, which includes an
# embedded molfile and runs to 11,500 characters for aspirin and 18,000 for three search hits.
MOLECULE_FIELDS = (
    "molecule_chembl_id",
    "pref_name",
    "max_phase",
    "molecule_type",
    "first_approval",
    "withdrawn_flag",
    "oral",
    "parenteral",
    "topical",
    "molecule_properties",
    "molecule_structures",
)

MOLECULE_SEARCH_FIELDS = (
    "molecule_chembl_id",
    "pref_name",
    "max_phase",
    "molecule_type",
)

TARGET_FIELDS = (
    "target_chembl_id",
    "pref_name",
    "target_type",
    "organism",
)

ACTIVITY_FIELDS = (
    "molecule_chembl_id",
    "target_chembl_id",
    "target_pref_name",
    "target_organism",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "assay_description",
    "document_year",
)


def descriptor(value: Any) -> Any:
    """Reads one computed property, which is shown as it was sent but only if it is short.

    :param value: The value as the service sent it, a number from PubChem or a string of one
        from ChEMBL.
    :return: A finite number, a shortened string, or None for anything else.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(value) else None

    return as_text(value) or None


def tidy(value: Any) -> str:
    """Renders a field that may be absent, without writing "None" into a report."""
    return "not stated" if value is None or value == "" else str(value)


@dataclass(frozen=True)
class Compound:
    """A compound as PubChem holds it.

    :param cid: The PubChem compound identifier.
    :param title: The name PubChem shows for it, which is the common name where there is one.
    :param formula: The molecular formula.
    :param weight: The molecular weight, in daltons.
    :param smiles: The SMILES with stereochemistry.
    :param connectivity_smiles: The SMILES without stereochemistry.
    :param inchikey: The InChIKey, which is the identifier to match against other databases.
    :param iupac_name: The systematic name.
    :param description: What PubChem says the compound is.
    :param properties: The computed descriptors, keyed by PubChem's name for each.
    """

    cid: int
    title: str = ""
    formula: str = ""
    weight: float | None = None
    smiles: str = ""
    connectivity_smiles: str = ""
    inchikey: str = ""
    iupac_name: str = ""
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        """Where a person can read the full record."""
        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{self.cid}"

    def report(self) -> str:
        """Renders the compound for a model."""
        # Capped here too, not only on the line below: with no curated title this falls back to
        # the systematic name, which for anything peptide-sized is the length of a paragraph
        named = truncate_text(self.title or self.iupac_name or "unnamed", MAX_STRUCTURE_CHARACTERS)
        lines = [f"PubChem CID {self.cid}: {named}"]

        if self.iupac_name and self.iupac_name != self.title:
            lines.append(f"IUPAC name: {truncate_text(self.iupac_name, MAX_STRUCTURE_CHARACTERS)}")

        lines.append(f"Formula: {tidy(self.formula)}")

        if self.weight is not None:
            lines.append(f"Molecular weight: {self.weight:.2f}")

        if self.smiles:
            lines.append(f"SMILES: {truncate_text(self.smiles, MAX_STRUCTURE_CHARACTERS)}")

        # Only worth the line when it differs, which is when the compound has stereochemistry
        if self.connectivity_smiles and self.connectivity_smiles != self.smiles:
            lines.append(
                "SMILES without stereochemistry: "
                f"{truncate_text(self.connectivity_smiles, MAX_STRUCTURE_CHARACTERS)}"
            )

        if self.inchikey:
            lines.append(f"InChIKey: {self.inchikey}")

        descriptors = [
            f"{label}: {self.properties[key]}"
            for key, label in (
                ("XLogP", "XLogP"),
                ("TPSA", "TPSA"),
                ("HBondDonorCount", "H-bond donors"),
                ("HBondAcceptorCount", "H-bond acceptors"),
                ("RotatableBondCount", "Rotatable bonds"),
                ("HeavyAtomCount", "Heavy atoms"),
                ("Charge", "Formal charge"),
            )
            if self.properties.get(key) is not None
        ]

        if descriptors:
            lines.append(f"\n{', '.join(descriptors)}")

        if self.description:
            lines.append(f"\n{truncate_text(self.description, MAX_DESCRIPTION_CHARACTERS)}")

        lines.append(f"\n{self.url}")

        return "\n".join(lines)


@dataclass(frozen=True)
class Drug:
    """A molecule as ChEMBL holds it, which is a compound plus what is known about its use.

    :param chembl_id: The ChEMBL identifier.
    :param name: The preferred name, which for an approved drug is the generic name.
    :param max_phase: The furthest clinical phase reached, 4 meaning approved.
    :param molecule_type: Small molecule, antibody, protein, and so on.
    :param first_approval: The year of first approval, if it was approved.
    :param withdrawn: Whether it was withdrawn from a market.
    :param routes: The routes of administration recorded for it.
    :param smiles: The canonical SMILES.
    :param inchikey: The InChIKey.
    :param properties: The computed descriptors, keyed by ChEMBL's name for each.
    :param mechanisms: What it acts on and how, as action, description, and target.
    """

    chembl_id: str
    name: str = ""
    max_phase: float | None = None
    molecule_type: str = ""
    first_approval: int | None = None
    withdrawn: bool = False
    routes: tuple[str, ...] = ()
    smiles: str = ""
    inchikey: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    mechanisms: tuple[tuple[str, str, str], ...] = ()

    @property
    def development_stage(self) -> str:
        """Describes how far the molecule got, rather than leaving a model to read a number.

        A model that reads "max phase 4" as "phase 4 trial ongoing" has it exactly backwards, and
        the difference between an approved drug and an abandoned one decides whether it is worth
        proposing.
        """
        # ChEMBL uses -1 for a molecule whose clinical status it does not know, which is 2,499 of
        # them including camptothecin. Rendering that as "reached clinical phase -1" reads as a
        # parsing failure, and comparing it as a number ranks it below a wholly untested compound.
        if self.max_phase is None or self.max_phase < 0:
            return "clinical status not recorded"

        if self.max_phase == 0:
            return "no clinical development recorded"

        if self.max_phase >= 4:
            approved = f" (first approved {self.first_approval})" if self.first_approval else ""
            return f"approved{approved}"

        return f"reached clinical phase {self.max_phase:g}"

    @property
    def url(self) -> str:
        """Where a person can read the full record."""
        return f"https://www.ebi.ac.uk/chembl/explore/compound/{self.chembl_id}"

    def report(self) -> str:
        """Renders the drug for a model."""
        lines = [
            f"ChEMBL {self.chembl_id}: {self.name or 'unnamed'}",
            f"Type: {tidy(self.molecule_type)}",
            f"Development: {self.development_stage}",
        ]

        if self.withdrawn:
            lines.append("Withdrawn from at least one market.")

        if self.routes:
            lines.append(f"Routes: {', '.join(self.routes)}")

        if self.smiles:
            lines.append(f"SMILES: {truncate_text(self.smiles, MAX_STRUCTURE_CHARACTERS)}")

        if self.inchikey:
            lines.append(f"InChIKey: {self.inchikey}")

        descriptors = [
            f"{label}: {self.properties[key]}"
            for key, label in (
                ("full_mwt", "Molecular weight"),
                ("alogp", "ALogP"),
                ("hbd", "H-bond donors"),
                ("hba", "H-bond acceptors"),
                ("psa", "TPSA"),
                ("num_ro5_violations", "Rule-of-five violations"),
                ("qed_weighted", "QED"),
            )
            if self.properties.get(key) is not None
        ]

        if descriptors:
            lines.append(f"\n{', '.join(descriptors)}")

        if self.mechanisms:
            shown = self.mechanisms[:MAX_MECHANISMS_REPORTED]
            lines.append("\nMechanism(s) of action:")
            lines.extend(
                f"  {action.lower() if action else 'acts on'} {description}"
                f"{f' ({target})' if target else ''}"
                for action, description, target in shown
            )

            if len(self.mechanisms) > len(shown):
                lines.append(f"  ... and {len(self.mechanisms) - len(shown)} more")

        lines.append(f"\n{self.url}")

        return "\n".join(lines)


@dataclass(frozen=True)
class DrugHit:
    """One result of a ChEMBL molecule search."""

    chembl_id: str
    name: str = ""
    max_phase: float | None = None
    molecule_type: str = ""

    def summary(self) -> str:
        """Renders the hit as one line."""
        if self.max_phase is None or self.max_phase < 0:
            stage = "clinical status unknown"
        elif self.max_phase >= 4:
            stage = "approved"
        elif self.max_phase == 0:
            stage = "not in development"
        else:
            stage = f"phase {self.max_phase:g}"

        return f"  {self.chembl_id}  {self.name or 'unnamed'} ({tidy(self.molecule_type)}, {stage})"


@dataclass(frozen=True)
class Target:
    """One result of a ChEMBL target search."""

    chembl_id: str
    name: str = ""
    target_type: str = ""
    organism: str = ""

    def summary(self) -> str:
        """Renders the target as one line."""
        return (
            f"  {self.chembl_id}  {self.name or 'unnamed'} "
            f"[{tidy(self.target_type)}, {tidy(self.organism)}]"
        )


@dataclass(frozen=True)
class Activity:
    """One measured interaction between a molecule and a target.

    :param molecule_chembl_id: The molecule measured.
    :param target_chembl_id: The target it was measured against.
    :param target_name: The target's name.
    :param organism: The organism the target came from.
    :param measurement: What was measured, such as IC50 or Ki.
    :param relation: Whether the value is exact, an upper bound, or a lower bound.
    :param value: The standardised value.
    :param units: The units of the standardised value.
    :param pchembl: The negative log of the molar value, which is comparable across measurements.
    :param assay_id: The assay it came from.
    :param assay_description: What the assay did.
    :param year: The year of the document it came from.
    """

    molecule_chembl_id: str
    target_chembl_id: str = ""
    target_name: str = ""
    organism: str = ""
    measurement: str = ""
    relation: str = "="
    value: float | None = None
    units: str = ""
    pchembl: float | None = None
    assay_id: str = ""
    assay_description: str = ""
    year: int | None = None

    def summary(self) -> str:
        """Renders the measurement as one line."""
        # The units joined to the number rather than appended after it: ChEMBL leaves them off a
        # ratio or a percentage, and the spare separator reads as a dropped field
        amount = "?" if self.value is None else f"{self.value:g}"
        measured = f"{amount} {self.units}" if self.units else amount
        strength = f", pChEMBL {self.pchembl:.1f}" if self.pchembl is not None else ""
        when = f", {self.year}" if self.year else ""

        return f"  {tidy(self.measurement)} {self.relation} {measured}{strength}{when}"


@dataclass(frozen=True)
class SearchHits:
    """What a ChEMBL search found.

    :param query: What was searched for.
    :param kind: What was being searched for, used in the report's first line.
    :param total: How many results the service said there were, which it rounds.
    :param hits: The results kept, already limited.
    """

    query: str
    kind: str
    total: int
    hits: tuple[Any, ...]

    def report(self) -> str:
        """Renders the results for a model."""
        if not self.hits:
            return f'No {self.kind} in ChEMBL match "{self.query}".'

        header = (
            f'{self.total:,} {self.kind} match "{self.query}", showing {len(self.hits)}:'
            if self.total > len(self.hits)
            else f'{len(self.hits)} {self.kind} match "{self.query}":'
        )

        return "\n".join([header, *(hit.summary() for hit in self.hits)])


@dataclass(frozen=True)
class ActivityResults:
    """The measurements found for a bioactivity question.

    :param subject: What was asked about, for the report's first line.
    :param total: How many measurements ChEMBL holds that match, before the cap.
    :param activities: The measurements kept, most potent first.
    :param grouped_by: Whether the lines are grouped by target or by molecule. This is always
        whichever of the two the caller did not pin: asking what is active against a target and
        then grouping by that target yields one heading and hides the molecules, which are the
        answer to the question.
    """

    subject: str
    total: int
    activities: tuple[Activity, ...]
    grouped_by: str = "target"

    def groups(self) -> tuple[tuple[str, tuple[Activity, ...]], ...]:
        """Gathers the measurements under what they were measured against.

        Order is the order the measurements arrived in, which is most potent first, so the
        strongest result still leads.

        :return: Each heading with its measurements.
        """
        collected: dict[str, list[Activity]] = {}

        for activity in self.activities:
            if self.grouped_by == "target":
                name = activity.target_name or activity.target_chembl_id or "unnamed target"
                organism = f" [{activity.organism}]" if activity.organism else ""
                # Not repeated when the name is already the identifier, which is what an
                # unnamed target falls back to and would render as "CHEMBL203 (CHEMBL203)"
                identifier = (
                    f" ({activity.target_chembl_id})"
                    if activity.target_chembl_id and activity.target_chembl_id != name
                    else ""
                )
                heading = f"{name}{identifier}{organism}"
            else:
                heading = activity.molecule_chembl_id or "unnamed molecule"

            collected.setdefault(heading, []).append(activity)

        return tuple((heading, tuple(found)) for heading, found in collected.items())

    def report(self) -> str:
        """Renders the measurements for a model, grouped so the pattern is visible."""
        if not self.activities:
            return (
                f"ChEMBL holds no potency measurements for {self.subject} "
                f"above pChEMBL {MIN_PCHEMBL_REPORTED:g}."
            )

        lines = [
            f"{self.total:,} measurement(s) recorded for {self.subject}, "
            f"showing the {len(self.activities)} most potent:"
        ]

        # Grouped because the interesting thing about fifteen measurements is how many distinct
        # molecules or targets they cover. Fifteen rows each repeating the same target name reads
        # as fifteen findings when it is one, and costs the tokens to say so.
        for heading, group in self.groups():
            lines.append(f"\n{heading}")

            for activity in group:
                lines.append(activity.summary())

                if activity.assay_description:
                    lines.append(
                        f"    {truncate_text(activity.assay_description, MAX_ASSAY_DESCRIPTION_CHARACTERS)}"
                    )

        lines.append(
            "\nA pChEMBL value is the negative log of the molar potency, so 9 is nanomolar and "
            "6 is micromolar. Measurements from different assays are not directly comparable."
        )

        return "\n".join(lines)


def compound_from(properties: dict[str, Any], description: str = "", title: str = "") -> Compound:
    """Builds a compound record from a PubChem property table row.

    :param properties: One entry of the property table.
    :param description: What PubChem says the compound is.
    :param title: The name PubChem shows for it.
    :return: The compound.
    """
    # Read through as_text and as_int rather than str() and int(), throughout: a key PubChem sends
    # as JSON null is present, so a default never applies and str() would write the word "None"
    # into a report as though it were the compound's formula. A null CID made int() raise.
    return Compound(
        cid=as_int(properties.get("CID"), 0) or 0,
        title=title,
        formula=as_text(properties.get("MolecularFormula")),
        weight=number_or_none(properties.get("MolecularWeight")),
        smiles=as_text(properties.get("SMILES"), limit=None),
        connectivity_smiles=as_text(properties.get("ConnectivitySMILES"), limit=None),
        inchikey=as_text(properties.get("InChIKey")),
        iupac_name=as_text(properties.get("IUPACName"), limit=None),
        description=description,
        properties={
            key: descriptor(properties[key]) for key in COMPOUND_PROPERTIES if key in properties
        },
    )


def describe_compound(cid: int) -> tuple[str, str]:
    """Fetches what PubChem says a compound is.

    Kept apart from the property lookup because it is a second request for something a caller may
    not need, and because a compound with no curated description is not a failed lookup.

    :param cid: The compound identifier.
    :return: The title and the description, either of which may be empty.
    """
    url = build_url(f"{PUBCHEM_BASE}/cid/{{cid}}/description/JSON", cid=str(cid))

    try:
        response = request_json(url)
    except WebRequestError:
        return "", ""

    entries = [
        entry
        for entry in as_list(as_dict(as_dict(response).get("InformationList")).get("Information"))
        if isinstance(entry, dict)
    ]

    # Kept whole, like the description below, because the report shortens it
    title = next(
        (as_text(entry["Title"], limit=None) for entry in entries if entry.get("Title")), ""
    )

    # The longest rather than the first. PubChem returns descriptions from several sources in no
    # useful order, and for aspirin the first is a one-line Proposition 65 hazard notice while
    # the one that says what the molecule is comes second. Length separates a definition from a
    # regulatory note reliably enough, and the alternative is ranking source names by hand.
    descriptions = [
        as_text(entry["Description"], limit=None) for entry in entries if entry.get("Description")
    ]

    return title, max(descriptions, key=len, default="")


def get_compound(identifier: str, namespace: str = "name") -> Compound:
    """Looks a compound up in PubChem.

    :param identifier: The name, CID, SMILES, or InChIKey.
    :param namespace: Which of those the identifier is.
    :raises ValueError: If the namespace is not one PubChem accepts.
    :raises RecordNotFoundError: If PubChem holds no such compound.
    :raises WebRequestError: If PubChem could not be reached.
    :return: The compound record.
    """
    if namespace not in COMPOUND_NAMESPACES:
        raise ValueError(
            f'Unknown namespace "{namespace}". Use one of: {", ".join(COMPOUND_NAMESPACES)}.'
        )

    # Checked here rather than left to encode_segment, which only sees the namespaces whose value
    # goes in the path. A blank SMILES would otherwise be sent as a parameter and asked about.
    if not identifier.strip():
        raise ValueError(f"Give a {namespace} to look up")

    wanted = ",".join(COMPOUND_PROPERTIES)

    if namespace in QUERY_NAMESPACES:
        url = f"{PUBCHEM_BASE}/{namespace}/property/{wanted}/JSON"
        params: dict[str, Any] | None = {namespace: identifier}
    else:
        url = build_url(
            f"{PUBCHEM_BASE}/{namespace}/{{identifier}}/property/{wanted}/JSON",
            identifier=identifier,
        )
        params = None

    try:
        response = request_json(url, params=params)
    except WebRequestError as error:
        if error.status_code not in NOT_FOUND_STATUSES:
            raise

        raise RecordNotFoundError(
            f'PubChem has no compound for {namespace} "{identifier}".'
        ) from error

    rows = as_list(as_dict(as_dict(response).get("PropertyTable")).get("Properties"))

    if not rows:
        raise RecordNotFoundError(f'PubChem returned no properties for {namespace} "{identifier}".')

    row = as_dict(rows[0])
    cid = as_int(row.get("CID"), 0) or 0

    # PubChem answers a structure it has never seen with 200 and CID 0 rather than an error, so
    # the only sign that nothing was found is the identifier itself
    if cid == 0:
        raise RecordNotFoundError(
            f'PubChem has no compound matching {namespace} "{identifier}". It parsed the input '
            f"but holds no such structure."
        )

    title, description = describe_compound(cid)

    return compound_from(row, description=description, title=title)


def molecule_from(record: dict[str, Any], mechanisms: tuple[tuple[str, str, str], ...]) -> Drug:
    """Builds a drug record from a ChEMBL molecule record."""
    structures = as_dict(record.get("molecule_structures"))
    routes = tuple(
        label
        for key, label in (("oral", "oral"), ("parenteral", "parenteral"), ("topical", "topical"))
        if record.get(key)
    )

    return Drug(
        chembl_id=as_text(record.get("molecule_chembl_id")),
        name=as_text(record.get("pref_name")),
        max_phase=number_or_none(record.get("max_phase")),
        molecule_type=as_text(record.get("molecule_type")),
        first_approval=as_int(record.get("first_approval")),
        withdrawn=bool(record.get("withdrawn_flag")),
        routes=routes,
        smiles=as_text(structures.get("canonical_smiles"), limit=None),
        inchikey=as_text(structures.get("standard_inchi_key")),
        properties={
            key: descriptor(value)
            for key, value in as_dict(record.get("molecule_properties")).items()
        },
        mechanisms=mechanisms,
    )


def mechanisms_of(chembl_id: str) -> tuple[tuple[str, str, str], ...]:
    """Fetches how a molecule acts, which is a separate endpoint from the molecule itself.

    :param chembl_id: The molecule to ask about.
    :raises WebRequestError: If the endpoint could not be reached.
    :return: Each mechanism as action, description, and target; empty if none are recorded.
    """
    try:
        response = request_json(
            f"{CHEMBL_BASE}/mechanism",
            params={
                "molecule_chembl_id": chembl_id,
                "format": "json",
                "only": "action_type,mechanism_of_action,target_chembl_id",
            },
        )
    except WebRequestError as error:
        # Only a refusal is an answer. Most molecules have no recorded mechanism, so an empty
        # result is ordinary and not worth failing a lookup over; returning that same empty
        # result when the endpoint could not be reached tells an agent that a drug whose
        # mechanism is well known acts on nothing, which it will not go back and check.
        if error.status_code in NOT_FOUND_STATUSES:
            return ()

        raise

    return tuple(
        (
            as_text(entry.get("action_type")),
            as_text(entry.get("mechanism_of_action")),
            as_text(entry.get("target_chembl_id")),
        )
        for entry in as_list(as_dict(response).get("mechanisms"))
        if isinstance(entry, dict)
    )


def get_drug(chembl_id: str) -> Drug:
    """Looks a molecule up in ChEMBL.

    :param chembl_id: A ChEMBL identifier such as CHEMBL25.
    :raises RecordNotFoundError: If ChEMBL holds no such molecule.
    :raises WebRequestError: If ChEMBL could not be reached.
    :return: The drug record.
    """
    identifier = chembl_id.strip().upper()
    url = build_url(f"{CHEMBL_BASE}/molecule/{{chembl_id}}.json", chembl_id=identifier)

    try:
        record = request_json(url, params={"only": ",".join(MOLECULE_FIELDS)})
    except WebRequestError as error:
        if error.status_code not in NOT_FOUND_STATUSES:
            raise

        raise RecordNotFoundError(f'ChEMBL has no molecule "{identifier}".') from error

    if not isinstance(record, dict) or not record.get("molecule_chembl_id"):
        raise RecordNotFoundError(f'ChEMBL returned no molecule for "{identifier}".')

    return molecule_from(record, mechanisms=mechanisms_of(identifier))


def search_drugs(query: str, limit: int = 10) -> SearchHits:
    """Searches ChEMBL for molecules by name.

    :param query: A drug or compound name.
    :param limit: How many results to return.
    :raises ValueError: If the limit is below one.
    :raises WebRequestError: If ChEMBL could not be reached.
    :return: The results.
    """
    size = bounded(limit, MAX_SEARCH_RESULTS, "limit")

    response = request_json(
        f"{CHEMBL_BASE}/molecule/search",
        params={
            "q": query,
            "format": "json",
            "limit": size,
            "only": ",".join(MOLECULE_SEARCH_FIELDS),
        },
    )

    records = [as_dict(record) for record in as_list(as_dict(response).get("molecules"))[:size]]
    hits = tuple(
        DrugHit(
            chembl_id=as_text(record.get("molecule_chembl_id")),
            name=as_text(record.get("pref_name")),
            max_phase=number_or_none(record.get("max_phase")),
            molecule_type=as_text(record.get("molecule_type")),
        )
        for record in records
    )

    return SearchHits(
        query=query,
        kind="molecules",
        total=total_count(response) or len(hits),
        hits=hits,
    )


def search_targets(
    query: str,
    organism: str = "",
    single_proteins_only: bool = False,
    limit: int = 10,
) -> SearchHits:
    """Searches ChEMBL for targets by name or gene symbol.

    Free text ranking is unhelpful here in a specific way. Searching for EGFR returns two
    protein-protein interaction entries and mouse EGFR above human EGFR, and restricting to
    Homo sapiens still leaves the interactions on top. Restricting to single proteins as well
    puts the obvious answer first, which is why both filters exist.

    :param query: A target name or gene symbol, such as EGFR.
    :param organism: Restrict to one organism, such as "Homo sapiens".
    :param single_proteins_only: Exclude complexes, cell lines, and protein-protein interactions.
    :param limit: How many results to return.
    :raises ValueError: If the limit is below one.
    :raises WebRequestError: If ChEMBL could not be reached.
    :return: The results.
    """
    size = bounded(limit, MAX_SEARCH_RESULTS, "limit")

    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "limit": size,
        "only": ",".join(TARGET_FIELDS),
    }

    if organism:
        params["organism__iexact"] = organism

    if single_proteins_only:
        params["target_type"] = "SINGLE PROTEIN"

    response = request_json(f"{CHEMBL_BASE}/target/search", params=params)

    records = [as_dict(record) for record in as_list(as_dict(response).get("targets"))[:size]]
    hits = tuple(
        Target(
            chembl_id=as_text(record.get("target_chembl_id")),
            name=as_text(record.get("pref_name")),
            target_type=as_text(record.get("target_type")),
            organism=as_text(record.get("organism")),
        )
        for record in records
    )

    return SearchHits(
        query=f"{query} in {organism}" if organism else query,
        kind="targets",
        total=total_count(response) or len(hits),
        hits=hits,
    )


def total_count(response: Any) -> int:
    """Reads how many results ChEMBL says a search has, or zero if it does not say."""
    return as_int(as_dict(as_dict(response).get("page_meta")).get("total_count"), 0) or 0


def activity_from(record: dict[str, Any]) -> Activity:
    """Builds a measurement from a ChEMBL activity record."""
    return Activity(
        molecule_chembl_id=as_text(record.get("molecule_chembl_id")),
        target_chembl_id=as_text(record.get("target_chembl_id")),
        target_name=as_text(record.get("target_pref_name")),
        organism=as_text(record.get("target_organism")),
        measurement=as_text(record.get("standard_type")),
        relation=as_text(record.get("standard_relation")) or "=",
        value=number_or_none(record.get("standard_value")),
        units=as_text(record.get("standard_units")),
        pchembl=number_or_none(record.get("pchembl_value")),
        assay_id=as_text(record.get("assay_chembl_id")),
        # Unescaped because the assay descriptions ChEMBL imported from PubChem BioAssay carry
        # HTML entities, so an agent is otherwise told about "Bloom&apos;s syndrome helicase"
        assay_description=html.unescape(as_text(record.get("assay_description"), limit=None)),
        year=as_int(record.get("document_year")),
    )


def get_activities(
    target_chembl_id: str = "",
    molecule_chembl_id: str = "",
    activity_type: str = "",
    limit: int = MAX_ACTIVITIES_REPORTED,
) -> ActivityResults:
    """Finds the most potent measured interactions for a target or a molecule.

    Only measurements carrying a pChEMBL value are requested, and they are ordered by it. That is
    not a detail. ChEMBL's standardisation sometimes loses a measurement's units, leaving a value
    that is wrong by orders of magnitude: ordering the IC50 values against EGFR by raw value puts
    a row reading "5.012E-9 nM" first, which would be a hundred million times tighter than any
    real binding, and whose original value was 17.3 in units the record no longer names. Those
    rows are exactly the ones with no pChEMBL value, so requiring one both fixes the ordering and
    drops the corrupted rows.

    :param target_chembl_id: The target to look up activity against.
    :param molecule_chembl_id: The molecule to look up activity for. One of the two is required.
    :param activity_type: Restrict to one kind of measurement, such as IC50, Ki, or EC50.
    :param limit: How many measurements to return.
    :raises ValueError: If neither identifier is given, or the limit is below one.
    :raises WebRequestError: If ChEMBL could not be reached.
    :return: The measurements, most potent first.
    """
    # Stripped before the check, not after it. An identifier of one space is not falsy, so it
    # would pass the check and then be sent as the empty string, which ChEMBL treats as no filter
    # at all and answers with the most potent measurements in the database.
    target_chembl_id = target_chembl_id.strip().upper()
    molecule_chembl_id = molecule_chembl_id.strip().upper()

    if not target_chembl_id and not molecule_chembl_id:
        raise ValueError("Give either a target_chembl_id or a molecule_chembl_id")

    size = bounded(limit, MAX_ACTIVITIES_REPORTED, "limit")

    params: dict[str, Any] = {
        "format": "json",
        "limit": size,
        "order_by": "-pchembl_value",
        "pchembl_value__isnull": "false",
        "pchembl_value__gte": MIN_PCHEMBL_REPORTED,
        "only": ",".join(ACTIVITY_FIELDS),
    }

    if target_chembl_id:
        params["target_chembl_id"] = target_chembl_id

    if molecule_chembl_id:
        params["molecule_chembl_id"] = molecule_chembl_id

    if activity_type:
        params["standard_type"] = activity_type.strip()

    response = request_json(f"{CHEMBL_BASE}/activity", params=params)

    records = as_list(as_dict(response).get("activities"))[:size]
    activities = tuple(activity_from(as_dict(record)) for record in records)

    if target_chembl_id and molecule_chembl_id:
        subject = f"{molecule_chembl_id} against {target_chembl_id}"
    elif target_chembl_id:
        subject = f"target {target_chembl_id}"
    else:
        subject = f"molecule {molecule_chembl_id}"

    if activity_type:
        subject = f"{subject} ({activity_type})"

    return ActivityResults(
        subject=subject,
        total=total_count(response) or len(activities),
        activities=activities,
        grouped_by="molecule" if target_chembl_id else "target",
    )
