"""What a lab has found, kept as findings that later work can be given.

A meeting's summary is written for the meeting after it. Passed on whole, the summaries of every
earlier meeting soon fill the context, and most of what is in them does not bear on the meeting
at hand. LabMemory keeps what the work established instead, as findings: each one claim, with the
evidence for it and the step it came from. A meeting is then given the findings it needs, chosen
either by whoever plans the work, from a list of every finding by its claim, as Biomni's agent
chooses the resources a task needs, or by how well each matches the meeting's agenda, by BM25.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from virtual_lab.utils import write_atomically

# BM25's usual constants: how quickly a term's weight saturates as it repeats, and how much a long
# finding is discounted against a short one
BM25_K1 = 1.5
BM25_B = 0.75

# Words too common to say what a finding is about
STOPWORDS = frozenset(
    "a about after all also an and any are as at be been but by can could did do does for from had has have "
    "how if in into is it its may might more most no not of on or our over so such than that the their them "
    "then there these they this those to under was we were what when where which while who why will with "
    "would".split()
)


class Finding(BaseModel):
    """One result that the work established."""

    claim: str = Field(
        description="One result the work established, stated so that it stands on its own, with the names, "
        "numbers, and conditions it depends on."
    )
    evidence: str = Field(
        description="What in the work shows it, such as the data, computation, source, or argument, and how "
        "strong that is."
    )


class Findings(BaseModel):
    """What a piece of work established."""

    findings: list[Finding] = Field(
        description="Each result the work established, as a finding of its own. Leave out what was only proposed "
        "or planned, and say in the claim when a result is uncertain. Empty if the work established nothing."
    )


@dataclass(frozen=True)
class MemoryEntry:
    """A finding as a lab's memory keeps it.

    :param id: Its id, F1, F2, and so on, in the order the findings were made.
    :param claim: What was found.
    :param evidence: What shows it.
    :param source: The step that found it, by name.
    :param round: The round of the project it was found in, if it was found in one.
    """

    id: str
    claim: str
    evidence: str
    source: str
    round: int | None = None

    def describe(self) -> str:
        """The finding in full, as a meeting is given it."""
        found = f"round {self.round}, {self.source}" if self.round is not None else self.source
        return f"[{self.id}] {self.claim}\n\nEvidence: {self.evidence}\n\n(Found in {found}.)"


def finding_key(finding_id: str) -> str:
    """An id as it is compared: without regard to case or surrounding space, as a model may write it."""
    return finding_id.strip().upper()


def tokenize(text: str) -> list[str]:
    """Splits text into the words BM25 matches, lower-cased, without stopwords.

    A word joined by a dot or hyphen, such as KP.3 or SARS-CoV-2, is kept whole as well as in its
    parts, so that it matches itself more strongly than it matches anything sharing one part.
    """
    words = []
    for word in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower()):
        parts = re.split(r"[.\-]", word)
        words += [word, *parts] if len(parts) > 1 else [word]

    return [word for word in words if word not in STOPWORDS]


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """How well each document matches the query, by Okapi BM25.

    The inverse document frequency is BM25+'s, which stays positive, so a term found in most
    documents still counts for a little rather than counting against them.
    """
    tokenized = [tokenize(document) for document in documents]
    if not tokenized:
        return []

    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    containing = Counter(term for tokens in tokenized for term in set(tokens))
    terms = set(tokenize(query))

    scores = []
    for tokens in tokenized:
        counts = Counter(tokens)
        score = 0.0
        for term in terms:
            if not (frequency := counts[term]):
                continue
            idf = math.log(1 + (len(tokenized) - containing[term] + 0.5) / (containing[term] + 0.5))
            length = 1 - BM25_B + BM25_B * len(tokens) / average_length
            score += idf * frequency * (BM25_K1 + 1) / (frequency + BM25_K1 * length)
        scores.append(score)

    return scores


class LabMemory:
    """The findings a lab has made, in the order it made them.

    :param entries: Findings kept already, such as those another project's memory saved.
    """

    def __init__(self, entries: Iterable[MemoryEntry] = ()) -> None:
        self._entries: list[MemoryEntry] = []
        for entry in entries:
            if finding_key(entry.id) in self._ids():
                raise ValueError(f"Two findings have the id {entry.id}")
            self._entries.append(entry)

    def _ids(self) -> set[str]:
        return {finding_key(entry.id) for entry in self._entries}

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[MemoryEntry]:
        return iter(tuple(self._entries))

    def add(self, findings: Iterable[Finding], source: str, round: int | None = None) -> list[MemoryEntry]:
        """Keeps findings, each under the next id; one with no claim is left out.

        :param findings: What was found.
        :param source: The step that found them, by name.
        :param round: The round of the project they were found in, if any.
        :return: What was kept.
        """
        added = []
        for finding in findings:
            if not finding.claim.strip():
                continue
            entry = MemoryEntry(
                id=self._next_id(),
                claim=finding.claim.strip(),
                evidence=finding.evidence.strip(),
                source=source,
                round=round,
            )
            self._entries.append(entry)
            added.append(entry)

        return added

    def _next_id(self) -> str:
        numbers = [int(key[1:]) for key in self._ids() if re.fullmatch(r"F\d+", key)]
        return f"F{max(numbers, default=0) + 1}"

    def unknown(self, ids: Iterable[str]) -> list[str]:
        """The ids given that are not of any finding kept."""
        known = self._ids()
        return [finding_id for finding_id in ids if finding_key(finding_id) not in known]

    def get(self, ids: Iterable[str]) -> list[MemoryEntry]:
        """The findings with these ids, each once, in the order they were made.

        :raises KeyError: If an id is not of any finding kept.
        """
        if unknown := self.unknown(ids := list(ids)):
            raise KeyError(f"No finding has the id {', '.join(unknown)}")
        wanted = {finding_key(finding_id) for finding_id in ids}

        return [entry for entry in self._entries if finding_key(entry.id) in wanted]

    def search(self, query: str, limit: int) -> list[MemoryEntry]:
        """The findings that best match a query by BM25, at most limit of them.

        Each finding's claim and evidence are matched. A finding sharing no word with the query is
        never returned, and findings that match equally well are returned in the order they were
        made.
        """
        if limit < 1:
            raise ValueError(f"limit must be at least 1, not {limit}")

        scores = bm25_scores(query, [f"{entry.claim} {entry.evidence}" for entry in self._entries])
        ranked = sorted((index for index, score in enumerate(scores) if score > 0), key=lambda index: (-scores[index], index))

        return [self._entries[index] for index in ranked[:limit]]

    def catalog(self) -> str:
        """Every finding by its id and claim, for choosing which are needed."""
        return "\n".join(f"[{entry.id}] {entry.claim}" for entry in self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {"findings": [asdict(entry) for entry in self._entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabMemory":
        return cls(MemoryEntry(**entry) for entry in data["findings"])

    def save(self, path: Path) -> None:
        """Saves the findings to a JSON file, replacing it in one step."""
        write_atomically(Path(path), json.dumps(self.to_dict(), indent=4).encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> "LabMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
