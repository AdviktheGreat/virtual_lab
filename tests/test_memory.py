"""Tests for a lab's memory: keeping findings, choosing them by id, and matching them by BM25."""

import json
import math
from pathlib import Path

import pytest

from virtual_lab.memory import Finding, LabMemory, MemoryEntry, bm25_scores, finding_key, tokenize


def finding(claim: str, evidence: str = "A measurement.") -> Finding:
    return Finding(claim=claim, evidence=evidence)


def memory(*claims: str) -> LabMemory:
    lab = LabMemory()
    lab.add([finding(claim) for claim in claims], source="round_01_meeting", round=1)
    return lab


class TestTokenize:
    def test_words_are_lower_cased_without_stopwords(self) -> None:
        assert tokenize("The Nanobody binds to the SPIKE") == ["nanobody", "binds", "spike"]

    def test_a_joined_word_is_kept_whole_and_in_its_parts(self) -> None:
        assert tokenize("KP.3 and SARS-CoV-2") == ["kp.3", "kp", "3", "sars-cov-2", "sars", "cov", "2"]

    def test_punctuation_around_a_word_is_not_part_of_it(self) -> None:
        assert tokenize("(KP.3).") == ["kp.3", "kp", "3"]


class TestBM25:
    def test_no_documents_have_no_scores(self) -> None:
        assert bm25_scores("spike", []) == []

    def test_a_document_sharing_no_word_scores_nothing(self) -> None:
        assert bm25_scores("spike", ["The spike binds.", "Solubility is high."])[1] == 0

    def test_a_rarer_term_counts_for_more(self) -> None:
        documents = ["spike binding", "spike solubility", "spike escape"]
        scores = bm25_scores("spike escape", documents)
        assert scores[2] > scores[0] == scores[1] > 0

    def test_a_term_in_every_document_still_counts(self) -> None:
        assert all(score > 0 for score in bm25_scores("spike", ["spike one", "spike two"]))

    def test_a_term_repeated_counts_for_more_but_saturates(self) -> None:
        once, twice, thrice = bm25_scores("escape", ["escape x y z", "escape escape y z", "escape escape escape z"])
        assert once < twice < thrice
        assert thrice - twice < twice - once

    def test_a_longer_document_is_discounted(self) -> None:
        short, long = bm25_scores("escape", ["escape here", "escape here and over there with more words around"])
        assert short > long

    def test_a_repeated_query_term_counts_once(self) -> None:
        documents = ["escape here", "binding there"]
        assert bm25_scores("escape escape", documents) == bm25_scores("escape", documents)

    def test_the_score_is_bm25s(self) -> None:
        # One of two documents has the term once, and is as long as the average
        idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
        assert bm25_scores("escape", ["escape here", "binding site"])[0] == pytest.approx(idf * 2.5 / (1 + 1.5))


class TestLabMemory:
    def test_findings_are_kept_in_order_under_new_ids(self) -> None:
        lab = memory("A binds.", "B binds.")
        added = lab.add([finding("C binds.", " Its affinity. ")], source="round_02_findings", round=2)

        assert [entry.id for entry in lab] == ["F1", "F2", "F3"]
        assert added == [MemoryEntry("F3", "C binds.", "Its affinity.", "round_02_findings", 2)]
        assert len(lab) == 3

    def test_a_finding_with_no_claim_is_left_out(self) -> None:
        lab = LabMemory()
        assert lab.add([finding("  "), finding(" A binds. ")], source="s") == [MemoryEntry("F1", "A binds.", "A measurement.", "s")]

    def test_ids_carry_on_from_entries_given(self) -> None:
        lab = LabMemory([MemoryEntry("F7", "A.", "E.", "s"), MemoryEntry("other", "B.", "E.", "s")])
        assert [entry.id for entry in lab.add([finding("C.")], source="s")] == ["F8"]

    def test_two_entries_with_one_id_are_refused(self) -> None:
        with pytest.raises(ValueError, match="the id f1"):
            LabMemory([MemoryEntry("F1", "A.", "E.", "s"), MemoryEntry("f1 ", "B.", "E.", "s")])

    def test_findings_are_got_by_id_once_each_in_order(self) -> None:
        lab = memory("A.", "B.", "C.")
        assert [entry.claim for entry in lab.get(["F3", " f1", "F3"])] == ["A.", "C."]

    def test_an_unknown_id_is_named(self) -> None:
        lab = memory("A.")
        assert lab.unknown(["F1", "F2", "x"]) == ["F2", "x"]
        with pytest.raises(KeyError, match="F2"):
            lab.get(["F1", "F2"])

    def test_the_catalog_lists_every_claim_by_id(self) -> None:
        assert memory("A binds.", "B binds.").catalog() == "[F1] A binds.\n[F2] B binds."

    def test_an_entry_is_described_in_full(self) -> None:
        entry = MemoryEntry("F1", "A binds.", "An assay.", "round_02_meeting", 2)
        assert entry.describe() == "[F1] A binds.\n\nEvidence: An assay.\n\n(Found in round 2, round_02_meeting.)"
        assert "(Found in seed.)" in MemoryEntry("F1", "A.", "E.", "seed").describe()

    def test_ids_are_compared_without_case_or_space(self) -> None:
        assert finding_key(" f12 ") == "F12"


class TestSearch:
    def test_the_best_matches_come_first_up_to_the_limit(self) -> None:
        lab = memory("Nanobody A escapes KP.3.", "Nanobody B is soluble.", "KP.3 escape is common in KP.3 lineages.")

        assert [entry.id for entry in lab.search("KP.3 escape", limit=2)] == ["F3", "F1"]
        assert [entry.id for entry in lab.search("KP.3 escape", limit=1)] == ["F3"]

    def test_a_finding_matching_nothing_is_never_given(self) -> None:
        assert memory("A binds.", "B is soluble.").search("solubility of B", limit=5)[0].claim == "B is soluble."
        assert memory("A binds.").search("toxicity", limit=5) == []

    def test_the_evidence_is_matched_as_well(self) -> None:
        lab = LabMemory()
        lab.add([finding("A is better.", "Surface plasmon resonance."), finding("B is worse.")], source="s")
        assert [entry.id for entry in lab.search("plasmon", limit=5)] == ["F1"]

    def test_equal_matches_keep_their_order(self) -> None:
        assert [entry.id for entry in memory("A binds.", "B binds.").search("binds", limit=5)] == ["F1", "F2"]

    def test_an_empty_memory_finds_nothing(self) -> None:
        assert LabMemory().search("anything", limit=3) == []

    def test_a_limit_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            memory("A.").search("A", limit=0)


class TestSaving:
    def test_a_memory_saved_is_loaded_as_it_was(self, tmp_path: Path) -> None:
        lab = memory("A binds.", "B binds.")
        lab.save(tmp_path / "memory.json")

        loaded = LabMemory.load(tmp_path / "memory.json")

        assert loaded.entries == lab.entries
        assert json.loads((tmp_path / "memory.json").read_text())["findings"][0] == {
            "id": "F1",
            "claim": "A binds.",
            "evidence": "A measurement.",
            "source": "round_01_meeting",
            "round": 1,
        }

    def test_a_loaded_memory_carries_on_its_ids(self, tmp_path: Path) -> None:
        memory("A.", "B.").save(tmp_path / "memory.json")
        loaded = LabMemory.load(tmp_path / "memory.json")
        assert loaded.add([finding("C.")], source="s")[0].id == "F3"
