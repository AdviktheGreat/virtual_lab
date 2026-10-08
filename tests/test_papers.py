"""Tests for reading a paper for its tasks, databases, and software."""

import io
import random
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from virtual_lab import papers
from virtual_lab.papers import (
    CHUNK_PROMPT,
    CONSOLIDATION_PROMPT,
    PaperBudgetExceededError,
    PaperDatabase,
    PaperFindings,
    PaperReadingError,
    PaperSoftware,
    PaperTask,
    extract_paper_findings,
    in_batches,
    merge_findings,
    normalize_name,
    read_paper,
    size_of,
    split_text,
    truncate_text,
)
from virtual_lab.utils import CostUnknownError, MeetingUsage, compute_token_cost

from conftest import TEST_MODEL, FakeClient, make_usage, parsed_response


def task(name: str, **fields: str) -> PaperTask:
    values = {
        "description": "Finds genes that differ.",
        "inputs": "A count matrix.",
        "outputs": "A table of results.",
        "code_implementation": "pydeseq2.",
        "frequency": "Very common.",
        "standard_methods": "Negative binomial models.",
        "example": "Used on the tumour samples.",
    }
    return PaperTask(task_name=name, **{**values, **fields})


def database(name: str, **fields: str) -> PaperDatabase:
    values = {"description": "Expression data.", "url": "", "usage": "Look up samples.", "example": "Used."}
    return PaperDatabase(name=name, **{**values, **fields})


def package(name: str, **fields: str) -> PaperSoftware:
    values = {"description": "Alignment.", "url": "", "usage": "Align reads.", "example": "Used."}
    return PaperSoftware(name=name, **{**values, **fields})


def findings(
    tasks: list[PaperTask] | None = None,
    databases: list[PaperDatabase] | None = None,
    software: list[PaperSoftware] | None = None,
) -> PaperFindings:
    return PaperFindings(tasks=tasks or [], databases=databases or [], software=software or [])


DESEQ = findings([task("RNA-seq differential expression with DESeq2")], [database("GEO")], [package("STAR")])


def words(text: str) -> set[str]:
    return set(text.split())


class TestSplitText:
    def test_a_text_that_fits_is_one_chunk(self) -> None:
        assert split_text("Short text.", 100, 10) == ["Short text."]

    def test_nothing_is_nothing_to_read(self) -> None:
        assert split_text("", 100, 10) == []
        assert split_text("  \n\n  ", 100, 10) == []

    def test_a_paper_is_cut_between_paragraphs_before_anything_finer(self) -> None:
        first = "First paragraph. It has two sentences."
        second = "Second paragraph. It has two as well."

        assert split_text(f"{first}\n\n{second}", 45, 0) == [first, second]

    def test_a_long_paragraph_is_cut_between_sentences_and_keeps_their_full_stops(self) -> None:
        text = "One sentence is here. Another sentence follows. A third one ends it."

        chunks = split_text(text, 35, 0)

        assert chunks == ["One sentence is here.", "Another sentence follows.", "A third one ends it."]

    def test_a_word_longer_than_a_chunk_is_cut_between_characters(self) -> None:
        chunks = split_text("x" * 25, 10, 0)

        assert chunks == ["x" * 10, "x" * 10, "x" * 5]

    def test_a_chunk_begins_with_the_end_of_the_one_before_by_up_to_the_overlap(self) -> None:
        text = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five."

        without = split_text(text, 25, 0)
        overlapping = split_text(text, 25, 12)

        assert len(set(without)) == len(without)
        assert not any(first.endswith(second.split(". ")[0] + ".") for first, second in zip(without, without[1:]))
        assert any(first.endswith(second.split(". ")[0] + ".") for first, second in zip(overlapping, overlapping[1:]))

    @pytest.mark.parametrize("seed", range(8))
    def test_no_chunk_is_too_long_and_none_of_the_text_is_lost(self, seed: int) -> None:
        rng = random.Random(seed)
        vocabulary = ["gene", "protein.", "cell,", "DESeq2", "x" * 15, "a", "\n", "\n\n", "data. "]
        text = " ".join(rng.choice(vocabulary) for _ in range(400))
        size = rng.choice([20, 50, 120])
        overlap = rng.choice([0, 5, size // 3])

        chunks = split_text(text, size, overlap)

        assert chunks and all(0 < len(chunk) <= size for chunk in chunks)
        assert all(chunk in text for chunk in chunks)
        assert words(" ".join(chunks)) >= words(text) - {""}

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 11)])
    def test_sizes_out_of_range_are_refused(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError, match="chunk_"):
            split_text("text", size, overlap)


class TestTruncateText:
    def test_a_text_within_the_limit_is_left_alone(self) -> None:
        assert truncate_text("A sentence.", 50) == "A sentence."

    def test_a_text_is_cut_at_the_last_end_of_a_sentence(self) -> None:
        assert truncate_text("One is here. Two is here. Three is cut off", 30) == "One is here. Two is here."

    def test_a_decimal_point_is_not_the_end_of_a_sentence(self) -> None:
        text = "The cutoff was 0.05 for every comparison made in the study"

        cut = truncate_text(text, 17)

        assert cut == "The cutoff was"

    def test_a_sentence_end_near_the_start_is_not_where_a_long_text_is_cut(self) -> None:
        text = "See e.g. " + "word " * 20

        cut = truncate_text(text, 60)

        assert len(cut) > 50 and cut.endswith("word")

    def test_a_text_with_no_sentence_or_space_is_cut_where_it_must_be(self) -> None:
        assert truncate_text("x" * 40, 10) == "x" * 10


class TestMerging:
    def test_names_are_compared_without_case_or_punctuation(self) -> None:
        assert normalize_name(" DESeq-2 ") == normalize_name("deseq 2") == normalize_name("DESEQ2") == "deseq2"

    def test_names_in_any_script_are_kept_and_c_and_its_successors_are_not_one(self) -> None:
        assert normalize_name("β-catenin") == "βcatenin" != normalize_name("beta-catenin")
        assert normalize_name("遺伝子") == "遺伝子"
        assert len({normalize_name(name) for name in ["C", "C++", "C#"]}) == 3
        assert normalize_name("...") == normalize_name("_") == ""

    def test_one_item_is_kept_for_each_name_and_the_first_is_as_written(self) -> None:
        merged = merge_findings(
            [
                findings([task("DESeq2 analysis", example="First.")], [database("GEO")]),
                findings([task("deseq2 analysis", example="Second.")], [database("geo"), database("PDB")]),
            ]
        )

        assert [item.example for item in merged.tasks] == ["First."]
        assert [item.name for item in merged.databases] == ["GEO", "PDB"]

    def test_what_the_first_left_empty_is_filled_in_from_the_rest(self) -> None:
        merged = merge_findings(
            [
                findings(software=[package("STAR", url="")]),
                findings(software=[package("star", url="https://github.com/alexdobin/STAR", usage="")]),
            ]
        )

        assert [item.url for item in merged.software] == ["https://github.com/alexdobin/STAR"]
        assert merged.software[0].usage == "Align reads."

    def test_an_item_with_no_name_is_dropped(self) -> None:
        merged = merge_findings([findings([task("  "), task("---"), task("Real task")])])

        assert [item.task_name for item in merged.tasks] == ["Real task"]


class TestBatches:
    def full(self, count: int) -> PaperFindings:
        return findings(
            [task(f"Task {number}", description="d" * 200) for number in range(count)],
            [database("GEO")],
            [package("STAR")],
        )

    def test_findings_that_fit_are_one_batch(self) -> None:
        everything = self.full(2)

        assert in_batches(everything, 100_000) == [everything]

    def test_findings_that_do_not_fit_are_in_batches_that_hold_each_item_once_in_order(self) -> None:
        everything = self.full(10)

        batches = in_batches(everything, 1_500)

        assert len(batches) > 1
        assert [item.task_name for batch in batches for item in batch.tasks] == [
            item.task_name for item in everything.tasks
        ]
        assert [item.name for batch in batches for item in batch.databases] == ["GEO"]
        assert [item.name for batch in batches for item in batch.software] == ["STAR"]
        assert all(len(batch.model_dump_json()) <= 1_500 + 300 for batch in batches)
        assert not any(batch.is_empty() for batch in batches)

    def test_an_item_larger_than_the_limit_is_a_batch_of_its_own(self) -> None:
        big = findings([task("Big", description="d" * 3_000), task("Small")])

        batches = in_batches(big, 1_000)

        assert [[item.task_name for item in batch.tasks] for batch in batches] == [["Big"], ["Small"]]


def pdf_of(*pages: str) -> bytes:
    """A small PDF of pages of text, built by hand so that no PDF writer is needed."""
    count = len(pages)
    font = 3 + 2 * count
    kids = " ".join(f"{3 + 2 * number} 0 R" for number in range(count))
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {count} >>"]
    for number, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * number} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()

    return out


class TestReadPaper:
    def test_text_and_markdown_files_are_read_as_they_are(self, tmp_path: Path) -> None:
        (tmp_path / "paper.txt").write_text("Methods: DESeq2 was used.", encoding="utf-8")
        (tmp_path / "paper.MD").write_text("# Methods\n\nSTAR aligned reads.", encoding="utf-8")

        assert read_paper(tmp_path / "paper.txt") == "Methods: DESeq2 was used."
        assert read_paper(str(tmp_path / "paper.MD")) == "# Methods\n\nSTAR aligned reads."

    def test_text_that_is_not_utf8_is_read_with_the_bad_bytes_replaced(self, tmp_path: Path) -> None:
        (tmp_path / "paper.txt").write_bytes(b"caf\xe9 culture")

        assert read_paper(tmp_path / "paper.txt") == "caf\ufffd culture"

    def test_the_text_of_a_pdf_is_read_page_by_page(self, tmp_path: Path) -> None:
        (tmp_path / "paper.pdf").write_bytes(pdf_of("DESeq2 differential expression", "STAR alignment"))

        text = read_paper(tmp_path / "paper.pdf")

        assert "DESeq2 differential expression" in text and "STAR alignment" in text
        assert text.index("DESeq2") < text.index("STAR")

    def test_a_missing_file_a_wrong_type_and_a_file_that_is_too_large_are_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "paper.docx").write_bytes(b"x")
        (tmp_path / "long.txt").write_text("x" * 20)

        with pytest.raises(FileNotFoundError, match="no paper"):
            read_paper(tmp_path / "nothing.pdf")
        with pytest.raises(ValueError, match="not a paper this reads"):
            read_paper(tmp_path / "paper.docx")
        monkeypatch.setattr(papers, "MAX_PAPER_FILE_BYTES", 10)
        with pytest.raises(ValueError, match="more than the 10"):
            read_paper(tmp_path / "long.txt")

    def test_a_file_that_is_not_a_pdf_is_refused_saying_so(self, tmp_path: Path) -> None:
        (tmp_path / "paper.pdf").write_bytes(b"this is not a pdf at all")

        with pytest.raises(ValueError, match="paper.pdf could not be read as a PDF"):
            read_paper(tmp_path / "paper.pdf")

    def test_an_encrypted_pdf_is_refused(self, tmp_path: Path) -> None:
        pytest.importorskip("cryptography")
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_of("Secret text"))))
        writer.encrypt("secret", algorithm="AES-256")
        output = io.BytesIO()
        writer.write(output)
        (tmp_path / "paper.pdf").write_bytes(output.getvalue())

        with pytest.raises(ValueError, match="needs a password"):
            read_paper(tmp_path / "paper.pdf")

    def test_a_pdf_locked_only_against_copying_is_read(self, tmp_path: Path) -> None:
        pytest.importorskip("cryptography")
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_of("Open text"))))
        writer.encrypt("", owner_password="owner", algorithm="AES-256")
        output = io.BytesIO()
        writer.write(output)
        (tmp_path / "paper.pdf").write_bytes(output.getvalue())

        assert "Open text" in read_paper(tmp_path / "paper.pdf")

    def test_a_pdf_with_more_pages_than_a_paper_has_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "paper.pdf").write_bytes(pdf_of("One", "Two"))
        monkeypatch.setattr(papers, "MAX_PAPER_PAGES", 1)

        with pytest.raises(ValueError, match="2 pages, more than the 1"):
            read_paper(tmp_path / "paper.pdf")

    def test_a_pdf_is_not_read_without_pypdf_and_the_error_says_what_to_install(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "paper.pdf").write_bytes(pdf_of("One"))
        monkeypatch.setitem(sys.modules, "pypdf", None)

        with pytest.raises(ImportError, match=r"virtual-lab\[papers\]"):
            read_paper(tmp_path / "paper.pdf")


# What a chunk of this text is cut into, one chunk to each paragraph
PAPER = "\n\n".join(
    [
        "Methods. Reads were aligned with STAR and counted.",
        "Differential expression used DESeq2 on the counts.",
        "Results. Genes were found in GEO samples.",
    ]
)
CHUNK = 55


def queue(client: FakeClient, *responses: BaseModel | BaseException | None, **options: Any) -> None:
    client.completions.parsed_responses = [
        response if isinstance(response, BaseException) else parsed_response(response, **options)
        for response in responses
    ]


def read(client: FakeClient, text: str = PAPER, **options: Any) -> Any:
    options = {"model": TEST_MODEL, "chunk_size": CHUNK, "chunk_overlap": 0, **options}
    return extract_paper_findings(text, chat_models=None, client=client, **options)


def sent(client: FakeClient, number: int) -> list[dict[str, Any]]:
    return client.completions.parse_calls[number]["messages"]


class TestExtractPaperFindings:
    def test_each_chunk_is_read_and_the_findings_are_consolidated_into_the_result(
        self, fake_client: FakeClient
    ) -> None:
        first = findings([task("STAR alignment")], software=[package("STAR")])
        second = findings([task("DESeq2 analysis")], software=[package("DESeq2")])
        third = findings(databases=[database("GEO")])
        final = findings([task("RNA-seq differential expression with DESeq2")], [database("GEO")], [package("STAR")])
        queue(fake_client, first, second, third, final)

        reading = read(fake_client)

        assert reading.findings == final
        assert reading.chunks == 3 and reading.characters == len(PAPER) and reading.truncated_from is None
        assert reading.failed_chunks == {} and reading.model == TEST_MODEL
        assert len(fake_client.completions.parse_calls) == 4
        assert reading.usage.num_calls == 4 and reading.cost == pytest.approx(4 * 0.00045)

    def test_each_chunk_is_sent_with_the_guidelines_where_it_is_in_the_paper_and_its_text(
        self, fake_client: FakeClient
    ) -> None:
        queue(fake_client, DESEQ, DESEQ, DESEQ, DESEQ)

        read(fake_client)

        for number, paragraph in enumerate(PAPER.split("\n\n")):
            messages = sent(fake_client, number)
            assert messages[0] == {"role": "system", "content": CHUNK_PROMPT}
            assert f"chunk {number + 1} of 3" in messages[1]["content"]
            assert paragraph in messages[1]["content"]
        assert fake_client.completions.parse_calls[0]["response_format"] is PaperFindings

    def test_text_with_braces_reaches_the_model_as_it_is(self, fake_client: FakeClient) -> None:
        queue(fake_client, DESEQ, DESEQ)

        read(fake_client, text="The model was y ~ {group} + {batch}.", chunk_size=100)

        assert "y ~ {group} + {batch}." in sent(fake_client, 0)[1]["content"]

    def test_the_consolidation_is_given_the_findings_of_all_the_chunks_merged_by_name(
        self, fake_client: FakeClient
    ) -> None:
        first = findings([task("DESeq2 analysis")], [database("GEO")])
        second = findings([task("deseq2 analysis")], [database("PDB")])
        queue(fake_client, first, second, findings(), DESEQ)

        read(fake_client)

        messages = sent(fake_client, 3)
        assert messages[0] == {"role": "system", "content": CONSOLIDATION_PROMPT}
        merged = PaperFindings.model_validate_json(messages[1]["content"].split("\n", 1)[1])
        assert [item.task_name for item in merged.tasks] == ["DESeq2 analysis"]
        assert [item.name for item in merged.databases] == ["GEO", "PDB"]

    def test_a_paper_that_shows_nothing_common_is_not_consolidated(self, fake_client: FakeClient) -> None:
        queue(fake_client, findings(), findings(), findings())

        reading = read(fake_client)

        assert reading.findings.is_empty() and len(fake_client.completions.parse_calls) == 3

    def test_the_temperature_is_low_by_default_and_the_models_own_when_none(self, fake_client: FakeClient) -> None:
        queue(fake_client, findings(), findings(), findings(), findings(), findings(), findings())

        read(fake_client)
        read(fake_client, temperature=None)

        assert fake_client.completions.parse_calls[0]["temperature"] == 0.2
        assert "temperature" not in fake_client.completions.parse_calls[3]

    def test_a_chunk_that_cannot_be_read_is_recorded_and_the_rest_are_used(self, fake_client: FakeClient) -> None:
        queue(fake_client, DESEQ, None, DESEQ, DESEQ)
        fake_client.completions.parsed_responses[1] = parsed_response(None, refusal="No.")

        reading = read(fake_client)

        assert list(reading.failed_chunks) == [2] and "refused" in reading.failed_chunks[2]
        assert not reading.findings.is_empty()
        assert reading.usage.num_calls == 4

    def test_a_paper_none_of_whose_chunks_can_be_read_is_refused_with_what_it_cost(
        self, fake_client: FakeClient
    ) -> None:
        queue(fake_client, None, None, None)

        with pytest.raises(PaperReadingError, match="None of the 3 chunks could be read") as caught:
            read(fake_client)

        assert caught.value.usage is not None and caught.value.usage.num_calls == 3
        assert caught.value.findings is None

    def test_a_consolidation_that_fails_keeps_what_the_chunks_found_unfiltered(self, fake_client: FakeClient) -> None:
        queue(fake_client, DESEQ, DESEQ, DESEQ, None)

        with pytest.raises(PaperReadingError, match="could not be consolidated") as caught:
            read(fake_client)

        assert caught.value.findings == merge_findings([DESEQ])
        assert caught.value.usage is not None and caught.value.usage.num_calls == 4

    def test_findings_too_many_for_one_request_are_consolidated_in_batches_and_then_as_a_whole(
        self, fake_client: FakeClient
    ) -> None:
        many = findings([task(f"Task {number}", description="d" * 400) for number in range(8)])
        smaller = findings([task("Task 0"), task("Task 1")])
        # Three chunks, then the findings of the first in three batches, then what they left as a whole
        queue(fake_client, many, findings(), findings(), smaller, smaller, smaller, DESEQ)

        reading = read(fake_client, max_consolidation_chars=2_000)

        assert reading.findings == DESEQ and reading.usage.num_calls == 7
        assert fake_client.completions.parsed_responses == []
        asked = [
            PaperFindings.model_validate_json(sent(fake_client, number)[1]["content"].split("\n", 1)[1])
            for number in range(3, 7)
        ]
        batched = [[item.task_name for item in batch.tasks] for batch in asked[:3]]
        assert batched == [["Task 0", "Task 1", "Task 2"], ["Task 3", "Task 4", "Task 5"], ["Task 6", "Task 7"]]
        assert all(size_of(batch) <= 2_000 for batch in asked[:3])
        assert [item.task_name for item in asked[3].tasks] == ["Task 0", "Task 1"]

    def test_batches_that_leave_nothing_are_not_asked_about_again(self, fake_client: FakeClient) -> None:
        many = findings([task(f"Task {number}", description="d" * 400) for number in range(8)])
        queue(fake_client, many, findings(), findings(), findings(), findings(), findings())

        reading = read(fake_client, max_consolidation_chars=2_000)

        assert len(fake_client.completions.parse_calls) == 6
        assert reading.findings.is_empty()

    def test_batches_that_do_not_make_the_findings_smaller_are_not_asked_again(self, fake_client: FakeClient) -> None:
        many = findings([task(f"Task {number}", description="d" * 400) for number in range(8)])
        queue(fake_client, many, findings(), findings(), many, many, many)

        reading = read(fake_client, max_consolidation_chars=2_000)

        assert len(fake_client.completions.parse_calls) == 6
        assert reading.findings == many

    def test_consolidation_passes_stop_at_the_limit_however_the_findings_shrink(
        self, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(papers, "MAX_CONSOLIDATION_PASSES", 2)

        def one(number: int) -> PaperFindings:
            return findings([task(f"Task {number}", description="d" * 400)])

        many = findings([one(number).tasks[0] for number in range(5)])
        # Each of the five is a batch of its own, and each pass drops the first
        first_pass = [findings(), *[one(number) for number in range(1, 5)]]
        second_pass = [findings(), *[one(number) for number in range(2, 5)]]
        queue(fake_client, many, findings(), findings(), *first_pass, *second_pass)

        reading = read(fake_client, max_consolidation_chars=1_000)

        assert len(fake_client.completions.parse_calls) == 3 + 5 + 4
        assert [item.task_name for item in reading.findings.tasks] == ["Task 2", "Task 3", "Task 4"]

    def test_a_long_paper_is_cut_short_at_a_sentence_and_says_so(self, fake_client: FakeClient) -> None:
        queue(fake_client, DESEQ, DESEQ)

        reading = read(fake_client, text="One is here. Two is here. Three is cut off.", max_chars=26, chunk_size=100)

        assert reading.truncated_from == 43 and reading.characters == len("One is here. Two is here.")
        assert "Three" not in sent(fake_client, 0)[1]["content"]

    def test_progress_is_reported_before_each_request(self, fake_client: FakeClient) -> None:
        queue(fake_client, DESEQ, DESEQ, DESEQ, DESEQ)
        lines: list[str] = []

        read(fake_client, on_progress=lines.append)

        assert lines == [
            "Reading chunk 1 of 3",
            "Reading chunk 2 of 3",
            "Reading chunk 3 of 3",
            "Consolidating the findings in 1 request",
        ]

    def test_nothing_is_asked_of_an_empty_paper_or_with_options_out_of_range(self, fake_client: FakeClient) -> None:
        for text, options in [
            ("  ", {}),
            (PAPER, {"max_chars": 0}),
            (PAPER, {"max_consolidation_chars": 999}),
            (PAPER, {"chunk_overlap": CHUNK}),
            (PAPER, {"max_cost": -1}),
            (PAPER, {"max_cost": float("nan")}),
        ]:
            with pytest.raises(ValueError):
                read(fake_client, text, **options)

        assert fake_client.completions.parse_calls == []


class TestBudget:
    def big(self) -> Any:
        return make_usage(prompt_tokens=1_000, completion_tokens=0)

    def test_reading_stops_before_the_request_the_limit_leaves_no_room_for(self, fake_client: FakeClient) -> None:
        queue(fake_client, DESEQ, DESEQ, DESEQ, DESEQ, usage=self.big())

        with pytest.raises(PaperBudgetExceededError, match="reading has cost") as caught:
            read(fake_client, max_cost=0.002)

        assert len(fake_client.completions.parse_calls) == 1
        assert caught.value.spent == pytest.approx(0.0025) and caught.value.limit == 0.002
        assert caught.value.findings == DESEQ and caught.value.usage.num_calls == 1

    def test_a_limit_met_before_any_chunk_is_read_leaves_no_findings(self, fake_client: FakeClient) -> None:
        with pytest.raises(PaperBudgetExceededError) as caught:
            read(fake_client, max_cost=0.0)

        assert caught.value.findings is None and caught.value.usage.num_calls == 0

    def test_a_limit_met_in_the_consolidation_keeps_what_the_chunks_found(self, fake_client: FakeClient) -> None:
        first = findings([task("STAR alignment")])
        second = findings([task("DESeq2 analysis")])
        queue(fake_client, first, second, findings(databases=[database("GEO")]), DESEQ, usage=self.big())

        with pytest.raises(PaperBudgetExceededError) as caught:
            read(fake_client, max_cost=0.0075)

        assert len(fake_client.completions.parse_calls) == 3
        assert caught.value.findings == merge_findings([first, second, findings(databases=[database("GEO")])])
        assert caught.value.usage.num_calls == 3

    def test_a_response_that_does_not_say_what_it_used_stops_a_limited_reading_with_the_findings(
        self, fake_client: FakeClient
    ) -> None:
        # A schema's own validators fail inside the SDK, before there is a response to count
        with pytest.raises(ValidationError) as invalid:
            PaperFindings.model_validate({})
        queue(fake_client, DESEQ, invalid.value, DESEQ)

        with pytest.raises(PaperReadingError, match="did not report what it used") as caught:
            read(fake_client, max_cost=1.0)

        assert caught.value.findings == DESEQ and caught.value.usage.num_calls == 2
        assert len(fake_client.completions.parse_calls) == 2

    def test_reading_stops_when_the_limit_is_reached_exactly(self, fake_client: FakeClient) -> None:
        queue(fake_client, DESEQ, DESEQ, DESEQ, DESEQ, usage=self.big())
        one_request = compute_token_cost(TEST_MODEL, 1_000, 0)

        with pytest.raises(PaperBudgetExceededError):
            read(fake_client, max_cost=one_request)

        assert len(fake_client.completions.parse_calls) == 1

    def test_one_limit_holds_across_papers_that_share_an_account(self, fake_client: FakeClient) -> None:
        queue(fake_client, *[DESEQ] * 8, usage=self.big())
        shared = MeetingUsage()

        with pytest.raises(PaperBudgetExceededError):
            for _ in range(4):
                read(fake_client, text="One short paper.", chunk_size=100, usage=shared, max_cost=0.006)

        assert shared.compute_cost() >= 0.006
        assert len(fake_client.completions.parse_calls) == 3

    def test_each_reading_keeps_its_own_count_beside_the_shared_one(self, fake_client: FakeClient) -> None:
        queue(fake_client, *[DESEQ] * 4)
        shared = MeetingUsage()

        first = read(fake_client, text="One short paper.", chunk_size=100, usage=shared)
        second = read(fake_client, text="Another short paper.", chunk_size=100, usage=shared)

        assert first.usage.num_calls == second.usage.num_calls == 2 and shared.num_calls == 4

    def test_a_model_with_no_known_price_cannot_be_given_a_limit(self, fake_client: FakeClient) -> None:
        with pytest.raises(CostUnknownError, match="max_cost cannot be enforced"):
            read(fake_client, model="a-model-nobody-priced", max_cost=1.0)

        assert fake_client.completions.parse_calls == []
