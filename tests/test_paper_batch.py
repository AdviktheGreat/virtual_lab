"""Tests for reading many papers and counting what they share."""

import csv
import json
import random
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from virtual_lab import paper_batch
from virtual_lab.literature import Article, ArticleText
from virtual_lab.paper_batch import (
    Paper,
    PaperResult,
    PaperUnavailableError,
    biorxiv_papers,
    clean_cell,
    combine_paper_summaries,
    key_from,
    load_result,
    paper_text,
    papers_in,
    read_biorxiv_subjects,
    read_papers,
    save_json,
    summarize_papers,
    tally,
)
from virtual_lab.papers import merge_findings
from virtual_lab.records import RecordNotFoundError
from virtual_lab.utils import CostUnknownError, MeetingUsage
from virtual_lab.web import WebRequestError

from conftest import TEST_MODEL, FakeClient, FakeListing, biorxiv_record, make_usage, open_article
from test_papers import CHUNK, DESEQ, PAPER, database, findings, package, queue, task

SHORT = "One short paper about differential expression."


def write_papers(directory: Path, *texts: str) -> list[Paper]:
    directory.mkdir(parents=True, exist_ok=True)
    for number, text in enumerate(texts, 1):
        (directory / f"paper{number}.txt").write_text(text, encoding="utf-8")

    return papers_in(directory)


class TestPapersIn:
    def test_the_pdf_text_markdown_and_latex_files_of_a_directory_are_papers_in_name_order(
        self, tmp_path: Path
    ) -> None:
        for name in ["b.txt", "a.PDF", "c.md", "d.tex", "e.csv", "f.docx", ".hidden.txt"]:
            (tmp_path / name).write_text("text")

        papers = papers_in(tmp_path)

        assert [paper.key for paper in papers] == ["a.PDF", "b.txt", "c.md", "d.tex"]
        assert [paper.title for paper in papers] == ["a", "b", "c", "d"]
        assert [paper.path for paper in papers] == [tmp_path / name for name in ["a.PDF", "b.txt", "c.md", "d.tex"]]

    def test_the_directories_within_it_are_read_only_if_asked(self, tmp_path: Path) -> None:
        (tmp_path / "top.txt").write_text("text")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.txt").write_text("text")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "hidden.txt").write_text("text")

        assert [paper.key for paper in papers_in(tmp_path)] == ["top.txt"]
        assert [paper.key for paper in papers_in(tmp_path, recursive=True)] == ["sub_deep.txt", "top.txt"]

    def test_two_paths_that_clean_to_one_key_are_told_apart(self, tmp_path: Path) -> None:
        (tmp_path / "a b.txt").write_text("text")
        (tmp_path / "a_b.txt").write_text("text")

        keys = [paper.key for paper in papers_in(tmp_path)]

        assert len(set(keys)) == 2 and keys[0] == "a_b.txt" and keys[1].startswith("a_b.txt-")

    def test_a_directory_that_is_not_there_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError, match="no directory of papers"):
            papers_in(tmp_path / "missing")


class TestPaper:
    @pytest.mark.parametrize("key", ["", ".hidden", "../x", "a/b", "-x", "x" * 151, "a b"])
    def test_a_key_that_could_name_anything_but_a_result_is_refused(self, key: str) -> None:
        with pytest.raises(ValueError, match="key must begin with a letter or digit"):
            Paper(key=key)

    def test_a_paper_is_labelled_as_biomni_labels_it(self) -> None:
        assert Paper(key="k", title="A title", doi="10.1/x").label == "A title (10.1/x)"
        assert Paper(key="k", title="A title").label == "A title"
        assert Paper(key="k").label == "k"

    def test_a_paper_is_saved_and_read_back_with_its_file(self, tmp_path: Path) -> None:
        paper = Paper(key="k", title="T", doi="10.1/x", published_doi="10.2/y", path=tmp_path / "k.pdf")

        assert Paper.from_dict(json.loads(json.dumps(paper.to_dict()))) == paper
        assert Paper.from_dict(Paper(key="k").to_dict()) == Paper(key="k")

    def test_a_key_is_made_of_whatever_text_is_given(self) -> None:
        assert key_from("10.1101/2023.12.30.573731") == "10.1101_2023.12.30.573731"
        assert key_from("../../etc/passwd") == "etc_passwd"
        assert key_from("///") == "paper"
        assert len(key_from("x" * 500)) == 150


class TestBiorxivPapers:
    def test_the_period_the_subject_and_the_page_are_asked_for(self, listing: FakeListing) -> None:
        biorxiv_papers("2024-01-01", "2024-01-31", subject="Developmental Biology", limit=1)

        url, params = listing.calls[0]
        assert url == "https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-31/0/json"
        assert params == {"category": "developmental_biology"}

    def test_every_subject_is_asked_for_with_all_and_the_period_ends_today_by_default(
        self, listing: FakeListing
    ) -> None:
        biorxiv_papers("2024-01-01", subject="all", limit=1)

        url, params = listing.calls[0]
        assert url.endswith(f"/2024-01-01/{date.today().isoformat()}/0/json") and params == {}

    def test_a_paper_has_what_the_listing_says_of_it(self, listing: FakeListing) -> None:
        listing.records = [biorxiv_record("10.1101/2023.12.30.573731", published="10.1038/x", abstract="Why.\nHow.")]

        (paper,) = biorxiv_papers("2024-01-01", "2024-01-31")

        assert paper == Paper(
            key="10.1101_2023.12.30.573731",
            title="Title of 10.1101/2023.12.30.573731",
            doi="10.1101/2023.12.30.573731",
            authors="A. Author",
            date="2024-01-02",
            category="neuroscience",
            abstract="Why.\nHow.",
            license="cc_by",
            published_doi="10.1038/x",
        )

    def test_the_first_ones_are_taken_and_no_more_pages_than_they_need_are_read(self, listing: FakeListing) -> None:
        papers = biorxiv_papers("2024-01-01", "2024-01-31", limit=3)

        assert [paper.doi for paper in papers] == ["10.1101/0", "10.1101/1", "10.1101/2"]
        assert [url.split("/")[-2] for url, _ in listing.calls] == ["0", "2"]

    def test_the_listing_is_read_to_its_end_and_no_further(self, listing: FakeListing) -> None:
        papers = biorxiv_papers("2024-01-01", "2024-01-31", limit=100)

        assert len(papers) == 7
        assert [url.split("/")[-2] for url, _ in listing.calls] == ["0", "2", "4", "6"]

    def test_only_those_published_are_kept_unless_all_are_asked_for(self, listing: FakeListing) -> None:
        listing.records = [
            biorxiv_record("a", "NA"),
            biorxiv_record("b", "10.9/b"),
            biorxiv_record("c", ""),
            biorxiv_record("d", "10.9/d"),
        ]

        assert [paper.doi for paper in biorxiv_papers("2024-01-01", "2024-01-31")] == ["b", "d"]
        everything = biorxiv_papers("2024-01-01", "2024-01-31", published_only=False)
        assert [paper.doi for paper in everything] == ["a", "b", "c", "d"]
        assert [paper.published_doi for paper in everything] == ["", "10.9/b", "", "10.9/d"]

    def test_a_preprint_listed_again_as_a_later_version_is_one_paper_with_the_version_that_qualifies(
        self, listing: FakeListing
    ) -> None:
        listing.records = [
            biorxiv_record("a", "NA", version="1"),
            biorxiv_record("a", "10.9/a", version="2"),
            biorxiv_record("a", "10.9/other", version="3"),
            biorxiv_record("b", "10.9/b"),
        ]

        papers = biorxiv_papers("2024-01-01", "2024-01-31")

        assert [(paper.doi, paper.published_doi) for paper in papers] == [("a", "10.9/a"), ("b", "10.9/b")]

    def test_a_sample_is_random_but_the_same_for_the_same_seed(self, listing: FakeListing) -> None:
        first = biorxiv_papers("2024-01-01", "2024-01-31", limit=3, random_sample=True)
        again = biorxiv_papers("2024-01-01", "2024-01-31", limit=3, random_sample=True)
        other = biorxiv_papers("2024-01-01", "2024-01-31", limit=3, random_sample=True, seed=7)

        everything = biorxiv_papers("2024-01-01", "2024-01-31", limit=100)
        assert first == again == random.Random(42).sample(everything, 3)
        assert other == random.Random(7).sample(everything, 3)
        assert first != other

    def test_a_sample_reads_the_whole_listing(self, listing: FakeListing) -> None:
        biorxiv_papers("2024-01-01", "2024-01-31", limit=1, random_sample=True)

        assert len(listing.calls) == 4

    def test_no_more_pages_are_read_than_are_allowed(self, listing: FakeListing) -> None:
        papers = biorxiv_papers("2024-01-01", "2024-01-31", limit=100, max_pages=2)

        assert len(listing.calls) == 2 and len(papers) == 4

    def test_a_period_with_nothing_posted_is_no_papers(self, listing: FakeListing) -> None:
        listing.records = []

        assert biorxiv_papers("2024-01-01", "2024-01-31") == []
        assert len(listing.calls) == 1

    def test_what_the_listing_gets_wrong_is_skipped(
        self, listing: FakeListing, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        listing.records = ["nonsense", {"title": "No DOI"}, biorxiv_record("ok", "10.9/ok"), None]  # type: ignore[list-item]

        assert [paper.doi for paper in biorxiv_papers("2024-01-01", "2024-01-31")] == ["ok"]

        monkeypatch.setattr(paper_batch, "request_json", lambda *args, **kwargs: ["not", "an", "object"])
        assert biorxiv_papers("2024-01-01", "2024-01-31") == []

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"since": "January"}, "YYYY-MM-DD"),
            ({"since": "2024-01-31", "until": "2024-01-01"}, "ends on 2024-01-01, before it begins"),
            ({"since": "2024-01-01", "until": "tomorrow"}, "YYYY-MM-DD"),
            ({"since": "2024-01-01", "limit": 0}, "limit must be at least 1"),
            ({"since": "2024-01-01", "max_pages": 0}, "max_pages must be at least 1"),
            ({"since": "2024-01-01", "subject": " "}, "Give a subject"),
        ],
    )
    def test_a_period_or_option_that_cannot_be_used_is_refused_before_anything_is_asked(
        self, listing: FakeListing, options: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            biorxiv_papers(**options)

        assert listing.calls == []


class TestPaperText:
    def test_a_file_is_read(self, tmp_path: Path) -> None:
        (tmp_path / "p.txt").write_text("The text.")

        assert paper_text(Paper(key="p", path=tmp_path / "p.txt")) == "The text."

    def test_a_pmcid_is_read_from_europe_pmc_with_its_title_and_sections(self, europe_pmc: dict[str, Any]) -> None:
        europe_pmc["texts"]["PMC1"] = ArticleText(
            article=open_article(),
            sections=(("Abstract", "What it is."), ("Methods", "We ran DESeq2.")),
        )

        text = paper_text(Paper(key="p", pmcid="PMC1"))

        assert text == "The article\n\nAbstract\nWhat it is.\n\nMethods\nWe ran DESeq2."
        assert europe_pmc["asked"] == [("text", "PMC1")]

    def test_a_preprint_is_read_as_its_published_version_where_that_is_open(self, europe_pmc: dict[str, Any]) -> None:
        europe_pmc["articles"]["10.9/x"] = open_article("PMC7")
        europe_pmc["texts"]["PMC7"] = ArticleText(article=open_article("PMC7"), sections=(("Methods", "Text."),))

        text = paper_text(Paper(key="p", doi="10.1101/x", published_doi="10.9/x"))

        assert "Methods\nText." in text
        assert europe_pmc["asked"] == [("article", "10.9/x"), ("text", "PMC7")]

    def test_a_published_version_that_is_not_open_is_unavailable_and_its_text_is_not_asked_for(
        self, europe_pmc: dict[str, Any]
    ) -> None:
        europe_pmc["articles"]["10.9/x"] = Article(article_id="1", pmcid="PMC7", open_access=False, in_europe_pmc=True)

        with pytest.raises(PaperUnavailableError, match="holds no open full text of 10.9/x"):
            paper_text(Paper(key="p", published_doi="10.9/x"))

        assert europe_pmc["asked"] == [("article", "10.9/x")]

    def test_a_paper_with_nothing_to_read_it_from_is_unavailable(self, europe_pmc: dict[str, Any]) -> None:
        with pytest.raises(PaperUnavailableError, match="no file, no PMCID, and no published version"):
            paper_text(Paper(key="p", doi="10.1101/x"))

        assert europe_pmc["asked"] == []

    def test_an_article_europe_pmc_does_not_have_is_not_found(self, europe_pmc: dict[str, Any]) -> None:
        with pytest.raises(RecordNotFoundError):
            paper_text(Paper(key="p", published_doi="10.9/unknown"))


class TestReadPapers:
    def read(self, fake_client: FakeClient, papers: list[Paper], save_dir: Path, **options: Any) -> Any:
        options = {"model": TEST_MODEL, "client": fake_client, **options}

        return read_papers(papers, save_dir, **options)

    def test_each_paper_is_read_and_saved_with_the_tables_that_count_what_they_share(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        papers = write_papers(tmp_path / "in", SHORT, "Another short paper.")
        queue(fake_client, DESEQ, DESEQ, DESEQ, DESEQ)
        lines: list[str] = []

        report = self.read(fake_client, papers, tmp_path / "out", on_progress=lines.append)

        assert (report.papers, report.read, report.unavailable, report.failed) == (2, 2, 0, 0)
        assert report.stopped is None and report.cost == pytest.approx(4 * 0.00045)
        assert report.spent == pytest.approx(4 * 0.00045)
        assert [result.paper.key for result in report.results] == ["paper1.txt", "paper2.txt"]
        saved = load_result(tmp_path / "out" / "results" / "paper1.txt.json")
        assert saved is not None and saved.findings == DESEQ and saved.chunks == 1 and saved.status == "read"
        assert saved.usage["num_calls"] == 2
        assert json.loads((tmp_path / "out" / "report.json").read_text())["read"] == 2
        frequencies = json.loads((tmp_path / "out" / "frequency_summary.json").read_text())
        assert frequencies["papers"] == 2
        assert frequencies["tasks"] == {"RNA-seq differential expression with DESeq2": 2}
        assert lines[0] == "Paper 1 of 2: paper1" and "  Reading chunk 1 of 1" in lines

    def test_the_run_is_recorded_with_what_it_was_read_with(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, DESEQ, DESEQ)

        self.read(fake_client, write_papers(tmp_path / "in", SHORT), tmp_path / "out", chunk_size=500)

        run = json.loads((tmp_path / "out" / "run.json").read_text())
        assert run["model"] == TEST_MODEL and run["chunk_size"] == 500 and run["max_chars"] == 200_000

    def test_a_paper_already_read_is_not_read_again_and_a_run_carries_on_where_it_stopped(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        papers = write_papers(tmp_path / "in", SHORT, "Second.", "Third.")
        queue(fake_client, *[DESEQ] * 4, usage=make_usage(prompt_tokens=1_000, completion_tokens=0))

        first = self.read(fake_client, papers, tmp_path / "out", max_cost=0.005)

        assert first.stopped is not None and "reaches its max_cost of $0.0050" in first.stopped
        assert [result.paper.key for result in first.results] == ["paper1.txt"]
        calls = len(fake_client.completions.parse_calls)
        assert calls == 2 and not (tmp_path / "out" / "results" / "paper2.txt.json").exists()

        queue(fake_client, *[DESEQ] * 4)
        second = self.read(fake_client, papers, tmp_path / "out")

        assert second.stopped is None and second.read == 3
        assert len(fake_client.completions.parse_calls) == calls + 4
        assert second.spent == pytest.approx(4 * 0.00045)

        third = self.read(fake_client, papers, tmp_path / "out")

        assert third.read == 3 and third.spent == 0 and len(fake_client.completions.parse_calls) == calls + 4

    def test_a_directory_holding_a_run_with_other_options_is_refused_before_anything_is_read(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        papers = write_papers(tmp_path / "in", SHORT)
        queue(fake_client, DESEQ, DESEQ)
        self.read(fake_client, papers, tmp_path / "out")

        with pytest.raises(ValueError, match="different chunk_size, max_chars"):
            self.read(fake_client, papers, tmp_path / "out", chunk_size=1_000, max_chars=5_000)

        assert len(fake_client.completions.parse_calls) == 2

    def test_a_paper_with_no_text_is_saved_as_unavailable_and_asked_nothing(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        gone = Paper(key="gone.txt", path=tmp_path / "gone.txt")
        unknown = Paper(key="unknown", doi="10.1101/x")

        report = self.read(fake_client, [gone, unknown], tmp_path / "out")

        assert (report.read, report.unavailable, report.failed) == (0, 2, 0) and report.stopped is None
        assert report.results[0].error is not None and report.results[0].error.startswith("FileNotFoundError")
        assert report.results[1].error is not None and "PaperUnavailableError" in report.results[1].error
        assert fake_client.completions.parse_calls == [] and report.cost == 0

    def test_a_preprint_whose_published_version_europe_pmc_does_not_know_is_unavailable(
        self, fake_client: FakeClient, tmp_path: Path, europe_pmc: dict[str, Any]
    ) -> None:
        report = self.read(fake_client, [Paper(key="p", published_doi="10.9/unknown")], tmp_path / "out")

        assert (report.unavailable, report.failed) == (1, 0)
        assert (report.results[0].error or "").startswith("RecordNotFoundError")

    def test_a_paper_is_unavailable_when_its_file_is_not_one_that_can_be_read(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        (tmp_path / "bad.pdf").write_bytes(b"not a pdf")

        report = self.read(fake_client, [Paper(key="bad.pdf", path=tmp_path / "bad.pdf")], tmp_path / "out")

        assert report.unavailable == 1 and "could not be read as a PDF" in (report.results[0].error or "")

    def test_the_papers_that_could_not_be_read_are_read_again_only_if_asked(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        path = tmp_path / "later.txt"
        papers = [Paper(key="later.txt", path=path)]
        assert self.read(fake_client, papers, tmp_path / "out").unavailable == 1
        path.write_text(SHORT)
        queue(fake_client, DESEQ, DESEQ)

        assert self.read(fake_client, papers, tmp_path / "out").unavailable == 1
        assert fake_client.completions.parse_calls == []

        again = self.read(fake_client, papers, tmp_path / "out", retry_failed=True)

        assert (again.read, again.unavailable) == (1, 0) and len(fake_client.completions.parse_calls) == 2

    def test_a_paper_that_cannot_be_read_by_the_model_is_saved_as_failed_with_what_it_cost(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, None, DESEQ, DESEQ)
        papers = write_papers(tmp_path / "in", SHORT, SHORT)

        report = self.read(fake_client, papers, tmp_path / "out")

        failed, read = report.results
        assert failed.status == "failed" and failed.error is not None
        assert "None of the 1 chunks could be read" in failed.error and failed.cost == pytest.approx(0.00045)
        assert failed.usage["num_calls"] == 1 and failed.findings is None
        assert read.status == "read" and report.stopped is None

    def test_three_failures_in_a_row_stop_the_run_and_a_paper_read_between_resets_the_count(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, None, None, None, None)

        stopped = self.read(fake_client, write_papers(tmp_path / "a", *[SHORT] * 4), tmp_path / "a-out")

        assert len(stopped.results) == 3 and stopped.stopped is not None
        assert stopped.stopped.startswith("3 papers in a row failed, the last with PaperReadingError")
        assert not (tmp_path / "a-out" / "results" / "paper4.txt.json").exists()

        queue(fake_client, None, None, DESEQ, DESEQ, None, None)
        carried = self.read(fake_client, write_papers(tmp_path / "b", *[SHORT] * 5), tmp_path / "b-out")

        assert carried.stopped is None and (carried.failed, carried.read) == (4, 1)

    def test_a_paper_with_no_text_neither_adds_to_a_run_of_failures_nor_ends_it(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, None, None, None)
        papers = write_papers(tmp_path / "in", SHORT, SHORT, SHORT, SHORT)
        papers.insert(2, Paper(key="gone", path=tmp_path / "gone.txt"))

        report = self.read(fake_client, papers, tmp_path / "out")

        assert report.stopped is not None and [result.status for result in report.results] == [
            "failed",
            "failed",
            "unavailable",
            "failed",
        ]

    def test_failures_never_stop_a_run_that_is_told_not_to(self, fake_client: FakeClient, tmp_path: Path) -> None:
        queue(fake_client, *[None] * 5)

        report = self.read(
            fake_client, write_papers(tmp_path / "in", *[SHORT] * 5), tmp_path / "out", max_consecutive_failures=None
        )

        assert report.failed == 5 and report.stopped is None

    def test_a_paper_stopped_by_the_runs_limit_is_left_unfinished_and_unsaved(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, *[DESEQ] * 4, usage=make_usage(prompt_tokens=1_000, completion_tokens=0))
        papers = write_papers(tmp_path / "in", SHORT, SHORT)

        report = self.read(fake_client, papers, tmp_path / "out", max_cost=0.0075)

        assert [result.paper.key for result in report.results] == ["paper1.txt"]
        assert report.stopped is not None and "while reading paper2.txt, which is left unfinished" in report.stopped
        assert report.spent == pytest.approx(0.0075)
        assert not (tmp_path / "out" / "results" / "paper2.txt.json").exists()

    def test_a_paper_stopped_by_its_own_limit_fails_with_what_it_found_and_the_run_goes_on(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        # Each request costs $0.0025. The first paper has three chunks, and the limit of $0.004 is
        # reached after two of them, but it leaves room for the two requests the second paper needs.
        queue(fake_client, *[DESEQ] * 4, usage=make_usage(prompt_tokens=1_000, completion_tokens=0))
        papers = write_papers(tmp_path / "in", PAPER, SHORT)

        report = self.read(
            fake_client, papers, tmp_path / "out", chunk_size=CHUNK, chunk_overlap=0, max_cost_per_paper=0.004
        )

        first, second = report.results
        assert first.status == "failed" and first.error is not None
        assert first.error.startswith("reached its limit on spending: PaperBudgetExceededError")
        assert first.cost == pytest.approx(0.005) and first.usage["num_calls"] == 2
        assert first.findings == merge_findings([DESEQ, DESEQ])
        assert second.status == "read" and second.findings == DESEQ and report.stopped is None

    def test_what_a_paper_found_before_it_failed_is_kept_and_not_counted(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        queue(fake_client, DESEQ, DESEQ, DESEQ, None)

        report = self.read(
            fake_client, write_papers(tmp_path / "in", PAPER), tmp_path / "out", chunk_size=CHUNK, chunk_overlap=0
        )

        (result,) = report.results
        assert result.status == "failed" and "could not be consolidated" in (result.error or "")
        assert result.findings == merge_findings([DESEQ]) and result.cost == pytest.approx(4 * 0.00045)
        saved = load_result(tmp_path / "out" / "results" / "paper1.txt.json")
        assert saved is not None and saved.findings == result.findings
        frequencies = json.loads((tmp_path / "out" / "frequency_summary.json").read_text())
        assert frequencies == {"papers": 0, "tasks": {}, "databases": {}, "software": {}}
        assert not (tmp_path / "out" / "tasks_summary.csv").exists()

    def test_a_limit_already_reached_starts_no_paper(self, fake_client: FakeClient, tmp_path: Path) -> None:
        report = self.read(fake_client, write_papers(tmp_path / "in", SHORT), tmp_path / "out", max_cost=0)

        assert report.results == [] and fake_client.completions.parse_calls == []
        assert report.stopped is not None and "while reading" not in report.stopped

    def test_the_time_each_paper_took_is_saved(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter([100.0, 102.5])
        monkeypatch.setattr(paper_batch, "time", SimpleNamespace(time=lambda: next(clock)))
        queue(fake_client, DESEQ, DESEQ)

        report = self.read(fake_client, write_papers(tmp_path / "in", SHORT), tmp_path / "out")

        assert report.results[0].elapsed == 2.5
        saved = load_result(tmp_path / "out" / "results" / "paper1.txt.json")
        assert saved is not None and saved.elapsed == 2.5

    def test_a_limit_is_for_every_run_that_shares_the_account(self, fake_client: FakeClient, tmp_path: Path) -> None:
        shared = MeetingUsage()
        queue(fake_client, *[DESEQ] * 4, usage=make_usage(prompt_tokens=1_000, completion_tokens=0))

        first = self.read(fake_client, write_papers(tmp_path / "a", SHORT), tmp_path / "a-out", usage=shared)
        second = self.read(
            fake_client, write_papers(tmp_path / "b", SHORT), tmp_path / "b-out", usage=shared, max_cost=0.005
        )

        assert first.stopped is None and first.spent == pytest.approx(0.005)
        assert second.stopped is not None and second.results == [] and second.spent == 0
        assert shared.num_calls == 2

    def test_a_limit_cannot_be_enforced_when_what_was_spent_cannot_be_worked_out(
        self, fake_client: FakeClient, tmp_path: Path
    ) -> None:
        shared = MeetingUsage()
        shared.add(TEST_MODEL, None)

        report = self.read(
            fake_client, write_papers(tmp_path / "in", SHORT), tmp_path / "out", usage=shared, max_cost=1.0
        )

        assert report.stopped is not None and "cannot be worked out" in report.stopped
        assert report.results == [] and report.spent is None and fake_client.completions.parse_calls == []

    @pytest.mark.parametrize(
        ("options", "error", "message"),
        [
            ({"chunk_size": 0}, ValueError, "chunk_size must be above zero"),
            ({"max_cost": -1.0}, ValueError, "max_cost must be a finite amount"),
            ({"max_cost_per_paper": float("nan")}, ValueError, "max_cost_per_paper must be a finite amount"),
            ({"max_consecutive_failures": 0}, ValueError, "max_consecutive_failures must be at least 1"),
            ({"temperature": 3.0}, ValueError, "temperature must be between 0 and 2"),
            ({"model": "a-model-nobody-priced", "max_cost": 1.0}, CostUnknownError, "cannot be enforced"),
            ({"model": "a-model-nobody-priced", "max_cost_per_paper": 1.0}, CostUnknownError, "cannot be enforced"),
        ],
    )
    def test_options_that_would_fail_every_paper_are_refused_before_anything_is_saved(
        self, fake_client: FakeClient, tmp_path: Path, options: dict[str, Any], error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=message):
            self.read(fake_client, write_papers(tmp_path / "in", SHORT), tmp_path / "out", **options)

        assert not (tmp_path / "out").exists() and fake_client.completions.parse_calls == []

    def test_two_papers_with_one_key_are_refused(self, fake_client: FakeClient, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Two papers have the same key"):
            self.read(fake_client, [Paper(key="same"), Paper(key="same")], tmp_path / "out")

    def test_a_missing_pdf_reader_stops_the_run_rather_than_failing_every_paper(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_pypdf(paper: Paper) -> str:
            raise ImportError('Reading a PDF needs pypdf: pip install "virtual-lab[papers]"')

        monkeypatch.setattr(paper_batch, "paper_text", no_pypdf)

        with pytest.raises(ImportError, match="virtual-lab\\[papers\\]"):
            self.read(fake_client, [Paper(key="a.pdf")], tmp_path / "out")

        assert not (tmp_path / "out" / "results").exists()

    def test_a_network_failure_fetching_a_text_is_a_failure_of_the_paper(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def down(paper: Paper) -> str:
            raise WebRequestError("Could not fetch https://www.ebi.ac.uk after 3 attempts")

        monkeypatch.setattr(paper_batch, "paper_text", down)

        report = self.read(fake_client, [Paper(key="a")], tmp_path / "out")

        assert report.failed == 1 and (report.results[0].error or "").startswith("WebRequestError")

    def test_a_failure_in_the_reading_that_is_not_the_papers_is_recorded_and_the_run_goes_on(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("the model is down")

        monkeypatch.setattr(paper_batch, "extract_paper_findings", broken)

        report = self.read(fake_client, write_papers(tmp_path / "in", SHORT), tmp_path / "out")

        assert report.failed == 1 and report.results[0].error == "RuntimeError: the model is down"

    def test_the_model_is_built_once_for_all_the_papers(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[str] = []
        real = paper_batch.resolve_chat_models

        def counting(models: list[str], **kwargs: Any) -> Any:
            built.extend(models)
            return real(models, **kwargs)

        monkeypatch.setattr(paper_batch, "resolve_chat_models", counting)
        queue(fake_client, *[DESEQ] * 6)

        self.read(fake_client, write_papers(tmp_path / "in", SHORT, SHORT, SHORT), tmp_path / "out")

        assert built == [TEST_MODEL]

    def test_a_result_that_cannot_be_read_back_is_read_again(self, fake_client: FakeClient, tmp_path: Path) -> None:
        papers = write_papers(tmp_path / "in", SHORT)
        queue(fake_client, DESEQ, DESEQ, DESEQ, DESEQ)
        self.read(fake_client, papers, tmp_path / "out")
        (tmp_path / "out" / "results" / "paper1.txt.json").write_text("{ not json")

        report = self.read(fake_client, papers, tmp_path / "out")

        assert report.read == 1 and len(fake_client.completions.parse_calls) == 4
        assert load_result(tmp_path / "out" / "results" / "paper1.txt.json") is not None

    def test_a_result_is_read_back_whole(self, tmp_path: Path) -> None:
        result = PaperResult(
            paper=Paper(key="k", title="T", doi="10.1/x"),
            status="read",
            findings=DESEQ,
            chunks=3,
            characters=900,
            truncated_from=1_200,
            failed_chunks={2: "refused"},
            usage={"num_calls": 4},
            cost=0.01,
            elapsed=1.5,
        )
        save_json(tmp_path / "r.json", result.to_dict())

        assert load_result(tmp_path / "r.json") == result
        assert load_result(tmp_path / "missing.json") is None

    def test_a_result_has_one_of_three_statuses(self) -> None:
        with pytest.raises(ValueError, match="status is one of read, unavailable, failed"):
            PaperResult(paper=Paper(key="k"), status="done")


def saved(directory: Path, key: str, found: Any, status: str = "read", **fields: Any) -> None:
    paper = Paper(key=key, title=f"Paper {key}", doi=f"10.1/{key}", **fields)
    save_json(directory / "results" / f"{key}.json", PaperResult(paper=paper, status=status, findings=found).to_dict())


class TestTally:
    def test_each_name_is_counted_once_in_each_paper_and_the_commonest_come_first(self) -> None:
        counts = tally([["STAR", "DESeq2", "STAR"], ["DESeq2"], ["DESeq2", "GEO"]])

        assert counts == {"DESeq2": 3, "STAR": 1, "GEO": 1} and list(counts) == ["DESeq2", "GEO", "STAR"]

    def test_names_that_differ_only_as_names_are_one_and_are_shown_as_most_often_written(self) -> None:
        counts = tally([["deseq2"], ["DESeq2"], ["DESeq2 "], ["Deseq-2"]])

        assert counts == {"DESeq2": 4}

    def test_a_tie_is_shown_as_first_written_and_a_name_with_nothing_in_it_is_left_out(self) -> None:
        assert tally([["Bowtie2", "..."], ["bowtie 2"], [""]]) == {"Bowtie2": 2}


class TestSummarizePapers:
    def test_the_tables_have_a_row_for_each_thing_found_in_each_paper_and_the_paper(self, tmp_path: Path) -> None:
        saved(tmp_path, "a", findings([task("DESeq2 analysis")], [database("GEO")], [package("STAR")]))
        saved(tmp_path, "b", findings([task("deseq2 analysis"), task("PCA")], [database("GEO")]))

        summary = summarize_papers(tmp_path)

        assert summary.papers == 2
        assert summary.tasks == {"DESeq2 analysis": 2, "PCA": 1} and summary.databases == {"GEO": 2}
        assert summary.software == {"STAR": 1}
        with open(tmp_path / "tasks_summary.csv", newline="") as file:
            rows = list(csv.DictReader(file))
        assert [(row["task_name"], row["paper"]) for row in rows] == [
            ("DESeq2 analysis", "Paper a (10.1/a)"),
            ("deseq2 analysis", "Paper b (10.1/b)"),
            ("PCA", "Paper b (10.1/b)"),
        ]
        assert list(rows[0]) == [
            "task_name",
            "description",
            "inputs",
            "outputs",
            "code_implementation",
            "frequency",
            "standard_methods",
            "example",
            "paper",
        ]
        with open(tmp_path / "databases_summary.csv", newline="") as file:
            assert [row["name"] for row in csv.DictReader(file)] == ["GEO", "GEO"]
        frequencies = json.loads((tmp_path / "frequency_summary.json").read_text())
        assert frequencies == {
            "papers": 2,
            "tasks": {"DESeq2 analysis": 2, "PCA": 1},
            "databases": {"GEO": 2},
            "software": {"STAR": 1},
        }

    def test_a_table_with_no_rows_is_not_written_and_a_paper_not_read_is_not_counted(self, tmp_path: Path) -> None:
        saved(tmp_path, "a", findings([task("PCA")]))
        saved(tmp_path, "b", None, status="failed")
        saved(tmp_path, "c", None, status="unavailable")

        summary = summarize_papers(tmp_path)

        assert summary.papers == 1 and summary.databases == {}
        assert (tmp_path / "tasks_summary.csv").is_file()
        assert not (tmp_path / "databases_summary.csv").exists() and not (tmp_path / "software_summary.csv").exists()

    def test_a_directory_with_nothing_read_is_summarized_as_nothing(self, tmp_path: Path) -> None:
        summary = summarize_papers(tmp_path)

        assert (summary.papers, summary.tasks) == (0, {})
        assert json.loads((tmp_path / "frequency_summary.json").read_text())["papers"] == 0

    def test_what_a_spreadsheet_would_run_is_made_text(self, tmp_path: Path) -> None:
        saved(tmp_path, "a", findings([task('=HYPERLINK("http://x")', description="@SUM(A1)")]))

        summarize_papers(tmp_path)

        with open(tmp_path / "tasks_summary.csv", newline="") as file:
            (row,) = list(csv.DictReader(file))
        assert row["task_name"] == '\'=HYPERLINK("http://x")' and row["description"] == "'@SUM(A1)"
        assert row["inputs"] == "A count matrix."

    def test_only_what_would_be_read_as_a_formula_is_changed(self) -> None:
        assert [clean_cell(value) for value in ["=1", "+1", "-1", "@a", "\tx", "\rx"]] == [
            "'=1",
            "'+1",
            "'-1",
            "'@a",
            "'\tx",
            "'\rx",
        ]
        assert [clean_cell(value) for value in ["a=1", "x-ray", "", 7, None]] == ["a=1", "x-ray", "", 7, None]


class TestCombine:
    def write(self, directory: Path, papers: int, **counts: dict[str, int]) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        save_json(
            directory / "frequency_summary.json",
            {"papers": papers, "tasks": {}, "databases": {}, "software": {}, **counts},
        )

        return directory

    def test_counts_are_added_by_name_and_written_most_common_first(self, tmp_path: Path) -> None:
        first = self.write(tmp_path / "a", 10, tasks={"PCA": 4, "DESeq2": 1}, software={"STAR": 2})
        second = self.write(tmp_path / "b", 5, tasks={"pca": 3, "UMAP": 3}, databases={"GEO": 1})

        combined = combine_paper_summaries([first, second], tmp_path / "all")

        assert combined.papers == 15
        assert combined.tasks == {"PCA": 7, "UMAP": 3, "DESeq2": 1} and list(combined.tasks) == [
            "PCA",
            "UMAP",
            "DESeq2",
        ]
        assert combined.databases == {"GEO": 1} and combined.software == {"STAR": 2}
        written = json.loads((tmp_path / "all" / "combined_summary.json").read_text())
        assert written["tasks"] == combined.tasks and written["papers"] == 15
        with open(tmp_path / "all" / "tasks_frequency.csv", newline="") as file:
            assert list(csv.DictReader(file)) == [
                {"Task": "PCA", "Frequency": "7"},
                {"Task": "UMAP", "Frequency": "3"},
                {"Task": "DESeq2", "Frequency": "1"},
            ]
        for name, column in [("databases", "Database"), ("software", "Software")]:
            with open(tmp_path / "all" / f"{name}_frequency.csv", newline="") as file:
                assert column in (csv.DictReader(file).fieldnames or [])

    def test_a_summary_with_counts_that_are_not_counts_is_added_up_without_them(self, tmp_path: Path) -> None:
        directory = self.write(
            tmp_path / "a", 1, tasks={"PCA": 2, "Bad": "many", "Zero": 0, "Flag": True, "Neg": -1, "": 3}
        )
        (directory / "frequency_summary.json").write_text(
            json.dumps({"tasks": {"PCA": 2, "Bad": "many", "Zero": 0, "Flag": True, "Neg": -1, "": 3}, "software": 5})
        )

        combined = combine_paper_summaries([directory], tmp_path / "all")

        assert combined.tasks == {"PCA": 2} and combined.software == {} and combined.papers == 0

    def test_a_directory_with_no_summary_is_refused_and_nothing_is_written(self, tmp_path: Path) -> None:
        first = self.write(tmp_path / "a", 1, tasks={"PCA": 1})

        with pytest.raises(FileNotFoundError, match="has no frequency_summary.json"):
            combine_paper_summaries([first, tmp_path / "empty"], tmp_path / "all")

        assert not (tmp_path / "all").exists()

    def test_nothing_to_add_is_an_empty_total(self, tmp_path: Path) -> None:
        combined = combine_paper_summaries([], tmp_path / "all")

        assert (combined.papers, combined.tasks) == (0, {}) and (tmp_path / "all" / "combined_summary.json").is_file()


class TestReadBiorxivSubjects:
    @pytest.fixture
    def subjects(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Paper]]:
        by_subject = {
            "neuroscience": write_papers(tmp_path / "n", SHORT, SHORT),
            "cell biology": write_papers(tmp_path / "c", SHORT),
            "ecology": [],
        }
        asked: list[dict[str, Any]] = []

        def fake_papers(since: str, until: str | None = None, subject: str = "all", **options: Any) -> list[Paper]:
            asked.append({"since": since, "until": until, "subject": subject, **options})
            return by_subject[subject]

        monkeypatch.setattr(paper_batch, "biorxiv_papers", fake_papers)
        by_subject["asked"] = asked  # type: ignore[assignment]

        return by_subject

    def test_each_subject_is_read_into_its_own_directory_and_the_counts_are_added_up(
        self, fake_client: FakeClient, tmp_path: Path, subjects: dict[str, Any]
    ) -> None:
        queue(fake_client, *[DESEQ] * 6)

        report = read_biorxiv_subjects(
            tmp_path / "out",
            "2024-01-01",
            "2024-01-31",
            subjects=["neuroscience", "ecology", "cell biology"],
            papers_per_subject=5,
            max_pages=3,
            model=TEST_MODEL,
            client=fake_client,
        )

        assert list(report.reports) == ["neuroscience", "cell biology"] and report.stopped is None
        assert [report.reports[name].read for name in report.reports] == [2, 1]
        assert (tmp_path / "out" / "neuroscience" / "report.json").is_file()
        assert (tmp_path / "out" / "cell_biology" / "frequency_summary.json").is_file()
        assert report.combined.papers == 3
        assert report.combined.tasks == {"RNA-seq differential expression with DESeq2": 3}
        assert json.loads((tmp_path / "out" / "combined_summary.json").read_text())["papers"] == 3
        assert report.spent == pytest.approx(6 * 0.00045)
        assert subjects["asked"][0] == {
            "since": "2024-01-01",
            "until": "2024-01-31",
            "subject": "neuroscience",
            "limit": 5,
            "published_only": True,
            "random_sample": False,
            "seed": 42,
            "max_pages": 3,
        }

    def test_one_limit_holds_for_every_subject_and_a_stop_ends_the_run_with_what_it_has(
        self, fake_client: FakeClient, tmp_path: Path, subjects: dict[str, Any]
    ) -> None:
        queue(fake_client, *[DESEQ] * 6, usage=make_usage(prompt_tokens=1_000, completion_tokens=0))

        report = read_biorxiv_subjects(
            tmp_path / "out",
            "2024-01-01",
            subjects=["neuroscience", "cell biology"],
            model=TEST_MODEL,
            client=fake_client,
            max_cost=0.005,
        )

        assert list(report.reports) == ["neuroscience"]
        assert report.stopped is not None and report.stopped.startswith("While reading neuroscience: ")
        assert report.combined.papers == 1 and not (tmp_path / "out" / "cell_biology").exists()

    def test_the_subjects_are_biomnis_by_default(
        self, fake_client: FakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[str] = []
        monkeypatch.setattr(
            paper_batch, "biorxiv_papers", lambda since, until=None, subject="all", **kw: asked.append(subject) or []
        )

        report = read_biorxiv_subjects(tmp_path / "out", "2024-01-01", model=TEST_MODEL, client=fake_client)

        assert len(asked) == 25 and asked[0] == "evolutionary biology" and asked[-1] == "pharmacology and toxicology"
        assert report.reports == {} and report.combined.papers == 0
