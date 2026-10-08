"""Tests for the virtual-lab-papers command."""

import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

from virtual_lab import papers_cli
from virtual_lab.constants import BIORXIV_SUBJECTS
from virtual_lab.literature import ArticleText
from virtual_lab.paper_batch import PaperRunReport, PaperSummary, SubjectsReport
from virtual_lab.papers_cli import main

from conftest import FakeClient, FakeListing, make_usage, open_article
from test_paper_batch import SHORT, write_papers
from test_papers import DESEQ, findings, queue, task

TASK = "RNA-seq differential expression with DESeq2"


def run(capsys: pytest.CaptureFixture[str], *arguments: str | Path) -> tuple[int, str, str]:
    status = main([str(argument) for argument in arguments])
    captured = capsys.readouterr()

    return status, captured.out, captured.err


@pytest.fixture
def papers(tmp_path: Path) -> Path:
    write_papers(tmp_path / "papers", SHORT, "Another short paper.")

    return tmp_path / "papers"


def open_papers(europe_pmc: dict[str, Any], *numbers: int) -> None:
    for number in numbers:
        pmcid = f"PMC{number}"
        europe_pmc["articles"][f"10.9/{number}"] = open_article(pmcid)
        europe_pmc["texts"][pmcid] = ArticleText(
            article=open_article(pmcid), sections=(("Methods", "We ran DESeq2 on the counts."),)
        )


class TestRead:
    def test_the_papers_in_a_directory_are_read_saved_and_counted(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, *[DESEQ] * 4)

        status, out, err = run(capsys, "read", papers, tmp_path / "out", "--model", "gpt-4o-2024-08-06")

        assert status == 0
        assert "Read 2 of 2 papers, spending $0.0018." in out and f"Saved in {tmp_path / 'out'}" in out
        assert "Counted over 2 papers read." in out and f"     2  {TASK}" in out
        assert "Most common databases:" in out and "Most common software:" in out
        assert "Paper 1 of 2: paper1" in err and "Paper 2 of 2: paper2" in err
        assert json.loads((tmp_path / "out" / "frequency_summary.json").read_text())["tasks"] == {TASK: 2}

    def test_the_progress_is_not_shown_when_it_is_not_wanted(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, *[DESEQ] * 4)

        status, out, err = run(capsys, "read", papers, tmp_path / "out", "--quiet", "--model", "gpt-4o-2024-08-06")

        assert status == 0 and err == "" and "Read 2 of 2" in out

    def test_a_run_that_is_run_again_carries_on_and_reads_nothing_twice(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, *[DESEQ] * 4, usage=make_usage(prompt_tokens=1_000, completion_tokens=0))
        arguments = ["read", papers, tmp_path / "out", "--model", "gpt-4o-2024-08-06", "--quiet"]

        first = run(capsys, *arguments, "--max-cost", "0.005")

        assert first[0] == 1 and "Read 1 of 2 papers" in first[1] and "Stopped: The run spent $0.0050" in first[1]
        queue(fake_client, *[DESEQ] * 2)

        second = run(capsys, *arguments)

        assert second[0] == 0 and "Read 2 of 2 papers, spending $0.0009." in second[1]
        assert len(fake_client.completions.parse_calls) == 2 + 2

    def test_what_is_read_with_is_what_was_asked(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, *[DESEQ] * 4)

        status, _, _ = run(
            capsys,
            *["read", papers, tmp_path / "out", "--quiet", "--model", "gpt-4o-2024-08-06"],
            *["--chunk-size", "500", "--chunk-overlap", "50", "--no-max-chars", "--model-temperature"],
            *["--max-consolidation-chars", "5000", "--max-completion-tokens", "900"],
        )

        assert status == 0
        run_options = json.loads((tmp_path / "out" / "run.json").read_text())
        assert run_options == {
            "model": "gpt-4o-2024-08-06",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "max_chars": None,
            "max_consolidation_chars": 5000,
            "temperature": None,
            "max_completion_tokens": 900,
        }
        assert "temperature" not in fake_client.completions.parse_calls[0]

    def test_the_defaults_are_the_librarys(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue(fake_client, *[DESEQ] * 4)

        run(capsys, "read", papers, tmp_path / "out", "--quiet")

        run_options = json.loads((tmp_path / "out" / "run.json").read_text())
        assert run_options["chunk_size"] == 4_000 and run_options["max_chars"] == 200_000
        assert run_options["temperature"] == 0.2 and run_options["model"] == "gpt-5.2"

    def test_the_directories_within_the_directory_are_read_only_if_asked(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "papers" / "deeper").mkdir(parents=True)
        (tmp_path / "papers" / "top.txt").write_text(SHORT)
        (tmp_path / "papers" / "deeper" / "inner.txt").write_text(SHORT)
        queue(fake_client, *[DESEQ] * 6)

        flat = run(capsys, "read", tmp_path / "papers", tmp_path / "flat", "--quiet", "--model", "gpt-4o-2024-08-06")
        deep = run(
            capsys,
            "read",
            tmp_path / "papers",
            tmp_path / "deep",
            "--quiet",
            "--recursive",
            "--model",
            "gpt-4o-2024-08-06",
        )

        assert "Read 1 of 1 papers" in flat[1] and "Read 2 of 2 papers" in deep[1]

    def test_failures_are_reported_and_a_run_of_them_stops_it_unless_told_to_go_on(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_papers(tmp_path / "papers", *[SHORT] * 4)
        queue(fake_client, *[None] * 4)
        arguments = ["read", tmp_path / "papers", "--quiet", "--model", "gpt-4o-2024-08-06"]

        stopped = run(capsys, arguments[0], arguments[1], tmp_path / "a", *arguments[2:])

        assert stopped[0] == 1 and "Read 0 of 4 papers (3 failed)" in stopped[1]
        assert "Stopped: 3 papers in a row failed" in stopped[1]

        queue(fake_client, *[None] * 4)
        through = run(capsys, arguments[0], arguments[1], tmp_path / "b", *arguments[2:], "--keep-going")

        assert through[0] == 0 and "Read 0 of 4 papers (4 failed)" in through[1]

    def test_a_paper_that_failed_is_read_again_only_if_asked(
        self, fake_client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_papers(tmp_path / "papers", SHORT)
        arguments = ["read", tmp_path / "papers", tmp_path / "out", "--quiet", "--model", "gpt-4o-2024-08-06"]
        queue(fake_client, None)
        assert "Read 0 of 1 papers (1 failed)" in run(capsys, *arguments)[1]
        queue(fake_client, DESEQ, DESEQ)

        assert "Read 0 of 1 papers (1 failed)" in run(capsys, *arguments)[1]
        assert len(fake_client.completions.parse_calls) == 1

        again = run(capsys, *arguments, "--retry-failed")

        assert "Read 1 of 1 papers, spending $0.0009." in again[1]

    def test_a_paper_stopped_by_its_own_limit_fails_and_the_run_goes_on(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A request costs $0.0025, so a paper may make one before reaching $0.002, and needs two
        queue(fake_client, *[DESEQ] * 4, usage=make_usage(prompt_tokens=1_000, completion_tokens=0))

        status, out, _ = run(
            capsys,
            *["read", papers, tmp_path / "out", "--quiet", "--model", "gpt-4o-2024-08-06"],
            *["--max-cost-per-paper", "0.002"],
        )

        assert status == 0 and "Read 0 of 2 papers (2 failed)" in out
        results = json.loads((tmp_path / "out" / "report.json").read_text())["results"]
        assert all("reached its limit on spending" in result["error"] for result in results)

    def test_papers_with_no_text_are_counted_as_unavailable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_client: FakeClient
    ) -> None:
        (tmp_path / "papers").mkdir()
        (tmp_path / "papers" / "bad.pdf").write_bytes(b"not a pdf")

        status, out, _ = run(capsys, "read", tmp_path / "papers", tmp_path / "out", "--quiet")

        assert status == 0 and "Read 0 of 1 papers (1 unavailable)" in out


class TestBiorxiv:
    def test_the_preprints_listed_are_read_from_their_published_versions(
        self,
        fake_client: FakeClient,
        listing: FakeListing,
        europe_pmc: dict[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        open_papers(europe_pmc, 0, 1)
        queue(fake_client, *[DESEQ] * 4)

        status, out, err = run(
            capsys,
            *["biorxiv", tmp_path / "out", "--since", "2024-01-01", "--until", "2024-01-31", "--limit", "2"],
            *["--subject", "Neuroscience", "--model", "gpt-4o-2024-08-06"],
        )

        assert status == 0 and "Read 2 of 2 papers, spending $0.0018." in out and f"     2  {TASK}" in out
        assert "Paper 1 of 2: Title of 10.1101/0 (10.1101/0)" in err
        assert listing.calls[0] == (
            "https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-31/0/json",
            {"category": "neuroscience"},
        )
        assert (tmp_path / "out" / "results" / "10.1101_0.json").is_file()

    def test_the_listing_options_are_passed_on(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: dict[str, Any] = {}

        def fake_papers(since: str, until: str | None = None, **options: Any) -> list[Any]:
            asked.update(since=since, until=until, **options)
            return []

        monkeypatch.setattr(papers_cli, "biorxiv_papers", fake_papers)

        status, out, _ = run(
            capsys,
            *["biorxiv", tmp_path / "out", "--since", "2024-02-01", "--random", "--seed", "7", "--max-pages", "3"],
            *["--include-unpublished", "--limit", "5"],
        )

        assert status == 0 and "bioRxiv lists no preprints for all in that period." in out
        assert asked == {
            "since": "2024-02-01",
            "until": None,
            "subject": "all",
            "limit": 5,
            "max_pages": 3,
            "published_only": False,
            "random_sample": True,
            "seed": 7,
        }
        assert not (tmp_path / "out").exists()

    def test_several_subjects_are_each_read_into_a_directory_of_their_own_and_counted_together(
        self,
        fake_client: FakeClient,
        listing: FakeListing,
        europe_pmc: dict[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        open_papers(europe_pmc, 0)
        queue(fake_client, *[DESEQ] * 4)

        status, out, _ = run(
            capsys,
            *["biorxiv", tmp_path / "out", "--since", "2024-01-01", "--limit", "1", "--quiet"],
            *["--subject", "neuroscience", "--subject", "cell biology", "--model", "gpt-4o-2024-08-06"],
        )

        assert status == 0
        assert "neuroscience: Read 1 of 1 papers, spending $0.0009." in out
        assert "cell biology: Read 1 of 1 papers, spending $0.0009." in out
        assert "Counted over 2 papers read." in out and f"     2  {TASK}" in out
        assert (tmp_path / "out" / "cell_biology" / "report.json").is_file()
        assert (tmp_path / "out" / "combined_summary.json").is_file()

    def test_every_subject_biomni_reads_is_asked_for_with_all_subjects(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: dict[str, Any] = {}

        def fake_subjects(save_dir: Path, since: str, until: str | None, **options: Any) -> SubjectsReport:
            asked.update(save_dir=save_dir, since=since, until=until, **options)
            report = PaperRunReport(results=[], papers=0, spent=0.0, stopped="Out of money", save_dir=save_dir)
            return SubjectsReport(
                reports={"ecology": report}, stopped="While reading ecology: Out of money", combined=empty(), spent=None
            )

        monkeypatch.setattr(papers_cli, "read_biorxiv_subjects", fake_subjects)

        status, out, _ = run(
            capsys,
            *["biorxiv", tmp_path / "out", "--since", "2024-01-01", "--all-subjects", "--limit", "40", "--quiet"],
            *["--max-pages", "3", "--seed", "7", "--random", "--include-unpublished", "--until", "2024-03-01"],
        )

        assert status == 1 and asked["subjects"] == list(BIORXIV_SUBJECTS) and len(BIORXIV_SUBJECTS) == 25
        assert asked["papers_per_subject"] == 40 and asked["max_cost"] is None
        assert (asked["max_pages"], asked["seed"], asked["random_sample"], asked["published_only"]) == (
            3,
            7,
            True,
            False,
        )
        assert asked["until"] == "2024-03-01"
        assert "ecology: Read 0 of 0 papers, spending $0.0000.\n  Stopped: Out of money" in out
        assert "Spent an amount that cannot be worked out in all." in out

    def test_subjects_and_all_subjects_together_are_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, out, err = run(
            capsys, "biorxiv", tmp_path / "out", "--since", "2024-01-01", "--subject", "ecology", "--all-subjects"
        )

        assert status == 2 and out == "" and "Give --subject or --all-subjects, not both" in err

    def test_a_date_that_is_not_one_is_refused_with_no_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, _, err = run(capsys, "biorxiv", tmp_path / "out", "--since", "January")

        assert status == 2 and err.startswith("virtual-lab-papers: ValueError: A period is given as YYYY-MM-DD")

    def test_a_period_is_required(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["biorxiv", str(tmp_path / "out")])

        assert caught.value.code == 2


def empty() -> PaperSummary:
    return PaperSummary(papers=0, tasks={}, databases={}, software={})


class TestSummarizeAndCombine:
    @pytest.fixture
    def runs(self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
        queue(fake_client, *[DESEQ] * 4)
        run(capsys, "read", papers, tmp_path / "one", "--quiet", "--model", "gpt-4o-2024-08-06")

        return tmp_path / "one"

    def test_a_directory_is_summarized_and_shows_only_the_most_common(
        self, runs: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (runs / "frequency_summary.json").unlink()

        status, out, _ = run(capsys, "summarize", runs, "--top", "1")

        assert status == 0 and f"     2  {TASK}" in out and f"Written to {runs}" in out
        assert (runs / "frequency_summary.json").is_file()
        assert out.count("     2  ") == 3

    def test_none_is_shown_of_a_kind_when_top_is_zero(self, runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(capsys, "summarize", runs, "--top", "0")

        assert "Counted over 2 papers read." in out and "Most common" not in out

    def test_a_top_below_zero_shows_none_rather_than_all_but_the_last(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        many = findings([task("Alignment"), task("Assembly"), task("Annotation")])
        queue(fake_client, many, many, many, many)
        run(capsys, "read", papers, tmp_path / "many", "--quiet", "--model", "gpt-4o-2024-08-06")

        _, out, _ = run(capsys, "summarize", tmp_path / "many", "--top=-1")

        assert "Counted over 2 papers read." in out and "Most common" not in out
        assert "Alignment" not in out and "Annotation" not in out

    def test_several_runs_are_added_up(self, runs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        status, out, _ = run(capsys, "combine", runs, runs, "--output", tmp_path / "total")

        assert status == 0 and "Counted over 4 papers read." in out and f"     4  {TASK}" in out
        assert json.loads((tmp_path / "total" / "combined_summary.json").read_text())["tasks"] == {TASK: 4}
        assert (tmp_path / "total" / "tasks_frequency.csv").is_file()

    def test_a_directory_that_has_not_been_summarized_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "empty").mkdir()

        status, _, err = run(capsys, "combine", tmp_path / "empty", "--output", tmp_path / "total")

        assert status == 2 and "FileNotFoundError" in err and "frequency_summary.json" in err


class TestErrors:
    def test_a_directory_that_is_not_there_is_refused_with_no_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, out, err = run(capsys, "read", tmp_path / "missing", tmp_path / "out")

        assert status == 2 and out == ""
        assert err.startswith("virtual-lab-papers: NotADirectoryError: There is no directory of papers")

    def test_an_error_that_is_not_expected_is_reported_by_its_name(
        self, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("The api_key client option must be set")

        monkeypatch.setattr(papers_cli, "read_papers", broken)

        status, _, err = run(capsys, "read", papers, tmp_path / "out")

        assert status == 2 and err == "virtual-lab-papers: RuntimeError: The api_key client option must be set\n"

    def test_an_interruption_says_the_command_carries_on(
        self, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(papers_cli, "read_papers", interrupted)

        status, out, err = run(capsys, "read", papers, tmp_path / "out")

        assert status == 130 and out == "" and "the same command carries on from there" in err

    def test_a_limit_that_cannot_be_enforced_is_refused(
        self, fake_client: FakeClient, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, _, err = run(
            capsys, "read", papers, tmp_path / "out", "--model", "a-model-nobody-priced", "--max-cost", "1"
        )

        assert status == 2 and "CostUnknownError" in err and not (tmp_path / "out").exists()

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit) as caught:
            main([])

        assert caught.value.code == 2

    def test_keys_are_taken_from_an_env_file_that_is_there(
        self,
        fake_client: FakeClient,
        papers: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Set and removed so that the test puts back whatever load_env sets
        monkeypatch.setenv("VL_PAPERS_CLI_TEST", "unset")
        monkeypatch.delenv("VL_PAPERS_CLI_TEST")
        (tmp_path / "keys.env").write_text("VL_PAPERS_CLI_TEST=from-the-file\n")
        queue(fake_client, *[DESEQ] * 4)

        status, _, _ = run(capsys, "read", papers, tmp_path / "out", "--env-file", tmp_path / "keys.env", "--quiet")

        assert status == 0 and os.environ["VL_PAPERS_CLI_TEST"] == "from-the-file"

    def test_an_env_file_that_is_not_there_is_refused_before_anything_is_read(
        self, papers: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status, _, err = run(capsys, "read", papers, tmp_path / "out", "--env-file", tmp_path / "nothing.env")

        assert status == 2 and "FileNotFoundError: There is no .env file" in err and not (tmp_path / "out").exists()


def test_the_command_is_installed_by_the_package() -> None:
    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["virtual-lab-papers"] == "virtual_lab.papers_cli:main"
