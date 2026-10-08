"""Reading many papers for their tasks, databases, and software, and counting what they share, as
Biomni's bioRxiv scripts do (snap-stanford/Biomni, biomni/biorxiv_scripts, Apache License 2.0).

Biomni lists the bioRxiv preprints of a subject that were later published, downloads each PDF
from bioRxiv, reads it, and counts which tasks, databases, and software turn up in how many
papers, so that the ones in hundreds of papers can become tools. Its download is the part that
no longer works: bioRxiv answers a script's request for a PDF with HTTP 429, however politely it
is made. So the papers here come from two places that do answer:

- a directory of files, which a person has downloaded, from anywhere, and
- bioRxiv's own listing, which is an API, with the full text of each preprint's published version
  from Europe PMC, for those that are open access. A preprint whose published version is not
  open, or is not published, is recorded as unavailable and not read.

A run saves each paper's result as soon as it is read, so a run that stops carries on where it
stopped, and it can be limited in what it spends.
"""

import csv
import hashlib
import io
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from virtual_lab.constants import (
    BIORXIV_SUBJECTS,
    CONSISTENT_TEMPERATURE,
    DEFAULT_MODEL,
    DEFAULT_PAPER_CHUNK_OVERLAP,
    DEFAULT_PAPER_CHUNK_SIZE,
    MAX_BIORXIV_PAGES,
    MAX_CONSOLIDATION_CHARS,
    MAX_PAPER_CHARS,
)
from virtual_lab.literature import get_article, get_article_text
from virtual_lab.llm import ModelSource, resolve_chat_models
from virtual_lab.papers import (
    TEXT_SUFFIXES,
    PaperBudgetExceededError,
    PaperFindings,
    PaperReadingError,
    check_reading_options,
    extract_paper_findings,
    normalize_name,
    read_paper,
)
from virtual_lab.records import RecordNotFoundError, as_dict, as_int, as_list, as_text
from virtual_lab.utils import CostUnknownError, MeetingUsage, compute_token_cost, write_atomically
from virtual_lab.web import build_url, request_json

BIORXIV_DETAILS_URL = "https://api.biorxiv.org/details/biorxiv/{since}/{until}/{cursor}/json"

RESULTS_DIR = "results"
RUN_FILE = "run.json"
REPORT_FILE = "report.json"
FREQUENCY_FILE = "frequency_summary.json"
COMBINED_FILE = "combined_summary.json"

READ = "read"
UNAVAILABLE = "unavailable"
FAILED = "failed"
STATUSES = (READ, UNAVAILABLE, FAILED)

# A key names a paper's result file, so it may not name anything else. It begins with a letter or
# digit, which keeps it from being a hidden file or a path.
SAFE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,149}")

# The fields of each kind of finding, in the order a table has them, with the paper they came from
# as its last column
TASK_COLUMNS = (
    "task_name",
    "description",
    "inputs",
    "outputs",
    "code_implementation",
    "frequency",
    "standard_methods",
    "example",
)
NAMED_COLUMNS = ("name", "description", "url", "usage", "example")


class PaperUnavailableError(Exception):
    """Raised when there is no text to read for a paper, as against one that failed to be read."""


@dataclass(frozen=True)
class Paper:
    """A paper to read: where its text is, and what is known about it.

    :param key: What names its result file, of letters, digits, '.', '_', and '-'.
    :param title: The title.
    :param doi: The paper's DOI, or a preprint's.
    :param authors: The authors, as one string.
    :param date: When it was posted, as a string.
    :param category: The subject it was posted under.
    :param abstract: The abstract.
    :param license: The license it was posted under, which says what may be done with its text.
    :param published_doi: For a preprint, the DOI of the version published in a journal.
    :param pmcid: A PubMed Central identifier, whose open full text is read from Europe PMC.
    :param path: A file to read: a PDF, or a text, Markdown, or LaTeX file.
    """

    key: str
    title: str = ""
    doi: str = ""
    authors: str = ""
    date: str = ""
    category: str = ""
    abstract: str = ""
    license: str = ""
    published_doi: str = ""
    pmcid: str = ""
    path: Path | None = None

    def __post_init__(self) -> None:
        if not SAFE_KEY.fullmatch(self.key):
            raise ValueError(
                f"A paper's key must begin with a letter or digit and be letters, digits, '.', '_', and '-', "
                f"at most 150 of them, to name its files, not {self.key!r}"
            )

    @property
    def label(self) -> str:
        """How the paper is named in a table: its title and DOI, as Biomni names it."""
        title = self.title or self.key

        return f"{title} ({self.doi})" if self.doi else title

    def to_dict(self) -> dict[str, Any]:
        data = dict(vars(self))
        data["path"] = str(self.path) if self.path is not None else None

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        known = {name: value for name, value in data.items() if name in cls.__dataclass_fields__}
        path = known.get("path")

        return cls(**{**known, "path": Path(path) if path else None})


def key_from(text: str) -> str:
    """A key made from some text, such as a DOI or a file's path."""
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", text).lstrip("._-")[:150]

    return key or "paper"


def papers_in(directory: Path | str, recursive: bool = False) -> list[Paper]:
    """The papers in a directory: its PDF, text, Markdown, and LaTeX files, in name order.

    A file's key is its path in the directory, so two papers never share a key; if cleaning two
    different paths down to a key gives one, the second is told apart by a few characters of a
    hash of its path. Hidden files are left out.

    :param directory: Where the papers are.
    :param recursive: Whether to look in the directories within it too.
    :raises NotADirectoryError: If there is no such directory.
    :return: The papers, each with its file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"There is no directory of papers at {directory}")

    suffixes = {".pdf", *TEXT_SUFFIXES}
    candidates = directory.rglob("*") if recursive else directory.glob("*")
    files = sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: path.relative_to(directory).as_posix(),
    )

    papers: list[Paper] = []
    taken: set[str] = set()
    for path in files:
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts):
            continue

        key = key_from(relative.as_posix())
        if key in taken:
            digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:8]
            key = f"{key[:140]}-{digest}"
        taken.add(key)
        papers.append(Paper(key=key, title=path.stem, path=path))

    return papers


def biorxiv_papers(
    since: str,
    until: str | None = None,
    subject: str = "all",
    limit: int = 10,
    published_only: bool = True,
    random_sample: bool = False,
    seed: int = 42,
    max_pages: int = MAX_BIORXIV_PAGES,
) -> list[Paper]:
    """The bioRxiv preprints posted in a period, from bioRxiv's own listing.

    Biomni reads the preprints of a subject that have been published, since a published paper has
    been reviewed, and takes the first ones, or a random sample with a fixed seed. These are the
    same choices. Only the listing is read here, for each preprint's DOI, subject, abstract, and
    the DOI of its published version: the full text is read from Europe PMC, as `paper_text`
    does. One record is kept for each DOI, the first listed that qualifies.

    :param since: The first day of the period, as YYYY-MM-DD.
    :param until: The last day, defaulting to today.
    :param subject: A bioRxiv subject such as "neuroscience", or "all".
    :param limit: How many preprints to return.
    :param published_only: Whether to keep only those that have been published in a journal.
    :param random_sample: Whether to take a random sample of the period's preprints, not the
        first ones. This reads the whole listing, up to max_pages pages, so it is slow for a
        long period.
    :param seed: The random seed, which is Biomni's.
    :param max_pages: The most pages of the listing to read, at about 30 preprints a page.
    :raises ValueError: If a date is not a date, the period ends before it begins, or the limit
        or max_pages is below one.
    :raises WebRequestError: If bioRxiv could not be reached.
    :return: The preprints.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, not {limit}")
    if max_pages < 1:
        raise ValueError(f"max_pages must be at least 1, not {max_pages}")
    if not subject.strip():
        raise ValueError('Give a subject such as "neuroscience", or "all"')

    try:
        first = date.fromisoformat(since)
        last = date.fromisoformat(until) if until is not None else date.today()
    except ValueError as error:
        raise ValueError(f"A period is given as YYYY-MM-DD dates: {error}") from error
    if last < first:
        raise ValueError(f"The period ends on {last}, before it begins on {first}")

    params = {} if subject.strip().lower() == "all" else {"category": subject.strip().lower().replace(" ", "_")}
    kept: dict[str, Paper] = {}
    cursor = 0

    for _ in range(max_pages):
        url = build_url(BIORXIV_DETAILS_URL, since=first.isoformat(), until=last.isoformat(), cursor=str(cursor))
        response = as_dict(request_json(url, params=params))
        records = as_list(response.get("collection"))
        if not records:
            break

        for record in records:
            paper = biorxiv_paper(as_dict(record))
            if paper is None or paper.doi in kept or (published_only and not paper.published_doi):
                continue
            kept[paper.doi] = paper

        cursor += len(records)
        total = as_int(as_dict(next(iter(as_list(response.get("messages"))), None)).get("total"), 0) or 0
        if cursor >= total or (not random_sample and len(kept) >= limit):
            break

    papers = list(kept.values())
    if len(papers) > limit:
        papers = random.Random(seed).sample(papers, limit) if random_sample else papers[:limit]

    return papers


def biorxiv_paper(record: dict[str, Any]) -> Paper | None:
    """A paper from one record of bioRxiv's listing, or None if it has no DOI."""
    doi = as_text(record.get("doi"), None).strip()
    if not doi:
        return None

    published = as_text(record.get("published"), None).strip()

    return Paper(
        key=key_from(doi),
        title=as_text(record.get("title")),
        doi=doi,
        authors=as_text(record.get("authors")),
        date=as_text(record.get("date")),
        category=as_text(record.get("category")),
        abstract=as_text(record.get("abstract"), None),
        license=as_text(record.get("license")),
        published_doi="" if published.upper() in ("", "NA") else published,
    )


def paper_text(paper: Paper) -> str:
    """The text of a paper: its file, or the open full text of it in Europe PMC.

    A paper with no file or PMCID is looked up in Europe PMC by the DOI of its published version.
    Of an article's text, the sections Europe PMC's reader leaves out, such as the references, are
    left out.

    :raises PaperUnavailableError: If there is nothing to read the paper from, or Europe PMC does
        not hold its full text.
    :raises RecordNotFoundError: If Europe PMC has no such article.
    :raises FileNotFoundError: If the file is missing.
    :raises ValueError: If the file is not one that can be read.
    :raises WebRequestError: If Europe PMC could not be reached.
    :return: The text.
    """
    if paper.path is not None:
        return read_paper(paper.path)

    pmcid = paper.pmcid
    if not pmcid:
        if not paper.published_doi:
            raise PaperUnavailableError(f"{paper.key} has no file, no PMCID, and no published version to read")

        article = get_article(paper.published_doi)
        if not article.has_full_text:
            raise PaperUnavailableError(
                f"Europe PMC holds no open full text of {paper.published_doi}, the published version of "
                f"{paper.doi or paper.key}"
            )
        pmcid = article.pmcid

    text = get_article_text(pmcid)
    heading = text.article.title or paper.title

    return "\n\n".join([heading, *(f"{name}\n{body}" for name, body in text.sections)]).strip()


@dataclass(frozen=True)
class PaperResult:
    """What came of reading one paper, as it is saved.

    :param paper: The paper.
    :param status: "read", "unavailable" when there was no text to read, or "failed" when
        reading it failed.
    :param findings: What it showed, if it was read. For a paper that failed after some of its chunks
        were read, what those showed, which has been paid for but was not consolidated, so it is kept
        and not counted.
    :param chunks: How many chunks it was cut into.
    :param characters: How much of its text was read.
    :param truncated_from: Its length, if it was cut short.
    :param failed_chunks: Why each chunk that could not be read could not, by number.
    :param usage: What the requests used.
    :param cost: What reading it cost, in USD, or None if that cannot be worked out.
    :param error: Why it was unavailable or failed.
    :param elapsed: How long it took, in seconds.
    """

    paper: Paper
    status: str
    findings: PaperFindings | None = None
    chunks: int = 0
    characters: int = 0
    truncated_from: int | None = None
    failed_chunks: dict[int, str] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = 0.0
    error: str | None = None
    elapsed: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"A result's status is one of {', '.join(STATUSES)}, not {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper": self.paper.to_dict(),
            "status": self.status,
            "findings": self.findings.model_dump(mode="json") if self.findings is not None else None,
            "chunks": self.chunks,
            "characters": self.characters,
            "truncated_from": self.truncated_from,
            "failed_chunks": {str(number): why for number, why in self.failed_chunks.items()},
            "usage": self.usage,
            "cost": self.cost,
            "error": self.error,
            "elapsed": self.elapsed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperResult":
        findings = data.get("findings")

        return cls(
            paper=Paper.from_dict(data["paper"]),
            status=data["status"],
            findings=PaperFindings.model_validate(findings) if findings is not None else None,
            chunks=data.get("chunks", 0),
            characters=data.get("characters", 0),
            truncated_from=data.get("truncated_from"),
            failed_chunks={int(number): why for number, why in data.get("failed_chunks", {}).items()},
            usage=data.get("usage", {}),
            cost=data.get("cost", 0.0),
            error=data.get("error"),
            elapsed=data.get("elapsed", 0.0),
        )


@dataclass
class PaperRunReport:
    """What a run over many papers did.

    :param results: The result of each paper that was finished, in the order they were asked for.
    :param papers: How many papers were asked for.
    :param spent: What this run spent, in USD, or None if that cannot be worked out. Papers whose
        results were already saved cost nothing.
    :param stopped: Why the run stopped before it had gone through every paper, if it did.
    :param save_dir: Where it was saved.
    """

    results: list[PaperResult]
    papers: int
    spent: float | None
    stopped: str | None
    save_dir: Path

    @property
    def read(self) -> int:
        return sum(result.status == READ for result in self.results)

    @property
    def unavailable(self) -> int:
        return sum(result.status == UNAVAILABLE for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.status == FAILED for result in self.results)

    @property
    def cost(self) -> float | None:
        """What the finished papers cost in all, or None if any cost cannot be worked out."""
        costs = [result.cost for result in self.results]

        return None if any(cost is None for cost in costs) else sum(cost for cost in costs if cost is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "papers": self.papers,
            "finished": len(self.results),
            "read": self.read,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "cost": self.cost,
            "spent": self.spent,
            "stopped": self.stopped,
            "results": [
                {"key": result.paper.key, "status": result.status, "cost": result.cost, "error": result.error}
                for result in self.results
            ],
        }


def load_result(path: Path) -> PaperResult | None:
    """A result read back from its file, or None if there is none or it cannot be read.

    A file that cannot be read is as good as no file: the paper is read again, and the file
    written over.
    """
    if not path.is_file():
        return None

    try:
        return PaperResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def save_json(path: Path, data: Any) -> None:
    write_atomically(path, json.dumps(data, indent=4).encode("utf-8"))


def read_papers(
    papers: Iterable[Paper],
    save_dir: Path | str,
    model: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_PAPER_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_PAPER_CHUNK_OVERLAP,
    max_chars: int | None = MAX_PAPER_CHARS,
    max_consolidation_chars: int = MAX_CONSOLIDATION_CHARS,
    temperature: float | None = CONSISTENT_TEMPERATURE,
    max_completion_tokens: int | None = None,
    max_cost: float | None = None,
    max_cost_per_paper: float | None = None,
    chat_models: ModelSource | None = None,
    client: Any = None,
    usage: MeetingUsage | None = None,
    retry_failed: bool = False,
    max_consecutive_failures: int | None = 3,
    on_progress: Callable[[str], None] | None = None,
) -> PaperRunReport:
    """Reads each paper for its tasks, databases, and software, and saves what it finds.

    Each paper's result is saved under save_dir/results/ as soon as it is read, and the tables
    that count what the papers share, as Biomni's are counted, are written to save_dir when the
    run ends. A paper already saved is not read again, so a run carries on where an earlier one
    on the same directory stopped; the earlier one must have used the same model and options.

    A paper whose text cannot be had, or whose reading fails, is saved as that and the run goes
    on. Three failures in a row stop it, since that is more likely a missing key or no network
    than three bad papers. A paper stopped by a limit is not saved, and is read again next time.

    :param papers: The papers, in the order to read them.
    :param save_dir: Where to save the run, and where an earlier run of it was saved.
    :param model: The model to read with.
    :param chunk_size: The most characters in a chunk, as extract_paper_findings takes it.
    :param chunk_overlap: The characters repeated between chunks.
    :param max_chars: The most of each paper to read.
    :param max_consolidation_chars: The most findings one consolidation request is given.
    :param temperature: The sampling temperature, or None for the model's default.
    :param max_completion_tokens: The most tokens an answer may use.
    :param max_cost: The most the run may spend, in USD, counting what usage already holds. No
        paper is started once it is spent, and one running when it is reached is stopped.
    :param max_cost_per_paper: The most one paper may spend. A paper that reaches it fails, and
        the run goes on.
    :param chat_models: The chat model to ask, as hold_meeting takes it.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param usage: A count to add what the run uses to, so that several runs can keep one account.
    :param retry_failed: Whether to read again the papers an earlier run could not read or find
        text for.
    :param max_consecutive_failures: How many papers in a row may fail before the run stops, or
        None to never stop. A paper with no text to read does not count.
    :param on_progress: Called with a line saying what is being done.
    :raises ValueError: If an option is out of range, a key is used twice, or save_dir holds a run
        with other options.
    :raises CostUnknownError: If a limit is given and the model's price is not known.
    :raises ImportError: If a PDF is to be read and pypdf is not installed.
    :return: What was found of each paper, and what it cost.
    """
    check_reading_options(model, chunk_size, chunk_overlap, max_chars, max_consolidation_chars, temperature, max_cost)
    for name, limit in (("max_cost", max_cost), ("max_cost_per_paper", max_cost_per_paper)):
        if limit is not None and not (math.isfinite(limit) and limit >= 0):
            raise ValueError(f"{name} must be a finite amount, zero or more, not {limit}")
    if max_consecutive_failures is not None and max_consecutive_failures < 1:
        raise ValueError(f"max_consecutive_failures must be at least 1, not {max_consecutive_failures}")
    if max_cost_per_paper is not None:
        try:
            compute_token_cost(model, 0, 0)
        except CostUnknownError as error:
            raise CostUnknownError(f"{error}, so a limit on spending cannot be enforced") from error

    asked = list(papers)
    if len({paper.key for paper in asked}) != len(asked):
        raise ValueError("Two papers have the same key")

    # Built before anything is read, so that a model that cannot be reached fails the run before
    # any paper is saved as having failed
    llm = resolve_chat_models([model], chat_models=chat_models, client=client)[model]

    save_dir = Path(save_dir)
    run = {
        "model": model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_chars": max_chars,
        "max_consolidation_chars": max_consolidation_chars,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    run_path = save_dir / RUN_FILE
    if run_path.is_file():
        earlier = json.loads(run_path.read_text(encoding="utf-8"))
        if earlier != run:
            different = ", ".join(name for name in run if earlier.get(name) != run[name])
            raise ValueError(
                f"{save_dir} holds a run with a different {different}, whose results cannot be "
                f"counted with this one's. Save this run somewhere else."
            )
    else:
        save_json(run_path, run)

    run_usage = usage if usage is not None else MeetingUsage()

    def spent() -> float | None:
        try:
            return run_usage.compute_cost()
        except CostUnknownError:
            return None

    def say(line: str) -> None:
        if on_progress is not None:
            on_progress(line)

    started_with = spent()
    results: dict[str, PaperResult] = {}
    stopped: str | None = None
    failures_in_a_row = 0

    for number, paper in enumerate(asked, 1):
        path = save_dir / RESULTS_DIR / f"{paper.key}.json"
        saved = load_result(path)
        if saved is not None and (saved.status == READ or not retry_failed):
            results[paper.key] = saved
            continue

        limits: list[float] = []
        if max_cost is not None or max_cost_per_paper is not None:
            so_far = spent()
            if so_far is None:
                stopped = "What the run has spent cannot be worked out, so its limits cannot be enforced"
                break
            if max_cost is not None:
                if so_far >= max_cost:
                    stopped = f"The run spent ${so_far:.4f}, which reaches its max_cost of ${max_cost:.4f}"
                    break
                limits.append(max_cost)
            if max_cost_per_paper is not None:
                limits.append(so_far + max_cost_per_paper)

        say(f"Paper {number} of {len(asked)}: {paper.label}")
        began = time.time()
        result: PaperResult

        try:
            text = paper_text(paper)
        except (PaperUnavailableError, RecordNotFoundError, FileNotFoundError, ValueError) as error:
            result = PaperResult(paper=paper, status=UNAVAILABLE, error=f"{type(error).__name__}: {error}")
        except ImportError:
            raise
        except Exception as error:
            result = PaperResult(paper=paper, status=FAILED, error=f"{type(error).__name__}: {error}")
        else:
            try:
                reading = extract_paper_findings(
                    text,
                    model=model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_chars=max_chars,
                    max_consolidation_chars=max_consolidation_chars,
                    temperature=temperature,
                    max_cost=min(limits) if limits else None,
                    max_completion_tokens=max_completion_tokens,
                    chat_models={model: llm},
                    usage=run_usage,
                    on_progress=lambda line: say(f"  {line}"),
                )
            except PaperBudgetExceededError as error:
                # Either the run's limit or this paper's, and only the run's stops the run
                if max_cost is not None and error.spent >= max_cost:
                    stopped = (
                        f"The run spent ${error.spent:.4f}, which reaches its max_cost of ${max_cost:.4f}, "
                        f"while reading {paper.key}, which is left unfinished"
                    )
                    break
                result = failure(paper, error, "reached its limit on spending")
            except PaperReadingError as error:
                result = failure(paper, error)
            except Exception as error:
                result = PaperResult(paper=paper, status=FAILED, error=f"{type(error).__name__}: {error}")
            else:
                result = PaperResult(
                    paper=paper,
                    status=READ,
                    findings=reading.findings,
                    chunks=reading.chunks,
                    characters=reading.characters,
                    truncated_from=reading.truncated_from,
                    failed_chunks=reading.failed_chunks,
                    usage=reading.usage.to_dict(),
                    cost=reading.cost,
                )

        result = replace(result, elapsed=round(time.time() - began, 3))
        save_json(path, result.to_dict())
        results[paper.key] = result

        # A paper with no text to read is no sign that anything is wrong, so it neither adds to
        # a run of failures nor ends one
        if result.status == FAILED:
            failures_in_a_row += 1
        elif result.status == READ:
            failures_in_a_row = 0
        if max_consecutive_failures is not None and failures_in_a_row >= max_consecutive_failures:
            stopped = f"{failures_in_a_row} papers in a row failed, the last with {result.error}"
            break

    finished = [results[paper.key] for paper in asked if paper.key in results]
    ended_with = spent()
    report = PaperRunReport(
        results=finished,
        papers=len(asked),
        spent=ended_with - started_with if ended_with is not None and started_with is not None else None,
        stopped=stopped,
        save_dir=save_dir,
    )
    save_json(save_dir / REPORT_FILE, report.to_dict())
    summarize_papers(save_dir)

    return report


def failure(paper: Paper, error: PaperReadingError | PaperBudgetExceededError, why: str = "") -> PaperResult:
    """The result of a paper whose reading failed, with what the reading found and used before it did."""
    message = f"{type(error).__name__}: {error}"
    cost: float | None = 0.0
    if error.usage is not None:
        try:
            cost = error.usage.compute_cost()
        except CostUnknownError:
            cost = None

    return PaperResult(
        paper=paper,
        status=FAILED,
        findings=error.findings,
        usage=error.usage.to_dict() if error.usage is not None else {},
        cost=cost,
        error=f"{why}: {message}" if why else message,
    )


def clean_cell(value: Any) -> Any:
    """A table cell that a spreadsheet will not read as a formula.

    A paper's text, and so a model's account of it, is not trusted: a name beginning with '=' or
    '@' is run by a spreadsheet that opens the file. A quote before it keeps it text.
    """
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"

    return value


def csv_bytes(columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: clean_cell(row.get(column, "")) for column in columns})

    return output.getvalue().encode("utf-8")


@dataclass(frozen=True)
class PaperSummary:
    """What the papers read in a directory have in common.

    :param papers: How many papers were read.
    :param tasks: Each task and in how many papers it was found, the most common first.
    :param databases: The same for databases.
    :param software: The same for software.
    """

    papers: int
    tasks: dict[str, int]
    databases: dict[str, int]
    software: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"papers": self.papers, "tasks": self.tasks, "databases": self.databases, "software": self.software}


def tally(names_by_paper: Iterable[Iterable[str]]) -> dict[str, int]:
    """In how many papers each name is found, the most common first.

    Names are compared as normalize_name compares them, so that 'DESeq2' and 'deseq2' are one,
    and each is shown as it was most often written, the first of those if they tie. A paper that
    names something twice counts once.
    """
    papers: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}

    for names in names_by_paper:
        seen: set[str] = set()
        for name in names:
            key = normalize_name(name)
            if key and key not in seen:
                seen.add(key)
                papers[key] += 1
                spellings.setdefault(key, Counter())[name.strip()] += 1

    shown = {key: spellings[key].most_common(1)[0][0] for key in papers}
    ordered = sorted(papers, key=lambda key: (-papers[key], shown[key].lower(), shown[key]))

    return {shown[key]: papers[key] for key in ordered}


def saved_results(save_dir: Path) -> list[PaperResult]:
    """Every result saved in a directory, in the order of their keys."""
    results_dir = save_dir / RESULTS_DIR
    if not results_dir.is_dir():
        return []

    loaded = (load_result(path) for path in sorted(results_dir.glob("*.json")))

    return [result for result in loaded if result is not None]


def summarize_papers(save_dir: Path | str) -> PaperSummary:
    """Counts what the papers read in a directory have in common, and writes it there.

    Written, as Biomni writes them, are tasks_summary.csv, databases_summary.csv, and
    software_summary.csv, with a row for each thing found in each paper and the paper it was
    found in, and frequency_summary.json, with each name and the number of papers it was found in.
    Every paper read in the directory is counted, whether it was asked for in this run or an
    earlier one. A cell that a spreadsheet would read as a formula is given a leading quote.

    :param save_dir: A directory a run was saved in.
    :return: The counts.
    """
    save_dir = Path(save_dir)
    read = [
        (result.paper, result.findings)
        for result in saved_results(save_dir)
        if result.status == READ and result.findings is not None
    ]

    rows: dict[str, list[dict[str, Any]]] = {"tasks": [], "databases": [], "software": []}
    names: dict[str, list[list[str]]] = {"tasks": [], "databases": [], "software": []}
    for paper, findings in read:
        for kind, label in (("tasks", "task_name"), ("databases", "name"), ("software", "name")):
            items = [item.model_dump() for item in getattr(findings, kind)]
            rows[kind] += [{**item, "paper": paper.label} for item in items]
            names[kind].append([item[label] for item in items])

    for kind, columns in (("tasks", TASK_COLUMNS), ("databases", NAMED_COLUMNS), ("software", NAMED_COLUMNS)):
        if rows[kind]:
            write_atomically(save_dir / f"{kind}_summary.csv", csv_bytes([*columns, "paper"], rows[kind]))

    summary = PaperSummary(
        papers=len(read),
        tasks=tally(names["tasks"]),
        databases=tally(names["databases"]),
        software=tally(names["software"]),
    )
    save_json(save_dir / FREQUENCY_FILE, summary.to_dict())

    return summary


def combine_paper_summaries(save_dirs: Sequence[Path | str], output_dir: Path | str) -> PaperSummary:
    """Adds up the counts of several runs, as Biomni adds up its subjects', and writes the total.

    Each directory's frequency_summary.json is read, and what is the same by normalize_name is
    one, with its counts added. Written to output_dir are combined_summary.json, and
    tasks_frequency.csv, databases_frequency.csv, and software_frequency.csv. A paper found in
    two of the directories is counted twice.

    :param save_dirs: Directories that summarize_papers wrote a frequency summary in.
    :param output_dir: Where to write the total.
    :raises FileNotFoundError: If a directory has no frequency summary.
    :return: The total, with the number of papers the directories' summaries were made of.
    """
    totals: dict[str, Counter[str]] = {"tasks": Counter(), "databases": Counter(), "software": Counter()}
    spellings: dict[str, dict[str, Counter[str]]] = {kind: {} for kind in totals}
    papers = 0

    for directory in save_dirs:
        path = Path(directory) / FREQUENCY_FILE
        if not path.is_file():
            raise FileNotFoundError(f"{directory} has no {FREQUENCY_FILE}. Summarize it with summarize_papers first.")

        frequencies = json.loads(path.read_text(encoding="utf-8"))
        papers += as_int(frequencies.get("papers"), 0) or 0
        for kind in totals:
            for name, count in as_dict(frequencies.get(kind)).items():
                key = normalize_name(name)
                if key and isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    totals[kind][key] += count
                    spellings[kind].setdefault(key, Counter())[name] += count

    def ordered(kind: str) -> dict[str, int]:
        shown = {key: spellings[kind][key].most_common(1)[0][0] for key in totals[kind]}
        keys = sorted(totals[kind], key=lambda key: (-totals[kind][key], shown[key].lower(), shown[key]))

        return {shown[key]: totals[kind][key] for key in keys}

    combined = PaperSummary(
        papers=papers, tasks=ordered("tasks"), databases=ordered("databases"), software=ordered("software")
    )

    output_dir = Path(output_dir)
    save_json(output_dir / COMBINED_FILE, combined.to_dict())
    for kind, column in (("tasks", "Task"), ("databases", "Database"), ("software", "Software")):
        counts = getattr(combined, kind)
        write_atomically(
            output_dir / f"{kind}_frequency.csv",
            csv_bytes([column, "Frequency"], ({column: name, "Frequency": count} for name, count in counts.items())),
        )

    return combined


@dataclass
class SubjectsReport:
    """What reading the papers of several subjects did.

    :param reports: The report of each subject that was read, by subject.
    :param stopped: Why it stopped before every subject was read, if it did.
    :param combined: The counts of every subject read, added up.
    :param spent: What it spent, in USD, or None if that cannot be worked out.
    """

    reports: dict[str, PaperRunReport]
    stopped: str | None
    combined: PaperSummary
    spent: float | None


def read_biorxiv_subjects(
    save_dir: Path | str,
    since: str,
    until: str | None = None,
    subjects: Sequence[str] = BIORXIV_SUBJECTS,
    papers_per_subject: int = 100,
    published_only: bool = True,
    random_sample: bool = False,
    seed: int = 42,
    max_pages: int = MAX_BIORXIV_PAGES,
    usage: MeetingUsage | None = None,
    **options: Any,
) -> SubjectsReport:
    """Reads the bioRxiv papers of each subject, and adds up what they share, as Biomni's
    process_all_subjects.py does, with the 25 subjects it reads by default.

    Each subject is read into a directory of its own under save_dir, and the total is written to
    save_dir. One account is kept for every subject, so max_cost is for all of them, and the run
    stops, with what it has, when a subject stops, as a limit or a run of failures there is more
    likely to be everywhere.

    :param save_dir: Where to save the run.
    :param since: The first day of the period to take the preprints from, as YYYY-MM-DD.
    :param until: The last day, defaulting to today.
    :param subjects: The bioRxiv subjects.
    :param papers_per_subject: How many preprints to take for each subject.
    :param published_only: Whether to take only those that have been published.
    :param random_sample: Whether to sample the period's preprints at random.
    :param seed: The sample's seed.
    :param max_pages: The most pages of each subject's listing to read.
    :param usage: A count to add what the run uses to.
    :param options: Anything else read_papers takes, such as model, max_cost, or client.
    :raises ValueError: As read_papers and biorxiv_papers raise it.
    :raises WebRequestError: If bioRxiv could not be reached. What the subjects before it read is
        saved, and running this again carries on from there.
    :return: The report of each subject, and the total.
    """
    save_dir = Path(save_dir)
    shared = usage if usage is not None else MeetingUsage()
    reports: dict[str, PaperRunReport] = {}
    stopped: str | None = None
    directories: dict[str, Path] = {}

    def spent() -> float | None:
        try:
            return shared.compute_cost()
        except CostUnknownError:
            return None

    started_with = spent()
    for subject in subjects:
        papers = biorxiv_papers(
            since,
            until,
            subject=subject,
            limit=papers_per_subject,
            published_only=published_only,
            random_sample=random_sample,
            seed=seed,
            max_pages=max_pages,
        )
        if not papers:
            continue

        directory = save_dir / key_from(subject.lower())
        directories[subject] = directory
        report = read_papers(papers, directory, usage=shared, **options)
        reports[subject] = report
        if report.stopped is not None:
            stopped = f"While reading {subject}: {report.stopped}"
            break

    directories = {subject: directory for subject, directory in directories.items() if subject in reports}
    combined = combine_paper_summaries(list(directories.values()), save_dir)
    ended_with = spent()

    return SubjectsReport(
        reports=reports,
        stopped=stopped,
        combined=combined,
        spent=ended_with - started_with if ended_with is not None and started_with is not None else None,
    )
