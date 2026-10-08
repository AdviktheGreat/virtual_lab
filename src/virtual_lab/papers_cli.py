"""virtual-lab-papers: reads papers for the tasks, databases, and software in them, and counts what
they share, from the command line.

    virtual-lab-papers read ~/papers ~/runs/mine --max-cost 5
    virtual-lab-papers biorxiv ~/runs/neuro --since 2024-01-01 --subject neuroscience --limit 20
    virtual-lab-papers biorxiv ~/runs/all --since 2024-01-01 --all-subjects --max-cost 50
    virtual-lab-papers summarize ~/runs/mine
    virtual-lab-papers combine ~/runs/a ~/runs/b --output ~/runs/total

A run saves each paper as it is read, so one that stops, for a limit, a run of failures, or Ctrl-C,
carries on where it stopped when the same command is run again.
"""

import argparse
import sys
from collections.abc import Sequence
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
from virtual_lab.paper_batch import (
    PaperRunReport,
    PaperSummary,
    biorxiv_papers,
    combine_paper_summaries,
    papers_in,
    read_biorxiv_subjects,
    read_papers,
    summarize_papers,
)

PROGRAM = "virtual-lab-papers"
INTERRUPTED = 130


def reading_parser() -> argparse.ArgumentParser:
    """The options that say how papers are read, which the commands that read share."""
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("how papers are read")
    group.add_argument("--model", default=DEFAULT_MODEL, help="the model to read with (default: %(default)s)")
    group.add_argument(
        "--chunk-size", type=int, default=DEFAULT_PAPER_CHUNK_SIZE, help="characters in a chunk (default: %(default)s)"
    )
    group.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_PAPER_CHUNK_OVERLAP,
        help="characters repeated between chunks (default: %(default)s)",
    )
    group.add_argument(
        "--max-chars", type=int, default=MAX_PAPER_CHARS, help="most of each paper to read (default: %(default)s)"
    )
    group.add_argument("--no-max-chars", action="store_true", help="read the whole of every paper, however long")
    group.add_argument(
        "--max-consolidation-chars",
        type=int,
        default=MAX_CONSOLIDATION_CHARS,
        help="most findings one consolidation request is given (default: %(default)s)",
    )
    group.add_argument(
        "--temperature", type=float, default=CONSISTENT_TEMPERATURE, help="sampling temperature (default: %(default)s)"
    )
    group.add_argument(
        "--model-temperature",
        action="store_true",
        help="leave the temperature to the model, for one that accepts no other",
    )
    group.add_argument("--max-completion-tokens", type=int, help="most tokens an answer may use")

    spending = parser.add_argument_group("spending, and what to do when papers fail")
    spending.add_argument("--max-cost", type=float, help="most the run may spend, in USD, in all")
    spending.add_argument("--max-cost-per-paper", type=float, help="most one paper may spend, in USD")
    spending.add_argument(
        "--retry-failed", action="store_true", help="read again the papers an earlier run could not read"
    )
    spending.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="papers that may fail in a row before the run stops (default: %(default)s)",
    )
    spending.add_argument("--keep-going", action="store_true", help="never stop for failures")

    parser.add_argument("--env-file", help="a .env file to take the API keys from")
    parser.add_argument("--quiet", action="store_true", help="do not say which paper is being read")

    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Read papers for the common tasks, databases, and software in them, and count what they share.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    reading = reading_parser()

    read = commands.add_parser(
        "read",
        parents=[reading],
        help="read the papers in a directory",
        description="Read the PDF, text, Markdown, and LaTeX files in a directory.",
    )
    read.add_argument("papers", type=Path, help="the directory of papers")
    read.add_argument("save_dir", type=Path, help="where to save the run, and where an earlier run was saved")
    read.add_argument("--recursive", action="store_true", help="read the directories within it too")

    biorxiv = commands.add_parser(
        "biorxiv",
        parents=[reading],
        help="read bioRxiv preprints that have been published",
        description=(
            "Read the preprints bioRxiv lists for a period, from the full text of their published versions "
            "in Europe PMC where it is open access. bioRxiv's own PDFs are not downloaded, since it refuses "
            "scripts. With more than one subject each is read into a directory of its own in save_dir, and "
            "the counts of all of them are added up in save_dir."
        ),
    )
    biorxiv.add_argument("save_dir", type=Path, help="where to save the run, and where an earlier run was saved")
    biorxiv.add_argument("--since", required=True, help="the first day of the period, as YYYY-MM-DD")
    biorxiv.add_argument("--until", help="the last day (default: today)")
    biorxiv.add_argument(
        "--subject",
        action="append",
        help='a bioRxiv subject such as "neuroscience"; give it more than once for several (default: all)',
    )
    biorxiv.add_argument(
        "--all-subjects", action="store_true", help=f"each of the {len(BIORXIV_SUBJECTS)} subjects Biomni reads"
    )
    biorxiv.add_argument(
        "--limit", type=int, default=10, help="preprints to take for each subject (default: %(default)s)"
    )
    biorxiv.add_argument(
        "--include-unpublished",
        action="store_true",
        help="list preprints that have not been published too, which are recorded as unavailable",
    )
    biorxiv.add_argument("--random", action="store_true", help="take a random sample of the period, not the first ones")
    biorxiv.add_argument("--seed", type=int, default=42, help="the sample's seed (default: %(default)s)")
    biorxiv.add_argument(
        "--max-pages",
        type=int,
        default=MAX_BIORXIV_PAGES,
        help="most pages of bioRxiv's listing to read, of about 30 preprints (default: %(default)s)",
    )

    summarize = commands.add_parser(
        "summarize",
        help="count what the papers read in a directory share",
        description=(
            "Count what the papers read in a directory share, and write tasks_summary.csv, "
            "databases_summary.csv, software_summary.csv, and frequency_summary.json there."
        ),
    )
    summarize.add_argument("save_dir", type=Path, help="a directory a run was saved in")
    summarize.add_argument("--top", type=int, default=10, help="how many of each to show (default: %(default)s)")

    combine = commands.add_parser(
        "combine",
        help="add up the counts of several runs",
        description="Add up the counts of several runs. A paper in two of them is counted twice.",
    )
    combine.add_argument("save_dirs", type=Path, nargs="+", help="directories that have been summarized")
    combine.add_argument("--output", type=Path, required=True, help="where to write the total")
    combine.add_argument("--top", type=int, default=10, help="how many of each to show (default: %(default)s)")

    return parser


def reading_options(arguments: argparse.Namespace) -> dict[str, Any]:
    """The options read_papers takes, from the command line's."""
    return {
        "model": arguments.model,
        "chunk_size": arguments.chunk_size,
        "chunk_overlap": arguments.chunk_overlap,
        "max_chars": None if arguments.no_max_chars else arguments.max_chars,
        "max_consolidation_chars": arguments.max_consolidation_chars,
        "temperature": None if arguments.model_temperature else arguments.temperature,
        "max_completion_tokens": arguments.max_completion_tokens,
        "max_cost": arguments.max_cost,
        "max_cost_per_paper": arguments.max_cost_per_paper,
        "retry_failed": arguments.retry_failed,
        "max_consecutive_failures": None if arguments.keep_going else arguments.max_consecutive_failures,
        "on_progress": None if arguments.quiet else lambda line: print(line, file=sys.stderr, flush=True),
    }


def dollars(amount: float | None) -> str:
    return "an amount that cannot be worked out" if amount is None else f"${amount:.4f}"


def describe(report: PaperRunReport, subject: str | None = None) -> str:
    """What a run did, in a sentence or two."""
    what = f"{subject}: " if subject else ""
    others = [
        f"{count} {name}" for count, name in ((report.unavailable, "unavailable"), (report.failed, "failed")) if count
    ]
    line = f"{what}Read {report.read} of {report.papers} papers"
    if others:
        line += f" ({', '.join(others)})"
    line += f", spending {dollars(report.spent)}."
    if report.stopped is not None:
        line += f"\n  Stopped: {report.stopped}"

    return line


def show_counts(summary: PaperSummary, top: int) -> list[str]:
    """The most common of each kind, as lines."""
    lines = [f"Counted over {summary.papers} papers read."]
    for kind in ("tasks", "databases", "software"):
        shown = list(getattr(summary, kind).items())[: max(top, 0)]
        if not shown:
            continue

        lines.append(f"\nMost common {kind}:")
        lines += [f"  {count:>4}  {name}" for name, count in shown]

    return lines


def command_read(arguments: argparse.Namespace) -> int:
    report = read_papers(
        papers_in(arguments.papers, recursive=arguments.recursive), arguments.save_dir, **reading_options(arguments)
    )
    summary = summarize_papers(arguments.save_dir)
    print(describe(report))
    print(f"Saved in {arguments.save_dir}")
    print("\n".join(show_counts(summary, 10)))

    return 0 if report.stopped is None else 1


def command_biorxiv(arguments: argparse.Namespace) -> int:
    subjects = list(BIORXIV_SUBJECTS) if arguments.all_subjects else arguments.subject or ["all"]
    if arguments.all_subjects and arguments.subject:
        raise ValueError("Give --subject or --all-subjects, not both")

    options = reading_options(arguments)
    listing = {
        "published_only": not arguments.include_unpublished,
        "random_sample": arguments.random,
        "seed": arguments.seed,
    }
    if len(subjects) == 1:
        papers = biorxiv_papers(
            arguments.since,
            arguments.until,
            subject=subjects[0],
            limit=arguments.limit,
            max_pages=arguments.max_pages,
            **listing,
        )
        if not papers:
            print(f"bioRxiv lists no preprints for {subjects[0]} in that period.")
            return 0

        report = read_papers(papers, arguments.save_dir, **options)
        summary = summarize_papers(arguments.save_dir)
        print(describe(report))
        print(f"Saved in {arguments.save_dir}")
        print("\n".join(show_counts(summary, 10)))

        return 0 if report.stopped is None else 1

    subjects_report = read_biorxiv_subjects(
        arguments.save_dir,
        arguments.since,
        arguments.until,
        subjects=subjects,
        papers_per_subject=arguments.limit,
        max_pages=arguments.max_pages,
        **listing,
        **options,
    )
    for subject, report in subjects_report.reports.items():
        print(describe(report, subject))
    if not subjects_report.reports:
        print("bioRxiv lists no preprints for those subjects in that period.")
    print(f"Spent {dollars(subjects_report.spent)} in all. Saved in {arguments.save_dir}")
    print("\n".join(show_counts(subjects_report.combined, 10)))

    return 0 if subjects_report.stopped is None else 1


def command_summarize(arguments: argparse.Namespace) -> int:
    summary = summarize_papers(arguments.save_dir)
    print("\n".join(show_counts(summary, arguments.top)))
    print(f"\nWritten to {arguments.save_dir}")

    return 0


def command_combine(arguments: argparse.Namespace) -> int:
    summary = combine_paper_summaries(arguments.save_dirs, arguments.output)
    print("\n".join(show_counts(summary, arguments.top)))
    print(f"\nWritten to {arguments.output}")

    return 0


COMMANDS = {
    "read": command_read,
    "biorxiv": command_biorxiv,
    "summarize": command_summarize,
    "combine": command_combine,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command virtual-lab-papers.

    :param argv: The arguments, defaulting to the command line's.
    :return: The exit status: 0 when it did everything asked, 1 when a run stopped early, for a
        limit or a run of failures, 2 when it failed with an error, and 130 when it was interrupted.
    """
    arguments = build_parser().parse_args(argv)

    try:
        if getattr(arguments, "env_file", None):
            from virtual_lab.env_file import load_env

            load_env(arguments.env_file)

        return COMMANDS[arguments.command](arguments)
    except KeyboardInterrupt:
        print(
            f"\n{PROGRAM}: interrupted. What was read is saved, and the same command carries on from there.",
            file=sys.stderr,
        )

        return INTERRUPTED
    except Exception as error:
        print(f"{PROGRAM}: {type(error).__name__}: {error}", file=sys.stderr)

        return 2


if __name__ == "__main__":
    sys.exit(main())
