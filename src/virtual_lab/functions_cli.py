"""virtual-lab-functions: writes a function for each task, checks it, and saves it, from the command line.

    virtual-lab-functions generate ~/functions --tasks tasks.json
    virtual-lab-functions generate ~/functions --tasks ~/runs/neuro --min-papers 3 --limit 20 --verify docker
    virtual-lab-functions generate ~/functions --task "Align paired-end reads with BWA-MEM" --max-cost 2
    virtual-lab-functions list ~/functions

The tasks are Biomni's file of descriptions, {"tasks": [...]}, or the summaries and the directories that
virtual-lab-papers writes. A run saves each function as it is written, so one that stops, for a limit, a
run of failures, or Ctrl-C, carries on where it stopped when the same command is run again.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from virtual_lab.constants import CONSISTENT_TEMPERATURE, DEFAULT_MAX_FUNCTION_ATTEMPTS, DEFAULT_MODEL
from virtual_lab.function_generator import (
    FAILED,
    FunctionRunReport,
    FunctionTask,
    function_tasks,
    generate_functions,
    saved_functions,
    unique_tasks,
)

PROGRAM = "virtual-lab-functions"
INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Write a Python function for each task, check it, and save it, as Biomni's "
        "generate_function.py does, with the checks and the retries it does not have.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="write a function for each task")
    generate.add_argument("save_dir", type=Path, help="where to save the functions, and where an earlier run was saved")
    tasks = generate.add_argument_group("the tasks (at least one)")
    tasks.add_argument(
        "--tasks",
        action="append",
        default=[],
        metavar="PATH",
        help='a JSON file of {"tasks": [...]}, or a summary of papers, or a directory papers were read into; '
        "may be repeated",
    )
    tasks.add_argument("--task", action="append", default=[], metavar="TEXT", help="a task, in words; may be repeated")
    tasks.add_argument(
        "--min-papers", type=int, default=1, help="of the tasks in a summary, only those found in this many papers"
    )
    tasks.add_argument("--limit", type=int, help="write only the first tasks of each file, after that")

    writing = generate.add_argument_group("how functions are written and checked")
    writing.add_argument("--model", default=DEFAULT_MODEL, help="the model to write with (default: %(default)s)")
    writing.add_argument(
        "--temperature", type=float, default=CONSISTENT_TEMPERATURE, help="sampling temperature (default: %(default)s)"
    )
    writing.add_argument(
        "--model-temperature",
        action="store_true",
        help="leave the temperature to the model, for one that accepts no other",
    )
    writing.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_FUNCTION_ATTEMPTS,
        help="times to ask for each function, the first included (default: %(default)s)",
    )
    writing.add_argument("--max-completion-tokens", type=int, help="most tokens an answer may use")
    writing.add_argument(
        "--verify",
        choices=("docker", "local"),
        help="also import each function in a Docker container, or here, with no isolation, before keeping it",
    )
    writing.add_argument("--image", help="the Docker image to verify in, if not the sandbox image")
    writing.add_argument("--platform", help="the platform to run the image as, such as linux/amd64")
    writing.add_argument("--timeout", type=float, help="seconds to allow each import")

    spending = generate.add_argument_group("spending, and what to do when functions fail")
    spending.add_argument("--max-cost", type=float, help="most the run may spend, in USD, in all")
    spending.add_argument("--max-cost-per-function", type=float, help="most one function may spend, in USD")
    spending.add_argument(
        "--retry-failed", action="store_true", help="write again the functions an earlier run could not make pass"
    )
    spending.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="functions that may fail in a row before the run stops (default: %(default)s)",
    )
    spending.add_argument("--keep-going", action="store_true", help="never stop for failures")
    generate.add_argument("--env-file", help="a .env file to take the API keys from")
    generate.add_argument("--quiet", action="store_true", help="do not say which function is being written")

    listing = commands.add_parser("list", help="list the functions saved in a directory")
    listing.add_argument("save_dir", type=Path)

    return parser


def dollars(amount: float | None) -> str:
    return "an amount that cannot be worked out" if amount is None else f"${amount:.4f}"


def describe(report: FunctionRunReport) -> str:
    """What a run did, in a sentence or two."""
    others = [f"{count} {name}" for count, name in ((report.unverified, "not run"), (report.failed, "failed")) if count]
    line = f"Wrote {report.verified + report.unverified} of {report.tasks} functions"
    if others:
        line += f" ({', '.join(others)})"
    line += f", spending {dollars(report.spent)}."
    if report.stopped is not None:
        line += f"\n  Stopped: {report.stopped}"

    return line


def executor_for(arguments: argparse.Namespace) -> Any:
    if arguments.verify is None:
        return None

    from virtual_lab.execution import DockerExecutor, LocalExecutor

    if arguments.verify == "local":
        return LocalExecutor(warn=False)

    options: dict[str, Any] = {}
    if arguments.image:
        options["image"] = arguments.image
    if arguments.platform:
        options["platform"] = arguments.platform

    return DockerExecutor(**options)


def command_generate(arguments: argparse.Namespace) -> int:
    tasks: list[FunctionTask] = []
    for path in arguments.tasks:
        tasks += function_tasks(path, min_papers=arguments.min_papers, limit=arguments.limit)
    tasks += [FunctionTask.of(text) for text in arguments.task]
    if not arguments.tasks and not arguments.task:
        raise ValueError("Give the tasks, with --tasks or --task")
    tasks = unique_tasks(tasks)
    if not tasks:
        print("There are no tasks to write functions for.")
        return 0

    report = generate_functions(
        tasks,
        arguments.save_dir,
        model=arguments.model,
        temperature=None if arguments.model_temperature else arguments.temperature,
        max_attempts=arguments.max_attempts,
        executor=executor_for(arguments),
        timeout=arguments.timeout,
        max_completion_tokens=arguments.max_completion_tokens,
        max_cost=arguments.max_cost,
        max_cost_per_function=arguments.max_cost_per_function,
        retry_failed=arguments.retry_failed,
        max_consecutive_failures=None if arguments.keep_going else arguments.max_consecutive_failures,
        on_progress=None if arguments.quiet else lambda line: print(line, file=sys.stderr, flush=True),
    )
    print(describe(report))
    print(f"Saved in {arguments.save_dir}")
    for result in report.results:
        if result.status == FAILED:
            print(f"  Failed: {result.name}: {result.error}")

    return 0 if report.stopped is None and report.failed == 0 else 1


def command_list(arguments: argparse.Namespace) -> int:
    saved = saved_functions(arguments.save_dir)
    if not saved:
        print(f"No functions are saved in {arguments.save_dir}.")
        return 0

    for function in saved:
        needs = f" (needs {', '.join(function.modules)})" if function.modules else ""
        print(f"{function.status:<10} {function.name}{needs}")
        if function.error:
            print(f"           {function.error[:200]}")

    return 0


COMMANDS = {"generate": command_generate, "list": command_list}


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command virtual-lab-functions.

    :param argv: The arguments, defaulting to the command line's.
    :return: The exit status: 0 when it did everything asked, 1 when a run stopped early, for a
        limit or a run of failures, or a function could not be written, 2 when it failed with an
        error, and 130 when it was interrupted.
    """
    arguments = build_parser().parse_args(argv)

    try:
        if getattr(arguments, "env_file", None):
            from virtual_lab.env_file import load_env

            load_env(arguments.env_file)

        return COMMANDS[arguments.command](arguments)
    except KeyboardInterrupt:
        print(
            f"\n{PROGRAM}: interrupted. What was written is saved, and the same command carries on from there.",
            file=sys.stderr,
        )

        return INTERRUPTED
    except Exception as error:
        print(f"{PROGRAM}: {type(error).__name__}: {error}", file=sys.stderr)

        return 2


if __name__ == "__main__":
    sys.exit(main())
