"""Running a meeting's code and giving its author the failures back.

A critic reviewing code that was never run can only judge whether it looks right. Running it
first replaces that judgement with evidence, and handing a traceback back to whoever wrote the
code is what turns a review into a correction.

The failure is returned to the author rather than the critic on purpose. The author holds the
reasoning that produced the code and knows what it was meant to do; the critic's job is
scientific judgement, not debugging. Every attempt costs another round of code generation, so
the number of them is capped and recorded rather than left to run until something works.
"""

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

from virtual_lab.__about__ import __version__
from virtual_lab.agent import Agent
from virtual_lab.artifacts import CodeArtifacts, CodeFile, save_artifacts
from virtual_lab.constants import (
    ARTIFACT_DIR_NAME,
    CONSISTENT_TEMPERATURE,
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    DEFAULT_MAX_RETRIES,
    EXECUTION_DIR_NAME,
    MAX_REPORTED_OUTPUT_CHARS,
)
from virtual_lab.execution import ExecutionResult, Executor, run_files
from virtual_lab.prompts import code_repair_prompt
from virtual_lab.provenance import describe_agent, utc_timestamp
from virtual_lab.structured import request_structured_output
from virtual_lab.utils import MeetingUsage


def error_signature(result: ExecutionResult) -> str:
    """Summarizes a failure closely enough to tell two of them apart.

    The last line of standard error is used because that is where an exception type and message
    land. Two consecutive attempts with the same signature mean the agent is not making progress.

    :param result: The failed execution.
    :return: A short string identifying the failure.
    """
    lines = [line.strip() for line in result.stderr.strip().splitlines() if line.strip()]

    return f"{result.exit_code}|{result.timed_out}|{lines[-1] if lines else ''}"


def merge_artifacts(previous: CodeArtifacts, repaired: CodeArtifacts) -> CodeArtifacts:
    """Applies a repair to a set of files, keeping files the repair did not mention.

    The prompt asks for every file back, but a model that returns only the file it changed
    should not thereby delete the rest of the work.

    :param previous: The files as they stood before the repair.
    :param repaired: The files the agent returned.
    :return: The merged files, in their original order, with new files appended.
    """
    by_filename = {file.filename: file for file in previous.files}
    order = list(by_filename)

    for file in repaired.files:
        if file.filename not in by_filename:
            order.append(file.filename)
        by_filename[file.filename] = file

    return CodeArtifacts(files=[by_filename[filename] for filename in order])


@dataclass(frozen=True)
class RepairAttempt:
    """One pass of running a meeting's files.

    :param index: Which attempt this was, counting from one.
    :param results: Each file that was run, paired with what happened.
    """

    index: int
    results: tuple[tuple[CodeFile, ExecutionResult], ...]

    @property
    def succeeded(self) -> bool:
        """Whether every file that ran succeeded. True when there was nothing to run."""
        return all(result.succeeded for _, result in self.results)

    @property
    def failure(self) -> tuple[CodeFile, ExecutionResult] | None:
        """The first file that failed, with its result, or None if none did."""
        for file, result in self.results:
            if not result.succeeded:
                return file, result

        return None

    def to_dict(self) -> dict[str, Any]:
        """Returns the attempt as a JSON-serializable dictionary."""
        return {
            "attempt": self.index,
            "succeeded": self.succeeded,
            "runs": [
                {"filename": file.filename, **result.to_dict()} for file, result in self.results
            ],
        }


@dataclass
class RepairOutcome:
    """What happened when a meeting's code was run, including any repairs.

    :param artifacts: The files as they finally stood, repaired if they were repaired.
    :param attempts: Each pass of running them, in order.
    :param paths: The files written for the final attempt.
    :param stopped_early: Whether the loop gave up because an attempt reproduced the same error.
    :param usage: What the repair requests cost. Executions themselves cost nothing.
    """

    artifacts: CodeArtifacts
    attempts: tuple[RepairAttempt, ...] = ()
    paths: tuple[Path, ...] = ()
    stopped_early: bool = False
    usage: MeetingUsage = field(default_factory=MeetingUsage)
    started_at: str = field(default_factory=utc_timestamp)
    ended_at: str | None = None
    elapsed_seconds: float | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the final attempt ran without failing."""
        return bool(self.attempts) and self.attempts[-1].succeeded

    @property
    def num_attempts(self) -> int:
        """How many times the code was run."""
        return len(self.attempts)

    @property
    def was_repaired(self) -> bool:
        """Whether the code that finally ran differed from the code the meeting produced."""
        return self.succeeded and self.num_attempts > 1

    def report(self, max_chars: int = MAX_REPORTED_OUTPUT_CHARS) -> str:
        """Describes the outcome in the form a critic is shown when reviewing results.

        This is deliberately explicit about failure. A critic told only that code was written
        will review the code; a critic told the code failed three times reviews the approach.

        :param max_chars: The most characters of output to include.
        :return: The description.
        """
        if not self.attempts:
            return "The code was not run."

        final = self.attempts[-1]
        attempts = f"{self.num_attempts} attempt{'s' if self.num_attempts > 1 else ''}"

        if self.succeeded and self.num_attempts == 1:
            heading = "The code ran successfully on the first attempt."
        elif self.succeeded:
            heading = f"The code ran successfully after {attempts}, having been corrected."
        elif self.stopped_early:
            heading = (
                f"The code failed on {attempts} and was abandoned because the same error "
                f"recurred without progress."
            )
        else:
            heading = f"The code failed on all {attempts} and was not made to work."

        sections = [heading]

        for file, result in final.results:
            sections.append(f"--- {file.filename} ---\n{result.report(max_chars=max_chars)}")

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Returns the outcome as a JSON-serializable dictionary."""
        return {
            "succeeded": self.succeeded,
            "num_attempts": self.num_attempts,
            "was_repaired": self.was_repaired,
            "stopped_early": self.stopped_early,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": self.elapsed_seconds,
            "usage": self.usage.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "final_files": [
                {
                    "filename": file.filename,
                    "language": file.language,
                    "description": file.description,
                    "contents": file.contents,
                }
                for file in self.artifacts.files
            ],
        }


def run_with_repair(
    artifacts: CodeArtifacts,
    author: Agent,
    save_dir: Path,
    executor: Executor,
    save_name: str = "discussion",
    client: OpenAI | None = None,
    model: str | None = None,
    temperature: float = CONSISTENT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
    timeout: float | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> RepairOutcome:
    """Writes a meeting's code, runs it, and asks its author to fix it when it fails.

    The loop stops as soon as the code runs, when the attempts are used up, or when an attempt
    reproduces the previous error, since an agent repeating itself will keep repeating itself
    and each further attempt costs another round of code generation.

    :param artifacts: The files the meeting produced.
    :param author: The agent that wrote them, who will be asked to fix them.
    :param save_dir: The directory the meeting was saved in.
    :param executor: What to run the code with. Use DockerExecutor unless you have a reason not to.
    :param save_name: The name the meeting was saved under.
    :param client: The OpenAI client to use, created if not given.
    :param model: The model to ask for repairs, defaulting to the author's own model.
    :param temperature: The sampling temperature for repair requests.
    :param max_attempts: The most times to run the code, counting the first run.
    :param timeout: Seconds to allow each execution, defaulting to the executor's own limit.
    :param max_retries: Retries for failed API calls, used only if a client is created here.
    :raises ValueError: If max_attempts is less than one.
    :raises StructuredOutputError: If the author does not return usable corrected files.
    :return: What happened, whether or not the code was made to work.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    if client is None:
        client = OpenAI(max_retries=max_retries)

    outcome = RepairOutcome(artifacts=artifacts)
    attempts: list[RepairAttempt] = []
    current = artifacts
    directory = save_dir / ARTIFACT_DIR_NAME / save_name
    start_time = time.time()

    for index in range(1, max_attempts + 1):
        outcome.paths = save_artifacts(
            save_dir=save_dir, save_name=save_name, artifacts=current
        )
        attempt = RepairAttempt(
            index=index,
            results=run_files(
                directory=directory, files=current.files, executor=executor, timeout=timeout
            ),
        )
        attempts.append(attempt)

        if attempt.succeeded:
            break

        # Having spent the last attempt, there is no point paying for a correction nobody will run
        if index == max_attempts:
            break

        if len(attempts) >= 2:
            latest, previous = attempts[-1].failure, attempts[-2].failure
            if (
                latest is not None
                and previous is not None
                and error_signature(latest[1]) == error_signature(previous[1])
            ):
                outcome.stopped_early = True
                break

        failed_file, failed_result = attempt.failure  # type: ignore[misc]
        current = merge_artifacts(
            previous=current,
            repaired=request_repair(
                client=client,
                author=author,
                model=model or author.model,
                files=current.files,
                filename=failed_file.filename,
                report=failed_result.report(),
                attempt=index,
                max_attempts=max_attempts,
                temperature=temperature,
                usage=outcome.usage,
            ),
        )

    outcome.artifacts = current
    outcome.attempts = tuple(attempts)
    outcome.ended_at = utc_timestamp()
    outcome.elapsed_seconds = time.time() - start_time

    return outcome


def request_repair(
    client: OpenAI,
    author: Agent,
    model: str,
    files: tuple[CodeFile, ...] | list[CodeFile],
    filename: str,
    report: str,
    attempt: int,
    max_attempts: int,
    temperature: float,
    usage: MeetingUsage,
) -> CodeArtifacts:
    """Asks an agent to fix code of its own that failed.

    The request is self-contained rather than a continuation of the meeting, so that a repair
    costs one focused call instead of resending the entire discussion.

    :param client: The OpenAI client to use.
    :param author: The agent that wrote the code.
    :param model: The model to ask.
    :param files: The files as they currently stand.
    :param filename: The file that failed.
    :param report: What happened when the code was run.
    :param attempt: Which attempt this is, counting from one.
    :param max_attempts: The most attempts allowed.
    :param temperature: The sampling temperature.
    :param usage: The usage to add this call's cost to.
    :raises StructuredOutputError: If the agent does not return usable corrected files.
    :return: The files the agent returned, which may not include every original file.
    """
    repaired, response = request_structured_output(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": author.prompt},
            {
                "role": "user",
                "content": code_repair_prompt(
                    agent=author,
                    files=files,
                    filename=filename,
                    report=report,
                    attempt=attempt,
                    max_attempts=max_attempts,
                ),
            },
        ],
        schema=CodeArtifacts,
        temperature=temperature,
    )
    usage.add(model=model, usage=response.usage)

    return repaired


def save_execution_record(
    save_dir: Path, save_name: str, outcome: RepairOutcome, author: Agent, executor: Executor
) -> Path:
    """Writes what happened when a meeting's code was run.

    :param save_dir: The directory the meeting was saved in.
    :param save_name: The name the meeting was saved under.
    :param outcome: What happened.
    :param author: The agent whose code it was.
    :param executor: What ran it.
    :return: The path written.
    """
    execution_dir = save_dir / EXECUTION_DIR_NAME
    execution_dir.mkdir(parents=True, exist_ok=True)
    path = execution_dir / f"{save_name}.json"

    record = {
        "virtual_lab_version": __version__,
        "openai_version": openai.__version__,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "save_name": save_name,
        "author": describe_agent(author),
        "executor": describe_executor(executor),
        **outcome.to_dict(),
    }

    with open(path, "w") as f:
        json.dump(record, f, indent=4)

    return path


def describe_executor(executor: Executor) -> dict[str, Any]:
    """Describes how code was run, including whether it was isolated at all.

    :param executor: The executor to describe.
    :return: The executor's type and its settings.
    """
    description: dict[str, Any] = {"type": type(executor).__name__}

    for attribute in ("image", "allow_network", "timeout", "memory_limit", "cpu_limit", "pids_limit"):
        if hasattr(executor, attribute):
            description[attribute] = getattr(executor, attribute)

    return description
