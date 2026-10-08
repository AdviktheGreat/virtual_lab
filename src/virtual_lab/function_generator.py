"""Writing a function for a task, and checking it, as Biomni's FunctionGenerator does for the tasks
its papers turn up (snap-stanford/Biomni, biomni/agent/function_generator.py and
biomni/biorxiv_scripts/generate_function.py, Apache License 2.0).

Biomni asks a model, as a senior Python engineer, for the code that does a task, takes the first
fenced block of the reply, and saves it in a file named for the first six words of the task. Here
the request is the same, and so are the name and the file; what is different is what is done with
the reply. It is read for what a tool needs of a function (see function_checks), and a function
that fails is given back to the model with what is wrong, up to a few times. With an executor, the
file is also imported in it, so that a library the code needs and does not have is found out
before the function is kept, and not when a model first calls it.

A run over many tasks saves each function as it is written, so one that stops, at a limit on
spending or a run of failures, carries on where it stopped; and a function is saved only if it
passed, with what was written for the tasks that did not kept in their records.
"""

import hashlib
import json
import math
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from virtual_lab.completions import check_temperature, send_request
from virtual_lab.constants import (
    CONSISTENT_TEMPERATURE,
    DEFAULT_MAX_FUNCTION_ATTEMPTS,
    DEFAULT_MODEL,
    MAX_FUNCTION_NAME_CHARS,
    MAX_FUNCTION_TASK_CHARS,
)
from virtual_lab.execution import ExecutionResult, Executor
from virtual_lab.function_checks import FunctionReading, FunctionSourceError, extract_code, read_function, script_name
from virtual_lab.llm import ModelSource, ask, resolve_chat_models
from virtual_lab.paper_batch import COMBINED_FILE, FREQUENCY_FILE, file_digest, save_json, saved_results
from virtual_lab.papers import PaperTask, normalize_name
from virtual_lab.repair import error_signature
from virtual_lab.utils import BudgetExceededError, CostUnknownError, MeetingUsage, compute_token_cost, write_atomically

RECORDS_DIR = "records"
REPORT_FILE = "report.json"

VERIFIED = "verified"
UNVERIFIED = "unverified"
FAILED = "failed"
STATUSES = (VERIFIED, UNVERIFIED, FAILED)

# Biomni's requirements of the code, as it states them, and then what a tool needs of it
SYSTEM_PROMPT = """You are a senior Python engineer. Generate robust, idiomatic Python code that solves the \
user's task. Requirements:
1. Output ONLY Python code, ideally inside a single triple-backtick code block.
2. Include minimal inline comments and a small docstring.
3. Add a `main()` and an `if __name__ == '__main__':` guard when appropriate.
4. Avoid external dependencies unless necessary; if used, show `pip` installs in comments.
5. Do not include prose before or after the code.
6. When applicable, prioritize the use of codes on public repositories, such as HuggingFace or Github
7. The code will be called as a tool, by a program that sends JSON, so it must define one function named
   `{name}` that does the task. Give every parameter of it a type hint, of one of the types JSON has: str, int,
   float, bool, list, dict, or an Optional, union, or Literal of them. Take the path of a file as a str. Give
   the function a docstring that says what it does, what each parameter is, and what it returns, and have it
   return a value that can be written as JSON.
8. Importing the file must only define things. Anything that runs the task belongs in `main()`, under the
   `if __name__ == '__main__':` guard."""

TASK_PROMPT = "Generate Python codes for the following task:\n{task}"

CORRECTION_PROMPT = """{task_prompt}

This is the code that was written for it:

```python
{source}
```

It cannot be used, for these reasons:
{problems}

Return the whole of the corrected code, as one code block, and nothing else."""

RETRY_PROMPT = """{task_prompt}

Your last reply could not be used: {problems}

Write the code again, as one code block, and nothing else."""

CHECK_FILE = "_virtual_lab_check.py"

# Imports the file under a name of its own, so that it cannot be taken for a library it imports,
# and says if it does not define the function
CHECK_SCRIPT = """\
import importlib.util
import sys

path, name = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("virtual_lab_function", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if not callable(getattr(module, name, None)):
    sys.exit(f"{path} defines no function named {name}")
"""


class FunctionGenerationError(RuntimeError):
    """Raised when a model did not write a function that can be used.

    :param source: The last code it wrote that could be read as Python, if any, so that a person
        can use it, though it did not pass.
    :param problems: What was wrong with it, or with the last reply.
    :param attempts: How many times it was asked.
    :param usage: What the requests used, which has been paid for.
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        problems: Sequence[str] = (),
        attempts: int = 0,
        usage: MeetingUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.problems = tuple(problems)
        self.attempts = attempts
        self.usage = usage


class FunctionUsageUnknownError(FunctionGenerationError):
    """Raised when a response did not say what it used, so that max_cost could no longer be enforced.

    What the model made of the task is not the cause, and so a run does not keep a record of it as
    the task's failure: another run, with a provider that reports its usage, may write the function.
    """


class FunctionBudgetExceededError(BudgetExceededError):
    """Raised before a request that the limit on writing functions leaves no room for.

    :param source: The last code written, which has been paid for, if any.
    :param usage: What the writing used before it stopped.
    """

    def __init__(
        self, spent: float, limit: float, source: str | None = None, usage: MeetingUsage | None = None
    ) -> None:
        super().__init__(spent=spent, limit=limit, what="writing")
        self.source = source
        self.usage = usage


def clean_name(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class FunctionTask:
    """A task to write a function for.

    :param name: What the task is called, which the function and its file are named from.
    :param brief: What the model is told of it: the name alone, or the name and what is known of
        the task, such as what it takes and gives.
    :param papers: In how many papers the task was found, if it came from reading them.
    """

    name: str
    brief: str
    papers: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A task needs a name")
        if not self.brief.strip():
            raise ValueError(f"The task {self.name!r} has nothing to say what it is")
        if len(self.brief) > MAX_FUNCTION_TASK_CHARS:
            raise ValueError(
                f"The task {self.name[:60]!r} is {len(self.brief):,} characters, more than the "
                f"{MAX_FUNCTION_TASK_CHARS:,} a function may be written for"
            )

    @classmethod
    def of(cls, task: "str | FunctionTask") -> "FunctionTask":
        """A task from what was given: a task already, or the words that say it."""
        if isinstance(task, FunctionTask):
            return task
        if not isinstance(task, str):
            raise TypeError(f"A task is a str or a FunctionTask, not {type(task).__name__}")

        return cls(name=clean_name(task), brief=task.strip())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "brief": self.brief, "papers": self.papers}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionTask":
        return cls(name=data["name"], brief=data["brief"], papers=data.get("papers"))


@dataclass(frozen=True)
class GeneratedFunction:
    """A function written for a task, and what came of checking it, as it is saved.

    :param task: What it was written for.
    :param name: What the function, its file, and the tool made of it are called.
    :param status: "verified" if it was also imported in an executor, "unverified" if it was
        read but nothing of it was run, or "failed" if it could not be made to pass.
    :param source: The code. For a function that failed, the last code written that could be
        read as Python, if any, which is not saved as a file.
    :param attempts: How many times the model was asked.
    :param problems: Why a function that failed did, which is what the last reply was refused for.
    :param modules: The modules it imports that are not Python's own, which must be installed.
    :param model: The model that wrote it.
    :param usage: What the requests used.
    :param cost: What they cost, in USD, or None if that cannot be worked out.
    :param error: Why it failed.
    :param elapsed: How long it took, in seconds.
    :param digest: The SHA-256 of the file as it was saved, which tells a function that was
        changed since from one that was not.
    """

    task: FunctionTask
    name: str
    status: str
    source: str | None = None
    attempts: int = 0
    problems: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = 0.0
    error: str | None = None
    elapsed: float = 0.0
    digest: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"A function's status is one of {', '.join(STATUSES)}, not {self.status!r}")

    @property
    def usable(self) -> bool:
        """Whether it passed, and so was saved as a file."""
        return self.status != FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "name": self.name,
            "status": self.status,
            "source": self.source,
            "attempts": self.attempts,
            "problems": list(self.problems),
            "modules": list(self.modules),
            "model": self.model,
            "usage": self.usage,
            "cost": self.cost,
            "error": self.error,
            "elapsed": self.elapsed,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedFunction":
        return cls(
            task=FunctionTask.from_dict(data["task"]),
            name=data["name"],
            status=data["status"],
            source=data.get("source"),
            attempts=data.get("attempts", 0),
            problems=tuple(data.get("problems", ())),
            modules=tuple(data.get("modules", ())),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            cost=data.get("cost", 0.0),
            error=data.get("error"),
            elapsed=data.get("elapsed", 0.0),
            digest=data.get("digest", ""),
        )


def numbered(problems: Sequence[str]) -> str:
    return "\n".join(f"{number}. {problem}" for number, problem in enumerate(problems, 1))


def check_generation_options(
    model: str,
    temperature: float | None,
    max_attempts: int,
    max_completion_tokens: int | None,
    max_cost: float | None,
) -> None:
    """Refuses options that every writing would fail on, so that many tasks are not each tried.

    :raises ValueError: If an option is out of range.
    :raises CostUnknownError: If max_cost is given and the model's price is not known, since a
        limit on spending is only a limit if the price is.
    """
    check_temperature(temperature)
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, not {max_attempts}")
    if max_completion_tokens is not None and max_completion_tokens < 1:
        raise ValueError(f"max_completion_tokens must be at least 1, not {max_completion_tokens}")
    if max_cost is not None:
        if not (math.isfinite(max_cost) and max_cost >= 0):
            raise ValueError(f"max_cost must be a finite amount, zero or more, not {max_cost}")
        try:
            compute_token_cost(model, 0, 0)
        except CostUnknownError as error:
            raise CostUnknownError(
                f"{error}, so a max_cost cannot be enforced. Add its prices to the tables in "
                "virtual_lab.constants, or run without a limit."
            ) from error


def check_function_name(name: str) -> None:
    if not (name.isidentifier() and name.isascii() and name == name.lower() and len(name) <= MAX_FUNCTION_NAME_CHARS):
        raise ValueError(
            f"{name!r} cannot name a function: use lowercase letters, digits, and '_', not beginning with a "
            f"digit, at most {MAX_FUNCTION_NAME_CHARS} of them"
        )


def import_in(executor: Executor, source: str, name: str, timeout: float | None = None) -> ExecutionResult:
    """Imports a function's file in an executor, and checks that it defines the function.

    The file is in a directory of its own, so that it is not on the path of the code that imports it,
    and cannot be mistaken for a library of the same name.
    """
    with tempfile.TemporaryDirectory(prefix="virtual_lab_function_") as directory:
        scratch = Path(directory)
        (scratch / "function").mkdir()
        (scratch / "function" / f"{name}.py").write_text(source, encoding="utf-8")
        (scratch / CHECK_FILE).write_text(CHECK_SCRIPT, encoding="utf-8")

        return executor.run(scratch, ("python3", f"./{CHECK_FILE}", f"function/{name}.py", name), timeout=timeout)


def generate_function(
    task: str | FunctionTask,
    name: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float | None = CONSISTENT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_FUNCTION_ATTEMPTS,
    executor: Executor | None = None,
    timeout: float | None = None,
    max_cost: float | None = None,
    max_completion_tokens: int | None = None,
    chat_models: ModelSource | None = None,
    client: Any = None,
    usage: MeetingUsage | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> GeneratedFunction:
    """Asks a model for a function that does a task, and checks it.

    The model is asked as Biomni's FunctionGenerator asks it, and the reply is checked by
    read_function: that it is Python, defines a function of the task's name that is documented and
    typed so that a model can call it as a tool, and does nothing but define things when imported.
    With an executor the file is imported in it as well. A function that fails is given back to the
    model, with what is wrong, to be corrected, until it passes, or max_attempts have been made, or a
    correction fails for the same reason as the one before it, since a model repeating itself will
    go on doing so.

    :param task: What the function is to do, as words, or a FunctionTask.
    :param name: What to call the function, defaulting to script_name of the task, which is
        Biomni's filename without its extension.
    :param model: The model to ask.
    :param temperature: The sampling temperature, or None for the model's default. Biomni uses 0.7.
    :param max_attempts: The most times to ask, the first writing counted.
    :param executor: What to import the file with, such as a DockerExecutor, or None to read it
        only. Without one nothing the model wrote is run, and the function is "unverified".
    :param timeout: Seconds to allow the import, defaulting to the executor's own.
    :param max_cost: The most to spend, in USD, checked before every request, counting what usage
        already holds, so that one limit can hold across many functions.
    :param max_completion_tokens: The most tokens an answer may use, or None for the model's limit.
    :param chat_models: The chat model to ask, as hold_meeting takes it.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param usage: A count to add what the requests use to as well, so that a caller writing many
        functions keeps one account.
    :param on_progress: Called with a line saying what is being done, before each request.
    :raises ValueError: If an option or the name is out of range, or the task too long.
    :raises FunctionBudgetExceededError: Before a request, if max_cost has been reached.
    :raises FunctionGenerationError: If no usable function was written. It holds the last code
        that could be read as Python.
    :raises FunctionUsageUnknownError: If a response did not report its usage so that max_cost
        could no longer be enforced, which is a FunctionGenerationError that holds the same.
    :return: The function, "verified" if the executor imported it and otherwise "unverified".
    """
    check_generation_options(model, temperature, max_attempts, max_completion_tokens, max_cost)
    task = FunctionTask.of(task)
    name = name if name is not None else script_name(task.name)
    check_function_name(name)

    own = MeetingUsage()
    accounts = [own] if usage is None or usage is own else [own, usage]
    limited = usage if usage is not None else own
    llm = resolve_chat_models([model], chat_models=chat_models, client=client)[model]
    began = time.time()

    def say(line: str) -> None:
        if on_progress is not None:
            on_progress(line)

    def request(system: str, user: str, source: str | None) -> Any:
        try:
            if max_cost is not None and (spent := limited.compute_cost()) >= max_cost:
                raise FunctionBudgetExceededError(spent, max_cost, source=source, usage=own)
        except CostUnknownError as error:
            raise FunctionUsageUnknownError(
                f"A response did not report what it used, so max_cost cannot be enforced: {error}",
                source=source,
                usage=own,
            ) from error

        reply = send_request(
            lambda sent: ask(
                llm,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=sent,
                max_tokens=max_completion_tokens,
            ),
            model=model,
            temperature=temperature,
        )
        for account in accounts:
            account.add(model, reply.usage)

        return reply

    system = SYSTEM_PROMPT.format(name=name)
    task_prompt = TASK_PROMPT.format(task=task.brief)
    # What was last written that could be read as Python, which is kept for a person to use even
    # if it never passed, and what the problems being corrected are of, which is the last reply's
    source: str | None = None
    problems: list[str] = []
    previous: tuple[str, ...] | None = None
    about: str | None = None

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            user = task_prompt
            say(f"Writing {name}")
        else:
            user = (
                CORRECTION_PROMPT.format(task_prompt=task_prompt, source=about, problems=numbered(problems))
                if about is not None
                else RETRY_PROMPT.format(task_prompt=task_prompt, problems=" ".join(problems))
            )
            say(f"  Attempt {attempt} of {max_attempts}, to correct: {' '.join(problems[0].split())[:100]}")

        reply = request(system, user, source)
        code = extract_code(reply.content)
        reading: FunctionReading | None = None
        about = None

        if reply.finish_reason == "length":
            problems = [
                "The reply ran out of tokens before the code was finished. Write less code: only what the task needs."
            ]
        elif code is None:
            problems = ["The reply had no Python code. Reply with the code in one ```python code block."]
        elif not code.complete:
            problems = ["The reply ended before the code block was closed. Write less code, and close the block."]
        else:
            source = about = code.source + "\n"
            try:
                reading = read_function(source, name)
                problems = []
            except FunctionSourceError as error:
                problems = error.problems
        signature = tuple(problems)

        if not problems and executor is not None:
            say("  Importing it in the executor")
            result = import_in(executor, source or "", name, timeout)
            if not result.succeeded:
                problems = [f"Importing the file failed. {result.report()}"]
                signature = ("import", error_signature(result))

        if not problems:
            assert reading is not None and source is not None

            return GeneratedFunction(
                task=task,
                name=name,
                status=VERIFIED if executor is not None else UNVERIFIED,
                source=source,
                attempts=attempt,
                modules=reading.modules,
                model=model,
                usage=own.to_dict(),
                cost=cost_of(own),
                elapsed=round(time.time() - began, 3),
            )

        # The same fault twice running is a model that is not going to fix it
        if signature == previous:
            break
        previous = signature

    raise FunctionGenerationError(
        f"No function that can be used was written in {attempt} {'attempt' if attempt == 1 else 'attempts'}: "
        f"{problems[0]}",
        source=source,
        problems=problems,
        attempts=attempt,
        usage=own,
    )


def cost_of(usage: MeetingUsage) -> float | None:
    try:
        return usage.compute_cost()
    except CostUnknownError:
        return None


def function_names(tasks: Sequence[FunctionTask]) -> list[str]:
    """What each task's function is called: script_name of it, and, where tasks share one, that
    with a few characters of a hash of the task's name, so that none is written over another."""
    bases = [script_name(task.name) for task in tasks]
    shared = {base for base in bases if bases.count(base) > 1}

    return [
        f"{base[: MAX_FUNCTION_NAME_CHARS - 7]}_{hashlib.sha256(task.name.encode()).hexdigest()[:6]}"
        if base in shared
        else base
        for base, task in zip(bases, tasks, strict=True)
    ]


def function_path(save_dir: Path, name: str) -> Path:
    return save_dir / f"{name}.py"


def record_path(save_dir: Path, name: str) -> Path:
    return save_dir / RECORDS_DIR / f"{name}.json"


def load_function(save_dir: Path | str, name: str) -> GeneratedFunction | None:
    """A function's record, as it was saved, or None if there is none or it cannot be read."""
    path = record_path(Path(save_dir), name)
    if not path.is_file():
        return None

    try:
        return GeneratedFunction.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def saved_functions(save_dir: Path | str) -> list[GeneratedFunction]:
    """The record of every function saved in a directory, in the order of their names."""
    records = Path(save_dir) / RECORDS_DIR
    if not records.is_dir():
        return []

    loaded = (load_function(save_dir, path.stem) for path in sorted(records.glob("*.json")))

    return [function for function in loaded if function is not None]


def save_function(save_dir: Path | str, function: GeneratedFunction) -> GeneratedFunction:
    """Saves a function's file, if it passed, and then its record, and returns the function with
    the digest of its file. The file is written first, so that a stop between the two leaves a
    file with no record, which a run refuses to write over, and not a record of another file."""
    save_dir = Path(save_dir)
    if function.usable:
        if function.source is None:
            raise ValueError(f"{function.name} passed, and has no code to save")
        write_atomically(function_path(save_dir, function.name), function.source.encode("utf-8"))
        function = replace(function, digest=file_digest(function_path(save_dir, function.name)))
    save_json(record_path(save_dir, function.name), function.to_dict())

    return function


@dataclass
class FunctionRunReport:
    """What a run over many tasks did.

    :param results: Each task's function that was finished, in the order the tasks were asked for.
    :param tasks: How many tasks were asked for.
    :param spent: What this run spent, in USD, or None if that cannot be worked out. Functions
        already saved cost nothing.
    :param stopped: Why the run stopped before it had gone through every task, if it did.
    :param save_dir: Where it was saved.
    """

    results: list[GeneratedFunction]
    tasks: int
    spent: float | None
    stopped: str | None
    save_dir: Path

    @property
    def verified(self) -> int:
        return sum(result.status == VERIFIED for result in self.results)

    @property
    def unverified(self) -> int:
        return sum(result.status == UNVERIFIED for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.status == FAILED for result in self.results)

    @property
    def cost(self) -> float | None:
        """What the finished functions cost in all, or None if any cost cannot be worked out."""
        costs = [result.cost for result in self.results]

        return None if any(cost is None for cost in costs) else sum(cost for cost in costs if cost is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "finished": len(self.results),
            "verified": self.verified,
            "unverified": self.unverified,
            "failed": self.failed,
            "cost": self.cost,
            "spent": self.spent,
            "stopped": self.stopped,
            "results": [
                {"name": result.name, "status": result.status, "cost": result.cost, "error": result.error}
                for result in self.results
            ],
        }


REUSE = "reuse"
AGAIN = "again"
CONFLICT = "conflict"
VERIFY = "verify"


def resume_decision(
    saved: GeneratedFunction | None,
    task: FunctionTask,
    name: str,
    save_dir: Path,
    retry_failed: bool,
    has_executor: bool,
) -> str:
    """What to do with what is saved under a task's name: REUSE it, write the function AGAIN,
    VERIFY a function that was never run, or refuse it as the CONFLICT of a file that is not this
    run's to write over.

    A record is of the same task if its brief is. A function of another brief is written again, but
    not over a file that has been changed since it was saved, which is someone's work. A file
    with no record of it is not this run's at all.
    """
    path = function_path(save_dir, name)
    if saved is None or saved.task.brief != task.brief:
        wanted = AGAIN
    elif saved.status == FAILED:
        wanted = AGAIN if retry_failed else REUSE
    elif not path.is_file():
        wanted = AGAIN
    else:
        return VERIFY if saved.status == UNVERIFIED and has_executor else REUSE

    if wanted == AGAIN and path.exists() and (saved is None or not saved.usable or file_digest(path) != saved.digest):
        return CONFLICT

    return wanted


def generate_functions(
    tasks: Iterable[str | FunctionTask],
    save_dir: Path | str,
    model: str = DEFAULT_MODEL,
    temperature: float | None = CONSISTENT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_FUNCTION_ATTEMPTS,
    executor: Executor | None = None,
    timeout: float | None = None,
    max_completion_tokens: int | None = None,
    max_cost: float | None = None,
    max_cost_per_function: float | None = None,
    chat_models: ModelSource | None = None,
    client: Any = None,
    usage: MeetingUsage | None = None,
    retry_failed: bool = False,
    max_consecutive_failures: int | None = 3,
    on_progress: Callable[[str], None] | None = None,
) -> FunctionRunReport:
    """Writes a function for each task, and saves those that pass.

    Each function is saved under save_dir as name.py, with a record of the task, the model, what
    it cost, and how it was checked, under save_dir/records/, as soon as it is written. A task
    whose function is already saved is not written again, so a run carries on where an earlier one
    stopped. A function saved without being run, when an executor is now given, is imported in it
    without asking the model again, and is written again only if it fails.

    A task whose function cannot be made to pass is saved as failed, with the last code written
    for it, and the run goes on. Three failures in a row stop it, since that is more likely a
    missing key or no network than three bad tasks. A task stopped by a limit is not saved, and nor
    is one that failed of an error that is not about it, such as a key that is wrong, or a response
    that did not report its usage so that a limit could not be kept, so that the next run writes it.

    :param tasks: The tasks, in the order to write them.
    :param save_dir: Where to save the functions, and where an earlier run of them was saved.
    :param model: The model to write them with.
    :param temperature: The sampling temperature, or None for the model's default.
    :param max_attempts: The most times to ask for each function, the first writing counted.
    :param executor: What to import each function with, or None to read them only.
    :param timeout: Seconds to allow each import.
    :param max_completion_tokens: The most tokens an answer may use.
    :param max_cost: The most the run may spend, in USD, counting what usage already holds.
    :param max_cost_per_function: The most one function may spend. One that reaches it fails, and
        the run goes on.
    :param chat_models: The chat model to ask, as hold_meeting takes it.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param usage: A count to add what the run uses to, so that several runs can keep one account.
    :param retry_failed: Whether to write again the functions an earlier run could not make pass.
    :param max_consecutive_failures: How many tasks in a row may fail before the run stops, or
        None to never stop.
    :param on_progress: Called with a line saying what is being done.
    :raises ValueError: If an option is out of range, two tasks have one name, or save_dir holds a
        file that is not this run's to write over.
    :raises CostUnknownError: If a limit is given and the model's price is not known.
    :raises ExecutionError: If the executor cannot run, such as Docker not being installed.
    :return: What was written for each task, and what it cost.
    """
    check_generation_options(model, temperature, max_attempts, max_completion_tokens, max_cost)
    for label, limit in (("max_cost", max_cost), ("max_cost_per_function", max_cost_per_function)):
        if limit is not None and not (math.isfinite(limit) and limit >= 0):
            raise ValueError(f"{label} must be a finite amount, zero or more, not {limit}")
    if max_consecutive_failures is not None and max_consecutive_failures < 1:
        raise ValueError(f"max_consecutive_failures must be at least 1, not {max_consecutive_failures}")
    if max_cost_per_function is not None:
        try:
            compute_token_cost(model, 0, 0)
        except CostUnknownError as error:
            raise CostUnknownError(f"{error}, so a limit on spending cannot be enforced") from error

    asked = [FunctionTask.of(task) for task in tasks]
    names = function_names(asked)
    if len(set(names)) != len(names):
        raise ValueError("Two tasks have the same name")

    # Checked before anything is written, so that a missing executor fails the run before any
    # function is saved as having failed
    if executor is not None and (check := getattr(executor, "check_available", None)) is not None:
        check()
    llm = resolve_chat_models([model], chat_models=chat_models, client=client)[model]

    save_dir = Path(save_dir)
    run_usage = usage if usage is not None else MeetingUsage()

    def say(line: str) -> None:
        if on_progress is not None:
            on_progress(line)

    # Everything already saved is checked against its task before any is written, so that a run is
    # refused rather than stopped half way, or worse, written over someone's file
    plan: dict[str, tuple[GeneratedFunction | None, str]] = {}
    conflicts: list[str] = []
    for task, name in zip(asked, names, strict=True):
        saved = load_function(save_dir, name)
        decision = resume_decision(saved, task, name, save_dir, retry_failed, executor is not None)
        if decision == CONFLICT:
            conflicts.append(f"{name}.py")
        plan[name] = (saved, decision)
    if conflicts:
        shown = ", ".join(conflicts[:5]) + (f", and {len(conflicts) - 5} more" if len(conflicts) > 5 else "")
        raise ValueError(
            f"{save_dir} holds {shown}, which this run did not write or has since been changed. Save this "
            "run somewhere else, or move them."
        )

    started_with = cost_of(run_usage)
    results: dict[str, GeneratedFunction] = {}
    stopped: str | None = None
    failures_in_a_row = 0

    # Whatever ends the loop, even Ctrl-C, the report is written, so that what was written is counted
    try:
        for number, (task, name) in enumerate(zip(asked, names, strict=True), 1):
            saved, decision = plan[name]
            if saved is not None and decision == REUSE:
                results[name] = saved
                continue

            if decision == VERIFY and saved is not None and executor is not None:
                say(f"Function {number} of {len(asked)}: {name}, which was saved without being run")
                if (verified := import_saved(saved, save_dir, executor, timeout, say)) is not None:
                    results[name] = verified
                    continue

            limits: list[float] = []
            if max_cost is not None or max_cost_per_function is not None:
                so_far = cost_of(run_usage)
                if so_far is None:
                    stopped = "What the run has spent cannot be worked out, so its limits cannot be enforced"
                    break
                if max_cost is not None:
                    if so_far >= max_cost:
                        stopped = f"The run spent ${so_far:.4f}, which reaches its max_cost of ${max_cost:.4f}"
                        break
                    limits.append(max_cost)
                if max_cost_per_function is not None:
                    limits.append(so_far + max_cost_per_function)

            say(f"Function {number} of {len(asked)}: {name}")
            began = time.time()
            remember = True
            try:
                written = generate_function(
                    task,
                    name=name,
                    model=model,
                    temperature=temperature,
                    max_attempts=max_attempts,
                    executor=executor,
                    timeout=timeout,
                    max_cost=min(limits) if limits else None,
                    max_completion_tokens=max_completion_tokens,
                    chat_models={model: llm},
                    usage=run_usage,
                    on_progress=lambda line: say(f"  {line.lstrip()}"),
                )
            except FunctionBudgetExceededError as error:
                # Either the run's limit or this function's, and only the run's stops the run
                if max_cost is not None and error.spent >= max_cost:
                    stopped = (
                        f"The run spent ${error.spent:.4f}, which reaches its max_cost of ${max_cost:.4f}, "
                        f"while writing {name}, which is left unfinished"
                    )
                    break
                written = failed_function(task, name, model, error, "reached its limit on spending")
            except FunctionUsageUnknownError as error:
                # What the provider reported, not what the model made of the task
                remember = False
                written = failed_function(task, name, model, error)
            except FunctionGenerationError as error:
                written = failed_function(task, name, model, error)
            except Exception as error:
                # Not what the model made of the task: a wrong key or a network that is down fails
                # every task, and a record of each would be reused as its answer by the next run
                remember = False
                written = GeneratedFunction(
                    task=task,
                    name=name,
                    status=FAILED,
                    model=model,
                    error=f"{type(error).__name__}: {error}",
                )

            written = replace(written, elapsed=round(time.time() - began, 3))
            if remember:
                written = save_function(save_dir, written)
            results[name] = written

            if written.status == FAILED:
                failures_in_a_row += 1
            else:
                failures_in_a_row = 0
            if max_consecutive_failures is not None and failures_in_a_row >= max_consecutive_failures:
                stopped = f"{failures_in_a_row} functions in a row failed, the last with {written.error}"
                break
    finally:
        ended_with = cost_of(run_usage)
        report = FunctionRunReport(
            results=[results[name] for name in names if name in results],
            tasks=len(asked),
            spent=ended_with - started_with if ended_with is not None and started_with is not None else None,
            stopped=stopped,
            save_dir=save_dir,
        )
        save_json(save_dir / REPORT_FILE, report.to_dict())

    return report


def import_saved(
    saved: GeneratedFunction,
    save_dir: Path,
    executor: Executor,
    timeout: float | None,
    say: Callable[[str], None],
) -> GeneratedFunction | None:
    """Imports a function that was saved without being run, in an executor, and marks it verified
    if it does. None if it does not, and so is to be written again, which it is not if its file has
    been changed since it was saved, which is left as it is."""
    path = function_path(save_dir, saved.name)
    problem: str | None = None
    try:
        source = path.read_text(encoding="utf-8")
        read_function(source, saved.name)
    except (OSError, ValueError) as error:
        problem = str(error)
    else:
        result = import_in(executor, source, saved.name, timeout)
        if result.succeeded:
            return save_function(save_dir, replace(saved, status=VERIFIED, source=source))
        problem = result.report()

    if file_digest(path) != saved.digest:
        say(f"  It has been changed since it was saved, and does not pass, so it is left as it is: {problem[:100]}")
        return saved

    return None


def failed_function(
    task: FunctionTask,
    name: str,
    model: str,
    error: FunctionGenerationError | FunctionBudgetExceededError,
    why: str = "",
) -> GeneratedFunction:
    """A function that failed, with what was written for it and what that used, which has been paid for."""
    message = f"{type(error).__name__}: {error}"

    return GeneratedFunction(
        task=task,
        name=name,
        status=FAILED,
        source=error.source,
        attempts=getattr(error, "attempts", 0),
        problems=tuple(getattr(error, "problems", ())),
        model=model,
        usage=error.usage.to_dict() if error.usage is not None else {},
        cost=cost_of(error.usage) if error.usage is not None else 0.0,
        error=f"{why}: {message}" if why else message,
    )


def paper_task_details(directory: Path) -> dict[str, PaperTask]:
    """What the papers read in a directory, and in its subdirectories, say of each task, by the
    task's name as normalize_name has it: the account with the most in it, where several have it."""
    directories = [directory, *(path for path in sorted(directory.iterdir()) if path.is_dir())]
    found: dict[str, PaperTask] = {}
    for where in directories:
        for result in saved_results(where):
            if result.findings is None:
                continue
            for task in result.findings.tasks:
                key = normalize_name(task.task_name)
                if key and (key not in found or detail_size(task) > detail_size(found[key])):
                    found[key] = task

    return found


def detail_size(task: PaperTask) -> int:
    return len(task.description) + len(task.inputs) + len(task.outputs) + len(task.code_implementation)


BRIEF_FIELDS = (
    ("What it does", "description"),
    ("What it takes", "inputs"),
    ("What it gives", "outputs"),
    ("How it is implemented", "code_implementation"),
    ("Standard methods", "standard_methods"),
)


def brief_of(name: str, details: Mapping[str, Any]) -> str:
    """What a model is told of a task that is more than a name: what it does, takes, and gives, and how."""
    lines = [clean_name(name)]
    for label, key in BRIEF_FIELDS:
        text = str(details.get(key) or "").strip()
        if text:
            lines.append(f"{label}: {text}")

    return "\n".join(lines)


def function_tasks(
    source: Path | str | Mapping[str, Any] | Sequence[Any],
    min_papers: int = 1,
    limit: int | None = None,
) -> list[FunctionTask]:
    """The tasks to write functions for, from the files that Biomni's pipeline, and this one, make.

    A JSON file or object may be:

    - {"tasks": ["a description", ...]}, as Biomni's generate_function.py reads, or a list of them,
    - {"tasks": {"task name": 12, ...}}, as the frequency_summary.json and combined_summary.json that
      summarize_papers and combine_paper_summaries write are, whose tasks are taken in the order
      given, with the number of papers each was found in, or
    - any of those with objects in place of descriptions, each with a "task_name", or "name", and
      what it does, takes, and gives, as "description", "inputs", and "outputs".

    A directory is one that papers were read into: the tasks of its combined_summary.json, or its
    frequency_summary.json, are taken, and what the papers it holds say of each is added to what the
    model is told, which Biomni's file of names alone does not give it.

    :param source: A path, or what a JSON file holds.
    :param min_papers: Take only the tasks found in at least this many papers. A task whose count is not
        known is not left out.
    :param limit: Take at most this many tasks, the first, after leaving out the rest.
    :raises ValueError: If the path is neither a file nor a directory, a file is not JSON of those
        shapes, or a directory holds no summary.
    :return: The tasks, each once, by its name as normalize_name has it, the first found kept.
    """
    details: dict[str, PaperTask] = {}
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_dir():
            summary = next((path / file for file in (COMBINED_FILE, FREQUENCY_FILE) if (path / file).is_file()), None)
            if summary is None:
                raise ValueError(
                    f"{path} has no {FREQUENCY_FILE} or {COMBINED_FILE}. Read papers into it, or summarize it, "
                    "with virtual-lab-papers"
                )
            details = paper_task_details(path)
            path = summary
        if not path.is_file():
            raise ValueError(f"{source} is neither a file nor a directory")
        try:
            source = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{path} cannot be read: {error}") from error

    if isinstance(source, Mapping):
        if "tasks" not in source:
            raise ValueError('The tasks are under "tasks", as in {"tasks": ["a description", ...]}')
        source = source["tasks"]

    # A mapping is of names to the number of papers each was found in, as a summary's is
    items: list[tuple[Any, int | None]]
    if isinstance(source, Mapping):
        items = [
            (name, count if isinstance(count, int) and not isinstance(count, bool) else None)
            for name, count in source.items()
        ]
    elif isinstance(source, Sequence) and not isinstance(source, str | bytes):
        items = [(item, None) for item in source]
    else:
        raise ValueError("The tasks are a list, or a mapping of names to counts")

    found = [task for item, count in items if (task := task_from(item, count, details)) is not None]
    tasks = [
        task
        for task in unique_tasks(found)
        if not (min_papers > 1 and task.papers is not None and task.papers < min_papers)
    ]

    return tasks[: max(limit, 0)] if limit is not None else tasks


def unique_tasks(tasks: Iterable[FunctionTask]) -> list[FunctionTask]:
    """The tasks, each name once, by the name as normalize_name has it, the first of them kept."""
    seen: set[str] = set()
    unique: list[FunctionTask] = []
    for task in tasks:
        if (key := normalize_name(task.name)) not in seen:
            seen.add(key)
            unique.append(task)

    return unique


def task_from(item: Any, count: int | None, details: Mapping[str, PaperTask]) -> FunctionTask | None:
    """A task from one entry of a file: a description, or an object, or None if it says nothing."""
    if isinstance(item, str):
        name = clean_name(item)
        if not name:
            return None
        found = details.get(normalize_name(name))

        brief = brief_of(found.task_name, found.model_dump()) if found is not None else item.strip()

        return FunctionTask(name=name, brief=brief, papers=count)

    if isinstance(item, Mapping):
        name = clean_name(str(item.get("task_name") or item.get("name") or ""))
        if not name:
            raise ValueError(f"A task that is an object needs a task_name or a name: {json.dumps(item)[:100]}")

        return FunctionTask(name=name, brief=brief_of(name, item), papers=count)

    raise ValueError(f"A task is a description or an object, not {type(item).__name__}")
