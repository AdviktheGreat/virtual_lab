"""Reading a paper for the computational tasks, databases, and software that are common in
biomedical research, as Biomni's PaperTaskExtractor does (snap-stanford/Biomni,
biomni/agent/env_collection.py, Apache License 2.0).

Biomni grew its tools from the literature: papers were read for the tasks that recur across
many of them, and each task became a function. The reading is a filter, and a hard one. A model
asked what a paper did will list everything the paper did, which is the paper's own methods and
no use to anyone else, so every request here is told to return nothing rather than something
specific to the paper, and a task is kept only if it has a concrete name, inputs, outputs, and a
way to implement it in code.

A paper is cut into chunks, each is read for its findings, and the findings of all of them are
merged and then consolidated once more, so that the filter is applied to the paper as a whole
and not only to each of its pieces. Where Biomni asks for JSON and parses it out of the reply,
falling back to a record of the failure, every request here is made against a schema, so what
comes back has the fields it should or is refused.
"""

import io
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from virtual_lab.constants import (
    CONSISTENT_TEMPERATURE,
    DEFAULT_MODEL,
    DEFAULT_PAPER_CHUNK_OVERLAP,
    DEFAULT_PAPER_CHUNK_SIZE,
    MAX_CONSOLIDATION_CHARS,
    MAX_CONSOLIDATION_PASSES,
    MAX_PAPER_CHARS,
    MAX_PAPER_FILE_BYTES,
    MAX_PAPER_PAGES,
)
from virtual_lab.llm import ModelSource, resolve_chat_models
from virtual_lab.structured import StructuredOutputError, request_structured_output
from virtual_lab.utils import BudgetExceededError, CostUnknownError, MeetingUsage, compute_token_cost

# Where a paper is cut, from the coarsest to the finest. A cut falls on the first of these that
# the text has, and a piece still too long is cut on the next, and at last between characters
SEPARATORS = ("\n\n", "\n", ". ", " ")

TEXT_SUFFIXES = (".txt", ".text", ".md", ".markdown", ".tex")

# How many characters of a failure to keep, for the record of a chunk that could not be read
MAX_FAILURE_CHARS = 300


class PaperTask(BaseModel):
    """A computational task that recurs across biomedical research, as a paper shows it."""

    task_name: str = Field(
        description="A specific, concrete name that includes the method, such as 'RNA-seq differential "
        "expression analysis with DESeq2' or 'Two-way ANOVA with Tukey's post-hoc test using SciPy', "
        "never a vague one such as 'statistical analysis'."
    )
    description: str = Field(description="What the computational task does, in a sentence or two.")
    inputs: str = Field(description="The specific data types or parameters the task needs.")
    outputs: str = Field(description="The specific data types or results the task produces.")
    code_implementation: str = Field(
        description="How the task is implemented in Python or Linux code: the key libraries or "
        "programs, and brief pseudocode."
    )
    frequency: str = Field(description="How common the task is in biomedical research.")
    standard_methods: str = Field(description="The established computational techniques that perform it.")
    example: str = Field(description="How this paper uses the task, with concrete details from it.")


class PaperDatabase(BaseModel):
    """A database that computational biomedical research commonly draws on."""

    name: str = Field(description="The database's name.")
    description: str = Field(description="What the database contains.")
    url: str = Field(description="Its URL if the paper gives one, otherwise an empty string.")
    usage: str = Field(description="How it is commonly used in computational biomedical research.")
    example: str = Field(description="How this paper uses it.")


class PaperSoftware(BaseModel):
    """A software package that computational biomedical research commonly uses."""

    name: str = Field(description="The package's name.")
    description: str = Field(description="What the software does.")
    url: str = Field(description="Its URL or reference if the paper gives one, otherwise an empty string.")
    usage: str = Field(description="How it is commonly used in computational biomedical research.")
    example: str = Field(description="How this paper uses it.")


class PaperFindings(BaseModel):
    """What a paper, or a part of one, shows to be common: its tasks, databases, and software.

    No field has a default, because strict structured output makes every field required.
    """

    tasks: list[PaperTask] = Field(description="The common computational tasks, or an empty list if there are none.")
    databases: list[PaperDatabase] = Field(description="The commonly used databases the text mentions.")
    software: list[PaperSoftware] = Field(description="The commonly used software packages the text mentions.")

    def is_empty(self) -> bool:
        return not (self.tasks or self.databases or self.software)


class PaperReadingError(RuntimeError):
    """Raised when a paper could not be read.

    :param findings: What the chunks showed, merged but not consolidated, if the failure came
        after they were read, since a person may want them even so. They are not filtered.
    :param usage: What the reading used before it failed.
    """

    def __init__(self, message: str, findings: PaperFindings | None = None, usage: MeetingUsage | None = None) -> None:
        super().__init__(message)
        self.findings = findings
        self.usage = usage


class PaperBudgetExceededError(BudgetExceededError):
    """Raised before a request that the limit on reading papers leaves no room for."""

    def __init__(self, spent: float, limit: float) -> None:
        super().__init__(spent=spent, limit=limit, what="reading")


@dataclass
class PaperReading:
    """What reading a paper found, and what it took.

    :param findings: The consolidated tasks, databases, and software.
    :param chunks: How many chunks the paper was cut into.
    :param characters: How much of the paper's text was read.
    :param truncated_from: The paper's length if it was cut short to max_chars, otherwise None.
    :param failed_chunks: Why each chunk that could not be read could not, by its number from 1.
        Those chunks add nothing to the findings.
    :param model: The model that read it.
    :param usage: What the requests used.
    """

    findings: PaperFindings
    chunks: int
    characters: int
    truncated_from: int | None
    failed_chunks: dict[int, str]
    model: str
    usage: MeetingUsage = field(default_factory=MeetingUsage)

    @property
    def cost(self) -> float | None:
        """What the reading cost in USD, or None if that cannot be worked out."""
        try:
            return self.usage.compute_cost()
        except CostUnknownError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": self.findings.model_dump(mode="json"),
            "chunks": self.chunks,
            "characters": self.characters,
            "truncated_from": self.truncated_from,
            "failed_chunks": {str(number): why for number, why in self.failed_chunks.items()},
            "model": self.model,
            "usage": self.usage.to_dict(),
        }


def pieces_of(text: str, separator: str) -> list[str]:
    """The text cut after each separator, so that putting the pieces together gives the text."""
    if not separator:
        return list(text)

    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])

    return pieces


def join_pieces(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Joins pieces, each no longer than size, into chunks no longer than size, in which a chunk
    begins with up to overlap characters of the one before."""
    chunks: list[str] = []
    current: list[str] = []
    total = 0

    for piece in pieces:
        if current and total + len(piece) > size:
            chunks.append("".join(current))
            while current and (total > overlap or total + len(piece) > size):
                total -= len(current.pop(0))

        current.append(piece)
        total += len(piece)

    if current:
        chunks.append("".join(current))

    return chunks


def cut(text: str, separators: tuple[str, ...], size: int, overlap: int) -> list[str]:
    separator = next((candidate for candidate in separators if candidate in text), "")
    finer = separators[separators.index(separator) + 1 :] if separator else ()
    chunks: list[str] = []
    run: list[str] = []

    for piece in pieces_of(text, separator):
        if len(piece) <= size:
            run.append(piece)
            continue

        chunks += join_pieces(run, size, overlap)
        run = []
        chunks += cut(piece, finer, size, overlap)

    return chunks + join_pieces(run, size, overlap)


def split_text(
    text: str,
    chunk_size: int = DEFAULT_PAPER_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_PAPER_CHUNK_OVERLAP,
) -> list[str]:
    """Cuts a text into chunks at the coarsest breaks it has: paragraphs, then lines, sentences,
    and words, as Biomni cuts a paper with LangChain's recursive splitter.

    A cut keeps what it cut at, so a sentence keeps its full stop, and every chunk is the text
    itself, with the spaces at its ends removed, not a rewrite of it.

    :param text: The text to cut.
    :param chunk_size: The most characters in a chunk.
    :param chunk_overlap: How many characters at the end of a chunk are repeated at the start of
        the next, as far as the text breaks there, so that nothing described across a cut is
        seen only in halves.
    :raises ValueError: If chunk_size is not above zero, or chunk_overlap is not from zero up to
        chunk_size.
    :return: The chunks, in order, none of them empty.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be above zero, not {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(f"chunk_overlap must be from zero up to chunk_size ({chunk_size}), not {chunk_overlap}")

    chunks = (chunk.strip() for chunk in cut(text, SEPARATORS, chunk_size, chunk_overlap))

    return [chunk for chunk in chunks if chunk]


def truncate_text(text: str, max_chars: int) -> str:
    """Cuts a text to at most max_chars, at the last end of a sentence, or failing that the last
    space, so that it does not stop in the middle of a word or a number."""
    if len(text) <= max_chars:
        return text

    head = text[:max_chars]
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=\s)", head)]
    if sentence_ends:
        return head[: sentence_ends[-1]]

    last_space = max(head.rfind(" "), head.rfind("\n"))

    return head[:last_space] if last_space > 0 else head


def normalize_name(name: str) -> str:
    """A name as it is compared, so that 'DESeq2', 'deseq2 ', and 'Deseq-2' are not three."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def fill_blanks(kept: BaseModel, other: BaseModel) -> BaseModel:
    """The item kept, with each field it leaves empty taken from another of the same name."""
    gaps = {name: value for name, value in other.model_dump().items() if value and not getattr(kept, name)}

    return kept.model_copy(update=gaps) if gaps else kept


def merge_findings(findings: Iterable[PaperFindings]) -> PaperFindings:
    """Puts findings together, one task, database, or package to each name, the first as it was
    written and with what it left empty filled in from the rest. An item with no name is dropped,
    since nothing could be done with it."""
    tasks: dict[str, PaperTask] = {}
    databases: dict[str, PaperDatabase] = {}
    software: dict[str, PaperSoftware] = {}

    def keep(table: dict[str, Any], name: str, item: Any) -> None:
        key = normalize_name(name)
        if not key:
            return
        table[key] = fill_blanks(table[key], item) if key in table else item

    for found in findings:
        for task in found.tasks:
            keep(tasks, task.task_name, task)
        for database in found.databases:
            keep(databases, database.name, database)
        for package in found.software:
            keep(software, package.name, package)

    return PaperFindings(
        tasks=list(tasks.values()), databases=list(databases.values()), software=list(software.values())
    )


def size_of(item: BaseModel) -> int:
    return len(item.model_dump_json())


def in_batches(findings: PaperFindings, limit: int) -> list[PaperFindings]:
    """The findings in batches that are each about limit characters, or one batch if they fit.

    An item is never cut up, so one larger than the limit is a batch of its own.
    """
    batches: list[PaperFindings] = []
    current: dict[str, list[Any]] = {"tasks": [], "databases": [], "software": []}
    total = 0

    def close() -> None:
        nonlocal current, total
        batches.append(PaperFindings(**current))
        current, total = {"tasks": [], "databases": [], "software": []}, 0

    for kind in ("tasks", "databases", "software"):
        for item in getattr(findings, kind):
            length = size_of(item)
            if total and total + length > limit:
                close()
            current[kind].append(item)
            total += length

    close()

    return batches


CHUNK_PROMPT = "\n".join(
    [
        "You are a research methodology expert who identifies the computational tasks and data analysis procedures "
        "in academic papers.",
        "",
        "You are given one chunk of a paper. Identify ONLY the most common, generalizable computational tasks in "
        "it: those that are standard across biomedical research and can be implemented with Python or Linux code.",
        "",
        "Strict guidelines:",
        "1. Extract only tasks that are extremely common and standard in computational biomedical research, the "
        "kind that appear in hundreds of papers.",
        "2. Each task must have clear, well-defined inputs and outputs, and be something that could be implemented "
        "as a function.",
        "3. A task must generalize across many papers and research questions. If it is specific to this paper, "
        "unclear, or not widely used, leave it out.",
        "4. A task must be concrete and specific, with its exact methodological details. Do not name a task "
        "'Statistical analysis'; name it 'Two-way ANOVA with Tukey's post-hoc test using SciPy'.",
        "5. Leave out wet lab procedures and anything that cannot be automated with code.",
        "6. Also list the commonly used databases and software packages the text mentions.",
        "",
        "It is better to return no tasks than to include one that is not extremely common, generalizable, and "
        "implementable with code. Quality over quantity.",
    ]
)

CONSOLIDATION_PROMPT = "\n".join(
    [
        "You are a research methodology expert. You are given the computational research tasks, databases, and "
        "software that were extracted from different chunks of one academic paper. Consolidate them into one list.",
        "",
        "Be extremely selective. Keep a task only if it is:",
        "1. Fundamental to computational biomedical research, and used in hundreds of papers across different "
        "subfields.",
        "2. Defined by clear inputs and outputs, as a standard computational approach that could be implemented as "
        "a function with Python or Linux code.",
        "3. Concrete and specific, with exact methodological details in its name.",
        "",
        "Remove any task that is specific to a particular paper or dataset, lacks clear inputs or outputs, is niche "
        "or specialized, has a generic name without methodological details, cannot be implemented with code, or "
        "needs physical equipment or manual work. Merge tasks, databases, and software that are the same thing "
        "under different names into one, keeping the most specific name and the clearest example.",
        "",
        "It is better to return a few truly common computational tasks than many that are not universal or "
        "implementable with code.",
    ]
)


def read_paper(path: Path | str) -> str:
    """The text of a paper in a file: a PDF, or plain text or Markdown.

    A PDF needs pypdf, which `pip install "virtual-lab[papers]"` installs. Its text is as pypdf
    finds it, so a two-column paper's columns, and its figures' labels, may be mixed in with the
    prose, which is as true of what Biomni reads.

    :param path: The file.
    :raises FileNotFoundError: If there is no such file.
    :raises ValueError: If the file is not a type this reads, is too large, or is a PDF that
        cannot be read, or that is encrypted or has more than MAX_PAPER_PAGES pages.
    :raises ImportError: If it is a PDF and pypdf is not installed.
    :return: The text.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if not path.is_file():
        raise FileNotFoundError(f"There is no paper at {path}")
    if suffix != ".pdf" and suffix not in TEXT_SUFFIXES:
        raise ValueError(f"{path.name} is not a paper this reads: give a PDF, or a {', '.join(TEXT_SUFFIXES)} file")
    if (size := path.stat().st_size) > MAX_PAPER_FILE_BYTES:
        raise ValueError(f"{path.name} is {size:,} bytes, more than the {MAX_PAPER_FILE_BYTES:,} a paper may be")

    data = path.read_bytes()
    if suffix == ".pdf":
        return pdf_text(data, path.name)

    return data.decode("utf-8", errors="replace")


def pdf_text(data: bytes, name: str = "the PDF") -> str:
    """The text of a PDF, page by page, with a blank line between pages."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError('Reading a PDF needs pypdf: pip install "virtual-lab[papers]"') from error

    # pypdf raises whatever the damage in a file leads it to, from its own errors to a KeyError
    # or a ZeroDivisionError, so a file that cannot be read is told by the failing, not the type
    try:
        reader = PdfReader(io.BytesIO(data))
        encrypted = reader.is_encrypted
    except Exception as error:
        raise ValueError(f"{name} could not be read as a PDF: {error}") from error
    if encrypted:
        raise ValueError(f"{name} is encrypted, so its text cannot be read")

    try:
        pages = len(reader.pages)
    except Exception as error:
        raise ValueError(f"{name} could not be read as a PDF: {error}") from error
    if pages > MAX_PAPER_PAGES:
        raise ValueError(f"{name} has {pages:,} pages, more than the {MAX_PAPER_PAGES} a paper may have")

    try:
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise ValueError(f"{name} could not be read as a PDF: {error}") from error


def extract_paper_findings(
    text: str,
    model: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_PAPER_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_PAPER_CHUNK_OVERLAP,
    max_chars: int | None = MAX_PAPER_CHARS,
    max_consolidation_chars: int = MAX_CONSOLIDATION_CHARS,
    temperature: float | None = CONSISTENT_TEMPERATURE,
    max_cost: float | None = None,
    max_completion_tokens: int | None = None,
    chat_models: ModelSource | None = None,
    client: Any = None,
    usage: MeetingUsage | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> PaperReading:
    """Reads a paper for the common computational tasks, databases, and software in it.

    The text is cut into chunks, each is read, and the findings of all of them are merged, one
    to each name, and consolidated by one more request that applies the same filter to the
    paper as a whole. If the findings are too many for one request they are consolidated in
    batches, and then those results again.

    A chunk the model cannot answer, because it refuses or its answer does not fit the schema,
    is left out and recorded in failed_chunks, so that one chunk does not cost the paper; if
    every chunk fails, the paper is refused.

    :param text: The paper's text, as read_paper gives it.
    :param model: The model to read it with.
    :param chunk_size: The most characters in a chunk.
    :param chunk_overlap: The characters repeated between one chunk and the next.
    :param max_chars: The most of the text to read, cut at a sentence, or None for all of it.
    :param max_consolidation_chars: The most of the findings one consolidation request is given.
    :param temperature: The sampling temperature, or None for the model's default.
    :param max_cost: The most to spend, in USD, checked before every request, or None. It counts
        what usage holds, if given, so that one limit can hold across many papers.
    :param max_completion_tokens: The most tokens an answer may use, or None for no limit.
    :param chat_models: The chat model to ask, as hold_meeting takes it.
    :param client: An OpenAI client, as hold_meeting takes it.
    :param usage: A count to add what the requests use to as well, so that a caller reading many
        papers keeps one account. The reading's own usage is always kept.
    :param on_progress: Called with a line saying what is being done, before each request.
    :raises ValueError: If the text is empty, or a size is out of range, or the model is not
        priced and max_cost is given.
    :raises PaperBudgetExceededError: Before a request, if max_cost has been reached.
    :raises CostUnknownError: If max_cost is given and a response's usage was not reported.
    :raises PaperReadingError: If no chunk could be read, or the consolidation failed, in which
        case the error holds the merged findings.
    :return: What was found, and what it took.
    """
    if max_chars is not None and max_chars < 1:
        raise ValueError(f"max_chars must be above zero, not {max_chars}")
    if max_consolidation_chars < 1_000:
        raise ValueError(f"max_consolidation_chars must be at least 1,000, not {max_consolidation_chars}")
    if max_cost is not None:
        if not (math.isfinite(max_cost) and max_cost >= 0):
            raise ValueError(f"max_cost must be a finite amount, zero or more, not {max_cost}")
        # A limit on spending is only a limit if the model's price is known
        try:
            compute_token_cost(model, 0, 0)
        except CostUnknownError as error:
            raise CostUnknownError(
                f"{error}, so a max_cost cannot be enforced. Add its prices to the tables in "
                "virtual_lab.constants, or run without a limit."
            ) from error
    if not text.strip():
        raise ValueError("The paper has no text to read")

    truncated_from = len(text) if max_chars is not None and len(text) > max_chars else None
    if max_chars is not None:
        text = truncate_text(text, max_chars)
    chunks = split_text(text, chunk_size, chunk_overlap)

    own = MeetingUsage()
    accounts = [own] if usage is None or usage is own else [own, usage]
    limited = usage if usage is not None else own
    llm = resolve_chat_models([model], chat_models=chat_models, client=client)[model]

    def say(line: str) -> None:
        if on_progress is not None:
            on_progress(line)

    def count(reported: Any) -> None:
        for account in accounts:
            account.add(model, reported)

    def ask(system: str, user: str) -> PaperFindings:
        if max_cost is not None and (spent := limited.compute_cost()) >= max_cost:
            raise PaperBudgetExceededError(spent=spent, limit=max_cost)

        try:
            found, reply = request_structured_output(
                llm=llm,
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                schema=PaperFindings,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
        except StructuredOutputError as error:
            count(error.usage)
            raise

        count(reply.usage)

        return found

    read: list[PaperFindings] = []
    failed: dict[int, str] = {}
    for number, chunk in enumerate(chunks, 1):
        say(f"Reading chunk {number} of {len(chunks)}")
        try:
            read.append(
                ask(CHUNK_PROMPT, f"This is chunk {number} of {len(chunks)} of the paper.\n\nPAPER CHUNK:\n{chunk}")
            )
        except StructuredOutputError as error:
            failed[number] = str(error)[:MAX_FAILURE_CHARS]

    if len(failed) == len(chunks):
        raise PaperReadingError(
            f"None of the {len(chunks)} chunks could be read. The last: {failed[len(chunks)]}", usage=own
        )

    def consolidate(findings: PaperFindings) -> PaperFindings:
        for _ in range(MAX_CONSOLIDATION_PASSES):
            batches = in_batches(findings, max_consolidation_chars)
            say(f"Consolidating the findings in {len(batches)} {'request' if len(batches) == 1 else 'requests'}")
            consolidated = merge_findings(
                ask(CONSOLIDATION_PROMPT, f"EXTRACTED INFORMATION FROM PAPER CHUNKS:\n{batch.model_dump_json()}")
                for batch in batches
            )
            if len(batches) == 1 or size_of(consolidated) >= size_of(findings):
                return consolidated
            findings = consolidated

        return findings

    merged = merge_findings(read)
    if merged.is_empty():
        findings = merged
    else:
        try:
            findings = consolidate(merged)
        except StructuredOutputError as error:
            raise PaperReadingError(
                f"The findings of the chunks could not be consolidated: {error}", findings=merged, usage=own
            ) from error

    return PaperReading(
        findings=findings,
        chunks=len(chunks),
        characters=len(text),
        truncated_from=truncated_from,
        failed_chunks=failed,
        model=model,
        usage=own,
    )
