"""Biomni's benchmarks: Biomni-Eval1, LAB-Bench, and Humanity's Last Exam.

Each benchmark is a list of questions, each with its prompt and its answer, and the rules
Biomni scores answers by. Their files are fetched once with download_benchmarks and checked
against the hashes in BENCHMARK_FILES every time they are read, so that two scores made on
"the same benchmark" were made on the same questions.

The prompts, the shuffling of LAB-Bench's options, the scoring rules, the metrics, and the output
classes are adapted from Biomni (https://github.com/snap-stanford/Biomni, commit
400c1f366b96a35ca253e13c9b06c5076af41d65: biomni/eval/biomni_eval1.py, biomni/task/lab_bench.py,
and biomni/task/hle.py), Copyright the Biomni authors, used under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Where they depart from Biomni, a comment says how.
BiomniEval1, and the get_example, get_iterator, evaluate, and output_class methods, keep Biomni's
names, so that code written against Biomni's classes runs against these.

Reading the files needs pyarrow, and shuffling LAB-Bench's options as Biomni does needs numpy:
pip install "virtual-lab[eval]".
"""

import ast
import hashlib
import importlib
import json
import os
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from virtual_lab.constants import (
    BENCHMARK_FILES,
    BIOMNI_BENCHMARK_ARCHIVE,
    BIOMNI_BENCHMARK_ARCHIVE_SHA256,
    BIOMNI_EVAL1_FILE,
    BIOMNI_EVAL1_URL,
    BIOMNI_RELEASE_URL,
)
from virtual_lab.environment import fetch_file


class BenchmarkError(ValueError):
    """Raised when a benchmark file is not the one its scores would be compared against."""


def optional_module(name: str) -> Any:
    """Imports a module that only the benchmarks need, saying how to install it if missing."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            f'The benchmarks need {name.split(".")[0]}. Install it with: pip install "virtual-lab[eval]"'
        ) from error


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, in hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024**2), b""):
            digest.update(block)

    return digest.hexdigest()


def is_verified(path: Path, name: str) -> bool:
    """Whether a benchmark file is present and is the one pinned in BENCHMARK_FILES."""
    return path.is_file() and sha256_of(path) == BENCHMARK_FILES[name]


@dataclass
class BenchmarkDownload:
    """What happened when the benchmarks were fetched.

    :param directory: Where the files are.
    :param downloaded: The files fetched this time.
    :param present: The files that were already there, as pinned, and were left alone.
    """

    directory: Path
    downloaded: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)


def write_atomically(path: Path, data: bytes) -> None:
    """Writes a file under a temporary name and renames it into place, so it is never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def download_benchmarks(
    directory: Path, base_url: str = BIOMNI_RELEASE_URL, eval1_url: str = BIOMNI_EVAL1_URL
) -> BenchmarkDownload:
    """Fetches the files of Biomni's benchmarks, about 4 MB, into a directory.

    Biomni-Eval1 comes from Hugging Face, at a pinned revision, and LAB-Bench and Humanity's Last
    Exam from the archive Biomni downloads into its benchmark directory. A file already present
    and as pinned is left alone, and every file is checked against BENCHMARK_FILES before it is
    kept, so a file that is present is never partial or changed.

    :param directory: Where to put the files.
    :param base_url: Where Biomni publishes its release files.
    :param eval1_url: Where Biomni-Eval1 is published.
    :raises BenchmarkError: If a file fetched is not the one pinned, as when its publisher has
        changed it.
    :raises requests.RequestException: If a file cannot be fetched.
    :return: What was fetched and what was already there.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outcome = BenchmarkDownload(directory=directory)

    eval1 = directory / BIOMNI_EVAL1_FILE
    if is_verified(eval1, BIOMNI_EVAL1_FILE):
        outcome.present.append(BIOMNI_EVAL1_FILE)
    else:
        eval1.parent.mkdir(parents=True, exist_ok=True)
        fetch_file(eval1_url, eval1)
        if not is_verified(eval1, BIOMNI_EVAL1_FILE):
            eval1.unlink(missing_ok=True)
            raise BenchmarkError(
                f"{eval1_url} is not the Biomni-Eval1 this version of virtual_lab scores against: "
                f"its SHA-256 is not {BENCHMARK_FILES[BIOMNI_EVAL1_FILE]}."
            )
        outcome.downloaded.append(BIOMNI_EVAL1_FILE)

    in_archive = [name for name in BENCHMARK_FILES if name != BIOMNI_EVAL1_FILE]
    missing = [name for name in in_archive if not is_verified(directory / name, name)]
    outcome.present += [name for name in in_archive if name not in missing]
    if not missing:
        return outcome

    url = f"{base_url.rstrip('/')}/{BIOMNI_BENCHMARK_ARCHIVE}"
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        archive = Path(scratch) / BIOMNI_BENCHMARK_ARCHIVE
        fetch_file(url, archive)
        if sha256_of(archive) != BIOMNI_BENCHMARK_ARCHIVE_SHA256:
            raise BenchmarkError(
                f"{url} is not the archive this version of virtual_lab scores against: its SHA-256 "
                f"is not {BIOMNI_BENCHMARK_ARCHIVE_SHA256}."
            )

        # Only the files pinned are read, each by its exact name, so nothing in the archive is
        # written anywhere but where it is expected
        with zipfile.ZipFile(archive) as unpacked:
            for name in missing:
                data = unpacked.read(name)
                if hashlib.sha256(data).hexdigest() != BENCHMARK_FILES[name]:
                    raise BenchmarkError(f"{name} in {url} is not the file pinned in BENCHMARK_FILES.")
                write_atomically(directory / name, data)
                outcome.downloaded.append(name)

    return outcome


def benchmark_file(directory: Path, name: str) -> Path:
    """Finds a benchmark file, checking it is the one pinned.

    :raises FileNotFoundError: If it has not been fetched.
    :raises BenchmarkError: If it is not the one pinned.
    """
    path = Path(directory) / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Fetch the benchmarks with download_benchmarks({str(directory)!r}).")
    if sha256_of(path) != BENCHMARK_FILES[name]:
        raise BenchmarkError(
            f"{path} is not the file pinned in BENCHMARK_FILES, so scores made on it could not be "
            f"compared with others. Fetch it again with download_benchmarks."
        )

    return path


def read_rows(directory: Path, name: str) -> list[dict[str, Any]]:
    """Reads a benchmark file's rows, after checking it is the one pinned."""
    parquet = optional_module("pyarrow.parquet")

    return parquet.read_table(benchmark_file(directory, name)).to_pylist()


@dataclass(frozen=True)
class Question:
    """One question of a benchmark.

    :param benchmark: The benchmark's name.
    :param task: The task it belongs to: one of Biomni-Eval1's ten, or the benchmark's name for a
        benchmark that is a single task.
    :param id: Its number within the task: Biomni-Eval1's task_instance_id, or its index in the
        benchmark, as Biomni's get_example takes it.
    :param prompt: What is asked, exactly as Biomni asks it.
    :param answer: The correct answer, as the benchmark records it. The scoring rule compares an
        answer with it, which is not always in the same form: Biomni-Eval1's patient gene
        detection records a gene, and scores a record listing genes.
    :param metadata: Anything else the benchmark records about it.
    """

    benchmark: str
    task: str
    id: int
    prompt: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def key(self) -> str:
        """The question's name within its benchmark, as task/id."""
        return f"{self.task}/{self.id}"


class MultipleChoiceOutput(BaseModel):
    """Multiple choice output."""

    choice: str | None = Field(
        description="Multiple choice answer. For example, if there is <answer>A</answer> in the prompt, the output should be 'A'."
    )


# Biomni-Eval1 has no output classes of its own. These follow the answer formats its dataset card
# gives for each task.


class CrisprDeliveryOutput(BaseModel):
    """The CRISPR delivery method chosen."""

    choice: str = Field(description="The letter of the most relevant delivery method, from a to f, for example 'e'.")


class GeneOutput(BaseModel):
    """The gene chosen."""

    gene: str = Field(
        description="The one gene chosen, as written in the list of genes given but without any braces around it, for example 'BRCA1'."
    )


class VariantOutput(BaseModel):
    """The variant chosen."""

    variant: str = Field(description="The one variant chosen, exactly as written in the list of variants given, for example 'rs1065852'.")


class RareDiseaseOutput(BaseModel):
    """The rare disease diagnosed."""

    disease_name: str = Field(description="The name of the disease.")
    OMIM_ID: str = Field(description="The disease's OMIM ID, digits only, for example '154700'.")


class PatientGeneOutput(BaseModel):
    """The causal gene found."""

    causal_gene: list[str] = Field(
        description="The causal gene, as its Ensembl ID exactly as written in the list of candidate genes, for example ['ENSG00000186847']."
    )


def answer_text(output: BaseModel) -> str:
    """An output as the answer Biomni's scoring rules take: a string, or JSON for a record."""
    if isinstance(output, MultipleChoiceOutput | CrisprDeliveryOutput):
        return output.choice or ""
    if isinstance(output, GeneOutput):
        return output.gene
    if isinstance(output, VariantOutput):
        return output.variant

    return json.dumps(output.model_dump(mode="json"))


class Benchmark:
    """A benchmark: its questions, what answers are asked for, and how they are scored.

    :param questions: The questions, in the benchmark's order.
    """

    name = "benchmark"

    def __init__(self, questions: Iterable[Question]) -> None:
        self.questions = tuple(questions)
        if len({question.key for question in self.questions}) != len(self.questions):
            raise ValueError(f"{self.name} has two questions with the same task and id")

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(questions={len(self)})"

    def tasks(self) -> list[str]:
        """The tasks of the benchmark, in order of first appearance."""
        return list(dict.fromkeys(question.task for question in self.questions))

    def output_schema(self, question: Question) -> type[BaseModel]:
        """The schema an answer to a question is extracted into."""
        return MultipleChoiceOutput

    def answer_from(self, question: Question, output: BaseModel) -> str:
        """The answer an output gives, as the scoring rule takes it."""
        return answer_text(output)

    def score(self, question: Question, answer: str | None) -> float:
        """Scores an answer from 0 to 1; no answer scores 0."""
        return float(answer == question.answer)

    def metrics(self, answers: Sequence[tuple[Question, str | None]]) -> dict[str, float | None]:
        """The benchmark's own measures over a set of answers, as Biomni computes them."""
        if not answers:
            return {"accuracy": None}

        return {"accuracy": sum(self.score(question, answer) for question, answer in answers) / len(answers)}

    def describe(self) -> dict[str, Any]:
        """Describes the benchmark for the record of a run: what it is, and on which files."""
        return {"name": self.name, "type": type(self).__name__, "questions": len(self)}


def eval1_reward(task_name: str, user_answer: Any, ground_truth: str) -> float:
    """Scores an answer to a Biomni-Eval1 question by the rule for its task, as Biomni does."""
    if task_name == "crispr_delivery":
        return 1.0 if user_answer.strip().lower() == ground_truth.strip().lower() else 0.0

    if task_name.startswith("gwas_causal_gene"):
        return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

    if task_name == "gwas_variant_prioritization":
        return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0

    if task_name == "hle" or task_name.startswith("lab_bench") or task_name == "screen_gene_retrieval":
        return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

    if task_name == "rare_disease_diagnosis":
        try:
            user_dict = parse_record(user_answer)
            truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            return 1.0 if user_dict.get("OMIM_ID") == truth.get("OMIM_ID") else 0.0
        except Exception:
            return 0.0

    if task_name == "patient_gene_detection":
        try:
            predicted = parse_record(user_answer).get("causal_gene", [])
            if not isinstance(predicted, list):
                predicted = [predicted]
            true_genes = [gene.strip() for gene in ground_truth.split(",")] if "," in ground_truth else [ground_truth]
            return 1.0 if predicted and set(true_genes) & set(predicted) else 0.0
        except Exception:
            return 0.0

    raise ValueError(f"Unknown task: {task_name}")


def parse_record(user_answer: Any) -> Any:
    """Reads an answer given as JSON, or as a Python literal, as Biomni's scoring rules do."""
    if not isinstance(user_answer, str):
        return user_answer
    try:
        return json.loads(user_answer)
    except json.JSONDecodeError:
        return ast.literal_eval(user_answer)


EVAL1_OUTPUTS: dict[str, type[BaseModel]] = {
    "crispr_delivery": CrisprDeliveryOutput,
    "gwas_causal_gene_gwas_catalog": GeneOutput,
    "gwas_causal_gene_opentargets": GeneOutput,
    "gwas_causal_gene_pharmaprojects": GeneOutput,
    "gwas_variant_prioritization": VariantOutput,
    "lab_bench_dbqa": MultipleChoiceOutput,
    "lab_bench_seqqa": MultipleChoiceOutput,
    "patient_gene_detection": PatientGeneOutput,
    "rare_disease_diagnosis": RareDiseaseOutput,
    "screen_gene_retrieval": GeneOutput,
}


class BiomniEval1Benchmark(Benchmark):
    """Biomni-Eval1: 433 questions in ten tasks, each answer scored 0 or 1 by its task's rule.

    :param directory: Where download_benchmarks put the files.
    :param tasks: The tasks to include, defaulting to all ten.
    """

    name = "biomni_eval1"

    def __init__(self, directory: Path, tasks: Sequence[str] | None = None) -> None:
        rows = read_rows(directory, BIOMNI_EVAL1_FILE)
        known = {row["task_name"] for row in rows}
        if tasks is not None and (unknown := sorted(set(tasks) - known)):
            raise ValueError(f"Biomni-Eval1 has no task {', '.join(unknown)}; it has {', '.join(sorted(known))}")

        self.files = {BIOMNI_EVAL1_FILE: BENCHMARK_FILES[BIOMNI_EVAL1_FILE]}
        super().__init__(
            Question(
                benchmark=self.name,
                task=row["task_name"],
                id=int(row["task_instance_id"]),
                prompt=row["prompt"],
                answer=row["answer"],
                metadata={"instance_id": int(row["instance_id"]), "split": row["split"]},
            )
            for row in rows
            if tasks is None or row["task_name"] in tasks
        )

    def output_schema(self, question: Question) -> type[BaseModel]:
        return EVAL1_OUTPUTS[question.task]

    def score(self, question: Question, answer: str | None) -> float:
        return 0.0 if answer is None else eval1_reward(question.task, answer, question.answer)

    def metrics(self, answers: Sequence[tuple[Question, str | None]]) -> dict[str, float | None]:
        """The mean score overall and in each task."""
        measures: dict[str, float | None] = dict(super().metrics(answers))
        for task in dict.fromkeys(question.task for question, _ in answers):
            in_task = [(question, answer) for question, answer in answers if question.task == task]
            measures[f"accuracy/{task}"] = sum(self.score(q, a) for q, a in in_task) / len(in_task)

        return measures

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "tasks": self.tasks(), "files": self.files}


LAB_BENCH_PROMPT = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Question: {question}
Options:
{options}

You MUST include the letter of the correct answer within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]<answer>[/ANSWER]',
where <answer> is the correct letter. Always answer in exactly this format
of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing.
            """

REFRAIN_OPTION = "Insufficient information to answer the question."

LAB_BENCH_FILES = {
    "test": "train-00000-of-00001_test.parquet",
    "sampled": "train-00000-of-00001_sampled.parquet",
    "all": "train-00000-of-00001.parquet",
}


class LabBench(Benchmark):
    """LAB-Bench's database (DbQA) or sequence (SeqQA) questions, as Biomni asks them.

    Each question's options are its distractors, its correct answer, and a refusal ("Insufficient
    information to answer the question."), shuffled with numpy's random generator seeded with 42,
    one question after another, as Biomni shuffles them, so that each question has the letters
    it has in Biomni.

    :param directory: Where download_benchmarks put the files.
    :param dataset: "DbQA" or "SeqQA".
    :param subset: Which of Biomni's files: "test", the one Biomni's lab_bench reads (60 DbQA and
        70 SeqQA questions); "sampled"; or "all" of LAB-Bench's questions (520 and 600), which
        Biomni-Eval1's LAB-Bench questions are drawn from, with the same letters.
    """

    def __init__(self, directory: Path, dataset: str = "DbQA", subset: Literal["test", "sampled", "all"] = "test") -> None:
        if dataset not in ["DbQA", "SeqQA"]:
            raise ValueError("dataset must be one of 'DbQA', 'SeqQA'")
        if subset not in LAB_BENCH_FILES:
            raise ValueError(f"subset must be one of {', '.join(map(repr, LAB_BENCH_FILES))}")

        numpy = optional_module("numpy")
        self.dataset = dataset
        self.subset = subset
        self.name = f"lab_bench_{dataset.lower()}"
        name = f"{dataset}/{LAB_BENCH_FILES[subset]}"
        self.files = {name: BENCHMARK_FILES[name]}
        self.prompt = LAB_BENCH_PROMPT

        # One generator for the whole file, drawn from in row order, is what seeding numpy's
        # global generator and shuffling inside DataFrame.apply amounts to
        generator = numpy.random.RandomState(42)
        questions = []
        for index, row in enumerate(read_rows(directory, name)):
            options = list(row["distractors"]) + [row["ideal"], REFRAIN_OPTION]
            generator.shuffle(options)
            letters = "\n".join(f"{chr(ord('A') + position)}.{option}" for position, option in enumerate(options))
            questions.append(
                Question(
                    benchmark=self.name,
                    task=self.name,
                    id=index,
                    prompt=self.prompt.format(question=row["question"], options=letters),
                    answer=chr(ord("A") + options.index(row["ideal"])),
                    metadata={
                        "refrain": chr(ord("A") + options.index(REFRAIN_OPTION)),
                        "lab_bench_id": row["id"],
                        "subtask": row["subtask"],
                    },
                )
            )

        super().__init__(questions)

    def metrics(self, answers: Sequence[tuple[Question, str | None]]) -> dict[str, float | None]:
        """Accuracy, coverage (how often a question was answered rather than refused), the
        share refused, and precision (accuracy on the questions answered), as Biomni's
        lab_bench.evaluate computes them. A missing answer counts as answered wrongly, as
        Biomni counts one."""
        if not answers:
            return {"accuracy": None, "coverage": None, "refrain_ratio": None, "precision": None}

        covered = [(question, answer) for question, answer in answers if answer != question.metadata["refrain"]]

        return {
            "accuracy": sum(answer == question.answer for question, answer in answers) / len(answers),
            "coverage": len(covered) / len(answers),
            "refrain_ratio": 1 - len(covered) / len(answers),
            # Biomni's is undefined, and warns, when every question was refused
            "precision": sum(answer == question.answer for question, answer in covered) / len(covered)
            if covered
            else None,
        }

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "dataset": self.dataset, "subset": self.subset, "files": self.files}

    def get_example(self, index: int | None = None) -> dict[str, str]:
        """A question's prompt and answer, as Biomni's lab_bench.get_example gives them."""
        if index is None:
            index = optional_module("numpy").random.randint(len(self.questions))

        return {"prompt": self.questions[index].prompt, "answer": self.questions[index].answer}

    def get_iterator(self) -> Iterator[dict[str, str]]:
        """Every question, as get_example gives it."""
        for index in range(len(self.questions)):
            yield self.get_example(index)

    def evaluate(self, response: Sequence[str | None]) -> dict[str, float | None]:
        """Scores one answer per question, in order, as Biomni's lab_bench.evaluate does."""
        if len(response) != len(self.questions):
            raise ValueError(f"Expected {len(self.questions)} answers, one per question, not {len(response)}")

        return self.metrics(list(zip(self.questions, response, strict=True)))

    def output_class(self) -> type[BaseModel]:
        """The schema Biomni extracts an answer into."""
        return MultipleChoiceOutput


HLE_CATEGORIES = [
    "Other",
    "Humanities/Social Science",
    "Math",
    "Physics",
    "Computer Science/AI",
    "Biology/Medicine",
    "Chemistry",
    "Engineering",
]


class HumanitysLastExam(Benchmark):
    """The 52 multiple choice questions of Humanity's Last Exam in biology and medicine that
    Biomni samples, asked as Biomni asks them: the question, choices included, and nothing else.

    Five of them refer to an image, which Biomni does not show its agent and neither does this;
    their metadata says has_image, so that they can be left out.

    :param directory: Where download_benchmarks put the files.
    :param category: The category of questions, as in Biomni. Biomni's file holds only
        Biology/Medicine, so any other has no questions.
    """

    name = "hle"

    def __init__(self, directory: Path, category: str = "Biology/Medicine") -> None:
        if category not in HLE_CATEGORIES:
            raise ValueError(f"category must be one of {HLE_CATEGORIES}")

        name = "hle/test_sampled_biology_medicine.parquet"
        self.category = category
        self.files = {name: BENCHMARK_FILES[name]}
        self.prompt = """Question: {question}"""

        rows = [
            row
            for row in read_rows(directory, name)
            if row["category"] == category and row["answer_type"] == "multipleChoice"
        ]
        super().__init__(
            Question(
                benchmark=self.name,
                task=self.name,
                id=index,
                prompt=self.prompt.format(question=row["question"]),
                answer=row["answer"][0],
                metadata={"hle_id": row["id"], "has_image": bool(row["image"])},
            )
            for index, row in enumerate(rows)
        )

    def metrics(self, answers: Sequence[tuple[Question, str | None]]) -> dict[str, float | None]:
        """Accuracy. Biomni's humanity_last_exam.evaluate also reports coverage and precision
        against a refusal option, but its questions have none, and it fails trying."""
        return super().metrics(answers)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "category": self.category, "files": self.files}

    def get_example(self, index: int | None = None) -> dict[str, str]:
        """A question's prompt and answer, as Biomni's humanity_last_exam.get_example gives them."""
        if index is None:
            index = optional_module("numpy").random.randint(len(self.questions))

        return {"prompt": self.questions[index].prompt, "answer": self.questions[index].answer}

    def get_iterator(self) -> Iterator[dict[str, str]]:
        """Every question, as get_example gives it."""
        for index in range(len(self.questions)):
            yield self.get_example(index)

    def evaluate(self, response: Sequence[str | None]) -> dict[str, float | None]:
        """Scores one answer per question, in order."""
        if len(response) != len(self.questions):
            raise ValueError(f"Expected {len(self.questions)} answers, one per question, not {len(response)}")

        return self.metrics(list(zip(self.questions, response, strict=True)))

    def output_class(self) -> type[BaseModel]:
        """The schema Biomni extracts an answer into."""
        return MultipleChoiceOutput


class BiomniEval1:
    """Biomni-Eval1 with the interface of Biomni's own BiomniEval1, for code written against it.

    Biomni reads the dataset from Hugging Face each time; this reads the copy download_benchmarks
    fetched, checked against its pinned hash.

    :param directory: Where download_benchmarks put the files.
    """

    def __init__(self, directory: Path) -> None:
        self.rows = read_rows(directory, BIOMNI_EVAL1_FILE)
        self.instance_map = {(row["task_name"], row["task_instance_id"]): index for index, row in enumerate(self.rows)}
        print(
            f"Loaded BiomniEval1 dataset: {len(self.rows)} instances across "
            f"{len({row['task_name'] for row in self.rows})} tasks"
        )

    @property
    def df(self) -> Any:
        """The dataset as a pandas DataFrame, as Biomni's has it."""
        return optional_module("pandas").DataFrame(self.rows)

    def row(self, task_name: str, task_instance_id: int) -> dict[str, Any]:
        key = (task_name, task_instance_id)
        if key not in self.instance_map:
            raise ValueError(f"Instance not found: task={task_name}, task_instance_id={task_instance_id}")

        return self.rows[self.instance_map[key]]

    def evaluate(self, task_name: str, task_instance_id: int, user_answer: str) -> float:
        """Scores an answer from 0 to 1 by its task's rule."""
        ground_truth = self.row(task_name, task_instance_id)["answer"]
        try:
            return float(eval1_reward(task_name, user_answer, ground_truth))
        except Exception as error:
            raise RuntimeError(f"Error computing reward for {task_name} instance {task_instance_id}: {error}") from error

    def get_instance(self, task_name: str, task_instance_id: int) -> dict[str, Any]:
        """A question's prompt, answer, and identifiers."""
        row = self.row(task_name, task_instance_id)

        return {
            "global_instance_id": row["instance_id"],
            "task_instance_id": row["task_instance_id"],
            "task_name": row["task_name"],
            "split": row["split"],
            "prompt": row["prompt"],
            "answer": row["answer"],
        }

    def list_tasks(self) -> list[str]:
        """Every task, sorted."""
        return sorted({row["task_name"] for row in self.rows})

    def get_task_stats(self, task_name: str | None = None) -> dict[str, Any]:
        """How many questions there are, in each split, overall or in one task."""
        rows = [row for row in self.rows if task_name is None or row["task_name"] == task_name]
        if task_name and not rows:
            raise ValueError(f"Task not found: {task_name}")

        stats: dict[str, Any] = {
            "total_instances": len(rows),
            "train_instances": sum(row["split"] == "train" for row in rows),
            "val_instances": sum(row["split"] == "val" for row in rows),
        }
        if not task_name:
            stats["tasks"] = {task: self.get_task_stats(task) for task in self.list_tasks()}

        return stats

    def batch_evaluate(self, evaluations: list[tuple[str, int, str]]) -> list[float]:
        """Scores several answers, scoring 0 for one that cannot be scored, as Biomni does."""
        results = []
        for task_name, task_instance_id, user_answer in evaluations:
            try:
                results.append(self.evaluate(task_name, task_instance_id, user_answer))
            except Exception as error:
                print(f"Error evaluating {task_name} instance {task_instance_id}: {error}")
                results.append(0.0)

        return results

    def get_instances_by_task(self, task_name: str, split: str | None = None) -> Any:
        """A task's questions, in one split if given, as a pandas DataFrame."""
        pandas = optional_module("pandas")
        rows = [row for row in self.rows if row["task_name"] == task_name and (not split or row["split"] == split)]

        return pandas.DataFrame(rows, columns=list(self.rows[0]) if self.rows else None)

    def __repr__(self) -> str:
        return f"BiomniEval1(instances={len(self.rows)}, tasks={len(self.list_tasks())})"

    def __len__(self) -> int:
        return len(self.rows)
