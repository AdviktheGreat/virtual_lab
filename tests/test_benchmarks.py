"""Tests for Biomni's benchmarks: fetching their files, reading their questions, and scoring answers."""

import hashlib
import io
import json
import os
import zipfile
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from virtual_lab.benchmarks import (  # noqa: E402
    EVAL1_OUTPUTS,
    LAB_BENCH_PROMPT,
    REFRAIN_OPTION,
    BenchmarkError,
    BiomniEval1,
    BiomniEval1Benchmark,
    CrisprDeliveryOutput,
    GeneOutput,
    HumanitysLastExam,
    LabBench,
    MultipleChoiceOutput,
    PatientGeneOutput,
    Question,
    RareDiseaseOutput,
    VariantOutput,
    answer_text,
    benchmark_file,
    download_benchmarks,
    eval1_reward,
)
from virtual_lab.constants import (  # noqa: E402
    BENCHMARK_FILES,
    BIOMNI_BENCHMARK_ARCHIVE,
    BIOMNI_EVAL1_FILE,
)

benchmarks_module = import_module("virtual_lab.benchmarks")

EVAL1_ROWS = [
    {"instance_id": 0, "task_instance_id": 0, "prompt": "Pick a method.", "task_name": "crispr_delivery", "split": "val", "answer": "e"},
    {"instance_id": 1, "task_instance_id": 1, "prompt": "Pick another.", "task_name": "crispr_delivery", "split": "val", "answer": "b"},
    {"instance_id": 2, "task_instance_id": 0, "prompt": "Which gene?", "task_name": "gwas_causal_gene_opentargets", "split": "val", "answer": "HNF1A"},
    {"instance_id": 3, "task_instance_id": 7, "prompt": "Which variant?", "task_name": "gwas_variant_prioritization", "split": "val", "answer": "rs4253311"},
    {"instance_id": 4, "task_instance_id": 0, "prompt": "Diagnose.", "task_name": "rare_disease_diagnosis", "split": "val", "answer": '{"disease_name": "Gordon syndrome", "OMIM_ID": "114300"}'},
    {"instance_id": 5, "task_instance_id": 0, "prompt": "Find the gene.", "task_name": "patient_gene_detection", "split": "train", "answer": "ENSG1"},
]


def lab_bench_rows(count: int) -> list[dict[str, Any]]:
    """LAB-Bench rows with three or four distractors, so that shuffles differ in length."""
    return [
        {
            "id": f"q{index}",
            "question": f"Question {index}?",
            "ideal": f"right {index}",
            "distractors": [f"wrong {index}.{choice}" for choice in range(3 + index % 2)],
            "subtask": "subtask",
        }
        for index in range(count)
    ]


HLE_ROWS = [
    {"id": "h0", "question": "What? Answer Choices: A. x B. y", "image": "", "answer": "B. y", "answer_type": "multipleChoice", "category": "Biology/Medicine"},
    {"id": "h1", "question": "Name it.", "image": "", "answer": "thing", "answer_type": "exactMatch", "category": "Biology/Medicine"},
    {"id": "h2", "question": "Which? Answer Choices: A. p B. q", "image": "data:image/png;base64,AAAA", "answer": "A", "answer_type": "multipleChoice", "category": "Biology/Medicine"},
    {"id": "h3", "question": "Sum? Answer Choices: A. 1 B. 2", "image": "", "answer": "A", "answer_type": "multipleChoice", "category": "Math"},
]


def parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)

    return buffer.getvalue()


def benchmark_files() -> dict[str, bytes]:
    """Small stand-ins for every pinned benchmark file."""
    files = {BIOMNI_EVAL1_FILE: parquet_bytes(EVAL1_ROWS), "hle/test_sampled_biology_medicine.parquet": parquet_bytes(HLE_ROWS)}
    for dataset, count in (("DbQA", 9), ("SeqQA", 7)):
        for suffix, rows in (("", count), ("_sampled", 3), ("_test", 5)):
            files[f"{dataset}/train-00000-of-00001{suffix}.parquet"] = parquet_bytes(lab_bench_rows(rows))

    return files


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def files(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Pins the stand-in files' hashes in place of the real ones."""
    made = benchmark_files()
    assert set(made) == set(BENCHMARK_FILES)
    for name, data in made.items():
        monkeypatch.setitem(BENCHMARK_FILES, name, sha256(data))

    return made


@pytest.fixture
def directory(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """A benchmark directory holding every stand-in file."""
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)

    return tmp_path


def archive_of(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    return buffer.getvalue()


class FakeFetch:
    """Stands in for fetch_file, serving bytes by URL and recording what was fetched."""

    def __init__(self, served: dict[str, bytes]) -> None:
        self.served = served
        self.fetched: list[str] = []

    def __call__(self, url: str, path: Path) -> None:
        self.fetched.append(url)
        path.write_bytes(self.served[url])


def serve(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes], archive: bytes | None = None, eval1: bytes | None = None) -> FakeFetch:
    in_archive = {name: data for name, data in files.items() if name != BIOMNI_EVAL1_FILE}
    archive = archive if archive is not None else archive_of(in_archive)
    monkeypatch.setattr(benchmarks_module, "BIOMNI_BENCHMARK_ARCHIVE_SHA256", sha256(archive))
    fetch = FakeFetch(
        {
            "https://eval1.test/eval1.parquet": eval1 if eval1 is not None else files[BIOMNI_EVAL1_FILE],
            f"https://release.test/{BIOMNI_BENCHMARK_ARCHIVE}": archive,
        }
    )
    monkeypatch.setattr(benchmarks_module, "fetch_file", fetch)

    return fetch


def download(directory: Path) -> Any:
    return download_benchmarks(directory, base_url="https://release.test/", eval1_url="https://eval1.test/eval1.parquet")


class TestDownload:
    def test_fetches_every_file_as_pinned(self, tmp_path: Path, files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
        fetch = serve(monkeypatch, files)

        outcome = download(tmp_path)

        assert sorted(outcome.downloaded) == sorted(files)
        assert outcome.present == []
        for name, data in files.items():
            assert (tmp_path / name).read_bytes() == data
        assert len(fetch.fetched) == 2

    def test_files_already_present_are_not_fetched_again(self, directory: Path, files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
        fetch = serve(monkeypatch, files)

        outcome = download(directory)

        assert fetch.fetched == []
        assert outcome.downloaded == []
        assert sorted(outcome.present) == sorted(files)

    def test_only_a_missing_file_is_extracted(self, directory: Path, files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
        fetch = serve(monkeypatch, files)
        (directory / "SeqQA/train-00000-of-00001_test.parquet").write_bytes(b"changed")

        outcome = download(directory)

        assert outcome.downloaded == ["SeqQA/train-00000-of-00001_test.parquet"]
        assert fetch.fetched == [f"https://release.test/{BIOMNI_BENCHMARK_ARCHIVE}"]
        assert (directory / "SeqQA/train-00000-of-00001_test.parquet").read_bytes() == files["SeqQA/train-00000-of-00001_test.parquet"]

    def test_a_changed_eval1_is_refused_and_removed(self, tmp_path: Path, files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
        serve(monkeypatch, files, eval1=b"not the pinned file")

        with pytest.raises(BenchmarkError, match="Biomni-Eval1"):
            download(tmp_path)

        assert not (tmp_path / BIOMNI_EVAL1_FILE).exists()

    def test_a_changed_archive_is_refused(self, tmp_path: Path, files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
        serve(monkeypatch, files)
        monkeypatch.setattr(benchmarks_module, "BIOMNI_BENCHMARK_ARCHIVE_SHA256", "0" * 64)

        with pytest.raises(BenchmarkError, match="archive"):
            download(tmp_path)

        assert not (tmp_path / "DbQA").exists()

    def test_a_changed_file_in_the_archive_is_refused(self, tmp_path: Path, files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
        members = {name: data for name, data in files.items() if name != BIOMNI_EVAL1_FILE}
        members["hle/test_sampled_biology_medicine.parquet"] = b"changed"
        serve(monkeypatch, files, archive=archive_of(members))

        with pytest.raises(BenchmarkError, match="hle/test_sampled_biology_medicine.parquet"):
            download(tmp_path)

        assert not (tmp_path / "hle/test_sampled_biology_medicine.parquet").exists()

    def test_nothing_but_the_pinned_files_is_written(self, tmp_path: Path, files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
        members = {name: data for name, data in files.items() if name != BIOMNI_EVAL1_FILE}
        members["../outside.txt"] = b"escaped"
        members["DbQA/extra.parquet"] = b"extra"
        serve(monkeypatch, files, archive=archive_of(members))

        download(tmp_path / "bench")

        assert not (tmp_path / "outside.txt").exists()
        assert not (tmp_path / "bench/DbQA/extra.parquet").exists()
        written = sorted(path.relative_to(tmp_path / "bench").as_posix() for path in (tmp_path / "bench").rglob("*") if path.is_file())
        assert written == sorted(files)


class TestBenchmarkFile:
    def test_a_missing_file_says_how_to_fetch_it(self, tmp_path: Path, files: dict[str, bytes]) -> None:
        with pytest.raises(FileNotFoundError, match="download_benchmarks"):
            benchmark_file(tmp_path, BIOMNI_EVAL1_FILE)

    def test_a_changed_file_is_refused(self, directory: Path) -> None:
        (directory / BIOMNI_EVAL1_FILE).write_bytes(b"changed")

        with pytest.raises(BenchmarkError, match="pinned"):
            BiomniEval1Benchmark(directory)

    def test_the_real_files_are_pinned(self) -> None:
        # Every pinned hash is a SHA-256, and every file sits where Biomni's tasks read it
        assert all(len(value) == 64 and int(value, 16) >= 0 for value in BENCHMARK_FILES.values())
        assert "DbQA/train-00000-of-00001_test.parquet" in BENCHMARK_FILES


class TestEval1Rewards:
    @pytest.mark.parametrize(
        "task, answer, truth, expected",
        [
            ("crispr_delivery", " E ", "e", 1.0),
            ("crispr_delivery", "f", "e", 0.0),
            ("gwas_causal_gene_gwas_catalog", "apoa4", "APOA4", 1.0),
            ("gwas_causal_gene_opentargets", "HNF1B", "HNF1A", 0.0),
            ("gwas_causal_gene_pharmaprojects", " xpo1", "XPO1", 1.0),
            ("gwas_variant_prioritization", " rs4253311 ", "rs4253311", 1.0),
            # Variants are compared case and all, as Biomni compares them
            ("gwas_variant_prioritization", "RS4253311", "rs4253311", 0.0),
            ("lab_bench_dbqa", "b", "B", 1.0),
            ("lab_bench_seqqa", "C", "B", 0.0),
            ("screen_gene_retrieval", "znf561-as1", "ZNF561-AS1", 1.0),
            ("hle", "a", "A", 1.0),
            ("rare_disease_diagnosis", '{"disease_name": "Other", "OMIM_ID": "114300"}', '{"disease_name": "Gordon", "OMIM_ID": "114300"}', 1.0),
            ("rare_disease_diagnosis", "{'disease_name': 'Gordon', 'OMIM_ID': '114300'}", '{"OMIM_ID": "114300"}', 1.0),
            ("rare_disease_diagnosis", '{"OMIM_ID": "114301"}', '{"OMIM_ID": "114300"}', 0.0),
            ("rare_disease_diagnosis", "Gordon syndrome", '{"OMIM_ID": "114300"}', 0.0),
            ("patient_gene_detection", '{"causal_gene": ["ENSG2", "ENSG1"]}', "ENSG1", 1.0),
            ("patient_gene_detection", '{"causal_gene": "ENSG1"}', "ENSG1", 1.0),
            ("patient_gene_detection", '{"causal_gene": ["ENSG3"]}', "ENSG1, ENSG3", 1.0),
            ("patient_gene_detection", '{"causal_gene": ["ENSG2"]}', "ENSG1", 0.0),
            ("patient_gene_detection", '{"causal_gene": []}', "ENSG1", 0.0),
            ("patient_gene_detection", "ENSG1", "ENSG1", 0.0),
        ],
    )
    def test_each_task_is_scored_by_its_rule(self, task: str, answer: str, truth: str, expected: float) -> None:
        assert eval1_reward(task, answer, truth) == expected

    def test_an_unknown_task_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            eval1_reward("made_up", "a", "a")


class TestBiomniEval1Benchmark:
    def test_reads_every_question(self, directory: Path) -> None:
        benchmark = BiomniEval1Benchmark(directory)

        assert len(benchmark) == len(EVAL1_ROWS)
        first = benchmark.questions[0]
        assert (first.benchmark, first.task, first.id, first.prompt, first.answer) == ("biomni_eval1", "crispr_delivery", 0, "Pick a method.", "e")
        assert first.metadata == {"instance_id": 0, "split": "val"}
        assert benchmark.questions[3].key == "gwas_variant_prioritization/7"
        assert benchmark.tasks() == list(dict.fromkeys(row["task_name"] for row in EVAL1_ROWS))

    def test_tasks_can_be_chosen(self, directory: Path) -> None:
        benchmark = BiomniEval1Benchmark(directory, tasks=["crispr_delivery"])

        assert [question.key for question in benchmark] == ["crispr_delivery/0", "crispr_delivery/1"]
        assert benchmark.describe()["tasks"] == ["crispr_delivery"]

    def test_an_unknown_task_is_refused(self, directory: Path) -> None:
        with pytest.raises(ValueError, match="no task made_up"):
            BiomniEval1Benchmark(directory, tasks=["made_up"])

    def test_each_task_has_its_schema(self, directory: Path) -> None:
        benchmark = BiomniEval1Benchmark(directory)

        assert {question.task: benchmark.output_schema(question) for question in benchmark} == {
            task: EVAL1_OUTPUTS[task] for task in benchmark.tasks()
        }
        assert set(EVAL1_OUTPUTS) == {
            "crispr_delivery",
            "gwas_causal_gene_gwas_catalog",
            "gwas_causal_gene_opentargets",
            "gwas_causal_gene_pharmaprojects",
            "gwas_variant_prioritization",
            "lab_bench_dbqa",
            "lab_bench_seqqa",
            "patient_gene_detection",
            "rare_disease_diagnosis",
            "screen_gene_retrieval",
        }

    def test_answers_extracted_into_the_schemas_score_as_biomni_scores_them(self, directory: Path) -> None:
        benchmark = BiomniEval1Benchmark(directory)
        outputs = {
            "crispr_delivery": CrisprDeliveryOutput(choice="E"),
            "gwas_causal_gene_opentargets": GeneOutput(gene="hnf1a"),
            "gwas_variant_prioritization": VariantOutput(variant="rs4253311"),
            "rare_disease_diagnosis": RareDiseaseOutput(disease_name="Gordon", OMIM_ID="114300"),
            "patient_gene_detection": PatientGeneOutput(causal_gene=["ENSG1"]),
        }
        scores = {
            question.key: benchmark.score(question, benchmark.answer_from(question, outputs[question.task])) for question in benchmark
        }

        assert scores == {
            "crispr_delivery/0": 1.0,
            "crispr_delivery/1": 0.0,
            "gwas_causal_gene_opentargets/0": 1.0,
            "gwas_variant_prioritization/7": 1.0,
            "rare_disease_diagnosis/0": 1.0,
            "patient_gene_detection/0": 1.0,
        }

    def test_no_answer_scores_zero(self, directory: Path) -> None:
        benchmark = BiomniEval1Benchmark(directory)

        assert benchmark.score(benchmark.questions[0], None) == 0.0

    def test_metrics_are_overall_and_per_task(self, directory: Path) -> None:
        benchmark = BiomniEval1Benchmark(directory, tasks=["crispr_delivery", "gwas_causal_gene_opentargets"])
        crispr_0, crispr_1, gene = benchmark.questions

        metrics = benchmark.metrics([(crispr_0, "e"), (crispr_1, "a"), (gene, None)])

        assert metrics == {
            "accuracy": pytest.approx(1 / 3),
            "accuracy/crispr_delivery": 0.5,
            "accuracy/gwas_causal_gene_opentargets": 0.0,
        }
        assert benchmark.metrics([]) == {"accuracy": None}

    def test_describe_names_the_file_it_read(self, directory: Path, files: dict[str, bytes]) -> None:
        described = BiomniEval1Benchmark(directory).describe()

        assert described["files"] == {BIOMNI_EVAL1_FILE: sha256(files[BIOMNI_EVAL1_FILE])}
        assert described["questions"] == len(EVAL1_ROWS)


def biomni_lab_bench(path: Path, dataset: str, filename: str) -> Any:
    """Biomni's own processing of a LAB-Bench file, from biomni/task/lab_bench.py, run as it is."""
    df = pd.read_parquet(path / dataset / filename)
    np.random.seed(42)

    def shuffle(x):  # noqa: ANN001, ANN202
        np.random.shuffle(x)
        return x

    df["options"] = df.apply(
        lambda x: shuffle(x.distractors.tolist() + [x.ideal] + ["Insufficient information to answer the question."]),
        axis=1,
    )
    df["options_letters"] = df.options.apply(lambda x: "\n".join([chr(ord("A") + i) + "." + item for i, item in enumerate(x)]))
    df["letter_answer"] = df.apply(lambda x: chr(ord("A") + np.where(np.array(x.options) == x.ideal)[0][0]), axis=1)
    df["letter_refrain"] = df.apply(
        lambda x: chr(ord("A") + np.where(np.array(x.options) == "Insufficient information to answer the question.")[0][0]),
        axis=1,
    )

    return df


class TestLabBench:
    @pytest.mark.parametrize("dataset", ["DbQA", "SeqQA"])
    @pytest.mark.parametrize("subset, filename", [("test", "train-00000-of-00001_test.parquet"), ("all", "train-00000-of-00001.parquet")])
    def test_options_are_lettered_as_biomni_letters_them(self, directory: Path, dataset: str, subset: str, filename: str) -> None:
        expected = biomni_lab_bench(directory, dataset, filename)

        benchmark = LabBench(directory, dataset, subset)  # type: ignore[arg-type]

        assert [question.answer for question in benchmark] == list(expected.letter_answer)
        assert [question.metadata["refrain"] for question in benchmark] == list(expected.letter_refrain)
        assert [question.prompt for question in benchmark] == [
            LAB_BENCH_PROMPT.format(question=row.question, options=row.options_letters) for row in expected.itertuples()
        ]

    def test_the_shuffle_is_not_the_identity(self, directory: Path) -> None:
        # Guards the parity test above against both sides leaving the options unshuffled
        benchmark = LabBench(directory, "DbQA", "all")

        assert {question.answer for question in benchmark} != {"D"}
        assert len({question.metadata["refrain"] for question in benchmark}) > 1

    def test_questions_are_named_by_their_index(self, directory: Path) -> None:
        benchmark = LabBench(directory, "SeqQA")

        assert benchmark.name == "lab_bench_seqqa"
        assert [question.key for question in benchmark] == [f"lab_bench_seqqa/{index}" for index in range(5)]
        assert benchmark.questions[2].metadata["lab_bench_id"] == "q2"
        assert benchmark.describe() == {
            "name": "lab_bench_seqqa",
            "type": "LabBench",
            "questions": 5,
            "dataset": "SeqQA",
            "subset": "test",
            "files": {"SeqQA/train-00000-of-00001_test.parquet": BENCHMARK_FILES["SeqQA/train-00000-of-00001_test.parquet"]},
        }

    def test_a_dataset_or_subset_biomni_lacks_is_refused(self, directory: Path) -> None:
        with pytest.raises(ValueError, match="DbQA"):
            LabBench(directory, "ProtocolQA")
        with pytest.raises(ValueError, match="subset"):
            LabBench(directory, "DbQA", "train")  # type: ignore[arg-type]

    def test_metrics_are_biomnis(self, directory: Path) -> None:
        benchmark = LabBench(directory)
        questions = benchmark.questions
        refrain = questions[1].metadata["refrain"]
        wrong = next(letter for letter in "ABCDE" if letter not in {questions[2].answer, questions[2].metadata["refrain"]})

        answers = [questions[0].answer, refrain, wrong, None, questions[4].answer]
        metrics = benchmark.evaluate(answers)

        assert metrics == {
            "accuracy": pytest.approx(2 / 5),
            "coverage": pytest.approx(4 / 5),
            "refrain_ratio": pytest.approx(1 / 5),
            "precision": pytest.approx(2 / 4),
        }

    def test_precision_is_none_when_every_question_is_refused(self, directory: Path) -> None:
        benchmark = LabBench(directory)

        metrics = benchmark.evaluate([question.metadata["refrain"] for question in benchmark])

        assert metrics["coverage"] == 0.0
        assert metrics["refrain_ratio"] == 1.0
        assert metrics["precision"] is None

    def test_evaluate_needs_one_answer_per_question(self, directory: Path) -> None:
        with pytest.raises(ValueError, match="Expected 5 answers"):
            LabBench(directory).evaluate(["A"])

    def test_biomnis_interface(self, directory: Path) -> None:
        benchmark = LabBench(directory)

        assert benchmark.get_example(3) == {"prompt": benchmark.questions[3].prompt, "answer": benchmark.questions[3].answer}
        assert list(benchmark.get_iterator()) == [benchmark.get_example(index) for index in range(5)]
        assert benchmark.get_example() in list(benchmark.get_iterator())
        assert benchmark.output_class() is MultipleChoiceOutput
        assert REFRAIN_OPTION in benchmark.questions[0].prompt


class TestHumanitysLastExam:
    def test_reads_the_multiple_choice_questions_of_the_category(self, directory: Path) -> None:
        benchmark = HumanitysLastExam(directory)

        assert [question.prompt for question in benchmark] == [f"Question: {HLE_ROWS[0]['question']}", f"Question: {HLE_ROWS[2]['question']}"]
        assert [question.answer for question in benchmark] == ["B", "A"]
        assert [question.metadata for question in benchmark] == [{"hle_id": "h0", "has_image": False}, {"hle_id": "h2", "has_image": True}]
        assert [question.key for question in benchmark] == ["hle/0", "hle/1"]

    def test_another_category(self, directory: Path) -> None:
        assert [question.answer for question in HumanitysLastExam(directory, "Math")] == ["A"]
        with pytest.raises(ValueError, match="category"):
            HumanitysLastExam(directory, "Biology")

    def test_accuracy_only(self, directory: Path) -> None:
        benchmark = HumanitysLastExam(directory)

        assert benchmark.evaluate(["B", "C"]) == {"accuracy": 0.5}
        assert benchmark.output_class() is MultipleChoiceOutput
        assert benchmark.get_example(1)["answer"] == "A"


class TestBiomniEval1Interface:
    def test_scores_and_looks_up_as_biomni_does(self, directory: Path, capsys: pytest.CaptureFixture[str]) -> None:
        eval1 = BiomniEval1(directory)

        assert "6 instances across 5 tasks" in capsys.readouterr().out
        assert eval1.evaluate("crispr_delivery", 0, "E") == 1.0
        assert eval1.get_instance("gwas_variant_prioritization", 7) == {
            "global_instance_id": 3,
            "task_instance_id": 7,
            "task_name": "gwas_variant_prioritization",
            "split": "val",
            "prompt": "Which variant?",
            "answer": "rs4253311",
        }
        assert eval1.list_tasks() == sorted({row["task_name"] for row in EVAL1_ROWS})
        assert len(eval1) == 6
        assert repr(eval1) == "BiomniEval1(instances=6, tasks=5)"

    def test_an_unknown_instance_is_refused(self, directory: Path) -> None:
        with pytest.raises(ValueError, match="Instance not found"):
            BiomniEval1(directory).evaluate("crispr_delivery", 9, "a")

    def test_task_stats(self, directory: Path) -> None:
        eval1 = BiomniEval1(directory)

        assert eval1.get_task_stats("crispr_delivery") == {"total_instances": 2, "train_instances": 0, "val_instances": 2}
        stats = eval1.get_task_stats()
        assert (stats["total_instances"], stats["train_instances"], stats["val_instances"]) == (6, 1, 5)
        assert stats["tasks"]["patient_gene_detection"]["train_instances"] == 1
        with pytest.raises(ValueError, match="Task not found"):
            eval1.get_task_stats("made_up")

    def test_batch_evaluate_scores_an_unscorable_answer_zero(self, directory: Path) -> None:
        eval1 = BiomniEval1(directory)

        scores = eval1.batch_evaluate([("crispr_delivery", 1, "b"), ("crispr_delivery", 9, "b"), ("crispr_delivery", 0, None)])

        assert scores == [1.0, 0.0, 0.0]

    def test_data_frames(self, directory: Path) -> None:
        eval1 = BiomniEval1(directory)

        assert list(eval1.df.columns) == list(EVAL1_ROWS[0])
        assert list(eval1.get_instances_by_task("crispr_delivery").task_instance_id) == [0, 1]
        assert len(eval1.get_instances_by_task("patient_gene_detection", split="val")) == 0
        # Indexed by row in the whole dataset, as Biomni's is, so .loc finds the same row in df
        variants = eval1.get_instances_by_task("gwas_variant_prioritization", split="val")
        assert list(variants.index) == [3]
        assert eval1.df.loc[3, "answer"] == variants.loc[3, "answer"] == "rs4253311"


class TestAnswerText:
    def test_each_schema_becomes_the_answer_scored(self) -> None:
        assert answer_text(MultipleChoiceOutput(choice="B")) == "B"
        assert answer_text(MultipleChoiceOutput(choice=None)) == ""
        assert answer_text(CrisprDeliveryOutput(choice="e")) == "e"
        assert answer_text(GeneOutput(gene="BRCA1")) == "BRCA1"
        assert answer_text(VariantOutput(variant="rs1")) == "rs1"
        assert json.loads(answer_text(RareDiseaseOutput(disease_name="X", OMIM_ID="1"))) == {"disease_name": "X", "OMIM_ID": "1"}
        assert json.loads(answer_text(PatientGeneOutput(causal_gene=["ENSG1"]))) == {"causal_gene": ["ENSG1"]}


def test_two_questions_with_one_key_are_refused() -> None:
    from virtual_lab.benchmarks import Benchmark

    question = Question(benchmark="b", task="t", id=0, prompt="p", answer="A")

    with pytest.raises(ValueError, match="same task and id"):
        Benchmark([question, Question(benchmark="b", task="t", id=0, prompt="other", answer="B")])


REAL_BENCHMARKS = os.environ.get("VIRTUAL_LAB_BENCHMARK_DIR")


@pytest.mark.skipif(not REAL_BENCHMARKS, reason="set VIRTUAL_LAB_BENCHMARK_DIR to a directory download_benchmarks filled")
def test_lab_bench_questions_are_biomni_eval1s() -> None:
    # Biomni-Eval1's LAB-Bench questions are LAB-Bench's, lettered by Biomni's shuffle of the
    # whole file, with task_instance_id as the row
    directory = Path(REAL_BENCHMARKS or "")
    eval1 = BiomniEval1Benchmark(directory, tasks=["lab_bench_dbqa", "lab_bench_seqqa"])
    for dataset in ("DbQA", "SeqQA"):
        lab_bench = LabBench(directory, dataset, "all")
        asked = [question for question in eval1 if question.task == f"lab_bench_{dataset.lower()}"]
        assert len(asked) == 50
        for question in asked:
            assert (lab_bench.questions[question.id].prompt, lab_bench.questions[question.id].answer) == (question.prompt, question.answer)
