"""Checks the benchmarks here against Biomni's own code, on the real files.

Both are given the same questions and the same answers, and must agree on every prompt, answer,
and score. Biomni's classes are imported from a clone of its repository, unchanged. Only where
BiomniEval1 reads its file from is redirected: Hugging Face for Biomni, the pinned copy here.

Skipped unless both are given:

    VIRTUAL_LAB_BENCHMARK_DIR  a directory download_benchmarks filled
    VIRTUAL_LAB_BIOMNI_DIR     a clone of https://github.com/snap-stanford/Biomni

and scikit-learn is installed, which Biomni's evaluate functions use.
"""

import contextlib
import io
import json
import os
import sys
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
metrics = pytest.importorskip("sklearn.metrics")

from virtual_lab.benchmarks import (  # noqa: E402
    BiomniEval1,
    BiomniEval1Benchmark,
    HumanitysLastExam,
    LabBench,
    eval1_reward,
)

BENCHMARKS = os.environ.get("VIRTUAL_LAB_BENCHMARK_DIR")
BIOMNI = os.environ.get("VIRTUAL_LAB_BIOMNI_DIR")

pytestmark = pytest.mark.skipif(
    not (BENCHMARKS and BIOMNI),
    reason="set VIRTUAL_LAB_BENCHMARK_DIR (filled by download_benchmarks) and VIRTUAL_LAB_BIOMNI_DIR (a Biomni clone)",
)

EVAL1_URL = "hf://datasets/biomni/Eval1/"

# What a model might answer with, whatever the task: right, differently written, empty, or not an answer
GENERIC_ANSWERS = ["", "None", "wrong", "A", "Z", "{}", "[]", "1", None, 3, "B", "C", "D", "E", "F", "G", "H"]


def directory() -> Path:
    return Path(BENCHMARKS or "")


@pytest.fixture(scope="module")
def biomni() -> Iterator[Any]:
    """Biomni's own modules, imported from its clone."""
    sys.path.insert(0, BIOMNI or "")
    try:
        yield {
            "eval1": import_module("biomni.eval.biomni_eval1").BiomniEval1,
            "lab_bench": import_module("biomni.task.lab_bench").lab_bench,
            "hle": import_module("biomni.task.hle").humanity_last_exam,
        }
    finally:
        sys.path.remove(BIOMNI or "")
        for name in [name for name in sys.modules if name == "biomni" or name.startswith("biomni.")]:
            del sys.modules[name]


def biomni_arrays() -> Any:
    """Biomni indexes numpy arrays of strings, which pandas 3 makes Arrow-backed unless it is told
    not to. Where pandas has no such option it does not need telling."""
    try:
        pd.get_option("future.infer_string")
    except (pd.errors.OptionError, KeyError):
        return contextlib.nullcontext()

    return pd.option_context("future.infer_string", False)


@pytest.fixture(scope="module")
def theirs(biomni: Any) -> Any:
    real = pd.read_parquet

    def redirected(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path).startswith(EVAL1_URL):
            path = directory() / "eval1" / "biomni_eval1_dataset.parquet"
        return real(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch, biomni_arrays(), contextlib.redirect_stdout(io.StringIO()):
        patch.setattr(pd, "read_parquet", redirected)
        return biomni["eval1"]()


@pytest.fixture(scope="module")
def ours() -> BiomniEval1:
    with contextlib.redirect_stdout(io.StringIO()):
        return BiomniEval1(directory())


def outcome(function: Any, *arguments: Any) -> tuple[str, Any]:
    """What a call gave, or the kind of error it raised."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return "gave", function(*arguments)
    except Exception as error:
        return "raised", type(error).__name__


def candidates(task: str, truth: str) -> list[Any]:
    """Answers to a question, the right one written several ways and many wrong ones."""
    answers: list[Any] = [truth, truth.lower(), truth.upper(), f"  {truth}  ", *GENERIC_ANSWERS]
    if task == "rare_disease_diagnosis":
        record = json.loads(truth)
        answers += [
            json.dumps(record),
            repr(record),
            json.dumps({**record, "OMIM_ID": "0"}),
            json.dumps({"OMIM_ID": record.get("OMIM_ID")}),
            json.dumps({"disease_name": record.get("disease_name")}),
            json.dumps([record]),
            "{'OMIM_ID': " + repr(record.get("OMIM_ID")) + "}",
            record,
        ]
    if task == "patient_gene_detection":
        genes = [gene.strip() for gene in truth.split(",")] if "," in truth else [truth]
        answers += [
            json.dumps({"causal_gene": genes}),
            json.dumps({"causal_gene": genes[:1]}),
            json.dumps({"causal_gene": genes[0]}),
            json.dumps({"causal_gene": []}),
            json.dumps({"causal_gene": ["NOT_A_GENE"]}),
            json.dumps({"causal_gene": [*genes, "X"]}),
            repr({"causal_gene": genes}),
            json.dumps(genes),
            json.dumps({"gene": genes}),
            {"causal_gene": genes},
        ]

    return answers


class TestBiomniEval1:
    def test_it_has_the_same_questions_in_the_same_tasks_and_splits(self, theirs: Any, ours: BiomniEval1) -> None:
        assert len(ours) == len(theirs) == 433
        assert ours.list_tasks() == theirs.list_tasks()
        assert ours.get_task_stats() == theirs.get_task_stats()
        for task in ours.list_tasks():
            assert ours.get_task_stats(task) == theirs.get_task_stats(task)

    def test_every_question_has_the_same_prompt_and_answer(self, theirs: Any, ours: BiomniEval1) -> None:
        different = [
            (row["task_name"], row["task_instance_id"])
            for row in ours.rows
            if ours.get_instance(row["task_name"], row["task_instance_id"])
            != theirs.get_instance(row["task_name"], row["task_instance_id"])
        ]

        assert different == []

    @pytest.mark.parametrize("split", [None, "train", "val"])
    def test_each_task_has_the_same_table_of_questions_as_biomni_returns(
        self, theirs: Any, ours: BiomniEval1, split: str | None
    ) -> None:
        for task in ours.list_tasks():
            expected, actual = theirs.get_instances_by_task(task, split), ours.get_instances_by_task(task, split)

            # Only the kind of string pandas stores differs, which depends on how each was read
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_column_type=False)

    def test_every_answer_is_scored_as_biomni_scores_it_or_fails_as_it_fails(
        self, theirs: Any, ours: BiomniEval1
    ) -> None:
        different, scored = [], 0
        for row in ours.rows:
            task, number = row["task_name"], row["task_instance_id"]
            for answer in candidates(task, row["answer"]):
                scored += 1
                if outcome(ours.evaluate, task, number, answer) != outcome(theirs.evaluate, task, number, answer):
                    different.append((task, number, answer))

        assert scored > 7_000
        assert different == []

    @pytest.mark.parametrize("truth", ["G1", "G1,G2", "G1, G2 ,G3", " G1 ", "g1,G2", ""])
    def test_the_causal_genes_a_patient_has_are_matched_as_biomni_matches_them(self, theirs: Any, truth: str) -> None:
        # Every truth in the file is one gene, so the rule for several is tried on truths made up here
        answers: list[Any] = [
            json.dumps({"causal_gene": genes})
            for genes in ([], ["G1"], ["G2"], ["G1", "G2"], ["G3", "X"], ["X"], ["g1"])
        ]
        answers += [
            json.dumps({"causal_gene": "G2"}),
            json.dumps({"causal_gene": "G1,G2"}),
            json.dumps({"causal_gene": None}),
            json.dumps({"causal_gene": ["G1", "G2", "G3"]}),
            repr({"causal_gene": ["G1", "G2"]}),
            {"causal_gene": ["G2"]},
            {"causal_gene": "G1"},
            json.dumps({}),
            "G1",
            None,
        ]

        different = [
            answer
            for answer in answers
            if outcome(eval1_reward, "patient_gene_detection", answer, truth)
            != outcome(theirs._compute_reward, "patient_gene_detection", answer, truth)
        ]

        assert different == []

    def test_a_batch_of_answers_is_scored_as_biomni_scores_it_including_those_that_cannot_be(
        self, theirs: Any, ours: BiomniEval1
    ) -> None:
        batch = [(row["task_name"], row["task_instance_id"], row["answer"]) for row in ours.rows[::7]]
        batch += [("crispr_delivery", 10**6, "x"), ("no_such_task", 0, "x"), ("crispr_delivery", 0, 5)]

        with contextlib.redirect_stdout(io.StringIO()):
            assert ours.batch_evaluate(batch) == theirs.batch_evaluate(batch)

    def test_the_benchmark_scores_the_answers_a_model_gives_as_biomni_does(self, theirs: Any) -> None:
        benchmark = BiomniEval1Benchmark(directory())
        different = []
        for question in benchmark:
            for answer in (question.answer, question.answer.lower(), question.answer.upper(), "wrong"):
                with contextlib.redirect_stdout(io.StringIO()):
                    expected = theirs.evaluate(question.task, question.id, answer)
                if benchmark.score(question, answer) != expected:
                    different.append((question.key, answer))

        assert len(benchmark) == 433 and different == []


@pytest.mark.parametrize("dataset", ["DbQA", "SeqQA"])
class TestLabBench:
    @pytest.fixture
    def pair(self, biomni: Any, dataset: str) -> tuple[Any, LabBench]:
        with biomni_arrays():
            return biomni["lab_bench"](path=str(directory()), dataset=dataset), LabBench(directory(), dataset)

    def test_every_question_has_the_prompt_answer_and_refusal_letter_biomni_gives_it(
        self, pair: tuple[Any, LabBench]
    ) -> None:
        theirs_, ours_ = pair

        assert len(ours_) == len(theirs_.query) and len(ours_) in (60, 70)
        assert [theirs_.get_example(index) for index in range(len(ours_))] == list(ours_.get_iterator())
        assert list(theirs_.refrain_label) == [question.metadata["refrain"] for question in ours_.questions]

    def test_the_answer_is_asked_for_in_the_same_schema(self, pair: tuple[Any, LabBench]) -> None:
        theirs_, ours_ = pair

        assert theirs_.output_class().model_json_schema() == ours_.output_class().model_json_schema()

    @pytest.mark.parametrize("mode", ["right", "some wrong", "half refused", "random", "all refused"])
    def test_answers_are_measured_as_biomni_measures_them(self, pair: tuple[Any, LabBench], mode: str) -> None:
        theirs_, ours_ = pair
        answers, refusals = np.array(theirs_.answer), np.array(theirs_.refrain_label)
        generator = np.random.default_rng(3)
        for _ in range(6):
            response = answers.copy()
            if mode == "some wrong":
                response[generator.random(len(response)) < 0.3] = "A"
            elif mode == "half refused":
                flip = generator.random(len(response)) < 0.5
                response[flip] = refusals[flip]
            elif mode == "random":
                response = generator.choice(list("ABCDEFGH"), size=len(response))
            elif mode == "all refused":
                response = refusals.copy()

            ours_measures = ours_.evaluate(list(response))
            kind, theirs_measures = outcome(theirs_.evaluate, list(response))

            if kind == "raised":
                # Every question refused: Biomni's precision is a mean of nothing, which newer
                # scikit-learn raises on. Here it is None, and the rest is as Biomni's
                assert mode == "all refused"
                assert ours_measures == {"accuracy": 0.0, "coverage": 0.0, "refrain_ratio": 1.0, "precision": None}
                continue

            assert set(ours_measures) == set(theirs_measures)
            for name, value in theirs_measures.items():
                expected = None if np.isnan(value) else value
                assert ours_measures[name] == (pytest.approx(expected) if expected is not None else None)


class TestHumanitysLastExam:
    @pytest.fixture
    def pair(self, biomni: Any) -> tuple[Any, HumanitysLastExam]:
        with biomni_arrays():
            return biomni["hle"](path=str(directory())), HumanitysLastExam(directory())

    def test_every_question_has_the_prompt_and_answer_biomni_gives_it(
        self, pair: tuple[Any, HumanitysLastExam]
    ) -> None:
        theirs_, ours_ = pair

        assert len(ours_) == len(theirs_.query) == 52
        assert list(theirs_.get_iterator()) == list(ours_.get_iterator())
        assert [question.answer for question in ours_.questions] == list(theirs_.answer)
        assert theirs_.output_class().model_json_schema() == ours_.output_class().model_json_schema()

    def test_accuracy_is_what_scikit_learn_makes_of_the_answers(self, pair: tuple[Any, HumanitysLastExam]) -> None:
        theirs_, ours_ = pair
        for trial in range(10):
            response = [
                question.answer if (index + trial) % (trial + 2) else "A"
                for index, question in enumerate(ours_.questions)
            ]

            expected = metrics.accuracy_score(np.array(theirs_.answer), np.array(response))

            assert ours_.evaluate(response)["accuracy"] == pytest.approx(expected)
