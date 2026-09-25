"""Tests that the declared environment can actually install and run the code."""

import sys
import tomllib
from pathlib import Path

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def load_pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def test_python_floor_matches_the_syntax_used() -> None:
    # The library uses backslashes inside f-string expressions and PEP 695 type parameters,
    # neither of which parses before 3.12.
    assert load_pyproject()["project"]["requires-python"] == ">=3.12"


def test_no_classifiers_below_the_python_floor() -> None:
    classifiers = load_pyproject()["project"]["classifiers"]

    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" not in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers


def test_nanobody_design_extra_covers_its_imports() -> None:
    extras = load_pyproject()["project"]["optional-dependencies"]["nanobody-design"]
    names = {requirement.split("==")[0].split(">=")[0].strip() for requirement in extras}

    # scipy is imported by nanobody_design/scripts and was missing from this extra
    for required in {"scipy", "torch", "biopython", "pandas", "seaborn", "transformers"}:
        assert required in names, f"{required} missing from the nanobody-design extra"


def test_running_on_a_supported_interpreter() -> None:
    assert sys.version_info >= (3, 12)
