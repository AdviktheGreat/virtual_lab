"""Tests for reading settings from a .env file, at import and on request."""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import virtual_lab
from virtual_lab.env_file import LOAD_ENV_VARIABLE, load_default_env, load_env

SOURCE = str(Path(virtual_lab.__file__).parent.parent)


@pytest.fixture(autouse=True)
def restored_environment():
    """load_env sets variables in os.environ itself, which monkeypatch cannot undo."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadEnv:
    def test_variables_are_set_from_the_file(self, tmp_path: Path) -> None:
        os.environ.pop("VL_TEST_KEY", None)
        file = write(tmp_path / "keys.env", "VL_TEST_KEY=secret-value\n# a comment\nVL_TEST_OTHER='quoted'\n")

        assert load_env(file) == ["VL_TEST_KEY", "VL_TEST_OTHER"]
        assert os.environ["VL_TEST_KEY"] == "secret-value"
        assert os.environ["VL_TEST_OTHER"] == "quoted"

    def test_a_variable_already_set_keeps_its_value(self, tmp_path: Path) -> None:
        os.environ["VL_TEST_KEY"] = "from the shell"
        file = write(tmp_path / ".env", "VL_TEST_KEY=from the file\nVL_TEST_NEW=new\n")

        assert load_env(file) == ["VL_TEST_NEW"]
        assert os.environ["VL_TEST_KEY"] == "from the shell"

    def test_override_replaces_it(self, tmp_path: Path) -> None:
        os.environ["VL_TEST_KEY"] = "from the shell"
        file = write(tmp_path / ".env", "VL_TEST_KEY=from the file\n")

        assert load_env(file, override=True) == ["VL_TEST_KEY"]
        assert os.environ["VL_TEST_KEY"] == "from the file"

    def test_a_value_that_is_already_the_same_is_not_counted(self, tmp_path: Path) -> None:
        os.environ["VL_TEST_KEY"] = "same"

        assert load_env(write(tmp_path / ".env", "VL_TEST_KEY=same\n"), override=True) == []

    def test_a_variable_can_refer_to_another(self, tmp_path: Path) -> None:
        os.environ["VL_TEST_HOST"] = "example.org"
        file = write(tmp_path / ".env", "VL_TEST_URL=https://${VL_TEST_HOST}/api\n")

        load_env(file)

        assert os.environ["VL_TEST_URL"] == "https://example.org/api"

    def test_the_default_is_the_working_directorys_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write(tmp_path / ".env", "VL_TEST_KEY=here\n")
        monkeypatch.chdir(tmp_path)

        assert load_env() == ["VL_TEST_KEY"]

    def test_a_path_in_the_home_directory_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        write(tmp_path / "lab.env", "VL_TEST_KEY=home\n")

        assert load_env("~/lab.env") == ["VL_TEST_KEY"]

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no .env file at"):
            load_env(tmp_path / "missing.env")

    def test_a_directory_is_not_a_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_env(tmp_path)


class TestLoadDefaultEnv:
    def test_the_working_directorys_file_is_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        os.environ.pop("VL_TEST_KEY", None)
        monkeypatch.delenv(LOAD_ENV_VARIABLE, raising=False)
        write(tmp_path / ".env", "VL_TEST_KEY=read\n")
        monkeypatch.chdir(tmp_path)

        load_default_env()

        assert os.environ["VL_TEST_KEY"] == "read"

    def test_it_can_be_turned_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        os.environ.pop("VL_TEST_KEY", None)
        monkeypatch.setenv(LOAD_ENV_VARIABLE, "0")
        write(tmp_path / ".env", "VL_TEST_KEY=read\n")
        monkeypatch.chdir(tmp_path)

        load_default_env()

        assert "VL_TEST_KEY" not in os.environ

    def test_no_file_is_no_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LOAD_ENV_VARIABLE, raising=False)
        monkeypatch.chdir(tmp_path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_default_env()

    def test_a_file_that_is_not_utf_8_is_warned_of(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LOAD_ENV_VARIABLE, raising=False)
        (tmp_path / ".env").write_bytes(b"VL_TEST_KEY=\xff\xfe\n")
        monkeypatch.chdir(tmp_path)

        with pytest.warns(UserWarning, match="Could not read .env: 'utf-8' codec can't decode"):
            load_default_env()

    @pytest.mark.skipif(not hasattr(os, "getuid") or os.getuid() == 0, reason="needs a file it cannot read")
    def test_a_file_that_cannot_be_read_is_warned_of(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LOAD_ENV_VARIABLE, raising=False)
        file = write(tmp_path / ".env", "VL_TEST_KEY=read\n")
        file.chmod(0)
        monkeypatch.chdir(tmp_path)

        try:
            with pytest.warns(UserWarning, match="Could not read .env"):
                load_default_env()
        finally:
            file.chmod(0o600)


def import_in(directory: Path, code: str, **environment: str) -> subprocess.CompletedProcess:
    env = {
        name: value
        for name, value in os.environ.items()
        if name != LOAD_ENV_VARIABLE and not name.startswith("VL_TEST_")
    }
    env.update(PYTHONPATH=SOURCE, **environment)

    return subprocess.run(
        [sys.executable, "-c", code], cwd=directory, env=env, capture_output=True, text=True, timeout=120
    )


class TestAtImport:
    CODE = "import os, virtual_lab; print(repr(os.environ.get('VL_TEST_KEY')))"

    def test_importing_virtual_lab_reads_the_file_and_prints_nothing_else(self, tmp_path: Path) -> None:
        write(tmp_path / ".env", "VL_TEST_KEY=at import\n")

        imported = import_in(tmp_path, self.CODE)

        assert imported.returncode == 0, imported.stderr
        assert imported.stdout == "'at import'\n"

    def test_what_the_environment_holds_is_kept(self, tmp_path: Path) -> None:
        write(tmp_path / ".env", "VL_TEST_KEY=at import\n")

        imported = import_in(tmp_path, self.CODE, VL_TEST_KEY="from the shell")

        assert imported.stdout == "'from the shell'\n"

    def test_it_can_be_turned_off(self, tmp_path: Path) -> None:
        write(tmp_path / ".env", "VL_TEST_KEY=at import\n")

        imported = import_in(tmp_path, self.CODE, **{LOAD_ENV_VARIABLE: "0"})

        assert imported.stdout == "None\n"

    def test_importing_a_module_of_the_package_reads_it_too(self, tmp_path: Path) -> None:
        write(tmp_path / ".env", "VL_TEST_KEY=through a module\n")

        imported = import_in(tmp_path, "import os, virtual_lab.tools; print(os.environ['VL_TEST_KEY'])")

        assert imported.stdout == "through a module\n"

    def test_the_package_reads_it_before_importing_its_modules(self) -> None:
        source = Path(virtual_lab.__file__).read_text(encoding="utf-8")

        assert source.index("load_default_env()") < source.index("from virtual_lab.actions")
