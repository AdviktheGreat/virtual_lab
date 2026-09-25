"""Tests for getting a meeting's code out as files, safely."""

import json

import pytest
from pydantic import ValidationError

from virtual_lab.agent import Agent
from virtual_lab.artifacts import (
    ARTIFACT_DIR_NAME,
    CodeArtifacts,
    CodeFile,
    UnsafeFilenameError,
    check_filename,
    save_artifacts,
)
from virtual_lab.run_meeting import run_meeting

from conftest import FakeClient, parsed_response


def code_file(filename: str = "rank_mutations.py", contents: str = "print('hello')") -> CodeFile:
    return CodeFile(
        filename=filename,
        language="python",
        description="Ranks mutations.",
        contents=contents,
    )


ARTIFACTS = CodeArtifacts(
    files=[
        code_file("rank_mutations.py", "import esm\nprint('ranking')\n"),
        code_file("src/scoring.py", "def score(seq):\n    return 0.0\n"),
    ]
)


class TestFilenameSafety:
    """Filenames come from a model, so they are untrusted input."""

    @pytest.mark.parametrize(
        "filename",
        [
            "../escape.py",
            "../../etc/passwd",
            "src/../../escape.py",
            "/etc/passwd",
            "/tmp/evil.py",
            "~/.bashrc",
            "~root/.ssh/authorized_keys",
            "C:\\Windows\\System32\\evil.py",
            "src\\windows\\style.py",
            "with:colon.py",
            "",
            "   ",
            ".",
            "..",
            "with\x00null.py",
        ],
    )
    def test_dangerous_filenames_are_rejected(self, filename: str) -> None:
        with pytest.raises(UnsafeFilenameError):
            check_filename(filename)

    @pytest.mark.parametrize(
        "filename",
        [
            "script.py",
            "src/module.py",
            "deeply/nested/path/module.py",
            "data.csv",
            "README.md",
            "name with spaces.py",
            ".gitignore",
            "dot.in.name.py",
        ],
    )
    def test_ordinary_filenames_are_accepted(self, filename: str) -> None:
        assert check_filename(filename) == filename

    def test_the_schema_enforces_it(self) -> None:
        with pytest.raises(ValidationError, match="traverse upwards"):
            code_file("../escape.py")

    def test_duplicate_filenames_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            CodeArtifacts(files=[code_file("a.py"), code_file("a.py")])


class TestSaveArtifacts:
    def test_files_are_written_with_their_contents(self, tmp_path) -> None:
        paths = save_artifacts(save_dir=tmp_path, save_name="discussion", artifacts=ARTIFACTS)

        assert len(paths) == 2
        assert paths[0].read_text() == "import esm\nprint('ranking')\n"
        assert paths[1].read_text() == "def score(seq):\n    return 0.0\n"

    def test_files_go_under_the_meeting_directory(self, tmp_path) -> None:
        paths = save_artifacts(save_dir=tmp_path, save_name="discussion", artifacts=ARTIFACTS)
        base = tmp_path / ARTIFACT_DIR_NAME / "discussion"

        assert paths[0] == base / "rank_mutations.py"
        assert all(path.is_relative_to(base) for path in paths)

    def test_subdirectories_are_created(self, tmp_path) -> None:
        save_artifacts(save_dir=tmp_path, save_name="discussion", artifacts=ARTIFACTS)

        assert (tmp_path / ARTIFACT_DIR_NAME / "discussion" / "src" / "scoring.py").exists()

    def test_two_meetings_do_not_collide(self, tmp_path) -> None:
        save_artifacts(save_dir=tmp_path, save_name="discussion_1", artifacts=ARTIFACTS)
        save_artifacts(
            save_dir=tmp_path,
            save_name="discussion_2",
            artifacts=CodeArtifacts(files=[code_file("rank_mutations.py", "different\n")]),
        )

        base = tmp_path / ARTIFACT_DIR_NAME

        assert (base / "discussion_1" / "rank_mutations.py").read_text().startswith("import esm")
        assert (base / "discussion_2" / "rank_mutations.py").read_text() == "different\n"

    def test_artifacts_are_not_matched_by_transcript_globs(self, tmp_path) -> None:
        save_artifacts(save_dir=tmp_path, save_name="discussion_1", artifacts=ARTIFACTS)

        assert list(tmp_path.glob("discussion_*.json")) == []

    def test_a_path_that_escapes_after_resolution_is_refused(self, tmp_path) -> None:
        # Defence in depth: bypass the schema the way a future caller might and confirm the
        # writer still refuses rather than trusting that validation already happened
        artifacts = CodeArtifacts.model_construct(
            files=[CodeFile.model_construct(filename="../escaped.py", contents="x", language="python", description="d")]
        )

        with pytest.raises(UnsafeFilenameError, match="outside"):
            save_artifacts(save_dir=tmp_path, save_name="discussion", artifacts=artifacts)

        assert not (tmp_path / "escaped.py").exists()
        assert not (tmp_path / ARTIFACT_DIR_NAME / "escaped.py").exists()

    def test_nothing_is_written_outside_the_save_dir(self, tmp_path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        inside = tmp_path / "meeting"

        save_artifacts(save_dir=inside, save_name="discussion", artifacts=ARTIFACTS)

        assert list(outside.iterdir()) == []


class TestMeetingProducingArtifacts:
    def test_a_meeting_can_hand_back_files(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=ARTIFACTS)]

        result = run_meeting(
            meeting_type="individual",
            agenda="Write a script that ranks mutations.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
            output_schema=CodeArtifacts,
        )

        assert isinstance(result, CodeArtifacts)
        assert [file.filename for file in result.files] == [
            "rank_mutations.py",
            "src/scoring.py",
        ]

    def test_the_code_reaches_disk_and_is_readable(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        # The end of the copy-and-paste path: code goes from a transcript to a runnable file
        fake_client.completions.parsed_responses = [parsed_response(parsed=ARTIFACTS)]

        result = run_meeting(
            meeting_type="individual",
            agenda="Write a script that ranks mutations.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
            output_schema=CodeArtifacts,
        )
        paths = save_artifacts(save_dir=tmp_path, save_name="discussion", artifacts=result)

        assert paths[0].read_text() == "import esm\nprint('ranking')\n"
        assert paths[0].suffix == ".py"

    def test_the_files_also_appear_in_the_saved_output(
        self, fake_client: FakeClient, team_member: Agent, tmp_path
    ) -> None:
        fake_client.completions.parsed_responses = [parsed_response(parsed=ARTIFACTS)]

        run_meeting(
            meeting_type="individual",
            agenda="Write a script.",
            save_dir=tmp_path,
            team_member=team_member,
            num_rounds=0,
            output_schema=CodeArtifacts,
        )

        saved = json.loads((tmp_path / "outputs" / "discussion.json").read_text())

        assert saved["files"][0]["filename"] == "rank_mutations.py"
