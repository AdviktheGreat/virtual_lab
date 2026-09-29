"""Tests for getting a meeting's code out as files, safely."""

import json
import os
import sys
import unicodedata

import pytest
from pydantic import ValidationError

import virtual_lab.artifacts as artifacts_module
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

    @pytest.mark.parametrize(
        "first,second",
        [
            ("A.py", "a.py"),
            ("caf\u00e9.py", "cafe\u0301.py"),
            ("./a.py", "a.py"),
            ("src//a.py", "src/a.py"),
            ("src/a.py/", "src/a.py"),
            ("Stra\u00dfe.py", "strasse.py"),
            ("A\u0345\u0301.py", "a\u0301\u0345.py"),
        ],
        ids=["case", "unicode normalization", "dot segment", "double slash", "trailing slash",
             "case folding", "combining marks in another order"],
    )
    def test_names_for_the_same_file_on_disk_are_rejected(self, first: str, second: str) -> None:
        # On macOS each pair names one file, so saving both would silently keep only the second
        with pytest.raises(ValidationError, match="unique"):
            CodeArtifacts(files=[code_file(first), code_file(second)])

    def test_folding_a_normalized_name_leaves_it_normalized(self) -> None:
        # filename_key relies on this to skip the second normalization of the caseless match
        unnormalized = []

        for code_point in range(sys.maxunicode + 1):
            if 0xD800 <= code_point <= 0xDFFF:
                continue
            folded = unicodedata.normalize("NFD", chr(code_point)).casefold()
            if unicodedata.normalize("NFD", folded) != folded:
                unnormalized.append(f"U+{code_point:04X}")

        assert unnormalized == []

    @pytest.mark.parametrize(
        "names", [("sub", "sub/x.py"), ("deep/x.py", "DEEP"), ("a/b/c.py", "a/b")]
    )
    def test_a_file_that_is_another_files_directory_is_rejected(self, names) -> None:
        with pytest.raises(ValidationError, match="cannot also be a directory"):
            CodeArtifacts(files=[code_file(name) for name in names])

    def test_a_file_beside_a_directory_of_a_similar_name_is_accepted(self) -> None:
        artifacts = CodeArtifacts(files=[code_file("sub.py"), code_file("sub/x.py")])

        assert len(artifacts.files) == 2


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

    def test_a_link_left_where_a_file_goes_is_replaced_not_followed(self, tmp_path) -> None:
        # Code from an earlier attempt can leave a link behind in the directory it was run in
        secret = tmp_path / "outside" / "secret"
        secret.parent.mkdir()
        secret.write_text("untouched")
        base = tmp_path / "meeting" / ARTIFACT_DIR_NAME / "discussion"
        base.mkdir(parents=True)
        (base / "data.py").symlink_to(secret)

        paths = save_artifacts(
            save_dir=tmp_path / "meeting",
            save_name="discussion",
            artifacts=CodeArtifacts(files=[code_file("data.py", "print('new')")]),
        )

        assert secret.read_text() == "untouched"
        assert not paths[0].is_symlink()
        assert paths[0].read_text() == "print('new')"

    def test_a_link_that_appears_after_the_check_is_not_followed(
        self, tmp_path, monkeypatch
    ) -> None:
        secret = tmp_path / "outside" / "secret"
        secret.parent.mkdir()
        secret.write_text("untouched")
        base = tmp_path / "meeting" / ARTIFACT_DIR_NAME / "discussion"
        base.mkdir(parents=True)
        (base / "data.py").symlink_to(secret)
        # Stands in for code that creates the link between the check and the open
        monkeypatch.setattr(artifacts_module, "is_leftover", lambda path: False)

        with pytest.raises(UnsafeFilenameError, match="Cannot write"):
            save_artifacts(
                save_dir=tmp_path / "meeting",
                save_name="discussion",
                artifacts=CodeArtifacts(files=[code_file("data.py")]),
            )

        assert secret.read_text() == "untouched"

    def test_a_dangling_link_does_not_create_its_target(self, tmp_path) -> None:
        target = tmp_path / "outside" / "planted.py"
        target.parent.mkdir()
        base = tmp_path / "meeting" / ARTIFACT_DIR_NAME / "discussion"
        base.mkdir(parents=True)
        (base / "data.py").symlink_to(target)

        save_artifacts(
            save_dir=tmp_path / "meeting",
            save_name="discussion",
            artifacts=CodeArtifacts(files=[code_file("data.py")]),
        )

        assert not target.exists()
        assert (base / "data.py").read_text() == "print('hello')"

    def test_a_pipe_left_where_a_file_goes_is_replaced_rather_than_waited_on(
        self, tmp_path
    ) -> None:
        # Opening a named pipe to write blocks until something reads it, so one left by code in
        # the sandbox would hang the next repair attempt for good
        base = tmp_path / "meeting" / ARTIFACT_DIR_NAME / "discussion"
        (base / "src").mkdir(parents=True)
        os.mkfifo(base / "data.py")
        os.mkfifo(base / "pipe")

        save_artifacts(
            save_dir=tmp_path / "meeting",
            save_name="discussion",
            artifacts=CodeArtifacts(
                files=[code_file("data.py"), code_file("pipe/module.py")]
            ),
        )

        assert (base / "data.py").read_text() == "print('hello')"
        assert (base / "pipe" / "module.py").read_text() == "print('hello')"

    def test_a_pipe_that_appears_after_the_check_is_not_waited_on(
        self, tmp_path, monkeypatch
    ) -> None:
        base = tmp_path / "meeting" / ARTIFACT_DIR_NAME / "discussion"
        base.mkdir(parents=True)
        os.mkfifo(base / "data.py")
        monkeypatch.setattr(artifacts_module, "is_leftover", lambda path: False)

        with pytest.raises(UnsafeFilenameError, match="Cannot write"):
            save_artifacts(
                save_dir=tmp_path / "meeting",
                save_name="discussion",
                artifacts=CodeArtifacts(files=[code_file("data.py")]),
            )

    def test_a_link_left_where_a_directory_goes_is_replaced_not_followed(self, tmp_path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        base = tmp_path / "meeting" / ARTIFACT_DIR_NAME / "discussion"
        base.mkdir(parents=True)
        (base / "src").symlink_to(outside, target_is_directory=True)

        save_artifacts(
            save_dir=tmp_path / "meeting",
            save_name="discussion",
            artifacts=CodeArtifacts(files=[code_file("src/module.py")]),
        )

        assert list(outside.iterdir()) == []
        assert not (base / "src").is_symlink()
        assert (base / "src" / "module.py").read_text() == "print('hello')"

    def test_a_file_that_cannot_be_written_is_reported_as_unsafe(self, tmp_path) -> None:
        # An earlier attempt's code wrote a file where this attempt needs a directory
        base = tmp_path / ARTIFACT_DIR_NAME / "discussion"
        base.mkdir(parents=True)
        (base / "sub").write_text("left by the code")

        with pytest.raises(UnsafeFilenameError, match="Cannot write"):
            save_artifacts(
                save_dir=tmp_path,
                save_name="discussion",
                artifacts=CodeArtifacts(files=[code_file("sub/x.py")]),
            )

    def test_the_directory_itself_is_not_a_file(self, tmp_path) -> None:
        artifacts = CodeArtifacts.model_construct(
            files=[CodeFile.model_construct(filename="sub/..", contents="x", language="python",
                                            description="d")]
        )

        with pytest.raises(UnsafeFilenameError, match="outside"):
            save_artifacts(save_dir=tmp_path, save_name="discussion", artifacts=artifacts)

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
