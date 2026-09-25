"""Code and files a meeting produced, and how to put them on disk safely.

Agents write code into their turns, where nothing can run it. Getting that code out as files is
what lets it be executed, and executing model-authored content means the filenames are untrusted
input. Everything here treats them that way.
"""

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator

from virtual_lab.constants import ARTIFACT_DIR_NAME


class UnsafeFilenameError(ValueError):
    """Raised when a model asks for a filename that would write outside its directory."""


def check_filename(filename: str) -> str:
    """Checks that a filename is a plain relative path inside its own directory.

    A meeting's filenames come from a model, so they are treated as untrusted. Absolute paths,
    parent traversal, home expansion, Windows drive letters, and null bytes are all rejected
    rather than normalized, because a request for any of them is a sign something is wrong and
    guessing at the intent would be worse than refusing.

    :param filename: The filename to check.
    :raises UnsafeFilenameError: If the filename is not a safe relative path.
    :return: The filename, unchanged.
    """
    if not filename or not filename.strip():
        raise UnsafeFilenameError("A filename may not be empty")

    if "\x00" in filename:
        raise UnsafeFilenameError("A filename may not contain a null byte")

    if "\\" in filename or ":" in filename:
        raise UnsafeFilenameError(
            f'Unsafe filename "{filename}": use forward slashes and no drive letters'
        )

    if filename.startswith("~"):
        raise UnsafeFilenameError(f'Unsafe filename "{filename}": must not start with "~"')

    path = PurePosixPath(filename)

    if path.is_absolute():
        raise UnsafeFilenameError(f'Unsafe filename "{filename}": must be relative')

    if any(part == ".." for part in path.parts):
        raise UnsafeFilenameError(f'Unsafe filename "{filename}": must not traverse upwards')

    if path.name in {"", ".", ".."}:
        raise UnsafeFilenameError(f'Unsafe filename "{filename}": must name a file')

    return filename


class CodeFile(BaseModel):
    """One file a meeting produced."""

    filename: str = Field(
        description="The file's path relative to the output directory, for example "
        "'rank_mutations.py' or 'src/esm_scoring.py'. It must not be absolute or contain '..'."
    )
    language: str = Field(
        description="The language the file is written in, lowercase, for example 'python', 'r', "
        "or 'bash'. Use 'text' for data or documentation."
    )
    description: str = Field(description="What the file does, in one sentence.")
    contents: str = Field(
        description="The complete contents of the file. It must be runnable as written, with no "
        "placeholders, no pseudocode, and no omitted sections."
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Rejects a filename that would write outside its directory."""
        return check_filename(filename)


class CodeArtifacts(BaseModel):
    """The files a meeting produced, ready to be written out and run."""

    files: list[CodeFile] = Field(
        description="Every file needed to run the work discussed, including any helper modules."
    )

    @model_validator(mode="after")
    def check_filenames_are_unique(self) -> "CodeArtifacts":
        """Rejects two files claiming the same path, where one would overwrite the other."""
        filenames = [file.filename for file in self.files]
        duplicates = sorted({name for name in filenames if filenames.count(name) > 1})

        if duplicates:
            raise ValueError(f"Filenames must be unique; repeated: {', '.join(duplicates)}")

        return self


def save_artifacts(save_dir: Path, save_name: str, artifacts: CodeArtifacts) -> tuple[Path, ...]:
    """Writes a meeting's files into their own directory.

    Each path is re-checked against the target directory after being resolved, not only when it
    was parsed, so that a filename which slips past the schema still cannot escape.

    :param save_dir: The directory the transcript was saved in.
    :param save_name: The name the transcript was saved under.
    :param artifacts: The files to write.
    :raises UnsafeFilenameError: If a resolved path would fall outside the target directory.
    :return: The paths written, in order.
    """
    base_dir = (save_dir / ARTIFACT_DIR_NAME / save_name).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for file in artifacts.files:
        path = (base_dir / file.filename).resolve()

        if not path.is_relative_to(base_dir):
            raise UnsafeFilenameError(
                f'Unsafe filename "{file.filename}": resolves to {path}, outside {base_dir}'
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(file.contents)

        written.append(path)

    return tuple(written)
