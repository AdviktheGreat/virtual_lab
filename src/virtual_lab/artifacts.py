"""Code and files a meeting produced, and how to put them on disk safely.

Agents write code into their turns, where nothing can run it. Getting that code out as files is
what lets it be executed, and executing model-authored content means the filenames are untrusted
input. Everything here treats them that way.
"""

import os
import stat
import unicodedata
from collections import Counter
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


def filename_key(filename: str) -> str:
    """Reduces a filename to what decides which file it names on disk.

    "./a.py" and "a.py" are the same file everywhere. "A.py" and "a.py" are the same file on the
    case-insensitive filesystems macOS and Windows use by default, and a name typed with an
    accented letter and the same name with the accent as a separate combining mark are the same
    file on macOS. Treating all of these as one name is stricter than a case-sensitive Linux disk
    needs, and it is what stops one file from silently replacing another on the others.

    The comparison is Unicode's canonical caseless match. Normalizing before case folding is
    part of it: folding can turn a combining mark into a letter, which then no longer moves into
    canonical order, so the same name with its marks typed in another order would otherwise
    differ. The match also normalizes after folding, which is left out because folding a
    normalized name leaves it normalized; a test checks that for every character.

    :param filename: A filename that has passed check_filename.
    :return: The key, equal for any two filenames that could name the same file.
    """
    return unicodedata.normalize("NFD", PurePosixPath(filename).as_posix()).casefold()


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
        """Rejects two files claiming the same path, where one would overwrite the other.

        A file whose path is another file's directory is rejected too, since the two cannot
        both be written.
        """
        keys = Counter(filename_key(file.filename) for file in self.files)
        duplicates = sorted(
            {file.filename for file in self.files if keys[filename_key(file.filename)] > 1}
        )

        if duplicates:
            raise ValueError(f"Filenames must be unique; repeated: {', '.join(duplicates)}")

        clashes = []

        for file in self.files:
            directories = PurePosixPath(filename_key(file.filename)).parents[:-1]
            if any(directory.as_posix() in keys for directory in directories):
                clashes.append(file.filename)

        if clashes:
            raise ValueError(
                "A file cannot also be a directory; these are inside a path another file "
                f"takes: {', '.join(sorted(clashes))}"
            )

        return self


def save_artifacts(save_dir: Path, save_name: str, artifacts: CodeArtifacts) -> tuple[Path, ...]:
    """Writes a meeting's files into their own directory.

    Each path is re-checked against the target directory when it is written, not only when it
    was parsed, so that a filename which slips past the schema still cannot escape.

    The directory is run in between attempts, so the code may have left symbolic links in it,
    including where the next attempt's files go. Such a link is removed and replaced by the file
    or directory, never followed, so it can neither redirect a write outside the directory nor
    stop the next attempt from being written.

    :param save_dir: The directory the transcript was saved in.
    :param save_name: The name the transcript was saved under.
    :param artifacts: The files to write.
    :raises UnsafeFilenameError: If a path would fall outside the target directory, or a file
        cannot be written where its path puts it.
    :return: The paths written, in order.
    """
    base_dir = (save_dir / ARTIFACT_DIR_NAME / save_name).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for file in artifacts.files:
        path = Path(os.path.normpath(base_dir / file.filename))

        if not path.is_relative_to(base_dir) or path == base_dir:
            raise UnsafeFilenameError(
                f'Unsafe filename "{file.filename}": resolves to {path}, outside {base_dir}'
            )

        try:
            write_inside(base_dir=base_dir, path=path, contents=file.contents)
        except OSError as error:
            raise UnsafeFilenameError(
                f'Cannot write "{file.filename}" to {path}: {error}'
            ) from error

        written.append(path)

    return tuple(written)


def is_leftover(path: Path) -> bool:
    """Whether a path holds something other than a file or directory, such as a link or a pipe.

    Code run in the directory can leave any of these behind. A link would be followed out of the
    directory, and a named pipe blocks whoever opens it for writing until something reads it,
    which nothing will.

    :param path: The path to look at, without following it.
    :return: Whether what is there should be removed before writing.
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False

    return not (stat.S_ISDIR(mode) or stat.S_ISREG(mode))


def write_inside(base_dir: Path, path: Path, contents: str) -> None:
    """Writes a file under a directory without following any symbolic link on the way.

    :param base_dir: The directory, already resolved.
    :param path: The file to write, lexically inside the directory.
    :param contents: What to write.
    :raises OSError: If the file or one of its directories cannot be created.
    """
    current = base_dir

    for part in path.relative_to(base_dir).parts[:-1]:
        current = current / part
        if is_leftover(current):
            current.unlink()
        current.mkdir(exist_ok=True)

    if is_leftover(path):
        path.unlink()

    # O_NOFOLLOW makes the open fail rather than follow a link created since the check above, and
    # O_NONBLOCK makes it fail rather than wait on a pipe; neither changes writing a plain file
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_TRUNC
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )

    with open(os.open(path, flags, 0o666), "w") as f:
        f.write(contents)
