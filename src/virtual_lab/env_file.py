"""API keys and other settings read from a .env file, as Biomni reads them.

Biomni loads the .env file in the directory it is run from when its agent is imported, without
replacing what the environment already holds. virtual_lab does the same when it is imported, and
load_env reads any other file. What is read reaches this process alone: a session's code is
given none of it unless it is forwarded by name.
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# The file read when virtual_lab is imported, from the directory Python was started in
ENV_FILE_NAME = ".env"

# Set to "0" to keep virtual_lab from reading the .env file when it is imported
LOAD_ENV_VARIABLE = "VIRTUAL_LAB_LOAD_ENV"


def load_env(path: str | os.PathLike[str] = ENV_FILE_NAME, override: bool = False) -> list[str]:
    """Sets environment variables from a .env file.

    Each line sets one, as NAME=value, and may use ${NAME} to refer to a variable set before it;
    see python-dotenv for the whole format.

    :param path: The file, defaulting to .env in the working directory.
    :param override: Whether to replace variables that are already set. By default those keep
        their values, as Biomni keeps them, so that a key set in the shell outranks the file's.
    :raises FileNotFoundError: If there is no such file.
    :return: The names of the variables it set, or changed, sorted.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"There is no .env file at {file.resolve()}")

    before = dict(os.environ)
    load_dotenv(file, override=override, encoding="utf-8")

    return sorted(name for name, value in os.environ.items() if before.get(name) != value)


def load_default_env() -> None:
    """Reads the .env file in the working directory, if there is one, without replacing what is
    set, unless VIRTUAL_LAB_LOAD_ENV is "0". Nothing is printed, since a server speaking over
    standard output must write nothing else there."""
    if os.environ.get(LOAD_ENV_VARIABLE, "").strip() == "0":
        return

    try:
        file = Path.cwd() / ENV_FILE_NAME
        if file.is_file():
            load_env(file)
    except (OSError, UnicodeDecodeError) as error:
        # Importing virtual_lab must not fail for a file it was not asked to read
        warnings.warn(f"Could not read {ENV_FILE_NAME}: {error}", UserWarning, stacklevel=2)
