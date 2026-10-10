"""The files a conversation made or was given, shown to the page.

What the agents' code writes is not to be trusted with the page it is shown in: a file it wrote can be
HTML or SVG with a script in it. So a file is only ever sent with a type that does not run anything,
and a CSP that would not let it, and a path that leaves the conversation's directory, by ".." or a link, is
not a file at all.
"""

import mimetypes
import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

from starlette.responses import FileResponse

from virtual_lab.server.errors import not_found

# The most files a listing gives, so that a directory of thousands does not make a page slow
MAX_LISTED_FILES = 5000

# Where a file is shown in the page itself, which is for these types and no others
INLINE_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "application/pdf",
        "text/plain",
        "text/csv",
        "text/tab-separated-values",
        "text/markdown",
        "application/json",
    }
)

# Types a browser would run or render as a page, which are sent as plain text instead
RUNNABLE_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/javascript", "application/javascript"})

# What is left out of a listing: what uploads are written to on the way, and what Python leaves behind
HIDDEN_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})
HIDDEN_PREFIXES = (".upload.",)

CONTENT_SECURITY_POLICY = "sandbox; default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"


def resolve_inside(root: Path, relative: str, shown: Collection[str] | None = None) -> Path:
    """A file under root, by its path from it.

    :param shown: The first parts of the paths of the files that are shown, or None for all of them.
    :raises ApiError: If the path is absolute, goes up, leads out of root by a link, or is not of a part that is
        shown, or there is no such file.
    """
    parts = Path(relative).parts
    if relative.startswith("/") or ".." in parts or "\x00" in relative:
        raise not_found("file there")
    if shown is not None and (not parts or parts[0] not in shown):
        raise not_found("file there")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise not_found("file there")

    return path


def is_hidden(path: Path) -> bool:
    return path.name in HIDDEN_NAMES or path.name.startswith(HIDDEN_PREFIXES)


def list_files(
    root: Path, limit: int = MAX_LISTED_FILES, shown: Collection[str] | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """The files under root, by path, with their size and when they were changed, and whether there were more
    than the limit. Links are not followed, and what is hidden is left out.

    :param shown: The first parts of the paths of the files that are shown, or None for all of them.
    """
    found: list[dict[str, Any]] = []
    truncated = False
    resolved = root.resolve()
    for directory, names, files in os.walk(resolved, followlinks=False):
        # What is shown is of the top of the root alone, so what is beside it there is left out
        limited = shown is not None and Path(directory) == resolved
        names[:] = sorted(name for name in names if not is_hidden(Path(name)) and not (limited and name not in shown))
        for name in [] if limited else sorted(files):
            path = Path(directory) / name
            if is_hidden(path) or path.is_symlink():
                continue
            if len(found) >= limit:
                truncated = True
                break
            try:
                stat = path.stat()
            except OSError:
                # The code that wrote it has taken it away since it was seen
                continue
            found.append(
                {
                    "path": path.relative_to(resolved).as_posix(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": media_type(path),
                }
            )
        if truncated:
            break

    return sorted(found, key=lambda item: item["path"]), truncated


def media_type(path: Path) -> str:
    """What a file is sent as: its type, unless a browser would run it, which is sent as text."""
    guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if guessed in RUNNABLE_TYPES:
        return "text/plain"

    return guessed


def file_response(path: Path, download: bool = False) -> FileResponse:
    """A file as a response: shown in the page where its type is one that is safe to show, and saved otherwise."""
    kind = media_type(path)
    inline = kind in INLINE_TYPES and not download
    disposition = "inline" if inline else "attachment"
    response = FileResponse(
        path,
        media_type=f"{kind}; charset=utf-8" if kind.startswith("text/") else kind,
        content_disposition_type=disposition,
        filename=path.name,
    )
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"

    return response
