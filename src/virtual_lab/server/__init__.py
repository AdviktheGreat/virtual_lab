"""The Virtual Lab's server, which a page talks to: conversations with the head of a lab, whose events it
follows as they happen, the files they make, and what is set up for them.

It needs FastAPI and Uvicorn: pip install "virtual-lab[server]". Start it with serve, or from the command
line with virtual-lab-server.
"""

from collections.abc import Sequence
from types import ModuleType
from typing import Any

NEEDS = 'The server needs FastAPI and Uvicorn: pip install "virtual-lab[server]"'


def launcher() -> ModuleType:
    """The module that starts the server, which needs FastAPI and Uvicorn, said so plainly if they are not installed."""
    try:
        from virtual_lab.server import cli
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] not in ("fastapi", "uvicorn", "starlette", "pydantic"):
            raise
        raise ImportError(NEEDS) from error

    return cli


def serve(*args: Any, **kwargs: Any) -> None:
    """Starts the server; see virtual_lab.server.cli.serve."""
    launcher().serve(*args, **kwargs)  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> None:
    """virtual-lab-server: starts the server; see virtual_lab.server.cli.main."""
    try:
        cli = launcher()
    except ImportError as error:
        raise SystemExit(f"virtual-lab-server: {error}") from None
    cli.main(argv)  # type: ignore[attr-defined]


__all__ = ["main", "serve"]
