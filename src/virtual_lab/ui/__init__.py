"""The Virtual Lab's web interface, where a person sets up a meeting or a project, follows it as
it happens, steers it, and reads back everything the lab has done.

It needs Gradio: pip install "virtual-lab[ui]". Start it with launch_ui, or from the command
line with virtual-lab-ui.
"""

from types import ModuleType
from typing import Any


def interface() -> ModuleType:
    """The module of the interface, which needs Gradio, said so plainly if Gradio is not installed."""
    try:
        from virtual_lab.ui import app
    except ModuleNotFoundError as error:
        if error.name != "gradio":
            raise
        raise ImportError('The web interface needs Gradio: pip install "virtual-lab[ui]"') from error

    return app


def launch_ui(*args: Any, **kwargs: Any) -> Any:
    """Starts the web interface; see virtual_lab.ui.app.launch_ui."""
    return interface().launch_ui(*args, **kwargs)  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> None:
    """virtual-lab-ui: starts the web interface; see virtual_lab.ui.app.main."""
    try:
        app = interface()
    except ImportError as error:
        raise SystemExit(f"virtual-lab-ui: {error}") from None
    app.main(argv)  # type: ignore[attr-defined]


__all__ = ["launch_ui", "main"]
