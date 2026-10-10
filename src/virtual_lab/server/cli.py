"""Starting the server, from the command line as virtual-lab-server, or from Python with serve."""

import argparse
import ipaddress
import os
import socket
import threading
import webbrowser
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import uvicorn

from virtual_lab.server.app import create_app
from virtual_lab.server.security import LOCAL_HOSTS, new_token
from virtual_lab.ui.workspace import DEFAULT_WORKSPACE

# The first port tried when none is asked for, and how many after it are, if it is taken
FIRST_PORT = 8765
PORTS_TRIED = 100

# Where the built pages of the interface are kept in an install that has them
PACKAGED_PAGES = Path(__file__).parent / "static"

# Seconds the server gives a page that is still connected to leave when it is stopped, which a stream of events
# does not do by itself
SHUTDOWN_GRACE_SECONDS = 3


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def listen(host: str, port: int | None) -> socket.socket:
    """A socket listening on the port asked for, or on the first free one from FIRST_PORT if none was.

    :raises OSError: If the port asked for is not free, or none of those tried were.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    candidates = [port] if port is not None else range(FIRST_PORT, FIRST_PORT + PORTS_TRIED)
    error: OSError | None = None
    for candidate in candidates:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate))
        except OSError as failure:
            sock.close()
            error = failure
            continue
        sock.listen(128)
        sock.set_inheritable(True)

        return sock

    assert error is not None
    raise error


def address_of(host: str, sock: socket.socket) -> str:
    port = sock.getsockname()[1]
    name = f"[{host}]" if ":" in host else host

    return f"http://{name}:{port}"


def serve(
    workspace: Path | str = DEFAULT_WORKSPACE,
    host: str = "127.0.0.1",
    port: int | None = None,
    token: str | None = None,
    allowed_hosts: Collection[str] = (),
    static_dir: Path | str | None = None,
    open_browser: bool = True,
    check_keys: bool = True,
    client: Any = None,
    chat_models: Any = None,
) -> None:
    """Starts the server and waits until it is stopped.

    It prints the address to open, which has the token in it: whoever opens it is the one the server serves.

    :param workspace: Where conversations and settings are kept.
    :param host: The address to listen on; 127.0.0.1 for this machine alone.
    :param port: The port, or None for the first free one from 8765.
    :param token: What the API wants from every request, or None for one made now.
    :param allowed_hosts: The names the server may be reached by, other than this machine's. A server that
        listens beyond this machine needs the name it is reached by.
    :param static_dir: The built pages of the interface, or None for those the install has, if it has any.
    :param open_browser: Whether to open the address in a browser, if there are pages to show.
    :param check_keys: Whether to say a key is not set before something is paid for.
    :param client: An OpenAI client for every conversation, as Chat takes one.
    :param chat_models: Chat models for every conversation, as Chat takes them.
    :raises ValueError: If it would listen beyond this machine without being told what it is reached by.
    :raises OSError: If the port is not free.
    """
    if not is_loopback(host) and not allowed_hosts:
        raise ValueError(
            "A server that listens beyond this machine needs the names it is reached by, as allowed hosts: "
            "without them it turns every request away"
        )
    pages = Path(static_dir) if static_dir is not None else (PACKAGED_PAGES if PACKAGED_PAGES.is_dir() else None)
    token = token or new_token()
    app = create_app(
        workspace,
        token=token,
        allowed_hosts=(*LOCAL_HOSTS, *allowed_hosts),
        static_dir=pages,
        check_keys=check_keys,
        client=client,
        chat_models=chat_models,
    )
    sock = listen(host, port)
    address = address_of(host, sock)
    link = f"{address}/?token={token}"
    if pages is None:
        print(f"The Virtual Lab's server is running, with no pages to show. Its API is at {address}/api")
    else:
        print(f"The Virtual Lab is running. Open {link}")
    if not is_loopback(host):
        print("It listens beyond this machine: whoever has this link can spend on your keys and run code.")
    if open_browser and pages is not None:
        threading.Timer(0.5, webbrowser.open, args=(link,)).start()

    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS,
    )
    uvicorn.Server(config).run(sockets=[sock])


def main(argv: Sequence[str] | None = None) -> None:
    """virtual-lab-server: starts the server."""
    parser = argparse.ArgumentParser(prog="virtual-lab-server", description="The Virtual Lab's server.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="where everything run is kept")
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on (default: this machine alone)")
    parser.add_argument("--port", type=int, default=None, help=f"the port (default: the first free from {FIRST_PORT})")
    parser.add_argument(
        "--token",
        default=os.environ.get("VIRTUAL_LAB_SERVER_TOKEN"),
        help="what the API wants from every request (default: $VIRTUAL_LAB_SERVER_TOKEN, or one made for this run)",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="a name the server may be reached by, other than this machine's, which can be given again",
    )
    parser.add_argument("--static-dir", default=None, help="the built pages of the interface to serve")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in a browser")
    options = parser.parse_args(argv)

    try:
        serve(
            workspace=options.workspace,
            host=options.host,
            port=options.port,
            token=options.token,
            allowed_hosts=options.allowed_host,
            static_dir=options.static_dir,
            open_browser=not options.no_browser,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
