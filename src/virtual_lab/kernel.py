"""The process that holds a session's state, run inside the sandbox.

This file is the program a session starts, passed to the interpreter whole with "python3 -c",
so that it runs in any image with a Python 3 and nothing else. It therefore imports only the
standard library, and nothing from virtual_lab, which is not installed where it runs.

It reads one request per line from its standard input, as JSON, and answers each with one line
of JSON. Before the first request it moves that conversation onto file descriptors of its own
and points standard input at /dev/null, so that code which prints, reads input, or starts a
program that writes straight to descriptor 1 cannot corrupt it. While a request runs,
descriptors 1 and 2 are a pipe read by a thread here, which keeps the start and the end of what
arrives and counts what it drops, so that code printing without bound costs neither memory nor
disk.

Python code runs in one namespace that outlives each request, the way a notebook's cells do. R
and shell code run as a fresh Rscript or bash process each time, as they do in Biomni.
"""

from __future__ import annotations

import ast
import json
import linecache
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings

# Figures are saved to files, never shown, so the backend must not need a display
os.environ.setdefault("MPLBACKEND", "Agg")

# Most bytes of output kept from one request, half from its start and half from its end
MAX_OUTPUT_BYTES = 50_000

# Seconds to wait for output still in the pipe once code has finished. A program the code left
# running in the background can hold the pipe open indefinitely, and is not waited for.
DRAIN_SECONDS = 2.0

# Where figures left open by Python code are saved, relative to the working directory
PLOT_DIR = "plots"

INTERPRETERS = {"r": ("Rscript",), "bash": ("bash",)}


class CellTimeout(BaseException):
    """Raised in the code when its time is up.

    A BaseException, like KeyboardInterrupt, so that the "except Exception" code commonly wraps
    its work in does not swallow it and carry on.
    """


class Capture:
    """Reads a pipe to its end, keeping its first and last bytes."""

    def __init__(self, descriptor: int, limit: int = MAX_OUTPUT_BYTES) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.half = limit // 2
        self.dropped = 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.read, args=(descriptor,), daemon=True)
        self.thread.start()

    def read(self, descriptor: int) -> None:
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                with self.lock:
                    room = self.half - len(self.head)
                    if room > 0:
                        self.head += chunk[:room]
                        chunk = chunk[room:]
                    self.tail += chunk
                    excess = len(self.tail) - self.half
                    if excess > 0:
                        del self.tail[:excess]
                        self.dropped += excess

    def text(self, wait: float) -> tuple[str, int]:
        self.thread.join(wait)
        with self.lock:
            head, tail, dropped = bytes(self.head), bytes(self.tail), self.dropped

        if not dropped:
            return (head + tail).decode("utf-8", errors="replace"), 0

        # The cut is at a byte count, so it can fall inside a character; the bytes of it that
        # are left at the start of the tail would decode as a replacement character
        start = 0
        while start < min(3, len(tail)) and 0x80 <= tail[start] <= 0xBF:
            start += 1
        dropped += start

        return (
            head.decode("utf-8", errors="replace")
            + f"\n[... {dropped:,} bytes of output truncated ...]\n"
            + tail[start:].decode("utf-8", errors="replace")
        ), dropped


class Kernel:
    """Runs requests one at a time, keeping the Python namespace between them."""

    def __init__(self) -> None:
        self.namespace: dict = {"__name__": "__main__", "__builtins__": __builtins__}
        self.cells = 0
        self.figures = 0
        # Fixed at the start, so that code which changes directory still saves where it is told
        self.home = os.getcwd()
        self.can_time_out = hasattr(signal, "setitimer")
        warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")

    def handle(self, request: dict) -> dict:
        self.cells += 1
        language = str(request.get("language", "python")).strip().casefold()
        code = str(request.get("code", ""))
        timeout = float(request.get("timeout") or 0)

        read_end, write_end = os.pipe()
        capture = Capture(read_end)
        saved = (os.dup(1), os.dup(2))
        flush()
        os.dup2(write_end, 1)
        os.dup2(write_end, 2)
        os.close(write_end)

        start = time.monotonic()
        status, error, plots = "ok", None, []

        try:
            self.start_timer(timeout)
            try:
                if language == "python":
                    error = self.run_python(code)
                elif language in INTERPRETERS:
                    error = self.run_program(INTERPRETERS[language], code, language)
                else:
                    error = f"Unknown language {language!r}: use python, r, or bash."
            finally:
                self.stop_timer()
        except CellTimeout:
            status, error = "timeout", f"Stopped at the time limit of {timeout:g} s."
            print(f"\n{error} Variables set before this point are kept.", file=sys.stderr)
        finally:
            if language == "python":
                plots = self.save_figures()
            flush()
            # Code that replaced the streams would otherwise leave every later cell silent
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])

        if error is not None and status == "ok":
            status = "error"

        output, dropped = capture.text(DRAIN_SECONDS)

        return {
            "id": request.get("id"),
            "status": status,
            "error": error,
            "output": output,
            "output_dropped": dropped,
            "plots": plots,
            "duration": time.monotonic() - start,
        }

    def start_timer(self, timeout: float) -> None:
        if timeout > 0 and self.can_time_out:
            signal.signal(signal.SIGALRM, raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout)

    def stop_timer(self) -> None:
        if self.can_time_out:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def run_python(self, code: str) -> str | None:
        """Runs a cell, printing the value of its last line if that is an expression."""
        filename = f"<cell {self.cells}>"
        # Registered so that a traceback can quote the lines of the cell that raised
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)

        try:
            tree = ast.parse(code, filename, "exec")
        except (SyntaxError, ValueError) as error:
            # Raised by the parser, whose own frames are no help in finding the mistake
            message = "".join(traceback.format_exception_only(type(error), error))
            sys.stderr.write(message)
            return message.strip()

        try:
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, filename, "exec"), self.namespace)
            if last is not None:
                value = eval(compile(last, filename, "eval"), self.namespace)
                if value is not None:
                    self.namespace["_"] = value
                    print(repr(value))
        except SystemExit as exit:
            return f"The code called exit({exit.code!r}). The session is still running."
        except CellTimeout:
            raise
        except BaseException as error:
            # The first frame is the exec above, which is not the code's own
            frames = error.__traceback__.tb_next if error.__traceback__ else None
            traceback.print_exception(type(error), error, frames)
            return "".join(traceback.format_exception_only(type(error), error)).strip()

        return None

    def run_program(self, interpreter: tuple[str, ...], code: str, language: str) -> str | None:
        """Runs R or shell code as a program of its own, in a process group of its own."""
        suffix = ".R" if language == "r" else ".sh"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as file:
            file.write(code)
        try:
            try:
                process = subprocess.Popen(
                    [*interpreter, file.name], stdin=subprocess.DEVNULL, start_new_session=True
                )
            except FileNotFoundError:
                return f"{interpreter[0]} is not installed in this environment."
            try:
                returncode = process.wait()
            except BaseException:
                kill_group(process)
                raise
            # Anything it left running in the background would outlive the request otherwise
            kill_group(process)
        finally:
            os.unlink(file.name)

        return None if returncode == 0 else f"{interpreter[0]} exited with code {returncode}."

    def save_figures(self) -> list[str]:
        """Saves every figure the code left open, then closes it."""
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is None:
            return []

        saved = []
        try:
            for number in pyplot.get_fignums():
                figure = pyplot.figure(number)
                os.makedirs(os.path.join(self.home, PLOT_DIR), exist_ok=True)
                # A restarted session counts from one again, and must not overwrite the figures
                # an earlier one saved
                while True:
                    self.figures += 1
                    name = f"{PLOT_DIR}/figure_{self.figures}.png"
                    if not os.path.exists(os.path.join(self.home, name)):
                        break
                figure.savefig(os.path.join(self.home, name), dpi=100, bbox_inches="tight")
                pyplot.close(figure)
                saved.append(name)
        except Exception as error:
            print(f"Could not save a figure: {type(error).__name__}: {error}", file=sys.stderr)

        return saved


def raise_timeout(signum: int, frame: object) -> None:
    raise CellTimeout()


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        pass
    process.wait()


def flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def main() -> None:
    inbox = os.fdopen(os.dup(0), "r", encoding="utf-8")
    outbox = os.fdopen(os.dup(1), "w", encoding="utf-8")
    nothing = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nothing, 0)
    os.close(nothing)
    sys.stdin = open(os.devnull, encoding="utf-8")

    kernel = Kernel()
    outbox.write(json.dumps({"ready": True, "python": sys.version.split()[0], "pid": os.getpid()}) + "\n")
    outbox.flush()

    for line in inbox:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("not an object")
        except ValueError as error:
            # Answered all the same, since the other end waits for one answer per request
            response = {
                "id": None,
                "status": "error",
                "error": f"Unreadable request: {error}",
                "output": "",
                "output_dropped": 0,
                "plots": [],
                "duration": 0.0,
            }
        else:
            response = kernel.handle(request)
        outbox.write(json.dumps(response) + "\n")
        outbox.flush()


if __name__ == "__main__":
    main()
