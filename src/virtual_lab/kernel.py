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

A session can also define functions in that namespace that run outside it, on the host: tools
of the user's own, which may need what the sandbox does not have, such as the network, a key, or
a GPU. Calling one writes the call to the session as a line of JSON, beside the answers, and
waits for the reply the session writes back on standard input, so each line arriving there is
either a request or the reply to a call.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import keyword
import linecache
import os
import queue
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

# Most characters of an error message sent back. Output is bounded by Capture, but an exception
# can carry a message of any size, and an answer too long to read would end the session.
MAX_ERROR_CHARS = 2_000

# Most figures saved from one request. The rest are closed unsaved, for the same reason.
MAX_FIGURES = 50

INTERPRETERS = {"r": ("Rscript",), "bash": ("bash",)}

# Most bytes of a call to a host tool, unless the session says otherwise. The session reads each
# line written to it up to a limit, and a longer one would end it.
MAX_CALL_BYTES = 2 * 1024**2

# Seconds between looks at the time limit while waiting for a host tool's reply. The signal that
# stops code at its limit can be delivered to another thread, which a wait without a timeout
# would not notice until the reply came.
REPLY_POLL_SECONDS = 0.1


class CellTimeout(BaseException):
    """Raised in the code when its time is up.

    A BaseException, like KeyboardInterrupt, so that the "except Exception" code commonly wraps
    its work in does not swallow it and carry on.
    """


class Unreadable:
    """A line that could not be read as JSON, kept to be answered as a request would be."""

    def __init__(self, error: str) -> None:
        self.error = error


class HostToolError(RuntimeError):
    """Raised in the code when a host tool it called failed, or could not be called."""


class Default:
    """Stands for a parameter's default, which the host tool applies itself, so that leaving the
    parameter out and passing its default are the same call. Shown as the default, where known."""

    def __init__(self, shown: str) -> None:
        self.shown = shown

    def __repr__(self) -> str:
        return self.shown


@contextlib.contextmanager
def alarm_deferred():
    """Holds the time limit's signal back while a line is written, since a line cut short would
    run into the next one and leave the session unable to read either."""
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    else:
        yield


def to_json(value: object) -> object:
    """Converts what json cannot write itself: numpy arrays and scalars, sets, and paths."""
    for method in ("tolist", "item"):
        if callable(getattr(value, method, None)):
            try:
                return getattr(value, method)()
            except Exception:
                pass
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)

    raise TypeError(
        f"a {type(value).__name__} cannot be passed as JSON; convert it first, a table to a list of "
        "rows with .to_dict('records'), say, or write it to a file and pass the file's path"
    )


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
        self.outbox = None
        self.write_lock = threading.Lock()
        # Guards which request is running, and the calls waiting for replies
        self.calls = threading.Condition()
        self.running = None
        self.call_count = 0
        self.replies: dict = {}
        self.max_call_bytes = MAX_CALL_BYTES

    def send(self, message: dict) -> None:
        """Writes one line to the session."""
        line = json.dumps(message) + "\n"
        with self.write_lock:
            self.outbox.write(line)
            self.outbox.flush()

    def handle(self, request: dict) -> dict:
        with self.calls:
            self.running = request.get("id")
        try:
            return self.run_request(request)
        finally:
            self.end_calls()

    def end_calls(self) -> None:
        """Ends the calls still waiting when the code that made them has finished.

        Only a thread the code started can still be waiting. The session stops listening for
        calls once it has the answer, so a call made after it would never be answered.
        """
        with self.calls:
            self.running = None
            self.replies.clear()
            self.calls.notify_all()

    def run_request(self, request: dict) -> dict:
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
            restore_streams()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])

        if error is not None and status == "ok":
            status = "error"

        if error is not None and len(error) > MAX_ERROR_CHARS:
            error = error[:MAX_ERROR_CHARS] + f" [... {len(error) - MAX_ERROR_CHARS:,} characters truncated]"

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

    def deliver(self, reply: dict) -> None:
        """Hands a reply to the call waiting for it. A reply nothing waits for is dropped: the
        call was given up at a time limit, or the code that made it has finished."""
        with self.calls:
            call = reply.get("reply")
            if call in self.replies and self.replies[call] is None:
                self.replies[call] = reply
                self.calls.notify_all()

    def call_host(self, name: str, arguments: dict) -> object:
        """Calls a host tool and waits for what it returns."""
        with self.calls:
            request = self.running
            if request is None:
                raise HostToolError(
                    f"{name} can only be called while code is running in the session, and the code "
                    "that started this thread has finished"
                )
            try:
                line = json.dumps(
                    {"call": self.call_count + 1, "request": request, "tool": name, "arguments": arguments},
                    default=to_json,
                )
            except (TypeError, ValueError) as error:
                raise TypeError(f"The arguments to {name} must be JSON: {error}") from None
            if len(line) + 1 > self.max_call_bytes:
                raise ValueError(
                    f"The arguments to {name} are {len(line):,} bytes of JSON, more than the "
                    f"{self.max_call_bytes - 1:,} a call can carry. Write them to a file and pass its path."
                )
            self.call_count += 1
            call = self.call_count
            self.replies[call] = None
            # Written while the lock is held, so that it cannot follow the request's answer
            with alarm_deferred(), self.write_lock:
                self.outbox.write(line + "\n")
                self.outbox.flush()

            try:
                while self.replies.get(call) is None:
                    if self.running != request or call not in self.replies:
                        raise HostToolError(f"{name} was still running when the code that called it finished")
                    self.calls.wait(REPLY_POLL_SECONDS)
                reply = self.replies[call]
            finally:
                self.replies.pop(call, None)

        if reply.get("error") is not None:
            raise HostToolError(f"{name} failed: {reply['error']}")

        return reply.get("result")

    def define_tools(self, request: dict) -> dict:
        """Defines a function in the namespace for each host tool, which calls it."""
        answer = {
            "id": request.get("id"),
            "status": "ok",
            "error": None,
            "output": "",
            "output_dropped": 0,
            "plots": [],
            "duration": 0.0,
        }
        try:
            self.max_call_bytes = int(request.get("max_call_bytes") or MAX_CALL_BYTES)
            stubs = {str(spec["name"]): self.make_stub(spec) for spec in request["tools"]}
        except Exception as error:
            answer.update(status="error", error=f"Could not define the tools: {type(error).__name__}: {error}")
            return answer
        self.namespace.update(stubs)
        return answer

    def make_stub(self, spec: dict):
        """A function taking the tool's parameters, as its signature on the host orders them."""
        name = str(spec["name"])
        parameters = []
        untakeable = False
        for item in spec.get("parameters", []):
            parameter = str(item["name"])
            # A parameter Python cannot name, such as "max-results", is passed with **
            if not parameter.isidentifier() or keyword.iskeyword(parameter):
                untakeable = True
                continue
            kind = inspect.Parameter.KEYWORD_ONLY if item.get("keyword_only") else inspect.Parameter.POSITIONAL_OR_KEYWORD
            default = inspect.Parameter.empty if item.get("required") else Default(str(item.get("default", "<default>")))
            parameters.append(inspect.Parameter(parameter, kind, default=default))
        if untakeable:
            rest = "arguments"
            while rest in {parameter.name for parameter in parameters}:
                rest = f"_{rest}"
            parameters.append(inspect.Parameter(rest, inspect.Parameter.VAR_KEYWORD))
        signature = inspect.Signature(parameters)
        kernel = self

        def stub(*args, **kwargs):
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError as error:
                raise TypeError(f"{name}() {error}") from None
            arguments = {}
            for key, value in bound.arguments.items():
                if signature.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                    arguments.update(value)
                else:
                    arguments[key] = value
            return kernel.call_host(name, arguments)

        stub.__name__ = stub.__qualname__ = name
        stub.__signature__ = signature
        stub.__doc__ = str(spec.get("description") or "")
        stub.__module__ = "host_tools"

        return stub

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
            # The first frame is the exec above, which is not the code's own, and the kernel's
            # frames below the code's, where it called a host tool, are not either
            frames = error.__traceback__.tb_next if error.__traceback__ else None
            report = traceback.TracebackException(type(error), error, frames)
            report.stack = traceback.StackSummary.from_list(
                [frame for frame in report.stack if frame.filename != KERNEL_FILE]
            )
            sys.stderr.write("".join(report.format()))
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
            numbers = pyplot.get_fignums()
            for number in numbers[MAX_FIGURES:]:
                pyplot.close(pyplot.figure(number))
            if len(numbers) > MAX_FIGURES:
                print(f"Only the first {MAX_FIGURES} of {len(numbers)} open figures were saved.", file=sys.stderr)
            for number in numbers[:MAX_FIGURES]:
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


# Where the kernel's own code is, as a traceback names it: "<string>" when run with -c
KERNEL_FILE = raise_timeout.__code__.co_filename


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        pass
    process.wait()


def restore_streams() -> None:
    """Gives the next request working standard streams, whatever this one did to them.

    Code that replaced sys.stdout would otherwise leave every later request silent, and code
    that closed it would leave every later print raising.
    """
    for name, descriptor in (("stdout", 1), ("stderr", 2)):
        original = getattr(sys, f"__{name}__")
        if original is None or original.closed:
            original = open(
                descriptor, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False
            )
            setattr(sys, f"__{name}__", original)
        setattr(sys, name, original)


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
    # Between requests, and after each one, descriptor 1 is the kernel's stderr rather than the
    # answers: a thread the code started that prints later would otherwise write into them
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, encoding="utf-8")

    kernel = Kernel()
    kernel.outbox = outbox
    kernel.send({"ready": True, "python": sys.version.split()[0], "pid": os.getpid()})

    # Read in a thread of its own, so that the reply to a host tool's call is read while the
    # code that made the call is still running
    requests: queue.Queue = queue.Queue()

    def read_inbox() -> None:
        for line in inbox:
            try:
                message = json.loads(line)
            except ValueError as error:
                requests.put(Unreadable(str(error)))
                continue
            if isinstance(message, dict) and "reply" in message:
                kernel.deliver(message)
            else:
                requests.put(message)
        requests.put(None)

    threading.Thread(target=read_inbox, daemon=True).start()

    while (request := requests.get()) is not None:
        if isinstance(request, dict) and "tools" in request:
            response = kernel.define_tools(request)
        elif isinstance(request, dict):
            response = kernel.handle(request)
        else:
            # Answered all the same, since the other end waits for one answer per request
            response = {
                "id": None,
                "status": "error",
                "error": f"Unreadable request: {request.error if isinstance(request, Unreadable) else 'not an object'}",
                "output": "",
                "output_dropped": 0,
                "plots": [],
                "duration": 0.0,
            }
        kernel.send(response)


if __name__ == "__main__":
    main()
