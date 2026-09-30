"""Code as an agent's action: reading the code an agent wrote in its turn, and what it gets back.

An agent can run code in the meeting's session in one of two ways. With tool calling, it calls
the run_code tool (see virtual_lab.session.session_tool). Without it, it writes the code between
<execute> and </execute> in its reply, the convention Biomni's agent uses, and is answered with
what the code printed between <observation> and </observation>. The second works with any
model, including ones served without tool support.

The markers Biomni uses to choose a language are kept: a block that begins with #!R is R, and
one that begins with #!BASH or #!CLI is a shell script. Anything else is Python.
"""

import json
import re
from dataclasses import dataclass

from virtual_lab.session import CellResult

EXECUTE_OPEN = "<execute>"
EXECUTE_CLOSE = "</execute>"

# Biomni's markers, with the looser comment forms its own parser accepts
LANGUAGE_MARKERS = (
    (re.compile(r"^(?:#!R\b|# R code\b|# R script\b)", re.IGNORECASE), "r"),
    (re.compile(r"^(?:#!BASH\b|#!CLI\b|# Bash script\b)", re.IGNORECASE), "bash"),
)

# A fence a model wraps its code in out of habit, which would be a syntax error if run
FENCE = re.compile(r"^```[\w+-]*[ \t]*\n(?P<code>.*?)\n?```\s*$", re.DOTALL)


@dataclass(frozen=True)
class CodeAction:
    """Code an agent asked to run by writing it in its reply.

    :param said: The reply up to and including the end of the block. Anything after it was
        written without seeing the code's result, and is dropped: a model that keeps going past
        </execute> tends to invent the output it expects.
    :param language: "python", "r", or "bash".
    :param code: The code, without its language marker.
    """

    said: str
    language: str
    code: str


def find_code_action(content: str) -> CodeAction | None:
    """Finds the first block of code in a reply, if it has one.

    A block missing its closing tag runs to the end of the reply, since a model that stopped
    at its token limit, or forgot the tag, still meant the code to run.

    :param content: The reply.
    :return: The first block, or None if the reply has none.
    """
    start = content.find(EXECUTE_OPEN)
    if start == -1:
        return None

    body_start = start + len(EXECUTE_OPEN)
    end = content.find(EXECUTE_CLOSE, body_start)
    if end == -1:
        body, said = content[body_start:], content + EXECUTE_CLOSE
    else:
        body, said = content[body_start:end], content[: end + len(EXECUTE_CLOSE)]

    code = body.strip()
    if (fenced := FENCE.match(code)) is not None:
        code = fenced.group("code").strip()

    language = "python"
    for marker, marked in LANGUAGE_MARKERS:
        if (found := marker.match(code)) is not None:
            language, code = marked, code[found.end() :].strip()
            break

    return CodeAction(said=said, language=language, code=code)


def observation(result: CellResult, runs_left: int) -> str:
    """What an agent is told after its code ran, in the form Biomni's agent is told it.

    :param result: What happened.
    :param runs_left: How many more blocks the agent may run in this turn.
    :return: The message.
    """
    note = (
        "\n\nThat was the last code you can run in this turn. Give your contribution to the "
        "discussion now, without an <execute> block."
        if runs_left <= 0
        else ""
    )

    return f"<observation>\n{result.report()}\n</observation>{note}"


def describe_code_run(language: str, code: str, report: str) -> str:
    """Shows a run of code in the transcript: the code, then what came of it."""
    fence = "```"
    while fence in code:
        fence += "`"

    return f"{fence}{language}\n{code}\n{fence}\n\n{report}"


def describe_tool_output(tool_call, output: str, code_tool: str) -> str:  # type: ignore[no-untyped-def]
    """Shows a tool's output in the transcript, with the code it ran if it ran code.

    The model is sent only the output, since it wrote the code itself; a reader of the
    transcript is not, and would otherwise see results without what produced them.
    """
    function = getattr(tool_call, "function", None)
    if getattr(function, "name", None) != code_tool:
        return output

    try:
        arguments = json.loads(function.arguments)
    except (TypeError, ValueError):
        return output
    if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
        return output

    return describe_code_run(str(arguments.get("language") or "python"), arguments["code"], output)
