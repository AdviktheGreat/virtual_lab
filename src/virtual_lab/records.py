"""What every lookup needs, whatever it is looking up.

Each source gets its own module, because the shape of a protein record has nothing to do with
the shape of a compound record and combining them only makes both harder to read. What they do
share is how a failure is classified, how a field too long to send is shortened, and how a
document written elsewhere is parsed, and those belong in one place so that a missing record
means the same thing whichever service was asked.
"""

import math
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from typing import Any

from virtual_lab.constants import MAX_FIELD_CHARACTERS


# Statuses that mean "there is no such record", as opposed to "the service could not answer".
# Only these become a RecordNotFoundError: telling an agent a compound does not exist because the
# service returned 503 is worse than telling it nothing, since it will stop looking. Services
# disagree on which status to use, so both are here: UniProt answers an unknown accession with
# 400, PubChem answers an unknown name with 404.
NO_SUCH_RECORD_STATUSES = frozenset({400, 404})

# Largest whole number a field is believed to hold. Counts, years, and identifiers are all far
# below it, and a number past it is damage that would otherwise print as hundreds of digits.
LARGEST_WHOLE_NUMBER = 10**15


class DatabaseError(Exception):
    """Raised when a database cannot answer a question about an identifier."""


class RecordNotFoundError(DatabaseError):
    """Raised when an identifier is well formed but names nothing."""


def bounded(value: int, most: int, name: str) -> int:
    """Holds a count a model chose within what is worth requesting.

    :param value: The requested count.
    :param most: The largest value allowed.
    :param name: What is being counted, for the error message.
    :raises ValueError: If the value is not a whole number, or is below one.
    :return: The count, reduced to the limit if it was over it.
    """
    # The value arrives from a model's JSON, so it can be a string, a float, or a boolean, which
    # Python counts as a whole number. A string failed the comparison below with TypeError, and
    # NaN passed it, since every comparison with NaN is false.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number, not {truncate_text(repr(value), 40)}")

    if value < 1:
        raise ValueError(f"{name} must be at least 1")

    return min(value, most)


def truncate_text(text: str, limit: int) -> str:
    """Shortens a piece of free text for display, saying how much it left out.

    A curated record carries kilobytes of prose. Capping the sequence and then spending the
    saving on a disease description is no saving at all.

    :param text: The full text.
    :param limit: The most characters to show.
    :return: The text, or its start with a note of how much was left out.
    """
    if len(text) <= limit:
        return text

    return f"{text[:limit]}... ({len(text) - limit:,} of {len(text):,} characters not shown)"


def as_dict(value: Any) -> dict[str, Any]:
    """Returns a value that should be a JSON object, or an empty one if it is anything else."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """Returns a value that should be a JSON array, or an empty one if it is anything else."""
    return value if isinstance(value, list) else []


def as_text(value: Any, limit: int | None = MAX_FIELD_CHARACTERS) -> str:
    """Reads a short text field, shortened to what a report can hold.

    :param value: The value as the service sent it.
    :param limit: The most characters to keep, or None for a field that is kept whole because
        its report shortens it, where shortening it twice would miscount what was left out.
    :return: The text, or the empty string for a null, a flag, or a nested value, none of which
        is text and all of which str() would render as though they were.
    """
    if value is None or isinstance(value, (bool, dict, list)):
        return ""

    return str(value) if limit is None else truncate_text(str(value), limit)


def listing(items: Sequence[str], most: int, separator: str = ", ") -> str:
    """Joins the first few of a list, and says how many more there are.

    :param items: The items.
    :param most: The most to name.
    :param separator: What goes between two items.
    :return: The items joined.
    """
    shown = separator.join(items[:most])

    return f"{shown} and {len(items) - most:,} more" if len(items) > most else shown


def number_or_none(value: Any) -> float | None:
    """Reads a number that a service may have sent as a string.

    ChEMBL sends max_phase as "4.0" and molecular weight as "180.16", and PubChem sends molecular
    weight as "180.16" too. Comparing those to a number silently does the wrong thing, and
    formatting them assumes a type they do not have. Infinity and NaN are refused as well: float()
    accepts "nan" and "1e999", and neither is a measurement.

    :param value: The value as the service sent it.
    :return: The number, or None if there was not one.
    """
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def as_int(value: Any, default: int | None = None) -> int | None:
    """Reads a whole number that a service may have sent as a string or as a float.

    int() is the wrong tool here. It accepts True as 1, raises on "\u00b2", which str.isdigit
    accepts, raises OverflowError on infinity, and refuses a string past 4,300 digits.

    :param value: The value as the service sent it.
    :param default: What to return if the value is not a whole number.
    :return: The number, or the default.
    """
    if isinstance(value, bool):
        return default

    if isinstance(value, float):
        value = int(value) if value.is_integer() else default
    elif isinstance(value, str):
        stripped = value.strip()
        digits = stripped.removeprefix("-")
        value = (
            int(stripped)
            if digits.isascii() and digits.isdecimal() and len(digits) <= 16
            else default
        )

    if not isinstance(value, int) or abs(value) > LARGEST_WHOLE_NUMBER:
        return default

    return value


def parse_xml(body: str, error: type[Exception], subject: str = "response") -> ElementTree.Element:
    """Parses XML from elsewhere, refusing a document that declares entities.

    ElementTree resolves internal entities, and a document declaring a few nested ones expands to
    whatever size its author chose while the parser holds the result in memory. It is not a
    theoretical concern: the parser here does expand them, which was checked rather than assumed.
    Nothing this library parses has any reason to carry a DOCTYPE, so the declaration is refused
    outright instead of the expansion being bounded.

    :param body: The document.
    :param error: The exception to raise, so that a caller sees a failure of the kind it handles
        rather than one belonging to another module.
    :param subject: What the document is, for the message.
    :raises error: If the body declares a document type, or is not XML.
    :return: The root element.
    """
    # Searched for anywhere rather than near the start: the declaration may follow a comment or a
    # processing instruction of any length, and a prefix that stopped at 4 KB let a 6.6 KB feed
    # through that expanded to five megabytes. An entity declaration is refused on its own too,
    # though a parser will only read one inside a DOCTYPE.
    if re.search(r"<!(DOCTYPE|ENTITY)", body, re.IGNORECASE):
        raise error(f"Refusing to parse a {subject} that declares a document type")

    try:
        return ElementTree.fromstring(body)
    except ElementTree.ParseError as problem:
        raise error(f"The {subject} was not usable XML: {problem}") from problem
