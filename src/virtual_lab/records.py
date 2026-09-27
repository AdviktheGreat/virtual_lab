"""What every database lookup needs, whatever it is looking up.

Each database gets its own module, because the shape of a protein record has nothing to do with
the shape of a compound record and combining them only makes both harder to read. What they do
share is how a failure is classified and how a field too long to send is shortened, and those
belong in one place so that a missing record means the same thing whichever service was asked.
"""


# Statuses that mean "there is no such record", as opposed to "the service could not answer".
# Only these become a RecordNotFoundError: telling an agent a compound does not exist because the
# service returned 503 is worse than telling it nothing, since it will stop looking. Services
# disagree on which status to use, so both are here: UniProt answers an unknown accession with
# 400, PubChem answers an unknown name with 404.
NO_SUCH_RECORD_STATUSES = frozenset({400, 404})


class DatabaseError(Exception):
    """Raised when a database cannot answer a question about an identifier."""


class RecordNotFoundError(DatabaseError):
    """Raised when an identifier is well formed but names nothing."""


def bounded(value: int, most: int, name: str) -> int:
    """Holds a count a model chose within what is worth requesting.

    :param value: The requested count.
    :param most: The largest value allowed.
    :param name: What is being counted, for the error message.
    :raises ValueError: If the value is below one.
    :return: The count, reduced to the limit if it was over it.
    """
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
