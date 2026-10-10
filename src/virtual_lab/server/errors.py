"""What the server answers with when it cannot do what it was asked."""

from typing import Any


class ApiError(Exception):
    """An error the server tells of, as an HTTP status and a body of the same shape every time:
    {"error": {"code": ..., "message": ..., ...}}.

    :param status: The HTTP status.
    :param code: What went wrong, in a word a page can act on, such as "busy" or "not_found".
    :param message: What went wrong, for a person to read.
    :param details: Anything else the page may use, such as the variables that are missing.
    """

    def __init__(self, status: int, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, **self.details}}


def not_found(what: str) -> ApiError:
    return ApiError(404, "not_found", f"There is no {what}")
