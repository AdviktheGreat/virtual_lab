"""Sending requests to the chat completions API.

Every request a meeting makes goes through here, for one reason: which parameters a model
accepts is not written down anywhere a program can read. GPT-5, GPT-5 mini and nano and the pro
models reject any temperature but the default; GPT-5.1 and later accept one only when reasoning
is off; and a model released next month will have its own rule. A request that sends a
temperature to a model that refuses one fails with a 400 on the first call of the meeting, and
a table of which models accept it is out of date as soon as it is written.

So the request is sent as asked, and if the API refuses the temperature it is sent again
without one, and the model is remembered so that the rest of the process does not pay for the
refusal again. The meeting record says which models ran at their default temperature, since a
transcript produced at 1.0 is not the same experiment as one produced at 0.2.
"""

import threading
from typing import Callable, TypeVar

import openai

ResponseT = TypeVar("ResponseT")

# Models that have refused a temperature in this process. Shared on purpose: the refusal is a
# property of the model, not of the meeting, and every meeting after the first would otherwise
# spend a failed request finding it out again.
MODELS_WITHOUT_TEMPERATURE: set[str] = set()
_LOCK = threading.Lock()


# The codes for a model that takes no temperature, as against one that takes this temperature
# but not the value given. A value out of range is also reported against "temperature", and
# treating it as a refusal would drop the caller's mistake silently and stop sending any
# temperature to that model for the rest of the process.
UNSUPPORTED_CODES = frozenset({"unsupported_value", "unsupported_parameter"})


def check_temperature(temperature: float | None) -> None:
    """Refuses a temperature no model accepts, before it is sent.

    :param temperature: The sampling temperature, or None for the model's default.
    :raises ValueError: If the temperature is outside 0 to 2.
    """
    # Written this way round so that NaN, which compares false with everything, is refused
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError(f"temperature must be between 0 and 2, not {temperature}")


# How providers without OpenAI's error codes say a model takes no temperature. A value out of
# range is worded differently ("must be", "range"), and is the caller's mistake to see.
REFUSAL_PHRASES = (
    "not supported",
    "unsupported",
    "does not support",
    "only the default",
    "may only be set",
    "cannot be used",
    "cannot both be specified",
    "is deprecated",
)


def rejects_temperature(error: BaseException) -> bool:
    """Whether a refused request was refused because the model takes no temperature.

    OpenAI names the offending parameter and gives a code, which are the parts of its error
    meant to be read by a program. Anthropic and the OpenAI-compatible servers give neither, so
    for them the message is read, and only a 400 that names the temperature as unsupported
    counts.

    :param error: The error the request raised.
    :return: Whether removing the temperature could make the request succeed.
    """
    if isinstance(error, openai.BadRequestError) and getattr(error, "code", None) is not None:
        return (
            getattr(error, "param", None) == "temperature"
            and getattr(error, "code", None) in UNSUPPORTED_CODES
        )

    if getattr(error, "status_code", None) != 400:
        return False

    message = str(getattr(error, "message", None) or error).lower()

    return "temperature" in message and any(phrase in message for phrase in REFUSAL_PHRASES)


def send_request(
    send: Callable[[float | None], ResponseT],
    model: str,
    temperature: float | None,
) -> ResponseT:
    """Sends a request, without its temperature if the model will not accept one.

    :param send: Sends the request at the temperature given, where None means send none.
    :param model: The model being asked, which is remembered if it refuses a temperature.
    :param temperature: The sampling temperature wanted, or None for the model's default.
    :raises ValueError: If the temperature is outside 0 to 2.
    :raises Exception: Whatever the request raised, if it was refused for any other reason.
    :return: The response.
    """
    check_temperature(temperature)

    if temperature is not None and model not in MODELS_WITHOUT_TEMPERATURE:
        try:
            return send(temperature)
        except Exception as error:
            if not rejects_temperature(error):
                raise

            with _LOCK:
                MODELS_WITHOUT_TEMPERATURE.add(model)

            print(
                f'Warning: "{model}" does not accept a temperature of {temperature}, so it is '
                f"running at its default. Results will vary more between runs than asked for."
            )

    return send(None)


def ran_without_temperature(models: list[str]) -> list[str]:
    """Which of some models have run at their default temperature in this process.

    :param models: The models to ask about.
    :return: Those that refused a temperature, in the order given.
    """
    return [model for model in models if model in MODELS_WITHOUT_TEMPERATURE]
