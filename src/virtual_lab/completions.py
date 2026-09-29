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
from typing import Any, Callable, TypeVar

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


def rejects_temperature(error: openai.BadRequestError) -> bool:
    """Whether a refused request was refused because the model takes no temperature.

    The API names the offending parameter and gives a code, which are the parts of the error
    meant to be read by a program; the message wording has changed between models.

    :param error: The error the API returned.
    :return: Whether removing the temperature could make the request succeed.
    """
    return (
        getattr(error, "param", None) == "temperature"
        and getattr(error, "code", None) in UNSUPPORTED_CODES
    )


def send_request(
    send: Callable[..., ResponseT],
    model: str,
    temperature: float | None,
    **kwargs: Any,
) -> ResponseT:
    """Sends a request, without its temperature if the model will not accept one.

    :param send: The client method to call, such as client.chat.completions.create.
    :param model: The model to ask.
    :param temperature: The sampling temperature wanted, or None for the model's default.
    :param kwargs: The rest of the request.
    :raises ValueError: If the temperature is outside 0 to 2.
    :raises openai.BadRequestError: If the request is refused for any other reason.
    :return: The response.
    """
    check_temperature(temperature)

    if temperature is not None and model not in MODELS_WITHOUT_TEMPERATURE:
        try:
            return send(model=model, temperature=temperature, **kwargs)
        except openai.BadRequestError as error:
            if not rejects_temperature(error):
                raise

            with _LOCK:
                MODELS_WITHOUT_TEMPERATURE.add(model)

            print(
                f'Warning: "{model}" does not accept a temperature of {temperature}, so it is '
                f"running at its default. Results will vary more between runs than asked for."
            )

    return send(model=model, **kwargs)


def ran_without_temperature(models: list[str]) -> list[str]:
    """Which of some models have run at their default temperature in this process.

    :param models: The models to ask about.
    :return: Those that refused a temperature, in the order given.
    """
    return [model for model in models if model in MODELS_WITHOUT_TEMPERATURE]
