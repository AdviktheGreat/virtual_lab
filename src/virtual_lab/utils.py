"""Contains useful utility functions."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam

from virtual_lab.constants import (
    CONTEXT_WARNING_THRESHOLD,
    DEFAULT_ENCODING,
    DEFAULT_FINETUNING_EPOCHS,
    FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN,
    MODEL_TO_INPUT_PRICE_PER_TOKEN,
    MODEL_TO_MAX_INPUT_TOKENS,
    MODEL_TO_OUTPUT_PRICE_PER_TOKEN,
    TOKENS_PER_MESSAGE,
)
from virtual_lab.prompts import format_references
from virtual_lab.web import WebRequestError, build_url, request_json


def get_pubmed_central_article(pmcid: str, abstract_only: bool = False) -> tuple[str | None, list[str] | None]:
    """Gets the title and content (abstract or full text) of a PubMed Central article given a PMC ID.

    Note: This only returns main text, ignoring tables, figures, and references.

    :param pmcid: The PMC ID of the article.
    :param abstract_only: Whether to return only the abstract instead of the full text.
    :return: The title and content (abstract or full text of the article as a list of paragraphs)
        or None if the article is not found.
    """
    # Requested through the checked layer, which applies a timeout, a size limit, retries, and a
    # polite request rate. A full text article is a large document from a service that asks
    # clients to identify themselves and rate limits those that do not.
    text_url = build_url(
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_JSON/PMC{pmcid}/unicode",
        pmcid=pmcid,
    )

    # An article that cannot be fetched or parsed is skipped rather than fatal, since the caller
    # is working through a list of search results and the next one may be fine
    try:
        article = request_json(text_url)
    except (WebRequestError, json.JSONDecodeError):
        return None, None

    # Get document
    document = article[0]["documents"][0]

    # Get title
    title = next(passage["text"] for passage in document["passages"] if passage["infons"]["section_type"] == "TITLE")

    # Get relevant passages
    passages = [passage for passage in document["passages"] if passage["infons"]["type"] in {"abstract", "paragraph"}]

    # Get abstract or full text of article (excluding references)
    if abstract_only:
        passages = [passage for passage in passages if passage["infons"]["section_type"] in ["ABSTRACT"]]
    else:
        passages = [
            passage
            for passage in passages
            if passage["infons"]["section_type"] in ["ABSTRACT", "INTRO", "RESULTS", "DISCUSS", "CONCL", "METHODS"]
        ]

    # Get content
    content = [passage["text"] for passage in passages]

    return title, content


def run_pubmed_search(query: str, num_articles: int = 3, abstract_only: bool = False) -> str:
    """Runs a PubMed search, returning the full text of the top matching article.

    :param query: The query to search PubMed with.
    :param num_articles: The number of articles to search for.
    :param abstract_only: Whether to return only the abstract instead of the full text.
    :return: The full text of the top matching article.
    """
    # Print search query
    print(
        f'Searching PubMed Central for {num_articles} articles ({"abstracts" if abstract_only else "full text"}) with query: "{query}"'
    )

    # Perform PubMed Central search for query to get PMC ID. The query is passed as a parameter
    # rather than interpolated into the URL, so that a query containing a separator cannot alter
    # the rest of the request.
    search = request_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pmc",
            "term": query,
            "retmax": 2 * num_articles,
            "retmode": "json",
            "sort": "relevance",
        },
    )
    pmcids_found = search["esearchresult"]["idlist"]

    # Loop through top articles
    texts = []
    titles = []
    pmcids = []

    for pmcid in pmcids_found:
        # Break if reached desired number of articles
        if len(pmcids) >= num_articles:
            break

        title, content = get_pubmed_central_article(
            pmcid=pmcid,
            abstract_only=abstract_only,
        )

        if title is None:
            continue

        texts.append(f"PMCID = {pmcid}\n\nTitle = {title}\n\n{'\n\n'.join(content or [])}")
        titles.append(title)
        pmcids.append(pmcid)

    # Print articles found
    article_count = len(texts)

    print(f"Found {article_count:,} articles on PubMed Central")

    # Combine texts
    if article_count == 0:
        combined_text = f'No articles found on PubMed Central for the query "{query}".'
    else:
        combined_text = format_references(
            references=tuple(texts),
            reference_type="paper",
            intro=f'Here are the top {article_count} articles on PubMed Central for the query "{query}":',
        )

    return combined_text


def count_tokens(string: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Returns the estimated number of tokens in a text string.

    This is an offline estimate. Prefer the token counts reported by the API (see MeetingUsage)
    when they are available, since only those account for reasoning tokens and caching.

    :param string: The text string to count tokens in.
    :param encoding_name: The name of the encoding to use.
    :return: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens


def update_token_counts(
    token_counts: dict[str, int],
    discussion: list[dict[str, str]],
    response: str,
) -> None:
    """Updates the token counts (in place) with a discussion and response.

    :param token_counts: The token counts to update.
    :param discussion: The discussion to update the token counts with.
    :param response: The response to update the token counts with.
    """
    new_input_token_count = sum(count_tokens(turn["message"]) for turn in discussion)
    new_output_token_count = count_tokens(response)

    token_counts["input"] += new_input_token_count
    token_counts["output"] += new_output_token_count

    token_counts["max"] = max(token_counts["max"], new_input_token_count + new_output_token_count)


def count_discussion_tokens(
    discussion: list[dict[str, str]],
) -> dict[str, int]:
    """Counts the number of tokens in a discussion.

    :param discussion: The discussion to count tokens in.
    :return: A dictionary of token counts.
    """
    token_counts = {
        "input": 0,
        "output": 0,
        "tool": 0,
        "max": 0,
    }

    for index, turn in enumerate(discussion):
        if turn["agent"] == "User":
            continue

        # Tool output is not generated by the model, so it is never an output token. It is sent
        # back to the model, so it is already counted as input via the prefixes of later turns.
        # It is tracked separately for reporting only.
        if turn["agent"] == "Tool":
            token_counts["tool"] += count_tokens(turn["message"])
            continue

        update_token_counts(
            token_counts=token_counts,
            discussion=discussion[:index],
            response=turn["message"],
        )

    return token_counts


def _find_model_key[T](model: str, model_dict: dict[str, T]) -> str | None:
    """Finds the matching key in a model-keyed dictionary for a model.

    First checks for an exact match, then finds the longest prefix match.

    :param model: The name of the model.
    :param model_dict: The model-keyed dictionary to search.
    :return: The matching key or None if no match found.
    """
    if model in model_dict:
        return model

    # Find the longest prefix match
    matching_keys = [key for key in model_dict if model.startswith(key)]
    if matching_keys:
        return max(matching_keys, key=len)

    return None


def compute_token_cost(model: str, input_token_count: int, output_token_count: int) -> float:
    """Computes the token cost of a model given input and output token counts.

    :param model: The name of the model.
    :param input_token_count: The number of tokens in the input.
    :param output_token_count: The number of tokens in the output.
    :return: The token cost of the model.
    """
    input_key = _find_model_key(model, MODEL_TO_INPUT_PRICE_PER_TOKEN)
    output_key = _find_model_key(model, MODEL_TO_OUTPUT_PRICE_PER_TOKEN)

    if input_key is None or output_key is None:
        raise ValueError(f'Cost of model "{model}" not known')

    return (
        input_token_count * MODEL_TO_INPUT_PRICE_PER_TOKEN[input_key]
        + output_token_count * MODEL_TO_OUTPUT_PRICE_PER_TOKEN[output_key]
    )


class ContextLengthExceededError(ValueError):
    """Raised when a request is too large for the model's input limit."""


def count_message_tokens(messages: list[ChatCompletionMessageParam]) -> int:
    """Estimates the number of tokens in a list of chat messages.

    :param messages: The messages to count tokens in.
    :return: The estimated number of tokens, including per-message framing overhead.
    """
    num_tokens = 0

    for message in messages:
        num_tokens += TOKENS_PER_MESSAGE

        content = message.get("content")

        if isinstance(content, str):
            num_tokens += count_tokens(content)
        elif isinstance(content, Iterable):
            # Multi-part content, e.g. text and images interleaved
            for part in content:
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str):
                    num_tokens += count_tokens(text)

        # Tool calls and their arguments also occupy the context
        for tool_call in message.get("tool_calls") or ():
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if isinstance(function, dict):
                num_tokens += count_tokens(str(function.get("name", "")))
                num_tokens += count_tokens(str(function.get("arguments", "")))

    return num_tokens


def get_max_input_tokens(model: str) -> int | None:
    """Returns the maximum number of input tokens a model accepts.

    :param model: The name of the model.
    :return: The input limit, or None if the model is not known.
    """
    key = _find_model_key(model, MODEL_TO_MAX_INPUT_TOKENS)

    return MODEL_TO_MAX_INPUT_TOKENS[key] if key is not None else None


def check_context_length(
    messages: list[ChatCompletionMessageParam],
    model: str,
    warning_threshold: float = CONTEXT_WARNING_THRESHOLD,
) -> tuple[int, bool]:
    """Checks that a request fits in a model's input limit before sending it.

    The estimate omits tool schemas and some framing, so it is a lower bound on the true
    request size. Exceeding the limit on the estimate therefore means the request cannot
    succeed, which makes it safe to fail before paying for the call.

    :param messages: The messages that would be sent.
    :param model: The model the messages would be sent to.
    :param warning_threshold: The fraction of the input limit above which to flag the request.
    :raises ContextLengthExceededError: If the estimate already exceeds the model's input limit.
    :return: The estimated token count, and whether it is above the warning threshold.
    """
    estimated_tokens = count_message_tokens(messages)
    max_input_tokens = get_max_input_tokens(model)

    # Unknown models are not checked rather than guessed at, to avoid blocking valid requests
    if max_input_tokens is None:
        return estimated_tokens, False

    if estimated_tokens > max_input_tokens:
        raise ContextLengthExceededError(
            f"Request of about {estimated_tokens:,} tokens exceeds the {max_input_tokens:,} token "
            f'input limit of "{model}". Reduce num_rounds, the number of team members, or the '
            f"size of the summaries and contexts passed in."
        )

    return estimated_tokens, estimated_tokens > warning_threshold * max_input_tokens


def print_cost_and_time(
    token_counts: dict[str, int],
    model: str,
    elapsed_time: float,
) -> None:
    """Prints the token counts, cost, and elapsed time of a meeting.

    :param token_counts: The token counts, as returned by count_discussion_tokens.
    :param model: The name of the model used for the meeting.
    :param elapsed_time: The elapsed time of the meeting in seconds.
    """
    # Print token counts
    print(f"Input token count: {token_counts['input']:,}")
    print(f"Output token count: {token_counts['output']:,}")
    print(f"Tool token count: {token_counts.get('tool', 0):,}")
    print(f"Max token length: {token_counts['max']:,}")

    # Compute and print cost (tool tokens are already included in the input count)
    try:
        cost = compute_token_cost(
            model=model,
            input_token_count=token_counts["input"],
            output_token_count=token_counts["output"],
        )
        print(f"Cost: ${cost:.2f}")
    except ValueError as e:
        print(f"Warning: {e}")

    # Print time
    print(f"Time: {int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}")


@dataclass
class ModelUsage:
    """Token usage reported by the API for a single model."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    max_input_tokens: int = 0
    num_calls: int = 0


@dataclass
class MeetingUsage:
    """Accumulates the token usage reported by the API across all calls in a meeting.

    Usage is tracked per model so that meetings with agents on different models are priced
    correctly. Unlike an offline tiktoken estimate, these counts include reasoning tokens,
    which are billed as output but never appear in the response content.
    """

    per_model: dict[str, ModelUsage] = field(default_factory=dict)

    def add(self, model: str, usage: CompletionUsage | None) -> None:
        """Records the usage from a single API response.

        :param model: The model that produced the response.
        :param usage: The usage reported by the API, or None if the API did not report any.
        """
        if usage is None:
            return

        model_usage = self.per_model.setdefault(model, ModelUsage())

        input_tokens = usage.prompt_tokens or 0
        model_usage.input_tokens += input_tokens
        # Reasoning tokens are already included in completion_tokens, so they are not added again
        model_usage.output_tokens += usage.completion_tokens or 0
        model_usage.max_input_tokens = max(model_usage.max_input_tokens, input_tokens)
        model_usage.num_calls += 1

        # Detail fields vary across API and SDK versions, so they are read defensively
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            model_usage.cached_input_tokens += getattr(prompt_details, "cached_tokens", 0) or 0

        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            model_usage.reasoning_tokens += getattr(completion_details, "reasoning_tokens", 0) or 0

    @property
    def input_tokens(self) -> int:
        """The total number of input tokens across all models."""
        return sum(model_usage.input_tokens for model_usage in self.per_model.values())

    @property
    def cached_input_tokens(self) -> int:
        """The total number of cached input tokens across all models."""
        return sum(model_usage.cached_input_tokens for model_usage in self.per_model.values())

    @property
    def output_tokens(self) -> int:
        """The total number of output tokens across all models."""
        return sum(model_usage.output_tokens for model_usage in self.per_model.values())

    @property
    def reasoning_tokens(self) -> int:
        """The total number of reasoning tokens across all models."""
        return sum(model_usage.reasoning_tokens for model_usage in self.per_model.values())

    @property
    def max_input_tokens(self) -> int:
        """The largest number of input tokens sent in any single call."""
        return max((model_usage.max_input_tokens for model_usage in self.per_model.values()), default=0)

    @property
    def num_calls(self) -> int:
        """The total number of API calls."""
        return sum(model_usage.num_calls for model_usage in self.per_model.values())

    def compute_cost(self) -> float:
        """Computes the total cost across all models.

        Cached input tokens are priced at the full input rate, so the result is an upper bound
        for models and providers that discount them.

        :raises ValueError: If the price of any model is not known.
        :return: The total cost in USD.
        """
        return sum(
            compute_token_cost(
                model=model,
                input_token_count=model_usage.input_tokens,
                output_token_count=model_usage.output_tokens,
            )
            for model, model_usage in self.per_model.items()
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns the usage as a JSON-serializable dictionary, including cost where known.

        :return: Totals, the per-model breakdown, and the cost, or None if any model is unpriced.
        """
        try:
            cost: float | None = self.compute_cost()
        except ValueError:
            cost = None

        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "max_input_tokens": self.max_input_tokens,
            "num_calls": self.num_calls,
            "cost": cost,
            "per_model": {
                model: dict(model_usage.__dict__) for model, model_usage in self.per_model.items()
            },
        }

    def print_summary(self, elapsed_time: float) -> None:
        """Prints the token usage, cost, and elapsed time.

        :param elapsed_time: The elapsed time of the meeting in seconds.
        """
        # Break down by model only when more than one model was used
        if len(self.per_model) > 1:
            for model, model_usage in sorted(self.per_model.items()):
                print(
                    f"{model}: {model_usage.num_calls:,} calls, "
                    f"{model_usage.input_tokens:,} input, {model_usage.output_tokens:,} output"
                )

        print(f"Input token count: {self.input_tokens:,} ({self.cached_input_tokens:,} cached)")
        print(f"Output token count: {self.output_tokens:,} ({self.reasoning_tokens:,} reasoning)")
        print(f"Max input token count: {self.max_input_tokens:,}")

        try:
            print(f"Cost: ${self.compute_cost():.2f}")
        except ValueError as e:
            print(f"Warning: {e}")

        print(f"Time: {int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}")


def compute_finetuning_cost(model: str, token_count: int, num_epochs: int = DEFAULT_FINETUNING_EPOCHS) -> float:
    """Computes the cost of fine-tuning a model.

    :param model: The model that will be finetuned.
    :param token_count: The number of training tokens for finetuning.
    :param num_epochs: Number of finetuning epochs.
    :return: The cost of finetuning.
    """
    if model not in FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN:
        raise ValueError(f'Cost of model "{model}" not known')

    return token_count * FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN[model] * num_epochs


def get_summary(discussion: list[dict[str, str]]) -> str:
    """Get the summary from a discussion.

    :param discussion: The discussion to extract the summary from.
    :return: The summary.
    """
    return discussion[-1]["message"]


def load_summaries(discussion_paths: list[Path]) -> tuple[str, ...]:
    """Load summaries from a list of discussion paths.

    :param discussion_paths: The paths to the discussion JSON files. The summary is the last entry in the discussion.
    :return: A tuple of summaries.
    """
    summaries = []
    for discussion_path in discussion_paths:
        with open(discussion_path, "r") as file:
            discussion = json.load(file)
        summaries.append(get_summary(discussion))

    return tuple(summaries)


def save_meeting(save_dir: Path, save_name: str, discussion: list[dict[str, str]]) -> None:
    """Save a meeting discussion to JSON and Markdown files.

    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param discussion: The discussion to save.
    """
    # Create the save directory if it does not exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the discussion as JSON
    with open(save_dir / f"{save_name}.json", "w") as f:
        json.dump(discussion, f, indent=4)

    # Save the discussion as Markdown
    with open(save_dir / f"{save_name}.md", "w", encoding="utf-8") as file:
        for turn in discussion:
            file.write(f"## {turn['agent']}\n\n{turn['message']}\n\n")
