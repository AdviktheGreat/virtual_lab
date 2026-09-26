"""Holds constants."""

DEFAULT_MODEL = "gpt-5.2"

# Prices in USD as of December 30, 2025 (https://openai.com/api/pricing/)
MODEL_TO_INPUT_PRICE_PER_TOKEN = {
    "gpt-3.5-turbo-0125": 0.5 / 10**6,
    "gpt-4o-2024-08-06": 2.5 / 10**6,
    "gpt-4o-2024-05-13": 5 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.15 / 10**6,
    "o1-mini-2024-09-12": 3 / 10**6,
    "gpt-5": 1.25 / 10**6,
    "gpt-5-mini": 0.25 / 10**6,
    "gpt-5-nano": 0.05 / 10**6,
    "gpt-5.2": 1.75 / 10**6,
    "gpt-5.2-pro": 21 / 10**6,
}

MODEL_TO_OUTPUT_PRICE_PER_TOKEN = {
    "gpt-3.5-turbo-0125": 1.5 / 10**6,
    "gpt-4o-2024-08-06": 10 / 10**6,
    "gpt-4o-2024-05-13": 15 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.6 / 10**6,
    "o1-mini-2024-09-12": 12 / 10**6,
    "gpt-5": 10 / 10**6,
    "gpt-5-mini": 2 / 10**6,
    "gpt-5-nano": 0.4 / 10**6,
    "gpt-5.2": 14 / 10**6,
    "gpt-5.2-pro": 168 / 10**6,
}

FINETUNING_MODEL_TO_INPUT_PRICE_PER_TOKEN = {
    "gpt-4o-2024-08-06": 3.75 / 10**6,
    "gpt-4o-mini-2024-07-18": 0.3 / 10**6,
}

FINETUNING_MODEL_TO_OUTPUT_PRICE_PER_TOKEN = {
    "gpt-4o-2024-08-06": 15 / 10**6,
    "gpt-4o-mini-2024-07-18": 1.2 / 10**6,
}

FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN = {
    "gpt-4o-2024-08-06": 25 / 10**6,
    "gpt-4o-mini-2024-07-18": 3 / 10**6,
}

DEFAULT_FINETUNING_EPOCHS = 4

# Encoding for offline token estimates. Correct for the gpt-4o and gpt-5 families;
# cl100k_base only applies to gpt-4 and gpt-3.5-turbo.
DEFAULT_ENCODING = "o200k_base"

# Maximum input tokens accepted by each model. Keys are matched as prefixes, so "gpt-5"
# also covers gpt-5-mini, gpt-5.2, and so on. Models with no matching entry are not checked.
MODEL_TO_MAX_INPUT_TOKENS = {
    "gpt-3.5-turbo": 16_385,
    "gpt-4o": 128_000,
    "o1-mini": 128_000,
    "gpt-5": 272_000,
}

# Fraction of a model's input limit at which to warn that a meeting is running out of room
CONTEXT_WARNING_THRESHOLD = 0.8

# Rough per-message framing overhead used when estimating the size of a request
TOKENS_PER_MESSAGE = 4

# The API caps message author names at 64 characters
MAX_AGENT_NAME_LENGTH = 64

# Subdirectory for the transcript of a meeting that failed partway through. Kept out of the
# meeting's own directory so that globs over finished meetings cannot match it.
PARTIAL_MEETING_DIR_NAME = "partial"

# Subdirectory holding one directory per meeting for the files that meeting produced
ARTIFACT_DIR_NAME = "artifacts"

# Subdirectory for the structured output of each meeting, mirroring the transcript's filename.
# In a subdirectory for the same reason as the provenance record below.
OUTPUT_DIR_NAME = "outputs"

# Subdirectory for the provenance record of each meeting, mirroring the transcript's filename.
# A sibling file would not do: "discussion_1.meta.json" matches the "discussion_*.json" globs
# the notebooks use to collect transcripts, which would feed a record in as if it were one.
METADATA_DIR_NAME = "metadata"

# Retries for transient API failures (rate limits, timeouts, 5xx), applied with
# exponential backoff by the OpenAI client. Higher than the SDK default of 2 because a
# failed call discards a whole meeting's worth of work.
DEFAULT_MAX_RETRIES = 5

CONSISTENT_TEMPERATURE = 0.2
CREATIVE_TEMPERATURE = 0.8

# Maximum consecutive rounds of tool calls allowed within a single agent's turn, so that an
# agent cannot search indefinitely. Tool definitions are withheld on the final attempt to
# force a text answer.
MAX_TOOL_ITERATIONS = 5

# Image used to run model-authored code. Pinned to a digest-free but explicit tag so that a
# meeting is not silently handed a different interpreter than the one it was written for.
DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"

# Where the meeting's files are mounted inside the container. Model-authored code sees only
# this directory, so paths it writes into its own output are relative to here.
SANDBOX_WORK_DIR = "/workspace"

# Wall clock limit for a single execution, in seconds. Model-authored code has no notion of
# how long it should take, so something has to stop an accidental infinite loop.
DEFAULT_EXECUTION_TIMEOUT = 300

# Resource ceilings for a single execution. A runaway allocation or a fork bomb has to hit a
# limit inside the container rather than exhaust the host.
DEFAULT_MEMORY_LIMIT = "2g"
DEFAULT_CPU_LIMIT = "2"
DEFAULT_PIDS_LIMIT = 256

# Size of the writable scratch space mounted at /tmp, since the container's root filesystem
# is read-only and many libraries expect somewhere to write
DEFAULT_TMPFS_SIZE = "256m"

# Most output that is kept from an execution, per stream. Untrusted code can print without
# bound, so output is spooled to disk and only the tail of it is read back.
MAX_CAPTURED_OUTPUT_CHARS = 50_000

# Most output shown to an agent when it is told how its code behaved. Far smaller than what is
# captured, because this goes into a request and is paid for by the token.
MAX_REPORTED_OUTPUT_CHARS = 4_000

# Times code may be run before giving up, counting the first run. Every attempt past the first
# costs another round of code generation, so this is a budget and not a limit to raise freely:
# three attempts means a failing meeting can cost roughly three times a passing one.
DEFAULT_MAX_REPAIR_ATTEMPTS = 3

# Subdirectory for the record of what happened when a meeting's code was run, kept separate
# from the meeting's own record for the same reason the others are in their own directories.
EXECUTION_DIR_NAME = "executions"
