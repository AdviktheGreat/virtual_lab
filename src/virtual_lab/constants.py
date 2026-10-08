"""Holds constants."""

DEFAULT_MODEL = "gpt-5.2"

# Prices in USD per token: OpenAI's as of December 30, 2025 (https://openai.com/api/pricing/),
# Anthropic's and Google's as of September 2026
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
    # Anthropic (https://platform.claude.com/docs/en/about-claude/pricing)
    "claude-opus-5": 5 / 10**6,
    "claude-sonnet-5": 2 / 10**6,
    "claude-opus-4-8": 5 / 10**6,
    "claude-opus-4-7": 5 / 10**6,
    "claude-opus-4-6": 5 / 10**6,
    "claude-sonnet-4-6": 3 / 10**6,
    "claude-opus-4-5": 5 / 10**6,
    "claude-sonnet-4-5": 3 / 10**6,
    "claude-haiku-4-5": 1 / 10**6,
    "claude-opus-4-1": 15 / 10**6,
    "claude-opus-4": 15 / 10**6,
    "claude-sonnet-4": 3 / 10**6,
    "claude-3-5-haiku": 0.8 / 10**6,
    # Google (https://ai.google.dev/gemini-api/docs/pricing), at the rate for prompts over
    # 200k tokens where there are two, so that a cost is never understated
    "gemini-3.1-pro-preview": 4 / 10**6,
    "gemini-2.5-pro": 2.5 / 10**6,
    "gemini-2.5-flash": 0.3 / 10**6,
    "gemini-2.5-flash-lite": 0.1 / 10**6,
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
    # Anthropic (https://platform.claude.com/docs/en/about-claude/pricing)
    "claude-opus-5": 25 / 10**6,
    "claude-sonnet-5": 10 / 10**6,
    "claude-opus-4-8": 25 / 10**6,
    "claude-opus-4-7": 25 / 10**6,
    "claude-opus-4-6": 25 / 10**6,
    "claude-sonnet-4-6": 15 / 10**6,
    "claude-opus-4-5": 25 / 10**6,
    "claude-sonnet-4-5": 15 / 10**6,
    "claude-haiku-4-5": 5 / 10**6,
    "claude-opus-4-1": 75 / 10**6,
    "claude-opus-4": 75 / 10**6,
    "claude-sonnet-4": 15 / 10**6,
    "claude-3-5-haiku": 4 / 10**6,
    # Google (https://ai.google.dev/gemini-api/docs/pricing), at the rate for prompts over
    # 200k tokens where there are two, so that a cost is never understated
    "gemini-3.1-pro-preview": 18 / 10**6,
    "gemini-2.5-pro": 15 / 10**6,
    "gemini-2.5-flash": 2.5 / 10**6,
    "gemini-2.5-flash-lite": 0.4 / 10**6,
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

# Who a note given through a meeting's steer is from, in its transcript
HUMAN_SPEAKER = "Human researcher"

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

# Output tokens allowed per response for providers that insist on a figure, Anthropic and
# self-hosted servers, as Biomni sets them. A meeting's max_completion_tokens overrides it.
DEFAULT_MAX_OUTPUT_TOKENS = 8_192

CONSISTENT_TEMPERATURE = 0.2
CREATIVE_TEMPERATURE = 0.8

# Maximum consecutive rounds of tool calls allowed within a single agent's turn, so that an
# agent cannot search indefinitely. Tool definitions are withheld on the final attempt to
# force a text answer.
MAX_TOOL_ITERATIONS = 5

# The same limit for a meeting with a session to run code in. An analysis takes many steps of
# code, each shaped by the output of the last: loading, inspecting, cleaning, and only then
# computing what was asked. Biomni's agent routinely takes dozens.
SESSION_MAX_TOOL_ITERATIONS = 20

# Most characters of one tool result given back to a model. Every lookup already budgets its own
# report, the largest being a full text at 20,000 characters plus its headings and citation, so
# this is set above all of them and only binds on output that has no budget of its own: the
# PubMed search returns whole articles, and one tool call is otherwise one megabyte of request.
MAX_TOOL_OUTPUT_CHARS = 25_000

# Characters of a tool name, or of an error, to repeat back to a model. The API accepts function
# names of at most 64 characters, so a longer one is not a tool, and an exception's message is
# written by whatever raised it and can quote an entire response.
MAX_TOOL_NAME_CHARACTERS = 64
MAX_TOOL_ERROR_CHARACTERS = 1_000

# Articles one PubMed search may return. Each is a full text fetched with a request of its own,
# two candidates are searched for every article wanted, and the model chooses the number.
MAX_PUBMED_ARTICLES = 5

# Characters of a tool call's arguments to keep in the meeting record. Enough for every search
# and lookup the tools take; a model can pass arguments of any length, and the record is a log of
# what was asked for, not a second copy of it.
MAX_RECORDED_ARGUMENT_CHARS = 2_000

# Image used to run model-authored code. Pinned to a digest-free but explicit tag so that a
# meeting is not silently handed a different interpreter than the one it was written for.
DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"

# The name the sandbox images built from Biomni's environment are tagged with (see
# virtual_lab.environment)
SANDBOX_IMAGE_NAME = "virtual-lab-sandbox"

# The platform those images are built for. Biomni's environment is only published for x86-64
# Linux: pystan has no ARM wheels, and several of its command-line tools are x86-64 binaries.
# On an ARM machine Docker runs the image under emulation.
SANDBOX_PLATFORM = "linux/amd64"

# Where Biomni publishes its data lake and benchmarks
BIOMNI_RELEASE_URL = "https://biomni-release.s3.amazonaws.com"

# How much of a data lake file is read at a time. The largest are several gigabytes.
DATA_LAKE_CHUNK_BYTES = 1024**2

# Biomni-Eval1, at the revision of its dataset that Biomni's BiomniEval1 reads. Pinned, so that
# a score can be compared with another made on the same questions.
BIOMNI_EVAL1_URL = (
    "https://huggingface.co/datasets/biomni/Eval1/resolve/"
    "51f97feb9e377fac7384faae49046bc067283c38/biomni_eval1_dataset.parquet"
)
BIOMNI_EVAL1_FILE = "eval1/biomni_eval1_dataset.parquet"

# Biomni's archive of the files its LAB-Bench and Humanity's Last Exam tasks read, unpacked as
# Biomni unpacks it into its benchmark directory
BIOMNI_BENCHMARK_ARCHIVE = "benchmark.zip"
BIOMNI_BENCHMARK_ARCHIVE_SHA256 = "27f4d9021dca7472efc225f6442f9a1dc595062a4851e4f708ba94d637ace6ff"

# Every benchmark file, by its path under the benchmark directory, with what it must hash to.
# Biomni's archive is not versioned, so a changed file is refused rather than scored as if it
# were the same benchmark.
BENCHMARK_FILES = {
    BIOMNI_EVAL1_FILE: "cb8443b059cfc81d3d94bab35473bb271b4e7e1935473f2b61b5fae58a2db861",
    "DbQA/train-00000-of-00001.parquet": "d984bf7690ec690bd9859b77706c9304cf70e45f5a8851b449d0e18c15c0bb4f",
    "DbQA/train-00000-of-00001_sampled.parquet": "c5910bbda84b2e2ab3e75828199895549ddaddcf5d2e45fded16cd838c74ce1d",
    "DbQA/train-00000-of-00001_test.parquet": "c373221f9803f45cf3de59d6435c20bfac12c8ff2c327aa94c4edf31c837f5a2",
    "SeqQA/train-00000-of-00001.parquet": "09c7b26ea1a4366dc6ce9e05a09e505d41e7073780a7c782a1018689d733c3f8",
    "SeqQA/train-00000-of-00001_sampled.parquet": "39f2b2ad880949ac75b69212c12afa7fb79c50a16115c1f630b7223ae919adf3",
    "SeqQA/train-00000-of-00001_test.parquet": "a21197ffe4f35fa878ed4e82971debfd19c6b4c91e1bc30f44c3006ffbd76ce8",
    "hle/test_sampled_biology_medicine.parquet": "9382aeba3e9aca2587020eec6369a9641f6f6c60d311ad4de2f202be365fe8ae",
}

# Reading an answer out of a transcript has one right result, so the model reading it is not
# asked to vary
EXTRACTION_TEMPERATURE = 0.0

# Where the data lake is mounted in the sandbox, read-only. The same path Biomni's tools read
# it from, relative to their data directory.
SANDBOX_DATA_LAKE_DIR = "/biomni_data/data_lake"

# The directory Biomni's tools take as their data path, which holds the data lake. Given to them
# as BIOMNI_PATH, their own configuration's way of setting it.
SANDBOX_BIOMNI_PATH = "/biomni_data"

# Where Biomni's own package, with its tool functions, is mounted in the sandbox, read-only, and
# put on the Python path, so that code imports them as Biomni's agent does:
# from biomni.tool.genomics import ...
SANDBOX_BIOMNI_PACKAGE_DIR = "/opt/biomni_package"

# Where the data given to a session is mounted in the sandbox, read-only, each under its file name
SANDBOX_USER_DATA_DIR = "/data"

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

# The same ceilings for a session (see virtual_lab.session), which holds a whole meeting's
# analysis in memory at once rather than running one script: an AnnData object of a single-cell
# experiment is gigabytes, and numba and BLAS start a thread per core, each counted as a process
DEFAULT_SESSION_MEMORY_LIMIT = "8g"
DEFAULT_SESSION_CPU_LIMIT = "4"
DEFAULT_SESSION_PIDS_LIMIT = 1024
DEFAULT_SESSION_TMPFS_SIZE = "2g"

# Seconds to wait for a session's interpreter to say it is ready. An image built for x86-64 is
# emulated on an ARM machine, where the interpreter alone can take several seconds to start.
SESSION_START_TIMEOUT = 120

# Seconds past its time limit to wait for code to be stopped before stopping the whole session.
# Code is interrupted at its limit, which takes effect at its next Python instruction; a call
# into compiled code that does not return is only ended by stopping the session.
SESSION_GRACE_SECONDS = 10

# Seconds to wait for a session to exit once its input is closed, before it is killed
SESSION_CLOSE_TIMEOUT = 5

# Largest answer read from a session's interpreter, in bytes. Its output is already bounded,
# so an answer past this is not one the interpreter wrote, and ends the session.
MAX_SESSION_RESPONSE_BYTES = 2 * 1024**2

# Most bytes of JSON a session's tool may return to the code that called it. A tool's arguments
# are bounded by MAX_SESSION_RESPONSE_BYTES, since they come back the way answers do.
MAX_HOST_TOOL_RESULT_BYTES = 16 * 1024**2

# Most bytes of JSON the value of code's last expression may take to be sent back from a session,
# as Session.evaluate asks for it. It comes back in the answer, beside the code's output, and
# the two must fit within MAX_SESSION_RESPONSE_BYTES.
MAX_SESSION_VALUE_BYTES = 1024**2

# Seconds a call to a tool of an MCP server may take before it is given up, and the agent told so
MCP_CALL_TIMEOUT = 300.0
# Seconds an MCP server may take to start, or be connected to, and list its tools. Generous, since
# a server run with npx or docker may first have to be downloaded.
MCP_START_TIMEOUT = 120.0
# Seconds an MCP server is given to stop once it is closed, before it is stopped regardless
MCP_STOP_TIMEOUT = 10.0
# Most characters of what an MCP server wrote to stderr shown when it fails to start or stops
MCP_LOG_TAIL_CHARS = 2_000
# Most pages an MCP server may list its tools in, so that a server whose pages never end cannot
# keep a connection from finishing
MCP_MAX_TOOL_PAGES = 1_000
# Most characters of what an MCP server says of how to use its tools that the agents are told,
# so that a server cannot fill their context with it
MCP_MAX_INSTRUCTIONS_CHARS = 8_000
# Most characters of a call's arguments shown to a person asked to approve it
APPROVAL_MAX_ARGUMENT_CHARS = 4_000
# Seconds a person has to sign in to an MCP server in their browser. The time it takes is not
# counted against the time a server has to be connected to, or a call to finish
MCP_SIGN_IN_TIMEOUT = 600.0
# Where the sign-ins to MCP servers are kept, and the variable that puts them elsewhere
MCP_AUTH_DIRECTORY = "~/.virtual_lab/mcp_auth"
MCP_AUTH_DIRECTORY_VARIABLE = "VIRTUAL_LAB_MCP_AUTH_DIR"
# Where serve_mcp listens over HTTP by default: this machine alone, at http://127.0.0.1:8000/mcp
MCP_SERVER_HOST = "127.0.0.1"
MCP_SERVER_PORT = 8000
MCP_SERVER_PATH = "/mcp"
# Where virtual-lab-mcp reads the token that every HTTP request must carry, and the fewest
# characters a token may have, which secrets.token_urlsafe(32) exceeds at 43
MCP_SERVER_TOKEN_VARIABLE = "VIRTUAL_LAB_MCP_TOKEN"
MCP_SERVER_MIN_TOKEN_CHARS = 16
# Most characters of a tool call's arguments kept in a session's record of it
MAX_RECORDED_ARGUMENT_CHARS = 10_000

# Most output that is kept from an execution, per stream, in bytes. Untrusted code can print
# without bound, so output is read as it arrives and only this much of its tail is held; none of
# it is written to disk, where a program printing in a loop would otherwise fill the host.
MAX_CAPTURED_OUTPUT_CHARS = 50_000

# Seconds to keep reading output after a run has ended and its processes have been killed. The
# pipes close as soon as the last writer is gone, so this only matters when something outside
# the process group still holds one open, and then the output read so far is used.
OUTPUT_DRAIN_TIMEOUT = 5

# Largest single file code may write, in bytes, enforced by the kernel as RLIMIT_FSIZE. A loop
# writing to a file is otherwise stopped only by a full disk, and on the local backend or the
# sandbox's mount that disk is the host's. A gigabyte is well past the tables and embeddings an
# analysis writes. It is a limit per file, not in total.
MAX_WRITTEN_FILE_BYTES = 1024**3

# Most output shown to an agent when it is told how its code behaved. Far smaller than what is
# captured, because this goes into a request and is paid for by the token.
MAX_REPORTED_OUTPUT_CHARS = 4_000

# Files listed by name when a run's report says what it wrote. A script writing one file per
# item can write tens of thousands, and 20,000 names made a 2.26 MB report; the full list stays
# in the execution record.
MAX_REPORTED_FILES = 50

# Times code may be run before giving up, counting the first run. Every attempt past the first
# costs another round of code generation, so this is a budget and not a limit to raise freely:
# three attempts means a failing meeting can cost roughly three times a passing one.
DEFAULT_MAX_REPAIR_ATTEMPTS = 3

# Subdirectory for the record of what happened when a meeting's code was run, kept separate
# from the meeting's own record for the same reason the others are in their own directories.
EXECUTION_DIR_NAME = "executions"

# Seconds for each part of the check of which of Biomni's software a session has: importing its
# tool modules, which takes seconds but loads torch in some, and listing the R packages. Each
# part runs in a process of its own and is abandoned at this limit, so that a check that hangs
# never costs the session the interpreter the analysis lives in.
SOFTWARE_CHECK_TIMEOUT = 300

# Most characters of a model's answer to which resources a meeting needs kept in the record. The
# answer is a few lists of numbers; anything longer is a model that did not follow the prompt.
MAX_RECORDED_RETRIEVAL_CHARS = 2_000

# Subdirectory for the log of the code a meeting ran in its session, cell by cell, in full. The
# meeting's record says which cells each turn ran; this holds their code and output.
SESSION_LOG_DIR_NAME = "sessions"

# A project's ledger of every step it took, in its directory, and the subdirectory its meetings
# and repairs are saved in, each under the step's name.
PROJECT_FILE_NAME = "project.json"
PROJECT_MEETINGS_DIR_NAME = "meetings"

# What a project run to its goal records of each round, and the report it ends with, in the
# project's directory
RESEARCH_LOG_FILE_NAME = "research_log.json"
REPORT_FILE_NAME = "report.json"
REPORT_MARKDOWN_FILE_NAME = "report.md"

# The findings a project run to its goal has made, as its LabMemory keeps them
MEMORY_FILE_NAME = "memory.json"

# Identifies this library to the services it queries. NCBI and EMBL-EBI both ask clients to say
# who they are, and an unidentified client is the first to be throttled.
WEB_USER_AGENT = "virtual-lab (https://github.com/zou-group/virtual_lab)"

# Seconds to allow for a connection and for each read. A service that stops responding mid-answer
# would otherwise hold a meeting open indefinitely.
WEB_TIMEOUT_SECONDS = 30

# Attempts for a request that fails in a way worth repeating, counting the first
WEB_MAX_ATTEMPTS = 3

# Base for exponential backoff between attempts, in seconds. Overridden by a Retry-After header.
WEB_BACKOFF_SECONDS = 1.0

# Redirects to follow before concluding a service is sending a client in circles
WEB_MAX_REDIRECTS = 5

# Smallest gap between two requests to the same host. NCBI limits unauthenticated clients to a
# few requests a second and refuses the rest, so several tool calls in a row would otherwise
# spend their retries on rate limits this library brought on itself.
MIN_SECONDS_BETWEEN_REQUESTS = 0.34

# Most bytes to accept from one response. A structure file or a full-text article can be far
# larger than a meeting can be given, and the body is abandoned rather than read once it is over
# this, so an unbounded response cannot exhaust memory either.
MAX_RESPONSE_BYTES = 5_000_000

# Responses remembered per process. Agents ask the same question more than once, and a repeat
# costs a service bandwidth for an answer already known.
MAX_WEB_CACHE_ENTRIES = 256

# Total size of the remembered responses. Needed alongside the entry count because a count on its
# own is not a bound on memory: 256 entries of the largest allowed response would be well over a
# gigabyte, which would undo the limit above it.
MAX_WEB_CACHE_CHARACTERS = 32_000_000

# Residues of a sequence to show a model. A sequence is the most useful thing a protein record
# holds and also the longest: titin runs to roughly 35,000 residues, which is most of a context
# window spent on one field. The whole sequence stays on the record for code to use.
MAX_SEQUENCE_RESIDUES_REPORTED = 1_200

# Most results a search may be asked for. A model chooses this number, and a request for hundreds
# would be answered, charged for, and then largely ignored.
MAX_SEARCH_RESULTS = 25

# Chains of a structure to describe. A ribosome has dozens, and the first few are enough to tell
# an agent what the structure contains.
MAX_CHAINS_REPORTED = 8

# Annotated positions to list per protein, since a well studied one carries hundreds
MAX_FEATURES_REPORTED = 25

# Free text annotations to show per protein, and how much of each. These are the largest thing in
# a protein report after the sequence: the comments on a well curated entry run to over 11,000
# characters, which is more than the capped sequence and the rest of the record put together.
MAX_COMMENTS_REPORTED = 8
MAX_COMMENT_CHARACTERS = 600

# Most bytes to accept for a structure file. Far larger than an ordinary response because these
# are written to disk for code to read rather than shown to a model, and a large complex runs to
# tens of megabytes.
MAX_STRUCTURE_FILE_BYTES = 30_000_000

# Characters of a download link a service sends to keep. AlphaFold's are under a hundred, and one
# is repeated in the report of what was downloaded.
MAX_URL_CHARACTERS = 2_000

# Free text to show for a compound. PubChem's description of a well known drug runs to several
# paragraphs, the same unbounded field problem a protein's comments were.
MAX_DESCRIPTION_CHARACTERS = 600

# Characters of an assay's description to show beside a measurement. Enough to tell a cell assay
# from a biochemical one, which is what decides whether two numbers can be compared.
MAX_ASSAY_DESCRIPTION_CHARACTERS = 200

# Measurements to report for one bioactivity question. ChEMBL holds roughly 19,000 IC50 values
# against a single well studied kinase, and the most potent handful is what answers the question.
MAX_ACTIVITIES_REPORTED = 15

# Characters of a structure string or a systematic name to show. A small molecule's SMILES is
# under a hundred, but ChEMBL holds peptides and antibody-drug conjugates whose SMILES runs to
# thousands, and a systematic name grows with it. One of those would be the whole report.
MAX_STRUCTURE_CHARACTERS = 500

# Mechanisms of action to list for one molecule. A promiscuous kinase inhibitor has dozens on
# record, and the report is meant to say what the drug does rather than enumerate every target.
MAX_MECHANISMS_REPORTED = 10

# Characters kept of any short field a service sends: a name, a title, a journal, an identifier,
# a descriptor. Real ones are well under this, the longest titles being around 300 characters,
# but nothing in a response is bounded except by its size, and a single field of a five megabyte
# response would otherwise become a five megabyte report. The same figure as a structure string,
# which is the longest thing of this kind a report shows on purpose.
MAX_FIELD_CHARACTERS = 500

# Items of a short list to name before saying how many more there are: the genes of a protein,
# the components bound in a structure, the sections of an article left out. Real ones are a
# handful, and a list of thousands says nothing more than its count does.
MAX_ITEMS_LISTED = 20

# Smallest pChEMBL value worth reporting, on a scale where 6 is a micromolar affinity and 9 is
# nanomolar. Below this a compound is not usefully a binder, and the rows are mostly the inactive
# arm of a screen.
MIN_PCHEMBL_REPORTED = 4.0

# Subdirectory that downloaded structure files are written to. Code an agent writes runs with no
# network, so anything it needs to read has to be fetched for it and left somewhere it can reach.
STRUCTURE_DIR_NAME = "structures"

# Characters of an abstract to show. Long enough for a structured abstract with its Background,
# Methods, Results, and Conclusions headings, which is where most of the length comes from.
MAX_ABSTRACT_CHARACTERS = 2_500

# Authors to name before saying how many more there are. A consortium genomics paper has several
# hundred, and the first few are what identifies the work.
MAX_AUTHORS_REPORTED = 8

# Subject categories to name for a preprint. A paper cross-listed into eight of them is telling
# you less with each one.
MAX_CATEGORIES_REPORTED = 5

# Characters of one section of a full text to show, and of the whole article. A research article
# runs to 36,000 characters of body text, which is most of a context window spent on one paper
# when the point of fetching it was to decide whether it is worth reading properly.
MAX_SECTION_CHARACTERS = 4_000
MAX_ARTICLE_CHARACTERS = 20_000

# Sections of a full text that are not what anyone asked for. References alone are 28% of the
# body text of a typical article and are a list of other papers' titles, which reads to a model
# as though this article had discussed all of them.
SKIPPED_SECTION_TITLES = (
    "reference",
    "bibliography",
    "acknowledg",
    "funding",
    "conflict",
    "competing interest",
    "author contribution",
    "author information",
    "supplementary",
    "supporting information",
    "associated data",
    "data availability",
    "ethics",
    "abbreviation",
)

# Most bytes to read from a data file. Large enough for the supplementary table of a genomics
# paper and small enough that a file left in the working directory by mistake cannot exhaust
# memory. Counted against what is actually read rather than the size on disk, because a
# spreadsheet is a zip archive and what it claims to hold is written inside it.
MAX_TABLE_BYTES = 50_000_000

# Rows to read before answering with what has been seen. Reading every row of a million-row file
# to report five example values is time spent for nothing, and the answer is the same. The report
# says when it stopped early, since "20 missing values" means something different in a file that
# was read to the end.
MAX_TABLE_ROWS_SCANNED = 50_000

# Columns to describe. A wide matrix of per-sample measurements runs to thousands, and a line
# each would be the whole context window for a file whose shape is the same in every column.
MAX_COLUMNS_REPORTED = 40

# Whole rows to show under the column descriptions, and how many columns of each. A few aligned
# rows show how the columns line up, which a per-column summary cannot, and are how a header on
# the wrong row gets noticed. Wide enough to see that and no wider: the rows are shown on one
# line each, and a hundred columns of them is a wall.
MAX_SAMPLE_ROWS = 4
MAX_SAMPLE_COLUMNS = 8

# Distinct values of one column to show, and how much of any single cell. A free text notes
# column holds paragraphs, and three of them would crowd out every other column in the file.
MAX_COLUMN_EXAMPLES = 4
MAX_CELL_CHARACTERS = 120

# Distinct values to track per column. Only used to say whether a column is an identifier or a
# handful of repeated categories, so the count stops mattering long before this.
MAX_DISTINCT_TRACKED = 1_000

# Files to list for one working directory, and how deep to look. A directory holding more than
# this is not a project's data, and the point is to tell an agent what it can open.
MAX_DATA_FILES_LISTED = 60

# Suffixes read as delimited text, and as spreadsheets. Anything else is refused by name rather
# than sniffed, since guessing at the format of a file that was not meant to be a table produces
# a confident description of nothing.
DELIMITED_SUFFIXES = (".csv", ".tsv", ".tab", ".txt")
SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm")

# Entries to read from a spreadsheet archive, and the most bytes to unpack from any one of them.
# A 204 KB zip can declare a 200 MB member, which was measured rather than supposed, so the cap
# is on the bytes taken out rather than on the size the archive claims.
MAX_ARCHIVE_ENTRIES = 256
MAX_SHEET_BYTES = 30_000_000

# The last column a spreadsheet can have, XFD. A reference beyond it is not in any file Excel
# wrote, and taking it at its word pads every row out to that width.
MAX_SPREADSHEET_COLUMNS = 16_384

# Cells to hold from one sheet, counting the blanks between the values in each row. The byte
# cap does not bound this: a cell's position costs nothing to write, so a 5.7 KB sheet with a
# cell in column XFD of each row was measured at 600 MB. The smallest real cell is 16 bytes of
# XML, so a dense sheet under the byte cap holds fewer than this.
MAX_TABLE_CELLS = 2_000_000

# Names to list in one warning about a header. A matrix of per-sample measurements runs to tens
# of thousands of columns, and a warning naming each blank one is the whole context window.
MAX_NAMES_LISTED = 5

# Sheets to name in a report. Unlike names in a warning, these are how a sheet is asked for, so
# a workbook's ordinary handful is always listed in full.
MAX_SHEETS_LISTED = 30
