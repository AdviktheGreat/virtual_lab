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
