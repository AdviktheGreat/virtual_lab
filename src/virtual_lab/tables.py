"""Reading the data files a project works on, and saying how they will mislead.

An agent given a data file and no way to look inside it writes code against the columns it
imagines are there. The obvious fix is to let it run a line of code and print the result, but
that is a round trip through the sandbox for every question, and the first question is always
the same one: what is in this file.

So the point of this module is not to load data. Nothing here returns a value for computing
with, and the sandbox is still where analysis happens. The point is to answer that first
question well enough that the code written next is written against the file that exists.

Which means the interesting part is not the parsing but the warnings. A table lies quietly. A
column of measurements with one "<0.001" in it is a column of text, and every mean taken from
it afterwards is wrong or missing without anything having failed. A header read from a
byte-order mark is "\\ufeffgene" and never matches "gene". Two columns with the same name leave
one of them silently dropped. A gene symbol that went through Excel is a date, and SEPT2 has
been 2-Sep in about a fifth of published genomics supplements for a decade. None of these raise
anything. Each of them is checked here and said out loud, because the cost of not saying it is
an analysis that runs to completion and reports a number nobody can reproduce.
"""

import csv
import datetime
import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from virtual_lab.artifacts import UnsafeFilenameError, check_filename
from virtual_lab.constants import (
    DELIMITED_SUFFIXES,
    MAX_ARCHIVE_ENTRIES,
    MAX_CELL_CHARACTERS,
    MAX_COLUMN_EXAMPLES,
    MAX_COLUMNS_REPORTED,
    MAX_DATA_FILES_LISTED,
    MAX_DISTINCT_TRACKED,
    MAX_SAMPLE_COLUMNS,
    MAX_SAMPLE_ROWS,
    MAX_SHEET_BYTES,
    MAX_TABLE_BYTES,
    MAX_TABLE_ROWS_SCANNED,
    SPREADSHEET_SUFFIXES,
)
from virtual_lab.records import parse_xml, truncate_text


class TableError(Exception):
    """Raised when a data file cannot be read, or cannot be read as a table."""


# What a number looks like, written out rather than left to float(), which accepts a good deal
# more than anyone means by a number: float("1_000") is 1000.0, float("nan") and float("inf")
# both succeed, and float("\u0661\u0662\u0663") is 123.0 because \d and the float constructor
# both take Unicode digits. A column of Arabic-Indic numerals is not the problem; a column whose
# missing values are spelled "nan" being reported as numeric is.
NUMBER = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")

# A decimal comma, which is what a spreadsheet saved in most of Europe writes, and is why the
# file is usually semicolon separated as well
DECIMAL_COMMA = re.compile(r"[+-]?[0-9]+,[0-9]+")

# A thousands separator, which makes a number into text and, in a comma separated file, into two
# columns
GROUPED_NUMBER = re.compile(r"[+-]?[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?")

# An identifier that happens to be digits. Reading it as a number loses the zeros and the
# identifiers stop matching anything.
LEADING_ZERO = re.compile(r"0[0-9]+")

# A measurement that fell outside the assay's range. These are the single most common reason a
# column of numbers is a column of text, and dropping them biases whatever is computed next,
# because they are not missing at random: they are the strongest and weakest samples.
CENSORED = re.compile(r"[<>]=?\s*[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?")

# Words a file uses to mean "no value". Counted as missing so that the statistics mean
# something, and then named in the report, because which one a file uses is information: "ND"
# is not detected, "NA" is usually absent, and in a table of elements "NA" is sodium.
MISSING_WORDS = frozenset(
    {
        "",
        "-",
        "--",
        "---",
        ".",
        "?",
        "na",
        "n/a",
        "#n/a",
        "nan",
        "null",
        "none",
        "nd",
        "n.d.",
        "missing",
    }
)

TRUE_WORDS = frozenset({"true", "yes", "y", "t"})
FALSE_WORDS = frozenset({"false", "no", "n", "f"})

# Integers beyond this cannot survive a round trip through a 64 bit float, which is what every
# analysis library will store them in. Sample identifiers and genomic coordinates get this far.
LARGEST_EXACT_INTEGER = 2**53

# Delimiters worth considering. Space is left out on purpose: a space separated file is
# ambiguous with a comma separated one holding text, and guessing wrong splits every sentence.
DELIMITERS = ",\t;|"

SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# Two relationship namespaces, which is not a mistake in the file. The <Relationship> elements
# belong to the package, and the r:id attribute pointing at one of them belongs to the document.
# Using either for both finds nothing, and finding nothing here reads as a workbook with no
# sheets in it rather than as a bug.
PACKAGE_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
DOCUMENT_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
WORKSHEET_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"

# Number format identifiers that Excel defines as dates and times without writing them down
# anywhere in the file. A custom format gets an entry in <numFmts> and an id of 164 or more;
# these do not, so a reader with no table of them sees an ordinary number and reports a gene
# column that Excel ate as a column of five digit integers.
BUILTIN_DATE_FORMATS = frozenset({14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47})

# Day zero of the serial numbers Excel stores dates as. It is two days before 1900-01-01 rather
# than one, because Excel believes 1900 was a leap year, and the offset is only correct for
# dates after that imaginary February. Dates before March 1900 are out by a day and are not
# worth special casing; a 1900 date in a data file is a corrupted value, not a measurement.
EXCEL_DAY_ZERO = datetime.date(1899, 12, 30)


def count(number: int, thing: str) -> str:
    """A number and what it counts, made plural when it needs to be."""
    return f"{number:,} {thing}" if number == 1 else f"{number:,} {thing}s"


def looks_like(pattern: re.Pattern[str], value: str) -> bool:
    """Whether a whole value matches a pattern, ignoring surrounding space."""
    return pattern.fullmatch(value.strip()) is not None


def is_missing(value: str) -> bool:
    """Whether a value is one of the ways a file says there is no value."""
    return value.strip().lower() in MISSING_WORDS


def column_number(reference: str) -> int:
    """Turns a cell reference such as AB12 into a one-based column number.

    A row in a spreadsheet only holds the cells that have something in them, so the third cell
    of a row is not necessarily the third column. Reading them in order puts every value after
    a blank into the wrong column, which is a worse outcome than failing, because the columns
    stay plausible.

    :param reference: The cell reference, such as A1 or AB12.
    :return: The column number, or 0 if the reference does not start with letters.
    """
    number = 0

    for character in reference:
        if not character.isalpha():
            break

        number = number * 26 + (ord(character.upper()) - ord("A") + 1)

    return number


@dataclass(frozen=True)
class Column:
    """One column of a data file, as it actually is rather than as it is named.

    :param name: The header, or a generated name if the file has none.
    :param position: The one-based position in the file.
    :param kind: What the values are: number, text, date, boolean, or empty.
    :param filled: How many scanned rows hold a value.
    :param missing: How many do not.
    :param distinct: How many different values were seen, up to the tracking limit.
    :param counted_all: Whether distinct is exact, or a floor.
    :param smallest: The smallest number, when the column holds numbers.
    :param largest: The largest number.
    :param examples: A few values, for recognising what the column holds.
    :param notes: What is wrong with the column, in the words of what it will cost.
    """

    name: str
    position: int
    kind: str
    filled: int
    missing: int
    distinct: int
    counted_all: bool
    smallest: float | None
    largest: float | None
    examples: tuple[str, ...]
    notes: tuple[str, ...]

    def describe(self) -> str:
        """Renders one line for the column, plus a line for each of its notes."""
        counted = f"{self.distinct:,}" if self.counted_all else f"over {self.distinct:,}"
        parts = [f"{count(self.filled, 'value')}, {counted} distinct"]

        if self.missing:
            parts.append(f"{self.missing:,} missing")

        if self.smallest is not None and self.largest is not None:
            parts.append(f"{self.smallest:g} to {self.largest:g}")

        if self.examples:
            parts.append("e.g. " + ", ".join(self.examples))

        lines = [f"{self.position:>3}. {self.name} [{self.kind}] {'; '.join(parts)}"]
        lines.extend(f"      ! {note}" for note in self.notes)

        return "\n".join(lines)


@dataclass(frozen=True)
class Table:
    """What one data file holds.

    :param name: How to open the file from where the code will run.
    :param rows: How many data rows were scanned, not counting the header.
    :param complete: Whether that is the whole file or where the scan stopped.
    :param columns: The columns, already limited to what is worth listing.
    :param total_columns: How many there are in the file.
    :param sample: The first few rows, as text, for seeing how the columns line up.
    :param reading: How the file was read, to be repeated by whatever reads it next.
    :param sheet: The sheet that was read, for a spreadsheet.
    :param other_sheets: The sheets that were not.
    :param warnings: What the file will get wrong for a reader who does not know.
    """

    name: str
    rows: int
    complete: bool
    columns: tuple[Column, ...]
    total_columns: int
    sample: tuple[tuple[str, ...], ...]
    reading: str
    sheet: str | None = None
    other_sheets: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def report(self) -> str:
        """Renders the file for a model."""
        counted = (
            count(self.rows, "row") if self.complete else f"the first {count(self.rows, 'row')}"
        )
        where = f' (sheet "{self.sheet}")' if self.sheet else ""
        lines = [
            f"{self.name}{where}: {counted}, {count(self.total_columns, 'column')}. "
            f"{self.reading}"
        ]

        if not self.complete:
            lines.append(
                f"The file is longer than {MAX_TABLE_ROWS_SCANNED:,} rows. Everything below "
                f"describes the rows that were read, not the whole file."
            )

        if self.other_sheets:
            lines.append(f"Other sheets in this workbook: {', '.join(self.other_sheets)}.")

        lines.append("\nColumns:")
        lines.extend(column.describe() for column in self.columns)

        if self.total_columns > len(self.columns):
            lines.append(f"      ({self.total_columns - len(self.columns):,} more not described)")

        if self.sample:
            lines.append("\nFirst rows:")
            lines.extend("    " + " | ".join(row) for row in self.sample)

        if self.warnings:
            lines.append("\nBefore using this file:")
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines)


class ColumnScan:
    """Accumulates what is known about one column while its values go past.

    A separate mutable thing from the Column it produces, because the scan is over rows and the
    answer is over columns, and a file wide enough to be worth limiting is too wide to hold in
    memory a column at a time.
    """

    def __init__(self, name: str, position: int) -> None:
        self.name = name
        self.position = position
        self.kinds: Counter[str] = Counter()
        self.absent_words: Counter[str] = Counter()
        self.distinct: set[str] = set()
        self.overflowed = False
        self.smallest: float | None = None
        self.largest: float | None = None
        self.examples: list[str] = []
        self.odd: Counter[str] = Counter()
        self.odd_examples: dict[str, str] = {}

    def add(self, value: str, declared: str | None = None) -> None:
        """Takes one cell.

        :param value: The cell as text.
        :param declared: The kind the file itself gave, for a spreadsheet, where a date and a
            number are the same bytes and only the format tells them apart. Left out for
            delimited text, where everything is text and the kind has to be inferred.
        """
        stripped = value.strip()

        if declared in {None, "text"} and is_missing(stripped):
            self.kinds["missing"] += 1
            self.absent_words[stripped.lower()] += 1
            return

        if declared == "blank":
            self.kinds["missing"] += 1
            self.absent_words[""] += 1
            return

        kind = declared if declared is not None else self.infer(stripped)
        self.kinds[kind] += 1

        if len(self.distinct) < MAX_DISTINCT_TRACKED:
            self.distinct.add(stripped)
        elif stripped not in self.distinct:
            self.overflowed = True

        if len(self.examples) < MAX_COLUMN_EXAMPLES and stripped not in self.examples:
            self.examples.append(truncate_text(stripped, MAX_CELL_CHARACTERS))

        if kind == "number":
            self.measure(stripped)

        self.flag(stripped, kind)

    def infer(self, value: str) -> str:
        """Decides what an untyped value is."""
        if looks_like(NUMBER, value):
            return "number"

        if value.lower() in TRUE_WORDS or value.lower() in FALSE_WORDS:
            return "boolean"

        return "text"

    def measure(self, value: str) -> None:
        """Widens the numeric range."""
        number = float(value)
        self.smallest = number if self.smallest is None else min(self.smallest, number)
        self.largest = number if self.largest is None else max(self.largest, number)

    def flag(self, value: str, kind: str) -> None:
        """Records the ways this one value will be read wrongly."""
        if kind == "number" and looks_like(LEADING_ZERO, value):
            self.note("leading zeros", value)

        if kind == "number" and value.lstrip("+-").isdigit():
            if abs(int(value)) > LARGEST_EXACT_INTEGER:
                self.note("too large to be exact", value)

        if kind == "text":
            if looks_like(CENSORED, value):
                self.note("outside the assay range", value)
            elif looks_like(DECIMAL_COMMA, value):
                self.note("decimal comma", value)
            elif looks_like(GROUPED_NUMBER, value):
                self.note("thousands separator", value)

        if kind == "error":
            self.note("spreadsheet error", value)

    def note(self, what: str, example: str) -> None:
        """Counts one oddity, keeping the first example of it."""
        self.odd[what] += 1
        self.odd_examples.setdefault(what, truncate_text(example, MAX_CELL_CHARACTERS))

    def kind(self) -> str:
        """The one word for what this column holds."""
        real = {name: seen for name, seen in self.kinds.items() if name != "missing"}

        if not real:
            return "empty"

        best, seen = max(real.items(), key=lambda pair: pair[1])

        return best if seen == sum(real.values()) else f"mostly {best}"

    def finish(self) -> Column:
        """Freezes the scan into a column."""
        kind = self.kind()
        numeric = kind in {"number", "mostly number"}

        return Column(
            name=self.name,
            position=self.position,
            kind=kind,
            filled=sum(seen for name, seen in self.kinds.items() if name != "missing"),
            missing=self.kinds["missing"],
            distinct=len(self.distinct),
            counted_all=not self.overflowed,
            smallest=self.smallest if numeric else None,
            largest=self.largest if numeric else None,
            examples=tuple(self.examples),
            notes=self.describe_problems(kind),
        )

    def describe_problems(self, kind: str) -> tuple[str, ...]:
        """Turns the counts into what each one will cost."""
        notes = []
        real = sum(seen for name, seen in self.kinds.items() if name != "missing")

        if kind.startswith("mostly "):
            minority = {
                name: seen
                for name, seen in self.kinds.items()
                if name not in {"missing", kind.removeprefix("mostly ")}
            }
            named = ", ".join(
                f"{number:,} {name}" for name, number in sorted(minority.items())
            )
            notes.append(
                f"Mixed types: {named} among {count(real, 'value')}. Most readers make the "
                f"whole column text, so arithmetic on it fails or is quietly skipped."
            )

        if len(self.absent_words) > 1 or (self.absent_words and "" not in self.absent_words):
            spelled = ", ".join(
                f'"{word}" ({seen:,})' if word else f"empty ({seen:,})"
                for word, seen in sorted(self.absent_words.items())
            )
            notes.append(f"Missing values are written as {spelled}.")

        for what, seen in sorted(self.odd.items()):
            notes.append(self.explain(what, seen, self.odd_examples[what]))

        # A column holding both dates and text is the signature of the one corruption that has
        # been in the literature for a decade: a spreadsheet turns SEPT2, MARCH1 and DEC1 into
        # dates as they are typed, and a gene column comes back part symbols, part dates. A
        # column that is entirely dates is probably a date. A column that is mostly symbols with
        # a scattering of dates in it is not.
        # A column of Y and N is a column of flags, and the trap is not in reading it but in
        # using it: "N" is a non-empty string, so a flag tested for truth directly is always on.
        if kind in {"boolean", "mostly boolean"} and any(
            len(value) == 1 for value in self.distinct
        ):
            notes.append(
                "Flags written as single letters. Tested for truth as they stand, both letters "
                'are true, since "N" is a non-empty string. Compare them to the letter.'
            )

        if self.kinds["date"] and self.kinds["text"]:
            notes.append(
                f"Dates among text: {count(self.kinds['date'], 'date')} and "
                f"{count(self.kinds['text'], 'text value')} in one column. In a column of names "
                f"or identifiers that is Excel's doing rather than the data's, since SEPT2 and "
                f"MARCH1 become dates as they are typed. The symbol cannot be recovered from "
                f"this file; check it against a copy that was never opened in a spreadsheet."
            )

        return tuple(notes)

    def explain(self, what: str, seen: int, example: str) -> str:
        """Says what one kind of oddity costs.

        Each reads as a noun phrase and a consequence rather than as a sentence about how many,
        so that one occurrence and a thousand are worded the same way and the count stays a
        number in the middle rather than something the grammar has to agree with.
        """
        filled = sum(number for name, number in self.kinds.items() if name != "missing")
        many = f"{seen:,} of {filled:,} values ({example})"

        reasons = {
            "leading zeros": (
                f"Leading zeros, on {many}. These are identifiers rather than quantities; read "
                f"as numbers they lose the zeros and stop matching anything."
            ),
            "too large to be exact": (
                f"Past {LARGEST_EXACT_INTEGER:,}, on {many}. A 64 bit float cannot hold them "
                f"exactly, so they change when read as numbers."
            ),
            "outside the assay range": (
                f"Bounds rather than measurements, on {many}. These are the strongest and "
                f"weakest samples, so dropping them biases whatever is computed next."
            ),
            "decimal comma": (
                f"A comma for the decimal point, on {many}. Read with a full stop they become "
                f"text, or lose everything after the comma."
            ),
            "thousands separator": (
                f"Digits grouped with commas, on {many}. They are text until the separator is "
                f"taken out."
            ),
            "spreadsheet error": (
                f"A spreadsheet error, on {many}. The formula that produced them failed, so "
                f"there was never a value."
            ),
        }

        return reasons[what]


def resolve(work_dir: Path, filename: str) -> Path:
    """Turns a filename a model chose into a path inside the working directory.

    The filename is untrusted in exactly the way a filename a meeting asks to write is, so it is
    checked the same way and then re-checked after resolution. Resolving first also settles
    symbolic links, so a link left in the directory pointing somewhere else is refused rather
    than followed.

    :param work_dir: The directory the file must be inside.
    :param filename: The name, relative to that directory.
    :raises UnsafeFilenameError: If the name is not a plain relative path, or escapes.
    :raises TableError: If nothing is there, or it is not a file.
    :return: The resolved path.
    """
    base = work_dir.resolve()
    path = (base / check_filename(filename)).resolve()

    if not path.is_relative_to(base):
        raise UnsafeFilenameError(f'Refusing to read "{filename}": it resolves outside {base}')

    if not path.exists():
        raise TableError(f'There is no file called "{filename}" here')

    if not path.is_file():
        raise TableError(f'"{filename}" is not a file')

    return path


def decode(raw: bytes) -> tuple[str, str, bool]:
    """Turns the bytes of a text file into text, saying how.

    Tried in order rather than detected, because there is no detecting it: cp1252 maps every
    possible byte, so it always succeeds and a wrong guess is silent. UTF-8 does not, which
    makes "UTF-8 first, and cp1252 only when that fails" the one ordering that is ever right.

    :param raw: The file's bytes.
    :return: The text, the encoding used, and whether a byte-order mark was removed.
    """
    mark = raw.startswith(b"\xef\xbb\xbf")

    try:
        return raw.decode("utf-8-sig"), "utf-8", mark
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252", mark


def find_delimiter(sample: str, suffix: str) -> tuple[str, str]:
    """Works out what separates the fields.

    :param sample: The first part of the file.
    :param suffix: The file's extension, lowercased, which is a claim and not evidence.
    :return: The delimiter and how it was arrived at, for the report.
    """
    if suffix in {".tsv", ".tab"}:
        return "\t", "tab separated, from the file's name"

    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=DELIMITERS)
        return sniffed.delimiter, f"{named(sniffed.delimiter)} separated"
    except csv.Error:
        pass

    # Sniffing raises on a file with one column, which a list of gene names is, and on one whose
    # rows are not all the same width, which a hand-edited file usually is. Both are ordinary
    # enough that refusing them is not an option.
    #
    # There is no point looking here for a delimiter that cuts every line into the same number
    # of fields: wherever one exists the sniffer has already found it, which was checked across
    # forty thousand generated tables rather than assumed. What is left is the ragged file, and
    # there the delimiter is whichever appears on the most lines. Calling it one column instead
    # would put each whole row into a single field and describe a file that is not there.
    lines = [line for line in sample.splitlines() if line.strip()][:20]
    scored = sorted(
        (
            -sum(1 for line in lines if candidate in line),
            -sum(line.count(candidate) for line in lines),
            position,
            candidate,
        )
        for position, candidate in enumerate(DELIMITERS)
    )
    present, _, _, best = scored[0]

    if lines and -present > len(lines) / 2:
        return best, f"{named(best)} separated, on {-present} of {len(lines)} lines"

    return "\t", "one column, no delimiter found"


def skip_preamble(text: str) -> tuple[str, int]:
    """Drops the commented lines a data file often starts with.

    A file written by an instrument or a pipeline carries a block of settings above the header,
    each line marked with a hash. Left in, the first of them is read as the header and the whole
    file becomes one column called "# generated by".

    :param text: The file's text.
    :return: The text from the first uncommented line, and how many lines were dropped.
    """
    lines = text.splitlines(keepends=True)
    dropped = 0

    for line in lines:
        if not line.startswith("#"):
            break

        dropped += 1

    return "".join(lines[dropped:]), dropped


def named(delimiter: str) -> str:
    """The English name of a delimiter, for a report a model reads."""
    return {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}.get(delimiter, delimiter)


def name_columns(header: list[str]) -> tuple[list[str], list[str]]:
    """Cleans up a header row and says what was wrong with it.

    :param header: The first row of the file.
    :return: The column names, and the warnings.
    """
    warnings = []
    spaced = [name for name in header if name != name.strip() and name.strip()]

    if spaced:
        listed = ", ".join(f'"{name}"' for name in spaced)
        warnings.append(
            f"Column names with spaces around them: {listed}. They are shown trimmed below, but "
            f"a lookup by the name as it reads in the file will not find them."
        )

    names = [name.strip() for name in header]
    blank = [position for position, name in enumerate(names, start=1) if not name]

    for position in blank:
        names[position - 1] = f"column {position}"

    if blank:
        listed = ", ".join(str(position) for position in blank)
        warnings.append(
            f"Columns with no name, at position {listed}. They are called "
            f'"column {blank[0]}" and so on below; nothing in the file calls them that.'
        )

    repeated = sorted({name for name in names if names.count(name) > 1})

    if repeated:
        warnings.append(
            f"Repeated column names: {', '.join(repeated)}. Readers that key rows by name keep "
            f"only the last of each, so the earlier ones are dropped without a word."
        )

    # A header of numbers is not a header. Every reader takes the first row as one anyway, which
    # costs a row of data and names every column after a measurement.
    numeric = [name for name in names if looks_like(NUMBER, name)]

    if numeric:
        warnings.append(
            f"The first row was used as the header, but {len(numeric)} of its {len(names)} "
            f"cells are numbers ({', '.join(numeric[:3])}). This file probably has no header "
            f"row, in which case the first row of data is missing from everything below."
        )

    return names, warnings


def scan(
    rows: list[list[str]],
    names: list[str],
    kinds: list[list[str]] | None = None,
) -> tuple[tuple[Column, ...], list[str]]:
    """Summarises the rows of a table column by column.

    :param rows: The data rows, without the header.
    :param names: The column names.
    :param kinds: The kind of each cell, where the file declares one. Delimited files do not.
    :return: The columns, limited to what is worth listing, and the warnings.
    """
    scans = [ColumnScan(name, position) for position, name in enumerate(names, start=1)]
    short = 0
    long = 0

    for index, row in enumerate(rows):
        if len(row) < len(names):
            short += 1
        elif len(row) > len(names):
            long += 1

        declared = kinds[index] if kinds is not None else None

        for position, value in enumerate(row[: len(names)]):
            scans[position].add(value, declared[position] if declared is not None else None)

        # A row that stops early is not a row of empty cells; the cells are absent. Counting
        # them as missing is what makes the missing count match what a reader will see.
        for position in range(len(row), len(names)):
            scans[position].add("", "blank" if kinds is not None else None)

    warnings = []

    if short or long:
        parts = []

        if short:
            parts.append(f"{short:,} shorter")

        if long:
            parts.append(f"{long:,} longer")

        warnings.append(
            f"The rows are not all the same width: {' and '.join(parts)} than the "
            f"{len(names)} columns in the header. Short rows are padded with empty values and "
            f"the extra fields of long rows are dropped, here and by most other readers."
        )

    return tuple(entry.finish() for entry in scans[:MAX_COLUMNS_REPORTED]), warnings


def read_delimited(path: Path, name: str) -> Table:
    """Reads a delimited text file.

    :param path: The file.
    :param name: How to refer to it, which is how code will open it.
    :raises TableError: If the file cannot be read as a table.
    :return: The table.
    """
    with open(path, "rb") as handle:
        raw = handle.read(MAX_TABLE_BYTES + 1)

    if not raw.strip():
        raise TableError(f'"{name}" is empty')

    capped = len(raw) > MAX_TABLE_BYTES

    if capped:
        # Cut back to a line ending rather than to the byte. A multi-byte character split down
        # the middle is not valid UTF-8, and the file would be reported as cp1252 on the
        # strength of one truncated character that was never in it.
        raw = raw[: raw[:MAX_TABLE_BYTES].rfind(b"\n") + 1]

    text, encoding, mark = decode(raw)
    text, commented = skip_preamble(text)

    if not text.strip():
        raise TableError(f'"{name}" is all comments and has no table in it')

    delimiter, how = find_delimiter(text[:65536], path.suffix.lower())

    try:
        # Read through a file object rather than from split lines. str.splitlines breaks on
        # eight characters besides the newline, among them U+2028, so a cell holding one becomes
        # two rows; and a quoted field spanning lines has its newline silently removed.
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as problem:
        # The field size limit is a module-wide setting, so raising it would change the
        # behaviour of anything else in the process that parses CSV. Reporting is the honest
        # answer: a single field over 128 KB is a file that needs handling, not a wider limit.
        raise TableError(f'"{name}" could not be parsed as a table: {problem}') from problem

    rows = [row for row in rows if any(cell.strip() for cell in row)]

    if not rows:
        raise TableError(f'"{name}" has no rows with anything in them')

    names, warnings = name_columns(rows[0])
    body = rows[1 : MAX_TABLE_ROWS_SCANNED + 1]
    columns, more = scan(body, names)
    warnings.extend(more)

    if commented:
        warnings.append(
            f"The file begins with {count(commented, 'commented line')}, skipped here. Most "
            f"readers do not skip them, and the first of them then becomes the header."
        )

    if mark:
        warnings.append(
            "The file starts with a byte-order mark. It is stripped here, but a reader told "
            f'the encoding is "utf-8" rather than "utf-8-sig" will see the first column named '
            f'"\\ufeff{names[0]}", which matches nothing.'
        )

    if encoding != "utf-8":
        warnings.append(
            "The file is not valid UTF-8 and was read as cp1252. Accented characters, Greek "
            "letters and the micro sign may not be what was written."
        )

    if capped:
        warnings.append(
            f"The file is larger than {MAX_TABLE_BYTES:,} bytes. Reading stopped at the last "
            f"whole line before that, so there is more of it than is described here."
        )

    # A single column file is where "the first row is the header" is least likely to be true and
    # least likely to be noticed, since there is no second column to look odd
    if len(names) == 1 and "no delimiter" in how:
        warnings.append(
            f'No delimiter was found, so this is being read as one column called "{names[0]}", '
            f"taken from the first line. If the file is a plain list with no header then that "
            f"first line is a value and is missing from the counts above."
        )

    return Table(
        name=name,
        rows=len(body),
        complete=len(rows) - 1 <= MAX_TABLE_ROWS_SCANNED and not capped,
        columns=columns,
        total_columns=len(names),
        sample=sample_of(body, len(names)),
        reading=f"{how}, {encoding}.",
        warnings=tuple(warnings),
    )


def sample_of(rows: list[list[str]], width: int) -> tuple[tuple[str, ...], ...]:
    """Takes the first few rows, padded and shortened, for showing under the columns.

    Padded because a short row read as it stands puts the next column's heading over a value
    belonging to this one, which is the mistake the sample exists to make visible.
    """
    shown = []

    for row in rows[:MAX_SAMPLE_ROWS]:
        padded = list(row[:width]) + [""] * (width - len(row))
        cells = [
            truncate_text(cell.strip(), MAX_CELL_CHARACTERS)
            for cell in padded[:MAX_SAMPLE_COLUMNS]
        ]

        # Said rather than left to be inferred from a row that stops. A row quietly cut at the
        # eighth column reads as a row with eight columns in it.
        if width > MAX_SAMPLE_COLUMNS:
            cells.append(f"... {width - MAX_SAMPLE_COLUMNS} more")

        shown.append(tuple(cells))

    return tuple(shown)


def entry_text(archive: zipfile.ZipFile, name: str) -> str:
    """Reads one member of a spreadsheet, refusing to unpack more than a sheet's worth.

    A zip records the size of each member inside itself, so the declared size is written by
    whoever wrote the file and is not a limit on anything. A 204 KB archive declaring a 200 MB
    member is an ordinary thing to construct, which was measured rather than supposed. So the
    cap is on the bytes actually taken out.

    :param archive: The open archive.
    :param name: The member's exact name.
    :raises TableError: If the member is missing or larger than the cap.
    :return: The member as text.
    """
    try:
        with archive.open(name) as member:
            raw = member.read(MAX_SHEET_BYTES + 1)
    except KeyError as problem:
        raise TableError(f"This spreadsheet is missing {name}") from problem

    if len(raw) > MAX_SHEET_BYTES:
        raise TableError(
            f"{name} in this spreadsheet unpacks to more than {MAX_SHEET_BYTES:,} bytes"
        )

    return raw.decode("utf-8", errors="replace")


def sheets_in(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Lists a workbook's sheets in the order they appear in it.

    The order of the members in the archive is not the order of the sheets, and the file called
    sheet1.xml is not necessarily the first one: a workbook whose sheets have been reordered or
    deleted keeps the old names. The relationship identifiers are the only route from a sheet's
    name to its contents.

    :param archive: The open archive.
    :raises TableError: If the workbook cannot be read.
    :return: Each sheet's name and the member holding it.
    """
    workbook = parse_xml(entry_text(archive, "xl/workbook.xml"), TableError, "spreadsheet")
    relationships = parse_xml(
        entry_text(archive, "xl/_rels/workbook.xml.rels"), TableError, "spreadsheet"
    )

    targets = {}

    for relationship in relationships.iter(f"{PACKAGE_NS}Relationship"):
        if relationship.get("Type") != WORKSHEET_TYPE:
            continue

        target = relationship.get("Target", "")
        # A target is written either relative to xl/ or absolute within the archive, and both
        # forms appear in files written by ordinary tools
        targets[relationship.get("Id", "")] = (
            target.lstrip("/") if target.startswith("/") else f"xl/{target}"
        )

    found = []

    for sheet in workbook.iter(f"{SHEET_NS}sheet"):
        identifier = sheet.get(f"{DOCUMENT_NS}id", "")

        if identifier in targets:
            found.append((sheet.get("name", "unnamed"), targets[identifier]))

    if not found:
        raise TableError("This spreadsheet has no sheets")

    return found


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Reads the table of strings a spreadsheet shares between its cells.

    Absent in files written by some libraries, which put every string in its cell instead, so
    its absence is ordinary rather than an error. Each entry may also be split into runs where
    the formatting changes partway through a cell, and the runs have to be joined: asking for
    the first <t> below an entry returns nothing at all, because the <t> elements are inside the
    runs rather than beside them.

    :param archive: The open archive.
    :return: The strings, in index order.
    """
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = parse_xml(entry_text(archive, "xl/sharedStrings.xml"), TableError, "spreadsheet")

    return [
        "".join(piece.text or "" for piece in entry.iter(f"{SHEET_NS}t"))
        for entry in root.iter(f"{SHEET_NS}si")
    ]


def date_styles(archive: zipfile.ZipFile) -> set[int]:
    """Finds which cell styles mean the number in the cell is a date.

    Without this a date is a five digit integer, which is how a gene symbol Excel turned into a
    date gets reported as a measurement.

    :param archive: The open archive.
    :return: The style indices that are dates.
    """
    if "xl/styles.xml" not in archive.namelist():
        return set()

    root = parse_xml(entry_text(archive, "xl/styles.xml"), TableError, "spreadsheet")
    dated = set(BUILTIN_DATE_FORMATS)

    for custom in root.iter(f"{SHEET_NS}numFmt"):
        code = (custom.get("formatCode") or "").lower()

        # Quoted literals, escapes and locale prefixes are removed before the letters are read,
        # because they are text that happens to contain letters: the format 0.00"m/s" is a speed
        # and would otherwise be a date, and every value under it a day in 1900.
        bare = re.sub(r'"[^"]*"|\\.|\[[^\]]*\]', "", code)

        # A format means a date if it positions the parts of one. There is no flag saying so;
        # the letters are the only evidence the file carries.
        if "y" in bare or ("d" in bare and "m" in bare):
            dated.add(int(custom.get("numFmtId", "-1")))

    styles = set()

    for cell_formats in root.iter(f"{SHEET_NS}cellXfs"):
        for index, entry in enumerate(cell_formats.iter(f"{SHEET_NS}xf")):
            if int(entry.get("numFmtId", "0")) in dated:
                styles.add(index)

    return styles


def cell_value(
    cell,  # type: ignore[no-untyped-def]
    strings: list[str],
    dated: set[int],
) -> tuple[str, str]:
    """Turns one cell into text and a kind.

    Six ways of holding a value, and a reader that knows one of them misreads the rest. A file
    written by Excel shares its strings and refers to them by index; one written by a library
    may put the text in the cell; a formula's cached result is a seventh thing again.

    :param cell: The <c> element.
    :param strings: The shared strings.
    :param dated: Style indices that mean a date.
    :return: The cell as text, and its kind.
    """
    kind = cell.get("t", "n")
    inline = cell.find(f"{SHEET_NS}is")
    holder = cell.find(f"{SHEET_NS}v")
    raw = (holder.text or "") if holder is not None else ""

    if kind == "inlineStr":
        runs = inline.iter(f"{SHEET_NS}t") if inline is not None else ()
        text = "".join(piece.text or "" for piece in runs)

        return text, "text" if text.strip() else "blank"

    if kind == "s":
        index = int(raw) if raw.lstrip("-").isdigit() else -1
        text = strings[index] if 0 <= index < len(strings) else ""
        return text, "text" if text.strip() else "blank"

    if kind == "str":
        return raw, "text" if raw.strip() else "blank"

    if kind == "e":
        return raw, "error"

    if kind == "b":
        return ("TRUE" if raw == "1" else "FALSE"), "boolean"

    if not looks_like(NUMBER, raw):
        # Two cases, and the same answer to both. A cell claiming to be a number and holding
        # something else is reported as what it is rather than trusted, since everything
        # downstream would otherwise call float() on it. An empty one is a formula whose result
        # was never calculated, which is what a file written by a library rather than by Excel
        # holds for every formula in it, and empty text is already counted as a missing value.
        return raw, "text"

    if int(cell.get("s", "-1") or "-1") in dated:
        return excel_date(float(raw)), "date"

    return raw, "number"


def excel_date(serial: float) -> str:
    """Turns a spreadsheet's day number into a date."""
    moment = EXCEL_DAY_ZERO + datetime.timedelta(days=serial)
    return moment.isoformat()


def read_spreadsheet(path: Path, name: str, sheet: str | None = None) -> Table:
    """Reads one sheet of a spreadsheet.

    :param path: The file.
    :param name: How to refer to it.
    :param sheet: Which sheet, by name. The first one when not given.
    :raises TableError: If the file is not a usable spreadsheet, or has no such sheet.
    :return: The table.
    """
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as problem:
        raise TableError(
            f'"{name}" is not a readable spreadsheet. A file saved as .xls rather than .xlsx is '
            f"a different format entirely and has to be re-saved."
        ) from problem

    with archive:
        if len(archive.namelist()) > MAX_ARCHIVE_ENTRIES:
            raise TableError(f'"{name}" holds more parts than a spreadsheet should')

        available = sheets_in(archive)
        chosen = pick_sheet(available, sheet, name)
        strings = shared_strings(archive)
        dated = date_styles(archive)
        root = parse_xml(entry_text(archive, chosen[1]), TableError, "spreadsheet")
        rows, kinds, capped = sheet_rows(root, strings, dated)

    if not rows:
        raise TableError(f'Sheet "{chosen[0]}" of "{name}" has nothing in it')

    names, warnings = name_columns(rows[0])
    body, body_kinds = rows[1:], kinds[1:]
    columns, more = scan(body, names, body_kinds)
    warnings.extend(more)

    return Table(
        name=name,
        rows=len(body),
        complete=not capped,
        columns=columns,
        total_columns=len(names),
        sample=sample_of(body, len(names)),
        reading="read as a spreadsheet.",
        sheet=chosen[0],
        other_sheets=tuple(other for other, _ in available if other != chosen[0]),
        warnings=tuple(warnings),
    )


def pick_sheet(available: list[tuple[str, str]], wanted: str | None, name: str) -> tuple[str, str]:
    """Chooses which sheet to read, refusing rather than guessing at a name that is not there."""
    if wanted is None:
        return available[0]

    for sheet in available:
        if sheet[0] == wanted:
            return sheet

    offered = ", ".join(sheet[0] for sheet in available)

    raise TableError(f'"{name}" has no sheet called "{wanted}". It has: {offered}.')


def sheet_rows(
    root,  # type: ignore[no-untyped-def]
    strings: list[str],
    dated: set[int],
) -> tuple[list[list[str]], list[list[str]], bool]:
    """Turns a sheet's XML into rows of text and rows of kinds.

    Cells are placed by their reference rather than taken in order, because a row holds only the
    cells that have something in them. A row whose second cell is blank arrives as two cells,
    and reading them in order moves every later value one column to the left, which produces a
    table that is wrong and looks right.

    :param root: The worksheet element.
    :param strings: The shared strings.
    :param dated: Style indices that mean a date.
    :return: The values, the kinds, and whether the scan stopped early.
    """
    rows: list[list[str]] = []
    kinds: list[list[str]] = []
    capped = False

    for row in root.iter(f"{SHEET_NS}row"):
        if len(rows) > MAX_TABLE_ROWS_SCANNED:
            capped = True
            break

        placed: dict[int, tuple[str, str]] = {}

        for cell in row.iter(f"{SHEET_NS}c"):
            position = column_number(cell.get("r", ""))

            if position:
                placed[position] = cell_value(cell, strings, dated)

        if not placed:
            continue

        width = max(placed)
        rows.append([placed.get(index, ("", "blank"))[0] for index in range(1, width + 1)])
        kinds.append([placed.get(index, ("", "blank"))[1] for index in range(1, width + 1)])

    if not rows:
        return [], [], capped

    # Every row is padded to the widest, because a header shorter than its data means the last
    # columns have no name rather than that the rows are ragged
    width = max(len(row) for row in rows)

    for row, kind in zip(rows, kinds):
        row.extend([""] * (width - len(row)))
        kind.extend(["blank"] * (width - len(kind)))

    return rows, kinds, capped


def describe_table(work_dir: Path, filename: str, sheet: str | None = None) -> Table:
    """Reads a data file and says what is in it.

    :param work_dir: The directory the file must be inside.
    :param filename: The file, relative to that directory.
    :param sheet: Which sheet, for a spreadsheet.
    :raises UnsafeFilenameError: If the filename escapes the directory.
    :raises TableError: If the file cannot be read as a table.
    :return: The table.
    """
    path = resolve(work_dir, filename)
    suffix = path.suffix.lower()

    if suffix in SPREADSHEET_SUFFIXES:
        return read_spreadsheet(path, filename, sheet)

    if suffix in DELIMITED_SUFFIXES:
        if sheet is not None:
            raise TableError(f'"{filename}" is a text file and has no sheets')

        return read_delimited(path, filename)

    readable = ", ".join(sorted(DELIMITED_SUFFIXES + SPREADSHEET_SUFFIXES))

    raise TableError(
        f'"{filename}" is not a kind of file this reads. It reads {readable}. Anything else has '
        f"to be opened by code in the sandbox."
    )


@dataclass(frozen=True)
class DataFile:
    """One file found in the working directory.

    :param name: How to open it from where the code runs.
    :param size: Its size in bytes.
    :param readable: Whether describe_table can read it.
    """

    name: str
    size: int
    readable: bool


@dataclass(frozen=True)
class DataFiles:
    """What is in the working directory.

    :param files: The files found, in order.
    :param complete: Whether that is all of them.
    """

    files: tuple[DataFile, ...]
    complete: bool

    def report(self) -> str:
        """Renders the listing for a model."""
        if not self.files:
            return (
                "There are no data files here yet. Files fetched by other tools are written "
                "into this directory, and code that runs writes into it too."
            )

        lines = [f"{len(self.files)} files:"]
        lines.extend(
            f"  {file.name} ({file.size:,} bytes)"
            + ("" if file.readable else " - not a table; open it in code")
            for file in self.files
        )

        if not self.complete:
            lines.append(f"  (more than {MAX_DATA_FILES_LISTED} files; the rest are not listed)")

        return "\n".join(lines)


def list_data_files(work_dir: Path) -> DataFiles:
    """Lists the files an agent can open.

    :param work_dir: The directory the code runs in.
    :return: The files, sorted by name.
    """
    base = work_dir.resolve()

    if not base.is_dir():
        return DataFiles(files=(), complete=True)

    found = []

    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        # Resolved and re-checked so that a symbolic link left in the directory is not listed as
        # though it were a file inside it
        if not path.resolve().is_relative_to(base):
            continue

        found.append(
            DataFile(
                name=str(path.relative_to(base)),
                size=path.stat().st_size,
                readable=path.suffix.lower() in DELIMITED_SUFFIXES + SPREADSHEET_SUFFIXES,
            )
        )

    return DataFiles(
        files=tuple(found[:MAX_DATA_FILES_LISTED]),
        complete=len(found) <= MAX_DATA_FILES_LISTED,
    )
