"""Tests for reading the data files a project works on.

Everything here is built on disk rather than mocked, because every interesting failure in this
module is a property of a real file: a byte-order mark is three bytes, a spreadsheet is a zip
archive, and a row that is missing its middle cell is missing it in the XML. A fake file agrees
with whatever the code believes about it, which is the one thing worth testing here.

The spreadsheets are assembled by hand for the same reason. A library that writes them would
write the dialect it reads, and the point of most of these is that there is more than one:
strings shared between cells or held in them, a relationship identifier in one namespace
pointing at a relationship element in another, a date that is only a date if you follow the
style to the number format.
"""

import io
import zipfile
from pathlib import Path

import pytest

from virtual_lab.artifacts import UnsafeFilenameError
from virtual_lab.constants import (
    MAX_ARCHIVE_ENTRIES,
    MAX_CELL_CHARACTERS,
    MAX_COLUMN_EXAMPLES,
    MAX_COLUMNS_REPORTED,
    MAX_DATA_FILES_LISTED,
    MAX_DISTINCT_TRACKED,
    MAX_SAMPLE_COLUMNS,
    MAX_SAMPLE_ROWS,
    MAX_SHEET_BYTES,
    MAX_SPREADSHEET_COLUMNS,
    MAX_TABLE_ROWS_SCANNED,
)
from virtual_lab.tables import (
    NUMBER,
    TableError,
    column_number,
    decode,
    describe_table,
    find_delimiter,
    list_data_files,
    looks_like,
    name_columns,
)

SHEET_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PACKAGE = "http://schemas.openxmlformats.org/package/2006/relationships"
DOCUMENT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def sheet(rows: str) -> str:
    """A worksheet holding the given <row> elements."""
    return f'<worksheet xmlns="{SHEET_MAIN}"><sheetData>{rows}</sheetData></worksheet>'


def row(number: int, cells: str) -> str:
    """One <row>."""
    return f'<row r="{number}">{cells}</row>'


def inline(reference: str, text: str) -> str:
    """A cell holding its own text, which is what several writing libraries produce."""
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def number(reference: str, value: str, style: int | None = None) -> str:
    """A numeric cell, optionally carrying a style index."""
    styled = f' s="{style}"' if style is not None else ""
    return f'<c r="{reference}"{styled}><v>{value}</v></c>'


def shared(reference: str, index: int) -> str:
    """A cell pointing into the shared string table, which is what Excel itself writes."""
    return f'<c r="{reference}" t="s"><v>{index}</v></c>'


def workbook(
    sheets: dict[str, str],
    strings: list[str] | None = None,
    formats: list[str] | None = None,
    custom: str = "",
    padding: int = 0,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    """Assembles a spreadsheet.

    :param sheets: Each sheet's name and its worksheet XML, in order.
    :param strings: The shared string table, left out entirely when not given.
    :param formats: A number format id per cell style, so a test can make a cell a date.
    :param custom: Extra <numFmt> entries, for a format the file has to define itself.
    :param padding: Extra members, for the archive size limit.
    :param compression: Stored rather than deflated when a test needs to edit a member's bytes.
    :return: The file's bytes.
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", compression) as archive:
        entries = []
        relationships = []

        for index, (name, body) in enumerate(sheets.items(), start=1):
            path = f"xl/worksheets/sheet{index}.xml"
            archive.writestr(path, body)
            entries.append(f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}" />')
            relationships.append(
                f'<Relationship Id="rId{index}" Target="/{path}" '
                f'Type="{DOCUMENT}/worksheet" />'
            )

        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{SHEET_MAIN}" xmlns:r="{DOCUMENT}">'
            f"<sheets>{''.join(entries)}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{PACKAGE}">{"".join(relationships)}</Relationships>',
        )

        if strings is not None:
            items = "".join(f"<si><t>{value}</t></si>" for value in strings)
            archive.writestr(
                "xl/sharedStrings.xml", f'<sst xmlns="{SHEET_MAIN}">{items}</sst>'
            )

        if formats is not None or custom:
            styles = "".join(f'<xf numFmtId="{entry}" />' for entry in formats or ["0"])
            archive.writestr(
                "xl/styles.xml",
                f'<styleSheet xmlns="{SHEET_MAIN}"><numFmts>{custom}</numFmts>'
                f"<cellXfs>{styles}</cellXfs></styleSheet>",
            )

        for index in range(padding):
            archive.writestr(f"xl/filler{index}.xml", "<a />")

    return buffer.getvalue()


def rebuild(raw: bytes, name: str, change) -> bytes:
    """Rewrites one member of a workbook, for a fixture a plain writer cannot produce."""
    source = zipfile.ZipFile(io.BytesIO(raw))
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for member in source.namelist():
            text = source.read(member).decode("utf-8")
            archive.writestr(member, change(text) if member == name else text)

    return buffer.getvalue()


def write(directory: Path, name: str, content: str | bytes) -> Path:
    """Puts a file in the working directory."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")

    return path


class TestWorkingOutWhatSeparatesTheFields:
    """The delimiter, which a file states only in its extension and often wrongly."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("gene,logFC\nTP53,1.2\nBRCA1,-2.0\n", ","),
            ("gene\tlogFC\nTP53\t1.2\nBRCA1\t-2.0\n", "\t"),
            ("gene;logFC\nTP53;1,2\nBRCA1;-2,0\n", ";"),
            ("gene|logFC\nTP53|1.2\nBRCA1|-2.0\n", "|"),
        ],
    )
    def test_the_usual_separators_are_recognised(self, text, expected) -> None:
        assert find_delimiter(text, ".csv")[0] == expected

    def test_a_named_tsv_is_tab_separated_whatever_it_contains(self) -> None:
        # A tsv holding free text with commas in it sniffs as comma separated, which splits
        # every sentence into columns
        text = "gene\tnote\nTP53\tup, strongly\nBRCA1\tdown, weakly\n"
        delimiter, how = find_delimiter(text, ".tsv")

        assert delimiter == "\t"
        assert "from the file's name" in how

    def test_a_single_column_file_is_read_rather_than_refused(self) -> None:
        # csv.Sniffer raises on this, and a list of gene names is an ordinary file
        delimiter, how = find_delimiter("TP53\nBRCA1\nEGFR\n", ".txt")

        assert delimiter == "\t"
        assert "no delimiter found" in how

    def test_a_ragged_file_is_still_read_as_the_table_it_is(self) -> None:
        # csv.Sniffer raises on this rather than picking the comma, and calling it one column
        # would put every row in a single field and describe a file that is not there
        delimiter, how = find_delimiter("a,b,c\n1,2,3\n4,5\n6,7,8,9\n", ".csv")

        assert delimiter == ","
        assert "on 4 of 4 lines" in how

    def test_when_two_candidates_are_on_every_line_the_commoner_one_wins(self) -> None:
        # Semicolons and commas both appear on all three lines and neither divides them evenly,
        # so the count is what separates them. Comma is tried first, and would win on order
        # alone; there are twice as many semicolons.
        delimiter, how = find_delimiter("gene;a;b,note\nTP53;1,x\nBRCA1;2;3;4,y,z\n", ".csv")

        assert delimiter == ";"
        assert "on 3 of 3 lines" in how

    def test_a_dead_tie_goes_to_the_likeliest_delimiter_not_the_lowest_byte(self) -> None:
        # Comma and tab appear on the same lines the same number of times. Left to the
        # characters themselves a tab sorts first, and a comma separated file becomes two
        # columns of nonsense.
        delimiter, _ = find_delimiter("a,b\tc\nd,e\tf\tg,h\n", ".csv")

        assert delimiter == ","

    def test_a_delimiter_on_a_minority_of_lines_is_not_the_delimiter(self) -> None:
        # One stray comma in a paragraph is not a column boundary
        delimiter, how = find_delimiter("TP53\nBRCA1\nEGFR, also known as ERBB1\nMYC\n", ".txt")

        assert "no delimiter found" in how

    def test_a_commented_preamble_is_skipped_rather_than_read_as_the_header(
        self, tmp_path
    ) -> None:
        # An instrument writes its settings above the table. Sniffing fails on the whole file,
        # and the first comment becomes a one-column header for everything below it.
        write(
            tmp_path,
            "run.csv",
            "# instrument: qExactive\n# date: 2025-09-02\ngene,logFC\nTP53,1.2\nBRCA1,-2.0\n",
        )
        table = describe_table(tmp_path, "run.csv")

        assert [column.name for column in table.columns] == ["gene", "logFC"]
        assert table.rows == 2
        assert any("2 commented lines" in warning for warning in table.warnings)

    def test_a_hash_inside_the_table_is_not_a_comment(self, tmp_path) -> None:
        write(tmp_path, "run.csv", "gene,note\nTP53,#1 hit\nBRCA1,ok\n")
        table = describe_table(tmp_path, "run.csv")

        assert table.rows == 2
        assert not any("commented" in warning for warning in table.warnings)

    def test_a_file_of_nothing_but_comments_is_refused(self, tmp_path) -> None:
        write(tmp_path, "run.csv", "# nothing\n# to see\n")

        with pytest.raises(TableError, match="all comments"):
            describe_table(tmp_path, "run.csv")

    def test_a_newline_inside_a_quoted_cell_does_not_start_a_row(self, tmp_path) -> None:
        write(tmp_path, "notes.csv", 'gene,note\nTP53,"line one\nline two"\nBRCA1,ok\n')
        table = describe_table(tmp_path, "notes.csv")

        assert table.rows == 2
        assert table.sample[0][1] == "line one\nline two"

    def test_a_unicode_line_separator_in_a_cell_does_not_start_a_row(self, tmp_path) -> None:
        # str.splitlines breaks on eight characters besides the newline. A cell holding one of
        # them becomes two rows, and the second is a row with one column in it.
        write(tmp_path, "notes.csv", "gene,note\nTP53,a\u2028b\nBRCA1,ok\n")
        table = describe_table(tmp_path, "notes.csv")

        assert table.rows == 2
        assert table.sample[0][1] == "a\u2028b"
        assert not any("same width" in warning for warning in table.warnings)

    def test_a_quoted_delimiter_does_not_become_a_column(self, tmp_path) -> None:
        write(
            tmp_path,
            "quoted.csv",
            'gene,description,logFC\nTP53,"tumor protein p53, isoform a",1.23\n',
        )
        table = describe_table(tmp_path, "quoted.csv")

        assert table.total_columns == 3
        assert table.sample[0][1] == "tumor protein p53, isoform a"


class TestTurningBytesIntoText:
    """The encoding, which is never stated and cannot be detected, only tried."""

    def test_a_utf_8_file_reads_as_utf_8(self) -> None:
        text, encoding, mark = decode("gene,conc\nTP53,5 \u00b5M\n".encode("utf-8"))

        assert encoding == "utf-8"
        assert mark is False
        assert "5 \u00b5M" in text

    def test_a_windows_file_is_not_read_as_broken_utf_8(self) -> None:
        text, encoding, mark = decode("TP53,5 \u00b5M".encode("cp1252"))

        assert encoding == "cp1252"
        assert "5 \u00b5M" in text

    def test_a_byte_order_mark_is_taken_off_the_first_column(self) -> None:
        text, encoding, mark = decode("gene,logFC\nTP53,1.2\n".encode("utf-8-sig"))

        assert mark is True
        assert text.startswith("gene")

    def test_the_mark_is_reported_because_another_reader_will_keep_it(self, tmp_path) -> None:
        write(tmp_path, "bom.csv", "gene,logFC\nTP53,1.2\n".encode("utf-8-sig"))
        table = describe_table(tmp_path, "bom.csv")

        assert table.columns[0].name == "gene"
        assert any("byte-order mark" in warning for warning in table.warnings)

    def test_a_file_that_is_not_utf_8_says_so(self, tmp_path) -> None:
        write(tmp_path, "latin.csv", "gene,conc\nTP53,5 \u00b5M\n".encode("cp1252"))
        table = describe_table(tmp_path, "latin.csv")

        assert any("cp1252" in warning for warning in table.warnings)


class TestWhatCountsAsANumber:
    """Numbers, which float() is far too generous about."""

    @pytest.mark.parametrize("value", ["1", "-1", "1.5", "+0.5", ".5", "1e5", "1E-5", "0012"])
    def test_a_number_is_a_number(self, value) -> None:
        assert looks_like(NUMBER, value)

    @pytest.mark.parametrize("value", ["1_000", "nan", "inf", "-inf", "Infinity", "\u0661\u0662"])
    def test_what_float_accepts_and_a_scientist_does_not(self, value) -> None:
        # Every one of these is accepted by float(). A column whose missing values are spelled
        # "nan" would otherwise be reported as numeric with a range running to nan.
        assert float(value.replace("\u0661\u0662", "12"))
        assert not looks_like(NUMBER, value)

    @pytest.mark.parametrize("value", ["1,000", "0x10", "", "-", "1.2.3", "5 uM", "<0.001"])
    def test_what_is_not_a_number_at_all(self, value) -> None:
        assert not looks_like(NUMBER, value)

    def test_a_column_of_nan_text_is_not_a_numeric_column(self, tmp_path) -> None:
        write(tmp_path, "nan.csv", "gene,score\nTP53,nan\nBRCA1,nan\n")
        table = describe_table(tmp_path, "nan.csv")

        assert table.columns[1].kind == "empty"
        assert table.columns[1].smallest is None


class TestTheHeaderRow:
    """Column names, which are the part of a file most often quietly wrong."""

    def test_names_are_trimmed_and_the_original_spacing_is_reported(self) -> None:
        names, warnings = name_columns(["gene ", " logFC"])

        assert names == ["gene", "logFC"]
        assert any('"gene "' in warning for warning in warnings)

    def test_an_unnamed_column_gets_a_name_that_says_it_is_made_up(self) -> None:
        names, warnings = name_columns(["gene", "", "logFC"])

        assert names[1] == "column 2"
        assert any("nothing in the file calls them that" in warning for warning in warnings)

    def test_a_repeated_name_is_reported_because_a_column_will_vanish(self) -> None:
        names, warnings = name_columns(["gene", "value", "value"])

        assert any("Repeated column names: value" in warning for warning in warnings)
        assert any("keep only the last" in warning for warning in warnings)

    def test_a_warning_about_a_wide_header_lists_only_the_first_few(self) -> None:
        # A matrix of samples runs to tens of thousands of columns, and a warning listing each
        # blank one was measured at 129 KB, which is a context window spent on commas
        names, warnings = name_columns([""] * 20 + ["x"] * 12 + [f"{n}.5" for n in range(9)])
        blank = next(warning for warning in warnings if "no name" in warning)
        repeated = next(warning for warning in warnings if "Repeated" in warning)
        numeric = next(warning for warning in warnings if "no header" in warning)

        assert "at position 1, 2, 3, 4, 5 and 15 more." in blank
        assert "column 20" not in repeated
        assert "x" in repeated
        assert "0.5, 1.5, 2.5, 3.5, 4.5 and 4 more" in numeric
        assert len(names) == 41

    def test_a_list_that_fits_is_not_said_to_have_more(self) -> None:
        names, warnings = name_columns([""] * 5 + ["gene"])

        assert any("at position 1, 2, 3, 4, 5." in warning for warning in warnings)
        assert not any("more" in warning for warning in warnings)

    def test_a_long_name_is_shortened_but_not_taken_for_a_repeat(self) -> None:
        # Shortening first would make two names that differ only past the cut look the same
        first, second = "x" * 500 + "a", "x" * 500 + "b"
        names, warnings = name_columns([first, second, " " + "y" * 5_000, "z" * 5_000, "z" * 5_000])

        assert any("Repeated" in warning for warning in warnings)
        assert not any("x" * 100 in warning for warning in warnings)
        assert all(len(name) < MAX_CELL_CHARACTERS + 60 for name in names)
        assert "characters not shown" in names[0]
        assert all(len(warning) < 1_000 for warning in warnings)

    def test_a_header_of_numbers_is_probably_not_a_header(self) -> None:
        names, warnings = name_columns(["TP53", "1.23", "0.001"])

        assert any("probably has no header row" in warning for warning in warnings)

    def test_an_ordinary_header_is_not_accused_of_being_data(self) -> None:
        names, warnings = name_columns(["gene", "logFC", "pval"])

        assert warnings == []

    def test_a_single_column_file_says_it_took_the_first_line_as_the_name(
        self, tmp_path
    ) -> None:
        write(tmp_path, "genes.txt", "TP53\nBRCA1\nEGFR\n")
        table = describe_table(tmp_path, "genes.txt")

        assert table.columns[0].name == "TP53"
        assert any("plain list with no header" in warning for warning in table.warnings)


class TestRowsThatDoNotFitTheHeader:
    """Ragged rows, which every reader handles silently and differently."""

    def test_a_short_row_and_a_long_one_are_both_reported(self, tmp_path) -> None:
        write(tmp_path, "ragged.csv", "a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
        table = describe_table(tmp_path, "ragged.csv")

        warning = next(w for w in table.warnings if "not all the same width" in w)

        assert "1 shorter" in warning
        assert "1 longer" in warning

    def test_rows_that_are_only_too_long_are_not_also_called_short(self, tmp_path) -> None:
        write(tmp_path, "ragged.csv", "a,b\n1,2\n3,4,5\n")
        table = describe_table(tmp_path, "ragged.csv")

        warning = next(w for w in table.warnings if "not all the same width" in w)

        assert "1 longer" in warning
        assert "shorter" not in warning

    def test_a_missing_cell_counts_as_missing_rather_than_as_nothing(self, tmp_path) -> None:
        write(tmp_path, "ragged.csv", "a,b,c\n1,2,3\n4,5\n")
        table = describe_table(tmp_path, "ragged.csv")

        assert table.columns[2].filled == 1
        assert table.columns[2].missing == 1
        # Padded in the sample too, so the short row still lines up under its headings
        assert table.sample[1] == ("4", "5", "")

    def test_an_even_file_is_not_warned_about(self, tmp_path) -> None:
        write(tmp_path, "even.csv", "a,b\n1,2\n3,4\n")
        table = describe_table(tmp_path, "even.csv")

        assert not any("same width" in warning for warning in table.warnings)


class TestTheWaysAFileSaysThereIsNoValue:
    """Missing values, which are words as often as they are blanks."""

    def test_the_words_are_counted_as_missing_and_then_named(self, tmp_path) -> None:
        write(tmp_path, "gaps.csv", "gene,score\nA,1\nB,NA\nC,ND\nD,\nE,n/a\n")
        table = describe_table(tmp_path, "gaps.csv")
        score = table.columns[1]

        assert score.filled == 1
        assert score.missing == 4

        note = next(note for note in score.notes if "Missing values" in note)

        for word in ['"na"', '"nd"', '"n/a"', "empty"]:
            assert word in note

    def test_an_ordinary_blank_is_not_worth_remarking_on(self, tmp_path) -> None:
        write(tmp_path, "gaps.csv", "gene,score\nA,1\nB,\n")
        table = describe_table(tmp_path, "gaps.csv")

        assert table.columns[1].missing == 1
        assert not any("Missing values" in note for note in table.columns[1].notes)

    def test_a_word_that_means_nothing_does_not_make_the_column_text(self, tmp_path) -> None:
        write(tmp_path, "gaps.csv", "gene,score\nA,1\nB,NA\nC,3\n")
        table = describe_table(tmp_path, "gaps.csv")

        assert table.columns[1].kind == "number"
        assert table.columns[1].smallest == 1


class TestValuesThatChangeWhenRead:
    """The whole point of the module: a file that loads, and is wrong afterwards."""

    def test_one_bound_makes_a_column_of_measurements_text(self, tmp_path) -> None:
        write(tmp_path, "ic50.csv", "gene,ic50\nA,12.4\nB,<0.001\nC,44.0\n")
        table = describe_table(tmp_path, "ic50.csv")
        ic50 = table.columns[1]

        assert ic50.kind == "mostly number"
        assert any("Mixed types" in note for note in ic50.notes)

        note = next(note for note in ic50.notes if "Bounds rather than" in note)

        assert "1 of 3 values (<0.001)" in note
        assert "biases" in note

    @pytest.mark.parametrize("value", ["<0.001", ">100", "<=5", ">= 2.5e-3"])
    def test_the_shapes_a_bound_is_written_in(self, tmp_path, value) -> None:
        write(tmp_path, "ic50.csv", f"gene,ic50\nA,1.0\nB,{value}\n")
        table = describe_table(tmp_path, "ic50.csv")

        assert any("Bounds rather than" in note for note in table.columns[1].notes)

    def test_leading_zeros_are_reported_because_a_numeric_read_drops_them(
        self, tmp_path
    ) -> None:
        write(tmp_path, "ids.csv", "sample,n\n00123,1\n00124,2\n")
        table = describe_table(tmp_path, "ids.csv")

        note = next(note for note in table.columns[0].notes if "Leading zeros" in note)

        assert "2 of 2 values (00123)" in note

    def test_an_ordinary_number_is_not_accused_of_leading_zeros(self, tmp_path) -> None:
        write(tmp_path, "ids.csv", "sample,n\n0,1\n0.5,2\n")
        table = describe_table(tmp_path, "ids.csv")

        assert not any("Leading zeros" in note for note in table.columns[0].notes)

    def test_an_integer_too_large_for_a_float_is_reported(self, tmp_path) -> None:
        write(tmp_path, "big.csv", "id,n\n9007199254740993,1\n")
        table = describe_table(tmp_path, "big.csv")

        note = next(note for note in table.columns[0].notes if "Past" in note)

        assert "9,007,199,254,740,992" in note

    def test_an_integer_a_float_can_hold_is_left_alone(self, tmp_path) -> None:
        write(tmp_path, "big.csv", "id,n\n9007199254740991,1\n")
        table = describe_table(tmp_path, "big.csv")

        assert not any("Past" in note for note in table.columns[0].notes)

    def test_a_decimal_comma_is_reported(self, tmp_path) -> None:
        write(tmp_path, "euro.csv", "gene;logFC\nA;1,23\nB;-2,10\n")
        table = describe_table(tmp_path, "euro.csv")

        note = next(
            note for note in table.columns[1].notes if "comma for the decimal point" in note
        )

        assert "2 of 2 values (1,23)" in note

    def test_grouped_digits_are_reported(self, tmp_path) -> None:
        write(tmp_path, "counts.tsv", "gene\tcount\nA\t1,234\nB\t2,345,678\n")
        table = describe_table(tmp_path, "counts.tsv")

        assert any("grouped with commas" in note for note in table.columns[1].notes)

    def test_single_letter_flags_are_reported_because_both_letters_are_true(
        self, tmp_path
    ) -> None:
        write(tmp_path, "flags.csv", "gene,passed\nA,Y\nB,N\nC,Y\n")
        table = describe_table(tmp_path, "flags.csv")

        assert table.columns[1].kind == "boolean"
        assert any("non-empty string" in note for note in table.columns[1].notes)

    def test_spelled_out_flags_are_not_accused_of_the_letter_problem(self, tmp_path) -> None:
        write(tmp_path, "flags.csv", "gene,passed\nA,true\nB,false\n")
        table = describe_table(tmp_path, "flags.csv")

        assert table.columns[1].kind == "boolean"
        assert not any("non-empty string" in note for note in table.columns[1].notes)


class TestRefusingToLeaveTheDirectory:
    """The filename comes from a model, so it is treated as an attempt until it is not."""

    @pytest.mark.parametrize(
        "filename",
        [
            "../secrets.csv",
            "../../etc/passwd",
            "/etc/passwd",
            "~/.ssh/id_rsa",
            "data/../../out.csv",
            "C:\\data.csv",
            "sub/../../escape.csv",
        ],
    )
    def test_a_path_leading_out_is_refused(self, tmp_path, filename) -> None:
        with pytest.raises(UnsafeFilenameError):
            describe_table(tmp_path, filename)

    def test_a_symbolic_link_out_of_the_directory_is_refused(self, tmp_path) -> None:
        outside = tmp_path.parent / "outside.csv"
        outside.write_text("secret,value\na,1\n")
        work = tmp_path / "work"
        work.mkdir()
        (work / "link.csv").symlink_to(outside)

        # The name is a plain relative one and the file is really there, so nothing but
        # resolving the link first catches this
        with pytest.raises(UnsafeFilenameError):
            describe_table(work, "link.csv")

    def test_a_link_inside_the_directory_is_fine(self, tmp_path) -> None:
        write(tmp_path, "real.csv", "gene,n\nTP53,1\n")
        (tmp_path / "link.csv").symlink_to(tmp_path / "real.csv")

        assert describe_table(tmp_path, "link.csv").total_columns == 2

    def test_a_subdirectory_is_allowed(self, tmp_path) -> None:
        write(tmp_path, "data/inner.csv", "gene,n\nTP53,1\n")

        assert describe_table(tmp_path, "data/inner.csv").rows == 1

    def test_a_missing_file_says_so_rather_than_raising_something_else(self, tmp_path) -> None:
        with pytest.raises(TableError, match="no file called"):
            describe_table(tmp_path, "nope.csv")

    def test_a_directory_is_not_a_file(self, tmp_path) -> None:
        (tmp_path / "folder.csv").mkdir()

        with pytest.raises(TableError, match="is not a file"):
            describe_table(tmp_path, "folder.csv")

    def test_a_kind_of_file_this_cannot_read_is_refused_by_name(self, tmp_path) -> None:
        write(tmp_path, "notes.md", "hello")

        with pytest.raises(TableError, match="not a kind of file"):
            describe_table(tmp_path, "notes.md")

    def test_an_empty_file_is_refused(self, tmp_path) -> None:
        write(tmp_path, "empty.csv", "   \n\n")

        with pytest.raises(TableError, match="empty"):
            describe_table(tmp_path, "empty.csv")

    def test_a_text_file_has_no_sheets(self, tmp_path) -> None:
        write(tmp_path, "plain.csv", "gene,n\nTP53,1\n")

        with pytest.raises(TableError, match="no sheets"):
            describe_table(tmp_path, "plain.csv", sheet="Sheet1")


class TestListingWhatIsThere:
    """The tool an agent needs before it can name a file at all."""

    def test_the_files_are_listed_with_their_sizes(self, tmp_path) -> None:
        write(tmp_path, "a.csv", "gene,n\nTP53,1\n")
        write(tmp_path, "b.xlsx", b"not really")
        listing = list_data_files(tmp_path)

        assert [file.name for file in listing.files] == ["a.csv", "b.xlsx"]
        assert listing.files[0].size == len("gene,n\nTP53,1\n")

    def test_a_file_this_cannot_read_is_listed_and_marked(self, tmp_path) -> None:
        write(tmp_path, "notes.md", "hello")
        listing = list_data_files(tmp_path)

        assert listing.files[0].readable is False
        assert "open it in code" in listing.report()

    def test_files_in_subdirectories_are_named_by_their_relative_path(self, tmp_path) -> None:
        write(tmp_path, "data/inner.csv", "gene,n\nTP53,1\n")
        listing = list_data_files(tmp_path)

        assert listing.files[0].name == str(Path("data/inner.csv"))

    def test_hidden_files_are_left_out(self, tmp_path) -> None:
        write(tmp_path, ".hidden.csv", "gene,n\nTP53,1\n")

        assert list_data_files(tmp_path).files == ()

    def test_what_is_inside_a_hidden_directory_is_left_out_too(self, tmp_path) -> None:
        # Checking the filename alone lists .git and .venv, which are not the project's data
        write(tmp_path, ".git/config", "[core]\n")
        write(tmp_path, ".venv/lib/packages.csv", "name,version\nrequests,2\n")
        write(tmp_path, "results/deep/.cache/notes.csv", "a,b\n1,2\n")
        write(tmp_path, "results/real.csv", "gene,n\nTP53,1\n")

        assert [found.name for found in list_data_files(tmp_path).files] == [
            str(Path("results/real.csv"))
        ]

    def test_a_link_pointing_out_of_the_directory_is_not_listed(self, tmp_path) -> None:
        outside = tmp_path.parent / "outside.csv"
        outside.write_text("secret,value\na,1\n")
        work = tmp_path / "work"
        work.mkdir()
        (work / "link.csv").symlink_to(outside)

        assert list_data_files(work).files == ()

    def test_an_empty_directory_says_what_would_put_something_in_it(self, tmp_path) -> None:
        assert "no data files here yet" in list_data_files(tmp_path).report()

    def test_a_directory_that_is_not_there_is_not_an_error(self, tmp_path) -> None:
        listing = list_data_files(tmp_path / "nothing")

        assert listing.files == ()
        assert listing.complete is True

    def test_a_crowded_directory_is_cut_short_and_says_so(self, tmp_path) -> None:
        for index in range(MAX_DATA_FILES_LISTED + 5):
            write(tmp_path, f"file{index:03d}.csv", "gene,n\nTP53,1\n")

        listing = list_data_files(tmp_path)

        assert len(listing.files) == MAX_DATA_FILES_LISTED
        assert listing.complete is False
        assert "the rest are not listed" in listing.report()


class TestReadingASpreadsheet:
    """The format a collaborator's data arrives in, read without a dependency."""

    def test_a_sheet_of_inline_strings_is_read(self, tmp_path) -> None:
        body = sheet(
            row(1, inline("A1", "gene") + inline("B1", "logFC"))
            + row(2, inline("A2", "TP53") + number("B2", "1.23"))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.sheet == "Results"
        assert [column.name for column in table.columns] == ["gene", "logFC"]
        assert table.columns[1].kind == "number"

    def test_a_sheet_of_shared_strings_is_read(self, tmp_path) -> None:
        # What Excel itself writes, where the cell holds an index rather than the text
        body = sheet(
            row(1, shared("A1", 0) + shared("B1", 1)) + row(2, shared("A2", 2) + number("B2", "9"))
        )
        write(
            tmp_path,
            "book.xlsx",
            workbook({"Results": body}, strings=["gene", "logFC", "TP53"]),
        )
        table = describe_table(tmp_path, "book.xlsx")

        assert [column.name for column in table.columns] == ["gene", "logFC"]
        assert table.sample[0][0] == "TP53"

    def test_an_index_past_the_end_of_the_table_does_not_raise(self, tmp_path) -> None:
        body = sheet(row(1, shared("A1", 0)) + row(2, shared("A2", 99)))
        write(tmp_path, "book.xlsx", workbook({"Results": body}, strings=["gene"]))

        assert describe_table(tmp_path, "book.xlsx").columns[0].name == "gene"

    def test_text_split_into_runs_is_joined_rather_than_lost(self, tmp_path) -> None:
        # Bold part-way through a cell splits it into runs, and the <t> elements are inside the
        # runs rather than beside them, so asking for the first one below <si> finds nothing
        strings = (
            f'<sst xmlns="{SHEET_MAIN}">'
            "<si><r><t>IC</t></r><r><t>50</t></r></si>"
            "<si><t>TP53</t></si></sst>"
        )
        body = sheet(row(1, shared("A1", 0)) + row(2, shared("A2", 1)))
        raw = workbook({"Results": body}, strings=[])
        buffer = io.BytesIO()

        with zipfile.ZipFile(io.BytesIO(raw)) as source:
            with zipfile.ZipFile(buffer, "w") as target:
                for item in source.infolist():
                    content = (
                        strings
                        if item.filename == "xl/sharedStrings.xml"
                        else source.read(item.filename)
                    )
                    target.writestr(item.filename, content)

        write(tmp_path, "book.xlsx", buffer.getvalue())

        assert describe_table(tmp_path, "book.xlsx").columns[0].name == "IC50"

    def test_the_named_sheet_is_read_and_the_others_are_named(self, tmp_path) -> None:
        first = sheet(row(1, inline("A1", "a")) + row(2, number("A2", "1")))
        second = sheet(row(1, inline("A1", "b")) + row(2, number("A2", "2")))
        write(tmp_path, "book.xlsx", workbook({"First": first, "Second": second}))
        table = describe_table(tmp_path, "book.xlsx", sheet="Second")

        assert table.sheet == "Second"
        assert table.columns[0].name == "b"
        assert table.other_sheets == ("First",)
        assert "Other sheets in this workbook: First." in table.report()

    def test_the_first_sheet_is_the_workbook_order_not_the_archive_order(
        self, tmp_path
    ) -> None:
        first = sheet(row(1, inline("A1", "a")) + row(2, number("A2", "1")))
        second = sheet(row(1, inline("A1", "b")) + row(2, number("A2", "2")))

        table = describe_table_bytes(tmp_path, workbook({"First": first, "Second": second}))

        assert table.sheet == "First"

    def test_a_sheet_that_is_not_there_lists_the_ones_that_are(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "a")) + row(2, number("A2", "1")))
        write(tmp_path, "book.xlsx", workbook({"Only": body}))

        with pytest.raises(TableError, match="has no sheet called"):
            describe_table(tmp_path, "book.xlsx", sheet="Missing")

    def test_an_empty_sheet_is_refused_rather_than_described(self, tmp_path) -> None:
        write(tmp_path, "book.xlsx", workbook({"Blank": sheet("")}))

        with pytest.raises(TableError, match="nothing in it"):
            describe_table(tmp_path, "book.xlsx")

    @pytest.mark.parametrize(
        "kind, raw, expected",
        [
            ('t="b"', "1", "TRUE"),
            ('t="b"', "0", "FALSE"),
            ('t="e"', "#DIV/0!", "#DIV/0!"),
            ('t="str"', "computed", "computed"),
        ],
    )
    def test_the_other_ways_a_cell_holds_a_value(self, tmp_path, kind, raw, expected) -> None:
        body = sheet(
            row(1, inline("A1", "value")) + row(2, f'<c r="A2" {kind}><v>{raw}</v></c>')
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}))

        assert describe_table(tmp_path, "book.xlsx").sample[0][0] == expected

    def test_a_spreadsheet_error_is_reported_as_a_value_that_never_existed(
        self, tmp_path
    ) -> None:
        body = sheet(
            row(1, inline("A1", "ratio"))
            + row(2, '<c r="A2" t="e"><v>#DIV/0!</v></c>')
            + row(3, number("A3", "1.5"))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert any("never a value" in note for note in table.columns[0].notes)

    def test_a_formula_with_no_cached_result_is_missing_rather_than_zero(
        self, tmp_path
    ) -> None:
        body = sheet(
            row(1, inline("A1", "total"))
            + row(2, '<c r="A2"><f>1+1</f><v /></c>')
            + row(3, number("A3", "5"))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.columns[0].missing == 1
        assert table.columns[0].filled == 1

    def test_a_cell_claiming_to_be_a_number_and_holding_text_is_text(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "value")) + row(2, number("A2", "not a number")))
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.columns[0].kind == "text"
        assert table.columns[0].smallest is None


def describe_table_bytes(directory: Path, content: bytes):
    """Writes a workbook and describes it, for a test that only cares about the result."""
    write(directory, "ordered.xlsx", content)
    return describe_table(directory, "ordered.xlsx")


class TestCellsThatAreNotWhereTheyLook:
    """A row holds only its non-empty cells, so position is not order."""

    def test_a_blank_middle_cell_does_not_shift_the_rest_left(self, tmp_path) -> None:
        body = sheet(
            row(1, inline("A1", "gene") + inline("B1", "pval") + inline("C1", "n"))
            # No B2 at all, which is how a spreadsheet writes a blank cell
            + row(2, inline("A2", "TP53") + number("C2", "12"))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.sample[0] == ("TP53", "", "12")
        assert table.columns[1].missing == 1
        assert table.columns[2].examples == ("12",)

    def test_a_row_starting_part_way_across_keeps_its_columns(self, tmp_path) -> None:
        body = sheet(
            row(1, inline("A1", "a") + inline("B1", "b") + inline("C1", "c"))
            + row(2, number("C2", "3"))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}))

        assert describe_table(tmp_path, "book.xlsx").sample[0] == ("", "", "3")

    @pytest.mark.parametrize(
        "reference, expected",
        [("A1", 1), ("B2", 2), ("Z9", 26), ("AA1", 27), ("AB1", 28), ("BA1", 53), ("ZZ1", 702)],
    )
    def test_the_column_letters_count_in_twenty_sixes(self, reference, expected) -> None:
        assert column_number(reference) == expected

    def test_a_reference_with_no_letters_is_not_a_column(self) -> None:
        assert column_number("123") == 0
        assert column_number("") == 0

    def test_a_letter_outside_the_alphabet_is_not_a_column(self) -> None:
        # str.isalpha takes both. "é" came out as column 137, and "ß" upper-cases to "SS",
        # which made ord() raise a TypeError out of a function that only raises TableError
        assert column_number("\u00e91") == 0
        assert column_number("\u00df1") == 0

    def test_the_last_column_a_spreadsheet_can_have_is_xfd(self) -> None:
        assert column_number("XFD1") == MAX_SPREADSHEET_COLUMNS

        with pytest.raises(TableError, match="past the last column"):
            column_number("XFE1")

    def test_a_reference_of_any_length_is_refused_without_being_counted_out(self) -> None:
        with pytest.raises(TableError, match="past the last column"):
            column_number("A" * 1_000_000 + "1")

    def test_a_cell_past_the_last_column_is_refused_as_damage(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "gene") + inline("ZZZZ1", "far")))
        write(tmp_path, "book.xlsx", workbook({"Results": body}))

        with pytest.raises(TableError, match="past the last column"):
            describe_table(tmp_path, "book.xlsx")

    def test_a_stray_value_far_to_the_right_stops_the_read_and_says_why(
        self, tmp_path, monkeypatch
    ) -> None:
        # Every row is padded to the widest, so one cell in column AD makes each row thirty
        # cells, and it is the padded total that the limit is on
        monkeypatch.setattr("virtual_lab.tables.MAX_TABLE_CELLS", 100)
        cells = "".join(row(index, inline(f"A{index}", f"g{index}")) for index in range(1, 5))
        body = sheet(cells + row(5, inline("A5", "g5") + inline("AD5", "stray")))
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.rows == 3
        assert table.complete is False
        assert any("30 columns wide" in warning for warning in table.warnings)
        assert any("first 4 rows" in warning for warning in table.warnings)
        assert "longer than" not in table.report()

    def test_the_cell_limit_counts_the_widest_row_so_far_not_the_one_in_hand(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("virtual_lab.tables.MAX_TABLE_CELLS", 100)
        header = row(1, inline("A1", "gene") + inline("AD1", "note"))
        cells = "".join(row(index, inline(f"A{index}", f"g{index}")) for index in range(2, 10))
        write(tmp_path, "book.xlsx", workbook({"Results": sheet(header + cells)}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.rows == 2
        assert table.complete is False

    def test_a_far_right_column_is_placed_rather_than_counted(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "first") + inline("AA1", "far")))
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.total_columns == 27
        assert table.columns[26].name == "far"


class TestTheDateAGeneSymbolBecame:
    """A date is a number plus a style, and a gene symbol that met Excel is a date."""

    def test_a_styled_number_is_read_as_the_date_it_is(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "when")) + row(2, number("A2", "45902", style=1)))
        write(
            tmp_path,
            "book.xlsx",
            # Style 0 is General, style 1 is the built-in d-mmm-yy, which no <numFmt> defines
            workbook({"Results": body}, formats=["0", "15"]),
        )
        table = describe_table(tmp_path, "book.xlsx")

        assert table.sample[0][0] == "2025-09-02"
        assert table.columns[0].kind == "date"

    def test_the_same_number_without_the_style_stays_a_number(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "count")) + row(2, number("A2", "45902", style=0)))
        write(tmp_path, "book.xlsx", workbook({"Results": body}, formats=["0", "15"]))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.sample[0][0] == "45902"
        assert table.columns[0].kind == "number"

    def test_a_format_the_file_defines_itself_is_recognised(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "when")) + row(2, number("A2", "45902", style=1)))
        write(
            tmp_path,
            "book.xlsx",
            workbook(
                {"Results": body},
                formats=["0", "164"],
                custom='<numFmt numFmtId="164" formatCode="yyyy-mm-dd h:mm:ss" />',
            ),
        )

        assert describe_table(tmp_path, "book.xlsx").sample[0][0] == "2025-09-02"

    def test_a_unit_in_a_format_does_not_make_a_speed_into_a_date(self, tmp_path) -> None:
        # The letters of a date are inside a quoted literal here, and reading them would put
        # every measurement in this column in 1900
        body = sheet(row(1, inline("A1", "speed")) + row(2, number("A2", "45902", style=1)))
        write(
            tmp_path,
            "book.xlsx",
            workbook(
                {"Results": body},
                formats=["0", "164"],
                custom='<numFmt numFmtId="164" formatCode="0.00&quot;m/d&quot;" />',
            ),
        )
        table = describe_table(tmp_path, "book.xlsx")

        assert table.columns[0].kind == "number"
        assert table.sample[0][0] == "45902"

    def test_a_column_of_symbols_and_dates_is_named_as_the_corruption_it_is(
        self, tmp_path
    ) -> None:
        body = sheet(
            row(1, inline("A1", "gene"))
            + row(2, inline("A2", "TP53"))
            + row(3, number("A3", "45902", style=1))
            + row(4, inline("A4", "BRCA1"))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}, formats=["0", "15"]))
        table = describe_table(tmp_path, "book.xlsx")

        note = next(note for note in table.columns[0].notes if "Dates among text" in note)

        assert "1 date and 2 text values" in note
        assert "SEPT2" in note
        assert "cannot be recovered" in note

    def test_a_column_that_is_all_dates_is_not_called_corrupted(self, tmp_path) -> None:
        body = sheet(
            row(1, inline("A1", "collected"))
            + row(2, number("A2", "45902", style=1))
            + row(3, number("A3", "45903", style=1))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}, formats=["0", "15"]))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.columns[0].kind == "date"
        assert not any("Dates among text" in note for note in table.columns[0].notes)


class TestRefusingAnUnsafeWorkbook:
    """A spreadsheet is a zip archive full of XML, and both are somebody else's bytes."""

    def test_a_file_that_is_not_a_zip_says_what_to_do_about_it(self, tmp_path) -> None:
        write(tmp_path, "old.xlsx", b"\xd0\xcf\x11\xe0not a zip at all")

        with pytest.raises(TableError, match="not a readable spreadsheet"):
            describe_table(tmp_path, "old.xlsx")

    def test_a_declared_document_type_is_refused_before_parsing(self, tmp_path) -> None:
        body = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE worksheet [<!ENTITY a "aaaaaaaaaa">]>'
            f'<worksheet xmlns="{SHEET_MAIN}"><sheetData>'
            f'{row(1, inline("A1", "&a;"))}</sheetData></worksheet>'
        )
        write(tmp_path, "bomb.xlsx", workbook({"Results": body}))

        with pytest.raises(TableError, match="declares a document type"):
            describe_table(tmp_path, "bomb.xlsx")

    def test_a_member_that_unpacks_to_more_than_the_cap_is_refused(self, tmp_path) -> None:
        # A small archive declaring an enormous member, which compresses to almost nothing
        body = f'<worksheet xmlns="{SHEET_MAIN}"><sheetData>{"<!-- x -->" * 10}'
        body += " " * (MAX_SHEET_BYTES + 1000)
        body += "</sheetData></worksheet>"
        raw = workbook({"Results": body})
        write(tmp_path, "bomb.xlsx", raw)

        assert len(raw) < MAX_SHEET_BYTES

        with pytest.raises(TableError, match="unpacks to more than"):
            describe_table(tmp_path, "bomb.xlsx")

    def test_the_cap_is_on_the_bytes_read_rather_than_checked_afterwards(
        self, tmp_path, monkeypatch
    ) -> None:
        # Refusing the member after unpacking it is not a limit on anything: the memory has
        # already been spent by the time the size is known. Nothing observable separates the two
        # except how many bytes were asked for, so that is what this looks at.
        asked = []
        opener = zipfile.ZipFile.open

        class Counted:
            def __init__(self, stream) -> None:
                self.stream = stream

            def read(self, size=None):
                asked.append(size)
                return self.stream.read(size) if size is not None else self.stream.read()

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *details):
                return self.stream.__exit__(*details)

        body = sheet(row(1, inline("A1", "gene")) + row(2, inline("A2", "TP53")))
        write(tmp_path, "book.xlsx", workbook({"Results": body}))

        def counted(self, name, mode="r", *rest, **named):
            stream = opener(self, name, mode, *rest, **named)
            return Counted(stream) if mode == "r" else stream

        monkeypatch.setattr(zipfile.ZipFile, "open", counted)
        describe_table(tmp_path, "book.xlsx")

        assert asked
        assert all(size is not None and size <= MAX_SHEET_BYTES + 1 for size in asked)

    def test_an_archive_of_too_many_parts_is_refused(self, tmp_path) -> None:
        body = sheet(row(1, inline("A1", "a")) + row(2, number("A2", "1")))
        write(
            tmp_path,
            "many.xlsx",
            workbook({"Results": body}, padding=MAX_ARCHIVE_ENTRIES + 1),
        )

        with pytest.raises(TableError, match="more parts than a spreadsheet should"):
            describe_table(tmp_path, "many.xlsx")

    def test_a_workbook_whose_sheets_point_nowhere_is_refused(self, tmp_path) -> None:
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "xl/workbook.xml",
                f'<workbook xmlns="{SHEET_MAIN}" xmlns:r="{DOCUMENT}">'
                f'<sheets><sheet name="Gone" sheetId="1" r:id="rId9" /></sheets></workbook>',
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{PACKAGE}" />'
            )

        write(tmp_path, "broken.xlsx", buffer.getvalue())

        with pytest.raises(TableError, match="no sheets"):
            describe_table(tmp_path, "broken.xlsx")

    def test_a_workbook_missing_its_parts_says_which(self, tmp_path) -> None:
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("docProps/core.xml", "<a />")

        write(tmp_path, "hollow.xlsx", buffer.getvalue())

        with pytest.raises(TableError, match="missing xl/workbook.xml"):
            describe_table(tmp_path, "hollow.xlsx")


    def test_a_chart_sheet_is_not_taken_for_a_sheet_of_data(self, tmp_path) -> None:
        # A chart sheet sits in the workbook's list of sheets like any other, and Excel puts it
        # first when a chart is moved to its own tab. Only its relationship's type says it holds
        # no cells; read as a worksheet, it would become the default sheet and have no rows.
        body = sheet(row(1, inline("A1", "gene")) + row(2, inline("A2", "TP53")))
        raw = workbook({"Results": body})
        raw = rebuild(raw, "xl/workbook.xml", lambda text: text.replace(
            "<sheets>", '<sheets><sheet name="Chart1" sheetId="9" r:id="rId9" />'
        ))
        raw = rebuild(raw, "xl/_rels/workbook.xml.rels", lambda text: text.replace(
            "</Relationships>",
            f'<Relationship Id="rId9" Target="chartsheets/sheet1.xml" '
            f'Type="{DOCUMENT}/chartsheet" />'
            f'<Relationship Id="rId8" Target="styles.xml" Type="{DOCUMENT}/styles" />'
            "</Relationships>",
        ))
        write(tmp_path, "book.xlsx", raw)
        table = describe_table(tmp_path, "book.xlsx")

        assert table.sheet == "Results"
        assert table.other_sheets == ()
        assert table.columns[0].name == "gene"

    def test_cells_without_a_reference_are_placed_after_the_one_before(self, tmp_path) -> None:
        # The r attribute on a cell is optional in the format, and writers that stream cells
        # out in order leave it off. Dropping those cells reads a full sheet as an empty one.
        body = sheet(
            '<row><c t="inlineStr"><is><t>gene</t></is></c>'
            '<c t="inlineStr"><is><t>logFC</t></is></c></row>'
            '<row><c t="inlineStr"><is><t>TP53</t></is></c><c><v>1.5</v></c></row>'
            # A referenced cell moves the position on, and the next unreferenced one follows it
            f'<row>{inline("A3", "BRCA1")}<c><v>2.5</v></c></row>'
        )
        write(tmp_path, "streamed.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "streamed.xlsx")

        assert [column.name for column in table.columns] == ["gene", "logFC"]
        assert table.rows == 2
        assert table.columns[0].examples == ("TP53", "BRCA1")
        assert table.columns[1].examples == ("1.5", "2.5")

    def test_a_row_with_nothing_in_it_does_not_become_a_row(self, tmp_path) -> None:
        # A spreadsheet writes <row/> for a row that has been emptied but not deleted
        body = sheet(
            row(1, inline("A1", "gene"))
            + row(2, inline("A2", "TP53"))
            + '<row r="3" />'
            + row(4, inline("A4", "BRCA1"))
        )
        write(tmp_path, "book.xlsx", workbook({"Results": body}))
        table = describe_table(tmp_path, "book.xlsx")

        assert table.rows == 2
        assert table.columns[0].examples == ("TP53", "BRCA1")

    def test_a_member_that_will_not_unpack_is_reported_as_a_table_problem(self, tmp_path) -> None:
        # zipfile.BadZipFile derives straight from Exception, so catching OSError misses the
        # commonest way a workbook fails: a member whose bytes no longer match their checksum
        body = sheet(row(1, inline("A1", "gene")))
        raw = bytearray(workbook({"Results": body}, compression=zipfile.ZIP_STORED))
        at = raw.find(b"<worksheet")
        raw[at + 1] = ord("X")
        write(tmp_path, "book.xlsx", bytes(raw))

        with pytest.raises(TableError, match="could not be read"):
            describe_table(tmp_path, "book.xlsx")


class TestWhatAReportCosts:
    """Every part of the report is bounded, because a data file has no size limit."""

    def test_a_wide_file_describes_only_as_many_columns_as_are_worth_it(
        self, tmp_path
    ) -> None:
        width = MAX_COLUMNS_REPORTED + 20
        header = ",".join(f"col{index}" for index in range(width))
        values = ",".join("1" for _ in range(width))
        write(tmp_path, "wide.csv", f"{header}\n{values}\n")
        table = describe_table(tmp_path, "wide.csv")

        assert table.total_columns == width
        assert len(table.columns) == MAX_COLUMNS_REPORTED
        assert f"{width - MAX_COLUMNS_REPORTED:,} more not described" in table.report()

    def test_columns_past_those_described_are_never_scanned(self, tmp_path, monkeypatch) -> None:
        # A scan per column costs a pass over every row each, so a wide header over short rows
        # costs the product of the two; only the described columns are worth it
        import virtual_lab.tables as tables

        made = []
        original = tables.ColumnScan.__init__

        def counted(self, *args) -> None:
            made.append(args)
            original(self, *args)

        monkeypatch.setattr(tables.ColumnScan, "__init__", counted)
        width = MAX_COLUMNS_REPORTED + 30
        header = ",".join(f"c{index}" for index in range(width))
        write(tmp_path, "wide.csv", f"{header}\n1,2\n3,4\n")
        table = describe_table(tmp_path, "wide.csv")

        assert len(made) == MAX_COLUMNS_REPORTED
        assert table.total_columns == width
        assert table.columns[0].missing == 0
        assert table.columns[-1].missing == 2

    def test_a_long_file_stops_and_says_where(self, tmp_path) -> None:
        rows = "\n".join(f"g{index},{index}" for index in range(MAX_TABLE_ROWS_SCANNED + 50))
        write(tmp_path, "long.csv", f"gene,n\n{rows}\n")
        table = describe_table(tmp_path, "long.csv")

        assert table.rows == MAX_TABLE_ROWS_SCANNED
        assert table.complete is False
        assert "Everything below describes the rows that were read" in table.report()
        assert any(f"longer than {MAX_TABLE_ROWS_SCANNED:,} rows" in w for w in table.warnings)

    def test_a_file_of_exactly_the_row_limit_is_read_to_the_end(self, tmp_path) -> None:
        rows = "\n".join(f"g{index},{index}" for index in range(MAX_TABLE_ROWS_SCANNED))
        write(tmp_path, "full.csv", f"gene,n\n{rows}\n")
        table = describe_table(tmp_path, "full.csv")

        assert table.rows == MAX_TABLE_ROWS_SCANNED
        assert table.complete is True
        assert not any("longer than" in warning for warning in table.warnings)

    def test_a_short_file_is_not_said_to_be_cut_short(self, tmp_path) -> None:
        write(tmp_path, "short.csv", "gene,n\nTP53,1\n")
        table = describe_table(tmp_path, "short.csv")

        assert table.complete is True
        assert "Everything below" not in table.report()

    def test_a_long_sheet_stops_too(self, tmp_path) -> None:
        rows = "".join(
            row(index + 1, inline(f"A{index + 1}", f"g{index}"))
            for index in range(MAX_TABLE_ROWS_SCANNED + 50)
        )
        write(tmp_path, "long.xlsx", workbook({"Results": sheet(rows)}))
        table = describe_table(tmp_path, "long.xlsx")

        assert table.rows == MAX_TABLE_ROWS_SCANNED
        assert table.complete is False
        assert any(f"longer than {MAX_TABLE_ROWS_SCANNED:,} rows" in w for w in table.warnings)

    def test_a_paragraph_in_a_cell_is_cut(self, tmp_path) -> None:
        write(tmp_path, "notes.csv", f"gene,note\nTP53,{'word ' * 400}\n")
        table = describe_table(tmp_path, "notes.csv")

        assert len(table.columns[1].examples[0]) < MAX_CELL_CHARACTERS + 60
        assert "characters not shown" in table.columns[1].examples[0]
        assert len(table.sample[0][1]) < MAX_CELL_CHARACTERS + 60

    def test_only_a_few_examples_are_given_per_column(self, tmp_path) -> None:
        rows = "\n".join(f"g{index}" for index in range(50))
        write(tmp_path, "many.csv", f"gene\n{rows}\n")
        table = describe_table(tmp_path, "many.csv")

        assert len(table.columns[0].examples) == MAX_COLUMN_EXAMPLES

    def test_a_file_cut_at_the_byte_limit_is_not_called_the_wrong_encoding(
        self, tmp_path, monkeypatch
    ) -> None:
        # Cutting to the byte splits a multi-byte character in half, which is not valid UTF-8,
        # and the file is then reported as cp1252 on the strength of a character never in it
        content = "gene,conc\nTP53,5 \u00b5M\nBRCA1,10 \u00b5M\nEGFR,2 \u00b5M\n"
        raw = content.encode("utf-8")

        # Between the two bytes of the last micro sign, so the limit is only survivable by
        # backing up to a line ending first
        limit = raw.rfind("\u00b5".encode("utf-8")) + 1

        with pytest.raises(UnicodeDecodeError):
            raw[:limit].decode("utf-8")

        monkeypatch.setattr("virtual_lab.tables.MAX_TABLE_BYTES", limit)
        write(tmp_path, "wide.csv", content)
        table = describe_table(tmp_path, "wide.csv")

        assert "cp1252" not in table.reading
        assert not any("cp1252" in warning for warning in table.warnings)
        assert any("larger than" in warning for warning in table.warnings)
        # The read stopped at the byte limit a few rows in, and the report used to put that
        # down to the file being longer than the row limit
        assert "longer than" not in table.report()

    def test_a_wide_sample_row_says_it_was_cut(self, tmp_path) -> None:
        width = MAX_SAMPLE_COLUMNS + 3
        header = ",".join(f"c{index}" for index in range(width))
        values = ",".join(str(index) for index in range(width))
        write(tmp_path, "wide.csv", f"{header}\n{values}\n")
        table = describe_table(tmp_path, "wide.csv")

        assert len(table.sample[0]) == MAX_SAMPLE_COLUMNS + 1
        assert table.sample[0][:2] == ("0", "1")
        assert table.sample[0][-1] == "... 3 more"

    def test_a_narrow_sample_row_is_not_given_a_marker(self, tmp_path) -> None:
        write(tmp_path, "narrow.csv", "a,b\n1,2\n")

        assert describe_table(tmp_path, "narrow.csv").sample[0] == ("1", "2")

    def test_a_column_with_too_many_different_values_says_the_count_is_a_floor(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("virtual_lab.tables.MAX_DISTINCT_TRACKED", 3)
        write(tmp_path, "ids.csv", "id\na\nb\nc\nd\ne\n")
        column = describe_table(tmp_path, "ids.csv").columns[0]

        assert column.distinct == 3
        assert column.counted_all is False
        assert "over 3" in column.describe()

    def test_a_column_inside_the_limit_says_the_count_is_exact(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("virtual_lab.tables.MAX_DISTINCT_TRACKED", 3)
        write(tmp_path, "ids.csv", "id\na\nb\na\n")
        column = describe_table(tmp_path, "ids.csv").columns[0]

        assert column.distinct == 2
        assert column.counted_all is True
        assert "over" not in column.describe()

    def test_a_value_seen_before_the_limit_does_not_count_as_going_over_it(
        self, tmp_path, monkeypatch
    ) -> None:
        # The limit is on how many values are remembered, so a column that fills it and then
        # repeats only what it has already shown still has an exact count
        monkeypatch.setattr("virtual_lab.tables.MAX_DISTINCT_TRACKED", 3)
        write(tmp_path, "ids.csv", "id\na\nb\nc\na\nc\n")
        column = describe_table(tmp_path, "ids.csv").columns[0]

        assert column.distinct == 3
        assert column.counted_all is True

    def test_a_file_of_nothing_but_separators_says_there_is_nothing_in_it(
        self, tmp_path
    ) -> None:
        write(tmp_path, "blank.csv", ",,\n,,\n")

        with pytest.raises(TableError, match="no rows with anything in them"):
            describe_table(tmp_path, "blank.csv")

    def test_a_file_too_long_with_no_line_ending_in_it_says_so(
        self, tmp_path, monkeypatch
    ) -> None:
        # Cutting back to a line ending when there is none leaves nothing, and the file would
        # otherwise be reported as a preamble of comments with no table under it
        monkeypatch.setattr("virtual_lab.tables.MAX_TABLE_BYTES", 40)
        write(tmp_path, "one.csv", "a," * 200)

        with pytest.raises(TableError, match="no line ending"):
            describe_table(tmp_path, "one.csv")

    def test_only_a_few_whole_rows_are_shown(self, tmp_path) -> None:
        rows = "\n".join(f"g{index},{index}" for index in range(50))
        write(tmp_path, "many.csv", f"gene,n\n{rows}\n")
        table = describe_table(tmp_path, "many.csv")

        assert len(table.sample) == MAX_SAMPLE_ROWS

    def test_a_field_too_long_for_the_parser_is_reported_rather_than_raised(
        self, tmp_path
    ) -> None:
        # csv's field size limit is a module-wide setting, so it is not raised here; a file with
        # a 200 KB cell in it needs handling rather than a wider limit
        write(tmp_path, "huge.csv", f"gene,blob\nTP53,{'x' * 200_000}\n")

        with pytest.raises(TableError, match="could not be parsed"):
            describe_table(tmp_path, "huge.csv")


class TestWhatTheReportSays:
    """The report is the whole output, so its shape is worth pinning."""

    def test_the_first_line_carries_the_shape_and_how_it_was_read(self, tmp_path) -> None:
        write(tmp_path, "assay.csv", "gene,logFC\nTP53,1.2\nBRCA1,-2.0\n")
        report = describe_table(tmp_path, "assay.csv").report()

        assert report.startswith("assay.csv: 2 rows, 2 columns. comma separated, utf-8.")

    def test_one_of_anything_is_not_reported_as_one_things(self, tmp_path) -> None:
        write(tmp_path, "one.csv", "gene\nTP53\n")
        report = describe_table(tmp_path, "one.csv").report()

        assert "1 row, 1 column." in report
        assert "1 value," in report

    def test_a_column_with_gaps_in_it_says_how_many(self, tmp_path) -> None:
        write(tmp_path, "assay.csv", "gene,logFC\nTP53,1.2\nBRCA1,\nEGFR,\n")
        report = describe_table(tmp_path, "assay.csv").report()

        assert "2 missing" in report

    def test_a_column_with_no_gaps_in_it_says_nothing_about_missing(self, tmp_path) -> None:
        write(tmp_path, "assay.csv", "gene,logFC\nTP53,1.2\n")

        assert "missing" not in describe_table(tmp_path, "assay.csv").report()

    def test_the_rows_are_shown_under_the_columns(self, tmp_path) -> None:
        write(tmp_path, "assay.csv", "gene,logFC\nTP53,1.2\n")
        report = describe_table(tmp_path, "assay.csv").report()

        assert "First rows:" in report
        assert "TP53 | 1.2" in report

    def test_a_clean_file_is_not_given_warnings_it_does_not_need(self, tmp_path) -> None:
        write(tmp_path, "clean.csv", "gene,logFC\nTP53,1.2\nBRCA1,-2.0\n")
        table = describe_table(tmp_path, "clean.csv")

        assert table.warnings == ()
        assert "Before using this file" not in table.report()
        assert all(column.notes == () for column in table.columns)

    def test_every_warning_reaches_the_report(self, tmp_path) -> None:
        write(tmp_path, "messy.csv", "gene ,value,value\nTP53,00123,<0.1\n".encode("utf-8-sig"))
        table = describe_table(tmp_path, "messy.csv")
        report = table.report()

        assert table.warnings
        for warning in table.warnings:
            assert warning in report

        for column in table.columns:
            for note in column.notes:
                assert note in report

    def test_the_numeric_range_is_shown_and_only_for_numbers(self, tmp_path) -> None:
        write(tmp_path, "assay.csv", "gene,logFC\nTP53,1.2\nBRCA1,-2.0\n")
        report = describe_table(tmp_path, "assay.csv").report()

        assert "-2 to 1.2" in report
        assert "[text]" in report
