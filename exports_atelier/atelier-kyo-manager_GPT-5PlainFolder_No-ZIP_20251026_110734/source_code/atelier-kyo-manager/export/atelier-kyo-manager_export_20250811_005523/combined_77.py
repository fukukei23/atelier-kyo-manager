
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\parsers\python_parser.py ===
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
from collections.abc import (
    Hashable,
    Iterator,
    Mapping,
    Sequence,
)
import csv
from io import StringIO
import re
from typing import (
    IO,
    TYPE_CHECKING,
    DefaultDict,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_numeric_dtype,
)
from pandas.core.dtypes.inference import is_dict_like

from pandas.io.common import (
    dedup_names,
    is_potential_multi_index,
)
from pandas.io.parsers.base_parser import (
    ParserBase,
    parser_defaults,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        ReadCsvBuffer,
        Scalar,
    )

    from pandas import (
        Index,
        MultiIndex,
    )

# BOM character (byte order mark)
# This exists at the beginning of a file to indicate endianness
# of a file (stream). Unfortunately, this marker screws up parsing,
# so we need to remove it if we see it.
_BOM = "\ufeff"


class PythonParser(ParserBase):
    _no_thousands_columns: set[int]

    def __init__(self, f: ReadCsvBuffer[str] | list, **kwds) -> None:
        """
        Workhorse function for processing nested list into DataFrame
        """
        super().__init__(kwds)

        self.data: Iterator[str] | None = None
        self.buf: list = []
        self.pos = 0
        self.line_pos = 0

        self.skiprows = kwds["skiprows"]

        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            self.skipfunc = lambda x: x in self.skiprows

        self.skipfooter = _validate_skipfooter_arg(kwds["skipfooter"])
        self.delimiter = kwds["delimiter"]

        self.quotechar = kwds["quotechar"]
        if isinstance(self.quotechar, str):
            self.quotechar = str(self.quotechar)

        self.escapechar = kwds["escapechar"]
        self.doublequote = kwds["doublequote"]
        self.skipinitialspace = kwds["skipinitialspace"]
        self.lineterminator = kwds["lineterminator"]
        self.quoting = kwds["quoting"]
        self.skip_blank_lines = kwds["skip_blank_lines"]

        self.has_index_names = False
        if "has_index_names" in kwds:
            self.has_index_names = kwds["has_index_names"]

        self.verbose = kwds["verbose"]

        self.thousands = kwds["thousands"]
        self.decimal = kwds["decimal"]

        self.comment = kwds["comment"]

        # Set self.data to something that can read lines.
        if isinstance(f, list):
            # read_excel: f is a list
            self.data = cast(Iterator[str], f)
        else:
            assert hasattr(f, "readline")
            self.data = self._make_reader(f)

        # Get columns in two steps: infer from data, then
        # infer column indices from self.usecols if it is specified.
        self._col_indices: list[int] | None = None
        columns: list[list[Scalar | None]]
        (
            columns,
            self.num_original_columns,
            self.unnamed_cols,
        ) = self._infer_columns()

        # Now self.columns has the set of columns that we will process.
        # The original set is stored in self.original_columns.
        # error: Cannot determine type of 'index_names'
        (
            self.columns,
            self.index_names,
            self.col_names,
            _,
        ) = self._extract_multi_indexer_columns(
            columns,
            self.index_names,  # type: ignore[has-type]
        )

        # get popped off for index
        self.orig_names: list[Hashable] = list(self.columns)

        # needs to be cleaned/refactored
        # multiple date column thing turning into a real spaghetti factory

        if not self._has_complex_date_col:
            (index_names, self.orig_names, self.columns) = self._get_index_name()
            self._name_processed = True
            if self.index_names is None:
                self.index_names = index_names

        if self._col_indices is None:
            self._col_indices = list(range(len(self.columns)))

        self._parse_date_cols = self._validate_parse_dates_presence(self.columns)
        self._no_thousands_columns = self._set_no_thousand_columns()

        if len(self.decimal) != 1:
            raise ValueError("Only length-1 decimal markers supported")

    @cache_readonly
    def num(self) -> re.Pattern:
        decimal = re.escape(self.decimal)
        if self.thousands is None:
            regex = rf"^[\-\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\-?[0-9]+)?$"
        else:
            thousands = re.escape(self.thousands)
            regex = (
                rf"^[\-\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?"
                rf"([0-9]?(E|e)\-?[0-9]+)?$"
            )
        return re.compile(regex)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]):
        sep = self.delimiter

        if sep is None or len(sep) == 1:
            if self.lineterminator:
                raise ValueError(
                    "Custom line terminators not supported in python parser (yet)"
                )

            class MyDialect(csv.Dialect):
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = self.escapechar
                doublequote = self.doublequote
                skipinitialspace = self.skipinitialspace
                quoting = self.quoting
                lineterminator = "\n"

            dia = MyDialect

            if sep is not None:
                dia.delimiter = sep
            else:
                # attempt to sniff the delimiter from the first valid line,
                # i.e. no comment line and not in skiprows
                line = f.readline()
                lines = self._check_comments([[line]])[0]
                while self.skipfunc(self.pos) or not lines:
                    self.pos += 1
                    line = f.readline()
                    lines = self._check_comments([[line]])[0]
                lines_str = cast(list[str], lines)

                # since `line` was a string, lines will be a list containing
                # only a single string
                line = lines_str[0]

                self.pos += 1
                self.line_pos += 1
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter

                # Note: encoding is irrelevant here
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))

            # Note: encoding is irrelevant here
            reader = csv.reader(f, dialect=dia, strict=True)

        else:

            def _read():
                line = f.readline()
                pat = re.compile(sep)

                yield pat.split(line.strip())

                for line in f:
                    yield pat.split(line.strip())

            reader = _read()

        return reader

    def read(
        self, rows: int | None = None
    ) -> tuple[
        Index | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]
    ]:
        try:
            content = self._get_lines(rows)
        except StopIteration:
            if self._first_chunk:
                content = []
            else:
                self.close()
                raise

        # done with first read, next time raise StopIteration
        self._first_chunk = False

        columns: Sequence[Hashable] = list(self.orig_names)
        if not len(content):  # pragma: no cover
            # DataFrame with the right metadata, even though it's length 0
            # error: Cannot determine type of 'index_col'
            names = dedup_names(
                self.orig_names,
                is_potential_multi_index(
                    self.orig_names,
                    self.index_col,  # type: ignore[has-type]
                ),
            )
            index, columns, col_dict = self._get_empty_meta(
                names,
                self.dtype,
            )
            conv_columns = self._maybe_make_multi_index_columns(columns, self.col_names)
            return index, conv_columns, col_dict

        # handle new style for names in index
        count_empty_content_vals = count_empty_vals(content[0])
        indexnamerow = None
        if self.has_index_names and count_empty_content_vals == len(columns):
            indexnamerow = content[0]
            content = content[1:]

        alldata = self._rows_to_cols(content)
        data, columns = self._exclude_implicit_index(alldata)

        conv_data = self._convert_data(data)
        columns, conv_data = self._do_date_conversions(columns, conv_data)

        index, result_columns = self._make_index(
            conv_data, alldata, columns, indexnamerow
        )

        return index, result_columns, conv_data

    def _exclude_implicit_index(
        self,
        alldata: list[np.ndarray],
    ) -> tuple[Mapping[Hashable, np.ndarray], Sequence[Hashable]]:
        # error: Cannot determine type of 'index_col'
        names = dedup_names(
            self.orig_names,
            is_potential_multi_index(
                self.orig_names,
                self.index_col,  # type: ignore[has-type]
            ),
        )

        offset = 0
        if self._implicit_index:
            # error: Cannot determine type of 'index_col'
            offset = len(self.index_col)  # type: ignore[has-type]

        len_alldata = len(alldata)
        self._check_data_length(names, alldata)

        return {
            name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata
        }, names

    # legacy
    def get_chunk(
        self, size: int | None = None
    ) -> tuple[
        Index | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]
    ]:
        if size is None:
            # error: "PythonParser" has no attribute "chunksize"
            size = self.chunksize  # type: ignore[attr-defined]
        return self.read(rows=size)

    def _convert_data(
        self,
        data: Mapping[Hashable, np.ndarray],
    ) -> Mapping[Hashable, ArrayLike]:
        # apply converters
        clean_conv = self._clean_mapping(self.converters)
        clean_dtypes = self._clean_mapping(self.dtype)

        # Apply NA values.
        clean_na_values = {}
        clean_na_fvalues = {}

        if isinstance(self.na_values, dict):
            for col in self.na_values:
                na_value = self.na_values[col]
                na_fvalue = self.na_fvalues[col]

                if isinstance(col, int) and col not in self.orig_names:
                    col = self.orig_names[col]

                clean_na_values[col] = na_value
                clean_na_fvalues[col] = na_fvalue
        else:
            clean_na_values = self.na_values
            clean_na_fvalues = self.na_fvalues

        return self._convert_to_ndarrays(
            data,
            clean_na_values,
            clean_na_fvalues,
            self.verbose,
            clean_conv,
            clean_dtypes,
        )

    @cache_readonly
    def _have_mi_columns(self) -> bool:
        if self.header is None:
            return False

        header = self.header
        if isinstance(header, (list, tuple, np.ndarray)):
            return len(header) > 1
        else:
            return False

    def _infer_columns(
        self,
    ) -> tuple[list[list[Scalar | None]], int, set[Scalar | None]]:
        names = self.names
        num_original_columns = 0
        clear_buffer = True
        unnamed_cols: set[Scalar | None] = set()

        if self.header is not None:
            header = self.header
            have_mi_columns = self._have_mi_columns

            if isinstance(header, (list, tuple, np.ndarray)):
                # we have a mi columns, so read an extra line
                if have_mi_columns:
                    header = list(header) + [header[-1] + 1]
            else:
                header = [header]

            columns: list[list[Scalar | None]] = []
            for level, hr in enumerate(header):
                try:
                    line = self._buffered_line()

                    while self.line_pos <= hr:
                        line = self._next_line()

                except StopIteration as err:
                    if 0 < self.line_pos <= hr and (
                        not have_mi_columns or hr != header[-1]
                    ):
                        # If no rows we want to raise a different message and if
                        # we have mi columns, the last line is not part of the header
                        joi = list(map(str, header[:-1] if have_mi_columns else header))
                        msg = f"[{','.join(joi)}], len of {len(joi)}, "
                        raise ValueError(
                            f"Passed header={msg}"
                            f"but only {self.line_pos} lines in file"
                        ) from err

                    # We have an empty file, so check
                    # if columns are provided. That will
                    # serve as the 'line' for parsing
                    if have_mi_columns and hr > 0:
                        if clear_buffer:
                            self._clear_buffer()
                        columns.append([None] * len(columns[-1]))
                        return columns, num_original_columns, unnamed_cols

                    if not self.names:
                        raise EmptyDataError("No columns to parse from file") from err

                    line = self.names[:]

                this_columns: list[Scalar | None] = []
                this_unnamed_cols = []

                for i, c in enumerate(line):
                    if c == "":
                        if have_mi_columns:
                            col_name = f"Unnamed: {i}_level_{level}"
                        else:
                            col_name = f"Unnamed: {i}"

                        this_unnamed_cols.append(i)
                        this_columns.append(col_name)
                    else:
                        this_columns.append(c)

                if not have_mi_columns:
                    counts: DefaultDict = defaultdict(int)
                    # Ensure that regular columns are used before unnamed ones
                    # to keep given names and mangle unnamed columns
                    col_loop_order = [
                        i
                        for i in range(len(this_columns))
                        if i not in this_unnamed_cols
                    ] + this_unnamed_cols

                    # TODO: Use pandas.io.common.dedup_names instead (see #50371)
                    for i in col_loop_order:
                        col = this_columns[i]
                        old_col = col
                        cur_count = counts[col]

                        if cur_count > 0:
                            while cur_count > 0:
                                counts[old_col] = cur_count + 1
                                col = f"{old_col}.{cur_count}"
                                if col in this_columns:
                                    cur_count += 1
                                else:
                                    cur_count = counts[col]

                            if (
                                self.dtype is not None
                                and is_dict_like(self.dtype)
                                and self.dtype.get(old_col) is not None
                                and self.dtype.get(col) is None
                            ):
                                self.dtype.update({col: self.dtype.get(old_col)})
                        this_columns[i] = col
                        counts[col] = cur_count + 1
                elif have_mi_columns:
                    # if we have grabbed an extra line, but its not in our
                    # format so save in the buffer, and create an blank extra
                    # line for the rest of the parsing code
                    if hr == header[-1]:
                        lc = len(this_columns)
                        # error: Cannot determine type of 'index_col'
                        sic = self.index_col  # type: ignore[has-type]
                        ic = len(sic) if sic is not None else 0
                        unnamed_count = len(this_unnamed_cols)

                        # if wrong number of blanks or no index, not our format
                        if (lc != unnamed_count and lc - ic > unnamed_count) or ic == 0:
                            clear_buffer = False
                            this_columns = [None] * lc
                            self.buf = [self.buf[-1]]

                columns.append(this_columns)
                unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})

                if len(columns) == 1:
                    num_original_columns = len(this_columns)

            if clear_buffer:
                self._clear_buffer()

            first_line: list[Scalar] | None
            if names is not None:
                # Read first row after header to check if data are longer
                try:
                    first_line = self._next_line()
                except StopIteration:
                    first_line = None

                len_first_data_row = 0 if first_line is None else len(first_line)

                if len(names) > len(columns[0]) and len(names) > len_first_data_row:
                    raise ValueError(
                        "Number of passed names did not match "
                        "number of header fields in the file"
                    )
                if len(columns) > 1:
                    raise TypeError("Cannot pass names with multi-index columns")

                if self.usecols is not None:
                    # Set _use_cols. We don't store columns because they are
                    # overwritten.
                    self._handle_usecols(columns, names, num_original_columns)
                else:
                    num_original_columns = len(names)
                if self._col_indices is not None and len(names) != len(
                    self._col_indices
                ):
                    columns = [[names[i] for i in sorted(self._col_indices)]]
                else:
                    columns = [names]
            else:
                columns = self._handle_usecols(
                    columns, columns[0], num_original_columns
                )
        else:
            ncols = len(self._header_line)
            num_original_columns = ncols

            if not names:
                columns = [list(range(ncols))]
                columns = self._handle_usecols(columns, columns[0], ncols)
            elif self.usecols is None or len(names) >= ncols:
                columns = self._handle_usecols([names], names, ncols)
                num_original_columns = len(names)
            elif not callable(self.usecols) and len(names) != len(self.usecols):
                raise ValueError(
                    "Number of passed names did not match number of "
                    "header fields in the file"
                )
            else:
                # Ignore output but set used columns.
                columns = [names]
                self._handle_usecols(columns, columns[0], ncols)

        return columns, num_original_columns, unnamed_cols

    @cache_readonly
    def _header_line(self):
        # Store line for reuse in _get_index_name
        if self.header is not None:
            return None

        try:
            line = self._buffered_line()
        except StopIteration as err:
            if not self.names:
                raise EmptyDataError("No columns to parse from file") from err

            line = self.names[:]
        return line

    def _handle_usecols(
        self,
        columns: list[list[Scalar | None]],
        usecols_key: list[Scalar | None],
        num_original_columns: int,
    ) -> list[list[Scalar | None]]:
        """
        Sets self._col_indices

        usecols_key is used if there are string usecols.
        """
        col_indices: set[int] | list[int]
        if self.usecols is not None:
            if callable(self.usecols):
                col_indices = self._evaluate_usecols(self.usecols, usecols_key)
            elif any(isinstance(u, str) for u in self.usecols):
                if len(columns) > 1:
                    raise ValueError(
                        "If using multiple headers, usecols must be integers."
                    )
                col_indices = []

                for col in self.usecols:
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            self._validate_usecols_names(self.usecols, usecols_key)
                    else:
                        col_indices.append(col)
            else:
                missing_usecols = [
                    col for col in self.usecols if col >= num_original_columns
                ]
                if missing_usecols:
                    raise ParserError(
                        "Defining usecols with out-of-bounds indices is not allowed. "
                        f"{missing_usecols} are out-of-bounds.",
                    )
                col_indices = self.usecols

            columns = [
                [n for i, n in enumerate(column) if i in col_indices]
                for column in columns
            ]
            self._col_indices = sorted(col_indices)
        return columns

    def _buffered_line(self) -> list[Scalar]:
        """
        Return a line from buffer, filling buffer if required.
        """
        if len(self.buf) > 0:
            return self.buf[0]
        else:
            return self._next_line()

    def _check_for_bom(self, first_row: list[Scalar]) -> list[Scalar]:
        """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
        # first_row will be a list, so we need to check
        # that that list is not empty before proceeding.
        if not first_row:
            return first_row

        # The first element of this row is the one that could have the
        # BOM that we want to remove. Check that the first element is a
        # string before proceeding.
        if not isinstance(first_row[0], str):
            return first_row

        # Check that the string is not empty, as that would
        # obviously not have a BOM at the start of it.
        if not first_row[0]:
            return first_row

        # Since the string is non-empty, check that it does
        # in fact begin with a BOM.
        first_elt = first_row[0][0]
        if first_elt != _BOM:
            return first_row

        first_row_bom = first_row[0]
        new_row: str

        if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
            start = 2
            quote = first_row_bom[1]
            end = first_row_bom[2:].index(quote) + 2

            # Extract the data between the quotation marks
            new_row = first_row_bom[start:end]

            # Extract any remaining data after the second
            # quotation mark.
            if len(first_row_bom) > end + 1:
                new_row += first_row_bom[end + 1 :]

        else:
            # No quotation so just remove BOM from first element
            new_row = first_row_bom[1:]

        new_row_list: list[Scalar] = [new_row]
        return new_row_list + first_row[1:]

    def _is_line_empty(self, line: list[Scalar]) -> bool:
        """
        Check if a line is empty or not.

        Parameters
        ----------
        line : str, array-like
            The line of data to check.

        Returns
        -------
        boolean : Whether or not the line is empty.
        """
        return not line or all(not x for x in line)

    def _next_line(self) -> list[Scalar]:
        if isinstance(self.data, list):
            while self.skipfunc(self.pos):
                if self.pos >= len(self.data):
                    break
                self.pos += 1

            while True:
                try:
                    line = self._check_comments([self.data[self.pos]])[0]
                    self.pos += 1
                    # either uncommented or blank to begin with
                    if not self.skip_blank_lines and (
                        self._is_line_empty(self.data[self.pos - 1]) or line
                    ):
                        break
                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                except IndexError:
                    raise StopIteration
        else:
            while self.skipfunc(self.pos):
                self.pos += 1
                # assert for mypy, data is Iterator[str] or None, would error in next
                assert self.data is not None
                next(self.data)

            while True:
                orig_line = self._next_iter_line(row_num=self.pos + 1)
                self.pos += 1

                if orig_line is not None:
                    line = self._check_comments([orig_line])[0]

                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])

                        if ret:
                            line = ret[0]
                            break
                    elif self._is_line_empty(orig_line) or line:
                        break

        # This was the first line of the file,
        # which could contain the BOM at the
        # beginning of it.
        if self.pos == 1:
            line = self._check_for_bom(line)

        self.line_pos += 1
        self.buf.append(line)
        return line

    def _alert_malformed(self, msg: str, row_num: int) -> None:
        """
        Alert a user about a malformed row, depending on value of
        `self.on_bad_lines` enum.

        If `self.on_bad_lines` is ERROR, the alert will be `ParserError`.
        If `self.on_bad_lines` is WARN, the alert will be printed out.

        Parameters
        ----------
        msg: str
            The error message to display.
        row_num: int
            The row number where the parsing error occurred.
            Because this row number is displayed, we 1-index,
            even though we 0-index internally.
        """
        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
            raise ParserError(msg)
        if self.on_bad_lines == self.BadLineHandleMethod.WARN:
            warnings.warn(
                f"Skipping line {row_num}: {msg}\n",
                ParserWarning,
                stacklevel=find_stack_level(),
            )

    def _next_iter_line(self, row_num: int) -> list[Scalar] | None:
        """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num: int
            The row number of the line being parsed.
        """
        try:
            # assert for mypy, data is Iterator[str] or None, would error in next
            assert self.data is not None
            line = next(self.data)
            # for mypy
            assert isinstance(line, list)
            return line
        except csv.Error as e:
            if self.on_bad_lines in (
                self.BadLineHandleMethod.ERROR,
                self.BadLineHandleMethod.WARN,
            ):
                msg = str(e)

                if "NULL byte" in msg or "line contains NUL" in msg:
                    msg = (
                        "NULL byte detected. This byte "
                        "cannot be processed in Python's "
                        "native csv library at the moment, "
                        "so please pass in engine='c' instead"
                    )

                if self.skipfooter > 0:
                    reason = (
                        "Error could possibly be due to "
                        "parsing errors in the skipped footer rows "
                        "(the skipfooter keyword is only applied "
                        "after Python's csv library has parsed "
                        "all rows)."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.comment is None:
            return lines
        ret = []
        for line in lines:
            rl = []
            for x in line:
                if (
                    not isinstance(x, str)
                    or self.comment not in x
                    or x in self.na_values
                ):
                    rl.append(x)
                else:
                    x = x[: x.find(self.comment)]
                    if len(x) > 0:
                        rl.append(x)
                    break
            ret.append(rl)
        return ret

    def _remove_empty_lines(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Iterate through the lines and remove any that are
        either empty or contain only one whitespace value

        Parameters
        ----------
        lines : list of list of Scalars
            The array of lines that we are to filter.

        Returns
        -------
        filtered_lines : list of list of Scalars
            The same array of lines with the "empty" ones removed.
        """
        # Remove empty lines and lines with only one whitespace value
        ret = [
            line
            for line in lines
            if (
                len(line) > 1
                or len(line) == 1
                and (not isinstance(line[0], str) or line[0].strip())
            )
        ]
        return ret

    def _check_thousands(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.thousands is None:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.thousands, replace=""
        )

    def _search_replace_num_columns(
        self, lines: list[list[Scalar]], search: str, replace: str
    ) -> list[list[Scalar]]:
        ret = []
        for line in lines:
            rl = []
            for i, x in enumerate(line):
                if (
                    not isinstance(x, str)
                    or search not in x
                    or i in self._no_thousands_columns
                    or not self.num.search(x.strip())
                ):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.decimal == parser_defaults["decimal"]:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.decimal, replace="."
        )

    def _clear_buffer(self) -> None:
        self.buf = []

    def _get_index_name(
        self,
    ) -> tuple[Sequence[Hashable] | None, list[Hashable], list[Hashable]]:
        """
        Try several cases to get lines:

        0) There are headers on row 0 and row 1 and their
        total summed lengths equals the length of the next line.
        Treat row 0 as columns and row 1 as indices
        1) Look for implicit index: there are more columns
        on row 1 than row 0. If this is true, assume that row
        1 lists index columns and row 0 lists normal columns.
        2) Get index from the columns if it was listed.
        """
        columns: Sequence[Hashable] = self.orig_names
        orig_names = list(columns)
        columns = list(columns)

        line: list[Scalar] | None
        if self._header_line is not None:
            line = self._header_line
        else:
            try:
                line = self._next_line()
            except StopIteration:
                line = None

        next_line: list[Scalar] | None
        try:
            next_line = self._next_line()
        except StopIteration:
            next_line = None

        # implicitly index_col=0 b/c 1 fewer column names
        implicit_first_cols = 0
        if line is not None:
            # leave it 0, #2442
            # Case 1
            # error: Cannot determine type of 'index_col'
            index_col = self.index_col  # type: ignore[has-type]
            if index_col is not False:
                implicit_first_cols = len(line) - self.num_original_columns

            # Case 0
            if (
                next_line is not None
                and self.header is not None
                and index_col is not False
            ):
                if len(next_line) == len(line) + self.num_original_columns:
                    # column and index names on diff rows
                    self.index_col = list(range(len(line)))
                    self.buf = self.buf[1:]

                    for c in reversed(line):
                        columns.insert(0, c)

                    # Update list of original names to include all indices.
                    orig_names = list(columns)
                    self.num_original_columns = len(columns)
                    return line, orig_names, columns

        if implicit_first_cols > 0:
            # Case 1
            self._implicit_index = True
            if self.index_col is None:
                self.index_col = list(range(implicit_first_cols))

            index_name = None

        else:
            # Case 2
            (index_name, _, self.index_col) = self._clean_index_names(
                columns, self.index_col
            )

        return index_name, orig_names, columns

    def _rows_to_cols(self, content: list[list[Scalar]]) -> list[np.ndarray]:
        col_len = self.num_original_columns

        if self._implicit_index:
            col_len += len(self.index_col)

        max_len = max(len(row) for row in content)

        # Check that there are no rows with too many
        # elements in their row (rows with too few
        # elements are padded with NaN).
        # error: Non-overlapping identity check (left operand type: "List[int]",
        # right operand type: "Literal[False]")
        if (
            max_len > col_len
            and self.index_col is not False  # type: ignore[comparison-overlap]
            and self.usecols is None
        ):
            footers = self.skipfooter if self.skipfooter else 0
            bad_lines = []

            iter_content = enumerate(content)
            content_len = len(content)
            content = []

            for i, _content in iter_content:
                actual_len = len(_content)

                if actual_len > col_len:
                    if callable(self.on_bad_lines):
                        new_l = self.on_bad_lines(_content)
                        if new_l is not None:
                            content.append(new_l)
                    elif self.on_bad_lines in (
                        self.BadLineHandleMethod.ERROR,
                        self.BadLineHandleMethod.WARN,
                    ):
                        row_num = self.pos - (content_len - i + footers)
                        bad_lines.append((row_num, actual_len))

                        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
                            break
                else:
                    content.append(_content)

            for row_num, actual_len in bad_lines:
                msg = (
                    f"Expected {col_len} fields in line {row_num + 1}, saw "
                    f"{actual_len}"
                )
                if (
                    self.delimiter
                    and len(self.delimiter) > 1
                    and self.quoting != csv.QUOTE_NONE
                ):
                    # see gh-13374
                    reason = (
                        "Error could possibly be due to quotes being "
                        "ignored when a multi-char delimiter is used."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num + 1)

        # see gh-13320
        zipped_content = list(lib.to_object_array(content, min_width=col_len).T)

        if self.usecols:
            assert self._col_indices is not None
            col_indices = self._col_indices

            if self._implicit_index:
                zipped_content = [
                    a
                    for i, a in enumerate(zipped_content)
                    if (
                        i < len(self.index_col)
                        or i - len(self.index_col) in col_indices
                    )
                ]
            else:
                zipped_content = [
                    a for i, a in enumerate(zipped_content) if i in col_indices
                ]
        return zipped_content

    def _get_lines(self, rows: int | None = None) -> list[list[Scalar]]:
        lines = self.buf
        new_rows = None

        # already fetched some number
        if rows is not None:
            # we already have the lines in the buffer
            if len(self.buf) >= rows:
                new_rows, self.buf = self.buf[:rows], self.buf[rows:]

            # need some lines
            else:
                rows -= len(self.buf)

        if new_rows is None:
            if isinstance(self.data, list):
                if self.pos > len(self.data):
                    raise StopIteration
                if rows is None:
                    new_rows = self.data[self.pos :]
                    new_pos = len(self.data)
                else:
                    new_rows = self.data[self.pos : self.pos + rows]
                    new_pos = self.pos + rows

                new_rows = self._remove_skipped_rows(new_rows)
                lines.extend(new_rows)
                self.pos = new_pos

            else:
                new_rows = []
                try:
                    if rows is not None:
                        row_index = 0
                        row_ct = 0
                        offset = self.pos if self.pos is not None else 0
                        while row_ct < rows:
                            # assert for mypy, data is Iterator[str] or None, would
                            # error in next
                            assert self.data is not None
                            new_row = next(self.data)
                            if not self.skipfunc(offset + row_index):
                                row_ct += 1
                            row_index += 1
                            new_rows.append(new_row)

                        len_new_rows = len(new_rows)
                        new_rows = self._remove_skipped_rows(new_rows)
                        lines.extend(new_rows)
                    else:
                        rows = 0

                        while True:
                            next_row = self._next_iter_line(row_num=self.pos + rows + 1)
                            rows += 1

                            if next_row is not None:
                                new_rows.append(next_row)
                        len_new_rows = len(new_rows)

                except StopIteration:
                    len_new_rows = len(new_rows)
                    new_rows = self._remove_skipped_rows(new_rows)
                    lines.extend(new_rows)
                    if len(lines) == 0:
                        raise
                self.pos += len_new_rows

            self.buf = []
        else:
            lines = new_rows

        if self.skipfooter:
            lines = lines[: -self.skipfooter]

        lines = self._check_comments(lines)
        if self.skip_blank_lines:
            lines = self._remove_empty_lines(lines)
        lines = self._check_thousands(lines)
        return self._check_decimal(lines)

    def _remove_skipped_rows(self, new_rows: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.skiprows:
            return [
                row for i, row in enumerate(new_rows) if not self.skipfunc(i + self.pos)
            ]
        return new_rows

    def _set_no_thousand_columns(self) -> set[int]:
        no_thousands_columns: set[int] = set()
        if self.columns and self.parse_dates:
            assert self._col_indices is not None
            no_thousands_columns = self._set_noconvert_dtype_columns(
                self._col_indices, self.columns
            )
        if self.columns and self.dtype:
            assert self._col_indices is not None
            for i, col in zip(self._col_indices, self.columns):
                if not isinstance(self.dtype, dict) and not is_numeric_dtype(
                    self.dtype
                ):
                    no_thousands_columns.add(i)
                if (
                    isinstance(self.dtype, dict)
                    and col in self.dtype
                    and (
                        not is_numeric_dtype(self.dtype[col])
                        or is_bool_dtype(self.dtype[col])
                    )
                ):
                    no_thousands_columns.add(i)
        return no_thousands_columns


class FixedWidthReader(abc.Iterator):
    """
    A reader of fixed-width lines.
    """

    def __init__(
        self,
        f: IO[str] | ReadCsvBuffer[str],
        colspecs: list[tuple[int, int]] | Literal["infer"],
        delimiter: str | None,
        comment: str | None,
        skiprows: set[int] | None = None,
        infer_nrows: int = 100,
    ) -> None:
        self.f = f
        self.buffer: Iterator | None = None
        self.delimiter = "\r\n" + delimiter if delimiter else "\n\r\t "
        self.comment = comment
        if colspecs == "infer":
            self.colspecs = self.detect_colspecs(
                infer_nrows=infer_nrows, skiprows=skiprows
            )
        else:
            self.colspecs = colspecs

        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(
                "column specifications must be a list or tuple, "
                f"input was a {type(colspecs).__name__}"
            )

        for colspec in self.colspecs:
            if not (
                isinstance(colspec, (tuple, list))
                and len(colspec) == 2
                and isinstance(colspec[0], (int, np.integer, type(None)))
                and isinstance(colspec[1], (int, np.integer, type(None)))
            ):
                raise TypeError(
                    "Each column specification must be "
                    "2 element tuple or list of integers"
                )

    def get_rows(self, infer_nrows: int, skiprows: set[int] | None = None) -> list[str]:
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        if skiprows is None:
            skiprows = set()
        buffer_rows = []
        detect_rows = []
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)
            buffer_rows.append(row)
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(
        self, infer_nrows: int = 100, skiprows: set[int] | None = None
    ) -> list[tuple[int, int]]:
        # Regex escape the delimiters
        delimiters = "".join([rf"\{x}" for x in self.delimiter])
        pattern = re.compile(f"([^{delimiters}]+)")
        rows = self.get_rows(infer_nrows, skiprows)
        if not rows:
            raise EmptyDataError("No rows from which to infer column width")
        max_len = max(map(len, rows))
        mask = np.zeros(max_len + 1, dtype=int)
        if self.comment is not None:
            rows = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start() : m.end()] = 1
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        edges = np.where((mask ^ shifted) == 1)[0]
        edge_pairs = list(zip(edges[::2], edges[1::2]))
        return edge_pairs

    def __next__(self) -> list[str]:
        # Argument 1 to "next" has incompatible type "Union[IO[str],
        # ReadCsvBuffer[str]]"; expected "SupportsNext[str]"
        if self.buffer is not None:
            try:
                line = next(self.buffer)
            except StopIteration:
                self.buffer = None
                line = next(self.f)  # type: ignore[arg-type]
        else:
            line = next(self.f)  # type: ignore[arg-type]
        # Note: 'colspecs' is a sequence of half-open intervals.
        return [line[from_:to].strip(self.delimiter) for (from_, to) in self.colspecs]


class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None:
        # Support iterators, convert to a list.
        self.colspecs = kwds.pop("colspecs")
        self.infer_nrows = kwds.pop("infer_nrows")
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> FixedWidthReader:
        return FixedWidthReader(
            f,
            self.colspecs,
            self.delimiter,
            self.comment,
            self.skiprows,
            self.infer_nrows,
        )

    def _remove_empty_lines(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Returns the list of lines without the empty ones. With fixed-width
        fields, empty lines become arrays of empty strings.

        See PythonParser._remove_empty_lines.
        """
        return [
            line
            for line in lines
            if any(not isinstance(e, str) or e.strip() for e in line)
        ]


def count_empty_vals(vals) -> int:
    return sum(1 for v in vals if v == "" or v is None)


def _validate_skipfooter_arg(skipfooter: int) -> int:
    """
    Validate the 'skipfooter' parameter.

    Checks whether 'skipfooter' is a non-negative integer.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    skipfooter : non-negative integer
        The number of rows to skip at the end of the file.

    Returns
    -------
    validated_skipfooter : non-negative integer
        The original input if the validation succeeds.

    Raises
    ------
    ValueError : 'skipfooter' was not a non-negative integer.
    """
    if not is_integer(skipfooter):
        raise ValueError("skipfooter must be an integer")

    if skipfooter < 0:
        raise ValueError("skipfooter cannot be negative")

    # Incompatible return value type (got "Union[int, integer[Any]]", expected "int")
    return skipfooter  # type: ignore[return-value]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\overlay.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Overlay (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime


@dataclass
class SourceOrderConfig:
    '''
    Configuration data for drawing the source order of an elements children.
    '''
    #: the color to outline the given element in.
    parent_outline_color: dom.RGBA

    #: the color to outline the child elements in.
    child_outline_color: dom.RGBA

    def to_json(self):
        json = dict()
        json['parentOutlineColor'] = self.parent_outline_color.to_json()
        json['childOutlineColor'] = self.child_outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            parent_outline_color=dom.RGBA.from_json(json['parentOutlineColor']),
            child_outline_color=dom.RGBA.from_json(json['childOutlineColor']),
        )


@dataclass
class GridHighlightConfig:
    '''
    Configuration data for the highlighting of Grid elements.
    '''
    #: Whether the extension lines from grid cells to the rulers should be shown (default: false).
    show_grid_extension_lines: typing.Optional[bool] = None

    #: Show Positive line number labels (default: false).
    show_positive_line_numbers: typing.Optional[bool] = None

    #: Show Negative line number labels (default: false).
    show_negative_line_numbers: typing.Optional[bool] = None

    #: Show area name labels (default: false).
    show_area_names: typing.Optional[bool] = None

    #: Show line name labels (default: false).
    show_line_names: typing.Optional[bool] = None

    #: Show track size labels (default: false).
    show_track_sizes: typing.Optional[bool] = None

    #: The grid container border highlight color (default: transparent).
    grid_border_color: typing.Optional[dom.RGBA] = None

    #: The cell border color (default: transparent). Deprecated, please use rowLineColor and columnLineColor instead.
    cell_border_color: typing.Optional[dom.RGBA] = None

    #: The row line color (default: transparent).
    row_line_color: typing.Optional[dom.RGBA] = None

    #: The column line color (default: transparent).
    column_line_color: typing.Optional[dom.RGBA] = None

    #: Whether the grid border is dashed (default: false).
    grid_border_dash: typing.Optional[bool] = None

    #: Whether the cell border is dashed (default: false). Deprecated, please us rowLineDash and columnLineDash instead.
    cell_border_dash: typing.Optional[bool] = None

    #: Whether row lines are dashed (default: false).
    row_line_dash: typing.Optional[bool] = None

    #: Whether column lines are dashed (default: false).
    column_line_dash: typing.Optional[bool] = None

    #: The row gap highlight fill color (default: transparent).
    row_gap_color: typing.Optional[dom.RGBA] = None

    #: The row gap hatching fill color (default: transparent).
    row_hatch_color: typing.Optional[dom.RGBA] = None

    #: The column gap highlight fill color (default: transparent).
    column_gap_color: typing.Optional[dom.RGBA] = None

    #: The column gap hatching fill color (default: transparent).
    column_hatch_color: typing.Optional[dom.RGBA] = None

    #: The named grid areas border color (Default: transparent).
    area_border_color: typing.Optional[dom.RGBA] = None

    #: The grid container background color (Default: transparent).
    grid_background_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.show_grid_extension_lines is not None:
            json['showGridExtensionLines'] = self.show_grid_extension_lines
        if self.show_positive_line_numbers is not None:
            json['showPositiveLineNumbers'] = self.show_positive_line_numbers
        if self.show_negative_line_numbers is not None:
            json['showNegativeLineNumbers'] = self.show_negative_line_numbers
        if self.show_area_names is not None:
            json['showAreaNames'] = self.show_area_names
        if self.show_line_names is not None:
            json['showLineNames'] = self.show_line_names
        if self.show_track_sizes is not None:
            json['showTrackSizes'] = self.show_track_sizes
        if self.grid_border_color is not None:
            json['gridBorderColor'] = self.grid_border_color.to_json()
        if self.cell_border_color is not None:
            json['cellBorderColor'] = self.cell_border_color.to_json()
        if self.row_line_color is not None:
            json['rowLineColor'] = self.row_line_color.to_json()
        if self.column_line_color is not None:
            json['columnLineColor'] = self.column_line_color.to_json()
        if self.grid_border_dash is not None:
            json['gridBorderDash'] = self.grid_border_dash
        if self.cell_border_dash is not None:
            json['cellBorderDash'] = self.cell_border_dash
        if self.row_line_dash is not None:
            json['rowLineDash'] = self.row_line_dash
        if self.column_line_dash is not None:
            json['columnLineDash'] = self.column_line_dash
        if self.row_gap_color is not None:
            json['rowGapColor'] = self.row_gap_color.to_json()
        if self.row_hatch_color is not None:
            json['rowHatchColor'] = self.row_hatch_color.to_json()
        if self.column_gap_color is not None:
            json['columnGapColor'] = self.column_gap_color.to_json()
        if self.column_hatch_color is not None:
            json['columnHatchColor'] = self.column_hatch_color.to_json()
        if self.area_border_color is not None:
            json['areaBorderColor'] = self.area_border_color.to_json()
        if self.grid_background_color is not None:
            json['gridBackgroundColor'] = self.grid_background_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_grid_extension_lines=bool(json['showGridExtensionLines']) if 'showGridExtensionLines' in json else None,
            show_positive_line_numbers=bool(json['showPositiveLineNumbers']) if 'showPositiveLineNumbers' in json else None,
            show_negative_line_numbers=bool(json['showNegativeLineNumbers']) if 'showNegativeLineNumbers' in json else None,
            show_area_names=bool(json['showAreaNames']) if 'showAreaNames' in json else None,
            show_line_names=bool(json['showLineNames']) if 'showLineNames' in json else None,
            show_track_sizes=bool(json['showTrackSizes']) if 'showTrackSizes' in json else None,
            grid_border_color=dom.RGBA.from_json(json['gridBorderColor']) if 'gridBorderColor' in json else None,
            cell_border_color=dom.RGBA.from_json(json['cellBorderColor']) if 'cellBorderColor' in json else None,
            row_line_color=dom.RGBA.from_json(json['rowLineColor']) if 'rowLineColor' in json else None,
            column_line_color=dom.RGBA.from_json(json['columnLineColor']) if 'columnLineColor' in json else None,
            grid_border_dash=bool(json['gridBorderDash']) if 'gridBorderDash' in json else None,
            cell_border_dash=bool(json['cellBorderDash']) if 'cellBorderDash' in json else None,
            row_line_dash=bool(json['rowLineDash']) if 'rowLineDash' in json else None,
            column_line_dash=bool(json['columnLineDash']) if 'columnLineDash' in json else None,
            row_gap_color=dom.RGBA.from_json(json['rowGapColor']) if 'rowGapColor' in json else None,
            row_hatch_color=dom.RGBA.from_json(json['rowHatchColor']) if 'rowHatchColor' in json else None,
            column_gap_color=dom.RGBA.from_json(json['columnGapColor']) if 'columnGapColor' in json else None,
            column_hatch_color=dom.RGBA.from_json(json['columnHatchColor']) if 'columnHatchColor' in json else None,
            area_border_color=dom.RGBA.from_json(json['areaBorderColor']) if 'areaBorderColor' in json else None,
            grid_background_color=dom.RGBA.from_json(json['gridBackgroundColor']) if 'gridBackgroundColor' in json else None,
        )


@dataclass
class FlexContainerHighlightConfig:
    '''
    Configuration data for the highlighting of Flex container elements.
    '''
    #: The style of the container border
    container_border: typing.Optional[LineStyle] = None

    #: The style of the separator between lines
    line_separator: typing.Optional[LineStyle] = None

    #: The style of the separator between items
    item_separator: typing.Optional[LineStyle] = None

    #: Style of content-distribution space on the main axis (justify-content).
    main_distributed_space: typing.Optional[BoxStyle] = None

    #: Style of content-distribution space on the cross axis (align-content).
    cross_distributed_space: typing.Optional[BoxStyle] = None

    #: Style of empty space caused by row gaps (gap/row-gap).
    row_gap_space: typing.Optional[BoxStyle] = None

    #: Style of empty space caused by columns gaps (gap/column-gap).
    column_gap_space: typing.Optional[BoxStyle] = None

    #: Style of the self-alignment line (align-items).
    cross_alignment: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.line_separator is not None:
            json['lineSeparator'] = self.line_separator.to_json()
        if self.item_separator is not None:
            json['itemSeparator'] = self.item_separator.to_json()
        if self.main_distributed_space is not None:
            json['mainDistributedSpace'] = self.main_distributed_space.to_json()
        if self.cross_distributed_space is not None:
            json['crossDistributedSpace'] = self.cross_distributed_space.to_json()
        if self.row_gap_space is not None:
            json['rowGapSpace'] = self.row_gap_space.to_json()
        if self.column_gap_space is not None:
            json['columnGapSpace'] = self.column_gap_space.to_json()
        if self.cross_alignment is not None:
            json['crossAlignment'] = self.cross_alignment.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None,
            line_separator=LineStyle.from_json(json['lineSeparator']) if 'lineSeparator' in json else None,
            item_separator=LineStyle.from_json(json['itemSeparator']) if 'itemSeparator' in json else None,
            main_distributed_space=BoxStyle.from_json(json['mainDistributedSpace']) if 'mainDistributedSpace' in json else None,
            cross_distributed_space=BoxStyle.from_json(json['crossDistributedSpace']) if 'crossDistributedSpace' in json else None,
            row_gap_space=BoxStyle.from_json(json['rowGapSpace']) if 'rowGapSpace' in json else None,
            column_gap_space=BoxStyle.from_json(json['columnGapSpace']) if 'columnGapSpace' in json else None,
            cross_alignment=LineStyle.from_json(json['crossAlignment']) if 'crossAlignment' in json else None,
        )


@dataclass
class FlexItemHighlightConfig:
    '''
    Configuration data for the highlighting of Flex item elements.
    '''
    #: Style of the box representing the item's base size
    base_size_box: typing.Optional[BoxStyle] = None

    #: Style of the border around the box representing the item's base size
    base_size_border: typing.Optional[LineStyle] = None

    #: Style of the arrow representing if the item grew or shrank
    flexibility_arrow: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.base_size_box is not None:
            json['baseSizeBox'] = self.base_size_box.to_json()
        if self.base_size_border is not None:
            json['baseSizeBorder'] = self.base_size_border.to_json()
        if self.flexibility_arrow is not None:
            json['flexibilityArrow'] = self.flexibility_arrow.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            base_size_box=BoxStyle.from_json(json['baseSizeBox']) if 'baseSizeBox' in json else None,
            base_size_border=LineStyle.from_json(json['baseSizeBorder']) if 'baseSizeBorder' in json else None,
            flexibility_arrow=LineStyle.from_json(json['flexibilityArrow']) if 'flexibilityArrow' in json else None,
        )


@dataclass
class LineStyle:
    '''
    Style information for drawing a line.
    '''
    #: The color of the line (default: transparent)
    color: typing.Optional[dom.RGBA] = None

    #: The line pattern (default: solid)
    pattern: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.color is not None:
            json['color'] = self.color.to_json()
        if self.pattern is not None:
            json['pattern'] = self.pattern
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            color=dom.RGBA.from_json(json['color']) if 'color' in json else None,
            pattern=str(json['pattern']) if 'pattern' in json else None,
        )


@dataclass
class BoxStyle:
    '''
    Style information for drawing a box.
    '''
    #: The background color for the box (default: transparent)
    fill_color: typing.Optional[dom.RGBA] = None

    #: The hatching color for the box (default: transparent)
    hatch_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.fill_color is not None:
            json['fillColor'] = self.fill_color.to_json()
        if self.hatch_color is not None:
            json['hatchColor'] = self.hatch_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fill_color=dom.RGBA.from_json(json['fillColor']) if 'fillColor' in json else None,
            hatch_color=dom.RGBA.from_json(json['hatchColor']) if 'hatchColor' in json else None,
        )


class ContrastAlgorithm(enum.Enum):
    AA = "aa"
    AAA = "aaa"
    APCA = "apca"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class HighlightConfig:
    '''
    Configuration data for the highlighting of page elements.
    '''
    #: Whether the node info tooltip should be shown (default: false).
    show_info: typing.Optional[bool] = None

    #: Whether the node styles in the tooltip (default: false).
    show_styles: typing.Optional[bool] = None

    #: Whether the rulers should be shown (default: false).
    show_rulers: typing.Optional[bool] = None

    #: Whether the a11y info should be shown (default: true).
    show_accessibility_info: typing.Optional[bool] = None

    #: Whether the extension lines from node to the rulers should be shown (default: false).
    show_extension_lines: typing.Optional[bool] = None

    #: The content box highlight fill color (default: transparent).
    content_color: typing.Optional[dom.RGBA] = None

    #: The padding highlight fill color (default: transparent).
    padding_color: typing.Optional[dom.RGBA] = None

    #: The border highlight fill color (default: transparent).
    border_color: typing.Optional[dom.RGBA] = None

    #: The margin highlight fill color (default: transparent).
    margin_color: typing.Optional[dom.RGBA] = None

    #: The event target element highlight fill color (default: transparent).
    event_target_color: typing.Optional[dom.RGBA] = None

    #: The shape outside fill color (default: transparent).
    shape_color: typing.Optional[dom.RGBA] = None

    #: The shape margin fill color (default: transparent).
    shape_margin_color: typing.Optional[dom.RGBA] = None

    #: The grid layout color (default: transparent).
    css_grid_color: typing.Optional[dom.RGBA] = None

    #: The color format used to format color styles (default: hex).
    color_format: typing.Optional[ColorFormat] = None

    #: The grid layout highlight configuration (default: all transparent).
    grid_highlight_config: typing.Optional[GridHighlightConfig] = None

    #: The flex container highlight configuration (default: all transparent).
    flex_container_highlight_config: typing.Optional[FlexContainerHighlightConfig] = None

    #: The flex item highlight configuration (default: all transparent).
    flex_item_highlight_config: typing.Optional[FlexItemHighlightConfig] = None

    #: The contrast algorithm to use for the contrast ratio (default: aa).
    contrast_algorithm: typing.Optional[ContrastAlgorithm] = None

    #: The container query container highlight configuration (default: all transparent).
    container_query_container_highlight_config: typing.Optional[ContainerQueryContainerHighlightConfig] = None

    def to_json(self):
        json = dict()
        if self.show_info is not None:
            json['showInfo'] = self.show_info
        if self.show_styles is not None:
            json['showStyles'] = self.show_styles
        if self.show_rulers is not None:
            json['showRulers'] = self.show_rulers
        if self.show_accessibility_info is not None:
            json['showAccessibilityInfo'] = self.show_accessibility_info
        if self.show_extension_lines is not None:
            json['showExtensionLines'] = self.show_extension_lines
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.padding_color is not None:
            json['paddingColor'] = self.padding_color.to_json()
        if self.border_color is not None:
            json['borderColor'] = self.border_color.to_json()
        if self.margin_color is not None:
            json['marginColor'] = self.margin_color.to_json()
        if self.event_target_color is not None:
            json['eventTargetColor'] = self.event_target_color.to_json()
        if self.shape_color is not None:
            json['shapeColor'] = self.shape_color.to_json()
        if self.shape_margin_color is not None:
            json['shapeMarginColor'] = self.shape_margin_color.to_json()
        if self.css_grid_color is not None:
            json['cssGridColor'] = self.css_grid_color.to_json()
        if self.color_format is not None:
            json['colorFormat'] = self.color_format.to_json()
        if self.grid_highlight_config is not None:
            json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        if self.flex_container_highlight_config is not None:
            json['flexContainerHighlightConfig'] = self.flex_container_highlight_config.to_json()
        if self.flex_item_highlight_config is not None:
            json['flexItemHighlightConfig'] = self.flex_item_highlight_config.to_json()
        if self.contrast_algorithm is not None:
            json['contrastAlgorithm'] = self.contrast_algorithm.to_json()
        if self.container_query_container_highlight_config is not None:
            json['containerQueryContainerHighlightConfig'] = self.container_query_container_highlight_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_info=bool(json['showInfo']) if 'showInfo' in json else None,
            show_styles=bool(json['showStyles']) if 'showStyles' in json else None,
            show_rulers=bool(json['showRulers']) if 'showRulers' in json else None,
            show_accessibility_info=bool(json['showAccessibilityInfo']) if 'showAccessibilityInfo' in json else None,
            show_extension_lines=bool(json['showExtensionLines']) if 'showExtensionLines' in json else None,
            content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None,
            padding_color=dom.RGBA.from_json(json['paddingColor']) if 'paddingColor' in json else None,
            border_color=dom.RGBA.from_json(json['borderColor']) if 'borderColor' in json else None,
            margin_color=dom.RGBA.from_json(json['marginColor']) if 'marginColor' in json else None,
            event_target_color=dom.RGBA.from_json(json['eventTargetColor']) if 'eventTargetColor' in json else None,
            shape_color=dom.RGBA.from_json(json['shapeColor']) if 'shapeColor' in json else None,
            shape_margin_color=dom.RGBA.from_json(json['shapeMarginColor']) if 'shapeMarginColor' in json else None,
            css_grid_color=dom.RGBA.from_json(json['cssGridColor']) if 'cssGridColor' in json else None,
            color_format=ColorFormat.from_json(json['colorFormat']) if 'colorFormat' in json else None,
            grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']) if 'gridHighlightConfig' in json else None,
            flex_container_highlight_config=FlexContainerHighlightConfig.from_json(json['flexContainerHighlightConfig']) if 'flexContainerHighlightConfig' in json else None,
            flex_item_highlight_config=FlexItemHighlightConfig.from_json(json['flexItemHighlightConfig']) if 'flexItemHighlightConfig' in json else None,
            contrast_algorithm=ContrastAlgorithm.from_json(json['contrastAlgorithm']) if 'contrastAlgorithm' in json else None,
            container_query_container_highlight_config=ContainerQueryContainerHighlightConfig.from_json(json['containerQueryContainerHighlightConfig']) if 'containerQueryContainerHighlightConfig' in json else None,
        )


class ColorFormat(enum.Enum):
    RGB = "rgb"
    HSL = "hsl"
    HWB = "hwb"
    HEX_ = "hex"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class GridNodeHighlightConfig:
    '''
    Configurations for Persistent Grid Highlight
    '''
    #: A descriptor for the highlight appearance.
    grid_highlight_config: GridHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class FlexNodeHighlightConfig:
    #: A descriptor for the highlight appearance of flex containers.
    flex_container_highlight_config: FlexContainerHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['flexContainerHighlightConfig'] = self.flex_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            flex_container_highlight_config=FlexContainerHighlightConfig.from_json(json['flexContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class ScrollSnapContainerHighlightConfig:
    #: The style of the snapport border (default: transparent)
    snapport_border: typing.Optional[LineStyle] = None

    #: The style of the snap area border (default: transparent)
    snap_area_border: typing.Optional[LineStyle] = None

    #: The margin highlight fill color (default: transparent).
    scroll_margin_color: typing.Optional[dom.RGBA] = None

    #: The padding highlight fill color (default: transparent).
    scroll_padding_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.snapport_border is not None:
            json['snapportBorder'] = self.snapport_border.to_json()
        if self.snap_area_border is not None:
            json['snapAreaBorder'] = self.snap_area_border.to_json()
        if self.scroll_margin_color is not None:
            json['scrollMarginColor'] = self.scroll_margin_color.to_json()
        if self.scroll_padding_color is not None:
            json['scrollPaddingColor'] = self.scroll_padding_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            snapport_border=LineStyle.from_json(json['snapportBorder']) if 'snapportBorder' in json else None,
            snap_area_border=LineStyle.from_json(json['snapAreaBorder']) if 'snapAreaBorder' in json else None,
            scroll_margin_color=dom.RGBA.from_json(json['scrollMarginColor']) if 'scrollMarginColor' in json else None,
            scroll_padding_color=dom.RGBA.from_json(json['scrollPaddingColor']) if 'scrollPaddingColor' in json else None,
        )


@dataclass
class ScrollSnapHighlightConfig:
    #: A descriptor for the highlight appearance of scroll snap containers.
    scroll_snap_container_highlight_config: ScrollSnapContainerHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['scrollSnapContainerHighlightConfig'] = self.scroll_snap_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            scroll_snap_container_highlight_config=ScrollSnapContainerHighlightConfig.from_json(json['scrollSnapContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class HingeConfig:
    '''
    Configuration for dual screen hinge
    '''
    #: A rectangle represent hinge
    rect: dom.Rect

    #: The content box highlight fill color (default: a dark color).
    content_color: typing.Optional[dom.RGBA] = None

    #: The content box highlight outline color (default: transparent).
    outline_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        json['rect'] = self.rect.to_json()
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.outline_color is not None:
            json['outlineColor'] = self.outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            rect=dom.Rect.from_json(json['rect']),
            content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None,
            outline_color=dom.RGBA.from_json(json['outlineColor']) if 'outlineColor' in json else None,
        )


@dataclass
class WindowControlsOverlayConfig:
    '''
    Configuration for Window Controls Overlay
    '''
    #: Whether the title bar CSS should be shown when emulating the Window Controls Overlay.
    show_css: bool

    #: Selected platforms to show the overlay.
    selected_platform: str

    #: The theme color defined in app manifest.
    theme_color: str

    def to_json(self):
        json = dict()
        json['showCSS'] = self.show_css
        json['selectedPlatform'] = self.selected_platform
        json['themeColor'] = self.theme_color
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_css=bool(json['showCSS']),
            selected_platform=str(json['selectedPlatform']),
            theme_color=str(json['themeColor']),
        )


@dataclass
class ContainerQueryHighlightConfig:
    #: A descriptor for the highlight appearance of container query containers.
    container_query_container_highlight_config: ContainerQueryContainerHighlightConfig

    #: Identifier of the container node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['containerQueryContainerHighlightConfig'] = self.container_query_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_query_container_highlight_config=ContainerQueryContainerHighlightConfig.from_json(json['containerQueryContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class ContainerQueryContainerHighlightConfig:
    #: The style of the container border.
    container_border: typing.Optional[LineStyle] = None

    #: The style of the descendants' borders.
    descendant_border: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.descendant_border is not None:
            json['descendantBorder'] = self.descendant_border.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None,
            descendant_border=LineStyle.from_json(json['descendantBorder']) if 'descendantBorder' in json else None,
        )


@dataclass
class IsolatedElementHighlightConfig:
    #: A descriptor for the highlight appearance of an element in isolation mode.
    isolation_mode_highlight_config: IsolationModeHighlightConfig

    #: Identifier of the isolated element to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['isolationModeHighlightConfig'] = self.isolation_mode_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            isolation_mode_highlight_config=IsolationModeHighlightConfig.from_json(json['isolationModeHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class IsolationModeHighlightConfig:
    #: The fill color of the resizers (default: transparent).
    resizer_color: typing.Optional[dom.RGBA] = None

    #: The fill color for resizer handles (default: transparent).
    resizer_handle_color: typing.Optional[dom.RGBA] = None

    #: The fill color for the mask covering non-isolated elements (default: transparent).
    mask_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.resizer_color is not None:
            json['resizerColor'] = self.resizer_color.to_json()
        if self.resizer_handle_color is not None:
            json['resizerHandleColor'] = self.resizer_handle_color.to_json()
        if self.mask_color is not None:
            json['maskColor'] = self.mask_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            resizer_color=dom.RGBA.from_json(json['resizerColor']) if 'resizerColor' in json else None,
            resizer_handle_color=dom.RGBA.from_json(json['resizerHandleColor']) if 'resizerHandleColor' in json else None,
            mask_color=dom.RGBA.from_json(json['maskColor']) if 'maskColor' in json else None,
        )


class InspectMode(enum.Enum):
    SEARCH_FOR_NODE = "searchForNode"
    SEARCH_FOR_UA_SHADOW_DOM = "searchForUAShadowDOM"
    CAPTURE_AREA_SCREENSHOT = "captureAreaScreenshot"
    SHOW_DISTANCES = "showDistances"
    NONE = "none"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.enable',
    }
    json = yield cmd_dict


def get_highlight_object_for_test(
        node_id: dom.NodeId,
        include_distance: typing.Optional[bool] = None,
        include_style: typing.Optional[bool] = None,
        color_format: typing.Optional[ColorFormat] = None,
        show_accessibility_info: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For testing.

    :param node_id: Id of the node to get highlight object for.
    :param include_distance: *(Optional)* Whether to include distance info.
    :param include_style: *(Optional)* Whether to include style info.
    :param color_format: *(Optional)* The color format to get config with (default: hex).
    :param show_accessibility_info: *(Optional)* Whether to show accessibility info (default: true).
    :returns: Highlight data for the node.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    if include_distance is not None:
        params['includeDistance'] = include_distance
    if include_style is not None:
        params['includeStyle'] = include_style
    if color_format is not None:
        params['colorFormat'] = color_format.to_json()
    if show_accessibility_info is not None:
        params['showAccessibilityInfo'] = show_accessibility_info
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getHighlightObjectForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlight'])


def get_grid_highlight_objects_for_test(
        node_ids: typing.List[dom.NodeId]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For Persistent Grid testing.

    :param node_ids: Ids of the node to get highlight object for.
    :returns: Grid Highlight data for the node ids provided.
    '''
    params: T_JSON_DICT = dict()
    params['nodeIds'] = [i.to_json() for i in node_ids]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getGridHighlightObjectsForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlights'])


def get_source_order_highlight_object_for_test(
        node_id: dom.NodeId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For Source Order Viewer testing.

    :param node_id: Id of the node to highlight.
    :returns: Source order highlight data for the node id provided.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getSourceOrderHighlightObjectForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlight'])


def hide_highlight() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Hides any highlight.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.hideHighlight',
    }
    json = yield cmd_dict


def highlight_frame(
        frame_id: page.FrameId,
        content_color: typing.Optional[dom.RGBA] = None,
        content_outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights owner element of the frame with given id.
    Deprecated: Doesn't work reliably and cannot be fixed due to process
    separation (the owner node might be in a different process). Determine
    the owner node in the client and use highlightNode.

    :param frame_id: Identifier of the frame to highlight.
    :param content_color: *(Optional)* The content box highlight fill color (default: transparent).
    :param content_outline_color: *(Optional)* The content box highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    if content_color is not None:
        params['contentColor'] = content_color.to_json()
    if content_outline_color is not None:
        params['contentOutlineColor'] = content_outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightFrame',
        'params': params,
    }
    json = yield cmd_dict


def highlight_node(
        highlight_config: HighlightConfig,
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        selector: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights DOM node with given id or with the given JavaScript object wrapper. Either nodeId or
    objectId must be specified.

    :param highlight_config: A descriptor for the highlight appearance.
    :param node_id: *(Optional)* Identifier of the node to highlight.
    :param backend_node_id: *(Optional)* Identifier of the backend node to highlight.
    :param object_id: *(Optional)* JavaScript object id of the node to be highlighted.
    :param selector: *(Optional)* Selectors to highlight relevant nodes.
    '''
    params: T_JSON_DICT = dict()
    params['highlightConfig'] = highlight_config.to_json()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if selector is not None:
        params['selector'] = selector
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightNode',
        'params': params,
    }
    json = yield cmd_dict


def highlight_quad(
        quad: dom.Quad,
        color: typing.Optional[dom.RGBA] = None,
        outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights given quad. Coordinates are absolute with respect to the main frame viewport.

    :param quad: Quad to highlight
    :param color: *(Optional)* The highlight fill color (default: transparent).
    :param outline_color: *(Optional)* The highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['quad'] = quad.to_json()
    if color is not None:
        params['color'] = color.to_json()
    if outline_color is not None:
        params['outlineColor'] = outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightQuad',
        'params': params,
    }
    json = yield cmd_dict


def highlight_rect(
        x: int,
        y: int,
        width: int,
        height: int,
        color: typing.Optional[dom.RGBA] = None,
        outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights given rectangle. Coordinates are absolute with respect to the main frame viewport.

    :param x: X coordinate
    :param y: Y coordinate
    :param width: Rectangle width
    :param height: Rectangle height
    :param color: *(Optional)* The highlight fill color (default: transparent).
    :param outline_color: *(Optional)* The highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    params['width'] = width
    params['height'] = height
    if color is not None:
        params['color'] = color.to_json()
    if outline_color is not None:
        params['outlineColor'] = outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightRect',
        'params': params,
    }
    json = yield cmd_dict


def highlight_source_order(
        source_order_config: SourceOrderConfig,
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights the source order of the children of the DOM node with given id or with the given
    JavaScript object wrapper. Either nodeId or objectId must be specified.

    :param source_order_config: A descriptor for the appearance of the overlay drawing.
    :param node_id: *(Optional)* Identifier of the node to highlight.
    :param backend_node_id: *(Optional)* Identifier of the backend node to highlight.
    :param object_id: *(Optional)* JavaScript object id of the node to be highlighted.
    '''
    params: T_JSON_DICT = dict()
    params['sourceOrderConfig'] = source_order_config.to_json()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightSourceOrder',
        'params': params,
    }
    json = yield cmd_dict


def set_inspect_mode(
        mode: InspectMode,
        highlight_config: typing.Optional[HighlightConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enters the 'inspect' mode. In this mode, elements that user is hovering over are highlighted.
    Backend then generates 'inspectNodeRequested' event upon element selection.

    :param mode: Set an inspection mode.
    :param highlight_config: *(Optional)* A descriptor for the highlight appearance of hovered-over nodes. May be omitted if ```enabled == false```.
    '''
    params: T_JSON_DICT = dict()
    params['mode'] = mode.to_json()
    if highlight_config is not None:
        params['highlightConfig'] = highlight_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setInspectMode',
        'params': params,
    }
    json = yield cmd_dict


def set_show_ad_highlights(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights owner element of all frames detected to be ads.

    :param show: True for showing ad highlights
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowAdHighlights',
        'params': params,
    }
    json = yield cmd_dict


def set_paused_in_debugger_message(
        message: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param message: *(Optional)* The message to display, also triggers resume and step over controls.
    '''
    params: T_JSON_DICT = dict()
    if message is not None:
        params['message'] = message
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setPausedInDebuggerMessage',
        'params': params,
    }
    json = yield cmd_dict


def set_show_debug_borders(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows debug borders on layers

    :param show: True for showing debug borders
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowDebugBorders',
        'params': params,
    }
    json = yield cmd_dict


def set_show_fps_counter(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows the FPS counter

    :param show: True for showing the FPS counter
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowFPSCounter',
        'params': params,
    }
    json = yield cmd_dict


def set_show_grid_overlays(
        grid_node_highlight_configs: typing.List[GridNodeHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlight multiple elements with the CSS Grid overlay.

    :param grid_node_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['gridNodeHighlightConfigs'] = [i.to_json() for i in grid_node_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowGridOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_flex_overlays(
        flex_node_highlight_configs: typing.List[FlexNodeHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param flex_node_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['flexNodeHighlightConfigs'] = [i.to_json() for i in flex_node_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowFlexOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_scroll_snap_overlays(
        scroll_snap_highlight_configs: typing.List[ScrollSnapHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scroll_snap_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['scrollSnapHighlightConfigs'] = [i.to_json() for i in scroll_snap_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowScrollSnapOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_container_query_overlays(
        container_query_highlight_configs: typing.List[ContainerQueryHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param container_query_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['containerQueryHighlightConfigs'] = [i.to_json() for i in container_query_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowContainerQueryOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_paint_rects(
        result: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows paint rectangles

    :param result: True for showing paint rectangles
    '''
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowPaintRects',
        'params': params,
    }
    json = yield cmd_dict


def set_show_layout_shift_regions(
        result: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows layout shift regions

    :param result: True for showing layout shift regions
    '''
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowLayoutShiftRegions',
        'params': params,
    }
    json = yield cmd_dict


def set_show_scroll_bottleneck_rects(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows scroll bottleneck rects

    :param show: True for showing scroll bottleneck rects
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowScrollBottleneckRects',
        'params': params,
    }
    json = yield cmd_dict


def set_show_hit_test_borders(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deprecated, no longer has any effect.

    :param show: True for showing hit-test borders
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowHitTestBorders',
        'params': params,
    }
    json = yield cmd_dict


def set_show_web_vitals(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deprecated, no longer has any effect.

    :param show:
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowWebVitals',
        'params': params,
    }
    json = yield cmd_dict


def set_show_viewport_size_on_resize(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Paints viewport size upon main frame resize.

    :param show: Whether to paint size or not.
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowViewportSizeOnResize',
        'params': params,
    }
    json = yield cmd_dict


def set_show_hinge(
        hinge_config: typing.Optional[HingeConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Add a dual screen device hinge

    :param hinge_config: *(Optional)* hinge data, null means hideHinge
    '''
    params: T_JSON_DICT = dict()
    if hinge_config is not None:
        params['hingeConfig'] = hinge_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowHinge',
        'params': params,
    }
    json = yield cmd_dict


def set_show_isolated_elements(
        isolated_element_highlight_configs: typing.List[IsolatedElementHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Show elements in isolation mode with overlays.

    :param isolated_element_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['isolatedElementHighlightConfigs'] = [i.to_json() for i in isolated_element_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowIsolatedElements',
        'params': params,
    }
    json = yield cmd_dict


def set_show_window_controls_overlay(
        window_controls_overlay_config: typing.Optional[WindowControlsOverlayConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Show Window Controls Overlay for PWA

    :param window_controls_overlay_config: *(Optional)* Window Controls Overlay data, null means hide Window Controls Overlay
    '''
    params: T_JSON_DICT = dict()
    if window_controls_overlay_config is not None:
        params['windowControlsOverlayConfig'] = window_controls_overlay_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowWindowControlsOverlay',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Overlay.inspectNodeRequested')
@dataclass
class InspectNodeRequested:
    '''
    Fired when the node should be inspected. This happens after call to ``setInspectMode`` or when
    user manually inspects an element.
    '''
    #: Id of the node to inspect.
    backend_node_id: dom.BackendNodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectNodeRequested:
        return cls(
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId'])
        )


@event_class('Overlay.nodeHighlightRequested')
@dataclass
class NodeHighlightRequested:
    '''
    Fired when the node should be highlighted. This happens after call to ``setInspectMode``.
    '''
    node_id: dom.NodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeHighlightRequested:
        return cls(
            node_id=dom.NodeId.from_json(json['nodeId'])
        )


@event_class('Overlay.screenshotRequested')
@dataclass
class ScreenshotRequested:
    '''
    Fired when user asks to capture screenshot of some area on the page.
    '''
    #: Viewport to capture, in device independent pixels (dip).
    viewport: page.Viewport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreenshotRequested:
        return cls(
            viewport=page.Viewport.from_json(json['viewport'])
        )


@event_class('Overlay.inspectModeCanceled')
@dataclass
class InspectModeCanceled:
    '''
    Fired when user cancels the inspect mode.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectModeCanceled:
        return cls(

        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\overlay.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Overlay (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime


@dataclass
class SourceOrderConfig:
    '''
    Configuration data for drawing the source order of an elements children.
    '''
    #: the color to outline the given element in.
    parent_outline_color: dom.RGBA

    #: the color to outline the child elements in.
    child_outline_color: dom.RGBA

    def to_json(self):
        json = dict()
        json['parentOutlineColor'] = self.parent_outline_color.to_json()
        json['childOutlineColor'] = self.child_outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            parent_outline_color=dom.RGBA.from_json(json['parentOutlineColor']),
            child_outline_color=dom.RGBA.from_json(json['childOutlineColor']),
        )


@dataclass
class GridHighlightConfig:
    '''
    Configuration data for the highlighting of Grid elements.
    '''
    #: Whether the extension lines from grid cells to the rulers should be shown (default: false).
    show_grid_extension_lines: typing.Optional[bool] = None

    #: Show Positive line number labels (default: false).
    show_positive_line_numbers: typing.Optional[bool] = None

    #: Show Negative line number labels (default: false).
    show_negative_line_numbers: typing.Optional[bool] = None

    #: Show area name labels (default: false).
    show_area_names: typing.Optional[bool] = None

    #: Show line name labels (default: false).
    show_line_names: typing.Optional[bool] = None

    #: Show track size labels (default: false).
    show_track_sizes: typing.Optional[bool] = None

    #: The grid container border highlight color (default: transparent).
    grid_border_color: typing.Optional[dom.RGBA] = None

    #: The cell border color (default: transparent). Deprecated, please use rowLineColor and columnLineColor instead.
    cell_border_color: typing.Optional[dom.RGBA] = None

    #: The row line color (default: transparent).
    row_line_color: typing.Optional[dom.RGBA] = None

    #: The column line color (default: transparent).
    column_line_color: typing.Optional[dom.RGBA] = None

    #: Whether the grid border is dashed (default: false).
    grid_border_dash: typing.Optional[bool] = None

    #: Whether the cell border is dashed (default: false). Deprecated, please us rowLineDash and columnLineDash instead.
    cell_border_dash: typing.Optional[bool] = None

    #: Whether row lines are dashed (default: false).
    row_line_dash: typing.Optional[bool] = None

    #: Whether column lines are dashed (default: false).
    column_line_dash: typing.Optional[bool] = None

    #: The row gap highlight fill color (default: transparent).
    row_gap_color: typing.Optional[dom.RGBA] = None

    #: The row gap hatching fill color (default: transparent).
    row_hatch_color: typing.Optional[dom.RGBA] = None

    #: The column gap highlight fill color (default: transparent).
    column_gap_color: typing.Optional[dom.RGBA] = None

    #: The column gap hatching fill color (default: transparent).
    column_hatch_color: typing.Optional[dom.RGBA] = None

    #: The named grid areas border color (Default: transparent).
    area_border_color: typing.Optional[dom.RGBA] = None

    #: The grid container background color (Default: transparent).
    grid_background_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.show_grid_extension_lines is not None:
            json['showGridExtensionLines'] = self.show_grid_extension_lines
        if self.show_positive_line_numbers is not None:
            json['showPositiveLineNumbers'] = self.show_positive_line_numbers
        if self.show_negative_line_numbers is not None:
            json['showNegativeLineNumbers'] = self.show_negative_line_numbers
        if self.show_area_names is not None:
            json['showAreaNames'] = self.show_area_names
        if self.show_line_names is not None:
            json['showLineNames'] = self.show_line_names
        if self.show_track_sizes is not None:
            json['showTrackSizes'] = self.show_track_sizes
        if self.grid_border_color is not None:
            json['gridBorderColor'] = self.grid_border_color.to_json()
        if self.cell_border_color is not None:
            json['cellBorderColor'] = self.cell_border_color.to_json()
        if self.row_line_color is not None:
            json['rowLineColor'] = self.row_line_color.to_json()
        if self.column_line_color is not None:
            json['columnLineColor'] = self.column_line_color.to_json()
        if self.grid_border_dash is not None:
            json['gridBorderDash'] = self.grid_border_dash
        if self.cell_border_dash is not None:
            json['cellBorderDash'] = self.cell_border_dash
        if self.row_line_dash is not None:
            json['rowLineDash'] = self.row_line_dash
        if self.column_line_dash is not None:
            json['columnLineDash'] = self.column_line_dash
        if self.row_gap_color is not None:
            json['rowGapColor'] = self.row_gap_color.to_json()
        if self.row_hatch_color is not None:
            json['rowHatchColor'] = self.row_hatch_color.to_json()
        if self.column_gap_color is not None:
            json['columnGapColor'] = self.column_gap_color.to_json()
        if self.column_hatch_color is not None:
            json['columnHatchColor'] = self.column_hatch_color.to_json()
        if self.area_border_color is not None:
            json['areaBorderColor'] = self.area_border_color.to_json()
        if self.grid_background_color is not None:
            json['gridBackgroundColor'] = self.grid_background_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_grid_extension_lines=bool(json['showGridExtensionLines']) if 'showGridExtensionLines' in json else None,
            show_positive_line_numbers=bool(json['showPositiveLineNumbers']) if 'showPositiveLineNumbers' in json else None,
            show_negative_line_numbers=bool(json['showNegativeLineNumbers']) if 'showNegativeLineNumbers' in json else None,
            show_area_names=bool(json['showAreaNames']) if 'showAreaNames' in json else None,
            show_line_names=bool(json['showLineNames']) if 'showLineNames' in json else None,
            show_track_sizes=bool(json['showTrackSizes']) if 'showTrackSizes' in json else None,
            grid_border_color=dom.RGBA.from_json(json['gridBorderColor']) if 'gridBorderColor' in json else None,
            cell_border_color=dom.RGBA.from_json(json['cellBorderColor']) if 'cellBorderColor' in json else None,
            row_line_color=dom.RGBA.from_json(json['rowLineColor']) if 'rowLineColor' in json else None,
            column_line_color=dom.RGBA.from_json(json['columnLineColor']) if 'columnLineColor' in json else None,
            grid_border_dash=bool(json['gridBorderDash']) if 'gridBorderDash' in json else None,
            cell_border_dash=bool(json['cellBorderDash']) if 'cellBorderDash' in json else None,
            row_line_dash=bool(json['rowLineDash']) if 'rowLineDash' in json else None,
            column_line_dash=bool(json['columnLineDash']) if 'columnLineDash' in json else None,
            row_gap_color=dom.RGBA.from_json(json['rowGapColor']) if 'rowGapColor' in json else None,
            row_hatch_color=dom.RGBA.from_json(json['rowHatchColor']) if 'rowHatchColor' in json else None,
            column_gap_color=dom.RGBA.from_json(json['columnGapColor']) if 'columnGapColor' in json else None,
            column_hatch_color=dom.RGBA.from_json(json['columnHatchColor']) if 'columnHatchColor' in json else None,
            area_border_color=dom.RGBA.from_json(json['areaBorderColor']) if 'areaBorderColor' in json else None,
            grid_background_color=dom.RGBA.from_json(json['gridBackgroundColor']) if 'gridBackgroundColor' in json else None,
        )


@dataclass
class FlexContainerHighlightConfig:
    '''
    Configuration data for the highlighting of Flex container elements.
    '''
    #: The style of the container border
    container_border: typing.Optional[LineStyle] = None

    #: The style of the separator between lines
    line_separator: typing.Optional[LineStyle] = None

    #: The style of the separator between items
    item_separator: typing.Optional[LineStyle] = None

    #: Style of content-distribution space on the main axis (justify-content).
    main_distributed_space: typing.Optional[BoxStyle] = None

    #: Style of content-distribution space on the cross axis (align-content).
    cross_distributed_space: typing.Optional[BoxStyle] = None

    #: Style of empty space caused by row gaps (gap/row-gap).
    row_gap_space: typing.Optional[BoxStyle] = None

    #: Style of empty space caused by columns gaps (gap/column-gap).
    column_gap_space: typing.Optional[BoxStyle] = None

    #: Style of the self-alignment line (align-items).
    cross_alignment: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.line_separator is not None:
            json['lineSeparator'] = self.line_separator.to_json()
        if self.item_separator is not None:
            json['itemSeparator'] = self.item_separator.to_json()
        if self.main_distributed_space is not None:
            json['mainDistributedSpace'] = self.main_distributed_space.to_json()
        if self.cross_distributed_space is not None:
            json['crossDistributedSpace'] = self.cross_distributed_space.to_json()
        if self.row_gap_space is not None:
            json['rowGapSpace'] = self.row_gap_space.to_json()
        if self.column_gap_space is not None:
            json['columnGapSpace'] = self.column_gap_space.to_json()
        if self.cross_alignment is not None:
            json['crossAlignment'] = self.cross_alignment.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None,
            line_separator=LineStyle.from_json(json['lineSeparator']) if 'lineSeparator' in json else None,
            item_separator=LineStyle.from_json(json['itemSeparator']) if 'itemSeparator' in json else None,
            main_distributed_space=BoxStyle.from_json(json['mainDistributedSpace']) if 'mainDistributedSpace' in json else None,
            cross_distributed_space=BoxStyle.from_json(json['crossDistributedSpace']) if 'crossDistributedSpace' in json else None,
            row_gap_space=BoxStyle.from_json(json['rowGapSpace']) if 'rowGapSpace' in json else None,
            column_gap_space=BoxStyle.from_json(json['columnGapSpace']) if 'columnGapSpace' in json else None,
            cross_alignment=LineStyle.from_json(json['crossAlignment']) if 'crossAlignment' in json else None,
        )


@dataclass
class FlexItemHighlightConfig:
    '''
    Configuration data for the highlighting of Flex item elements.
    '''
    #: Style of the box representing the item's base size
    base_size_box: typing.Optional[BoxStyle] = None

    #: Style of the border around the box representing the item's base size
    base_size_border: typing.Optional[LineStyle] = None

    #: Style of the arrow representing if the item grew or shrank
    flexibility_arrow: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.base_size_box is not None:
            json['baseSizeBox'] = self.base_size_box.to_json()
        if self.base_size_border is not None:
            json['baseSizeBorder'] = self.base_size_border.to_json()
        if self.flexibility_arrow is not None:
            json['flexibilityArrow'] = self.flexibility_arrow.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            base_size_box=BoxStyle.from_json(json['baseSizeBox']) if 'baseSizeBox' in json else None,
            base_size_border=LineStyle.from_json(json['baseSizeBorder']) if 'baseSizeBorder' in json else None,
            flexibility_arrow=LineStyle.from_json(json['flexibilityArrow']) if 'flexibilityArrow' in json else None,
        )


@dataclass
class LineStyle:
    '''
    Style information for drawing a line.
    '''
    #: The color of the line (default: transparent)
    color: typing.Optional[dom.RGBA] = None

    #: The line pattern (default: solid)
    pattern: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.color is not None:
            json['color'] = self.color.to_json()
        if self.pattern is not None:
            json['pattern'] = self.pattern
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            color=dom.RGBA.from_json(json['color']) if 'color' in json else None,
            pattern=str(json['pattern']) if 'pattern' in json else None,
        )


@dataclass
class BoxStyle:
    '''
    Style information for drawing a box.
    '''
    #: The background color for the box (default: transparent)
    fill_color: typing.Optional[dom.RGBA] = None

    #: The hatching color for the box (default: transparent)
    hatch_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.fill_color is not None:
            json['fillColor'] = self.fill_color.to_json()
        if self.hatch_color is not None:
            json['hatchColor'] = self.hatch_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fill_color=dom.RGBA.from_json(json['fillColor']) if 'fillColor' in json else None,
            hatch_color=dom.RGBA.from_json(json['hatchColor']) if 'hatchColor' in json else None,
        )


class ContrastAlgorithm(enum.Enum):
    AA = "aa"
    AAA = "aaa"
    APCA = "apca"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class HighlightConfig:
    '''
    Configuration data for the highlighting of page elements.
    '''
    #: Whether the node info tooltip should be shown (default: false).
    show_info: typing.Optional[bool] = None

    #: Whether the node styles in the tooltip (default: false).
    show_styles: typing.Optional[bool] = None

    #: Whether the rulers should be shown (default: false).
    show_rulers: typing.Optional[bool] = None

    #: Whether the a11y info should be shown (default: true).
    show_accessibility_info: typing.Optional[bool] = None

    #: Whether the extension lines from node to the rulers should be shown (default: false).
    show_extension_lines: typing.Optional[bool] = None

    #: The content box highlight fill color (default: transparent).
    content_color: typing.Optional[dom.RGBA] = None

    #: The padding highlight fill color (default: transparent).
    padding_color: typing.Optional[dom.RGBA] = None

    #: The border highlight fill color (default: transparent).
    border_color: typing.Optional[dom.RGBA] = None

    #: The margin highlight fill color (default: transparent).
    margin_color: typing.Optional[dom.RGBA] = None

    #: The event target element highlight fill color (default: transparent).
    event_target_color: typing.Optional[dom.RGBA] = None

    #: The shape outside fill color (default: transparent).
    shape_color: typing.Optional[dom.RGBA] = None

    #: The shape margin fill color (default: transparent).
    shape_margin_color: typing.Optional[dom.RGBA] = None

    #: The grid layout color (default: transparent).
    css_grid_color: typing.Optional[dom.RGBA] = None

    #: The color format used to format color styles (default: hex).
    color_format: typing.Optional[ColorFormat] = None

    #: The grid layout highlight configuration (default: all transparent).
    grid_highlight_config: typing.Optional[GridHighlightConfig] = None

    #: The flex container highlight configuration (default: all transparent).
    flex_container_highlight_config: typing.Optional[FlexContainerHighlightConfig] = None

    #: The flex item highlight configuration (default: all transparent).
    flex_item_highlight_config: typing.Optional[FlexItemHighlightConfig] = None

    #: The contrast algorithm to use for the contrast ratio (default: aa).
    contrast_algorithm: typing.Optional[ContrastAlgorithm] = None

    #: The container query container highlight configuration (default: all transparent).
    container_query_container_highlight_config: typing.Optional[ContainerQueryContainerHighlightConfig] = None

    def to_json(self):
        json = dict()
        if self.show_info is not None:
            json['showInfo'] = self.show_info
        if self.show_styles is not None:
            json['showStyles'] = self.show_styles
        if self.show_rulers is not None:
            json['showRulers'] = self.show_rulers
        if self.show_accessibility_info is not None:
            json['showAccessibilityInfo'] = self.show_accessibility_info
        if self.show_extension_lines is not None:
            json['showExtensionLines'] = self.show_extension_lines
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.padding_color is not None:
            json['paddingColor'] = self.padding_color.to_json()
        if self.border_color is not None:
            json['borderColor'] = self.border_color.to_json()
        if self.margin_color is not None:
            json['marginColor'] = self.margin_color.to_json()
        if self.event_target_color is not None:
            json['eventTargetColor'] = self.event_target_color.to_json()
        if self.shape_color is not None:
            json['shapeColor'] = self.shape_color.to_json()
        if self.shape_margin_color is not None:
            json['shapeMarginColor'] = self.shape_margin_color.to_json()
        if self.css_grid_color is not None:
            json['cssGridColor'] = self.css_grid_color.to_json()
        if self.color_format is not None:
            json['colorFormat'] = self.color_format.to_json()
        if self.grid_highlight_config is not None:
            json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        if self.flex_container_highlight_config is not None:
            json['flexContainerHighlightConfig'] = self.flex_container_highlight_config.to_json()
        if self.flex_item_highlight_config is not None:
            json['flexItemHighlightConfig'] = self.flex_item_highlight_config.to_json()
        if self.contrast_algorithm is not None:
            json['contrastAlgorithm'] = self.contrast_algorithm.to_json()
        if self.container_query_container_highlight_config is not None:
            json['containerQueryContainerHighlightConfig'] = self.container_query_container_highlight_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_info=bool(json['showInfo']) if 'showInfo' in json else None,
            show_styles=bool(json['showStyles']) if 'showStyles' in json else None,
            show_rulers=bool(json['showRulers']) if 'showRulers' in json else None,
            show_accessibility_info=bool(json['showAccessibilityInfo']) if 'showAccessibilityInfo' in json else None,
            show_extension_lines=bool(json['showExtensionLines']) if 'showExtensionLines' in json else None,
            content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None,
            padding_color=dom.RGBA.from_json(json['paddingColor']) if 'paddingColor' in json else None,
            border_color=dom.RGBA.from_json(json['borderColor']) if 'borderColor' in json else None,
            margin_color=dom.RGBA.from_json(json['marginColor']) if 'marginColor' in json else None,
            event_target_color=dom.RGBA.from_json(json['eventTargetColor']) if 'eventTargetColor' in json else None,
            shape_color=dom.RGBA.from_json(json['shapeColor']) if 'shapeColor' in json else None,
            shape_margin_color=dom.RGBA.from_json(json['shapeMarginColor']) if 'shapeMarginColor' in json else None,
            css_grid_color=dom.RGBA.from_json(json['cssGridColor']) if 'cssGridColor' in json else None,
            color_format=ColorFormat.from_json(json['colorFormat']) if 'colorFormat' in json else None,
            grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']) if 'gridHighlightConfig' in json else None,
            flex_container_highlight_config=FlexContainerHighlightConfig.from_json(json['flexContainerHighlightConfig']) if 'flexContainerHighlightConfig' in json else None,
            flex_item_highlight_config=FlexItemHighlightConfig.from_json(json['flexItemHighlightConfig']) if 'flexItemHighlightConfig' in json else None,
            contrast_algorithm=ContrastAlgorithm.from_json(json['contrastAlgorithm']) if 'contrastAlgorithm' in json else None,
            container_query_container_highlight_config=ContainerQueryContainerHighlightConfig.from_json(json['containerQueryContainerHighlightConfig']) if 'containerQueryContainerHighlightConfig' in json else None,
        )


class ColorFormat(enum.Enum):
    RGB = "rgb"
    HSL = "hsl"
    HWB = "hwb"
    HEX_ = "hex"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class GridNodeHighlightConfig:
    '''
    Configurations for Persistent Grid Highlight
    '''
    #: A descriptor for the highlight appearance.
    grid_highlight_config: GridHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class FlexNodeHighlightConfig:
    #: A descriptor for the highlight appearance of flex containers.
    flex_container_highlight_config: FlexContainerHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['flexContainerHighlightConfig'] = self.flex_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            flex_container_highlight_config=FlexContainerHighlightConfig.from_json(json['flexContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class ScrollSnapContainerHighlightConfig:
    #: The style of the snapport border (default: transparent)
    snapport_border: typing.Optional[LineStyle] = None

    #: The style of the snap area border (default: transparent)
    snap_area_border: typing.Optional[LineStyle] = None

    #: The margin highlight fill color (default: transparent).
    scroll_margin_color: typing.Optional[dom.RGBA] = None

    #: The padding highlight fill color (default: transparent).
    scroll_padding_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.snapport_border is not None:
            json['snapportBorder'] = self.snapport_border.to_json()
        if self.snap_area_border is not None:
            json['snapAreaBorder'] = self.snap_area_border.to_json()
        if self.scroll_margin_color is not None:
            json['scrollMarginColor'] = self.scroll_margin_color.to_json()
        if self.scroll_padding_color is not None:
            json['scrollPaddingColor'] = self.scroll_padding_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            snapport_border=LineStyle.from_json(json['snapportBorder']) if 'snapportBorder' in json else None,
            snap_area_border=LineStyle.from_json(json['snapAreaBorder']) if 'snapAreaBorder' in json else None,
            scroll_margin_color=dom.RGBA.from_json(json['scrollMarginColor']) if 'scrollMarginColor' in json else None,
            scroll_padding_color=dom.RGBA.from_json(json['scrollPaddingColor']) if 'scrollPaddingColor' in json else None,
        )


@dataclass
class ScrollSnapHighlightConfig:
    #: A descriptor for the highlight appearance of scroll snap containers.
    scroll_snap_container_highlight_config: ScrollSnapContainerHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['scrollSnapContainerHighlightConfig'] = self.scroll_snap_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            scroll_snap_container_highlight_config=ScrollSnapContainerHighlightConfig.from_json(json['scrollSnapContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class HingeConfig:
    '''
    Configuration for dual screen hinge
    '''
    #: A rectangle represent hinge
    rect: dom.Rect

    #: The content box highlight fill color (default: a dark color).
    content_color: typing.Optional[dom.RGBA] = None

    #: The content box highlight outline color (default: transparent).
    outline_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        json['rect'] = self.rect.to_json()
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.outline_color is not None:
            json['outlineColor'] = self.outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            rect=dom.Rect.from_json(json['rect']),
            content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None,
            outline_color=dom.RGBA.from_json(json['outlineColor']) if 'outlineColor' in json else None,
        )


@dataclass
class WindowControlsOverlayConfig:
    '''
    Configuration for Window Controls Overlay
    '''
    #: Whether the title bar CSS should be shown when emulating the Window Controls Overlay.
    show_css: bool

    #: Selected platforms to show the overlay.
    selected_platform: str

    #: The theme color defined in app manifest.
    theme_color: str

    def to_json(self):
        json = dict()
        json['showCSS'] = self.show_css
        json['selectedPlatform'] = self.selected_platform
        json['themeColor'] = self.theme_color
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_css=bool(json['showCSS']),
            selected_platform=str(json['selectedPlatform']),
            theme_color=str(json['themeColor']),
        )


@dataclass
class ContainerQueryHighlightConfig:
    #: A descriptor for the highlight appearance of container query containers.
    container_query_container_highlight_config: ContainerQueryContainerHighlightConfig

    #: Identifier of the container node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['containerQueryContainerHighlightConfig'] = self.container_query_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_query_container_highlight_config=ContainerQueryContainerHighlightConfig.from_json(json['containerQueryContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class ContainerQueryContainerHighlightConfig:
    #: The style of the container border.
    container_border: typing.Optional[LineStyle] = None

    #: The style of the descendants' borders.
    descendant_border: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.descendant_border is not None:
            json['descendantBorder'] = self.descendant_border.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None,
            descendant_border=LineStyle.from_json(json['descendantBorder']) if 'descendantBorder' in json else None,
        )


@dataclass
class IsolatedElementHighlightConfig:
    #: A descriptor for the highlight appearance of an element in isolation mode.
    isolation_mode_highlight_config: IsolationModeHighlightConfig

    #: Identifier of the isolated element to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['isolationModeHighlightConfig'] = self.isolation_mode_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            isolation_mode_highlight_config=IsolationModeHighlightConfig.from_json(json['isolationModeHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class IsolationModeHighlightConfig:
    #: The fill color of the resizers (default: transparent).
    resizer_color: typing.Optional[dom.RGBA] = None

    #: The fill color for resizer handles (default: transparent).
    resizer_handle_color: typing.Optional[dom.RGBA] = None

    #: The fill color for the mask covering non-isolated elements (default: transparent).
    mask_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.resizer_color is not None:
            json['resizerColor'] = self.resizer_color.to_json()
        if self.resizer_handle_color is not None:
            json['resizerHandleColor'] = self.resizer_handle_color.to_json()
        if self.mask_color is not None:
            json['maskColor'] = self.mask_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            resizer_color=dom.RGBA.from_json(json['resizerColor']) if 'resizerColor' in json else None,
            resizer_handle_color=dom.RGBA.from_json(json['resizerHandleColor']) if 'resizerHandleColor' in json else None,
            mask_color=dom.RGBA.from_json(json['maskColor']) if 'maskColor' in json else None,
        )


class InspectMode(enum.Enum):
    SEARCH_FOR_NODE = "searchForNode"
    SEARCH_FOR_UA_SHADOW_DOM = "searchForUAShadowDOM"
    CAPTURE_AREA_SCREENSHOT = "captureAreaScreenshot"
    SHOW_DISTANCES = "showDistances"
    NONE = "none"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.enable',
    }
    json = yield cmd_dict


def get_highlight_object_for_test(
        node_id: dom.NodeId,
        include_distance: typing.Optional[bool] = None,
        include_style: typing.Optional[bool] = None,
        color_format: typing.Optional[ColorFormat] = None,
        show_accessibility_info: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For testing.

    :param node_id: Id of the node to get highlight object for.
    :param include_distance: *(Optional)* Whether to include distance info.
    :param include_style: *(Optional)* Whether to include style info.
    :param color_format: *(Optional)* The color format to get config with (default: hex).
    :param show_accessibility_info: *(Optional)* Whether to show accessibility info (default: true).
    :returns: Highlight data for the node.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    if include_distance is not None:
        params['includeDistance'] = include_distance
    if include_style is not None:
        params['includeStyle'] = include_style
    if color_format is not None:
        params['colorFormat'] = color_format.to_json()
    if show_accessibility_info is not None:
        params['showAccessibilityInfo'] = show_accessibility_info
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getHighlightObjectForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlight'])


def get_grid_highlight_objects_for_test(
        node_ids: typing.List[dom.NodeId]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For Persistent Grid testing.

    :param node_ids: Ids of the node to get highlight object for.
    :returns: Grid Highlight data for the node ids provided.
    '''
    params: T_JSON_DICT = dict()
    params['nodeIds'] = [i.to_json() for i in node_ids]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getGridHighlightObjectsForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlights'])


def get_source_order_highlight_object_for_test(
        node_id: dom.NodeId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For Source Order Viewer testing.

    :param node_id: Id of the node to highlight.
    :returns: Source order highlight data for the node id provided.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getSourceOrderHighlightObjectForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlight'])


def hide_highlight() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Hides any highlight.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.hideHighlight',
    }
    json = yield cmd_dict


def highlight_frame(
        frame_id: page.FrameId,
        content_color: typing.Optional[dom.RGBA] = None,
        content_outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights owner element of the frame with given id.
    Deprecated: Doesn't work reliably and cannot be fixed due to process
    separation (the owner node might be in a different process). Determine
    the owner node in the client and use highlightNode.

    :param frame_id: Identifier of the frame to highlight.
    :param content_color: *(Optional)* The content box highlight fill color (default: transparent).
    :param content_outline_color: *(Optional)* The content box highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    if content_color is not None:
        params['contentColor'] = content_color.to_json()
    if content_outline_color is not None:
        params['contentOutlineColor'] = content_outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightFrame',
        'params': params,
    }
    json = yield cmd_dict


def highlight_node(
        highlight_config: HighlightConfig,
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        selector: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights DOM node with given id or with the given JavaScript object wrapper. Either nodeId or
    objectId must be specified.

    :param highlight_config: A descriptor for the highlight appearance.
    :param node_id: *(Optional)* Identifier of the node to highlight.
    :param backend_node_id: *(Optional)* Identifier of the backend node to highlight.
    :param object_id: *(Optional)* JavaScript object id of the node to be highlighted.
    :param selector: *(Optional)* Selectors to highlight relevant nodes.
    '''
    params: T_JSON_DICT = dict()
    params['highlightConfig'] = highlight_config.to_json()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if selector is not None:
        params['selector'] = selector
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightNode',
        'params': params,
    }
    json = yield cmd_dict


def highlight_quad(
        quad: dom.Quad,
        color: typing.Optional[dom.RGBA] = None,
        outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights given quad. Coordinates are absolute with respect to the main frame viewport.

    :param quad: Quad to highlight
    :param color: *(Optional)* The highlight fill color (default: transparent).
    :param outline_color: *(Optional)* The highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['quad'] = quad.to_json()
    if color is not None:
        params['color'] = color.to_json()
    if outline_color is not None:
        params['outlineColor'] = outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightQuad',
        'params': params,
    }
    json = yield cmd_dict


def highlight_rect(
        x: int,
        y: int,
        width: int,
        height: int,
        color: typing.Optional[dom.RGBA] = None,
        outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights given rectangle. Coordinates are absolute with respect to the main frame viewport.

    :param x: X coordinate
    :param y: Y coordinate
    :param width: Rectangle width
    :param height: Rectangle height
    :param color: *(Optional)* The highlight fill color (default: transparent).
    :param outline_color: *(Optional)* The highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    params['width'] = width
    params['height'] = height
    if color is not None:
        params['color'] = color.to_json()
    if outline_color is not None:
        params['outlineColor'] = outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightRect',
        'params': params,
    }
    json = yield cmd_dict


def highlight_source_order(
        source_order_config: SourceOrderConfig,
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights the source order of the children of the DOM node with given id or with the given
    JavaScript object wrapper. Either nodeId or objectId must be specified.

    :param source_order_config: A descriptor for the appearance of the overlay drawing.
    :param node_id: *(Optional)* Identifier of the node to highlight.
    :param backend_node_id: *(Optional)* Identifier of the backend node to highlight.
    :param object_id: *(Optional)* JavaScript object id of the node to be highlighted.
    '''
    params: T_JSON_DICT = dict()
    params['sourceOrderConfig'] = source_order_config.to_json()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightSourceOrder',
        'params': params,
    }
    json = yield cmd_dict


def set_inspect_mode(
        mode: InspectMode,
        highlight_config: typing.Optional[HighlightConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enters the 'inspect' mode. In this mode, elements that user is hovering over are highlighted.
    Backend then generates 'inspectNodeRequested' event upon element selection.

    :param mode: Set an inspection mode.
    :param highlight_config: *(Optional)* A descriptor for the highlight appearance of hovered-over nodes. May be omitted if ```enabled == false```.
    '''
    params: T_JSON_DICT = dict()
    params['mode'] = mode.to_json()
    if highlight_config is not None:
        params['highlightConfig'] = highlight_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setInspectMode',
        'params': params,
    }
    json = yield cmd_dict


def set_show_ad_highlights(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights owner element of all frames detected to be ads.

    :param show: True for showing ad highlights
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowAdHighlights',
        'params': params,
    }
    json = yield cmd_dict


def set_paused_in_debugger_message(
        message: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param message: *(Optional)* The message to display, also triggers resume and step over controls.
    '''
    params: T_JSON_DICT = dict()
    if message is not None:
        params['message'] = message
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setPausedInDebuggerMessage',
        'params': params,
    }
    json = yield cmd_dict


def set_show_debug_borders(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows debug borders on layers

    :param show: True for showing debug borders
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowDebugBorders',
        'params': params,
    }
    json = yield cmd_dict


def set_show_fps_counter(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows the FPS counter

    :param show: True for showing the FPS counter
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowFPSCounter',
        'params': params,
    }
    json = yield cmd_dict


def set_show_grid_overlays(
        grid_node_highlight_configs: typing.List[GridNodeHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlight multiple elements with the CSS Grid overlay.

    :param grid_node_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['gridNodeHighlightConfigs'] = [i.to_json() for i in grid_node_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowGridOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_flex_overlays(
        flex_node_highlight_configs: typing.List[FlexNodeHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param flex_node_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['flexNodeHighlightConfigs'] = [i.to_json() for i in flex_node_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowFlexOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_scroll_snap_overlays(
        scroll_snap_highlight_configs: typing.List[ScrollSnapHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scroll_snap_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['scrollSnapHighlightConfigs'] = [i.to_json() for i in scroll_snap_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowScrollSnapOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_container_query_overlays(
        container_query_highlight_configs: typing.List[ContainerQueryHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param container_query_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['containerQueryHighlightConfigs'] = [i.to_json() for i in container_query_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowContainerQueryOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_paint_rects(
        result: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows paint rectangles

    :param result: True for showing paint rectangles
    '''
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowPaintRects',
        'params': params,
    }
    json = yield cmd_dict


def set_show_layout_shift_regions(
        result: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows layout shift regions

    :param result: True for showing layout shift regions
    '''
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowLayoutShiftRegions',
        'params': params,
    }
    json = yield cmd_dict


def set_show_scroll_bottleneck_rects(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows scroll bottleneck rects

    :param show: True for showing scroll bottleneck rects
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowScrollBottleneckRects',
        'params': params,
    }
    json = yield cmd_dict


def set_show_hit_test_borders(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deprecated, no longer has any effect.

    :param show: True for showing hit-test borders
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowHitTestBorders',
        'params': params,
    }
    json = yield cmd_dict


def set_show_web_vitals(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deprecated, no longer has any effect.

    :param show:
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowWebVitals',
        'params': params,
    }
    json = yield cmd_dict


def set_show_viewport_size_on_resize(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Paints viewport size upon main frame resize.

    :param show: Whether to paint size or not.
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowViewportSizeOnResize',
        'params': params,
    }
    json = yield cmd_dict


def set_show_hinge(
        hinge_config: typing.Optional[HingeConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Add a dual screen device hinge

    :param hinge_config: *(Optional)* hinge data, null means hideHinge
    '''
    params: T_JSON_DICT = dict()
    if hinge_config is not None:
        params['hingeConfig'] = hinge_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowHinge',
        'params': params,
    }
    json = yield cmd_dict


def set_show_isolated_elements(
        isolated_element_highlight_configs: typing.List[IsolatedElementHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Show elements in isolation mode with overlays.

    :param isolated_element_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['isolatedElementHighlightConfigs'] = [i.to_json() for i in isolated_element_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowIsolatedElements',
        'params': params,
    }
    json = yield cmd_dict


def set_show_window_controls_overlay(
        window_controls_overlay_config: typing.Optional[WindowControlsOverlayConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Show Window Controls Overlay for PWA

    :param window_controls_overlay_config: *(Optional)* Window Controls Overlay data, null means hide Window Controls Overlay
    '''
    params: T_JSON_DICT = dict()
    if window_controls_overlay_config is not None:
        params['windowControlsOverlayConfig'] = window_controls_overlay_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowWindowControlsOverlay',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Overlay.inspectNodeRequested')
@dataclass
class InspectNodeRequested:
    '''
    Fired when the node should be inspected. This happens after call to ``setInspectMode`` or when
    user manually inspects an element.
    '''
    #: Id of the node to inspect.
    backend_node_id: dom.BackendNodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectNodeRequested:
        return cls(
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId'])
        )


@event_class('Overlay.nodeHighlightRequested')
@dataclass
class NodeHighlightRequested:
    '''
    Fired when the node should be highlighted. This happens after call to ``setInspectMode``.
    '''
    node_id: dom.NodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeHighlightRequested:
        return cls(
            node_id=dom.NodeId.from_json(json['nodeId'])
        )


@event_class('Overlay.screenshotRequested')
@dataclass
class ScreenshotRequested:
    '''
    Fired when user asks to capture screenshot of some area on the page.
    '''
    #: Viewport to capture, in device independent pixels (dip).
    viewport: page.Viewport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreenshotRequested:
        return cls(
            viewport=page.Viewport.from_json(json['viewport'])
        )


@event_class('Overlay.inspectModeCanceled')
@dataclass
class InspectModeCanceled:
    '''
    Fired when user cancels the inspect mode.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectModeCanceled:
        return cls(

        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\overlay.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Overlay (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime


@dataclass
class SourceOrderConfig:
    '''
    Configuration data for drawing the source order of an elements children.
    '''
    #: the color to outline the given element in.
    parent_outline_color: dom.RGBA

    #: the color to outline the child elements in.
    child_outline_color: dom.RGBA

    def to_json(self):
        json = dict()
        json['parentOutlineColor'] = self.parent_outline_color.to_json()
        json['childOutlineColor'] = self.child_outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            parent_outline_color=dom.RGBA.from_json(json['parentOutlineColor']),
            child_outline_color=dom.RGBA.from_json(json['childOutlineColor']),
        )


@dataclass
class GridHighlightConfig:
    '''
    Configuration data for the highlighting of Grid elements.
    '''
    #: Whether the extension lines from grid cells to the rulers should be shown (default: false).
    show_grid_extension_lines: typing.Optional[bool] = None

    #: Show Positive line number labels (default: false).
    show_positive_line_numbers: typing.Optional[bool] = None

    #: Show Negative line number labels (default: false).
    show_negative_line_numbers: typing.Optional[bool] = None

    #: Show area name labels (default: false).
    show_area_names: typing.Optional[bool] = None

    #: Show line name labels (default: false).
    show_line_names: typing.Optional[bool] = None

    #: Show track size labels (default: false).
    show_track_sizes: typing.Optional[bool] = None

    #: The grid container border highlight color (default: transparent).
    grid_border_color: typing.Optional[dom.RGBA] = None

    #: The cell border color (default: transparent). Deprecated, please use rowLineColor and columnLineColor instead.
    cell_border_color: typing.Optional[dom.RGBA] = None

    #: The row line color (default: transparent).
    row_line_color: typing.Optional[dom.RGBA] = None

    #: The column line color (default: transparent).
    column_line_color: typing.Optional[dom.RGBA] = None

    #: Whether the grid border is dashed (default: false).
    grid_border_dash: typing.Optional[bool] = None

    #: Whether the cell border is dashed (default: false). Deprecated, please us rowLineDash and columnLineDash instead.
    cell_border_dash: typing.Optional[bool] = None

    #: Whether row lines are dashed (default: false).
    row_line_dash: typing.Optional[bool] = None

    #: Whether column lines are dashed (default: false).
    column_line_dash: typing.Optional[bool] = None

    #: The row gap highlight fill color (default: transparent).
    row_gap_color: typing.Optional[dom.RGBA] = None

    #: The row gap hatching fill color (default: transparent).
    row_hatch_color: typing.Optional[dom.RGBA] = None

    #: The column gap highlight fill color (default: transparent).
    column_gap_color: typing.Optional[dom.RGBA] = None

    #: The column gap hatching fill color (default: transparent).
    column_hatch_color: typing.Optional[dom.RGBA] = None

    #: The named grid areas border color (Default: transparent).
    area_border_color: typing.Optional[dom.RGBA] = None

    #: The grid container background color (Default: transparent).
    grid_background_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.show_grid_extension_lines is not None:
            json['showGridExtensionLines'] = self.show_grid_extension_lines
        if self.show_positive_line_numbers is not None:
            json['showPositiveLineNumbers'] = self.show_positive_line_numbers
        if self.show_negative_line_numbers is not None:
            json['showNegativeLineNumbers'] = self.show_negative_line_numbers
        if self.show_area_names is not None:
            json['showAreaNames'] = self.show_area_names
        if self.show_line_names is not None:
            json['showLineNames'] = self.show_line_names
        if self.show_track_sizes is not None:
            json['showTrackSizes'] = self.show_track_sizes
        if self.grid_border_color is not None:
            json['gridBorderColor'] = self.grid_border_color.to_json()
        if self.cell_border_color is not None:
            json['cellBorderColor'] = self.cell_border_color.to_json()
        if self.row_line_color is not None:
            json['rowLineColor'] = self.row_line_color.to_json()
        if self.column_line_color is not None:
            json['columnLineColor'] = self.column_line_color.to_json()
        if self.grid_border_dash is not None:
            json['gridBorderDash'] = self.grid_border_dash
        if self.cell_border_dash is not None:
            json['cellBorderDash'] = self.cell_border_dash
        if self.row_line_dash is not None:
            json['rowLineDash'] = self.row_line_dash
        if self.column_line_dash is not None:
            json['columnLineDash'] = self.column_line_dash
        if self.row_gap_color is not None:
            json['rowGapColor'] = self.row_gap_color.to_json()
        if self.row_hatch_color is not None:
            json['rowHatchColor'] = self.row_hatch_color.to_json()
        if self.column_gap_color is not None:
            json['columnGapColor'] = self.column_gap_color.to_json()
        if self.column_hatch_color is not None:
            json['columnHatchColor'] = self.column_hatch_color.to_json()
        if self.area_border_color is not None:
            json['areaBorderColor'] = self.area_border_color.to_json()
        if self.grid_background_color is not None:
            json['gridBackgroundColor'] = self.grid_background_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_grid_extension_lines=bool(json['showGridExtensionLines']) if 'showGridExtensionLines' in json else None,
            show_positive_line_numbers=bool(json['showPositiveLineNumbers']) if 'showPositiveLineNumbers' in json else None,
            show_negative_line_numbers=bool(json['showNegativeLineNumbers']) if 'showNegativeLineNumbers' in json else None,
            show_area_names=bool(json['showAreaNames']) if 'showAreaNames' in json else None,
            show_line_names=bool(json['showLineNames']) if 'showLineNames' in json else None,
            show_track_sizes=bool(json['showTrackSizes']) if 'showTrackSizes' in json else None,
            grid_border_color=dom.RGBA.from_json(json['gridBorderColor']) if 'gridBorderColor' in json else None,
            cell_border_color=dom.RGBA.from_json(json['cellBorderColor']) if 'cellBorderColor' in json else None,
            row_line_color=dom.RGBA.from_json(json['rowLineColor']) if 'rowLineColor' in json else None,
            column_line_color=dom.RGBA.from_json(json['columnLineColor']) if 'columnLineColor' in json else None,
            grid_border_dash=bool(json['gridBorderDash']) if 'gridBorderDash' in json else None,
            cell_border_dash=bool(json['cellBorderDash']) if 'cellBorderDash' in json else None,
            row_line_dash=bool(json['rowLineDash']) if 'rowLineDash' in json else None,
            column_line_dash=bool(json['columnLineDash']) if 'columnLineDash' in json else None,
            row_gap_color=dom.RGBA.from_json(json['rowGapColor']) if 'rowGapColor' in json else None,
            row_hatch_color=dom.RGBA.from_json(json['rowHatchColor']) if 'rowHatchColor' in json else None,
            column_gap_color=dom.RGBA.from_json(json['columnGapColor']) if 'columnGapColor' in json else None,
            column_hatch_color=dom.RGBA.from_json(json['columnHatchColor']) if 'columnHatchColor' in json else None,
            area_border_color=dom.RGBA.from_json(json['areaBorderColor']) if 'areaBorderColor' in json else None,
            grid_background_color=dom.RGBA.from_json(json['gridBackgroundColor']) if 'gridBackgroundColor' in json else None,
        )


@dataclass
class FlexContainerHighlightConfig:
    '''
    Configuration data for the highlighting of Flex container elements.
    '''
    #: The style of the container border
    container_border: typing.Optional[LineStyle] = None

    #: The style of the separator between lines
    line_separator: typing.Optional[LineStyle] = None

    #: The style of the separator between items
    item_separator: typing.Optional[LineStyle] = None

    #: Style of content-distribution space on the main axis (justify-content).
    main_distributed_space: typing.Optional[BoxStyle] = None

    #: Style of content-distribution space on the cross axis (align-content).
    cross_distributed_space: typing.Optional[BoxStyle] = None

    #: Style of empty space caused by row gaps (gap/row-gap).
    row_gap_space: typing.Optional[BoxStyle] = None

    #: Style of empty space caused by columns gaps (gap/column-gap).
    column_gap_space: typing.Optional[BoxStyle] = None

    #: Style of the self-alignment line (align-items).
    cross_alignment: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.line_separator is not None:
            json['lineSeparator'] = self.line_separator.to_json()
        if self.item_separator is not None:
            json['itemSeparator'] = self.item_separator.to_json()
        if self.main_distributed_space is not None:
            json['mainDistributedSpace'] = self.main_distributed_space.to_json()
        if self.cross_distributed_space is not None:
            json['crossDistributedSpace'] = self.cross_distributed_space.to_json()
        if self.row_gap_space is not None:
            json['rowGapSpace'] = self.row_gap_space.to_json()
        if self.column_gap_space is not None:
            json['columnGapSpace'] = self.column_gap_space.to_json()
        if self.cross_alignment is not None:
            json['crossAlignment'] = self.cross_alignment.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None,
            line_separator=LineStyle.from_json(json['lineSeparator']) if 'lineSeparator' in json else None,
            item_separator=LineStyle.from_json(json['itemSeparator']) if 'itemSeparator' in json else None,
            main_distributed_space=BoxStyle.from_json(json['mainDistributedSpace']) if 'mainDistributedSpace' in json else None,
            cross_distributed_space=BoxStyle.from_json(json['crossDistributedSpace']) if 'crossDistributedSpace' in json else None,
            row_gap_space=BoxStyle.from_json(json['rowGapSpace']) if 'rowGapSpace' in json else None,
            column_gap_space=BoxStyle.from_json(json['columnGapSpace']) if 'columnGapSpace' in json else None,
            cross_alignment=LineStyle.from_json(json['crossAlignment']) if 'crossAlignment' in json else None,
        )


@dataclass
class FlexItemHighlightConfig:
    '''
    Configuration data for the highlighting of Flex item elements.
    '''
    #: Style of the box representing the item's base size
    base_size_box: typing.Optional[BoxStyle] = None

    #: Style of the border around the box representing the item's base size
    base_size_border: typing.Optional[LineStyle] = None

    #: Style of the arrow representing if the item grew or shrank
    flexibility_arrow: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.base_size_box is not None:
            json['baseSizeBox'] = self.base_size_box.to_json()
        if self.base_size_border is not None:
            json['baseSizeBorder'] = self.base_size_border.to_json()
        if self.flexibility_arrow is not None:
            json['flexibilityArrow'] = self.flexibility_arrow.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            base_size_box=BoxStyle.from_json(json['baseSizeBox']) if 'baseSizeBox' in json else None,
            base_size_border=LineStyle.from_json(json['baseSizeBorder']) if 'baseSizeBorder' in json else None,
            flexibility_arrow=LineStyle.from_json(json['flexibilityArrow']) if 'flexibilityArrow' in json else None,
        )


@dataclass
class LineStyle:
    '''
    Style information for drawing a line.
    '''
    #: The color of the line (default: transparent)
    color: typing.Optional[dom.RGBA] = None

    #: The line pattern (default: solid)
    pattern: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.color is not None:
            json['color'] = self.color.to_json()
        if self.pattern is not None:
            json['pattern'] = self.pattern
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            color=dom.RGBA.from_json(json['color']) if 'color' in json else None,
            pattern=str(json['pattern']) if 'pattern' in json else None,
        )


@dataclass
class BoxStyle:
    '''
    Style information for drawing a box.
    '''
    #: The background color for the box (default: transparent)
    fill_color: typing.Optional[dom.RGBA] = None

    #: The hatching color for the box (default: transparent)
    hatch_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.fill_color is not None:
            json['fillColor'] = self.fill_color.to_json()
        if self.hatch_color is not None:
            json['hatchColor'] = self.hatch_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fill_color=dom.RGBA.from_json(json['fillColor']) if 'fillColor' in json else None,
            hatch_color=dom.RGBA.from_json(json['hatchColor']) if 'hatchColor' in json else None,
        )


class ContrastAlgorithm(enum.Enum):
    AA = "aa"
    AAA = "aaa"
    APCA = "apca"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class HighlightConfig:
    '''
    Configuration data for the highlighting of page elements.
    '''
    #: Whether the node info tooltip should be shown (default: false).
    show_info: typing.Optional[bool] = None

    #: Whether the node styles in the tooltip (default: false).
    show_styles: typing.Optional[bool] = None

    #: Whether the rulers should be shown (default: false).
    show_rulers: typing.Optional[bool] = None

    #: Whether the a11y info should be shown (default: true).
    show_accessibility_info: typing.Optional[bool] = None

    #: Whether the extension lines from node to the rulers should be shown (default: false).
    show_extension_lines: typing.Optional[bool] = None

    #: The content box highlight fill color (default: transparent).
    content_color: typing.Optional[dom.RGBA] = None

    #: The padding highlight fill color (default: transparent).
    padding_color: typing.Optional[dom.RGBA] = None

    #: The border highlight fill color (default: transparent).
    border_color: typing.Optional[dom.RGBA] = None

    #: The margin highlight fill color (default: transparent).
    margin_color: typing.Optional[dom.RGBA] = None

    #: The event target element highlight fill color (default: transparent).
    event_target_color: typing.Optional[dom.RGBA] = None

    #: The shape outside fill color (default: transparent).
    shape_color: typing.Optional[dom.RGBA] = None

    #: The shape margin fill color (default: transparent).
    shape_margin_color: typing.Optional[dom.RGBA] = None

    #: The grid layout color (default: transparent).
    css_grid_color: typing.Optional[dom.RGBA] = None

    #: The color format used to format color styles (default: hex).
    color_format: typing.Optional[ColorFormat] = None

    #: The grid layout highlight configuration (default: all transparent).
    grid_highlight_config: typing.Optional[GridHighlightConfig] = None

    #: The flex container highlight configuration (default: all transparent).
    flex_container_highlight_config: typing.Optional[FlexContainerHighlightConfig] = None

    #: The flex item highlight configuration (default: all transparent).
    flex_item_highlight_config: typing.Optional[FlexItemHighlightConfig] = None

    #: The contrast algorithm to use for the contrast ratio (default: aa).
    contrast_algorithm: typing.Optional[ContrastAlgorithm] = None

    #: The container query container highlight configuration (default: all transparent).
    container_query_container_highlight_config: typing.Optional[ContainerQueryContainerHighlightConfig] = None

    def to_json(self):
        json = dict()
        if self.show_info is not None:
            json['showInfo'] = self.show_info
        if self.show_styles is not None:
            json['showStyles'] = self.show_styles
        if self.show_rulers is not None:
            json['showRulers'] = self.show_rulers
        if self.show_accessibility_info is not None:
            json['showAccessibilityInfo'] = self.show_accessibility_info
        if self.show_extension_lines is not None:
            json['showExtensionLines'] = self.show_extension_lines
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.padding_color is not None:
            json['paddingColor'] = self.padding_color.to_json()
        if self.border_color is not None:
            json['borderColor'] = self.border_color.to_json()
        if self.margin_color is not None:
            json['marginColor'] = self.margin_color.to_json()
        if self.event_target_color is not None:
            json['eventTargetColor'] = self.event_target_color.to_json()
        if self.shape_color is not None:
            json['shapeColor'] = self.shape_color.to_json()
        if self.shape_margin_color is not None:
            json['shapeMarginColor'] = self.shape_margin_color.to_json()
        if self.css_grid_color is not None:
            json['cssGridColor'] = self.css_grid_color.to_json()
        if self.color_format is not None:
            json['colorFormat'] = self.color_format.to_json()
        if self.grid_highlight_config is not None:
            json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        if self.flex_container_highlight_config is not None:
            json['flexContainerHighlightConfig'] = self.flex_container_highlight_config.to_json()
        if self.flex_item_highlight_config is not None:
            json['flexItemHighlightConfig'] = self.flex_item_highlight_config.to_json()
        if self.contrast_algorithm is not None:
            json['contrastAlgorithm'] = self.contrast_algorithm.to_json()
        if self.container_query_container_highlight_config is not None:
            json['containerQueryContainerHighlightConfig'] = self.container_query_container_highlight_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_info=bool(json['showInfo']) if 'showInfo' in json else None,
            show_styles=bool(json['showStyles']) if 'showStyles' in json else None,
            show_rulers=bool(json['showRulers']) if 'showRulers' in json else None,
            show_accessibility_info=bool(json['showAccessibilityInfo']) if 'showAccessibilityInfo' in json else None,
            show_extension_lines=bool(json['showExtensionLines']) if 'showExtensionLines' in json else None,
            content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None,
            padding_color=dom.RGBA.from_json(json['paddingColor']) if 'paddingColor' in json else None,
            border_color=dom.RGBA.from_json(json['borderColor']) if 'borderColor' in json else None,
            margin_color=dom.RGBA.from_json(json['marginColor']) if 'marginColor' in json else None,
            event_target_color=dom.RGBA.from_json(json['eventTargetColor']) if 'eventTargetColor' in json else None,
            shape_color=dom.RGBA.from_json(json['shapeColor']) if 'shapeColor' in json else None,
            shape_margin_color=dom.RGBA.from_json(json['shapeMarginColor']) if 'shapeMarginColor' in json else None,
            css_grid_color=dom.RGBA.from_json(json['cssGridColor']) if 'cssGridColor' in json else None,
            color_format=ColorFormat.from_json(json['colorFormat']) if 'colorFormat' in json else None,
            grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']) if 'gridHighlightConfig' in json else None,
            flex_container_highlight_config=FlexContainerHighlightConfig.from_json(json['flexContainerHighlightConfig']) if 'flexContainerHighlightConfig' in json else None,
            flex_item_highlight_config=FlexItemHighlightConfig.from_json(json['flexItemHighlightConfig']) if 'flexItemHighlightConfig' in json else None,
            contrast_algorithm=ContrastAlgorithm.from_json(json['contrastAlgorithm']) if 'contrastAlgorithm' in json else None,
            container_query_container_highlight_config=ContainerQueryContainerHighlightConfig.from_json(json['containerQueryContainerHighlightConfig']) if 'containerQueryContainerHighlightConfig' in json else None,
        )


class ColorFormat(enum.Enum):
    RGB = "rgb"
    HSL = "hsl"
    HWB = "hwb"
    HEX_ = "hex"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class GridNodeHighlightConfig:
    '''
    Configurations for Persistent Grid Highlight
    '''
    #: A descriptor for the highlight appearance.
    grid_highlight_config: GridHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class FlexNodeHighlightConfig:
    #: A descriptor for the highlight appearance of flex containers.
    flex_container_highlight_config: FlexContainerHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['flexContainerHighlightConfig'] = self.flex_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            flex_container_highlight_config=FlexContainerHighlightConfig.from_json(json['flexContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class ScrollSnapContainerHighlightConfig:
    #: The style of the snapport border (default: transparent)
    snapport_border: typing.Optional[LineStyle] = None

    #: The style of the snap area border (default: transparent)
    snap_area_border: typing.Optional[LineStyle] = None

    #: The margin highlight fill color (default: transparent).
    scroll_margin_color: typing.Optional[dom.RGBA] = None

    #: The padding highlight fill color (default: transparent).
    scroll_padding_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.snapport_border is not None:
            json['snapportBorder'] = self.snapport_border.to_json()
        if self.snap_area_border is not None:
            json['snapAreaBorder'] = self.snap_area_border.to_json()
        if self.scroll_margin_color is not None:
            json['scrollMarginColor'] = self.scroll_margin_color.to_json()
        if self.scroll_padding_color is not None:
            json['scrollPaddingColor'] = self.scroll_padding_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            snapport_border=LineStyle.from_json(json['snapportBorder']) if 'snapportBorder' in json else None,
            snap_area_border=LineStyle.from_json(json['snapAreaBorder']) if 'snapAreaBorder' in json else None,
            scroll_margin_color=dom.RGBA.from_json(json['scrollMarginColor']) if 'scrollMarginColor' in json else None,
            scroll_padding_color=dom.RGBA.from_json(json['scrollPaddingColor']) if 'scrollPaddingColor' in json else None,
        )


@dataclass
class ScrollSnapHighlightConfig:
    #: A descriptor for the highlight appearance of scroll snap containers.
    scroll_snap_container_highlight_config: ScrollSnapContainerHighlightConfig

    #: Identifier of the node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['scrollSnapContainerHighlightConfig'] = self.scroll_snap_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            scroll_snap_container_highlight_config=ScrollSnapContainerHighlightConfig.from_json(json['scrollSnapContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class HingeConfig:
    '''
    Configuration for dual screen hinge
    '''
    #: A rectangle represent hinge
    rect: dom.Rect

    #: The content box highlight fill color (default: a dark color).
    content_color: typing.Optional[dom.RGBA] = None

    #: The content box highlight outline color (default: transparent).
    outline_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        json['rect'] = self.rect.to_json()
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.outline_color is not None:
            json['outlineColor'] = self.outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            rect=dom.Rect.from_json(json['rect']),
            content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None,
            outline_color=dom.RGBA.from_json(json['outlineColor']) if 'outlineColor' in json else None,
        )


@dataclass
class WindowControlsOverlayConfig:
    '''
    Configuration for Window Controls Overlay
    '''
    #: Whether the title bar CSS should be shown when emulating the Window Controls Overlay.
    show_css: bool

    #: Selected platforms to show the overlay.
    selected_platform: str

    #: The theme color defined in app manifest.
    theme_color: str

    def to_json(self):
        json = dict()
        json['showCSS'] = self.show_css
        json['selectedPlatform'] = self.selected_platform
        json['themeColor'] = self.theme_color
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            show_css=bool(json['showCSS']),
            selected_platform=str(json['selectedPlatform']),
            theme_color=str(json['themeColor']),
        )


@dataclass
class ContainerQueryHighlightConfig:
    #: A descriptor for the highlight appearance of container query containers.
    container_query_container_highlight_config: ContainerQueryContainerHighlightConfig

    #: Identifier of the container node to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['containerQueryContainerHighlightConfig'] = self.container_query_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_query_container_highlight_config=ContainerQueryContainerHighlightConfig.from_json(json['containerQueryContainerHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class ContainerQueryContainerHighlightConfig:
    #: The style of the container border.
    container_border: typing.Optional[LineStyle] = None

    #: The style of the descendants' borders.
    descendant_border: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.descendant_border is not None:
            json['descendantBorder'] = self.descendant_border.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None,
            descendant_border=LineStyle.from_json(json['descendantBorder']) if 'descendantBorder' in json else None,
        )


@dataclass
class IsolatedElementHighlightConfig:
    #: A descriptor for the highlight appearance of an element in isolation mode.
    isolation_mode_highlight_config: IsolationModeHighlightConfig

    #: Identifier of the isolated element to highlight.
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['isolationModeHighlightConfig'] = self.isolation_mode_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            isolation_mode_highlight_config=IsolationModeHighlightConfig.from_json(json['isolationModeHighlightConfig']),
            node_id=dom.NodeId.from_json(json['nodeId']),
        )


@dataclass
class IsolationModeHighlightConfig:
    #: The fill color of the resizers (default: transparent).
    resizer_color: typing.Optional[dom.RGBA] = None

    #: The fill color for resizer handles (default: transparent).
    resizer_handle_color: typing.Optional[dom.RGBA] = None

    #: The fill color for the mask covering non-isolated elements (default: transparent).
    mask_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.resizer_color is not None:
            json['resizerColor'] = self.resizer_color.to_json()
        if self.resizer_handle_color is not None:
            json['resizerHandleColor'] = self.resizer_handle_color.to_json()
        if self.mask_color is not None:
            json['maskColor'] = self.mask_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            resizer_color=dom.RGBA.from_json(json['resizerColor']) if 'resizerColor' in json else None,
            resizer_handle_color=dom.RGBA.from_json(json['resizerHandleColor']) if 'resizerHandleColor' in json else None,
            mask_color=dom.RGBA.from_json(json['maskColor']) if 'maskColor' in json else None,
        )


class InspectMode(enum.Enum):
    SEARCH_FOR_NODE = "searchForNode"
    SEARCH_FOR_UA_SHADOW_DOM = "searchForUAShadowDOM"
    CAPTURE_AREA_SCREENSHOT = "captureAreaScreenshot"
    SHOW_DISTANCES = "showDistances"
    NONE = "none"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.enable',
    }
    json = yield cmd_dict


def get_highlight_object_for_test(
        node_id: dom.NodeId,
        include_distance: typing.Optional[bool] = None,
        include_style: typing.Optional[bool] = None,
        color_format: typing.Optional[ColorFormat] = None,
        show_accessibility_info: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For testing.

    :param node_id: Id of the node to get highlight object for.
    :param include_distance: *(Optional)* Whether to include distance info.
    :param include_style: *(Optional)* Whether to include style info.
    :param color_format: *(Optional)* The color format to get config with (default: hex).
    :param show_accessibility_info: *(Optional)* Whether to show accessibility info (default: true).
    :returns: Highlight data for the node.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    if include_distance is not None:
        params['includeDistance'] = include_distance
    if include_style is not None:
        params['includeStyle'] = include_style
    if color_format is not None:
        params['colorFormat'] = color_format.to_json()
    if show_accessibility_info is not None:
        params['showAccessibilityInfo'] = show_accessibility_info
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getHighlightObjectForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlight'])


def get_grid_highlight_objects_for_test(
        node_ids: typing.List[dom.NodeId]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For Persistent Grid testing.

    :param node_ids: Ids of the node to get highlight object for.
    :returns: Grid Highlight data for the node ids provided.
    '''
    params: T_JSON_DICT = dict()
    params['nodeIds'] = [i.to_json() for i in node_ids]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getGridHighlightObjectsForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlights'])


def get_source_order_highlight_object_for_test(
        node_id: dom.NodeId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    For Source Order Viewer testing.

    :param node_id: Id of the node to highlight.
    :returns: Source order highlight data for the node id provided.
    '''
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.getSourceOrderHighlightObjectForTest',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['highlight'])


def hide_highlight() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Hides any highlight.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.hideHighlight',
    }
    json = yield cmd_dict


def highlight_frame(
        frame_id: page.FrameId,
        content_color: typing.Optional[dom.RGBA] = None,
        content_outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights owner element of the frame with given id.
    Deprecated: Doesn't work reliably and cannot be fixed due to process
    separation (the owner node might be in a different process). Determine
    the owner node in the client and use highlightNode.

    :param frame_id: Identifier of the frame to highlight.
    :param content_color: *(Optional)* The content box highlight fill color (default: transparent).
    :param content_outline_color: *(Optional)* The content box highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    if content_color is not None:
        params['contentColor'] = content_color.to_json()
    if content_outline_color is not None:
        params['contentOutlineColor'] = content_outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightFrame',
        'params': params,
    }
    json = yield cmd_dict


def highlight_node(
        highlight_config: HighlightConfig,
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None,
        selector: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights DOM node with given id or with the given JavaScript object wrapper. Either nodeId or
    objectId must be specified.

    :param highlight_config: A descriptor for the highlight appearance.
    :param node_id: *(Optional)* Identifier of the node to highlight.
    :param backend_node_id: *(Optional)* Identifier of the backend node to highlight.
    :param object_id: *(Optional)* JavaScript object id of the node to be highlighted.
    :param selector: *(Optional)* Selectors to highlight relevant nodes.
    '''
    params: T_JSON_DICT = dict()
    params['highlightConfig'] = highlight_config.to_json()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if selector is not None:
        params['selector'] = selector
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightNode',
        'params': params,
    }
    json = yield cmd_dict


def highlight_quad(
        quad: dom.Quad,
        color: typing.Optional[dom.RGBA] = None,
        outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights given quad. Coordinates are absolute with respect to the main frame viewport.

    :param quad: Quad to highlight
    :param color: *(Optional)* The highlight fill color (default: transparent).
    :param outline_color: *(Optional)* The highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['quad'] = quad.to_json()
    if color is not None:
        params['color'] = color.to_json()
    if outline_color is not None:
        params['outlineColor'] = outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightQuad',
        'params': params,
    }
    json = yield cmd_dict


def highlight_rect(
        x: int,
        y: int,
        width: int,
        height: int,
        color: typing.Optional[dom.RGBA] = None,
        outline_color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights given rectangle. Coordinates are absolute with respect to the main frame viewport.

    :param x: X coordinate
    :param y: Y coordinate
    :param width: Rectangle width
    :param height: Rectangle height
    :param color: *(Optional)* The highlight fill color (default: transparent).
    :param outline_color: *(Optional)* The highlight outline color (default: transparent).
    '''
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    params['width'] = width
    params['height'] = height
    if color is not None:
        params['color'] = color.to_json()
    if outline_color is not None:
        params['outlineColor'] = outline_color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightRect',
        'params': params,
    }
    json = yield cmd_dict


def highlight_source_order(
        source_order_config: SourceOrderConfig,
        node_id: typing.Optional[dom.NodeId] = None,
        backend_node_id: typing.Optional[dom.BackendNodeId] = None,
        object_id: typing.Optional[runtime.RemoteObjectId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights the source order of the children of the DOM node with given id or with the given
    JavaScript object wrapper. Either nodeId or objectId must be specified.

    :param source_order_config: A descriptor for the appearance of the overlay drawing.
    :param node_id: *(Optional)* Identifier of the node to highlight.
    :param backend_node_id: *(Optional)* Identifier of the backend node to highlight.
    :param object_id: *(Optional)* JavaScript object id of the node to be highlighted.
    '''
    params: T_JSON_DICT = dict()
    params['sourceOrderConfig'] = source_order_config.to_json()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.highlightSourceOrder',
        'params': params,
    }
    json = yield cmd_dict


def set_inspect_mode(
        mode: InspectMode,
        highlight_config: typing.Optional[HighlightConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enters the 'inspect' mode. In this mode, elements that user is hovering over are highlighted.
    Backend then generates 'inspectNodeRequested' event upon element selection.

    :param mode: Set an inspection mode.
    :param highlight_config: *(Optional)* A descriptor for the highlight appearance of hovered-over nodes. May be omitted if ```enabled == false```.
    '''
    params: T_JSON_DICT = dict()
    params['mode'] = mode.to_json()
    if highlight_config is not None:
        params['highlightConfig'] = highlight_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setInspectMode',
        'params': params,
    }
    json = yield cmd_dict


def set_show_ad_highlights(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlights owner element of all frames detected to be ads.

    :param show: True for showing ad highlights
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowAdHighlights',
        'params': params,
    }
    json = yield cmd_dict


def set_paused_in_debugger_message(
        message: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param message: *(Optional)* The message to display, also triggers resume and step over controls.
    '''
    params: T_JSON_DICT = dict()
    if message is not None:
        params['message'] = message
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setPausedInDebuggerMessage',
        'params': params,
    }
    json = yield cmd_dict


def set_show_debug_borders(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows debug borders on layers

    :param show: True for showing debug borders
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowDebugBorders',
        'params': params,
    }
    json = yield cmd_dict


def set_show_fps_counter(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows the FPS counter

    :param show: True for showing the FPS counter
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowFPSCounter',
        'params': params,
    }
    json = yield cmd_dict


def set_show_grid_overlays(
        grid_node_highlight_configs: typing.List[GridNodeHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Highlight multiple elements with the CSS Grid overlay.

    :param grid_node_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['gridNodeHighlightConfigs'] = [i.to_json() for i in grid_node_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowGridOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_flex_overlays(
        flex_node_highlight_configs: typing.List[FlexNodeHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param flex_node_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['flexNodeHighlightConfigs'] = [i.to_json() for i in flex_node_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowFlexOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_scroll_snap_overlays(
        scroll_snap_highlight_configs: typing.List[ScrollSnapHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scroll_snap_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['scrollSnapHighlightConfigs'] = [i.to_json() for i in scroll_snap_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowScrollSnapOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_container_query_overlays(
        container_query_highlight_configs: typing.List[ContainerQueryHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param container_query_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['containerQueryHighlightConfigs'] = [i.to_json() for i in container_query_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowContainerQueryOverlays',
        'params': params,
    }
    json = yield cmd_dict


def set_show_paint_rects(
        result: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows paint rectangles

    :param result: True for showing paint rectangles
    '''
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowPaintRects',
        'params': params,
    }
    json = yield cmd_dict


def set_show_layout_shift_regions(
        result: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows layout shift regions

    :param result: True for showing layout shift regions
    '''
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowLayoutShiftRegions',
        'params': params,
    }
    json = yield cmd_dict


def set_show_scroll_bottleneck_rects(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that backend shows scroll bottleneck rects

    :param show: True for showing scroll bottleneck rects
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowScrollBottleneckRects',
        'params': params,
    }
    json = yield cmd_dict


def set_show_hit_test_borders(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deprecated, no longer has any effect.

    :param show: True for showing hit-test borders
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowHitTestBorders',
        'params': params,
    }
    json = yield cmd_dict


def set_show_web_vitals(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deprecated, no longer has any effect.

    :param show:
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowWebVitals',
        'params': params,
    }
    json = yield cmd_dict


def set_show_viewport_size_on_resize(
        show: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Paints viewport size upon main frame resize.

    :param show: Whether to paint size or not.
    '''
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowViewportSizeOnResize',
        'params': params,
    }
    json = yield cmd_dict


def set_show_hinge(
        hinge_config: typing.Optional[HingeConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Add a dual screen device hinge

    :param hinge_config: *(Optional)* hinge data, null means hideHinge
    '''
    params: T_JSON_DICT = dict()
    if hinge_config is not None:
        params['hingeConfig'] = hinge_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowHinge',
        'params': params,
    }
    json = yield cmd_dict


def set_show_isolated_elements(
        isolated_element_highlight_configs: typing.List[IsolatedElementHighlightConfig]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Show elements in isolation mode with overlays.

    :param isolated_element_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    '''
    params: T_JSON_DICT = dict()
    params['isolatedElementHighlightConfigs'] = [i.to_json() for i in isolated_element_highlight_configs]
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowIsolatedElements',
        'params': params,
    }
    json = yield cmd_dict


def set_show_window_controls_overlay(
        window_controls_overlay_config: typing.Optional[WindowControlsOverlayConfig] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Show Window Controls Overlay for PWA

    :param window_controls_overlay_config: *(Optional)* Window Controls Overlay data, null means hide Window Controls Overlay
    '''
    params: T_JSON_DICT = dict()
    if window_controls_overlay_config is not None:
        params['windowControlsOverlayConfig'] = window_controls_overlay_config.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Overlay.setShowWindowControlsOverlay',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Overlay.inspectNodeRequested')
@dataclass
class InspectNodeRequested:
    '''
    Fired when the node should be inspected. This happens after call to ``setInspectMode`` or when
    user manually inspects an element.
    '''
    #: Id of the node to inspect.
    backend_node_id: dom.BackendNodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectNodeRequested:
        return cls(
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId'])
        )


@event_class('Overlay.nodeHighlightRequested')
@dataclass
class NodeHighlightRequested:
    '''
    Fired when the node should be highlighted. This happens after call to ``setInspectMode``.
    '''
    node_id: dom.NodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeHighlightRequested:
        return cls(
            node_id=dom.NodeId.from_json(json['nodeId'])
        )


@event_class('Overlay.screenshotRequested')
@dataclass
class ScreenshotRequested:
    '''
    Fired when user asks to capture screenshot of some area on the page.
    '''
    #: Viewport to capture, in device independent pixels (dip).
    viewport: page.Viewport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreenshotRequested:
        return cls(
            viewport=page.Viewport.from_json(json['viewport'])
        )


@event_class('Overlay.inspectModeCanceled')
@dataclass
class InspectModeCanceled:
    '''
    Fired when user cancels the inspect mode.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectModeCanceled:
        return cls(

        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_dtls.py ===
# Implementation of DTLS 1.2, using pyopenssl
# https://datatracker.ietf.org/doc/html/rfc6347
#
# OpenSSL's APIs for DTLS are extremely awkward and limited, which forces us to jump
# through a *lot* of hoops and implement important chunks of the protocol ourselves.
# Hopefully they fix this before implementing DTLS 1.3, because it's a very different
# protocol, and it's probably impossible to pull tricks like we do here.

from __future__ import annotations

import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
    Union,
)
from weakref import ReferenceType, WeakValueDictionary

import attrs

import trio

from ._util import NoPublicConstructor, final

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Iterator
    from types import TracebackType

    # See DTLSEndpoint.__init__ for why this is imported here
    from OpenSSL import SSL  # noqa: TC004
    from typing_extensions import Self, TypeAlias, TypeVarTuple, Unpack

    from trio._socket import AddressFormat
    from trio.socket import SocketType

    PosArgsT = TypeVarTuple("PosArgsT")

MAX_UDP_PACKET_SIZE = 65527


def packet_header_overhead(sock: SocketType) -> int:
    if sock.family == trio.socket.AF_INET:
        return 28
    else:
        return 48


def worst_case_mtu(sock: SocketType) -> int:
    if sock.family == trio.socket.AF_INET:
        return 576 - packet_header_overhead(sock)
    else:
        return 1280 - packet_header_overhead(sock)  # TODO: test this line


def best_guess_mtu(sock: SocketType) -> int:
    return 1500 - packet_header_overhead(sock)


# There are a bunch of different RFCs that define these codes, so for a
# comprehensive collection look here:
# https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml
class ContentType(enum.IntEnum):
    change_cipher_spec = 20
    alert = 21
    handshake = 22
    application_data = 23
    heartbeat = 24


class HandshakeType(enum.IntEnum):
    hello_request = 0
    client_hello = 1
    server_hello = 2
    hello_verify_request = 3
    new_session_ticket = 4
    end_of_early_data = 4
    encrypted_extensions = 8
    certificate = 11
    server_key_exchange = 12
    certificate_request = 13
    server_hello_done = 14
    certificate_verify = 15
    client_key_exchange = 16
    finished = 20
    certificate_url = 21
    certificate_status = 22
    supplemental_data = 23
    key_update = 24
    compressed_certificate = 25
    ekt_key = 26
    message_hash = 254


class ProtocolVersion:
    DTLS10 = bytes([254, 255])
    DTLS12 = bytes([254, 253])


EPOCH_MASK = 0xFFFF << (6 * 8)


# Conventions:
# - All functions that handle network data end in _untrusted.
# - All functions end in _untrusted MUST make sure that bad data from the
#   network cannot *only* cause BadPacket to be raised. No IndexError or
#   struct.error or whatever.
class BadPacket(Exception):
    pass


# This checks that the DTLS 'epoch' field is 0, which is true iff we're in the
# initial handshake. It doesn't check the ContentType, because not all
# handshake messages have ContentType==handshake -- for example,
# ChangeCipherSpec is used during the handshake but has its own ContentType.
#
# Cannot fail.
def part_of_handshake_untrusted(packet: bytes) -> bool:
    # If the packet is too short, then slicing will successfully return a
    # short string, which will necessarily fail to match.
    return packet[3:5] == b"\x00\x00"


# Cannot fail
def is_client_hello_untrusted(packet: bytes) -> bool:
    try:
        return (
            packet[0] == ContentType.handshake
            and packet[13] == HandshakeType.client_hello
        )
    except IndexError:
        # Invalid DTLS record
        return False


# DTLS records are:
# - 1 byte content type
# - 2 bytes version
# - 8 bytes epoch+seqno
#    Technically this is 2 bytes epoch then 6 bytes seqno, but we treat it as
#    a single 8-byte integer, where epoch changes are represented as jumping
#    forward by 2**(6*8).
# - 2 bytes payload length (unsigned big-endian)
# - payload
RECORD_HEADER = struct.Struct("!B2sQH")


def to_hex(data: bytes) -> str:  # pragma: no cover
    return data.hex()


@attrs.frozen
class Record:
    content_type: int
    version: bytes = attrs.field(repr=to_hex)
    epoch_seqno: int
    payload: bytes = attrs.field(repr=to_hex)


def records_untrusted(packet: bytes) -> Iterator[Record]:
    i = 0
    while i < len(packet):
        try:
            ct, version, epoch_seqno, payload_len = RECORD_HEADER.unpack_from(packet, i)
        # Marked as no-cover because at time of writing, this code is unreachable
        # (records_untrusted only gets called on packets that are either trusted or that
        # have passed is_client_hello_untrusted, which filters out short packets)
        except struct.error as exc:  # pragma: no cover
            raise BadPacket("invalid record header") from exc
        i += RECORD_HEADER.size
        payload = packet[i : i + payload_len]
        if len(payload) != payload_len:
            raise BadPacket("short record")
        i += payload_len
        yield Record(ct, version, epoch_seqno, payload)


def encode_record(record: Record) -> bytes:
    header = RECORD_HEADER.pack(
        record.content_type,
        record.version,
        record.epoch_seqno,
        len(record.payload),
    )
    return header + record.payload


# Handshake messages are:
# - 1 byte message type
# - 3 bytes total message length
# - 2 bytes message sequence number
# - 3 bytes fragment offset
# - 3 bytes fragment length
HANDSHAKE_MESSAGE_HEADER = struct.Struct("!B3sH3s3s")


@attrs.frozen
class HandshakeFragment:
    msg_type: int
    msg_len: int
    msg_seq: int
    frag_offset: int
    frag_len: int
    frag: bytes = attrs.field(repr=to_hex)


def decode_handshake_fragment_untrusted(payload: bytes) -> HandshakeFragment:
    # Raises BadPacket if decoding fails
    try:
        (
            msg_type,
            msg_len_bytes,
            msg_seq,
            frag_offset_bytes,
            frag_len_bytes,
        ) = HANDSHAKE_MESSAGE_HEADER.unpack_from(payload)
    except struct.error as exc:  # TODO: test this line
        raise BadPacket("bad handshake message header") from exc
    # 'struct' doesn't have built-in support for 24-bit integers, so we
    # have to do it by hand. These can't fail.
    msg_len = int.from_bytes(msg_len_bytes, "big")
    frag_offset = int.from_bytes(frag_offset_bytes, "big")
    frag_len = int.from_bytes(frag_len_bytes, "big")
    frag = payload[HANDSHAKE_MESSAGE_HEADER.size :]
    if len(frag) != frag_len:
        raise BadPacket("handshake fragment length doesn't match record length")
    return HandshakeFragment(
        msg_type,
        msg_len,
        msg_seq,
        frag_offset,
        frag_len,
        frag,
    )


def encode_handshake_fragment(hsf: HandshakeFragment) -> bytes:
    hs_header = HANDSHAKE_MESSAGE_HEADER.pack(
        hsf.msg_type,
        hsf.msg_len.to_bytes(3, "big"),
        hsf.msg_seq,
        hsf.frag_offset.to_bytes(3, "big"),
        hsf.frag_len.to_bytes(3, "big"),
    )
    return hs_header + hsf.frag


def decode_client_hello_untrusted(packet: bytes) -> tuple[int, bytes, bytes]:
    # Raises BadPacket if parsing fails
    # Returns (record epoch_seqno, cookie from the packet, data that should be
    # hashed into cookie)
    try:
        # ClientHello has to be the first record in the packet
        record = next(records_untrusted(packet))
        # no-cover because at time of writing, this is unreachable:
        # decode_client_hello_untrusted is only called on packets that have passed
        # is_client_hello_untrusted, which confirms the content type.
        if record.content_type != ContentType.handshake:  # pragma: no cover
            raise BadPacket("not a handshake record")
        fragment = decode_handshake_fragment_untrusted(record.payload)
        if fragment.msg_type != HandshakeType.client_hello:
            raise BadPacket("not a ClientHello")
        # ClientHello can't be fragmented, because reassembly requires holding
        # per-connection state, and we refuse to allocate per-connection state
        # until after we get a valid ClientHello.
        if fragment.frag_offset != 0:
            raise BadPacket("fragmented ClientHello")
        if fragment.frag_len != fragment.msg_len:
            raise BadPacket("fragmented ClientHello")

        # As per RFC 6347:
        #
        #   When responding to a HelloVerifyRequest, the client MUST use the
        #   same parameter values (version, random, session_id, cipher_suites,
        #   compression_method) as it did in the original ClientHello.  The
        #   server SHOULD use those values to generate its cookie and verify that
        #   they are correct upon cookie receipt.
        #
        # However, the record-layer framing can and will change (e.g. the
        # second ClientHello will have a new record-layer sequence number). So
        # we need to pull out the handshake message alone, discarding the
        # record-layer stuff, and then we're going to hash all of it *except*
        # the cookie.

        body = fragment.frag
        # ClientHello is:
        #
        # - 2 bytes client_version
        # - 32 bytes random
        # - 1 byte session_id length
        # - session_id
        # - 1 byte cookie length
        # - cookie
        # - everything else
        #
        # So to find the cookie, so we need to figure out how long the
        # session_id is and skip past it.
        session_id_len = body[2 + 32]
        cookie_len_offset = 2 + 32 + 1 + session_id_len
        cookie_len = body[cookie_len_offset]

        cookie_start = cookie_len_offset + 1
        cookie_end = cookie_start + cookie_len

        before_cookie = body[:cookie_len_offset]
        cookie = body[cookie_start:cookie_end]
        after_cookie = body[cookie_end:]

        if len(cookie) != cookie_len:
            raise BadPacket("short cookie")
        return (record.epoch_seqno, cookie, before_cookie + after_cookie)

    except (struct.error, IndexError) as exc:
        raise BadPacket("bad ClientHello") from exc


@attrs.frozen
class HandshakeMessage:
    record_version: bytes = attrs.field(repr=to_hex)
    msg_type: HandshakeType
    msg_seq: int
    body: bytearray = attrs.field(repr=to_hex)


# ChangeCipherSpec is part of the handshake, but it's not a "handshake
# message" and can't be fragmented the same way. Sigh.
@attrs.frozen
class PseudoHandshakeMessage:
    record_version: bytes = attrs.field(repr=to_hex)
    content_type: int
    payload: bytes = attrs.field(repr=to_hex)


# The final record in a handshake is Finished, which is encrypted, can't be fragmented
# (at least by us), and keeps its record number (because it's in a new epoch). So we
# just pass it through unchanged. (Fortunately, the payload is only a single hash value,
# so the largest it will ever be is 64 bytes for a 512-bit hash. Which is small enough
# that it never requires fragmenting to fit into a UDP packet.
@attrs.frozen
class OpaqueHandshakeMessage:
    record: Record


_AnyHandshakeMessage: TypeAlias = Union[
    HandshakeMessage,
    PseudoHandshakeMessage,
    OpaqueHandshakeMessage,
]


# This takes a raw outgoing handshake volley that openssl generated, and
# reconstructs the handshake messages inside it, so that we can repack them
# into records while retransmitting. So the data ought to be well-behaved --
# it's not coming from the network.
def decode_volley_trusted(
    volley: bytes,
) -> list[_AnyHandshakeMessage]:
    messages: list[_AnyHandshakeMessage] = []
    messages_by_seq = {}
    for record in records_untrusted(volley):
        # ChangeCipherSpec isn't a handshake message, so it can't be fragmented.
        # Handshake messages with epoch > 0 are encrypted, so we can't fragment them
        # either. Fortunately, ChangeCipherSpec has a 1 byte payload, and the only
        # encrypted handshake message is Finished, whose payload is a single hash value
        # -- so 32 bytes for SHA-256, 64 for SHA-512, etc. Neither is going to be so
        # large that it has to be fragmented to fit into a single packet.
        if record.epoch_seqno & EPOCH_MASK:
            messages.append(OpaqueHandshakeMessage(record))
        elif record.content_type in (ContentType.change_cipher_spec, ContentType.alert):
            messages.append(
                PseudoHandshakeMessage(
                    record.version,
                    record.content_type,
                    record.payload,
                ),
            )
        else:
            assert record.content_type == ContentType.handshake
            fragment = decode_handshake_fragment_untrusted(record.payload)
            msg_type = HandshakeType(fragment.msg_type)
            if fragment.msg_seq not in messages_by_seq:
                msg = HandshakeMessage(
                    record.version,
                    msg_type,
                    fragment.msg_seq,
                    bytearray(fragment.msg_len),
                )
                messages.append(msg)
                messages_by_seq[fragment.msg_seq] = msg
            else:
                msg = messages_by_seq[fragment.msg_seq]
            assert msg.msg_type == fragment.msg_type
            assert msg.msg_seq == fragment.msg_seq
            assert len(msg.body) == fragment.msg_len

            msg.body[
                fragment.frag_offset : fragment.frag_offset + fragment.frag_len
            ] = fragment.frag

    return messages


class RecordEncoder:
    def __init__(self) -> None:
        self._record_seq = count()

    def set_first_record_number(self, n: int) -> None:
        self._record_seq = count(n)

    def encode_volley(
        self,
        messages: Iterable[_AnyHandshakeMessage],
        mtu: int,
    ) -> list[bytearray]:
        packets = []
        packet = bytearray()
        for message in messages:
            if isinstance(message, OpaqueHandshakeMessage):
                encoded = encode_record(message.record)
                if mtu - len(packet) - len(encoded) <= 0:  # TODO: test this line
                    packets.append(packet)
                    packet = bytearray()
                packet += encoded
                assert len(packet) <= mtu
            elif isinstance(message, PseudoHandshakeMessage):
                space = mtu - len(packet) - RECORD_HEADER.size - len(message.payload)
                if space <= 0:  # TODO: test this line
                    packets.append(packet)
                    packet = bytearray()
                packet += RECORD_HEADER.pack(
                    message.content_type,
                    message.record_version,
                    next(self._record_seq),
                    len(message.payload),
                )
                packet += message.payload
                assert len(packet) <= mtu
            else:
                msg_len_bytes = len(message.body).to_bytes(3, "big")
                frag_offset = 0
                frags_encoded = 0
                # If message.body is empty, then we still want to encode it in one
                # fragment, not zero.
                while frag_offset < len(message.body) or not frags_encoded:
                    space = (
                        mtu
                        - len(packet)
                        - RECORD_HEADER.size
                        - HANDSHAKE_MESSAGE_HEADER.size
                    )
                    if space <= 0:
                        packets.append(packet)
                        packet = bytearray()
                        continue
                    frag = message.body[frag_offset : frag_offset + space]
                    frag_offset_bytes = frag_offset.to_bytes(3, "big")
                    frag_len_bytes = len(frag).to_bytes(3, "big")
                    frag_offset += len(frag)

                    packet += RECORD_HEADER.pack(
                        ContentType.handshake,
                        message.record_version,
                        next(self._record_seq),
                        HANDSHAKE_MESSAGE_HEADER.size + len(frag),
                    )

                    packet += HANDSHAKE_MESSAGE_HEADER.pack(
                        message.msg_type,
                        msg_len_bytes,
                        message.msg_seq,
                        frag_offset_bytes,
                        frag_len_bytes,
                    )

                    packet += frag

                    frags_encoded += 1
                    assert len(packet) <= mtu

        if packet:
            packets.append(packet)

        return packets


# This bit requires implementing a bona fide cryptographic protocol, so even though it's
# a simple one let's take a moment to discuss the design.
#
# Our goal is to force new incoming handshakes that claim to be coming from a
# given ip:port to prove that they can also receive packets sent to that
# ip:port. (There's nothing in UDP to stop someone from forging the return
# address, and it's often used for stuff like DoS reflection attacks, where
# an attacker tries to trick us into sending data at some innocent victim.)
# For more details, see:
#
#    https://datatracker.ietf.org/doc/html/rfc6347#section-4.2.1
#
# To do this, when we receive an initial ClientHello, we calculate a magic
# cookie, and send it back as a HelloVerifyRequest. Then the client sends us a
# second ClientHello, this time with the magic cookie included, and after we
# check that this cookie is valid we go ahead and start the handshake proper.
#
# So the magic cookie needs the following properties:
# - No-one can forge it without knowing our secret key
# - It ensures that the ip, port, and ClientHello contents from the response
#   match those in the challenge
# - It expires after a short-ish period (so that if an attacker manages to steal one, it
#   won't be useful for long)
# - It doesn't require storing any peer-specific state on our side
#
# To do that, we take the ip/port/ClientHello data and compute an HMAC of them, using a
# secret key we generate on startup. We also include:
#
# - The current time (using Trio's clock), rounded to the nearest 30 seconds
# - A random salt
#
# Then the cookie is the salt and the HMAC digest concatenated together.
#
# When verifying a cookie, we use the salt + new ip/port/ClientHello data to recompute
# the HMAC digest, for both the current time and the current time minus 30 seconds, and
# if either of them match, we consider the cookie good.
#
# Including the rounded-off time like this means that each cookie is good for at least
# 30 seconds, and possibly as much as 60 seconds.
#
# The salt is probably not necessary -- I'm pretty sure that all it does is make it hard
# for an attacker to figure out when our clock ticks over a 30 second boundary. Which is
# probably pretty harmless? But it's easier to add the salt than to convince myself that
# it's *completely* harmless, so, salt it is.

COOKIE_REFRESH_INTERVAL = 30  # seconds
KEY_BYTES = 32
COOKIE_HASH = "sha256"
SALT_BYTES = 8
# 32 bytes was the maximum cookie length in DTLS 1.0. DTLS 1.2 raised it to 255. I doubt
# there are any DTLS 1.0 implementations still in the wild, but really 32 bytes is
# plenty, and it also gets rid of a confusing warning in Wireshark output.
#
# We truncate the cookie to 32 bytes, of which 8 bytes is salt, so that leaves 24 bytes
# of truncated HMAC = 192 bit security, which is still massive overkill. (TCP uses 32
# *bits* for this.) HMAC truncation is explicitly noted as safe in RFC 2104:
#   https://datatracker.ietf.org/doc/html/rfc2104#section-5
COOKIE_LENGTH = 32


def _current_cookie_tick() -> int:
    return int(trio.current_time() / COOKIE_REFRESH_INTERVAL)


# Simple deterministic and invertible serializer -- i.e., a useful tool for converting
# structured data into something we can cryptographically sign.
def _signable(*fields: bytes) -> bytes:
    out: list[bytes] = []
    for field in fields:
        out.extend((struct.pack("!Q", len(field)), field))
    return b"".join(out)


def _make_cookie(
    key: bytes,
    salt: bytes,
    tick: int,
    address: AddressFormat,
    client_hello_bits: bytes,
) -> bytes:
    assert len(salt) == SALT_BYTES
    assert len(key) == KEY_BYTES

    signable_data = _signable(
        salt,
        struct.pack("!Q", tick),
        # address is a mix of strings and ints, and variable length, so pack
        # it into a single nested field
        _signable(*(str(part).encode() for part in address)),
        client_hello_bits,
    )

    return (salt + hmac.digest(key, signable_data, COOKIE_HASH))[:COOKIE_LENGTH]


def valid_cookie(
    key: bytes,
    cookie: bytes,
    address: AddressFormat,
    client_hello_bits: bytes,
) -> bool:
    if len(cookie) > SALT_BYTES:
        salt = cookie[:SALT_BYTES]

        tick = _current_cookie_tick()

        cur_cookie = _make_cookie(key, salt, tick, address, client_hello_bits)
        old_cookie = _make_cookie(
            key,
            salt,
            max(tick - 1, 0),
            address,
            client_hello_bits,
        )

        # I doubt using a short-circuiting 'or' here would leak any meaningful
        # information, but why risk it when '|' is just as easy.
        return hmac.compare_digest(cookie, cur_cookie) | hmac.compare_digest(
            cookie,
            old_cookie,
        )
    else:
        return False


def challenge_for(
    key: bytes,
    address: AddressFormat,
    epoch_seqno: int,
    client_hello_bits: bytes,
) -> bytes:
    salt = os.urandom(SALT_BYTES)
    tick = _current_cookie_tick()
    cookie = _make_cookie(key, salt, tick, address, client_hello_bits)

    # HelloVerifyRequest body is:
    # - 2 bytes version
    # - length-prefixed cookie
    #
    # The DTLS 1.2 spec says that for this message specifically we should use
    # the DTLS 1.0 version.
    #
    # (It also says the opposite of that, but that part is a mistake:
    #    https://www.rfc-editor.org/errata/eid4103
    # ).
    #
    # And I guess we use this for both the message-level and record-level
    # ProtocolVersions, since we haven't negotiated anything else yet?
    body = ProtocolVersion.DTLS10 + bytes([len(cookie)]) + cookie

    # RFC says have to copy the client's record number
    # Errata says it should be handshake message number
    # Openssl copies back record sequence number, and always sets message seq
    # number 0. So I guess we'll follow openssl.
    hs = HandshakeFragment(
        msg_type=HandshakeType.hello_verify_request,
        msg_len=len(body),
        msg_seq=0,
        frag_offset=0,
        frag_len=len(body),
        frag=body,
    )
    payload = encode_handshake_fragment(hs)

    packet = encode_record(
        Record(ContentType.handshake, ProtocolVersion.DTLS10, epoch_seqno, payload),
    )
    return packet


_T = TypeVar("_T")


class _Queue(Generic[_T]):
    def __init__(self, incoming_packets_buffer: int | float) -> None:  # noqa: PYI041
        self.s, self.r = trio.open_memory_channel[_T](incoming_packets_buffer)


def _read_loop(read_fn: Callable[[int], bytes]) -> bytes:
    chunks = []
    while True:
        try:
            chunk = read_fn(2**14)  # max TLS record size
        except SSL.WantReadError:
            break
        chunks.append(chunk)
    return b"".join(chunks)


async def handle_client_hello_untrusted(
    endpoint: DTLSEndpoint,
    address: AddressFormat,
    packet: bytes,
) -> None:
    # it's trivial to write a simple function that directly calls this to
    # get code coverage, but it should maybe:
    # 1. be removed
    # 2. be asserted
    # 3. Write a complicated test case where this happens "organically"
    if endpoint._listening_context is None:  # pragma: no cover
        return

    try:
        epoch_seqno, cookie, bits = decode_client_hello_untrusted(packet)
    except BadPacket:
        return

    if endpoint._listening_key is None:
        endpoint._listening_key = os.urandom(KEY_BYTES)

    if not valid_cookie(endpoint._listening_key, cookie, address, bits):
        challenge_packet = challenge_for(
            endpoint._listening_key,
            address,
            epoch_seqno,
            bits,
        )
        try:
            async with endpoint._send_lock:
                await endpoint.socket.sendto(challenge_packet, address)
        except (OSError, trio.ClosedResourceError):
            pass
    else:
        # We got a real, valid ClientHello!
        stream = DTLSChannel._create(endpoint, address, endpoint._listening_context)
        # Our HelloRetryRequest had some sequence number. We need our future sequence
        # numbers to be larger than it, so our peer knows that our future records aren't
        # stale/duplicates. But, we don't know what this sequence number was. What we do
        # know is:
        # - the HelloRetryRequest seqno was copied it from the initial ClientHello
        # - the new ClientHello has a higher seqno than the initial ClientHello
        # So, if we copy the new ClientHello's seqno into our first real handshake
        # record and increment from there, that should work.
        stream._record_encoder.set_first_record_number(epoch_seqno)
        # Process the ClientHello
        try:
            stream._ssl.bio_write(packet)
            stream._ssl.DTLSv1_listen()
        except SSL.Error:  # pragma: no cover
            # ...OpenSSL didn't like it, so I guess we didn't have a valid ClientHello
            # after all.
            return

        # Check if we have an existing association
        old_stream = endpoint._streams.get(address)
        if old_stream is not None:
            if old_stream._client_hello == (cookie, bits):
                # ...This was just a duplicate of the last ClientHello, so never mind.
                return
            else:
                # Ok, this *really is* a new handshake; the old stream should go away.
                old_stream._set_replaced()
        stream._client_hello = (cookie, bits)
        endpoint._streams[address] = stream
        endpoint._incoming_connections_q.s.send_nowait(stream)


async def dtls_receive_loop(
    endpoint_ref: ReferenceType[DTLSEndpoint],
    sock: SocketType,
) -> None:
    try:
        while True:
            try:
                packet, address = await sock.recvfrom(MAX_UDP_PACKET_SIZE)
            except OSError as exc:
                if exc.errno == errno.ECONNRESET:
                    # Windows only: "On a UDP-datagram socket [ECONNRESET]
                    # indicates a previous send operation resulted in an ICMP Port
                    # Unreachable message" -- https://docs.microsoft.com/en-us/windows/win32/api/winsock/nf-winsock-recvfrom
                    #
                    # This is totally useless -- there's nothing we can do with this
                    # information. So we just ignore it and retry the recv.
                    continue
                else:
                    raise
            endpoint = endpoint_ref()
            try:
                if endpoint is None:
                    return
                if is_client_hello_untrusted(packet):
                    await handle_client_hello_untrusted(endpoint, address, packet)
                elif address in endpoint._streams:
                    stream = endpoint._streams[address]
                    if stream._did_handshake and part_of_handshake_untrusted(packet):
                        # The peer just sent us more handshake messages, that aren't a
                        # ClientHello, and we thought the handshake was done. Some of
                        # the packets that we sent to finish the handshake must have
                        # gotten lost. So re-send them. We do this directly here instead
                        # of just putting it into the queue and letting the receiver do
                        # it, because there's no guarantee that anyone is reading from
                        # the queue, because we think the handshake is done!
                        await stream._resend_final_volley()
                    else:
                        try:
                            # mypy for some reason cannot determine type of _q
                            stream._q.s.send_nowait(packet)  # type:ignore[has-type]
                        except trio.WouldBlock:
                            stream._packets_dropped_in_trio += 1
                else:
                    # Drop packet
                    pass
            finally:
                del endpoint
    except trio.ClosedResourceError:
        # socket was closed
        return
    except OSError as exc:
        if exc.errno in (errno.EBADF, errno.ENOTSOCK):
            # socket was closed
            return
        else:  # pragma: no cover
            # ??? shouldn't happen
            raise


@attrs.frozen
class DTLSChannelStatistics:
    """Currently this has only one attribute:

    - ``incoming_packets_dropped_in_trio`` (``int``): Gives a count of the number of
      incoming packets from this peer that Trio successfully received from the
      network, but then got dropped because the internal channel buffer was full. If
      this is non-zero, then you might want to call ``receive`` more often, or use a
      larger ``incoming_packets_buffer``, or just not worry about it because your
      UDP-based protocol should be able to handle the occasional lost packet, right?

    """

    incoming_packets_dropped_in_trio: int


@final
class DTLSChannel(trio.abc.Channel[bytes], metaclass=NoPublicConstructor):
    """A DTLS connection.

    This class has no public constructor – you get instances by calling
    `DTLSEndpoint.serve` or `~DTLSEndpoint.connect`.

    .. attribute:: endpoint

       The `DTLSEndpoint` that this connection is using.

    .. attribute:: peer_address

       The IP/port of the remote peer that this connection is associated with.

    """

    def __init__(
        self,
        endpoint: DTLSEndpoint,
        peer_address: AddressFormat,
        ctx: SSL.Context,
    ) -> None:
        self.endpoint = endpoint
        self.peer_address = peer_address
        self._packets_dropped_in_trio = 0
        self._client_hello = None
        self._did_handshake = False
        # These are mandatory for all DTLS connections. OP_NO_QUERY_MTU is required to
        # stop openssl from trying to query the memory BIO's MTU and then breaking, and
        # OP_NO_RENEGOTIATION disables renegotiation, which is too complex for us to
        # support and isn't useful anyway -- especially for DTLS where it's equivalent
        # to just performing a new handshake.
        ctx.set_options(
            SSL.OP_NO_QUERY_MTU | SSL.OP_NO_RENEGOTIATION,  # type: ignore[attr-defined]
        )
        self._ssl = SSL.Connection(ctx)
        self._handshake_mtu = 0
        # This calls self._ssl.set_ciphertext_mtu, which is important, because if you
        # don't call it then openssl doesn't work.
        self.set_ciphertext_mtu(best_guess_mtu(self.endpoint.socket))
        self._replaced = False
        self._closed = False
        self._q = _Queue[bytes](endpoint.incoming_packets_buffer)
        self._handshake_lock = trio.Lock()
        self._record_encoder: RecordEncoder = RecordEncoder()

        self._final_volley: list[_AnyHandshakeMessage] = []

    def _set_replaced(self) -> None:
        self._replaced = True
        # Any packets we already received could maybe possibly still be processed, but
        # there are no more coming. So we close this on the sender side.
        self._q.s.close()

    def _check_replaced(self) -> None:
        if self._replaced:
            raise trio.BrokenResourceError(
                "peer tore down this connection to start a new one",
            )

    # XX on systems where we can (maybe just Linux?) take advantage of the kernel's PMTU
    # estimate

    # XX should we send close-notify when closing? It seems particularly pointless for
    # DTLS where packets are all independent and can be lost anyway. We do at least need
    # to handle receiving it properly though, which might be easier if we send it...

    def close(self) -> None:
        """Close this connection.

        `DTLSChannel`\\s don't actually own any OS-level resources – the
        socket is owned by the `DTLSEndpoint`, not the individual connections. So
        you don't really *have* to call this. But it will interrupt any other tasks
        calling `receive` with a `ClosedResourceError`, and cause future attempts to use
        this connection to fail.

        You can also use this object as a synchronous or asynchronous context manager.

        """
        if self._closed:
            return
        self._closed = True
        if self.endpoint._streams.get(self.peer_address) is self:
            del self.endpoint._streams[self.peer_address]
        # Will wake any tasks waiting on self._q.get with a
        # ClosedResourceError
        self._q.r.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return self.close()

    async def aclose(self) -> None:
        """Close this connection, but asynchronously.

        This is included to satisfy the `trio.abc.Channel` contract. It's
        identical to `close`, but async.

        """
        self.close()
        await trio.lowlevel.checkpoint()

    async def _send_volley(self, volley_messages: list[_AnyHandshakeMessage]) -> None:
        packets = self._record_encoder.encode_volley(
            volley_messages,
            self._handshake_mtu,
        )
        for packet in packets:
            async with self.endpoint._send_lock:
                await self.endpoint.socket.sendto(packet, self.peer_address)

    async def _resend_final_volley(self) -> None:
        await self._send_volley(self._final_volley)

    async def do_handshake(self, *, initial_retransmit_timeout: float = 1.0) -> None:
        """Perform the handshake.

        Calling this is optional – if you don't, then it will be automatically called
        the first time you call `send` or `receive`. But calling it explicitly can be
        useful in case you want to control the retransmit timeout, use a cancel scope to
        place an overall timeout on the handshake, or catch errors from the handshake
        specifically.

        It's safe to call this multiple times, or call it simultaneously from multiple
        tasks – the first call will perform the handshake, and the rest will be no-ops.

        Args:

          initial_retransmit_timeout (float): Since UDP is an unreliable protocol, it's
            possible that some of the packets we send during the handshake will get
            lost. To handle this, DTLS uses a timer to automatically retransmit
            handshake packets that don't receive a response. This lets you set the
            timeout we use to detect packet loss. Ideally, it should be set to ~1.5
            times the round-trip time to your peer, but 1 second is a reasonable
            default. There's `some useful guidance here
            <https://tlswg.org/dtls13-spec/draft-ietf-tls-dtls13.html#name-timer-values>`__.

            This is the *initial* timeout, because if packets keep being lost then Trio
            will automatically back off to longer values, to avoid overloading the
            network.

        """
        async with self._handshake_lock:
            if self._did_handshake:
                return

            timeout = initial_retransmit_timeout
            volley_messages: list[_AnyHandshakeMessage] = []
            volley_failed_sends = 0

            def read_volley() -> list[_AnyHandshakeMessage]:
                volley_bytes = _read_loop(self._ssl.bio_read)
                new_volley_messages = decode_volley_trusted(volley_bytes)
                if (
                    new_volley_messages
                    and volley_messages
                    and isinstance(new_volley_messages[0], HandshakeMessage)
                    and isinstance(volley_messages[0], HandshakeMessage)
                    and new_volley_messages[0].msg_seq == volley_messages[0].msg_seq
                ):
                    # openssl decided to retransmit; discard because we handle
                    # retransmits ourselves
                    return []
                else:
                    return new_volley_messages

            # If we're a client, we send the initial volley. If we're a server, then
            # the initial ClientHello has already been inserted into self._ssl's
            # read BIO. So either way, we start by generating a new volley.
            with contextlib.suppress(SSL.WantReadError):
                self._ssl.do_handshake()
            volley_messages = read_volley()
            # If we don't have messages to send in our initial volley, then something
            # has gone very wrong. (I'm not sure this can actually happen without an
            # error from OpenSSL, but we check just in case.)
            if not volley_messages:  # pragma: no cover
                raise SSL.Error("something wrong with peer's ClientHello")

            while True:
                # -- at this point, we need to either send or re-send a volley --
                assert volley_messages
                self._check_replaced()
                await self._send_volley(volley_messages)
                # -- then this is where we wait for a reply --
                self.endpoint._ensure_receive_loop()
                with trio.move_on_after(timeout) as cscope:
                    async for packet in self._q.r:
                        self._ssl.bio_write(packet)
                        try:
                            self._ssl.do_handshake()
                        # We ignore generic SSL.Error here, because you can get those
                        # from random invalid packets
                        except (SSL.WantReadError, SSL.Error):
                            pass
                        else:
                            # No exception -> the handshake is done, and we can
                            # switch into data transfer mode.
                            self._did_handshake = True
                            # Might be empty, but that's ok -- we'll just send no
                            # packets.
                            self._final_volley = read_volley()
                            await self._send_volley(self._final_volley)
                            return
                        maybe_volley = read_volley()
                        if maybe_volley:
                            if (
                                isinstance(maybe_volley[0], PseudoHandshakeMessage)
                                and maybe_volley[0].content_type == ContentType.alert
                            ):  # TODO: test this line
                                # we're sending an alert (e.g. due to a corrupted
                                # packet). We want to send it once, but don't save it to
                                # retransmit -- keep the last volley as the current
                                # volley.
                                await self._send_volley(maybe_volley)
                            else:
                                # We managed to get all of the peer's volley and
                                # generate a new one ourselves! break out of the 'for'
                                # loop and restart the timer.
                                volley_messages = maybe_volley
                                # "Implementations SHOULD retain the current timer value
                                # until a transmission without loss occurs, at which
                                # time the value may be reset to the initial value."
                                if volley_failed_sends == 0:
                                    timeout = initial_retransmit_timeout
                                volley_failed_sends = 0
                                break
                    else:
                        assert self._replaced
                        self._check_replaced()
                if cscope.cancelled_caught:
                    # Timeout expired. Double timeout for backoff, with a limit of 60
                    # seconds (this matches what openssl does, and also the
                    # recommendation in draft-ietf-tls-dtls13).
                    timeout = min(2 * timeout, 60.0)
                    volley_failed_sends += 1
                    if volley_failed_sends == 2:
                        # We tried sending this twice and they both failed. Maybe our
                        # PMTU estimate is wrong? Let's try dropping it to the minimum
                        # and hope that helps.
                        self._handshake_mtu = min(
                            self._handshake_mtu,
                            worst_case_mtu(self.endpoint.socket),
                        )

    async def send(self, data: bytes) -> None:
        """Send a packet of data, securely."""

        if self._closed:
            raise trio.ClosedResourceError
        if not data:
            raise ValueError("openssl doesn't support sending empty DTLS packets")
        if not self._did_handshake:
            await self.do_handshake()
        self._check_replaced()
        self._ssl.write(data)
        async with self.endpoint._send_lock:
            await self.endpoint.socket.sendto(
                _read_loop(self._ssl.bio_read),
                self.peer_address,
            )

    async def receive(self) -> bytes:
        """Fetch the next packet of data from this connection's peer, waiting if
        necessary.

        This is safe to call from multiple tasks simultaneously, in case you have some
        reason to do that. And more importantly, it's cancellation-safe, meaning that
        cancelling a call to `receive` will never cause a packet to be lost or corrupt
        the underlying connection.

        """
        if not self._did_handshake:
            await self.do_handshake()
        # If the packet isn't really valid, then openssl can decode it to the empty
        # string (e.g. b/c it's a late-arriving handshake packet, or a duplicate copy of
        # a data packet). Skip over these instead of returning them.
        while True:
            try:
                packet = await self._q.r.receive()
            except trio.EndOfChannel:
                assert self._replaced
                self._check_replaced()
            self._ssl.bio_write(packet)
            cleartext = _read_loop(self._ssl.read)
            if cleartext:
                return cleartext

    def set_ciphertext_mtu(self, new_mtu: int) -> None:
        """Tells Trio the `largest amount of data that can be sent in a single packet to
        this peer <https://en.wikipedia.org/wiki/Maximum_transmission_unit>`__.

        Trio doesn't actually enforce this limit – if you pass a huge packet to `send`,
        then we'll dutifully encrypt it and attempt to send it. But calling this method
        does have two useful effects:

        - If called before the handshake is performed, then Trio will automatically
          fragment handshake messages to fit within the given MTU. It also might
          fragment them even smaller, if it detects signs of packet loss, so setting
          this should never be necessary to make a successful connection. But, the
          packet loss detection only happens after multiple timeouts have expired, so if
          you have reason to believe that a smaller MTU is required, then you can set
          this to skip those timeouts and establish the connection more quickly.

        - It changes the value returned from `get_cleartext_mtu`. So if you have some
          kind of estimate of the network-level MTU, then you can use this to figure out
          how much overhead DTLS will need for hashes/padding/etc., and how much space
          you have left for your application data.

        The MTU here is measuring the largest UDP *payload* you think can be sent, the
        amount of encrypted data that can be handed to the operating system in a single
        call to `send`. It should *not* include IP/UDP headers. Note that OS estimates
        of the MTU often are link-layer MTUs, so you have to subtract off 28 bytes on
        IPv4 and 48 bytes on IPv6 to get the ciphertext MTU.

        By default, Trio assumes an MTU of 1472 bytes on IPv4, and 1452 bytes on IPv6,
        which correspond to the common Ethernet MTU of 1500 bytes after accounting for
        IP/UDP overhead.

        """
        self._handshake_mtu = new_mtu
        self._ssl.set_ciphertext_mtu(new_mtu)

    def get_cleartext_mtu(self) -> int:
        """Returns the largest number of bytes that you can pass in a single call to
        `send` while still fitting within the network-level MTU.

        See `set_ciphertext_mtu` for more details.

        """
        if not self._did_handshake:
            raise trio.NeedHandshakeError
        return self._ssl.get_cleartext_mtu()  # type: ignore[no-any-return]

    def statistics(self) -> DTLSChannelStatistics:
        """Returns a `DTLSChannelStatistics` object with statistics about this connection."""
        return DTLSChannelStatistics(self._packets_dropped_in_trio)


@final
class DTLSEndpoint:
    """A DTLS endpoint.

    A single UDP socket can handle arbitrarily many DTLS connections simultaneously,
    acting as a client or server as needed. A `DTLSEndpoint` object holds a UDP socket
    and manages these connections, which are represented as `DTLSChannel` objects.

    Args:
      socket: (trio.socket.SocketType): A ``SOCK_DGRAM`` socket. If you want to accept
        incoming connections in server mode, then you should probably bind the socket to
        some known port.
      incoming_packets_buffer (int): Each `DTLSChannel` using this socket has its own
        buffer that holds incoming packets until you call `~DTLSChannel.receive` to read
        them. This lets you adjust the size of this buffer. `~DTLSChannel.statistics`
        lets you check if the buffer has overflowed.

    .. attribute:: socket
                   incoming_packets_buffer

       Both constructor arguments are also exposed as attributes, in case you need to
       access them later.

    """

    def __init__(
        self,
        socket: SocketType,
        *,
        incoming_packets_buffer: int = 10,
    ) -> None:
        # We do this lazily on first construction, so only people who actually use DTLS
        # have to install PyOpenSSL.
        global SSL
        from OpenSSL import SSL

        # for __del__, in case the next line raises
        self._initialized: bool = False
        if socket.type != trio.socket.SOCK_DGRAM:
            raise ValueError("DTLS requires a SOCK_DGRAM socket")
        self._initialized = True
        self.socket: SocketType = socket

        self.incoming_packets_buffer = incoming_packets_buffer
        self._token = trio.lowlevel.current_trio_token()
        # We don't need to track handshaking vs non-handshake connections
        # separately. We only keep one connection per remote address; as soon
        # as a peer provides a valid cookie, we can immediately tear down the
        # old connection.
        # {remote address: DTLSChannel}
        self._streams: WeakValueDictionary[AddressFormat, DTLSChannel] = (
            WeakValueDictionary()
        )
        self._listening_context: SSL.Context | None = None
        self._listening_key: bytes | None = None
        self._incoming_connections_q = _Queue[DTLSChannel](float("inf"))
        self._send_lock = trio.Lock()
        self._closed = False
        self._receive_loop_spawned = False

    def _ensure_receive_loop(self) -> None:
        # We have to spawn this lazily, because on Windows it will immediately error out
        # if the socket isn't already bound -- which for clients might not happen until
        # after we send our first packet.
        if not self._receive_loop_spawned:
            trio.lowlevel.spawn_system_task(
                dtls_receive_loop,
                weakref.ref(self),
                self.socket,
            )
            self._receive_loop_spawned = True

    def __del__(self) -> None:
        # Do nothing if this object was never fully constructed
        if not self._initialized:
            return
        # Close the socket in Trio context (if our Trio context still exists), so that
        # the background task gets notified about the closure and can exit.
        if not self._closed:
            with contextlib.suppress(RuntimeError):
                self._token.run_sync_soon(self.close)
            # Do this last, because it might raise an exception
            warnings.warn(
                f"unclosed DTLS endpoint {self!r}",
                ResourceWarning,
                source=self,
                stacklevel=1,
            )

    def close(self) -> None:
        """Close this socket, and all associated DTLS connections.

        This object can also be used as a context manager.

        """
        self._closed = True
        self.socket.close()
        for stream in list(self._streams.values()):
            stream.close()
        self._incoming_connections_q.s.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return self.close()

    def _check_closed(self) -> None:
        if self._closed:
            raise trio.ClosedResourceError

    async def serve(
        self,
        ssl_context: SSL.Context,
        async_fn: Callable[[DTLSChannel, Unpack[PosArgsT]], Awaitable[object]],
        *args: Unpack[PosArgsT],
        task_status: trio.TaskStatus[None] = trio.TASK_STATUS_IGNORED,
    ) -> None:
        """Listen for incoming connections, and spawn a handler for each using an
        internal nursery.

        Similar to `~trio.serve_tcp`, this function never returns until cancelled, or
        the `DTLSEndpoint` is closed and all handlers have exited.

        Usage commonly looks like::

            async def handler(dtls_channel):
                ...

            async with trio.open_nursery() as nursery:
                await nursery.start(dtls_endpoint.serve, ssl_context, handler)
                # ... do other things here ...

        The ``dtls_channel`` passed into the handler function has already performed the
        "cookie exchange" part of the DTLS handshake, so the peer address is
        trustworthy. But the actual cryptographic handshake doesn't happen until you
        start using it, giving you a chance for any last minute configuration, and the
        option to catch and handle handshake errors.

        Args:
          ssl_context (OpenSSL.SSL.Context): The PyOpenSSL context object to use for
            incoming connections.
          async_fn: The handler function that will be invoked for each incoming
            connection.
          *args: Additional arguments to pass to the handler function.

        """
        self._check_closed()
        if self._listening_context is not None:
            raise trio.BusyResourceError("another task is already listening")
        try:
            self.socket.getsockname()
        except OSError:  # TODO: test this line
            raise RuntimeError(
                "DTLS socket must be bound before it can serve",
            ) from None
        self._ensure_receive_loop()
        # We do cookie verification ourselves, so tell OpenSSL not to worry about it.
        # (See also _inject_client_hello_untrusted.)
        ssl_context.set_cookie_verify_callback(lambda *_: True)
        try:
            self._listening_context = ssl_context
            task_status.started()

            async def handler_wrapper(stream: DTLSChannel) -> None:
                with stream:
                    await async_fn(stream, *args)

            async with trio.open_nursery() as nursery:
                async for stream in self._incoming_connections_q.r:  # pragma: no branch
                    nursery.start_soon(handler_wrapper, stream)
        finally:
            self._listening_context = None

    def connect(
        self,
        address: tuple[str, int],
        ssl_context: SSL.Context,
    ) -> DTLSChannel:
        """Initiate an outgoing DTLS connection.

        Notice that this is a synchronous method. That's because it doesn't actually
        initiate any I/O – it just sets up a `DTLSChannel` object. The actual handshake
        doesn't occur until you start using the `DTLSChannel`. This gives you a chance
        to do further configuration first, like setting MTU etc.

        Args:
          address: The address to connect to. Usually a (host, port) tuple, like
            ``("127.0.0.1", 12345)``.
          ssl_context (OpenSSL.SSL.Context): The PyOpenSSL context object to use for
            this connection.

        Returns:
          DTLSChannel

        """
        # it would be nice if we could detect when 'address' is our own endpoint (a
        # loopback connection), because that can't work
        # but I don't see how to do it reliably
        self._check_closed()
        channel = DTLSChannel._create(self, address, ssl_context)
        channel._ssl.set_connect_state()
        old_channel = self._streams.get(address)
        if old_channel is not None:
            old_channel._set_replaced()
        self._streams[address] = channel
        return channel

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\domain.py ===
"""Implementation of :class:`Domain` class. """

from __future__ import annotations
from typing import Any

from sympy.core.numbers import AlgebraicNumber
from sympy.core import Basic, sympify
from sympy.core.sorting import ordered
from sympy.external.gmpy import GROUND_TYPES
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import UnificationFailed, CoercionFailed, DomainError
from sympy.polys.polyutils import _unify_gens, _not_a_coeff
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence


@public
class Domain:
    """Superclass for all domains in the polys domains system.

    See :ref:`polys-domainsintro` for an introductory explanation of the
    domains system.

    The :py:class:`~.Domain` class is an abstract base class for all of the
    concrete domain types. There are many different :py:class:`~.Domain`
    subclasses each of which has an associated ``dtype`` which is a class
    representing the elements of the domain. The coefficients of a
    :py:class:`~.Poly` are elements of a domain which must be a subclass of
    :py:class:`~.Domain`.

    Examples
    ========

    The most common example domains are the integers :ref:`ZZ` and the
    rationals :ref:`QQ`.

    >>> from sympy import Poly, symbols, Domain
    >>> x, y = symbols('x, y')
    >>> p = Poly(x**2 + y)
    >>> p
    Poly(x**2 + y, x, y, domain='ZZ')
    >>> p.domain
    ZZ
    >>> isinstance(p.domain, Domain)
    True
    >>> Poly(x**2 + y/2)
    Poly(x**2 + 1/2*y, x, y, domain='QQ')

    The domains can be used directly in which case the domain object e.g.
    (:ref:`ZZ` or :ref:`QQ`) can be used as a constructor for elements of
    ``dtype``.

    >>> from sympy import ZZ, QQ
    >>> ZZ(2)
    2
    >>> ZZ.dtype  # doctest: +SKIP
    <class 'int'>
    >>> type(ZZ(2))  # doctest: +SKIP
    <class 'int'>
    >>> QQ(1, 2)
    1/2
    >>> type(QQ(1, 2))  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    The corresponding domain elements can be used with the arithmetic
    operations ``+,-,*,**`` and depending on the domain some combination of
    ``/,//,%`` might be usable. For example in :ref:`ZZ` both ``//`` (floor
    division) and ``%`` (modulo division) can be used but ``/`` (true
    division) cannot. Since :ref:`QQ` is a :py:class:`~.Field` its elements
    can be used with ``/`` but ``//`` and ``%`` should not be used. Some
    domains have a :py:meth:`~.Domain.gcd` method.

    >>> ZZ(2) + ZZ(3)
    5
    >>> ZZ(5) // ZZ(2)
    2
    >>> ZZ(5) % ZZ(2)
    1
    >>> QQ(1, 2) / QQ(2, 3)
    3/4
    >>> ZZ.gcd(ZZ(4), ZZ(2))
    2
    >>> QQ.gcd(QQ(2,7), QQ(5,3))
    1/21
    >>> ZZ.is_Field
    False
    >>> QQ.is_Field
    True

    There are also many other domains including:

        1. :ref:`GF(p)` for finite fields of prime order.
        2. :ref:`RR` for real (floating point) numbers.
        3. :ref:`CC` for complex (floating point) numbers.
        4. :ref:`QQ(a)` for algebraic number fields.
        5. :ref:`K[x]` for polynomial rings.
        6. :ref:`K(x)` for rational function fields.
        7. :ref:`EX` for arbitrary expressions.

    Each domain is represented by a domain object and also an implementation
    class (``dtype``) for the elements of the domain. For example the
    :ref:`K[x]` domains are represented by a domain object which is an
    instance of :py:class:`~.PolynomialRing` and the elements are always
    instances of :py:class:`~.PolyElement`. The implementation class
    represents particular types of mathematical expressions in a way that is
    more efficient than a normal SymPy expression which is of type
    :py:class:`~.Expr`. The domain methods :py:meth:`~.Domain.from_sympy` and
    :py:meth:`~.Domain.to_sympy` are used to convert from :py:class:`~.Expr`
    to a domain element and vice versa.

    >>> from sympy import Symbol, ZZ, Expr
    >>> x = Symbol('x')
    >>> K = ZZ[x]           # polynomial ring domain
    >>> K
    ZZ[x]
    >>> type(K)             # class of the domain
    <class 'sympy.polys.domains.polynomialring.PolynomialRing'>
    >>> K.dtype             # doctest: +SKIP
    <class 'sympy.polys.rings.PolyElement'>
    >>> p_expr = x**2 + 1   # Expr
    >>> p_expr
    x**2 + 1
    >>> type(p_expr)
    <class 'sympy.core.add.Add'>
    >>> isinstance(p_expr, Expr)
    True
    >>> p_domain = K.from_sympy(p_expr)
    >>> p_domain            # domain element
    x**2 + 1
    >>> type(p_domain)
    <class 'sympy.polys.rings.PolyElement'>
    >>> K.to_sympy(p_domain) == p_expr
    True

    The :py:meth:`~.Domain.convert_from` method is used to convert domain
    elements from one domain to another.

    >>> from sympy import ZZ, QQ
    >>> ez = ZZ(2)
    >>> eq = QQ.convert_from(ez, ZZ)
    >>> type(ez)  # doctest: +SKIP
    <class 'int'>
    >>> type(eq)  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    Elements from different domains should not be mixed in arithmetic or other
    operations: they should be converted to a common domain first.  The domain
    method :py:meth:`~.Domain.unify` is used to find a domain that can
    represent all the elements of two given domains.

    >>> from sympy import ZZ, QQ, symbols
    >>> x, y = symbols('x, y')
    >>> ZZ.unify(QQ)
    QQ
    >>> ZZ[x].unify(QQ)
    QQ[x]
    >>> ZZ[x].unify(QQ[y])
    QQ[x,y]

    If a domain is a :py:class:`~.Ring` then is might have an associated
    :py:class:`~.Field` and vice versa. The :py:meth:`~.Domain.get_field` and
    :py:meth:`~.Domain.get_ring` methods will find or create the associated
    domain.

    >>> from sympy import ZZ, QQ, Symbol
    >>> x = Symbol('x')
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ
    >>> QQ.has_assoc_Ring
    True
    >>> QQ.get_ring()
    ZZ
    >>> K = QQ[x]
    >>> K
    QQ[x]
    >>> K.get_field()
    QQ(x)

    See also
    ========

    DomainElement: abstract base class for domain elements
    construct_domain: construct a minimal domain for some expressions

    """

    dtype: type | None = None
    """The type (class) of the elements of this :py:class:`~.Domain`:

    >>> from sympy import ZZ, QQ, Symbol
    >>> ZZ.dtype
    <class 'int'>
    >>> z = ZZ(2)
    >>> z
    2
    >>> type(z)
    <class 'int'>
    >>> type(z) == ZZ.dtype
    True

    Every domain has an associated **dtype** ("datatype") which is the
    class of the associated domain elements.

    See also
    ========

    of_type
    """

    zero: Any = None
    """The zero element of the :py:class:`~.Domain`:

    >>> from sympy import QQ
    >>> QQ.zero
    0
    >>> QQ.of_type(QQ.zero)
    True

    See also
    ========

    of_type
    one
    """

    one: Any = None
    """The one element of the :py:class:`~.Domain`:

    >>> from sympy import QQ
    >>> QQ.one
    1
    >>> QQ.of_type(QQ.one)
    True

    See also
    ========

    of_type
    zero
    """

    is_Ring = False
    """Boolean flag indicating if the domain is a :py:class:`~.Ring`.

    >>> from sympy import ZZ
    >>> ZZ.is_Ring
    True

    Basically every :py:class:`~.Domain` represents a ring so this flag is
    not that useful.

    See also
    ========

    is_PID
    is_Field
    get_ring
    has_assoc_Ring
    """

    is_Field = False
    """Boolean flag indicating if the domain is a :py:class:`~.Field`.

    >>> from sympy import ZZ, QQ
    >>> ZZ.is_Field
    False
    >>> QQ.is_Field
    True

    See also
    ========

    is_PID
    is_Ring
    get_field
    has_assoc_Field
    """

    has_assoc_Ring = False
    """Boolean flag indicating if the domain has an associated
    :py:class:`~.Ring`.

    >>> from sympy import QQ
    >>> QQ.has_assoc_Ring
    True
    >>> QQ.get_ring()
    ZZ

    See also
    ========

    is_Field
    get_ring
    """

    has_assoc_Field = False
    """Boolean flag indicating if the domain has an associated
    :py:class:`~.Field`.

    >>> from sympy import ZZ
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ

    See also
    ========

    is_Field
    get_field
    """

    is_FiniteField = is_FF = False
    is_IntegerRing = is_ZZ = False
    is_RationalField = is_QQ = False
    is_GaussianRing = is_ZZ_I = False
    is_GaussianField = is_QQ_I = False
    is_RealField = is_RR = False
    is_ComplexField = is_CC = False
    is_AlgebraicField = is_Algebraic = False
    is_PolynomialRing = is_Poly = False
    is_FractionField = is_Frac = False
    is_SymbolicDomain = is_EX = False
    is_SymbolicRawDomain = is_EXRAW = False
    is_FiniteExtension = False

    is_Exact = True
    is_Numerical = False

    is_Simple = False
    is_Composite = False

    is_PID = False
    """Boolean flag indicating if the domain is a `principal ideal domain`_.

    >>> from sympy import ZZ
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ

    .. _principal ideal domain: https://en.wikipedia.org/wiki/Principal_ideal_domain

    See also
    ========

    is_Field
    get_field
    """

    has_CharacteristicZero = False

    rep: str | None = None
    alias: str | None = None

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return self.rep

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype))

    def new(self, *args):
        return self.dtype(*args)

    @property
    def tp(self):
        """Alias for :py:attr:`~.Domain.dtype`"""
        return self.dtype

    def __call__(self, *args):
        """Construct an element of ``self`` domain from ``args``. """
        return self.new(*args)

    def normal(self, *args):
        return self.dtype(*args)

    def convert_from(self, element, base):
        """Convert ``element`` to ``self.dtype`` given the base domain. """
        if base.alias is not None:
            method = "from_" + base.alias
        else:
            method = "from_" + base.__class__.__name__

        _convert = getattr(self, method)

        if _convert is not None:
            result = _convert(element, base)

            if result is not None:
                return result

        raise CoercionFailed("Cannot convert %s of type %s from %s to %s" % (element, type(element), base, self))

    def convert(self, element, base=None):
        """Convert ``element`` to ``self.dtype``. """

        if base is not None:
            if _not_a_coeff(element):
                raise CoercionFailed('%s is not in any domain' % element)
            return self.convert_from(element, base)

        if self.of_type(element):
            return element

        if _not_a_coeff(element):
            raise CoercionFailed('%s is not in any domain' % element)

        from sympy.polys.domains import ZZ, QQ, RealField, ComplexField

        if ZZ.of_type(element):
            return self.convert_from(element, ZZ)

        if isinstance(element, int):
            return self.convert_from(ZZ(element), ZZ)

        if GROUND_TYPES != 'python':
            if isinstance(element, ZZ.tp):
                return self.convert_from(element, ZZ)
            if isinstance(element, QQ.tp):
                return self.convert_from(element, QQ)

        if isinstance(element, float):
            parent = RealField()
            return self.convert_from(parent(element), parent)

        if isinstance(element, complex):
            parent = ComplexField()
            return self.convert_from(parent(element), parent)

        if type(element).__name__ == 'mpf':
            parent = RealField()
            return self.convert_from(parent(element), parent)

        if type(element).__name__ == 'mpc':
            parent = ComplexField()
            return self.convert_from(parent(element), parent)

        if isinstance(element, DomainElement):
            return self.convert_from(element, element.parent())

        # TODO: implement this in from_ methods
        if self.is_Numerical and getattr(element, 'is_ground', False):
            return self.convert(element.LC())

        if isinstance(element, Basic):
            try:
                return self.from_sympy(element)
            except (TypeError, ValueError):
                pass
        else: # TODO: remove this branch
            if not is_sequence(element):
                try:
                    element = sympify(element, strict=True)
                    if isinstance(element, Basic):
                        return self.from_sympy(element)
                except (TypeError, ValueError):
                    pass

        raise CoercionFailed("Cannot convert %s of type %s to %s" % (element, type(element), self))

    def of_type(self, element):
        """Check if ``a`` is of type ``dtype``. """
        return isinstance(element, self.tp)

    def __contains__(self, a):
        """Check if ``a`` belongs to this domain. """
        try:
            if _not_a_coeff(a):
                raise CoercionFailed
            self.convert(a)  # this might raise, too
        except CoercionFailed:
            return False

        return True

    def to_sympy(self, a):
        """Convert domain element *a* to a SymPy expression (Expr).

        Explanation
        ===========

        Convert a :py:class:`~.Domain` element *a* to :py:class:`~.Expr`. Most
        public SymPy functions work with objects of type :py:class:`~.Expr`.
        The elements of a :py:class:`~.Domain` have a different internal
        representation. It is not possible to mix domain elements with
        :py:class:`~.Expr` so each domain has :py:meth:`~.Domain.to_sympy` and
        :py:meth:`~.Domain.from_sympy` methods to convert its domain elements
        to and from :py:class:`~.Expr`.

        Parameters
        ==========

        a: domain element
            An element of this :py:class:`~.Domain`.

        Returns
        =======

        expr: Expr
            A normal SymPy expression of type :py:class:`~.Expr`.

        Examples
        ========

        Construct an element of the :ref:`QQ` domain and then convert it to
        :py:class:`~.Expr`.

        >>> from sympy import QQ, Expr
        >>> q_domain = QQ(2)
        >>> q_domain
        2
        >>> q_expr = QQ.to_sympy(q_domain)
        >>> q_expr
        2

        Although the printed forms look similar these objects are not of the
        same type.

        >>> isinstance(q_domain, Expr)
        False
        >>> isinstance(q_expr, Expr)
        True

        Construct an element of :ref:`K[x]` and convert to
        :py:class:`~.Expr`.

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> K = QQ[x]
        >>> x_domain = K.gens[0]  # generator x as a domain element
        >>> p_domain = x_domain**2/3 + 1
        >>> p_domain
        1/3*x**2 + 1
        >>> p_expr = K.to_sympy(p_domain)
        >>> p_expr
        x**2/3 + 1

        The :py:meth:`~.Domain.from_sympy` method is used for the opposite
        conversion from a normal SymPy expression to a domain element.

        >>> p_domain == p_expr
        False
        >>> K.from_sympy(p_expr) == p_domain
        True
        >>> K.to_sympy(p_domain) == p_expr
        True
        >>> K.from_sympy(K.to_sympy(p_domain)) == p_domain
        True
        >>> K.to_sympy(K.from_sympy(p_expr)) == p_expr
        True

        The :py:meth:`~.Domain.from_sympy` method makes it easier to construct
        domain elements interactively.

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> K = QQ[x]
        >>> K.from_sympy(x**2/3 + 1)
        1/3*x**2 + 1

        See also
        ========

        from_sympy
        convert_from
        """
        raise NotImplementedError

    def from_sympy(self, a):
        """Convert a SymPy expression to an element of this domain.

        Explanation
        ===========

        See :py:meth:`~.Domain.to_sympy` for explanation and examples.

        Parameters
        ==========

        expr: Expr
            A normal SymPy expression of type :py:class:`~.Expr`.

        Returns
        =======

        a: domain element
            An element of this :py:class:`~.Domain`.

        See also
        ========

        to_sympy
        convert_from
        """
        raise NotImplementedError

    def sum(self, args):
        return sum(args, start=self.zero)

    def from_FF(K1, a, K0):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return None

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return None

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return None

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return None

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to ``dtype``. """
        return None

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return None

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return None

    def from_RealField(K1, a, K0):
        """Convert a real element object to ``dtype``. """
        return None

    def from_ComplexField(K1, a, K0):
        """Convert a complex element to ``dtype``. """
        return None

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        return None

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        if a.is_ground:
            return K1.convert(a.LC, K0.dom)

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        return None

    def from_MonogenicFiniteExtension(K1, a, K0):
        """Convert an ``ExtensionElement`` to ``dtype``. """
        return K1.convert_from(a.rep, K0.ring)

    def from_ExpressionDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
        return K1.from_sympy(a.ex)

    def from_ExpressionRawDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
        return K1.from_sympy(a)

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        if a.degree() <= 0:
            return K1.convert(a.LC(), K0.dom)

    def from_GeneralizedPolynomialRing(K1, a, K0):
        return K1.from_FractionField(a, K0)

    def unify_with_symbols(K0, K1, symbols):
        if (K0.is_Composite and (set(K0.symbols) & set(symbols))) or (K1.is_Composite and (set(K1.symbols) & set(symbols))):
            raise UnificationFailed("Cannot unify %s with %s, given %s generators" % (K0, K1, tuple(symbols)))

        return K0.unify(K1)

    def unify_composite(K0, K1):
        """Unify two domains where at least one is composite."""
        K0_ground = K0.dom if K0.is_Composite else K0
        K1_ground = K1.dom if K1.is_Composite else K1

        K0_symbols = K0.symbols if K0.is_Composite else ()
        K1_symbols = K1.symbols if K1.is_Composite else ()

        domain = K0_ground.unify(K1_ground)
        symbols = _unify_gens(K0_symbols, K1_symbols)
        order = K0.order if K0.is_Composite else K1.order

        # E.g. ZZ[x].unify(QQ.frac_field(x)) -> ZZ.frac_field(x)
        if ((K0.is_FractionField and K1.is_PolynomialRing or
             K1.is_FractionField and K0.is_PolynomialRing) and
             (not K0_ground.is_Field or not K1_ground.is_Field) and domain.is_Field
             and domain.has_assoc_Ring):
            domain = domain.get_ring()

        if K0.is_Composite and (not K1.is_Composite or K0.is_FractionField or K1.is_PolynomialRing):
            cls = K0.__class__
        else:
            cls = K1.__class__

        # Here cls might be PolynomialRing, FractionField, GlobalPolynomialRing
        # (dense/old Polynomialring) or dense/old FractionField.

        from sympy.polys.domains.old_polynomialring import GlobalPolynomialRing
        if cls == GlobalPolynomialRing:
            return cls(domain, symbols)

        return cls(domain, symbols, order)

    def unify(K0, K1, symbols=None):
        """
        Construct a minimal domain that contains elements of ``K0`` and ``K1``.

        Known domains (from smallest to largest):

        - ``GF(p)``
        - ``ZZ``
        - ``QQ``
        - ``RR(prec, tol)``
        - ``CC(prec, tol)``
        - ``ALG(a, b, c)``
        - ``K[x, y, z]``
        - ``K(x, y, z)``
        - ``EX``

        """
        if symbols is not None:
            return K0.unify_with_symbols(K1, symbols)

        if K0 == K1:
            return K0

        if not (K0.has_CharacteristicZero and K1.has_CharacteristicZero):
            # Reject unification of domains with different characteristics.
            if K0.characteristic() != K1.characteristic():
                raise UnificationFailed("Cannot unify %s with %s" % (K0, K1))

            # We do not get here if K0 == K1. The two domains have the same
            # characteristic but are unequal so at least one is composite and
            # we are unifying something like GF(3).unify(GF(3)[x]).
            return K0.unify_composite(K1)

        # From here we know both domains have characteristic zero and it can be
        # acceptable to fall back on EX.

        if K0.is_EXRAW:
            return K0
        if K1.is_EXRAW:
            return K1

        if K0.is_EX:
            return K0
        if K1.is_EX:
            return K1

        if K0.is_FiniteExtension or K1.is_FiniteExtension:
            if K1.is_FiniteExtension:
                K0, K1 = K1, K0
            if K1.is_FiniteExtension:
                # Unifying two extensions.
                # Try to ensure that K0.unify(K1) == K1.unify(K0)
                if list(ordered([K0.modulus, K1.modulus]))[1] == K0.modulus:
                    K0, K1 = K1, K0
                return K1.set_domain(K0)
            else:
                # Drop the generator from other and unify with the base domain
                K1 = K1.drop(K0.symbol)
                K1 = K0.domain.unify(K1)
                return K0.set_domain(K1)

        if K0.is_Composite or K1.is_Composite:
            return K0.unify_composite(K1)

        if K1.is_ComplexField:
            K0, K1 = K1, K0
        if K0.is_ComplexField:
            if K1.is_ComplexField or K1.is_RealField:
                if K0.precision >= K1.precision:
                    return K0
                else:
                    from sympy.polys.domains.complexfield import ComplexField
                    return ComplexField(prec=K1.precision)
            else:
                return K0

        if K1.is_RealField:
            K0, K1 = K1, K0
        if K0.is_RealField:
            if K1.is_RealField:
                if K0.precision >= K1.precision:
                    return K0
                else:
                    return K1
            elif K1.is_GaussianRing or K1.is_GaussianField:
                from sympy.polys.domains.complexfield import ComplexField
                return ComplexField(prec=K0.precision)
            else:
                return K0

        if K1.is_AlgebraicField:
            K0, K1 = K1, K0
        if K0.is_AlgebraicField:
            if K1.is_GaussianRing:
                K1 = K1.get_field()
            if K1.is_GaussianField:
                K1 = K1.as_AlgebraicField()
            if K1.is_AlgebraicField:
                return K0.__class__(K0.dom.unify(K1.dom), *_unify_gens(K0.orig_ext, K1.orig_ext))
            else:
                return K0

        if K0.is_GaussianField:
            return K0
        if K1.is_GaussianField:
            return K1

        if K0.is_GaussianRing:
            if K1.is_RationalField:
                K0 = K0.get_field()
            return K0
        if K1.is_GaussianRing:
            if K0.is_RationalField:
                K1 = K1.get_field()
            return K1

        if K0.is_RationalField:
            return K0
        if K1.is_RationalField:
            return K1

        if K0.is_IntegerRing:
            return K0
        if K1.is_IntegerRing:
            return K1

        from sympy.polys.domains import EX
        return EX

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        # XXX: Remove this.
        return isinstance(other, Domain) and self.dtype == other.dtype

    def __ne__(self, other):
        """Returns ``False`` if two domains are equivalent. """
        return not self == other

    def map(self, seq):
        """Rersively apply ``self`` to all elements of ``seq``. """
        result = []

        for elt in seq:
            if isinstance(elt, list):
                result.append(self.map(elt))
            else:
                result.append(self(elt))

        return result

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError('there is no ring associated with %s' % self)

    def get_field(self):
        """Returns a field associated with ``self``. """
        raise DomainError('there is no field associated with %s' % self)

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        return self

    def __getitem__(self, symbols):
        """The mathematical way to make a polynomial ring. """
        if hasattr(symbols, '__iter__'):
            return self.poly_ring(*symbols)
        else:
            return self.poly_ring(symbols)

    def poly_ring(self, *symbols, order=lex):
        """Returns a polynomial ring, i.e. `K[X]`. """
        from sympy.polys.domains.polynomialring import PolynomialRing
        return PolynomialRing(self, symbols, order)

    def frac_field(self, *symbols, order=lex):
        """Returns a fraction field, i.e. `K(X)`. """
        from sympy.polys.domains.fractionfield import FractionField
        return FractionField(self, symbols, order)

    def old_poly_ring(self, *symbols, **kwargs):
        """Returns a polynomial ring, i.e. `K[X]`. """
        from sympy.polys.domains.old_polynomialring import PolynomialRing
        return PolynomialRing(self, *symbols, **kwargs)

    def old_frac_field(self, *symbols, **kwargs):
        """Returns a fraction field, i.e. `K(X)`. """
        from sympy.polys.domains.old_fractionfield import FractionField
        return FractionField(self, *symbols, **kwargs)

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `K(\alpha, \ldots)`. """
        raise DomainError("Cannot create algebraic field over %s" % self)

    def alg_field_from_poly(self, poly, alias=None, root_index=-1):
        r"""
        Convenience method to construct an algebraic extension on a root of a
        polynomial, chosen by root index.

        Parameters
        ==========

        poly : :py:class:`~.Poly`
            The polynomial whose root generates the extension.
        alias : str, optional (default=None)
            Symbol name for the generator of the extension.
            E.g. "alpha" or "theta".
        root_index : int, optional (default=-1)
            Specifies which root of the polynomial is desired. The ordering is
            as defined by the :py:class:`~.ComplexRootOf` class. The default of
            ``-1`` selects the most natural choice in the common cases of
            quadratic and cyclotomic fields (the square root on the positive
            real or imaginary axis, resp. $\mathrm{e}^{2\pi i/n}$).

        Examples
        ========

        >>> from sympy import QQ, Poly
        >>> from sympy.abc import x
        >>> f = Poly(x**2 - 2)
        >>> K = QQ.alg_field_from_poly(f)
        >>> K.ext.minpoly == f
        True
        >>> g = Poly(8*x**3 - 6*x - 1)
        >>> L = QQ.alg_field_from_poly(g, "alpha")
        >>> L.ext.minpoly == g
        True
        >>> L.to_sympy(L([1, 1, 1]))
        alpha**2 + alpha + 1

        """
        from sympy.polys.rootoftools import CRootOf
        root = CRootOf(poly, root_index)
        alpha = AlgebraicNumber(root, alias=alias)
        return self.algebraic_field(alpha, alias=alias)

    def cyclotomic_field(self, n, ss=False, alias="zeta", gen=None, root_index=-1):
        r"""
        Convenience method to construct a cyclotomic field.

        Parameters
        ==========

        n : int
            Construct the nth cyclotomic field.
        ss : boolean, optional (default=False)
            If True, append *n* as a subscript on the alias string.
        alias : str, optional (default="zeta")
            Symbol name for the generator.
        gen : :py:class:`~.Symbol`, optional (default=None)
            Desired variable for the cyclotomic polynomial that defines the
            field. If ``None``, a dummy variable will be used.
        root_index : int, optional (default=-1)
            Specifies which root of the polynomial is desired. The ordering is
            as defined by the :py:class:`~.ComplexRootOf` class. The default of
            ``-1`` selects the root $\mathrm{e}^{2\pi i/n}$.

        Examples
        ========

        >>> from sympy import QQ, latex
        >>> K = QQ.cyclotomic_field(5)
        >>> K.to_sympy(K([-1, 1]))
        1 - zeta
        >>> L = QQ.cyclotomic_field(7, True)
        >>> a = L.to_sympy(L([-1, 1]))
        >>> print(a)
        1 - zeta7
        >>> print(latex(a))
        1 - \zeta_{7}

        """
        from sympy.polys.specialpolys import cyclotomic_poly
        if ss:
            alias += str(n)
        return self.alg_field_from_poly(cyclotomic_poly(n, gen), alias=alias,
                                        root_index=root_index)

    def inject(self, *symbols):
        """Inject generators into this domain. """
        raise NotImplementedError

    def drop(self, *symbols):
        """Drop generators from this domain. """
        if self.is_Simple:
            return self
        raise NotImplementedError  # pragma: no cover

    def is_zero(self, a):
        """Returns True if ``a`` is zero. """
        return not a

    def is_one(self, a):
        """Returns True if ``a`` is one. """
        return a == self.one

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return a > 0

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return a < 0

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return a <= 0

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return a >= 0

    def canonical_unit(self, a):
        if self.is_negative(a):
            return -self.one
        else:
            return self.one

    def abs(self, a):
        """Absolute value of ``a``, implies ``__abs__``. """
        return abs(a)

    def neg(self, a):
        """Returns ``a`` negated, implies ``__neg__``. """
        return -a

    def pos(self, a):
        """Returns ``a`` positive, implies ``__pos__``. """
        return +a

    def add(self, a, b):
        """Sum of ``a`` and ``b``, implies ``__add__``.  """
        return a + b

    def sub(self, a, b):
        """Difference of ``a`` and ``b``, implies ``__sub__``.  """
        return a - b

    def mul(self, a, b):
        """Product of ``a`` and ``b``, implies ``__mul__``.  """
        return a * b

    def pow(self, a, b):
        """Raise ``a`` to power ``b``, implies ``__pow__``.  """
        return a ** b

    def exquo(self, a, b):
        """Exact quotient of *a* and *b*. Analogue of ``a / b``.

        Explanation
        ===========

        This is essentially the same as ``a / b`` except that an error will be
        raised if the division is inexact (if there is any remainder) and the
        result will always be a domain element. When working in a
        :py:class:`~.Domain` that is not a :py:class:`~.Field` (e.g. :ref:`ZZ`
        or :ref:`K[x]`) ``exquo`` should be used instead of ``/``.

        The key invariant is that if ``q = K.exquo(a, b)`` (and ``exquo`` does
        not raise an exception) then ``a == b*q``.

        Examples
        ========

        We can use ``K.exquo`` instead of ``/`` for exact division.

        >>> from sympy import ZZ
        >>> ZZ.exquo(ZZ(4), ZZ(2))
        2
        >>> ZZ.exquo(ZZ(5), ZZ(2))
        Traceback (most recent call last):
            ...
        ExactQuotientFailed: 2 does not divide 5 in ZZ

        Over a :py:class:`~.Field` such as :ref:`QQ`, division (with nonzero
        divisor) is always exact so in that case ``/`` can be used instead of
        :py:meth:`~.Domain.exquo`.

        >>> from sympy import QQ
        >>> QQ.exquo(QQ(5), QQ(2))
        5/2
        >>> QQ(5) / QQ(2)
        5/2

        Parameters
        ==========

        a: domain element
            The dividend
        b: domain element
            The divisor

        Returns
        =======

        q: domain element
            The exact quotient

        Raises
        ======

        ExactQuotientFailed: if exact division is not possible.
        ZeroDivisionError: when the divisor is zero.

        See also
        ========

        quo: Analogue of ``a // b``
        rem: Analogue of ``a % b``
        div: Analogue of ``divmod(a, b)``

        Notes
        =====

        Since the default :py:attr:`~.Domain.dtype` for :ref:`ZZ` is ``int``
        (or ``mpz``) division as ``a / b`` should not be used as it would give
        a ``float`` which is not a domain element.

        >>> ZZ(4) / ZZ(2) # doctest: +SKIP
        2.0
        >>> ZZ(5) / ZZ(2) # doctest: +SKIP
        2.5

        On the other hand with `SYMPY_GROUND_TYPES=flint` elements of :ref:`ZZ`
        are ``flint.fmpz`` and division would raise an exception:

        >>> ZZ(4) / ZZ(2) # doctest: +SKIP
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for /: 'fmpz' and 'fmpz'

        Using ``/`` with :ref:`ZZ` will lead to incorrect results so
        :py:meth:`~.Domain.exquo` should be used instead.

        """
        raise NotImplementedError

    def quo(self, a, b):
        """Quotient of *a* and *b*. Analogue of ``a // b``.

        ``K.quo(a, b)`` is equivalent to ``K.div(a, b)[0]``. See
        :py:meth:`~.Domain.div` for more explanation.

        See also
        ========

        rem: Analogue of ``a % b``
        div: Analogue of ``divmod(a, b)``
        exquo: Analogue of ``a / b``
        """
        raise NotImplementedError

    def rem(self, a, b):
        """Modulo division of *a* and *b*. Analogue of ``a % b``.

        ``K.rem(a, b)`` is equivalent to ``K.div(a, b)[1]``. See
        :py:meth:`~.Domain.div` for more explanation.

        See also
        ========

        quo: Analogue of ``a // b``
        div: Analogue of ``divmod(a, b)``
        exquo: Analogue of ``a / b``
        """
        raise NotImplementedError

    def div(self, a, b):
        """Quotient and remainder for *a* and *b*. Analogue of ``divmod(a, b)``

        Explanation
        ===========

        This is essentially the same as ``divmod(a, b)`` except that is more
        consistent when working over some :py:class:`~.Field` domains such as
        :ref:`QQ`. When working over an arbitrary :py:class:`~.Domain` the
        :py:meth:`~.Domain.div` method should be used instead of ``divmod``.

        The key invariant is that if ``q, r = K.div(a, b)`` then
        ``a == b*q + r``.

        The result of ``K.div(a, b)`` is the same as the tuple
        ``(K.quo(a, b), K.rem(a, b))`` except that if both quotient and
        remainder are needed then it is more efficient to use
        :py:meth:`~.Domain.div`.

        Examples
        ========

        We can use ``K.div`` instead of ``divmod`` for floor division and
        remainder.

        >>> from sympy import ZZ, QQ
        >>> ZZ.div(ZZ(5), ZZ(2))
        (2, 1)

        If ``K`` is a :py:class:`~.Field` then the division is always exact
        with a remainder of :py:attr:`~.Domain.zero`.

        >>> QQ.div(QQ(5), QQ(2))
        (5/2, 0)

        Parameters
        ==========

        a: domain element
            The dividend
        b: domain element
            The divisor

        Returns
        =======

        (q, r): tuple of domain elements
            The quotient and remainder

        Raises
        ======

        ZeroDivisionError: when the divisor is zero.

        See also
        ========

        quo: Analogue of ``a // b``
        rem: Analogue of ``a % b``
        exquo: Analogue of ``a / b``

        Notes
        =====

        If ``gmpy`` is installed then the ``gmpy.mpq`` type will be used as
        the :py:attr:`~.Domain.dtype` for :ref:`QQ`. The ``gmpy.mpq`` type
        defines ``divmod`` in a way that is undesirable so
        :py:meth:`~.Domain.div` should be used instead of ``divmod``.

        >>> a = QQ(1)
        >>> b = QQ(3, 2)
        >>> a               # doctest: +SKIP
        mpq(1,1)
        >>> b               # doctest: +SKIP
        mpq(3,2)
        >>> divmod(a, b)    # doctest: +SKIP
        (mpz(0), mpq(1,1))
        >>> QQ.div(a, b)    # doctest: +SKIP
        (mpq(2,3), mpq(0,1))

        Using ``//`` or ``%`` with :ref:`QQ` will lead to incorrect results so
        :py:meth:`~.Domain.div` should be used instead.

        """
        raise NotImplementedError

    def invert(self, a, b):
        """Returns inversion of ``a mod b``, implies something. """
        raise NotImplementedError

    def revert(self, a):
        """Returns ``a**(-1)`` if possible. """
        raise NotImplementedError

    def numer(self, a):
        """Returns numerator of ``a``. """
        raise NotImplementedError

    def denom(self, a):
        """Returns denominator of ``a``. """
        raise NotImplementedError

    def half_gcdex(self, a, b):
        """Half extended GCD of ``a`` and ``b``. """
        s, t, h = self.gcdex(a, b)
        return s, h

    def gcdex(self, a, b):
        """Extended GCD of ``a`` and ``b``. """
        raise NotImplementedError

    def cofactors(self, a, b):
        """Returns GCD and cofactors of ``a`` and ``b``. """
        gcd = self.gcd(a, b)
        cfa = self.quo(a, gcd)
        cfb = self.quo(b, gcd)
        return gcd, cfa, cfb

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        raise NotImplementedError

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        raise NotImplementedError

    def log(self, a, b):
        """Returns b-base logarithm of ``a``. """
        raise NotImplementedError

    def sqrt(self, a):
        """Returns a (possibly inexact) square root of ``a``.

        Explanation
        ===========
        There is no universal definition of "inexact square root" for all
        domains. It is not recommended to implement this method for domains
        other then :ref:`ZZ`.

        See also
        ========
        exsqrt
        """
        raise NotImplementedError

    def is_square(self, a):
        """Returns whether ``a`` is a square in the domain.

        Explanation
        ===========
        Returns ``True`` if there is an element ``b`` in the domain such that
        ``b * b == a``, otherwise returns ``False``. For inexact domains like
        :ref:`RR` and :ref:`CC`, a tiny difference in this equality can be
        tolerated.

        See also
        ========
        exsqrt
        """
        raise NotImplementedError

    def exsqrt(self, a):
        """Principal square root of a within the domain if ``a`` is square.

        Explanation
        ===========
        The implementation of this method should return an element ``b`` in the
        domain such that ``b * b == a``, or ``None`` if there is no such ``b``.
        For inexact domains like :ref:`RR` and :ref:`CC`, a tiny difference in
        this equality can be tolerated. The choice of a "principal" square root
        should follow a consistent rule whenever possible.

        See also
        ========
        sqrt, is_square
        """
        raise NotImplementedError

    def evalf(self, a, prec=None, **options):
        """Returns numerical approximation of ``a``. """
        return self.to_sympy(a).evalf(prec, **options)

    n = evalf

    def real(self, a):
        return a

    def imag(self, a):
        return self.zero

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return a == b

    def characteristic(self):
        """Return the characteristic of this domain. """
        raise NotImplementedError('characteristic()')


__all__ = ['Domain']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\point.py ===
"""Geometrical Points.

Contains
========
Point
Point2D
Point3D

When methods of Point require 1 or more points as arguments, they
can be passed as a sequence of coordinates or Points:

>>> from sympy import Point
>>> Point(1, 1).is_collinear((2, 2), (3, 4))
False
>>> Point(1, 1).is_collinear(Point(2, 2), Point(3, 4))
False

"""

import warnings

from sympy.core import S, sympify, Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.numbers import Float
from sympy.core.parameters import global_parameters
from sympy.simplify.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.matrices.expressions import Transpose
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable

from .entity import GeometryEntity

from mpmath.libmp.libmpf import prec_to_dps


class Point(GeometryEntity):
    """A point in a n-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of n-coordinate values. In the special
        case where n=2 or 3, a Point2D or Point3D will be created
        as appropriate.
    evaluate : if `True` (default), all floats are turn into
        exact types.
    dim : number of coordinates the point should have.  If coordinates
        are unspecified, they are padded with zeros.
    on_morph : indicates what should happen when the number of
        coordinates of a point need to be changed by adding or
        removing zeros.  Possible values are `'warn'`, `'error'`, or
        `ignore` (default).  No warning or error is given when `*args`
        is empty and `dim` is given. An error is always raised when
        trying to remove nonzero coordinates.


    Attributes
    ==========

    length
    origin: A `Point` representing the origin of the
        appropriately-dimensioned space.

    Raises
    ======

    TypeError : When instantiating with anything but a Point or sequence
    ValueError : when instantiating with a sequence with length < 2 or
        when trying to reduce dimensions if keyword `on_morph='error'` is
        set.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point
    >>> from sympy.abc import x
    >>> Point(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point([1, 2])
    Point2D(1, 2)
    >>> Point(0, x)
    Point2D(0, x)
    >>> Point(dim=4)
    Point(0, 0, 0, 0)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    is_Point = True

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        on_morph = kwargs.get('on_morph', 'ignore')

        # unpack into coords
        coords = args[0] if len(args) == 1 else args

        # check args and handle quickly handle Point instances
        if isinstance(coords, Point):
            # even if we're mutating the dimension of a point, we
            # don't reevaluate its coordinates
            evaluate = False
            if len(coords) == kwargs.get('dim', len(coords)):
                return coords

        if not is_sequence(coords):
            raise TypeError(filldedent('''
                Expecting sequence of coordinates, not `{}`'''
                                       .format(func_name(coords))))
        # A point where only `dim` is specified is initialized
        # to zeros.
        if len(coords) == 0 and kwargs.get('dim', None):
            coords = (S.Zero,)*kwargs.get('dim')

        coords = Tuple(*coords)
        dim = kwargs.get('dim', len(coords))

        if len(coords) < 2:
            raise ValueError(filldedent('''
                Point requires 2 or more coordinates or
                keyword `dim` > 1.'''))
        if len(coords) != dim:
            message = ("Dimension of {} needs to be changed "
                       "from {} to {}.").format(coords, len(coords), dim)
            if on_morph == 'ignore':
                pass
            elif on_morph == "error":
                raise ValueError(message)
            elif on_morph == 'warn':
                warnings.warn(message, stacklevel=2)
            else:
                raise ValueError(filldedent('''
                        on_morph value should be 'error',
                        'warn' or 'ignore'.'''))
        if any(coords[dim:]):
            raise ValueError('Nonzero coordinates cannot be removed.')
        if any(a.is_number and im(a).is_zero is False for a in coords):
            raise ValueError('Imaginary coordinates are not permitted.')
        if not all(isinstance(a, Expr) for a in coords):
            raise TypeError('Coordinates must be valid SymPy expressions.')

        # pad with zeros appropriately
        coords = coords[:dim] + (S.Zero,)*(dim - len(coords))

        # Turn any Floats into rationals and simplify
        # any expressions before we instantiate
        if evaluate:
            coords = coords.xreplace({
                f: simplify(nsimplify(f, rational=True))
                 for f in coords.atoms(Float)})

        # return 2D or 3D instances
        if len(coords) == 2:
            kwargs['_nocheck'] = True
            return Point2D(*coords, **kwargs)
        elif len(coords) == 3:
            kwargs['_nocheck'] = True
            return Point3D(*coords, **kwargs)

        # the general Point
        return GeometryEntity.__new__(cls, *coords)

    def __abs__(self):
        """Returns the distance between this point and the origin."""
        origin = Point([0]*len(self))
        return Point.distance(origin, self)

    def __add__(self, other):
        """Add other to self by incrementing self's coordinates by
        those of other.

        Notes
        =====

        >>> from sympy import Point

        When sequences of coordinates are passed to Point methods, they
        are converted to a Point internally. This __add__ method does
        not do that so if floating point values are used, a floating
        point result (in terms of SymPy Floats) will be returned.

        >>> Point(1, 2) + (.1, .2)
        Point2D(1.1, 2.2)

        If this is not desired, the `translate` method can be used or
        another Point can be added:

        >>> Point(1, 2).translate(.1, .2)
        Point2D(11/10, 11/5)
        >>> Point(1, 2) + Point(.1, .2)
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.translate

        """
        try:
            s, o = Point._normalize_dimension(self, Point(other, evaluate=False))
        except TypeError:
            raise GeometryError("Don't know how to add {} and a Point object".format(other))

        coords = [simplify(a + b) for a, b in zip(s, o)]
        return Point(coords, evaluate=False)

    def __contains__(self, item):
        return item in self.args

    def __truediv__(self, divisor):
        """Divide point's coordinates by a factor."""
        divisor = sympify(divisor)
        coords = [simplify(x/divisor) for x in self.args]
        return Point(coords, evaluate=False)

    def __eq__(self, other):
        if not isinstance(other, Point) or len(self.args) != len(other.args):
            return False
        return self.args == other.args

    def __getitem__(self, key):
        return self.args[key]

    def __hash__(self):
        return hash(self.args)

    def __iter__(self):
        return self.args.__iter__()

    def __len__(self):
        return len(self.args)

    def __mul__(self, factor):
        """Multiply point's coordinates by a factor.

        Notes
        =====

        >>> from sympy import Point

        When multiplying a Point by a floating point number,
        the coordinates of the Point will be changed to Floats:

        >>> Point(1, 2)*0.1
        Point2D(0.1, 0.2)

        If this is not desired, the `scale` method can be used or
        else only multiply or divide by integers:

        >>> Point(1, 2).scale(1.1, 1.1)
        Point2D(11/10, 11/5)
        >>> Point(1, 2)*11/10
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.scale
        """
        factor = sympify(factor)
        coords = [simplify(x*factor) for x in self.args]
        return Point(coords, evaluate=False)

    def __rmul__(self, factor):
        """Multiply a factor by point's coordinates."""
        return self.__mul__(factor)

    def __neg__(self):
        """Negate the point."""
        coords = [-x for x in self.args]
        return Point(coords, evaluate=False)

    def __sub__(self, other):
        """Subtract two points, or subtract a factor from this point's
        coordinates."""
        return self + [-x for x in other]

    @classmethod
    def _normalize_dimension(cls, *points, **kwargs):
        """Ensure that points have the same dimension.
        By default `on_morph='warn'` is passed to the
        `Point` constructor."""
        # if we have a built-in ambient dimension, use it
        dim = getattr(cls, '_ambient_dimension', None)
        # override if we specified it
        dim = kwargs.get('dim', dim)
        # if no dim was given, use the highest dimensional point
        if dim is None:
            dim = max(i.ambient_dimension for i in points)
        if all(i.ambient_dimension == dim for i in points):
            return list(points)
        kwargs['dim'] = dim
        kwargs['on_morph'] = kwargs.get('on_morph', 'warn')
        return [Point(i, **kwargs) for i in points]

    @staticmethod
    def affine_rank(*args):
        """The affine rank of a set of points is the dimension
        of the smallest affine space containing all the points.
        For example, if the points lie on a line (and are not all
        the same) their affine rank is 1.  If the points lie on a plane
        but not a line, their affine rank is 2.  By convention, the empty
        set has affine rank -1."""

        if len(args) == 0:
            return -1
        # make sure we're genuinely points
        # and translate every point to the origin
        points = Point._normalize_dimension(*[Point(i) for i in args])
        origin = points[0]
        points = [i - origin for i in points[1:]]

        m = Matrix([i.args for i in points])
        # XXX fragile -- what is a better way?
        return m.rank(iszerofunc = lambda x:
            abs(x.n(2)) < 1e-12 if x.is_number else x.is_zero)

    @property
    def ambient_dimension(self):
        """Number of components this point has."""
        return getattr(self, '_ambient_dimension', len(self))

    @classmethod
    def are_coplanar(cls, *points):
        """Return True if there exists a plane in which all the points
        lie.  A trivial True value is returned if `len(points) < 3` or
        all Points are 2-dimensional.

        Parameters
        ==========

        A set of points

        Raises
        ======

        ValueError : if less than 3 unique points are given

        Returns
        =======

        boolean

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 2)
        >>> p2 = Point3D(2, 7, 2)
        >>> p3 = Point3D(0, 0, 2)
        >>> p4 = Point3D(1, 1, 2)
        >>> Point3D.are_coplanar(p1, p2, p3, p4)
        True
        >>> p5 = Point3D(0, 1, 3)
        >>> Point3D.are_coplanar(p1, p2, p3, p5)
        False

        """
        if len(points) <= 1:
            return True

        points = cls._normalize_dimension(*[Point(i) for i in points])
        # quick exit if we are in 2D
        if points[0].ambient_dimension == 2:
            return True
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 2

    def distance(self, other):
        """The Euclidean distance between self and another GeometricEntity.

        Returns
        =======

        distance : number or symbolic expression.

        Raises
        ======

        TypeError : if other is not recognized as a GeometricEntity or is a
                    GeometricEntity for which distance is not defined.

        See Also
        ========

        sympy.geometry.line.Segment.length
        sympy.geometry.point.Point.taxicab_distance

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> l = Line((3, 1), (2, 2))
        >>> p1.distance(p2)
        5
        >>> p1.distance(l)
        sqrt(2)

        The computed distance may be symbolic, too:

        >>> from sympy.abc import x, y
        >>> p3 = Point(x, y)
        >>> p3.distance((0, 0))
        sqrt(x**2 + y**2)

        """
        if not isinstance(other, GeometryEntity):
            try:
                other = Point(other, dim=self.ambient_dimension)
            except TypeError:
                raise TypeError("not recognized as a GeometricEntity: %s" % type(other))
        if isinstance(other, Point):
            s, p = Point._normalize_dimension(self, Point(other))
            return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
        distance = getattr(other, 'distance', None)
        if distance is None:
            raise TypeError("distance between Point and %s is not defined" % type(other))
        return distance(self)

    def dot(self, p):
        """Return dot product of self with another Point."""
        if not is_sequence(p):
            p = Point(p)  # raise the error via Point
        return Add(*(a*b for a, b in zip(self, p)))

    def equals(self, other):
        """Returns whether the coordinates of self and other agree."""
        # a point is equal to another point if all its components are equal
        if not isinstance(other, Point) or len(self) != len(other):
            return False
        return all(a.equals(b) for a, b in zip(self, other))

    def _eval_evalf(self, prec=15, **options):
        """Evaluate the coordinates of the point.

        This method will, where possible, create and return a new Point
        where the coordinates are evaluated as floating point numbers to
        the precision indicated (default=15).

        Parameters
        ==========

        prec : int

        Returns
        =======

        point : Point

        Examples
        ========

        >>> from sympy import Point, Rational
        >>> p1 = Point(Rational(1, 2), Rational(3, 2))
        >>> p1
        Point2D(1/2, 3/2)
        >>> p1.evalf()
        Point2D(0.5, 1.5)

        """
        dps = prec_to_dps(prec)
        coords = [x.evalf(n=dps, **options) for x in self.args]
        return Point(*coords, evaluate=False)

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point2D(0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other)
        if isinstance(other, Point):
            if self == other:
                return [self]
            p1, p2 = Point._normalize_dimension(self, other)
            if p1 == self and p1 == p2:
                return [self]
            return []
        return other.intersection(self)

    def is_collinear(self, *args):
        """Returns `True` if there exists a line
        that contains `self` and `points`.  Returns `False` otherwise.
        A trivially True value is returned if no points are given.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point
        >>> from sympy.abc import x
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4, p5 = Point(2, 2), Point(x, x), Point(1, 2)
        >>> Point.is_collinear(p1, p2, p3, p4)
        True
        >>> Point.is_collinear(p1, p2, p3, p5)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 1

    def is_concyclic(self, *args):
        """Do `self` and the given sequence of points lie in a circle?

        Returns True if the set of points are concyclic and
        False otherwise. A trivial value of True is returned
        if there are fewer than 2 other points.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_concyclic : boolean


        Examples
        ========

        >>> from sympy import Point

        Define 4 points that are on the unit circle:

        >>> p1, p2, p3, p4 = Point(1, 0), (0, 1), (-1, 0), (0, -1)

        >>> p1.is_concyclic() == p1.is_concyclic(p2, p3, p4) == True
        True

        Define a point not on that circle:

        >>> p = Point(1, 1)

        >>> p.is_concyclic(p1, p2, p3)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        if not Point.affine_rank(*points) <= 2:
            return False
        origin = points[0]
        points = [p - origin for p in points]
        # points are concyclic if they are coplanar and
        # there is a point c so that ||p_i-c|| == ||p_j-c|| for all
        # i and j.  Rearranging this equation gives us the following
        # condition: the matrix `mat` must not a pivot in the last
        # column.
        mat = Matrix([list(i) + [i.dot(i)] for i in points])
        rref, pivots = mat.rref()
        if len(origin) not in pivots:
            return True
        return False

    @property
    def is_nonzero(self):
        """True if any coordinate is nonzero, False if every coordinate is zero,
        and None if it cannot be determined."""
        is_zero = self.is_zero
        if is_zero is None:
            return None
        return not is_zero

    def is_scalar_multiple(self, p):
        """Returns whether each coordinate of `self` is a scalar
        multiple of the corresponding coordinate in point p.
        """
        s, o = Point._normalize_dimension(self, Point(p))
        # 2d points happen a lot, so optimize this function call
        if s.ambient_dimension == 2:
            (x1, y1), (x2, y2) = s.args, o.args
            rv = (x1*y2 - x2*y1).equals(0)
            if rv is None:
                raise Undecidable(filldedent(
                    '''Cannot determine if %s is a scalar multiple of
                    %s''' % (s, o)))

        # if the vectors p1 and p2 are linearly dependent, then they must
        # be scalar multiples of each other
        m = Matrix([s.args, o.args])
        return m.rank() < 2

    @property
    def is_zero(self):
        """True if every coordinate is zero, False if any coordinate is not zero,
        and None if it cannot be determined."""
        nonzero = [x.is_nonzero for x in self.args]
        if any(nonzero):
            return False
        if any(x is None for x in nonzero):
            return None
        return True

    @property
    def length(self):
        """
        Treating a Point as a Line, this returns 0 for the length of a Point.

        Examples
        ========

        >>> from sympy import Point
        >>> p = Point(0, 1)
        >>> p.length
        0
        """
        return S.Zero

    def midpoint(self, p):
        """The midpoint between self and point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        midpoint : Point

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(13, 5)
        >>> p1.midpoint(p2)
        Point2D(7, 3)

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Point([simplify((a + b)*S.Half) for a, b in zip(s, p)])

    @property
    def origin(self):
        """A point of all zeros of the same ambient dimension
        as the current point"""
        return Point([0]*len(self), evaluate=False)

    @property
    def orthogonal_direction(self):
        """Returns a non-zero point that is orthogonal to the
        line containing `self` and the origin.

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2, 3)
        >>> a.orthogonal_direction
        Point3D(-2, 1, 0)
        >>> b = _
        >>> Line(b, b.origin).is_perpendicular(Line(a, a.origin))
        True
        """
        dim = self.ambient_dimension
        # if a coordinate is zero, we can put a 1 there and zeros elsewhere
        if self[0].is_zero:
            return Point([1] + (dim - 1)*[0])
        if self[1].is_zero:
            return Point([0,1] + (dim - 2)*[0])
        # if the first two coordinates aren't zero, we can create a non-zero
        # orthogonal vector by swapping them, negating one, and padding with zeros
        return Point([-self[1], self[0]] + (dim - 2)*[0])

    @staticmethod
    def project(a, b):
        """Project the point `a` onto the line between the origin
        and point `b` along the normal direction.

        Parameters
        ==========

        a : Point
        b : Point

        Returns
        =======

        p : Point

        See Also
        ========

        sympy.geometry.line.LinearEntity.projection

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2)
        >>> b = Point(2, 5)
        >>> z = a.origin
        >>> p = Point.project(a, b)
        >>> Line(p, a).is_perpendicular(Line(p, b))
        True
        >>> Point.is_collinear(z, p, b)
        True
        """
        a, b = Point._normalize_dimension(Point(a), Point(b))
        if b.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return b*(a.dot(b) / b.dot(b))

    def taxicab_distance(self, p):
        """The Taxicab Distance from self to point p.

        Returns the sum of the horizontal and vertical distances to point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        taxicab_distance : The sum of the horizontal
        and vertical distances to point p.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.taxicab_distance(p2)
        7

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Add(*(abs(a - b) for a, b in zip(s, p)))

    def canberra_distance(self, p):
        """The Canberra Distance from self to point p.

        Returns the weighted sum of horizontal and vertical distances to
        point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        canberra_distance : The weighted sum of horizontal and vertical
        distances to point p. The weight used is the sum of absolute values
        of the coordinates.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(3, 3)
        >>> p1.canberra_distance(p2)
        1
        >>> p1, p2 = Point(0, 0), Point(3, 3)
        >>> p1.canberra_distance(p2)
        2

        Raises
        ======

        ValueError when both vectors are zero.

        See Also
        ========

        sympy.geometry.point.Point.distance

        """

        s, p = Point._normalize_dimension(self, Point(p))
        if self.is_zero and p.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return Add(*((abs(a - b)/(abs(a) + abs(b))) for a, b in zip(s, p)))

    @property
    def unit(self):
        """Return the Point that is in the same direction as `self`
        and a distance of 1 from the origin"""
        return self / abs(self)


class Point2D(Point):
    """A point in a 2-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 2 coordinate values.

    Attributes
    ==========

    x
    y
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When trying to create a point with more than two dimensions.
        When `intersection` is called with object other than a Point.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point2D
    >>> from sympy.abc import x
    >>> Point2D(1, 2)
    Point2D(1, 2)
    >>> Point2D([1, 2])
    Point2D(1, 2)
    >>> Point2D(0, x)
    Point2D(0, x)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point2D(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point2D(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    _ambient_dimension = 2

    def __new__(cls, *args, _nocheck=False, **kwargs):
        if not _nocheck:
            kwargs['dim'] = 2
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        return (self.x, self.y, self.x, self.y)

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        See Also
        ========

        translate, scale

        Examples
        ========

        >>> from sympy import Point2D, pi
        >>> t = Point2D(1, 0)
        >>> t.rotate(pi/2)
        Point2D(0, 1)
        >>> t.rotate(pi/2, (2, 0))
        Point2D(2, -1)

        """
        c = cos(angle)
        s = sin(angle)

        rv = self
        if pt is not None:
            pt = Point(pt, dim=2)
            rv -= pt
        x, y = rv.args
        rv = Point(c*x - s*y, s*x + c*y)
        if pt is not None:
            rv += pt
        return rv

    def scale(self, x=1, y=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(1, 1)
        >>> t.scale(2)
        Point2D(2, 1)
        >>> t.scale(2, 2)
        Point2D(2, 2)

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return Point(self.x*x, self.y*y)

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 3x3 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point2D.rotate
        sympy.geometry.point.Point2D.scale
        sympy.geometry.point.Point2D.translate
        """
        if not (matrix.is_Matrix and matrix.shape == (3, 3)):
            raise ValueError("matrix must be a 3x3 matrix")
        x, y = self.args
        return Point(*(Matrix(1, 3, [x, y, 1])*matrix).tolist()[0][:2])

    def translate(self, x=0, y=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        sympy.geometry.point.Point2D.rotate, scale

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(0, 1)
        >>> t.translate(2)
        Point2D(2, 1)
        >>> t.translate(2, 2)
        Point2D(2, 3)
        >>> t + Point2D(2, 2)
        Point2D(2, 3)

        """
        return Point(self.x + x, self.y + y)

    @property
    def coordinates(self):
        """
        Returns the two coordinates of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.coordinates
        (0, 1)
        """
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.y
        1
        """
        return self.args[1]

class Point3D(Point):
    """A point in a 3-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 3 coordinate values.

    Attributes
    ==========

    x
    y
    z
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When `intersection` is called with object other than a Point.

    Examples
    ========

    >>> from sympy import Point3D
    >>> from sympy.abc import x
    >>> Point3D(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point3D([1, 2, 3])
    Point3D(1, 2, 3)
    >>> Point3D(0, x, 3)
    Point3D(0, x, 3)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point3D(0.5, 0.25, 2)
    Point3D(1/2, 1/4, 2)
    >>> Point3D(0.5, 0.25, 3, evaluate=False)
    Point3D(0.5, 0.25, 3)

    """

    _ambient_dimension = 3

    def __new__(cls, *args, _nocheck=False, **kwargs):
        if not _nocheck:
            kwargs['dim'] = 3
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self

    @staticmethod
    def are_collinear(*points):
        """Is a sequence of points collinear?

        Test whether or not a set of points are collinear. Returns True if
        the set of points are collinear, or False otherwise.

        Parameters
        ==========

        points : sequence of Point

        Returns
        =======

        are_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line3D

        Examples
        ========

        >>> from sympy import Point3D
        >>> from sympy.abc import x
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
        >>> p3, p4, p5 = Point3D(2, 2, 2), Point3D(x, x, x), Point3D(1, 2, 6)
        >>> Point3D.are_collinear(p1, p2, p3, p4)
        True
        >>> Point3D.are_collinear(p1, p2, p3, p5)
        False
        """
        return Point.is_collinear(*points)

    def direction_cosine(self, point):
        """
        Gives the direction cosine between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_cosine(Point3D(2, 3, 5))
        [sqrt(6)/6, sqrt(6)/6, sqrt(6)/3]
        """
        a = self.direction_ratio(point)
        b = sqrt(Add(*(i**2 for i in a)))
        return [(point.x - self.x) / b,(point.y - self.y) / b,
                (point.z - self.z) / b]

    def direction_ratio(self, point):
        """
        Gives the direction ratio between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_ratio(Point3D(2, 3, 5))
        [1, 1, 2]
        """
        return [(point.x - self.x),(point.y - self.y),(point.z - self.z)]

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point3D(0, 0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=3)
        if isinstance(other, Point3D):
            if self == other:
                return [self]
            return []
        return other.intersection(self)

    def scale(self, x=1, y=1, z=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        translate

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(1, 1, 1)
        >>> t.scale(2)
        Point3D(2, 1, 1)
        >>> t.scale(2, 2)
        Point3D(2, 2, 1)

        """
        if pt:
            pt = Point3D(pt)
            return self.translate(*(-pt).args).scale(x, y, z).translate(*pt.args)
        return Point3D(self.x*x, self.y*y, self.z*z)

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 4x4 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point3D.scale
        sympy.geometry.point.Point3D.translate
        """
        if not (matrix.is_Matrix and matrix.shape == (4, 4)):
            raise ValueError("matrix must be a 4x4 matrix")
        x, y, z = self.args
        m = Transpose(matrix)
        return Point3D(*(Matrix(1, 4, [x, y, z, 1])*m).tolist()[0][:3])

    def translate(self, x=0, y=0, z=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        scale

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(0, 1, 1)
        >>> t.translate(2)
        Point3D(2, 1, 1)
        >>> t.translate(2, 2)
        Point3D(2, 3, 1)
        >>> t + Point3D(2, 2, 2)
        Point3D(2, 3, 3)

        """
        return Point3D(self.x + x, self.y + y, self.z + z)

    @property
    def coordinates(self):
        """
        Returns the three coordinates of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.coordinates
        (0, 1, 2)
        """
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 3)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.y
        1
        """
        return self.args[1]

    @property
    def z(self):
        """
        Returns the Z coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 1)
        >>> p.z
        1
        """
        return self.args[2]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_ssl.py ===
from __future__ import annotations

import os
import socket as stdlib_socket
import ssl
import sys
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from ssl import SSLContext
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

import pytest

from trio import StapledStream
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from trio.abc import ReceiveStream, SendStream
from trio.testing import (
    Matcher,
    MemoryReceiveStream,
    MemorySendStream,
    RaisesGroup,
)

try:
    import trustme
    from OpenSSL import SSL
except ImportError as error:
    skip_if_optional_else_raise(error)

import trio

from .. import _core, socket as tsocket
from .._abc import Stream
from .._core import BrokenResourceError, ClosedResourceError
from .._core._tests.tutil import slow
from .._highlevel_generic import aclose_forcefully
from .._highlevel_open_tcp_stream import open_tcp_stream
from .._highlevel_socket import SocketListener, SocketStream
from .._ssl import NeedHandshakeError, SSLListener, SSLStream, _is_eof
from .._util import ConflictDetector
from ..testing import (
    Sequencer,
    assert_checkpoints,
    check_two_way_stream,
    lockstep_stream_pair,
    memory_stream_pair,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from typing_extensions import TypeAlias

    from trio._core import MockClock
    from trio._ssl import T_Stream

    from .._core._run import CancelScope

# We have two different kinds of echo server fixtures we use for testing. The
# first is a real server written using the stdlib ssl module and blocking
# sockets. It runs in a thread and we talk to it over a real socketpair(), to
# validate interoperability in a semi-realistic setting.
#
# The second is a very weird virtual echo server that lives inside a custom
# Stream class. It lives entirely inside the Python object space; there are no
# operating system calls in it at all. No threads, no I/O, nothing. It's
# 'send_all' call takes encrypted data from a client and feeds it directly into
# the server-side TLS state engine to decrypt, then takes that data, feeds it
# back through to get the encrypted response, and returns it from 'receive_some'. This
# gives us full control and reproducibility. This server is written using
# PyOpenSSL, so that we can trigger renegotiations on demand. It also allows
# us to insert random (virtual) delays, to really exercise all the weird paths
# in SSLStream's state engine.
#
# Both present a certificate for "trio-test-1.example.org".

TRIO_TEST_CA = trustme.CA()
TRIO_TEST_1_CERT = TRIO_TEST_CA.issue_server_cert("trio-test-1.example.org")

SERVER_CTX = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
    SERVER_CTX.options &= ~ssl.OP_IGNORE_UNEXPECTED_EOF

TRIO_TEST_1_CERT.configure_cert(SERVER_CTX)


# TLS 1.3 has a lot of changes from previous versions. So we want to run tests
# with both TLS 1.3, and TLS 1.2.
# "tls13" means that we're willing to negotiate TLS 1.3. Usually that's
# what will happen, but the renegotiation tests explicitly force a
# downgrade on the server side. "tls12" means we refuse to negotiate TLS
# 1.3, so we'll almost certainly use TLS 1.2.
@pytest.fixture(scope="module", params=["tls13", "tls12"])
def client_ctx(request: pytest.FixtureRequest) -> ssl.SSLContext:
    ctx = ssl.create_default_context()

    if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
        ctx.options &= ~ssl.OP_IGNORE_UNEXPECTED_EOF

    TRIO_TEST_CA.configure_trust(ctx)
    if request.param in ["default", "tls13"]:
        return ctx
    elif request.param == "tls12":
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        return ctx
    else:  # pragma: no cover
        raise AssertionError()


# The blocking socket server.
def ssl_echo_serve_sync(
    sock: stdlib_socket.socket,
    *,
    expect_fail: bool = False,
) -> None:
    try:
        wrapped = SERVER_CTX.wrap_socket(
            sock,
            server_side=True,
            suppress_ragged_eofs=False,
        )
        with wrapped:
            wrapped.do_handshake()
            while True:
                data = wrapped.recv(4096)
                if not data:
                    # other side has initiated a graceful shutdown; we try to
                    # respond in kind but it's legal for them to have already
                    # gone away.
                    with suppress(BrokenPipeError, ssl.SSLZeroReturnError):
                        wrapped.unwrap()
                    return
                wrapped.sendall(data)
    # This is an obscure workaround for an openssl bug. In server mode, in
    # some versions, openssl sends some extra data at the end of do_handshake
    # that it shouldn't send. Normally this is harmless, but, if the other
    # side shuts down the connection before it reads that data, it might cause
    # the OS to report a ECONNREST or even ECONNABORTED (which is just wrong,
    # since ECONNABORTED is supposed to mean that connect() failed, but what
    # can you do). In this case the other side did nothing wrong, but there's
    # no way to recover, so we let it pass, and just cross our fingers its not
    # hiding any (other) real bugs. For more details see:
    #
    #   https://github.com/python-trio/trio/issues/1293
    #
    # Also, this happens frequently but non-deterministically, so we have to
    # 'no cover' it to avoid coverage flapping.
    except (ConnectionResetError, ConnectionAbortedError):  # pragma: no cover
        return
    except Exception as exc:
        if expect_fail:
            print("ssl_echo_serve_sync got error as expected:", exc)
        else:  # pragma: no cover
            print("ssl_echo_serve_sync got unexpected error:", exc)
            raise
    else:
        if expect_fail:  # pragma: no cover
            raise RuntimeError("failed to fail?")
    finally:
        sock.close()


# Fixture that gives a raw socket connected to a trio-test-1 echo server
# (running in a thread). Useful for testing making connections with different
# SSLContexts.
@asynccontextmanager
async def ssl_echo_server_raw(expect_fail: bool = False) -> AsyncIterator[SocketStream]:
    a, b = stdlib_socket.socketpair()
    async with trio.open_nursery() as nursery:
        # Exiting the 'with a, b' context manager closes the sockets, which
        # causes the thread to exit (possibly with an error), which allows the
        # nursery context manager to exit too.
        with a, b:
            nursery.start_soon(
                trio.to_thread.run_sync,
                partial(ssl_echo_serve_sync, b, expect_fail=expect_fail),
            )

            yield SocketStream(tsocket.from_stdlib_socket(a))


# Fixture that gives a properly set up SSLStream connected to a trio-test-1
# echo server (running in a thread)
@asynccontextmanager
async def ssl_echo_server(
    client_ctx: SSLContext,
    expect_fail: bool = False,
) -> AsyncIterator[SSLStream[Stream]]:
    async with ssl_echo_server_raw(expect_fail=expect_fail) as sock:
        yield SSLStream(sock, client_ctx, server_hostname="trio-test-1.example.org")


# The weird in-memory server ... thing.
# Doesn't inherit from Stream because I left out the methods that we don't
# actually need.
# jakkdl: it seems to implement all the abstract methods (now), so I made it inherit
#         from Stream for the sake of typechecking.
class PyOpenSSLEchoStream(Stream):
    def __init__(
        self,
        sleeper: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        ctx = SSL.Context(SSL.SSLv23_METHOD)
        # TLS 1.3 removes renegotiation support. Which is great for them, but
        # we still have to support versions before that, and that means we
        # need to test renegotiation support, which means we need to force this
        # to use a lower version where this test server can trigger
        # renegotiations.
        from cryptography.hazmat.bindings.openssl.binding import Binding

        b = Binding()
        ctx.set_options(b.lib.SSL_OP_NO_TLSv1_3)

        # Unfortunately there's currently no way to say "use 1.3 or worse", we
        # can only disable specific versions. And if the two sides start
        # negotiating 1.4 at some point in the future, it *might* mean that
        # our tests silently stop working properly. So the next line is a
        # tripwire to remind us we need to revisit this stuff in 5 years or
        # whatever when the next TLS version is released:
        assert not hasattr(SSL, "OP_NO_TLSv1_4")
        TRIO_TEST_1_CERT.configure_cert(ctx)
        self._conn = SSL.Connection(ctx, None)
        self._conn.set_accept_state()
        self._lot = _core.ParkingLot()
        self._pending_cleartext = bytearray()

        self._send_all_conflict_detector = ConflictDetector(
            "simultaneous calls to PyOpenSSLEchoStream.send_all",
        )
        self._receive_some_conflict_detector = ConflictDetector(
            "simultaneous calls to PyOpenSSLEchoStream.receive_some",
        )

        self.sleeper: Callable[[str], Awaitable[None]]
        if sleeper is None:

            async def no_op_sleeper(_: object) -> None:
                return

            self.sleeper = no_op_sleeper
        else:
            self.sleeper = sleeper

    async def aclose(self) -> None:
        self._conn.bio_shutdown()

    def renegotiate_pending(self) -> bool:
        return self._conn.renegotiate_pending()

    def renegotiate(self) -> None:
        # Returns false if a renegotiation is already in progress, meaning
        # nothing happens.
        assert self._conn.renegotiate()

    async def wait_send_all_might_not_block(self) -> None:
        with self._send_all_conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            await self.sleeper("wait_send_all_might_not_block")

    async def send_all(self, data: bytes) -> None:
        print("  --> transport_stream.send_all")
        with self._send_all_conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            await self.sleeper("send_all")
            self._conn.bio_write(data)
            while True:
                await self.sleeper("send_all")
                try:
                    data = self._conn.recv(1)
                except SSL.ZeroReturnError:
                    self._conn.shutdown()
                    print("renegotiations:", self._conn.total_renegotiations())
                    break
                except SSL.WantReadError:
                    break
                else:
                    self._pending_cleartext += data
            self._lot.unpark_all()
            await self.sleeper("send_all")
            print("  <-- transport_stream.send_all finished")

    async def receive_some(self, nbytes: int | None = None) -> bytes:
        print("  --> transport_stream.receive_some")
        if nbytes is None:
            nbytes = 65536  # arbitrary
        with self._receive_some_conflict_detector:
            try:
                await _core.checkpoint()
                await _core.checkpoint()
                while True:
                    await self.sleeper("receive_some")
                    try:
                        return self._conn.bio_read(nbytes)
                    except SSL.WantReadError:
                        # No data in our ciphertext buffer; try to generate
                        # some.
                        if self._pending_cleartext:
                            # We have some cleartext; maybe we can encrypt it
                            # and then return it.
                            print("    trying", self._pending_cleartext)
                            try:
                                # PyOpenSSL bug: doesn't accept bytearray
                                # https://github.com/pyca/pyopenssl/issues/621
                                next_byte = self._pending_cleartext[0:1]
                                self._conn.send(bytes(next_byte))
                            # Apparently this next bit never gets hit in the
                            # test suite, but it's not an interesting omission
                            # so let's pragma it.
                            except SSL.WantReadError:  # pragma: no cover
                                # We didn't manage to send the cleartext (and
                                # in particular we better leave it there to
                                # try again, due to openssl's retry
                                # semantics), but it's possible we pushed a
                                # renegotiation forward and *now* we have data
                                # to send.
                                try:
                                    return self._conn.bio_read(nbytes)
                                except SSL.WantReadError:
                                    # Nope. We're just going to have to wait
                                    # for someone to call send_all() to give
                                    # use more data.
                                    print("parking (a)")
                                    await self._lot.park()
                            else:
                                # We successfully sent that byte, so we don't
                                # have to again.
                                del self._pending_cleartext[0:1]
                        else:
                            # no pending cleartext; nothing to do but wait for
                            # someone to call send_all
                            print("parking (b)")
                            await self._lot.park()
            finally:
                await self.sleeper("receive_some")
                print("  <-- transport_stream.receive_some finished")


async def test_PyOpenSSLEchoStream_gives_resource_busy_errors() -> None:
    # Make sure that PyOpenSSLEchoStream complains if two tasks call send_all
    # at the same time, or ditto for receive_some. The tricky cases where SSLStream
    # might accidentally do this are during renegotiation, which we test using
    # PyOpenSSLEchoStream, so this makes sure that if we do have a bug then
    # PyOpenSSLEchoStream will notice and complain.

    async def do_test(
        func1: str,
        args1: tuple[object, ...],
        func2: str,
        args2: tuple[object, ...],
    ) -> None:
        s = PyOpenSSLEchoStream()
        with RaisesGroup(Matcher(_core.BusyResourceError, "simultaneous")):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(getattr(s, func1), *args1)
                nursery.start_soon(getattr(s, func2), *args2)

    await do_test("send_all", (b"x",), "send_all", (b"x",))
    await do_test("send_all", (b"x",), "wait_send_all_might_not_block", ())
    await do_test(
        "wait_send_all_might_not_block",
        (),
        "wait_send_all_might_not_block",
        (),
    )
    await do_test("receive_some", (1,), "receive_some", (1,))


@contextmanager
def virtual_ssl_echo_server(
    client_ctx: SSLContext,
    sleeper: Callable[[str], Awaitable[None]] | None = None,
) -> Iterator[SSLStream[PyOpenSSLEchoStream]]:
    fakesock = PyOpenSSLEchoStream(sleeper=sleeper)
    yield SSLStream(fakesock, client_ctx, server_hostname="trio-test-1.example.org")


def ssl_wrap_pair(  # type: ignore[explicit-any]
    client_ctx: SSLContext,
    client_transport: T_Stream,
    server_transport: T_Stream,
    *,
    client_kwargs: dict[str, Any] | None = None,
    server_kwargs: dict[str, Any] | None = None,
) -> tuple[SSLStream[T_Stream], SSLStream[T_Stream]]:
    if server_kwargs is None:
        server_kwargs = {}
    if client_kwargs is None:
        client_kwargs = {}
    client_ssl = SSLStream(
        client_transport,
        client_ctx,
        server_hostname="trio-test-1.example.org",
        **client_kwargs,
    )
    server_ssl = SSLStream(
        server_transport,
        SERVER_CTX,
        server_side=True,
        **server_kwargs,
    )
    return client_ssl, server_ssl


MemoryStapledStream: TypeAlias = StapledStream[MemorySendStream, MemoryReceiveStream]


def ssl_memory_stream_pair(
    client_ctx: SSLContext,
    client_kwargs: dict[str, str | bytes | bool | None] | None = None,
    server_kwargs: dict[str, str | bytes | bool | None] | None = None,
) -> tuple[
    SSLStream[MemoryStapledStream],
    SSLStream[MemoryStapledStream],
]:
    client_transport, server_transport = memory_stream_pair()
    return ssl_wrap_pair(
        client_ctx,
        client_transport,
        server_transport,
        client_kwargs=client_kwargs,
        server_kwargs=server_kwargs,
    )


MyStapledStream: TypeAlias = StapledStream[SendStream, ReceiveStream]


def ssl_lockstep_stream_pair(
    client_ctx: SSLContext,
    client_kwargs: dict[str, str | bytes | bool | None] | None = None,
    server_kwargs: dict[str, str | bytes | bool | None] | None = None,
) -> tuple[
    SSLStream[MyStapledStream],
    SSLStream[MyStapledStream],
]:
    client_transport, server_transport = lockstep_stream_pair()
    return ssl_wrap_pair(
        client_ctx,
        client_transport,
        server_transport,
        client_kwargs=client_kwargs,
        server_kwargs=server_kwargs,
    )


# Simple smoke test for handshake/send/receive/shutdown talking to a
# synchronous server, plus make sure that we do the bare minimum of
# certificate checking (even though this is really Python's responsibility)
async def test_ssl_client_basics(client_ctx: SSLContext) -> None:
    # Everything OK
    async with ssl_echo_server(client_ctx) as s:
        assert not s.server_side
        await s.send_all(b"x")
        assert await s.receive_some(1) == b"x"
        await s.aclose()

    # Didn't configure the CA file, should fail
    async with ssl_echo_server_raw(expect_fail=True) as sock:
        bad_client_ctx = ssl.create_default_context()
        s = SSLStream(sock, bad_client_ctx, server_hostname="trio-test-1.example.org")
        assert not s.server_side
        with pytest.raises(BrokenResourceError) as excinfo:
            await s.send_all(b"x")
        assert isinstance(excinfo.value.__cause__, ssl.SSLError)

    # Trusted CA, but wrong host name
    async with ssl_echo_server_raw(expect_fail=True) as sock:
        s = SSLStream(sock, client_ctx, server_hostname="trio-test-2.example.org")
        assert not s.server_side
        with pytest.raises(BrokenResourceError) as excinfo:
            await s.send_all(b"x")
        assert isinstance(excinfo.value.__cause__, ssl.CertificateError)


async def test_ssl_server_basics(client_ctx: SSLContext) -> None:
    a, b = stdlib_socket.socketpair()
    with a, b:
        server_sock = tsocket.from_stdlib_socket(b)
        server_transport = SSLStream(
            SocketStream(server_sock),
            SERVER_CTX,
            server_side=True,
        )
        assert server_transport.server_side

        def client() -> None:
            with client_ctx.wrap_socket(
                a,
                server_hostname="trio-test-1.example.org",
            ) as client_sock:
                client_sock.sendall(b"x")
                assert client_sock.recv(1) == b"y"
                client_sock.sendall(b"z")
                client_sock.unwrap()

        t = threading.Thread(target=client)
        t.start()

        assert await server_transport.receive_some(1) == b"x"
        await server_transport.send_all(b"y")
        assert await server_transport.receive_some(1) == b"z"
        assert await server_transport.receive_some(1) == b""
        await server_transport.aclose()

        t.join()


async def test_attributes(client_ctx: SSLContext) -> None:
    async with ssl_echo_server_raw(expect_fail=True) as sock:
        good_ctx = client_ctx
        bad_ctx = ssl.create_default_context()
        s = SSLStream(sock, good_ctx, server_hostname="trio-test-1.example.org")

        assert s.transport_stream is sock

        # Forwarded attribute getting
        assert s.context is good_ctx
        assert s.server_side == False  # noqa
        assert s.server_hostname == "trio-test-1.example.org"
        with pytest.raises(AttributeError):
            s.asfdasdfsa  # noqa: B018  # "useless expression"

        # __dir__
        assert "transport_stream" in dir(s)
        assert "context" in dir(s)

        # Setting the attribute goes through to the underlying object

        # most attributes on SSLObject are read-only
        with pytest.raises(AttributeError):
            s.server_side = True
        with pytest.raises(AttributeError):
            s.server_hostname = "asdf"

        # but .context is *not*. Check that we forward attribute setting by
        # making sure that after we set the bad context our handshake indeed
        # fails:
        s.context = bad_ctx
        assert s.context is bad_ctx
        with pytest.raises(BrokenResourceError) as excinfo:
            await s.do_handshake()
        assert isinstance(excinfo.value.__cause__, ssl.SSLError)


# Note: this test fails horribly if we force TLS 1.2 and trigger a
# renegotiation at the beginning (e.g. by switching to the pyopenssl
# server). Usually the client crashes in SSLObject.write with "UNEXPECTED
# RECORD"; sometimes we get something more exotic like a SyscallError. This is
# odd because openssl isn't doing any syscalls, but so it goes. After lots of
# websearching I'm pretty sure this is due to a bug in OpenSSL, where it just
# can't reliably handle full-duplex communication combined with
# renegotiation. Nice, eh?
#
#   https://rt.openssl.org/Ticket/Display.html?id=3712
#   https://rt.openssl.org/Ticket/Display.html?id=2481
#   http://openssl.6102.n7.nabble.com/TLS-renegotiation-failure-on-receiving-application-data-during-handshake-td48127.html
#   https://stackoverflow.com/questions/18728355/ssl-renegotiation-with-full-duplex-socket-communication
#
# In some variants of this test (maybe only against the java server?) I've
# also seen cases where our send_all blocks waiting to write, and then our receive_some
# also blocks waiting to write, and they never wake up again. It looks like
# some kind of deadlock. I suspect there may be an issue where we've filled up
# the send buffers, and the remote side is trying to handle the renegotiation
# from inside a write() call, so it has a problem: there's all this application
# data clogging up the pipe, but it can't process and return it to the
# application because it's in write(), and it doesn't want to buffer infinite
# amounts of data, and... actually I guess those are the only two choices.
#
# NSS even documents that you shouldn't try to do a renegotiation except when
# the connection is idle:
#
#   https://developer.mozilla.org/en-US/docs/Mozilla/Projects/NSS/SSL_functions/sslfnc.html#1061582
#
# I begin to see why HTTP/2 forbids renegotiation and TLS 1.3 removes it...


async def test_full_duplex_basics(client_ctx: SSLContext) -> None:
    CHUNKS = 30
    CHUNK_SIZE = 32768
    EXPECTED = CHUNKS * CHUNK_SIZE

    sent = bytearray()
    received = bytearray()

    async def sender(s: Stream) -> None:
        nonlocal sent
        for i in range(CHUNKS):
            print(i)
            chunk = bytes([i] * CHUNK_SIZE)
            sent += chunk
            await s.send_all(chunk)

    async def receiver(s: Stream) -> None:
        nonlocal received
        while len(received) < EXPECTED:
            chunk = await s.receive_some(CHUNK_SIZE // 2)
            received += chunk

    async with ssl_echo_server(client_ctx) as s:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sender, s)
            nursery.start_soon(receiver, s)
            # And let's have some doing handshakes too, everyone
            # simultaneously
            nursery.start_soon(s.do_handshake)
            nursery.start_soon(s.do_handshake)

        await s.aclose()

    assert len(sent) == len(received) == EXPECTED
    assert sent == received


async def test_renegotiation_simple(client_ctx: SSLContext) -> None:
    with virtual_ssl_echo_server(client_ctx) as s:
        await s.do_handshake()
        s.transport_stream.renegotiate()
        await s.send_all(b"a")
        assert await s.receive_some(1) == b"a"

        # Have to send some more data back and forth to make sure the
        # renegotiation is finished before shutting down the
        # connection... otherwise openssl raises an error. I think this is a
        # bug in openssl but what can ya do.
        await s.send_all(b"b")
        assert await s.receive_some(1) == b"b"

        await s.aclose()


@slow
async def test_renegotiation_randomized(
    mock_clock: MockClock,
    client_ctx: SSLContext,
) -> None:
    # The only blocking things in this function are our random sleeps, so 0 is
    # a good threshold.
    mock_clock.autojump_threshold = 0

    import random

    r = random.Random(0)

    async def sleeper(_: object) -> None:
        await trio.sleep(r.uniform(0, 10))

    async def clear() -> None:
        while s.transport_stream.renegotiate_pending():
            with assert_checkpoints():
                await send(b"-")
            with assert_checkpoints():
                await expect(b"-")
        print("-- clear --")

    async def send(byte: bytes) -> None:
        await s.transport_stream.sleeper("outer send")
        print("calling SSLStream.send_all", byte)
        with assert_checkpoints():
            await s.send_all(byte)

    async def expect(expected: bytes) -> None:
        await s.transport_stream.sleeper("expect")
        print("calling SSLStream.receive_some, expecting", expected)
        assert len(expected) == 1
        with assert_checkpoints():
            assert await s.receive_some(1) == expected

    with virtual_ssl_echo_server(client_ctx, sleeper=sleeper) as s:
        await s.do_handshake()

        await send(b"a")
        s.transport_stream.renegotiate()
        await expect(b"a")

        await clear()

        for i in range(100):
            b1 = bytes([i % 0xFF])
            b2 = bytes([(2 * i) % 0xFF])
            s.transport_stream.renegotiate()
            async with _core.open_nursery() as nursery:
                nursery.start_soon(send, b1)
                nursery.start_soon(expect, b1)
            async with _core.open_nursery() as nursery:
                nursery.start_soon(expect, b2)
                nursery.start_soon(send, b2)
            await clear()

        for i in range(100):
            b1 = bytes([i % 0xFF])
            b2 = bytes([(2 * i) % 0xFF])
            await send(b1)
            s.transport_stream.renegotiate()
            await expect(b1)
            async with _core.open_nursery() as nursery:
                nursery.start_soon(expect, b2)
                nursery.start_soon(send, b2)
            await clear()

    # Checking that wait_send_all_might_not_block and receive_some don't
    # conflict:

    # 1) Set up a situation where expect (receive_some) is blocked sending,
    # and wait_send_all_might_not_block comes in.

    # Our receive_some() call will get stuck when it hits send_all
    async def sleeper_with_slow_send_all(method: str) -> None:
        if method == "send_all":
            # ignore ASYNC116, not sleep_forever, trying to test a large but finite sleep
            await trio.sleep(100000)  # noqa: ASYNC116

    # And our wait_send_all_might_not_block call will give it time to get
    # stuck, and then start
    async def sleep_then_wait_writable() -> None:
        await trio.sleep(1000)
        await s.wait_send_all_might_not_block()

    with virtual_ssl_echo_server(client_ctx, sleeper=sleeper_with_slow_send_all) as s:
        await send(b"x")
        s.transport_stream.renegotiate()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect, b"x")
            nursery.start_soon(sleep_then_wait_writable)

        await clear()

        await s.aclose()

    # 2) Same, but now wait_send_all_might_not_block is stuck when
    # receive_some tries to send.

    async def sleeper_with_slow_wait_writable_and_expect(method: str) -> None:
        if method == "wait_send_all_might_not_block":
            # ignore ASYNC116, not sleep_forever, trying to test a large but finite sleep
            await trio.sleep(100000)  # noqa: ASYNC116
        elif method == "expect":
            await trio.sleep(1000)

    with virtual_ssl_echo_server(
        client_ctx,
        sleeper=sleeper_with_slow_wait_writable_and_expect,
    ) as s:
        await send(b"x")
        s.transport_stream.renegotiate()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect, b"x")
            nursery.start_soon(s.wait_send_all_might_not_block)

        await clear()

        await s.aclose()


async def test_resource_busy_errors(client_ctx: SSLContext) -> None:
    S: TypeAlias = trio.SSLStream[
        trio.StapledStream[trio.abc.SendStream, trio.abc.ReceiveStream]
    ]

    async def do_send_all(s: S) -> None:
        with assert_checkpoints():
            await s.send_all(b"x")

    async def do_receive_some(s: S) -> None:
        with assert_checkpoints():
            await s.receive_some(1)

    async def do_wait_send_all_might_not_block(s: S) -> None:
        with assert_checkpoints():
            await s.wait_send_all_might_not_block()

    async def do_test(
        func1: Callable[[S], Awaitable[None]],
        func2: Callable[[S], Awaitable[None]],
    ) -> None:
        s, _ = ssl_lockstep_stream_pair(client_ctx)
        with RaisesGroup(Matcher(_core.BusyResourceError, "another task")):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(func1, s)
                nursery.start_soon(func2, s)

    await do_test(do_send_all, do_send_all)
    await do_test(do_receive_some, do_receive_some)
    await do_test(do_send_all, do_wait_send_all_might_not_block)
    await do_test(do_wait_send_all_might_not_block, do_wait_send_all_might_not_block)


async def test_wait_writable_calls_underlying_wait_writable() -> None:
    record = []

    class NotAStream(Stream):
        async def wait_send_all_might_not_block(self) -> None:
            record.append("ok")

        # define methods that are abstract in Stream
        async def aclose(self) -> None:
            raise AssertionError("Should not get called")  # pragma: no cover

        async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray:
            raise AssertionError("Should not get called")  # pragma: no cover

        async def send_all(self, data: bytes | bytearray | memoryview) -> None:
            raise AssertionError("Should not get called")  # pragma: no cover

    ctx = ssl.create_default_context()
    s = SSLStream(NotAStream(), ctx, server_hostname="x")
    await s.wait_send_all_might_not_block()
    assert record == ["ok"]


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info >= (3, 10),
    reason="frequently fails on Windows + Python 3.10",
)
async def test_checkpoints(client_ctx: SSLContext) -> None:
    async with ssl_echo_server(client_ctx) as s:
        with assert_checkpoints():
            await s.do_handshake()
        with assert_checkpoints():
            await s.do_handshake()
        with assert_checkpoints():
            await s.wait_send_all_might_not_block()
        with assert_checkpoints():
            await s.send_all(b"xxx")
        with assert_checkpoints():
            await s.receive_some(1)
        # These receive_some's in theory could return immediately, because the
        # "xxx" was sent in a single record and after the first
        # receive_some(1) the rest are sitting inside the SSLObject's internal
        # buffers.
        with assert_checkpoints():
            await s.receive_some(1)
        with assert_checkpoints():
            await s.receive_some(1)
        with assert_checkpoints():
            await s.unwrap()

    async with ssl_echo_server(client_ctx) as s:
        await s.do_handshake()
        with assert_checkpoints():
            await s.aclose()


async def test_send_all_empty_string(client_ctx: SSLContext) -> None:
    async with ssl_echo_server(client_ctx) as s:
        await s.do_handshake()

        # underlying SSLObject interprets writing b"" as indicating an EOF,
        # for some reason. Make sure we don't inherit this.
        with assert_checkpoints():
            await s.send_all(b"")
        with assert_checkpoints():
            await s.send_all(b"")
        await s.send_all(b"x")
        assert await s.receive_some(1) == b"x"

        await s.aclose()


@pytest.mark.parametrize("https_compatible", [False, True])
async def test_SSLStream_generic(
    client_ctx: SSLContext,
    https_compatible: bool,
) -> None:
    async def stream_maker() -> tuple[
        SSLStream[MemoryStapledStream],
        SSLStream[MemoryStapledStream],
    ]:
        return ssl_memory_stream_pair(
            client_ctx,
            client_kwargs={"https_compatible": https_compatible},
            server_kwargs={"https_compatible": https_compatible},
        )

    async def clogged_stream_maker() -> tuple[
        SSLStream[MyStapledStream],
        SSLStream[MyStapledStream],
    ]:
        client, server = ssl_lockstep_stream_pair(client_ctx)
        # If we don't do handshakes up front, then we run into a problem in
        # the following situation:
        # - server does wait_send_all_might_not_block
        # - client does receive_some to unclog it
        # Then the client's receive_some will actually send some data to start
        # the handshake, and itself get stuck.
        async with _core.open_nursery() as nursery:
            nursery.start_soon(client.do_handshake)
            nursery.start_soon(server.do_handshake)
        return client, server

    await check_two_way_stream(stream_maker, clogged_stream_maker)


async def test_unwrap(client_ctx: SSLContext) -> None:
    client_ssl, server_ssl = ssl_memory_stream_pair(client_ctx)
    client_transport = client_ssl.transport_stream
    server_transport = server_ssl.transport_stream

    seq = Sequencer()

    async def client() -> None:
        await client_ssl.do_handshake()
        await client_ssl.send_all(b"x")
        assert await client_ssl.receive_some(1) == b"y"
        await client_ssl.send_all(b"z")

        # After sending that, disable outgoing data from our end, to make
        # sure the server doesn't see our EOF until after we've sent some
        # trailing data
        async with seq(0):
            send_all_hook = client_transport.send_stream.send_all_hook
            client_transport.send_stream.send_all_hook = None

        assert await client_ssl.receive_some(1) == b""
        assert client_ssl.transport_stream is client_transport
        # We just received EOF. Unwrap the connection and send some more.
        raw, trailing = await client_ssl.unwrap()
        assert raw is client_transport
        assert trailing == b""
        assert client_ssl.transport_stream is None
        await raw.send_all(b"trailing")

        # Reconnect the streams. Now the server will receive both our shutdown
        # acknowledgement + the trailing data in a single lump.
        client_transport.send_stream.send_all_hook = send_all_hook
        await client_transport.send_stream.send_all_hook()

    async def server() -> None:
        await server_ssl.do_handshake()
        assert await server_ssl.receive_some(1) == b"x"
        await server_ssl.send_all(b"y")
        assert await server_ssl.receive_some(1) == b"z"
        # Now client is blocked waiting for us to send something, but
        # instead we close the TLS connection (with sequencer to make sure
        # that the client won't see and automatically respond before we've had
        # a chance to disable the client->server transport)
        async with seq(1):
            raw, trailing = await server_ssl.unwrap()
        assert raw is server_transport
        assert trailing == b"trailing"
        assert server_ssl.transport_stream is None

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client)
        nursery.start_soon(server)


async def test_closing_nice_case(client_ctx: SSLContext) -> None:
    # the nice case: graceful closes all around

    client_ssl, server_ssl = ssl_memory_stream_pair(client_ctx)
    client_transport = client_ssl.transport_stream

    # Both the handshake and the close require back-and-forth discussion, so
    # we need to run them concurrently
    async def client_closer() -> None:
        with assert_checkpoints():
            await client_ssl.aclose()

    async def server_closer() -> None:
        assert await server_ssl.receive_some(10) == b""
        assert await server_ssl.receive_some(10) == b""
        with assert_checkpoints():
            await server_ssl.aclose()

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client_closer)
        nursery.start_soon(server_closer)

    # closing the SSLStream also closes its transport
    with pytest.raises(ClosedResourceError):
        await client_transport.send_all(b"123")

    # once closed, it's OK to close again
    with assert_checkpoints():
        await client_ssl.aclose()
    with assert_checkpoints():
        await client_ssl.aclose()

    # Trying to send more data does not work
    with pytest.raises(ClosedResourceError):
        await server_ssl.send_all(b"123")

    # And once the connection is has been closed *locally*, then instead of
    # getting empty bytestrings we get a proper error
    with pytest.raises(ClosedResourceError):
        assert await client_ssl.receive_some(10) == b""

    with pytest.raises(ClosedResourceError):
        await client_ssl.unwrap()

    with pytest.raises(ClosedResourceError):
        await client_ssl.do_handshake()

    # Check that a graceful close *before* handshaking gives a clean EOF on
    # the other side
    client_ssl, server_ssl = ssl_memory_stream_pair(client_ctx)

    async def expect_eof_server() -> None:
        with assert_checkpoints():
            assert await server_ssl.receive_some(10) == b""
        with assert_checkpoints():
            await server_ssl.aclose()

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client_ssl.aclose)
        nursery.start_soon(expect_eof_server)


async def test_send_all_fails_in_the_middle(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(client_ctx)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    async def bad_hook() -> NoReturn:
        raise KeyError

    client.transport_stream.send_stream.send_all_hook = bad_hook

    with pytest.raises(KeyError):
        await client.send_all(b"x")

    with pytest.raises(BrokenResourceError):
        await client.wait_send_all_might_not_block()

    closed = 0

    def close_hook() -> None:
        nonlocal closed
        closed += 1

    client.transport_stream.send_stream.close_hook = close_hook
    client.transport_stream.receive_stream.close_hook = close_hook
    await client.aclose()

    assert closed == 2


async def test_ssl_over_ssl(client_ctx: SSLContext) -> None:
    client_0, server_0 = memory_stream_pair()

    client_1 = SSLStream(
        client_0,
        client_ctx,
        server_hostname="trio-test-1.example.org",
    )
    server_1 = SSLStream(server_0, SERVER_CTX, server_side=True)

    client_2 = SSLStream(
        client_1,
        client_ctx,
        server_hostname="trio-test-1.example.org",
    )
    server_2 = SSLStream(server_1, SERVER_CTX, server_side=True)

    async def client() -> None:
        await client_2.send_all(b"hi")
        assert await client_2.receive_some(10) == b"bye"

    async def server() -> None:
        assert await server_2.receive_some(10) == b"hi"
        await server_2.send_all(b"bye")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client)
        nursery.start_soon(server)


async def test_ssl_bad_shutdown(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(client_ctx)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    await trio.aclose_forcefully(client)
    # now the server sees a broken stream
    with pytest.raises(BrokenResourceError):
        await server.receive_some(10)
    with pytest.raises(BrokenResourceError):
        await server.send_all(b"x" * 10)

    await server.aclose()


async def test_ssl_bad_shutdown_but_its_ok(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(
        client_ctx,
        server_kwargs={"https_compatible": True},
        client_kwargs={"https_compatible": True},
    )

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    await trio.aclose_forcefully(client)
    # the server sees that as a clean shutdown
    assert await server.receive_some(10) == b""
    with pytest.raises(BrokenResourceError):
        await server.send_all(b"x" * 10)

    await server.aclose()


async def test_ssl_handshake_failure_during_aclose() -> None:
    # Weird scenario: aclose() triggers an automatic handshake, and this
    # fails. This also exercises a bit of code in aclose() that was otherwise
    # uncovered, for re-raising exceptions after calling aclose_forcefully on
    # the underlying transport.
    async with ssl_echo_server_raw(expect_fail=True) as sock:
        # Don't configure trust correctly
        client_ctx = ssl.create_default_context()
        s = SSLStream(sock, client_ctx, server_hostname="trio-test-1.example.org")
        # It's a little unclear here whether aclose should swallow the error
        # or let it escape. We *do* swallow the error if it arrives when we're
        # sending close_notify, because both sides closing the connection
        # simultaneously is allowed. But I guess when https_compatible=False
        # then it's bad if we can get through a whole connection with a peer
        # that has no valid certificate, and never raise an error.
        with pytest.raises(BrokenResourceError):
            await s.aclose()


async def test_ssl_only_closes_stream_once(client_ctx: SSLContext) -> None:
    # We used to have a bug where if transport_stream.aclose() raised an
    # error, we would call it again. This checks that that's fixed.
    client, server = ssl_memory_stream_pair(client_ctx)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    client_orig_close_hook = client.transport_stream.send_stream.close_hook
    transport_close_count = 0

    def close_hook() -> NoReturn:
        nonlocal transport_close_count
        assert client_orig_close_hook is not None
        client_orig_close_hook()
        transport_close_count += 1
        raise KeyError

    client.transport_stream.send_stream.close_hook = close_hook

    with pytest.raises(KeyError):
        await client.aclose()
    assert transport_close_count == 1


async def test_ssl_https_compatibility_disagreement(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(
        client_ctx,
        server_kwargs={"https_compatible": False},
        client_kwargs={"https_compatible": True},
    )

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    # client is in HTTPS-mode, server is not
    # so client doing graceful_shutdown causes an error on server
    async def receive_and_expect_error() -> None:
        with pytest.raises(BrokenResourceError) as excinfo:
            await server.receive_some(10)

        assert _is_eof(excinfo.value.__cause__)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.aclose)
        nursery.start_soon(receive_and_expect_error)


async def test_https_mode_eof_before_handshake(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(
        client_ctx,
        server_kwargs={"https_compatible": True},
        client_kwargs={"https_compatible": True},
    )

    async def server_expect_clean_eof() -> None:
        assert await server.receive_some(10) == b""

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.aclose)
        nursery.start_soon(server_expect_clean_eof)


async def test_send_error_during_handshake(client_ctx: SSLContext) -> None:
    client, _server = ssl_memory_stream_pair(client_ctx)

    async def bad_hook() -> NoReturn:
        raise KeyError

    client.transport_stream.send_stream.send_all_hook = bad_hook

    with pytest.raises(KeyError):
        with assert_checkpoints():
            await client.do_handshake()

    with pytest.raises(BrokenResourceError):
        with assert_checkpoints():
            await client.do_handshake()


async def test_receive_error_during_handshake(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(client_ctx)

    async def bad_hook() -> NoReturn:
        raise KeyError

    client.transport_stream.receive_stream.receive_some_hook = bad_hook

    async def client_side(cancel_scope: CancelScope) -> None:
        with pytest.raises(KeyError):
            with assert_checkpoints():
                await client.do_handshake()
        cancel_scope.cancel()

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client_side, nursery.cancel_scope)
        nursery.start_soon(server.do_handshake)

    with pytest.raises(BrokenResourceError):
        with assert_checkpoints():
            await client.do_handshake()


def test_selected_alpn_protocol_before_handshake(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(client_ctx)

    with pytest.raises(NeedHandshakeError):
        client.selected_alpn_protocol()

    with pytest.raises(NeedHandshakeError):
        server.selected_alpn_protocol()


async def test_selected_alpn_protocol_when_not_set(client_ctx: SSLContext) -> None:
    # ALPN protocol still returns None when it's not set,
    # instead of raising an exception
    client, server = ssl_memory_stream_pair(client_ctx)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    assert client.selected_alpn_protocol() is None
    assert server.selected_alpn_protocol() is None

    assert client.selected_alpn_protocol() == server.selected_alpn_protocol()


def test_selected_npn_protocol_before_handshake(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(client_ctx)

    with pytest.raises(NeedHandshakeError):
        client.selected_npn_protocol()

    with pytest.raises(NeedHandshakeError):
        server.selected_npn_protocol()


@pytest.mark.filterwarnings(
    r"ignore: ssl module. NPN is deprecated, use ALPN instead:UserWarning",
    r"ignore:ssl NPN is deprecated, use ALPN instead:DeprecationWarning",
)
async def test_selected_npn_protocol_when_not_set(client_ctx: SSLContext) -> None:
    # NPN protocol still returns None when it's not set,
    # instead of raising an exception
    client, server = ssl_memory_stream_pair(client_ctx)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    assert client.selected_npn_protocol() is None
    assert server.selected_npn_protocol() is None

    assert client.selected_npn_protocol() == server.selected_npn_protocol()


def test_get_channel_binding_before_handshake(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(client_ctx)

    with pytest.raises(NeedHandshakeError):
        client.get_channel_binding()

    with pytest.raises(NeedHandshakeError):
        server.get_channel_binding()


async def test_get_channel_binding_after_handshake(client_ctx: SSLContext) -> None:
    client, server = ssl_memory_stream_pair(client_ctx)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    assert client.get_channel_binding() is not None
    assert server.get_channel_binding() is not None

    assert client.get_channel_binding() == server.get_channel_binding()


async def test_getpeercert(client_ctx: SSLContext) -> None:
    # Make sure we're not affected by https://bugs.python.org/issue29334
    client, server = ssl_memory_stream_pair(client_ctx)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(client.do_handshake)
        nursery.start_soon(server.do_handshake)

    assert server.getpeercert() is None
    print(client.getpeercert())
    assert ("DNS", "trio-test-1.example.org") in client.getpeercert()["subjectAltName"]


async def test_SSLListener(client_ctx: SSLContext) -> None:
    async def setup(
        https_compatible: bool = False,
    ) -> tuple[tsocket.SocketType, SSLListener[SocketStream], SSLStream[SocketStream]]:
        listen_sock = tsocket.socket()
        await listen_sock.bind(("127.0.0.1", 0))
        listen_sock.listen(1)
        socket_listener = SocketListener(listen_sock)
        ssl_listener = SSLListener(
            socket_listener,
            SERVER_CTX,
            https_compatible=https_compatible,
        )

        transport_client = await open_tcp_stream(*listen_sock.getsockname())
        ssl_client = SSLStream(
            transport_client,
            client_ctx,
            server_hostname="trio-test-1.example.org",
        )
        return listen_sock, ssl_listener, ssl_client

    listen_sock, ssl_listener, ssl_client = await setup()

    async with ssl_client:
        ssl_server = await ssl_listener.accept()

        async with ssl_server:
            assert not ssl_server._https_compatible

            # Make sure the connection works
            async with _core.open_nursery() as nursery:
                nursery.start_soon(ssl_client.do_handshake)
                nursery.start_soon(ssl_server.do_handshake)

        # Test SSLListener.aclose
        await ssl_listener.aclose()
        assert listen_sock.fileno() == -1

    ################

    # Test https_compatible
    _, ssl_listener, ssl_client = await setup(https_compatible=True)

    ssl_server = await ssl_listener.accept()

    assert ssl_server._https_compatible

    await aclose_forcefully(ssl_listener)
    await aclose_forcefully(ssl_client)
    await aclose_forcefully(ssl_server)