
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\format.py ===
"""
Internal module for formatting output data in csv, html, xml,
and latex files. This module also applies to display formatting.
"""
from __future__ import annotations

from collections.abc import (
    Generator,
    Hashable,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    cast,
)

import numpy as np

from pandas._config.config import (
    get_option,
    set_option,
)

from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
    NaT,
    Timedelta,
    Timestamp,
)
from pandas._libs.tslibs.nattype import NaTType

from pandas.core.dtypes.common import (
    is_complex_dtype,
    is_float,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    ExtensionArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    PeriodIndex,
    ensure_index,
)
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat

from pandas.io.common import (
    check_parent_directory,
    stringify_path,
)
from pandas.io.formats import printing

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Axes,
        ColspaceArgType,
        ColspaceType,
        CompressionOptions,
        FilePath,
        FloatFormatType,
        FormattersType,
        IndexLabel,
        SequenceNotStr,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import (
        DataFrame,
        Series,
    )


common_docstring: Final = """
        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        columns : array-like, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : %(col_space_type)s, optional
            %(col_space)s.
        header : %(header_type)s, optional
            %(header)s.
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of ``NaN`` to use.
        formatters : list, tuple or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List/tuple must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. This function must return a unicode string and will be
            applied only to the non-``NaN`` elements, with ``NaN`` being
            handled by ``na_rep``.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
    """

VALID_JUSTIFY_PARAMETERS = (
    "left",
    "right",
    "center",
    "justify",
    "justify-all",
    "start",
    "end",
    "inherit",
    "match-parent",
    "initial",
    "unset",
)

return_docstring: Final = """
        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns
            None.
    """


class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """

    def __init__(
        self,
        series: Series,
        *,
        length: bool | str = True,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        name: bool = False,
        float_format: str | None = None,
        dtype: bool = True,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> None:
        self.series = series
        self.buf = StringIO()
        self.name = name
        self.na_rep = na_rep
        self.header = header
        self.length = length
        self.index = index
        self.max_rows = max_rows
        self.min_rows = min_rows

        if float_format is None:
            float_format = get_option("display.float_format")
        self.float_format = float_format
        self.dtype = dtype
        self.adj = printing.get_adjustment()

        self._chk_truncate()

    def _chk_truncate(self) -> None:
        self.tr_row_num: int | None

        min_rows = self.min_rows
        max_rows = self.max_rows
        # truncation determined by max_rows, actual truncated number of rows
        # used below by min_rows
        is_truncated_vertically = max_rows and (len(self.series) > max_rows)
        series = self.series
        if is_truncated_vertically:
            max_rows = cast(int, max_rows)
            if min_rows:
                # if min_rows is set (not None or 0), set max_rows to minimum
                # of both
                max_rows = min(min_rows, max_rows)
            if max_rows == 1:
                row_num = max_rows
                series = series.iloc[:max_rows]
            else:
                row_num = max_rows // 2
                series = concat((series.iloc[:row_num], series.iloc[-row_num:]))
            self.tr_row_num = row_num
        else:
            self.tr_row_num = None
        self.tr_series = series
        self.is_truncated_vertically = is_truncated_vertically

    def _get_footer(self) -> str:
        name = self.series.name
        footer = ""

        index = self.series.index
        if (
            isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex))
            and index.freq is not None
        ):
            footer += f"Freq: {index.freqstr}"

        if self.name is not False and name is not None:
            if footer:
                footer += ", "

            series_name = printing.pprint_thing(name, escape_chars=("\t", "\r", "\n"))
            footer += f"Name: {series_name}"

        if self.length is True or (
            self.length == "truncate" and self.is_truncated_vertically
        ):
            if footer:
                footer += ", "
            footer += f"Length: {len(self.series)}"

        if self.dtype is not False and self.dtype is not None:
            dtype_name = getattr(self.tr_series.dtype, "name", None)
            if dtype_name:
                if footer:
                    footer += ", "
                footer += f"dtype: {printing.pprint_thing(dtype_name)}"

        # level infos are added to the end and in a new line, like it is done
        # for Categoricals
        if isinstance(self.tr_series.dtype, CategoricalDtype):
            level_info = self.tr_series._values._get_repr_footer()
            if footer:
                footer += "\n"
            footer += level_info

        return str(footer)

    def _get_formatted_values(self) -> list[str]:
        return format_array(
            self.tr_series._values,
            None,
            float_format=self.float_format,
            na_rep=self.na_rep,
            leading_space=self.index,
        )

    def to_string(self) -> str:
        series = self.tr_series
        footer = self._get_footer()

        if len(series) == 0:
            return f"{type(self.series).__name__}([], {footer})"

        index = series.index
        have_header = _has_names(index)
        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(include_names=True, sparsify=None)
            adj = printing.get_adjustment()
            fmt_index = adj.adjoin(2, *fmt_index).split("\n")
        else:
            fmt_index = index._format_flat(include_name=True)
        fmt_values = self._get_formatted_values()

        if self.is_truncated_vertically:
            n_header_rows = 0
            row_num = self.tr_row_num
            row_num = cast(int, row_num)
            width = self.adj.len(fmt_values[row_num - 1])
            if width > 3:
                dot_str = "..."
            else:
                dot_str = ".."
            # Series uses mode=center because it has single value columns
            # DataFrame uses mode=left
            dot_str = self.adj.justify([dot_str], width, mode="center")[0]
            fmt_values.insert(row_num + n_header_rows, dot_str)
            fmt_index.insert(row_num + 1, "")

        if self.index:
            result = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            result = self.adj.adjoin(3, fmt_values)

        if self.header and have_header:
            result = fmt_index[0] + "\n" + result

        if footer:
            result += "\n" + footer

        return str("".join(result))


def get_dataframe_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(dataFrame) calls using DataFrame.to_string.

    Supplying these parameters to DataFrame.to_string is equivalent to calling
    ``repr(DataFrame)``. This is useful if you want to adjust the repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame([[1, 2], [3, 4]])
    >>> repr_params = pd.io.formats.format.get_dataframe_repr_params()
    >>> repr(df) == df.to_string(**repr_params)
    True
    """
    from pandas.io.formats import console

    if get_option("display.expand_frame_repr"):
        line_width, _ = console.get_console_size()
    else:
        line_width = None
    return {
        "max_rows": get_option("display.max_rows"),
        "min_rows": get_option("display.min_rows"),
        "max_cols": get_option("display.max_columns"),
        "max_colwidth": get_option("display.max_colwidth"),
        "show_dimensions": get_option("display.show_dimensions"),
        "line_width": line_width,
    }


def get_series_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(Series) calls using Series.to_string.

    Supplying these parameters to Series.to_string is equivalent to calling
    ``repr(series)``. This is useful if you want to adjust the series repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> ser = pd.Series([1, 2, 3, 4])
    >>> repr_params = pd.io.formats.format.get_series_repr_params()
    >>> repr(ser) == ser.to_string(**repr_params)
    True
    """
    width, height = get_terminal_size()
    max_rows_opt = get_option("display.max_rows")
    max_rows = height if max_rows_opt == 0 else max_rows_opt
    min_rows = height if max_rows_opt == 0 else get_option("display.min_rows")

    return {
        "name": True,
        "dtype": True,
        "min_rows": min_rows,
        "max_rows": max_rows,
        "length": get_option("display.show_dimensions"),
    }


class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """

    __doc__ = __doc__ if __doc__ else ""
    __doc__ += common_docstring + return_docstring

    def __init__(
        self,
        frame: DataFrame,
        columns: Axes | None = None,
        col_space: ColspaceArgType | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: FormattersType | None = None,
        justify: str | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        max_rows: int | None = None,
        min_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: bool | str = False,
        decimal: str = ".",
        bold_rows: bool = False,
        escape: bool = True,
    ) -> None:
        self.frame = frame
        self.columns = self._initialize_columns(columns)
        self.col_space = self._initialize_colspace(col_space)
        self.header = header
        self.index = index
        self.na_rep = na_rep
        self.formatters = self._initialize_formatters(formatters)
        self.justify = self._initialize_justify(justify)
        self.float_format = float_format
        self.sparsify = self._initialize_sparsify(sparsify)
        self.show_index_names = index_names
        self.decimal = decimal
        self.bold_rows = bold_rows
        self.escape = escape
        self.max_rows = max_rows
        self.min_rows = min_rows
        self.max_cols = max_cols
        self.show_dimensions = show_dimensions

        self.max_cols_fitted = self._calc_max_cols_fitted()
        self.max_rows_fitted = self._calc_max_rows_fitted()

        self.tr_frame = self.frame
        self.truncate()
        self.adj = printing.get_adjustment()

    def get_strcols(self) -> list[list[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols = self._get_strcols_without_index()

        if self.index:
            str_index = self._get_formatted_index(self.tr_frame)
            strcols.insert(0, str_index)

        return strcols

    @property
    def should_show_dimensions(self) -> bool:
        return self.show_dimensions is True or (
            self.show_dimensions == "truncate" and self.is_truncated
        )

    @property
    def is_truncated(self) -> bool:
        return bool(self.is_truncated_horizontally or self.is_truncated_vertically)

    @property
    def is_truncated_horizontally(self) -> bool:
        return bool(self.max_cols_fitted and (len(self.columns) > self.max_cols_fitted))

    @property
    def is_truncated_vertically(self) -> bool:
        return bool(self.max_rows_fitted and (len(self.frame) > self.max_rows_fitted))

    @property
    def dimensions_info(self) -> str:
        return f"\n\n[{len(self.frame)} rows x {len(self.frame.columns)} columns]"

    @property
    def has_index_names(self) -> bool:
        return _has_names(self.frame.index)

    @property
    def has_column_names(self) -> bool:
        return _has_names(self.frame.columns)

    @property
    def show_row_idx_names(self) -> bool:
        return all((self.has_index_names, self.index, self.show_index_names))

    @property
    def show_col_idx_names(self) -> bool:
        return all((self.has_column_names, self.show_index_names, self.header))

    @property
    def max_rows_displayed(self) -> int:
        return min(self.max_rows or len(self.frame), len(self.frame))

    def _initialize_sparsify(self, sparsify: bool | None) -> bool:
        if sparsify is None:
            return get_option("display.multi_sparse")
        return sparsify

    def _initialize_formatters(
        self, formatters: FormattersType | None
    ) -> FormattersType:
        if formatters is None:
            return {}
        elif len(self.frame.columns) == len(formatters) or isinstance(formatters, dict):
            return formatters
        else:
            raise ValueError(
                f"Formatters length({len(formatters)}) should match "
                f"DataFrame number of columns({len(self.frame.columns)})"
            )

    def _initialize_justify(self, justify: str | None) -> str:
        if justify is None:
            return get_option("display.colheader_justify")
        else:
            return justify

    def _initialize_columns(self, columns: Axes | None) -> Index:
        if columns is not None:
            cols = ensure_index(columns)
            self.frame = self.frame[cols]
            return cols
        else:
            return self.frame.columns

    def _initialize_colspace(self, col_space: ColspaceArgType | None) -> ColspaceType:
        result: ColspaceType

        if col_space is None:
            result = {}
        elif isinstance(col_space, (int, str)):
            result = {"": col_space}
            result.update({column: col_space for column in self.frame.columns})
        elif isinstance(col_space, Mapping):
            for column in col_space.keys():
                if column not in self.frame.columns and column != "":
                    raise ValueError(
                        f"Col_space is defined for an unknown column: {column}"
                    )
            result = col_space
        else:
            if len(self.frame.columns) != len(col_space):
                raise ValueError(
                    f"Col_space length({len(col_space)}) should match "
                    f"DataFrame number of columns({len(self.frame.columns)})"
                )
            result = dict(zip(self.frame.columns, col_space))
        return result

    def _calc_max_cols_fitted(self) -> int | None:
        """Number of columns fitting the screen."""
        if not self._is_in_terminal():
            return self.max_cols

        width, _ = get_terminal_size()
        if self._is_screen_narrow(width):
            return width
        else:
            return self.max_cols

    def _calc_max_rows_fitted(self) -> int | None:
        """Number of rows with data fitting the screen."""
        max_rows: int | None

        if self._is_in_terminal():
            _, height = get_terminal_size()
            if self.max_rows == 0:
                # rows available to fill with actual data
                return height - self._get_number_of_auxiliary_rows()

            if self._is_screen_short(height):
                max_rows = height
            else:
                max_rows = self.max_rows
        else:
            max_rows = self.max_rows

        return self._adjust_max_rows(max_rows)

    def _adjust_max_rows(self, max_rows: int | None) -> int | None:
        """Adjust max_rows using display logic.

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options

        GH #37359
        """
        if max_rows:
            if (len(self.frame) > max_rows) and self.min_rows:
                # if truncated, set max_rows showed to min_rows
                max_rows = min(self.min_rows, max_rows)
        return max_rows

    def _is_in_terminal(self) -> bool:
        """Check if the output is to be shown in terminal."""
        return bool(self.max_cols == 0 or self.max_rows == 0)

    def _is_screen_narrow(self, max_width) -> bool:
        return bool(self.max_cols == 0 and len(self.frame.columns) > max_width)

    def _is_screen_short(self, max_height) -> bool:
        return bool(self.max_rows == 0 and len(self.frame) > max_height)

    def _get_number_of_auxiliary_rows(self) -> int:
        """Get number of rows occupied by prompt, dots and dimension info."""
        dot_row = 1
        prompt_row = 1
        num_rows = dot_row + prompt_row

        if self.show_dimensions:
            num_rows += len(self.dimensions_info.splitlines())

        if self.header:
            num_rows += 1

        return num_rows

    def truncate(self) -> None:
        """
        Check whether the frame should be truncated. If so, slice the frame up.
        """
        if self.is_truncated_horizontally:
            self._truncate_horizontally()

        if self.is_truncated_vertically:
            self._truncate_vertically()

    def _truncate_horizontally(self) -> None:
        """Remove columns, which are not to be displayed and adjust formatters.

        Attributes affected:
            - tr_frame
            - formatters
            - tr_col_num
        """
        assert self.max_cols_fitted is not None
        col_num = self.max_cols_fitted // 2
        if col_num >= 1:
            left = self.tr_frame.iloc[:, :col_num]
            right = self.tr_frame.iloc[:, -col_num:]
            self.tr_frame = concat((left, right), axis=1)

            # truncate formatter
            if isinstance(self.formatters, (list, tuple)):
                self.formatters = [
                    *self.formatters[:col_num],
                    *self.formatters[-col_num:],
                ]
        else:
            col_num = cast(int, self.max_cols)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]
        self.tr_col_num = col_num

    def _truncate_vertically(self) -> None:
        """Remove rows, which are not to be displayed.

        Attributes affected:
            - tr_frame
            - tr_row_num
        """
        assert self.max_rows_fitted is not None
        row_num = self.max_rows_fitted // 2
        if row_num >= 1:
            _len = len(self.tr_frame)
            _slice = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
            self.tr_frame = self.tr_frame.iloc[_slice]
        else:
            row_num = cast(int, self.max_rows)
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        self.tr_row_num = row_num

    def _get_strcols_without_index(self) -> list[list[str]]:
        strcols: list[list[str]] = []

        if not is_list_like(self.header) and not self.header:
            for i, c in enumerate(self.tr_frame):
                fmt_values = self.format_col(i)
                fmt_values = _make_fixed_width(
                    strings=fmt_values,
                    justify=self.justify,
                    minimum=int(self.col_space.get(c, 0)),
                    adj=self.adj,
                )
                strcols.append(fmt_values)
            return strcols

        if is_list_like(self.header):
            # cast here since can't be bool if is_list_like
            self.header = cast(list[str], self.header)
            if len(self.header) != len(self.columns):
                raise ValueError(
                    f"Writing {len(self.columns)} cols "
                    f"but got {len(self.header)} aliases"
                )
            str_columns = [[label] for label in self.header]
        else:
            str_columns = self._get_formatted_column_labels(self.tr_frame)

        if self.show_row_idx_names:
            for x in str_columns:
                x.append("")

        for i, c in enumerate(self.tr_frame):
            cheader = str_columns[i]
            header_colwidth = max(
                int(self.col_space.get(c, 0)), *(self.adj.len(x) for x in cheader)
            )
            fmt_values = self.format_col(i)
            fmt_values = _make_fixed_width(
                fmt_values, self.justify, minimum=header_colwidth, adj=self.adj
            )

            max_len = max(*(self.adj.len(x) for x in fmt_values), header_colwidth)
            cheader = self.adj.justify(cheader, max_len, mode=self.justify)
            strcols.append(cheader + fmt_values)

        return strcols

    def format_col(self, i: int) -> list[str]:
        frame = self.tr_frame
        formatter = self._get_formatter(i)
        return format_array(
            frame.iloc[:, i]._values,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            space=self.col_space.get(frame.columns[i]),
            decimal=self.decimal,
            leading_space=self.index,
        )

    def _get_formatter(self, i: str | int) -> Callable | None:
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                i = cast(int, i)
                return self.formatters[i]
            else:
                return None
        else:
            if is_integer(i) and i not in self.columns:
                i = self.columns[i]
            return self.formatters.get(i, None)

    def _get_formatted_column_labels(self, frame: DataFrame) -> list[list[str]]:
        from pandas.core.indexes.multi import sparsify_labels

        columns = frame.columns

        if isinstance(columns, MultiIndex):
            fmt_columns = columns._format_multi(sparsify=False, include_names=False)
            fmt_columns = list(zip(*fmt_columns))
            dtypes = self.frame.dtypes._values

            # if we have a Float level, they don't use leading space at all
            restrict_formatting = any(level.is_floating for level in columns.levels)
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))

            def space_format(x, y):
                if (
                    y not in self.formatters
                    and need_leadsp[x]
                    and not restrict_formatting
                ):
                    return " " + y
                return y

            str_columns_tuple = list(
                zip(*([space_format(x, y) for y in x] for x in fmt_columns))
            )
            if self.sparsify and len(str_columns_tuple):
                str_columns_tuple = sparsify_labels(str_columns_tuple)

            str_columns = [list(x) for x in zip(*str_columns_tuple)]
        else:
            fmt_columns = columns._format_flat(include_name=False)
            dtypes = self.frame.dtypes
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))
            str_columns = [
                [" " + x if not self._get_formatter(i) and need_leadsp[x] else x]
                for i, x in enumerate(fmt_columns)
            ]
        # self.str_columns = str_columns
        return str_columns

    def _get_formatted_index(self, frame: DataFrame) -> list[str]:
        # Note: this is only used by to_string() and to_latex(), not by
        # to_html(). so safe to cast col_space here.
        col_space = {k: cast(int, v) for k, v in self.col_space.items()}
        index = frame.index
        columns = frame.columns
        fmt = self._get_formatter("__index__")

        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(
                sparsify=self.sparsify,
                include_names=self.show_row_idx_names,
                formatter=fmt,
            )
        else:
            fmt_index = [
                index._format_flat(include_name=self.show_row_idx_names, formatter=fmt)
            ]

        fmt_index = [
            tuple(
                _make_fixed_width(
                    list(x), justify="left", minimum=col_space.get("", 0), adj=self.adj
                )
            )
            for x in fmt_index
        ]

        adjoined = self.adj.adjoin(1, *fmt_index).split("\n")

        # empty space for columns
        if self.show_col_idx_names:
            col_header = [str(x) for x in self._get_column_name_list()]
        else:
            col_header = [""] * columns.nlevels

        if self.header:
            return col_header + adjoined
        else:
            return adjoined

    def _get_column_name_list(self) -> list[Hashable]:
        names: list[Hashable] = []
        columns = self.frame.columns
        if isinstance(columns, MultiIndex):
            names.extend("" if name is None else name for name in columns.names)
        else:
            names.append("" if columns.name is None else columns.name)
        return names


class DataFrameRenderer:
    """Class for creating dataframe output in multiple formats.

    Called in pandas.core.generic.NDFrame:
        - to_csv
        - to_latex

    Called in pandas.core.frame.DataFrame:
        - to_html
        - to_string

    Parameters
    ----------
    fmt : DataFrameFormatter
        Formatter with the formatting options.
    """

    def __init__(self, fmt: DataFrameFormatter) -> None:
        self.fmt = fmt

    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        classes: str | list | tuple | None = None,
        notebook: bool = False,
        border: int | bool | None = None,
        table_id: str | None = None,
        render_links: bool = False,
    ) -> str | None:
        """
        Render a DataFrame to a html table.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        classes : str or list-like
            classes to include in the `class` attribute of the opening
            ``<table>`` tag, in addition to the default "dataframe".
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            ``<table>`` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        """
        from pandas.io.formats.html import (
            HTMLFormatter,
            NotebookFormatter,
        )

        Klass = NotebookFormatter if notebook else HTMLFormatter

        html_formatter = Klass(
            self.fmt,
            classes=classes,
            border=border,
            table_id=table_id,
            render_links=render_links,
        )
        string = html_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        line_width: int | None = None,
    ) -> str | None:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding: str, default “utf-8”
            Set character encoding.
        line_width : int, optional
            Width to wrap a line in characters.
        """
        from pandas.io.formats.string import StringFormatter

        string_formatter = StringFormatter(self.fmt, line_width=line_width)
        string = string_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        sep: str = ",",
        columns: Sequence[Hashable] | None = None,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        quotechar: str = '"',
        lineterminator: str | None = None,
        chunksize: int | None = None,
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        errors: str = "strict",
        storage_options: StorageOptions | None = None,
    ) -> str | None:
        """
        Render dataframe as comma-separated file.
        """
        from pandas.io.formats.csvs import CSVFormatter

        if path_or_buf is None:
            created_buffer = True
            path_or_buf = StringIO()
        else:
            created_buffer = False

        csv_formatter = CSVFormatter(
            path_or_buf=path_or_buf,
            lineterminator=lineterminator,
            sep=sep,
            encoding=encoding,
            errors=errors,
            compression=compression,
            quoting=quoting,
            cols=columns,
            index_label=index_label,
            mode=mode,
            chunksize=chunksize,
            quotechar=quotechar,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            storage_options=storage_options,
            formatter=self.fmt,
        )
        csv_formatter.save()

        if created_buffer:
            assert isinstance(path_or_buf, StringIO)
            content = path_or_buf.getvalue()
            path_or_buf.close()
            return content

        return None


def save_to_buffer(
    string: str,
    buf: FilePath | WriteBuffer[str] | None = None,
    encoding: str | None = None,
) -> str | None:
    """
    Perform serialization. Write to buf or return as string if buf is None.
    """
    with _get_buffer(buf, encoding=encoding) as fd:
        fd.write(string)
        if buf is None:
            # error: "WriteBuffer[str]" has no attribute "getvalue"
            return fd.getvalue()  # type: ignore[attr-defined]
        return None


@contextmanager
def _get_buffer(
    buf: FilePath | WriteBuffer[str] | None, encoding: str | None = None
) -> Generator[WriteBuffer[str], None, None] | Generator[StringIO, None, None]:
    """
    Context manager to open, yield and close buffer for filenames or Path-like
    objects, otherwise yield buf unchanged.
    """
    if buf is not None:
        buf = stringify_path(buf)
    else:
        buf = StringIO()

    if encoding is None:
        encoding = "utf-8"
    elif not isinstance(buf, str):
        raise ValueError("buf is not a file name and encoding is specified.")

    if hasattr(buf, "write"):
        # Incompatible types in "yield" (actual type "Union[str, WriteBuffer[str],
        # StringIO]", expected type "Union[WriteBuffer[str], StringIO]")
        yield buf  # type: ignore[misc]
    elif isinstance(buf, str):
        check_parent_directory(str(buf))
        with open(buf, "w", encoding=encoding, newline="") as f:
            # GH#30034 open instead of codecs.open prevents a file leak
            #  if we have an invalid encoding argument.
            # newline="" is needed to roundtrip correctly on
            #  windows test_to_latex_filename
            yield f
    else:
        raise TypeError("buf is not a file name and it has no write method")


# ----------------------------------------------------------------------
# Array formatters


def format_array(
    values: ArrayLike,
    formatter: Callable | None,
    float_format: FloatFormatType | None = None,
    na_rep: str = "NaN",
    digits: int | None = None,
    space: str | int | None = None,
    justify: str = "right",
    decimal: str = ".",
    leading_space: bool | None = True,
    quoting: int | None = None,
    fallback_formatter: Callable | None = None,
) -> list[str]:
    """
    Format an array for printing.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    formatter
    float_format
    na_rep
    digits
    space
    justify
    decimal
    leading_space : bool, optional, default True
        Whether the array should be formatted with a leading space.
        When an array as a column of a Series or DataFrame, we do want
        the leading space to pad between columns.

        When formatting an Index subclass
        (e.g. IntervalIndex._get_values_for_csv), we don't want the
        leading space since it should be left-aligned.
    fallback_formatter

    Returns
    -------
    List[str]
    """
    fmt_klass: type[_GenericArrayFormatter]
    if lib.is_np_dtype(values.dtype, "M"):
        fmt_klass = _Datetime64Formatter
        values = cast(DatetimeArray, values)
    elif isinstance(values.dtype, DatetimeTZDtype):
        fmt_klass = _Datetime64TZFormatter
        values = cast(DatetimeArray, values)
    elif lib.is_np_dtype(values.dtype, "m"):
        fmt_klass = _Timedelta64Formatter
        values = cast(TimedeltaArray, values)
    elif isinstance(values.dtype, ExtensionDtype):
        fmt_klass = _ExtensionArrayFormatter
    elif lib.is_np_dtype(values.dtype, "fc"):
        fmt_klass = FloatArrayFormatter
    elif lib.is_np_dtype(values.dtype, "iu"):
        fmt_klass = _IntArrayFormatter
    else:
        fmt_klass = _GenericArrayFormatter

    if space is None:
        space = 12

    if float_format is None:
        float_format = get_option("display.float_format")

    if digits is None:
        digits = get_option("display.precision")

    fmt_obj = fmt_klass(
        values,
        digits=digits,
        na_rep=na_rep,
        float_format=float_format,
        formatter=formatter,
        space=space,
        justify=justify,
        decimal=decimal,
        leading_space=leading_space,
        quoting=quoting,
        fallback_formatter=fallback_formatter,
    )

    return fmt_obj.get_result()


class _GenericArrayFormatter:
    def __init__(
        self,
        values: ArrayLike,
        digits: int = 7,
        formatter: Callable | None = None,
        na_rep: str = "NaN",
        space: str | int = 12,
        float_format: FloatFormatType | None = None,
        justify: str = "right",
        decimal: str = ".",
        quoting: int | None = None,
        fixed_width: bool = True,
        leading_space: bool | None = True,
        fallback_formatter: Callable | None = None,
    ) -> None:
        self.values = values
        self.digits = digits
        self.na_rep = na_rep
        self.space = space
        self.formatter = formatter
        self.float_format = float_format
        self.justify = justify
        self.decimal = decimal
        self.quoting = quoting
        self.fixed_width = fixed_width
        self.leading_space = leading_space
        self.fallback_formatter = fallback_formatter

    def get_result(self) -> list[str]:
        fmt_values = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)

    def _format_strings(self) -> list[str]:
        if self.float_format is None:
            float_format = get_option("display.float_format")
            if float_format is None:
                precision = get_option("display.precision")
                float_format = lambda x: _trim_zeros_single_float(
                    f"{x: .{precision:d}f}"
                )
        else:
            float_format = self.float_format

        if self.formatter is not None:
            formatter = self.formatter
        elif self.fallback_formatter is not None:
            formatter = self.fallback_formatter
        else:
            quote_strings = self.quoting is not None and self.quoting != QUOTE_NONE
            formatter = partial(
                printing.pprint_thing,
                escape_chars=("\t", "\r", "\n"),
                quote_strings=quote_strings,
            )

        def _format(x):
            if self.na_rep is not None and is_scalar(x) and isna(x):
                if x is None:
                    return "None"
                elif x is NA:
                    return str(NA)
                elif lib.is_float(x) and np.isinf(x):
                    # TODO(3.0): this will be unreachable when use_inf_as_na
                    #  deprecation is enforced
                    return str(x)
                elif x is NaT or isinstance(x, (np.datetime64, np.timedelta64)):
                    return "NaT"
                return self.na_rep
            elif isinstance(x, PandasObject):
                return str(x)
            elif isinstance(x, StringDtype):
                return repr(x)
            else:
                # object dtype
                return str(formatter(x))

        vals = self.values
        if not isinstance(vals, np.ndarray):
            raise TypeError(
                "ExtensionArray formatting should use _ExtensionArrayFormatter"
            )
        inferred = lib.map_infer(vals, is_float)
        is_float_type = (
            inferred
            # vals may have 2 or more dimensions
            & np.all(notna(vals), axis=tuple(range(1, len(vals.shape))))
        )
        leading_space = self.leading_space
        if leading_space is None:
            leading_space = is_float_type.any()

        fmt_values = []
        for i, v in enumerate(vals):
            if (not is_float_type[i] or self.formatter is not None) and leading_space:
                fmt_values.append(f" {_format(v)}")
            elif is_float_type[i]:
                fmt_values.append(float_format(v))
            else:
                if leading_space is False:
                    # False specifically, so that the default is
                    # to include a space if we get here.
                    tpl = "{v}"
                else:
                    tpl = " {v}"
                fmt_values.append(tpl.format(v=_format(v)))

        return fmt_values


class FloatArrayFormatter(_GenericArrayFormatter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # float_format is expected to be a string
        # formatter should be used to pass a function
        if self.float_format is not None and self.formatter is None:
            # GH21625, GH22270
            self.fixed_width = False
            if callable(self.float_format):
                self.formatter = self.float_format
                self.float_format = None

    def _value_formatter(
        self,
        float_format: FloatFormatType | None = None,
        threshold: float | None = None,
    ) -> Callable:
        """Returns a function to be applied on each value to format it"""
        # the float_format parameter supersedes self.float_format
        if float_format is None:
            float_format = self.float_format

        # we are going to compose different functions, to first convert to
        # a string, then replace the decimal symbol, and finally chop according
        # to the threshold

        # when there is no float_format, we use str instead of '%g'
        # because str(0.0) = '0.0' while '%g' % 0.0 = '0'
        if float_format:

            def base_formatter(v):
                assert float_format is not None  # for mypy
                # error: "str" not callable
                # error: Unexpected keyword argument "value" for "__call__" of
                # "EngFormatter"
                return (
                    float_format(value=v)  # type: ignore[operator,call-arg]
                    if notna(v)
                    else self.na_rep
                )

        else:

            def base_formatter(v):
                return str(v) if notna(v) else self.na_rep

        if self.decimal != ".":

            def decimal_formatter(v):
                return base_formatter(v).replace(".", self.decimal, 1)

        else:
            decimal_formatter = base_formatter

        if threshold is None:
            return decimal_formatter

        def formatter(value):
            if notna(value):
                if abs(value) > threshold:
                    return decimal_formatter(value)
                else:
                    return decimal_formatter(0.0)
            else:
                return self.na_rep

        return formatter

    def get_result_as_array(self) -> np.ndarray:
        """
        Returns the float values converted into strings using
        the parameters given at initialisation, as a numpy array
        """

        def format_with_na_rep(values: ArrayLike, formatter: Callable, na_rep: str):
            mask = isna(values)
            formatted = np.array(
                [
                    formatter(val) if not m else na_rep
                    for val, m in zip(values.ravel(), mask.ravel())
                ]
            ).reshape(values.shape)
            return formatted

        def format_complex_with_na_rep(
            values: ArrayLike, formatter: Callable, na_rep: str
        ):
            real_values = np.real(values).ravel()  # type: ignore[arg-type]
            imag_values = np.imag(values).ravel()  # type: ignore[arg-type]
            real_mask, imag_mask = isna(real_values), isna(imag_values)
            formatted_lst = []
            for val, real_val, imag_val, re_isna, im_isna in zip(
                values.ravel(),
                real_values,
                imag_values,
                real_mask,
                imag_mask,
            ):
                if not re_isna and not im_isna:
                    formatted_lst.append(formatter(val))
                elif not re_isna:  # xxx+nanj
                    formatted_lst.append(f"{formatter(real_val)}+{na_rep}j")
                elif not im_isna:  # nan[+/-]xxxj
                    # The imaginary part may either start with a "-" or a space
                    imag_formatted = formatter(imag_val).strip()
                    if imag_formatted.startswith("-"):
                        formatted_lst.append(f"{na_rep}{imag_formatted}j")
                    else:
                        formatted_lst.append(f"{na_rep}+{imag_formatted}j")
                else:  # nan+nanj
                    formatted_lst.append(f"{na_rep}+{na_rep}j")
            return np.array(formatted_lst).reshape(values.shape)

        if self.formatter is not None:
            return format_with_na_rep(self.values, self.formatter, self.na_rep)

        if self.fixed_width:
            threshold = get_option("display.chop_threshold")
        else:
            threshold = None

        # if we have a fixed_width, we'll need to try different float_format
        def format_values_with(float_format):
            formatter = self._value_formatter(float_format, threshold)

            # default formatter leaves a space to the left when formatting
            # floats, must be consistent for left-justifying NaNs (GH #25061)
            na_rep = " " + self.na_rep if self.justify == "left" else self.na_rep

            # different formatting strategies for complex and non-complex data
            # need to distinguish complex and float NaNs (GH #53762)
            values = self.values
            is_complex = is_complex_dtype(values)

            # separate the wheat from the chaff
            if is_complex:
                values = format_complex_with_na_rep(values, formatter, na_rep)
            else:
                values = format_with_na_rep(values, formatter, na_rep)

            if self.fixed_width:
                if is_complex:
                    result = _trim_zeros_complex(values, self.decimal)
                else:
                    result = _trim_zeros_float(values, self.decimal)
                return np.asarray(result, dtype="object")

            return values

        # There is a special default string when we are fixed-width
        # The default is otherwise to use str instead of a formatting string
        float_format: FloatFormatType | None
        if self.float_format is None:
            if self.fixed_width:
                if self.leading_space is True:
                    fmt_str = "{value: .{digits:d}f}"
                else:
                    fmt_str = "{value:.{digits:d}f}"
                float_format = partial(fmt_str.format, digits=self.digits)
            else:
                float_format = self.float_format
        else:
            float_format = lambda value: self.float_format % value

        formatted_values = format_values_with(float_format)

        if not self.fixed_width:
            return formatted_values

        # we need do convert to engineering format if some values are too small
        # and would appear as 0, or if some values are too big and take too
        # much space

        if len(formatted_values) > 0:
            maxlen = max(len(x) for x in formatted_values)
            too_long = maxlen > self.digits + 6
        else:
            too_long = False

        abs_vals = np.abs(self.values)
        # this is pretty arbitrary for now
        # large values: more that 8 characters including decimal symbol
        # and first digit, hence > 1e6
        has_large_values = (abs_vals > 1e6).any()
        has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()

        if has_small_values or (too_long and has_large_values):
            if self.leading_space is True:
                fmt_str = "{value: .{digits:d}e}"
            else:
                fmt_str = "{value:.{digits:d}e}"
            float_format = partial(fmt_str.format, digits=self.digits)
            formatted_values = format_values_with(float_format)

        return formatted_values

    def _format_strings(self) -> list[str]:
        return list(self.get_result_as_array())


class _IntArrayFormatter(_GenericArrayFormatter):
    def _format_strings(self) -> list[str]:
        if self.leading_space is False:
            formatter_str = lambda x: f"{x:d}".format(x=x)
        else:
            formatter_str = lambda x: f"{x: d}".format(x=x)
        formatter = self.formatter or formatter_str
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


class _Datetime64Formatter(_GenericArrayFormatter):
    values: DatetimeArray

    def __init__(
        self,
        values: DatetimeArray,
        nat_rep: str = "NaT",
        date_format: None = None,
        **kwargs,
    ) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.date_format = date_format

    def _format_strings(self) -> list[str]:
        """we by definition have DO NOT have a TZ"""
        values = self.values

        if self.formatter is not None:
            return [self.formatter(x) for x in values]

        fmt_values = values._format_native_types(
            na_rep=self.nat_rep, date_format=self.date_format
        )
        return fmt_values.tolist()


class _ExtensionArrayFormatter(_GenericArrayFormatter):
    values: ExtensionArray

    def _format_strings(self) -> list[str]:
        values = self.values

        formatter = self.formatter
        fallback_formatter = None
        if formatter is None:
            fallback_formatter = values._formatter(boxed=True)

        if isinstance(values, Categorical):
            # Categorical is special for now, so that we can preserve tzinfo
            array = values._internal_get_values()
        else:
            array = np.asarray(values, dtype=object)

        fmt_values = format_array(
            array,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            digits=self.digits,
            space=self.space,
            justify=self.justify,
            decimal=self.decimal,
            leading_space=self.leading_space,
            quoting=self.quoting,
            fallback_formatter=fallback_formatter,
        )
        return fmt_values


def format_percentiles(
    percentiles: (np.ndarray | Sequence[float]),
) -> list[str]:
    """
    Outputs rounded and formatted percentiles.

    Parameters
    ----------
    percentiles : list-like, containing floats from interval [0,1]

    Returns
    -------
    formatted : list of strings

    Notes
    -----
    Rounding precision is chosen so that: (1) if any two elements of
    ``percentiles`` differ, they remain different after rounding
    (2) no entry is *rounded* to 0% or 100%.
    Any non-integer is always rounded to at least 1 decimal place.

    Examples
    --------
    Keeps all entries different after rounding:

    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']

    No element is rounded to 0% or 100% (unless already equal to it).
    Duplicates are allowed:

    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
    """
    percentiles = np.asarray(percentiles)

    # It checks for np.nan as well
    if (
        not is_numeric_dtype(percentiles)
        or not np.all(percentiles >= 0)
        or not np.all(percentiles <= 1)
    ):
        raise ValueError("percentiles should all be in the interval [0,1]")

    percentiles = 100 * percentiles
    prec = get_precision(percentiles)
    percentiles_round_type = percentiles.round(prec).astype(int)

    int_idx = np.isclose(percentiles_round_type, percentiles)

    if np.all(int_idx):
        out = percentiles_round_type.astype(str)
        return [i + "%" for i in out]

    unique_pcts = np.unique(percentiles)
    prec = get_precision(unique_pcts)
    out = np.empty_like(percentiles, dtype=object)
    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)

    out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
    return [i + "%" for i in out]


def get_precision(array: np.ndarray | Sequence[float]) -> int:
    to_begin = array[0] if array[0] > 0 else None
    to_end = 100 - array[-1] if array[-1] < 100 else None
    diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
    diff = abs(diff)
    prec = -np.floor(np.log10(np.min(diff))).astype(int)
    prec = max(1, prec)
    return prec


def _format_datetime64(x: NaTType | Timestamp, nat_rep: str = "NaT") -> str:
    if x is NaT:
        return nat_rep

    # Timestamp.__str__ falls back to datetime.datetime.__str__ = isoformat(sep=' ')
    # so it already uses string formatting rather than strftime (faster).
    return str(x)


def _format_datetime64_dateonly(
    x: NaTType | Timestamp,
    nat_rep: str = "NaT",
    date_format: str | None = None,
) -> str:
    if isinstance(x, NaTType):
        return nat_rep

    if date_format:
        return x.strftime(date_format)
    else:
        # Timestamp._date_repr relies on string formatting (faster than strftime)
        return x._date_repr


def get_format_datetime64(
    is_dates_only: bool, nat_rep: str = "NaT", date_format: str | None = None
) -> Callable:
    """Return a formatter callable taking a datetime64 as input and providing
    a string as output"""

    if is_dates_only:
        return lambda x: _format_datetime64_dateonly(
            x, nat_rep=nat_rep, date_format=date_format
        )
    else:
        return lambda x: _format_datetime64(x, nat_rep=nat_rep)


class _Datetime64TZFormatter(_Datetime64Formatter):
    values: DatetimeArray

    def _format_strings(self) -> list[str]:
        """we by definition have a TZ"""
        ido = self.values._is_dates_only
        values = self.values.astype(object)
        formatter = self.formatter or get_format_datetime64(
            ido, date_format=self.date_format
        )
        fmt_values = [formatter(x) for x in values]

        return fmt_values


class _Timedelta64Formatter(_GenericArrayFormatter):
    values: TimedeltaArray

    def __init__(
        self,
        values: TimedeltaArray,
        nat_rep: str = "NaT",
        **kwargs,
    ) -> None:
        # TODO: nat_rep is never passed, na_rep is.
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep

    def _format_strings(self) -> list[str]:
        formatter = self.formatter or get_format_timedelta64(
            self.values, nat_rep=self.nat_rep, box=False
        )
        return [formatter(x) for x in self.values]


def get_format_timedelta64(
    values: TimedeltaArray,
    nat_rep: str | float = "NaT",
    box: bool = False,
) -> Callable:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
    even_days = values._is_dates_only

    if even_days:
        format = None
    else:
        format = "long"

    def _formatter(x):
        if x is None or (is_scalar(x) and isna(x)):
            return nat_rep

        if not isinstance(x, Timedelta):
            x = Timedelta(x)

        # Timedelta._repr_base uses string formatting (faster than strftime)
        result = x._repr_base(format=format)
        if box:
            result = f"'{result}'"
        return result

    return _formatter


def _make_fixed_width(
    strings: list[str],
    justify: str = "right",
    minimum: int | None = None,
    adj: printing._TextAdjustment | None = None,
) -> list[str]:
    if len(strings) == 0 or justify == "all":
        return strings

    if adj is None:
        adjustment = printing.get_adjustment()
    else:
        adjustment = adj

    max_len = max(adjustment.len(x) for x in strings)

    if minimum is not None:
        max_len = max(minimum, max_len)

    conf_max = get_option("display.max_colwidth")
    if conf_max is not None and max_len > conf_max:
        max_len = conf_max

    def just(x: str) -> str:
        if conf_max is not None:
            if (conf_max > 3) & (adjustment.len(x) > max_len):
                x = x[: max_len - 3] + "..."
        return x

    strings = [just(x) for x in strings]
    result = adjustment.justify(strings, max_len, mode=justify)
    return result


def _trim_zeros_complex(str_complexes: ArrayLike, decimal: str = ".") -> list[str]:
    """
    Separates the real and imaginary parts from the complex number, and
    executes the _trim_zeros_float method on each of those.
    """
    real_part, imag_part = [], []
    for x in str_complexes:
        # Complex numbers are represented as "(-)xxx(+/-)xxxj"
        # The split will give [{"", "-"}, "xxx", "+/-", "xxx", "j", ""]
        # Therefore, the imaginary part is the 4th and 3rd last elements,
        # and the real part is everything before the imaginary part
        trimmed = re.split(r"([j+-])", x)
        real_part.append("".join(trimmed[:-4]))
        imag_part.append("".join(trimmed[-4:-2]))

    # We want to align the lengths of the real and imaginary parts of each complex
    # number, as well as the lengths the real (resp. complex) parts of all numbers
    # in the array
    n = len(str_complexes)
    padded_parts = _trim_zeros_float(real_part + imag_part, decimal)
    if len(padded_parts) == 0:
        return []
    padded_length = max(len(part) for part in padded_parts) - 1
    padded = [
        real_pt  # real part, possibly NaN
        + imag_pt[0]  # +/-
        + f"{imag_pt[1:]:>{padded_length}}"  # complex part (no sign), possibly nan
        + "j"
        for real_pt, imag_pt in zip(padded_parts[:n], padded_parts[n:])
    ]
    return padded


def _trim_zeros_single_float(str_float: str) -> str:
    """
    Trims trailing zeros after a decimal point,
    leaving just one if necessary.
    """
    str_float = str_float.rstrip("0")
    if str_float.endswith("."):
        str_float += "0"

    return str_float


def _trim_zeros_float(
    str_floats: ArrayLike | list[str], decimal: str = "."
) -> list[str]:
    """
    Trims the maximum number of trailing zeros equally from
    all numbers containing decimals, leaving just one if
    necessary.
    """
    trimmed = str_floats
    number_regex = re.compile(rf"^\s*[\+-]?[0-9]+\{decimal}[0-9]*$")

    def is_number_with_decimal(x) -> bool:
        return re.match(number_regex, x) is not None

    def should_trim(values: ArrayLike | list[str]) -> bool:
        """
        Determine if an array of strings should be trimmed.

        Returns True if all numbers containing decimals (defined by the
        above regular expression) within the array end in a zero, otherwise
        returns False.
        """
        numbers = [x for x in values if is_number_with_decimal(x)]
        return len(numbers) > 0 and all(x.endswith("0") for x in numbers)

    while should_trim(trimmed):
        trimmed = [x[:-1] if is_number_with_decimal(x) else x for x in trimmed]

    # leave one 0 after the decimal points if need be.
    result = [
        x + "0" if is_number_with_decimal(x) and x.endswith(decimal) else x
        for x in trimmed
    ]
    return result


def _has_names(index: Index) -> bool:
    if isinstance(index, MultiIndex):
        return com.any_not_none(*index.names)
    else:
        return index.name is not None


class EngFormatter:
    """
    Formats float values according to engineering format.

    Based on matplotlib.ticker.EngFormatter
    """

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "u",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }

    def __init__(
        self, accuracy: int | None = None, use_eng_prefix: bool = False
    ) -> None:
        self.accuracy = accuracy
        self.use_eng_prefix = use_eng_prefix

    def __call__(self, num: float) -> str:
        """
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:
        >>> format_eng = EngFormatter(accuracy=0, use_eng_prefix=True)
        >>> format_eng(0)
        ' 0'
        >>> format_eng = EngFormatter(accuracy=1, use_eng_prefix=True)
        >>> format_eng(1_000_000)
        ' 1.0M'
        >>> format_eng = EngFormatter(accuracy=2, use_eng_prefix=False)
        >>> format_eng("-1e-6")
        '-1.00E-06'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        """
        dnum = Decimal(str(num))

        if Decimal.is_nan(dnum):
            return "NaN"

        if Decimal.is_infinite(dnum):
            return "inf"

        sign = 1

        if dnum < 0:  # pragma: no cover
            sign = -1
            dnum = -dnum

        if dnum != 0:
            pow10 = Decimal(int(math.floor(dnum.log10() / 3) * 3))
        else:
            pow10 = Decimal(0)

        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
        int_pow10 = int(pow10)

        if self.use_eng_prefix:
            prefix = self.ENG_PREFIXES[int_pow10]
        elif int_pow10 < 0:
            prefix = f"E-{-int_pow10:02d}"
        else:
            prefix = f"E+{int_pow10:02d}"

        mant = sign * dnum / (10**pow10)

        if self.accuracy is None:  # pragma: no cover
            format_str = "{mant: g}{prefix}"
        else:
            format_str = f"{{mant: .{self.accuracy:d}f}}{{prefix}}"

        formatted = format_str.format(mant=mant, prefix=prefix)

        return formatted


def set_eng_float_format(accuracy: int = 3, use_eng_prefix: bool = False) -> None:
    """
    Format float representation in DataFrame with SI notation.

    Parameters
    ----------
    accuracy : int, default 3
        Number of decimal digits after the floating point.
    use_eng_prefix : bool, default False
        Whether to represent a value with SI prefixes.

    Returns
    -------
    None

    Examples
    --------
    >>> df = pd.DataFrame([1e-9, 1e-3, 1, 1e3, 1e6])
    >>> df
                  0
    0  1.000000e-09
    1  1.000000e-03
    2  1.000000e+00
    3  1.000000e+03
    4  1.000000e+06

    >>> pd.set_eng_float_format(accuracy=1)
    >>> df
             0
    0  1.0E-09
    1  1.0E-03
    2  1.0E+00
    3  1.0E+03
    4  1.0E+06

    >>> pd.set_eng_float_format(use_eng_prefix=True)
    >>> df
            0
    0  1.000n
    1  1.000m
    2   1.000
    3  1.000k
    4  1.000M

    >>> pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
    >>> df
          0
    0  1.0n
    1  1.0m
    2   1.0
    3  1.0k
    4  1.0M

    >>> pd.set_option("display.float_format", None)  # unset option
    """
    set_option("display.float_format", EngFormatter(accuracy, use_eng_prefix))


def get_level_lengths(
    levels: Any, sentinel: bool | object | str = ""
) -> list[dict[int, int]]:
    """
    For each index in each level the function returns lengths of indexes.

    Parameters
    ----------
    levels : list of lists
        List of values on for level.
    sentinel : string, optional
        Value which states that no new index starts on there.

    Returns
    -------
    Returns list of maps. For each level returns map of indexes (key is index
    in row and value is length of index).
    """
    if len(levels) == 0:
        return []

    control = [True] * len(levels[0])

    result = []
    for level in levels:
        last_index = 0

        lengths = {}
        for i, key in enumerate(level):
            if control[i] and key == sentinel:
                pass
            else:
                control[i] = False
                lengths[last_index] = i - last_index
                last_index = i

        lengths[last_index] = len(level) - last_index

        result.append(lengths)

    return result


def buffer_put_lines(buf: WriteBuffer[str], lines: list[str]) -> None:
    """
    Appends lines to a buffer.

    Parameters
    ----------
    buf
        The buffer to write to
    lines
        The lines to append.
    """
    if any(isinstance(x, str) for x in lines):
        lines = [str(x) for x in lines]
    buf.write("\n".join(lines))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\apply.py ===
from __future__ import annotations

import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import option_context

from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
    AggFuncType,
    AggFuncTypeBase,
    AggFuncTypeDict,
    AggObjType,
    Axis,
    AxisInt,
    NDFrameT,
    npt,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
    is_dict_like,
    is_extension_array_dtype,
    is_list_like,
    is_numeric_dtype,
    is_sequence,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCNDFrame,
    ABCSeries,
)

from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Hashable,
        Iterable,
        MutableMapping,
        Sequence,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby import GroupBy
    from pandas.core.resample import Resampler
    from pandas.core.window.rolling import BaseWindow


ResType = dict[int, Any]


def frame_apply(
    obj: DataFrame,
    func: AggFuncType,
    axis: Axis = 0,
    raw: bool = False,
    result_type: str | None = None,
    by_row: Literal[False, "compat"] = "compat",
    engine: str = "python",
    engine_kwargs: dict[str, bool] | None = None,
    args=None,
    kwargs=None,
) -> FrameApply:
    """construct and return a row or column based frame apply object"""
    axis = obj._get_axis_number(axis)
    klass: type[FrameApply]
    if axis == 0:
        klass = FrameRowApply
    elif axis == 1:
        klass = FrameColumnApply

    _, func, _, _ = reconstruct_func(func, **kwargs)
    assert func is not None

    return klass(
        obj,
        func,
        raw=raw,
        result_type=result_type,
        by_row=by_row,
        engine=engine,
        engine_kwargs=engine_kwargs,
        args=args,
        kwargs=kwargs,
    )


class Apply(metaclass=abc.ABCMeta):
    axis: AxisInt

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: str | None,
        *,
        by_row: Literal[False, "compat", "_compat"] = "compat",
        engine: str = "python",
        engine_kwargs: dict[str, bool] | None = None,
        args,
        kwargs,
    ) -> None:
        self.obj = obj
        self.raw = raw

        assert by_row is False or by_row in ["compat", "_compat"]
        self.by_row = by_row

        self.args = args or ()
        self.kwargs = kwargs or {}

        self.engine = engine
        self.engine_kwargs = {} if engine_kwargs is None else engine_kwargs

        if result_type not in [None, "reduce", "broadcast", "expand"]:
            raise ValueError(
                "invalid value for result_type, must be one "
                "of {None, 'reduce', 'broadcast', 'expand'}"
            )

        self.result_type = result_type

        self.func = func

    @abc.abstractmethod
    def apply(self) -> DataFrame | Series:
        pass

    @abc.abstractmethod
    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        pass

    @abc.abstractmethod
    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        pass

    def agg(self) -> DataFrame | Series | None:
        """
        Provide an implementation for the aggregators.

        Returns
        -------
        Result of aggregation, or None if agg cannot be performed by
        this method.
        """
        obj = self.obj
        func = self.func
        args = self.args
        kwargs = self.kwargs

        if isinstance(func, str):
            return self.apply_str()

        if is_dict_like(func):
            return self.agg_dict_like()
        elif is_list_like(func):
            # we require a list, but not a 'str'
            return self.agg_list_like()

        if callable(func):
            f = com.get_cython_func(func)
            if f and not args and not kwargs:
                warn_alias_replacement(obj, func, f)
                return getattr(obj, f)()

        # caller can react
        return None

    def transform(self) -> DataFrame | Series:
        """
        Transform a DataFrame or Series.

        Returns
        -------
        DataFrame or Series
            Result of applying ``func`` along the given axis of the
            Series or DataFrame.

        Raises
        ------
        ValueError
            If the transform function fails or does not transform.
        """
        obj = self.obj
        func = self.func
        axis = self.axis
        args = self.args
        kwargs = self.kwargs

        is_series = obj.ndim == 1

        if obj._get_axis_number(axis) == 1:
            assert not is_series
            return obj.T.transform(func, 0, *args, **kwargs).T

        if is_list_like(func) and not is_dict_like(func):
            func = cast(list[AggFuncTypeBase], func)
            # Convert func equivalent dict
            if is_series:
                func = {com.get_callable_name(v) or v: v for v in func}
            else:
                func = {col: func for col in obj}

        if is_dict_like(func):
            func = cast(AggFuncTypeDict, func)
            return self.transform_dict_like(func)

        # func is either str or callable
        func = cast(AggFuncTypeBase, func)
        try:
            result = self.transform_str_or_callable(func)
        except TypeError:
            raise
        except Exception as err:
            raise ValueError("Transform function failed") from err

        # Functions that transform may return empty Series/DataFrame
        # when the dtype is not appropriate
        if (
            isinstance(result, (ABCSeries, ABCDataFrame))
            and result.empty
            and not obj.empty
        ):
            raise ValueError("Transform function failed")
        # error: Argument 1 to "__get__" of "AxisProperty" has incompatible type
        # "Union[Series, DataFrame, GroupBy[Any], SeriesGroupBy,
        # DataFrameGroupBy, BaseWindow, Resampler]"; expected "Union[DataFrame,
        # Series]"
        if not isinstance(result, (ABCSeries, ABCDataFrame)) or not result.index.equals(
            obj.index  # type: ignore[arg-type]
        ):
            raise ValueError("Function did not transform")

        return result

    def transform_dict_like(self, func) -> DataFrame:
        """
        Compute transform in the case of a dict-like func
        """
        from pandas.core.reshape.concat import concat

        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        # transform is currently only for Series/DataFrame
        assert isinstance(obj, ABCNDFrame)

        if len(func) == 0:
            raise ValueError("No transform functions were provided")

        func = self.normalize_dictlike_arg("transform", obj, func)

        results: dict[Hashable, DataFrame | Series] = {}
        for name, how in func.items():
            colg = obj._gotitem(name, ndim=1)
            results[name] = colg.transform(how, 0, *args, **kwargs)
        return concat(results, axis=1)

    def transform_str_or_callable(self, func) -> DataFrame | Series:
        """
        Compute transform in the case of a string or callable func
        """
        obj = self.obj
        args = self.args
        kwargs = self.kwargs

        if isinstance(func, str):
            return self._apply_str(obj, func, *args, **kwargs)

        if not args and not kwargs:
            f = com.get_cython_func(func)
            if f:
                warn_alias_replacement(obj, func, f)
                return getattr(obj, f)()

        # Two possible ways to use a UDF - apply or call directly
        try:
            return obj.apply(func, args=args, **kwargs)
        except Exception:
            return func(obj, *args, **kwargs)

    def agg_list_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_list_like(op_name="agg")

    def compute_list_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable] | Index, list[Any]]:
        """
        Compute agg/apply results for like-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[Hashable] or Index
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python objects.
        """
        func = cast(list[AggFuncTypeBase], self.func)
        obj = self.obj

        results = []
        keys = []

        # degenerate case
        if selected_obj.ndim == 1:
            for a in func:
                colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(a, *args, **kwargs)
                results.append(new_res)

                # make sure we find a good name
                name = com.get_callable_name(a) or a
                keys.append(name)

        else:
            indices = []
            for index, col in enumerate(selected_obj):
                colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                new_res = getattr(colg, op_name)(func, *args, **kwargs)
                results.append(new_res)
                indices.append(index)
            # error: Incompatible types in assignment (expression has type "Any |
            # Index", variable has type "list[Any | Callable[..., Any] | str]")
            keys = selected_obj.columns.take(indices)  # type: ignore[assignment]

        return keys, results

    def wrap_results_list_like(
        self, keys: Iterable[Hashable], results: list[Series | DataFrame]
    ):
        from pandas.core.reshape.concat import concat

        obj = self.obj

        try:
            return concat(results, keys=keys, axis=1, sort=False)
        except TypeError as err:
            # we are concatting non-NDFrame objects,
            # e.g. a list of scalars
            from pandas import Series

            result = Series(results, index=keys, name=obj.name)
            if is_nested_object(result):
                raise ValueError(
                    "cannot combine transform and aggregation operations"
                ) from err
            return result

    def agg_dict_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_dict_like(op_name="agg")

    def compute_dict_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        selection: Hashable | Sequence[Hashable],
        kwargs: dict[str, Any],
    ) -> tuple[list[Hashable], list[Any]]:
        """
        Compute agg/apply results for dict-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        selection : hashable or sequence of hashables
            Used by GroupBy, Window, and Resample if selection is applied to the object.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[hashable]
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python object.
        """
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        obj = self.obj
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
        func = cast(AggFuncTypeDict, self.func)
        func = self.normalize_dictlike_arg(op_name, selected_obj, func)

        is_non_unique_col = (
            selected_obj.ndim == 2
            and selected_obj.columns.nunique() < len(selected_obj.columns)
        )

        if selected_obj.ndim == 1:
            # key only used for output
            colg = obj._gotitem(selection, ndim=1)
            results = [getattr(colg, op_name)(how, **kwargs) for _, how in func.items()]
            keys = list(func.keys())
        elif not is_groupby and is_non_unique_col:
            # key used for column selection and output
            # GH#51099
            results = []
            keys = []
            for key, how in func.items():
                indices = selected_obj.columns.get_indexer_for([key])
                labels = selected_obj.columns.take(indices)
                label_to_indices = defaultdict(list)
                for index, label in zip(indices, labels):
                    label_to_indices[label].append(index)

                key_data = [
                    getattr(selected_obj._ixs(indice, axis=1), op_name)(how, **kwargs)
                    for label, indices in label_to_indices.items()
                    for indice in indices
                ]

                keys += [key] * len(key_data)
                results += key_data
        else:
            # key used for column selection and output
            results = [
                getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs)
                for key, how in func.items()
            ]
            keys = list(func.keys())

        return keys, results

    def wrap_results_dict_like(
        self,
        selected_obj: Series | DataFrame,
        result_index: list[Hashable],
        result_data: list,
    ):
        from pandas import Index
        from pandas.core.reshape.concat import concat

        obj = self.obj

        # Avoid making two isinstance calls in all and any below
        is_ndframe = [isinstance(r, ABCNDFrame) for r in result_data]

        if all(is_ndframe):
            results = dict(zip(result_index, result_data))
            keys_to_use: Iterable[Hashable]
            keys_to_use = [k for k in result_index if not results[k].empty]
            # Have to check, if at least one DataFrame is not empty.
            keys_to_use = keys_to_use if keys_to_use != [] else result_index
            if selected_obj.ndim == 2:
                # keys are columns, so we can preserve names
                ktu = Index(keys_to_use)
                ktu._set_names(selected_obj.columns.names)
                keys_to_use = ktu

            axis: AxisInt = 0 if isinstance(obj, ABCSeries) else 1
            result = concat(
                {k: results[k] for k in keys_to_use},
                axis=axis,
                keys=keys_to_use,
            )
        elif any(is_ndframe):
            # There is a mix of NDFrames and scalars
            raise ValueError(
                "cannot perform both aggregation "
                "and transformation operations "
                "simultaneously"
            )
        else:
            from pandas import Series

            # we have a list of scalars
            # GH 36212 use name only if obj is a series
            if obj.ndim == 1:
                obj = cast("Series", obj)
                name = obj.name
            else:
                name = None

            result = Series(result_data, index=result_index, name=name)

        return result

    def apply_str(self) -> DataFrame | Series:
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
        """
        # Caller is responsible for checking isinstance(self.f, str)
        func = cast(str, self.func)

        obj = self.obj

        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        # Support for `frame.transform('method')`
        # Some methods (shift, etc.) require the axis argument, others
        # don't, so inspect and insert if necessary.
        method = getattr(obj, func, None)
        if callable(method):
            sig = inspect.getfullargspec(method)
            arg_names = (*sig.args, *sig.kwonlyargs)
            if self.axis != 0 and (
                "axis" not in arg_names or func in ("corrwith", "skew")
            ):
                raise ValueError(f"Operation {func} does not support axis=1")
            if "axis" in arg_names:
                if isinstance(obj, (SeriesGroupBy, DataFrameGroupBy)):
                    # Try to avoid FutureWarning for deprecated axis keyword;
                    # If self.axis matches the axis we would get by not passing
                    #  axis, we safely exclude the keyword.

                    default_axis = 0
                    if func in ["idxmax", "idxmin"]:
                        # DataFrameGroupBy.idxmax, idxmin axis defaults to self.axis,
                        # whereas other axis keywords default to 0
                        default_axis = self.obj.axis

                    if default_axis != self.axis:
                        self.kwargs["axis"] = self.axis
                else:
                    self.kwargs["axis"] = self.axis
        return self._apply_str(obj, func, *self.args, **self.kwargs)

    def apply_list_or_dict_like(self) -> DataFrame | Series:
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.func is a list-like or dict-like, None otherwise.
        """

        if self.engine == "numba":
            raise NotImplementedError(
                "The 'numba' engine doesn't support list-like/"
                "dict likes of callables yet."
            )

        if self.axis == 1 and isinstance(self.obj, ABCDataFrame):
            return self.obj.T.apply(self.func, 0, args=self.args, **self.kwargs).T

        func = self.func
        kwargs = self.kwargs

        if is_dict_like(func):
            result = self.agg_or_apply_dict_like(op_name="apply")
        else:
            result = self.agg_or_apply_list_like(op_name="apply")

        result = reconstruct_and_relabel_result(result, func, **kwargs)

        return result

    def normalize_dictlike_arg(
        self, how: str, obj: DataFrame | Series, func: AggFuncTypeDict
    ) -> AggFuncTypeDict:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        assert how in ("apply", "agg", "transform")

        # Can't use func.values(); wouldn't work for a Series
        if (
            how == "agg"
            and isinstance(obj, ABCSeries)
            and any(is_list_like(v) for _, v in func.items())
        ) or (any(is_dict_like(v) for _, v in func.items())):
            # GH 15931 - deprecation of renaming keys
            raise SpecificationError("nested renamer is not supported")

        if obj.ndim != 1:
            # Check for missing columns on a frame
            from pandas import Index

            cols = Index(list(func.keys())).difference(obj.columns, sort=True)
            if len(cols) > 0:
                raise KeyError(f"Column(s) {list(cols)} do not exist")

        aggregator_types = (list, tuple, dict)

        # if we have a dict of any non-scalars
        # eg. {'A' : ['mean']}, normalize all to
        # be list-likes
        # Cannot use func.values() because arg may be a Series
        if any(isinstance(x, aggregator_types) for _, x in func.items()):
            new_func: AggFuncTypeDict = {}
            for k, v in func.items():
                if not isinstance(v, aggregator_types):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            func = new_func
        return func

    def _apply_str(self, obj, func: str, *args, **kwargs):
        """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on obj
        - try to find a numpy function
        - raise
        """
        assert isinstance(func, str)

        if hasattr(obj, func):
            f = getattr(obj, func)
            if callable(f):
                return f(*args, **kwargs)

            # people may aggregate on a non-callable attribute
            # but don't let them think they can pass args to it
            assert len(args) == 0
            assert len([kwarg for kwarg in kwargs if kwarg not in ["axis"]]) == 0
            return f
        elif hasattr(np, func) and hasattr(obj, "__array__"):
            # in particular exclude Window
            f = getattr(np, func)
            return f(obj, *args, **kwargs)
        else:
            msg = f"'{func}' is not a valid function for '{type(obj).__name__}' object"
            raise AttributeError(msg)


class NDFrameApply(Apply):
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """

    obj: DataFrame | Series

    @property
    def index(self) -> Index:
        return self.obj.index

    @property
    def agg_axis(self) -> Index:
        return self.obj._get_agg_axis(self.axis)

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        obj = self.obj
        kwargs = self.kwargs

        if op_name == "apply":
            if isinstance(self, FrameApply):
                by_row = self.by_row

            elif isinstance(self, SeriesApply):
                by_row = "_compat" if self.by_row else False
            else:
                by_row = False
            kwargs = {**kwargs, "by_row": by_row}

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        keys, results = self.compute_list_like(op_name, obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        assert op_name in ["agg", "apply"]
        obj = self.obj

        kwargs = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        selection = None
        result_index, result_data = self.compute_dict_like(
            op_name, obj, selection, kwargs
        )
        result = self.wrap_results_dict_like(obj, result_index, result_data)
        return result


class FrameApply(NDFrameApply):
    obj: DataFrame

    def __init__(
        self,
        obj: AggObjType,
        func: AggFuncType,
        raw: bool,
        result_type: str | None,
        *,
        by_row: Literal[False, "compat"] = False,
        engine: str = "python",
        engine_kwargs: dict[str, bool] | None = None,
        args,
        kwargs,
    ) -> None:
        if by_row is not False and by_row != "compat":
            raise ValueError(f"by_row={by_row} not allowed")
        super().__init__(
            obj,
            func,
            raw,
            result_type,
            by_row=by_row,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    # ---------------------------------------------------------------
    # Abstract Methods

    @property
    @abc.abstractmethod
    def result_index(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def result_columns(self) -> Index:
        pass

    @property
    @abc.abstractmethod
    def series_generator(self) -> Generator[Series, None, None]:
        pass

    @staticmethod
    @functools.cache
    @abc.abstractmethod
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        pass

    @abc.abstractmethod
    def apply_with_numba(self):
        pass

    def validate_values_for_numba(self):
        # Validate column dtyps all OK
        for colname, dtype in self.obj.dtypes.items():
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Column {colname} must have a numeric dtype. "
                    f"Found '{dtype}' instead"
                )
            if is_extension_array_dtype(dtype):
                raise ValueError(
                    f"Column {colname} is backed by an extension array, "
                    f"which is not supported by the numba engine."
                )

    @abc.abstractmethod
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        pass

    # ---------------------------------------------------------------

    @property
    def res_columns(self) -> Index:
        return self.result_columns

    @property
    def columns(self) -> Index:
        return self.obj.columns

    @cache_readonly
    def values(self):
        return self.obj.values

    def apply(self) -> DataFrame | Series:
        """compute the results"""

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support lists of callables yet"
                )
            return self.apply_list_or_dict_like()

        # all empty
        if len(self.columns) == 0 and len(self.index) == 0:
            return self.apply_empty_result()

        # string dispatch
        if isinstance(self.func, str):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support using "
                    "a string as the callable function"
                )
            return self.apply_str()

        # ufunc
        elif isinstance(self.func, np.ufunc):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support "
                    "using a numpy ufunc as the callable function"
                )
            with np.errstate(all="ignore"):
                results = self.obj._mgr.apply("apply", func=self.func)
            # _constructor will retain self.index and self.columns
            return self.obj._constructor_from_mgr(results, axes=results.axes)

        # broadcasting
        if self.result_type == "broadcast":
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support result_type='broadcast'"
                )
            return self.apply_broadcast(self.obj)

        # one axis empty
        elif not all(self.obj.shape):
            return self.apply_empty_result()

        # raw
        elif self.raw:
            return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)

        return self.apply_standard()

    def agg(self):
        obj = self.obj
        axis = self.axis

        # TODO: Avoid having to change state
        self.obj = self.obj if self.axis == 0 else self.obj.T
        self.axis = 0

        result = None
        try:
            result = super().agg()
        finally:
            self.obj = obj
            self.axis = axis

        if axis == 1:
            result = result.T if result is not None else result

        if result is None:
            result = self.obj.apply(self.func, axis, args=self.args, **self.kwargs)

        return result

    def apply_empty_result(self):
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
        assert callable(self.func)

        # we are not asked to reduce or infer reduction
        # so just return a copy of the existing object
        if self.result_type not in ["reduce", None]:
            return self.obj.copy()

        # we may need to infer
        should_reduce = self.result_type == "reduce"

        from pandas import Series

        if not should_reduce:
            try:
                if self.axis == 0:
                    r = self.func(
                        Series([], dtype=np.float64), *self.args, **self.kwargs
                    )
                else:
                    r = self.func(
                        Series(index=self.columns, dtype=np.float64),
                        *self.args,
                        **self.kwargs,
                    )
            except Exception:
                pass
            else:
                should_reduce = not isinstance(r, Series)

        if should_reduce:
            if len(self.agg_axis):
                r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
            else:
                r = np.nan

            return self.obj._constructor_sliced(r, index=self.agg_axis)
        else:
            return self.obj.copy()

    def apply_raw(self, engine="python", engine_kwargs=None):
        """apply to the values as a numpy array"""

        def wrap_function(func):
            """
            Wrap user supplied function to work around numpy issue.

            see https://github.com/numpy/numpy/issues/8352
            """

            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    result = np.array(result, dtype=object)
                return result

            return wrapper

        if engine == "numba":
            engine_kwargs = {} if engine_kwargs is None else engine_kwargs

            # error: Argument 1 to "__call__" of "_lru_cache_wrapper" has
            # incompatible type "Callable[..., Any] | str | list[Callable
            # [..., Any] | str] | dict[Hashable,Callable[..., Any] | str |
            # list[Callable[..., Any] | str]]"; expected "Hashable"
            nb_looper = generate_apply_looper(
                self.func, **engine_kwargs  # type: ignore[arg-type]
            )
            result = nb_looper(self.values, self.axis)
            # If we made the result 2-D, squeeze it back to 1-D
            result = np.squeeze(result)
        else:
            result = np.apply_along_axis(
                wrap_function(self.func),
                self.axis,
                self.values,
                *self.args,
                **self.kwargs,
            )

        # TODO: mixed type case
        if result.ndim == 2:
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        assert callable(self.func)

        result_values = np.empty_like(target.values)

        # axis which we want to compare compliance
        result_compare = target.shape[0]

        for i, col in enumerate(target.columns):
            res = self.func(target[col], *self.args, **self.kwargs)
            ares = np.asarray(res).ndim

            # must be a scalar or 1d
            if ares > 1:
                raise ValueError("too many dims to broadcast")
            if ares == 1:
                # must match return dim
                if result_compare != len(res):
                    raise ValueError("cannot broadcast result")

            result_values[:, i] = res

        # we *always* preserve the original index / columns
        result = self.obj._constructor(
            result_values, index=target.index, columns=target.columns
        )
        return result

    def apply_standard(self):
        if self.engine == "python":
            results, res_index = self.apply_series_generator()
        else:
            results, res_index = self.apply_series_numba()

        # wrap results
        return self.wrap_results(results, res_index)

    def apply_series_generator(self) -> tuple[ResType, Index]:
        assert callable(self.func)

        series_gen = self.series_generator
        res_index = self.result_index

        results = {}

        with option_context("mode.chained_assignment", None):
            for i, v in enumerate(series_gen):
                # ignore SettingWithCopy here in case the user mutates
                results[i] = self.func(v, *self.args, **self.kwargs)
                if isinstance(results[i], ABCSeries):
                    # If we have a view on v, we need to make a copy because
                    #  series_generator will swap out the underlying data
                    results[i] = results[i].copy(deep=False)

        return results, res_index

    def apply_series_numba(self):
        if self.engine_kwargs.get("parallel", False):
            raise NotImplementedError(
                "Parallel apply is not supported when raw=False and engine='numba'"
            )
        if not self.obj.index.is_unique or not self.columns.is_unique:
            raise NotImplementedError(
                "The index/columns must be unique when raw=False and engine='numba'"
            )
        self.validate_values_for_numba()
        results = self.apply_with_numba()
        return results, self.result_index

    def wrap_results(self, results: ResType, res_index: Index) -> DataFrame | Series:
        from pandas import Series

        # see if we can infer the results
        if len(results) > 0 and 0 in results and is_sequence(results[0]):
            return self.wrap_results_for_axis(results, res_index)

        # dict of scalars

        # the default dtype of an empty Series is `object`, but this
        # code can be hit by df.mean() where the result should have dtype
        # float64 even if it's an empty Series.
        constructor_sliced = self.obj._constructor_sliced
        if len(results) == 0 and constructor_sliced is Series:
            result = constructor_sliced(results, dtype=np.float64)
        else:
            result = constructor_sliced(results)
        result.index = res_index

        return result

    def apply_str(self) -> DataFrame | Series:
        # Caller is responsible for checking isinstance(self.func, str)
        # TODO: GH#39993 - Avoid special-casing by replacing with lambda
        if self.func == "size":
            # Special-cased because DataFrame.size returns a single scalar
            obj = self.obj
            value = obj.shape[self.axis]
            return obj._constructor_sliced(value, index=self.agg_axis)
        return super().apply_str()


class FrameRowApply(FrameApply):
    axis: AxisInt = 0

    @property
    def series_generator(self) -> Generator[Series, None, None]:
        return (self.obj._ixs(i, axis=1) for i in range(len(self.columns)))

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        numba = import_optional_dependency("numba")
        from pandas import Series

        # Import helper from extensions to cast string object -> np strings
        # Note: This also has the side effect of loading our numba extensions
        from pandas.core._numba.extensions import maybe_cast_str

        jitted_udf = numba.extending.register_jitable(func)

        # Currently the parallel argument doesn't get passed through here
        # (it's disabled) since the dicts in numba aren't thread-safe.
        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names, df_index):
            results = {}
            for j in range(values.shape[1]):
                # Create the series
                ser = Series(
                    values[:, j], index=df_index, name=maybe_cast_str(col_names[j])
                )
                results[j] = jitted_udf(ser)
            return results

        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        nb_func = self.generate_numba_apply_func(
            cast(Callable, self.func), **self.engine_kwargs
        )
        from pandas.core._numba.extensions import set_numba_data

        index = self.obj.index
        columns = self.obj.columns

        # Convert from numba dict to regular dict
        # Our isinstance checks in the df constructor don't pass for numbas typed dict
        with set_numba_data(index) as index, set_numba_data(columns) as columns:
            res = dict(nb_func(self.values, columns, index))
        return res

    @property
    def result_index(self) -> Index:
        return self.columns

    @property
    def result_columns(self) -> Index:
        return self.index

    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        """return the results for the rows"""

        if self.result_type == "reduce":
            # e.g. test_apply_dict GH#8735
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res

        elif self.result_type is None and all(
            isinstance(x, dict) for x in results.values()
        ):
            # Our operation was a to_dict op e.g.
            #  test_apply_dict GH#8735, test_apply_reduce_to_dict GH#25196 #37544
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res

        try:
            result = self.obj._constructor(data=results)
        except ValueError as err:
            if "All arrays must be of the same length" in str(err):
                # e.g. result = [[2, 3], [1.5], ['foo', 'bar']]
                #  see test_agg_listlike_result GH#29587
                res = self.obj._constructor_sliced(results)
                res.index = res_index
                return res
            else:
                raise

        if not isinstance(results[0], ABCSeries):
            if len(result.index) == len(self.res_columns):
                result.index = self.res_columns

        if len(result.columns) == len(res_index):
            result.columns = res_index

        return result


class FrameColumnApply(FrameApply):
    axis: AxisInt = 1

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        result = super().apply_broadcast(target.T)
        return result.T

    @property
    def series_generator(self) -> Generator[Series, None, None]:
        values = self.values
        values = ensure_wrapped_if_datetimelike(values)
        assert len(values) > 0

        # We create one Series object, and will swap out the data inside
        #  of it.  Kids: don't do this at home.
        ser = self.obj._ixs(0, axis=0)
        mgr = ser._mgr

        is_view = mgr.blocks[0].refs.has_reference()  # type: ignore[union-attr]

        if isinstance(ser.dtype, ExtensionDtype):
            # values will be incorrect for this block
            # TODO(EA2D): special case would be unnecessary with 2D EAs
            obj = self.obj
            for i in range(len(obj)):
                yield obj._ixs(i, axis=0)

        else:
            for arr, name in zip(values, self.index):
                # GH#35462 re-pin mgr in case setitem changed it
                ser._mgr = mgr
                mgr.set_values(arr)
                object.__setattr__(ser, "_name", name)
                if not is_view:
                    # In apply_series_generator we store the a shallow copy of the
                    # result, which potentially increases the ref count of this reused
                    # `ser` object (depending on the result of the applied function)
                    # -> if that happened and `ser` is already a copy, then we reset
                    # the refs here to avoid triggering a unnecessary CoW inside the
                    # applied function (https://github.com/pandas-dev/pandas/pull/56212)
                    mgr.blocks[0].refs = BlockValuesRefs(mgr.blocks[0])  # type: ignore[union-attr]
                yield ser

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        numba = import_optional_dependency("numba")
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str

        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names_index, index):
            results = {}
            # Currently the parallel argument doesn't get passed through here
            # (it's disabled) since the dicts in numba aren't thread-safe.
            for i in range(values.shape[0]):
                # Create the series
                # TODO: values corrupted without the copy
                ser = Series(
                    values[i].copy(),
                    index=col_names_index,
                    name=maybe_cast_str(index[i]),
                )
                results[i] = jitted_udf(ser)

            return results

        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        nb_func = self.generate_numba_apply_func(
            cast(Callable, self.func), **self.engine_kwargs
        )

        from pandas.core._numba.extensions import set_numba_data

        # Convert from numba dict to regular dict
        # Our isinstance checks in the df constructor don't pass for numbas typed dict
        with set_numba_data(self.obj.index) as index, set_numba_data(
            self.columns
        ) as columns:
            res = dict(nb_func(self.values, columns, index))

        return res

    @property
    def result_index(self) -> Index:
        return self.index

    @property
    def result_columns(self) -> Index:
        return self.columns

    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        """return the results for the columns"""
        result: DataFrame | Series

        # we have requested to expand
        if self.result_type == "expand":
            result = self.infer_to_same_shape(results, res_index)

        # we have a non-series and don't want inference
        elif not isinstance(results[0], ABCSeries):
            result = self.obj._constructor_sliced(results)
            result.index = res_index

        # we may want to infer results
        else:
            result = self.infer_to_same_shape(results, res_index)

        return result

    def infer_to_same_shape(self, results: ResType, res_index: Index) -> DataFrame:
        """infer the results to the same shape as the input object"""
        result = self.obj._constructor(data=results)
        result = result.T

        # set the index
        result.index = res_index

        # infer dtypes
        result = result.infer_objects(copy=False)

        return result


class SeriesApply(NDFrameApply):
    obj: Series
    axis: AxisInt = 0
    by_row: Literal[False, "compat", "_compat"]  # only relevant for apply()

    def __init__(
        self,
        obj: Series,
        func: AggFuncType,
        *,
        convert_dtype: bool | lib.NoDefault = lib.no_default,
        by_row: Literal[False, "compat", "_compat"] = "compat",
        args,
        kwargs,
    ) -> None:
        if convert_dtype is lib.no_default:
            convert_dtype = True
        else:
            warnings.warn(
                "the convert_dtype parameter is deprecated and will be removed in a "
                "future version.  Do ``ser.astype(object).apply()`` "
                "instead if you want ``convert_dtype=False``.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        self.convert_dtype = convert_dtype

        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            by_row=by_row,
            args=args,
            kwargs=kwargs,
        )

    def apply(self) -> DataFrame | Series:
        obj = self.obj

        if len(obj) == 0:
            return self.apply_empty_result()

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()

        if isinstance(self.func, str):
            # if we are a string, try to dispatch
            return self.apply_str()

        if self.by_row == "_compat":
            return self.apply_compat()

        # self.func is Callable
        return self.apply_standard()

    def agg(self):
        result = super().agg()
        if result is None:
            obj = self.obj
            func = self.func
            # string, list-like, and dict-like are entirely handled in super
            assert callable(func)

            # GH53325: The setup below is just to keep current behavior while emitting a
            # deprecation message. In the future this will all be replaced with a simple
            # `result = f(self.obj, *self.args, **self.kwargs)`.
            try:
                result = obj.apply(func, args=self.args, **self.kwargs)
            except (ValueError, AttributeError, TypeError):
                result = func(obj, *self.args, **self.kwargs)
            else:
                msg = (
                    f"using {func} in {type(obj).__name__}.agg cannot aggregate and "
                    f"has been deprecated. Use {type(obj).__name__}.transform to "
                    f"keep behavior unchanged."
                )
                warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())

        return result

    def apply_empty_result(self) -> Series:
        obj = self.obj
        return obj._constructor(dtype=obj.dtype, index=obj.index).__finalize__(
            obj, method="apply"
        )

    def apply_compat(self):
        """compat apply method for funcs in listlikes and dictlikes.

         Used for each callable when giving listlikes and dictlikes of callables to
         apply. Needed for compatibility with Pandas < v2.1.

        .. versionadded:: 2.1.0
        """
        obj = self.obj
        func = self.func

        if callable(func):
            f = com.get_cython_func(func)
            if f and not self.args and not self.kwargs:
                return obj.apply(func, by_row=False)

        try:
            result = obj.apply(func, by_row="compat")
        except (ValueError, AttributeError, TypeError):
            result = obj.apply(func, by_row=False)
        return result

    def apply_standard(self) -> DataFrame | Series:
        # caller is responsible for ensuring that f is Callable
        func = cast(Callable, self.func)
        obj = self.obj

        if isinstance(func, np.ufunc):
            with np.errstate(all="ignore"):
                return func(obj, *self.args, **self.kwargs)
        elif not self.by_row:
            return func(obj, *self.args, **self.kwargs)

        if self.args or self.kwargs:
            # _map_values does not support args/kwargs
            def curried(x):
                return func(x, *self.args, **self.kwargs)

        else:
            curried = func

        # row-wise access
        # apply doesn't have a `na_action` keyword and for backward compat reasons
        # we need to give `na_action="ignore"` for categorical data.
        # TODO: remove the `na_action="ignore"` when that default has been changed in
        #  Categorical (GH51645).
        action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
        mapped = obj._map_values(
            mapper=curried, na_action=action, convert=self.convert_dtype
        )

        if len(mapped) and isinstance(mapped[0], ABCSeries):
            # GH#43986 Need to do list(mapped) in order to get treated as nested
            #  See also GH#25959 regarding EA support
            return obj._constructor_expanddim(list(mapped), index=obj.index)
        else:
            return obj._constructor(mapped, index=obj.index).__finalize__(
                obj, method="apply"
            )


class GroupByApply(Apply):
    obj: GroupBy | Resampler | BaseWindow

    def __init__(
        self,
        obj: GroupBy[NDFrameT],
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:
        kwargs = kwargs.copy()
        self.axis = obj.obj._get_axis_number(kwargs.get("axis", 0))
        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

    def apply(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        obj = self.obj
        kwargs = self.kwargs
        if op_name == "apply":
            kwargs = {**kwargs, "by_row": False}

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        if obj._selected_obj.ndim == 1:
            # For SeriesGroupBy this matches _obj_with_exclusions
            selected_obj = obj._selected_obj
        else:
            selected_obj = obj._obj_with_exclusions

        # Only set as_index=True on groupby objects, not Window or Resample
        # that inherit from this class.
        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        assert op_name in ["agg", "apply"]

        obj = self.obj
        kwargs = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        selected_obj = obj._selected_obj
        selection = obj._selection

        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))

        # Numba Groupby engine/engine-kwargs passthrough
        if is_groupby:
            engine = self.kwargs.get("engine", None)
            engine_kwargs = self.kwargs.get("engine_kwargs", None)
            kwargs.update({"engine": engine, "engine_kwargs": engine_kwargs})

        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            result_index, result_data = self.compute_dict_like(
                op_name, selected_obj, selection, kwargs
            )
        result = self.wrap_results_dict_like(selected_obj, result_index, result_data)
        return result


class ResamplerWindowApply(GroupByApply):
    axis: AxisInt = 0
    obj: Resampler | BaseWindow

    def __init__(
        self,
        obj: Resampler | BaseWindow,
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:
        super(GroupByApply, self).__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

    def apply(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType, tuple[str, ...] | None, npt.NDArray[np.intp] | None]:
    """
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. 'min') or
    Callable, or list of them (e.g. ['min', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {'A': 'min'}, or {'A': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. 'min' or Callable) or list of agg functions
        (e.g. ['min', np.max]) or dictionary (e.g. {'A': ['min', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: tuple of column names
    order: array of columns indices

    Examples
    --------
    >>> reconstruct_func(None, **{"foo": ("col", "min")})
    (True, defaultdict(<class 'list'>, {'col': ['min']}), ('foo',), array([0]))

    >>> reconstruct_func("min")
    (False, 'min', None, None)
    """
    relabeling = func is None and is_multi_agg_with_relabel(**kwargs)
    columns: tuple[str, ...] | None = None
    order: npt.NDArray[np.intp] | None = None

    if not relabeling:
        if isinstance(func, list) and len(func) > len(set(func)):
            # GH 28426 will raise error if duplicated function names are used and
            # there is no reassigned name
            raise SpecificationError(
                "Function names must be unique if there is no new column names "
                "assigned"
            )
        if func is None:
            # nicer error message
            raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")

    if relabeling:
        # error: Incompatible types in assignment (expression has type
        # "MutableMapping[Hashable, list[Callable[..., Any] | str]]", variable has type
        # "Callable[..., Any] | str | list[Callable[..., Any] | str] |
        # MutableMapping[Hashable, Callable[..., Any] | str | list[Callable[..., Any] |
        # str]] | None")
        func, columns, order = normalize_keyword_aggregation(  # type: ignore[assignment]
            kwargs
        )
    assert func is not None

    return relabeling, func, columns, order


def is_multi_agg_with_relabel(**kwargs) -> bool:
    """
    Check whether kwargs passed to .agg look like multi-agg with relabeling.

    Parameters
    ----------
    **kwargs : dict

    Returns
    -------
    bool

    Examples
    --------
    >>> is_multi_agg_with_relabel(a="max")
    False
    >>> is_multi_agg_with_relabel(a_max=("a", "max"), a_min=("a", "min"))
    True
    >>> is_multi_agg_with_relabel()
    False
    """
    return all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values()) and (
        len(kwargs) > 0
    )


def normalize_keyword_aggregation(
    kwargs: dict,
) -> tuple[
    MutableMapping[Hashable, list[AggFuncTypeBase]],
    tuple[str, ...],
    npt.NDArray[np.intp],
]:
    """
    Normalize user-provided "named aggregation" kwargs.
    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs
    to the old Dict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : tuple[str, ...]
        The user-provided keys.
    col_idx_order : List[int]
        List of columns indices.

    Examples
    --------
    >>> normalize_keyword_aggregation({"output": ("input", "sum")})
    (defaultdict(<class 'list'>, {'input': ['sum']}), ('output',), array([0]))
    """
    from pandas.core.indexes.base import Index

    # Normalize the aggregation functions as Mapping[column, List[func]],
    # process normally, then fixup the names.
    # TODO: aggspec type: typing.Dict[str, List[AggScalar]]
    aggspec = defaultdict(list)
    order = []
    columns, pairs = list(zip(*kwargs.items()))

    for column, aggfunc in pairs:
        aggspec[column].append(aggfunc)
        order.append((column, com.get_callable_name(aggfunc) or aggfunc))

    # uniquify aggfunc name if duplicated in order list
    uniquified_order = _make_unique_kwarg_list(order)

    # GH 25719, due to aggspec will change the order of assigned columns in aggregation
    # uniquified_aggspec will store uniquified order list and will compare it with order
    # based on index
    aggspec_order = [
        (column, com.get_callable_name(aggfunc) or aggfunc)
        for column, aggfuncs in aggspec.items()
        for aggfunc in aggfuncs
    ]
    uniquified_aggspec = _make_unique_kwarg_list(aggspec_order)

    # get the new index of columns by comparison
    col_idx_order = Index(uniquified_aggspec).get_indexer(uniquified_order)
    return aggspec, columns, col_idx_order


def _make_unique_kwarg_list(
    seq: Sequence[tuple[Any, Any]]
) -> Sequence[tuple[Any, Any]]:
    """
    Uniquify aggfunc name of the pairs in the order list

    Examples:
    --------
    >>> kwarg_list = [('a', '<lambda>'), ('a', '<lambda>'), ('b', '<lambda>')]
    >>> _make_unique_kwarg_list(kwarg_list)
    [('a', '<lambda>_0'), ('a', '<lambda>_1'), ('b', '<lambda>')]
    """
    return [
        (pair[0], f"{pair[1]}_{seq[:i].count(pair)}") if seq.count(pair) > 1 else pair
        for i, pair in enumerate(seq)
    ]


def relabel_result(
    result: DataFrame | Series,
    func: dict[str, list[Callable | str]],
    columns: Iterable[Hashable],
    order: Iterable[int],
) -> dict[Hashable, Series]:
    """
    Internal function to reorder result if relabelling is True for
    dataframe.agg, and return the reordered result in dict.

    Parameters:
    ----------
    result: Result from aggregation
    func: Dict of (column name, funcs)
    columns: New columns name for relabelling
    order: New order for relabelling

    Examples
    --------
    >>> from pandas.core.apply import relabel_result
    >>> result = pd.DataFrame(
    ...     {"A": [np.nan, 2, np.nan], "C": [6, np.nan, np.nan], "B": [np.nan, 4, 2.5]},
    ...     index=["max", "mean", "min"]
    ... )
    >>> funcs = {"A": ["max"], "C": ["max"], "B": ["mean", "min"]}
    >>> columns = ("foo", "aab", "bar", "dat")
    >>> order = [0, 1, 2, 3]
    >>> result_in_dict = relabel_result(result, funcs, columns, order)
    >>> pd.DataFrame(result_in_dict, index=columns)
           A    C    B
    foo  2.0  NaN  NaN
    aab  NaN  6.0  NaN
    bar  NaN  NaN  4.0
    dat  NaN  NaN  2.5
    """
    from pandas.core.indexes.base import Index

    reordered_indexes = [
        pair[0] for pair in sorted(zip(columns, order), key=lambda t: t[1])
    ]
    reordered_result_in_dict: dict[Hashable, Series] = {}
    idx = 0

    reorder_mask = not isinstance(result, ABCSeries) and len(result.columns) > 1
    for col, fun in func.items():
        s = result[col].dropna()

        # In the `_aggregate`, the callable names are obtained and used in `result`, and
        # these names are ordered alphabetically. e.g.
        #           C2   C1
        # <lambda>   1  NaN
        # amax     NaN  4.0
        # max      NaN  4.0
        # sum     18.0  6.0
        # Therefore, the order of functions for each column could be shuffled
        # accordingly so need to get the callable name if it is not parsed names, and
        # reorder the aggregated result for each column.
        # e.g. if df.agg(c1=("C2", sum), c2=("C2", lambda x: min(x))), correct order is
        # [sum, <lambda>], but in `result`, it will be [<lambda>, sum], and we need to
        # reorder so that aggregated values map to their functions regarding the order.

        # However there is only one column being used for aggregation, not need to
        # reorder since the index is not sorted, and keep as is in `funcs`, e.g.
        #         A
        # min   1.0
        # mean  1.5
        # mean  1.5
        if reorder_mask:
            fun = [
                com.get_callable_name(f) if not isinstance(f, str) else f for f in fun
            ]
            col_idx_order = Index(s.index).get_indexer(fun)
            s = s.iloc[col_idx_order]

        # assign the new user-provided "named aggregation" as index names, and reindex
        # it based on the whole user-provided names.
        s.index = reordered_indexes[idx : idx + len(fun)]
        reordered_result_in_dict[col] = s.reindex(columns, copy=False)
        idx = idx + len(fun)
    return reordered_result_in_dict


def reconstruct_and_relabel_result(result, func, **kwargs) -> DataFrame | Series:
    from pandas import DataFrame

    relabeling, func, columns, order = reconstruct_func(func, **kwargs)

    if relabeling:
        # This is to keep the order to columns occurrence unchanged, and also
        # keep the order of new columns occurrence unchanged

        # For the return values of reconstruct_func, if relabeling is
        # False, columns and order will be None.
        assert columns is not None
        assert order is not None

        result_in_dict = relabel_result(result, func, columns, order)
        result = DataFrame(result_in_dict, index=columns)

    return result


# TODO: Can't use, because mypy doesn't like us setting __name__
#   error: "partial[Any]" has no attribute "__name__"
# the type is:
#   typing.Sequence[Callable[..., ScalarResult]]
#     -> typing.Sequence[Callable[..., ScalarResult]]:


def _managle_lambda_list(aggfuncs: Sequence[Any]) -> Sequence[Any]:
    """
    Possibly mangle a list of aggfuncs.

    Parameters
    ----------
    aggfuncs : Sequence

    Returns
    -------
    mangled: list-like
        A new AggSpec sequence, where lambdas have been converted
        to have unique names.

    Notes
    -----
    If just one aggfunc is passed, the name will not be mangled.
    """
    if len(aggfuncs) <= 1:
        # don't mangle for .agg([lambda x: .])
        return aggfuncs
    i = 0
    mangled_aggfuncs = []
    for aggfunc in aggfuncs:
        if com.get_callable_name(aggfunc) == "<lambda>":
            aggfunc = partial(aggfunc)
            aggfunc.__name__ = f"<lambda_{i}>"
            i += 1
        mangled_aggfuncs.append(aggfunc)

    return mangled_aggfuncs


def maybe_mangle_lambdas(agg_spec: Any) -> Any:
    """
    Make new lambdas with unique names.

    Parameters
    ----------
    agg_spec : Any
        An argument to GroupBy.agg.
        Non-dict-like `agg_spec` are pass through as is.
        For dict-like `agg_spec` a new spec is returned
        with name-mangled lambdas.

    Returns
    -------
    mangled : Any
        Same type as the input.

    Examples
    --------
    >>> maybe_mangle_lambdas('sum')
    'sum'
    >>> maybe_mangle_lambdas([lambda: 1, lambda: 2])  # doctest: +SKIP
    [<function __main__.<lambda_0>,
     <function pandas...._make_lambda.<locals>.f(*args, **kwargs)>]
    """
    is_dict = is_dict_like(agg_spec)
    if not (is_dict or is_list_like(agg_spec)):
        return agg_spec
    mangled_aggspec = type(agg_spec)()  # dict or OrderedDict

    if is_dict:
        for key, aggfuncs in agg_spec.items():
            if is_list_like(aggfuncs) and not is_dict_like(aggfuncs):
                mangled_aggfuncs = _managle_lambda_list(aggfuncs)
            else:
                mangled_aggfuncs = aggfuncs

            mangled_aggspec[key] = mangled_aggfuncs
    else:
        mangled_aggspec = _managle_lambda_list(agg_spec)

    return mangled_aggspec


def validate_func_kwargs(
    kwargs: dict,
) -> tuple[list[str], list[str | Callable[..., Any]]]:
    """
    Validates types of user-provided "named aggregation" kwargs.
    `TypeError` is raised if aggfunc is not `str` or callable.

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    columns : List[str]
        List of user-provided keys.
    func : List[Union[str, callable[...,Any]]]
        List of user-provided aggfuncs

    Examples
    --------
    >>> validate_func_kwargs({'one': 'min', 'two': 'max'})
    (['one', 'two'], ['min', 'max'])
    """
    tuple_given_message = "func is expected but received {} in **kwargs."
    columns = list(kwargs)
    func = []
    for col_func in kwargs.values():
        if not (isinstance(col_func, str) or callable(col_func)):
            raise TypeError(tuple_given_message.format(type(col_func).__name__))
        func.append(col_func)
    if not columns:
        no_arg_message = "Must provide 'func' or named aggregation **kwargs."
        raise TypeError(no_arg_message)
    return columns, func


def include_axis(op_name: Literal["agg", "apply"], colg: Series | DataFrame) -> bool:
    return isinstance(colg, ABCDataFrame) or (
        isinstance(colg, ABCSeries) and op_name == "agg"
    )


def warn_alias_replacement(
    obj: AggObjType,
    func: Callable,
    alias: str,
) -> None:
    if alias.startswith("np."):
        full_alias = alias
    else:
        full_alias = f"{type(obj).__name__}.{alias}"
        alias = f'"{alias}"'
    warnings.warn(
        f"The provided callable {func} is currently using "
        f"{full_alias}. In a future version of pandas, "
        f"the provided callable will be used directly. To keep current "
        f"behavior pass the string {alias} instead.",
        category=FutureWarning,
        stacklevel=find_stack_level(),
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\constant.py ===
from __future__ import annotations

from codecs import BOM_UTF8, BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE
from encodings.aliases import aliases
from re import IGNORECASE
from re import compile as re_compile

# Contain for each eligible encoding a list of/item bytes SIG/BOM
ENCODING_MARKS: dict[str, bytes | list[bytes]] = {
    "utf_8": BOM_UTF8,
    "utf_7": [
        b"\x2b\x2f\x76\x38",
        b"\x2b\x2f\x76\x39",
        b"\x2b\x2f\x76\x2b",
        b"\x2b\x2f\x76\x2f",
        b"\x2b\x2f\x76\x38\x2d",
    ],
    "gb18030": b"\x84\x31\x95\x33",
    "utf_32": [BOM_UTF32_BE, BOM_UTF32_LE],
    "utf_16": [BOM_UTF16_BE, BOM_UTF16_LE],
}

TOO_SMALL_SEQUENCE: int = 32
TOO_BIG_SEQUENCE: int = int(10e6)

UTF8_MAXIMAL_ALLOCATION: int = 1_112_064

# Up-to-date Unicode ucd/15.0.0
UNICODE_RANGES_COMBINED: dict[str, range] = {
    "Control character": range(32),
    "Basic Latin": range(32, 128),
    "Latin-1 Supplement": range(128, 256),
    "Latin Extended-A": range(256, 384),
    "Latin Extended-B": range(384, 592),
    "IPA Extensions": range(592, 688),
    "Spacing Modifier Letters": range(688, 768),
    "Combining Diacritical Marks": range(768, 880),
    "Greek and Coptic": range(880, 1024),
    "Cyrillic": range(1024, 1280),
    "Cyrillic Supplement": range(1280, 1328),
    "Armenian": range(1328, 1424),
    "Hebrew": range(1424, 1536),
    "Arabic": range(1536, 1792),
    "Syriac": range(1792, 1872),
    "Arabic Supplement": range(1872, 1920),
    "Thaana": range(1920, 1984),
    "NKo": range(1984, 2048),
    "Samaritan": range(2048, 2112),
    "Mandaic": range(2112, 2144),
    "Syriac Supplement": range(2144, 2160),
    "Arabic Extended-B": range(2160, 2208),
    "Arabic Extended-A": range(2208, 2304),
    "Devanagari": range(2304, 2432),
    "Bengali": range(2432, 2560),
    "Gurmukhi": range(2560, 2688),
    "Gujarati": range(2688, 2816),
    "Oriya": range(2816, 2944),
    "Tamil": range(2944, 3072),
    "Telugu": range(3072, 3200),
    "Kannada": range(3200, 3328),
    "Malayalam": range(3328, 3456),
    "Sinhala": range(3456, 3584),
    "Thai": range(3584, 3712),
    "Lao": range(3712, 3840),
    "Tibetan": range(3840, 4096),
    "Myanmar": range(4096, 4256),
    "Georgian": range(4256, 4352),
    "Hangul Jamo": range(4352, 4608),
    "Ethiopic": range(4608, 4992),
    "Ethiopic Supplement": range(4992, 5024),
    "Cherokee": range(5024, 5120),
    "Unified Canadian Aboriginal Syllabics": range(5120, 5760),
    "Ogham": range(5760, 5792),
    "Runic": range(5792, 5888),
    "Tagalog": range(5888, 5920),
    "Hanunoo": range(5920, 5952),
    "Buhid": range(5952, 5984),
    "Tagbanwa": range(5984, 6016),
    "Khmer": range(6016, 6144),
    "Mongolian": range(6144, 6320),
    "Unified Canadian Aboriginal Syllabics Extended": range(6320, 6400),
    "Limbu": range(6400, 6480),
    "Tai Le": range(6480, 6528),
    "New Tai Lue": range(6528, 6624),
    "Khmer Symbols": range(6624, 6656),
    "Buginese": range(6656, 6688),
    "Tai Tham": range(6688, 6832),
    "Combining Diacritical Marks Extended": range(6832, 6912),
    "Balinese": range(6912, 7040),
    "Sundanese": range(7040, 7104),
    "Batak": range(7104, 7168),
    "Lepcha": range(7168, 7248),
    "Ol Chiki": range(7248, 7296),
    "Cyrillic Extended-C": range(7296, 7312),
    "Georgian Extended": range(7312, 7360),
    "Sundanese Supplement": range(7360, 7376),
    "Vedic Extensions": range(7376, 7424),
    "Phonetic Extensions": range(7424, 7552),
    "Phonetic Extensions Supplement": range(7552, 7616),
    "Combining Diacritical Marks Supplement": range(7616, 7680),
    "Latin Extended Additional": range(7680, 7936),
    "Greek Extended": range(7936, 8192),
    "General Punctuation": range(8192, 8304),
    "Superscripts and Subscripts": range(8304, 8352),
    "Currency Symbols": range(8352, 8400),
    "Combining Diacritical Marks for Symbols": range(8400, 8448),
    "Letterlike Symbols": range(8448, 8528),
    "Number Forms": range(8528, 8592),
    "Arrows": range(8592, 8704),
    "Mathematical Operators": range(8704, 8960),
    "Miscellaneous Technical": range(8960, 9216),
    "Control Pictures": range(9216, 9280),
    "Optical Character Recognition": range(9280, 9312),
    "Enclosed Alphanumerics": range(9312, 9472),
    "Box Drawing": range(9472, 9600),
    "Block Elements": range(9600, 9632),
    "Geometric Shapes": range(9632, 9728),
    "Miscellaneous Symbols": range(9728, 9984),
    "Dingbats": range(9984, 10176),
    "Miscellaneous Mathematical Symbols-A": range(10176, 10224),
    "Supplemental Arrows-A": range(10224, 10240),
    "Braille Patterns": range(10240, 10496),
    "Supplemental Arrows-B": range(10496, 10624),
    "Miscellaneous Mathematical Symbols-B": range(10624, 10752),
    "Supplemental Mathematical Operators": range(10752, 11008),
    "Miscellaneous Symbols and Arrows": range(11008, 11264),
    "Glagolitic": range(11264, 11360),
    "Latin Extended-C": range(11360, 11392),
    "Coptic": range(11392, 11520),
    "Georgian Supplement": range(11520, 11568),
    "Tifinagh": range(11568, 11648),
    "Ethiopic Extended": range(11648, 11744),
    "Cyrillic Extended-A": range(11744, 11776),
    "Supplemental Punctuation": range(11776, 11904),
    "CJK Radicals Supplement": range(11904, 12032),
    "Kangxi Radicals": range(12032, 12256),
    "Ideographic Description Characters": range(12272, 12288),
    "CJK Symbols and Punctuation": range(12288, 12352),
    "Hiragana": range(12352, 12448),
    "Katakana": range(12448, 12544),
    "Bopomofo": range(12544, 12592),
    "Hangul Compatibility Jamo": range(12592, 12688),
    "Kanbun": range(12688, 12704),
    "Bopomofo Extended": range(12704, 12736),
    "CJK Strokes": range(12736, 12784),
    "Katakana Phonetic Extensions": range(12784, 12800),
    "Enclosed CJK Letters and Months": range(12800, 13056),
    "CJK Compatibility": range(13056, 13312),
    "CJK Unified Ideographs Extension A": range(13312, 19904),
    "Yijing Hexagram Symbols": range(19904, 19968),
    "CJK Unified Ideographs": range(19968, 40960),
    "Yi Syllables": range(40960, 42128),
    "Yi Radicals": range(42128, 42192),
    "Lisu": range(42192, 42240),
    "Vai": range(42240, 42560),
    "Cyrillic Extended-B": range(42560, 42656),
    "Bamum": range(42656, 42752),
    "Modifier Tone Letters": range(42752, 42784),
    "Latin Extended-D": range(42784, 43008),
    "Syloti Nagri": range(43008, 43056),
    "Common Indic Number Forms": range(43056, 43072),
    "Phags-pa": range(43072, 43136),
    "Saurashtra": range(43136, 43232),
    "Devanagari Extended": range(43232, 43264),
    "Kayah Li": range(43264, 43312),
    "Rejang": range(43312, 43360),
    "Hangul Jamo Extended-A": range(43360, 43392),
    "Javanese": range(43392, 43488),
    "Myanmar Extended-B": range(43488, 43520),
    "Cham": range(43520, 43616),
    "Myanmar Extended-A": range(43616, 43648),
    "Tai Viet": range(43648, 43744),
    "Meetei Mayek Extensions": range(43744, 43776),
    "Ethiopic Extended-A": range(43776, 43824),
    "Latin Extended-E": range(43824, 43888),
    "Cherokee Supplement": range(43888, 43968),
    "Meetei Mayek": range(43968, 44032),
    "Hangul Syllables": range(44032, 55216),
    "Hangul Jamo Extended-B": range(55216, 55296),
    "High Surrogates": range(55296, 56192),
    "High Private Use Surrogates": range(56192, 56320),
    "Low Surrogates": range(56320, 57344),
    "Private Use Area": range(57344, 63744),
    "CJK Compatibility Ideographs": range(63744, 64256),
    "Alphabetic Presentation Forms": range(64256, 64336),
    "Arabic Presentation Forms-A": range(64336, 65024),
    "Variation Selectors": range(65024, 65040),
    "Vertical Forms": range(65040, 65056),
    "Combining Half Marks": range(65056, 65072),
    "CJK Compatibility Forms": range(65072, 65104),
    "Small Form Variants": range(65104, 65136),
    "Arabic Presentation Forms-B": range(65136, 65280),
    "Halfwidth and Fullwidth Forms": range(65280, 65520),
    "Specials": range(65520, 65536),
    "Linear B Syllabary": range(65536, 65664),
    "Linear B Ideograms": range(65664, 65792),
    "Aegean Numbers": range(65792, 65856),
    "Ancient Greek Numbers": range(65856, 65936),
    "Ancient Symbols": range(65936, 66000),
    "Phaistos Disc": range(66000, 66048),
    "Lycian": range(66176, 66208),
    "Carian": range(66208, 66272),
    "Coptic Epact Numbers": range(66272, 66304),
    "Old Italic": range(66304, 66352),
    "Gothic": range(66352, 66384),
    "Old Permic": range(66384, 66432),
    "Ugaritic": range(66432, 66464),
    "Old Persian": range(66464, 66528),
    "Deseret": range(66560, 66640),
    "Shavian": range(66640, 66688),
    "Osmanya": range(66688, 66736),
    "Osage": range(66736, 66816),
    "Elbasan": range(66816, 66864),
    "Caucasian Albanian": range(66864, 66928),
    "Vithkuqi": range(66928, 67008),
    "Linear A": range(67072, 67456),
    "Latin Extended-F": range(67456, 67520),
    "Cypriot Syllabary": range(67584, 67648),
    "Imperial Aramaic": range(67648, 67680),
    "Palmyrene": range(67680, 67712),
    "Nabataean": range(67712, 67760),
    "Hatran": range(67808, 67840),
    "Phoenician": range(67840, 67872),
    "Lydian": range(67872, 67904),
    "Meroitic Hieroglyphs": range(67968, 68000),
    "Meroitic Cursive": range(68000, 68096),
    "Kharoshthi": range(68096, 68192),
    "Old South Arabian": range(68192, 68224),
    "Old North Arabian": range(68224, 68256),
    "Manichaean": range(68288, 68352),
    "Avestan": range(68352, 68416),
    "Inscriptional Parthian": range(68416, 68448),
    "Inscriptional Pahlavi": range(68448, 68480),
    "Psalter Pahlavi": range(68480, 68528),
    "Old Turkic": range(68608, 68688),
    "Old Hungarian": range(68736, 68864),
    "Hanifi Rohingya": range(68864, 68928),
    "Rumi Numeral Symbols": range(69216, 69248),
    "Yezidi": range(69248, 69312),
    "Arabic Extended-C": range(69312, 69376),
    "Old Sogdian": range(69376, 69424),
    "Sogdian": range(69424, 69488),
    "Old Uyghur": range(69488, 69552),
    "Chorasmian": range(69552, 69600),
    "Elymaic": range(69600, 69632),
    "Brahmi": range(69632, 69760),
    "Kaithi": range(69760, 69840),
    "Sora Sompeng": range(69840, 69888),
    "Chakma": range(69888, 69968),
    "Mahajani": range(69968, 70016),
    "Sharada": range(70016, 70112),
    "Sinhala Archaic Numbers": range(70112, 70144),
    "Khojki": range(70144, 70224),
    "Multani": range(70272, 70320),
    "Khudawadi": range(70320, 70400),
    "Grantha": range(70400, 70528),
    "Newa": range(70656, 70784),
    "Tirhuta": range(70784, 70880),
    "Siddham": range(71040, 71168),
    "Modi": range(71168, 71264),
    "Mongolian Supplement": range(71264, 71296),
    "Takri": range(71296, 71376),
    "Ahom": range(71424, 71504),
    "Dogra": range(71680, 71760),
    "Warang Citi": range(71840, 71936),
    "Dives Akuru": range(71936, 72032),
    "Nandinagari": range(72096, 72192),
    "Zanabazar Square": range(72192, 72272),
    "Soyombo": range(72272, 72368),
    "Unified Canadian Aboriginal Syllabics Extended-A": range(72368, 72384),
    "Pau Cin Hau": range(72384, 72448),
    "Devanagari Extended-A": range(72448, 72544),
    "Bhaiksuki": range(72704, 72816),
    "Marchen": range(72816, 72896),
    "Masaram Gondi": range(72960, 73056),
    "Gunjala Gondi": range(73056, 73136),
    "Makasar": range(73440, 73472),
    "Kawi": range(73472, 73568),
    "Lisu Supplement": range(73648, 73664),
    "Tamil Supplement": range(73664, 73728),
    "Cuneiform": range(73728, 74752),
    "Cuneiform Numbers and Punctuation": range(74752, 74880),
    "Early Dynastic Cuneiform": range(74880, 75088),
    "Cypro-Minoan": range(77712, 77824),
    "Egyptian Hieroglyphs": range(77824, 78896),
    "Egyptian Hieroglyph Format Controls": range(78896, 78944),
    "Anatolian Hieroglyphs": range(82944, 83584),
    "Bamum Supplement": range(92160, 92736),
    "Mro": range(92736, 92784),
    "Tangsa": range(92784, 92880),
    "Bassa Vah": range(92880, 92928),
    "Pahawh Hmong": range(92928, 93072),
    "Medefaidrin": range(93760, 93856),
    "Miao": range(93952, 94112),
    "Ideographic Symbols and Punctuation": range(94176, 94208),
    "Tangut": range(94208, 100352),
    "Tangut Components": range(100352, 101120),
    "Khitan Small Script": range(101120, 101632),
    "Tangut Supplement": range(101632, 101760),
    "Kana Extended-B": range(110576, 110592),
    "Kana Supplement": range(110592, 110848),
    "Kana Extended-A": range(110848, 110896),
    "Small Kana Extension": range(110896, 110960),
    "Nushu": range(110960, 111360),
    "Duployan": range(113664, 113824),
    "Shorthand Format Controls": range(113824, 113840),
    "Znamenny Musical Notation": range(118528, 118736),
    "Byzantine Musical Symbols": range(118784, 119040),
    "Musical Symbols": range(119040, 119296),
    "Ancient Greek Musical Notation": range(119296, 119376),
    "Kaktovik Numerals": range(119488, 119520),
    "Mayan Numerals": range(119520, 119552),
    "Tai Xuan Jing Symbols": range(119552, 119648),
    "Counting Rod Numerals": range(119648, 119680),
    "Mathematical Alphanumeric Symbols": range(119808, 120832),
    "Sutton SignWriting": range(120832, 121520),
    "Latin Extended-G": range(122624, 122880),
    "Glagolitic Supplement": range(122880, 122928),
    "Cyrillic Extended-D": range(122928, 123024),
    "Nyiakeng Puachue Hmong": range(123136, 123216),
    "Toto": range(123536, 123584),
    "Wancho": range(123584, 123648),
    "Nag Mundari": range(124112, 124160),
    "Ethiopic Extended-B": range(124896, 124928),
    "Mende Kikakui": range(124928, 125152),
    "Adlam": range(125184, 125280),
    "Indic Siyaq Numbers": range(126064, 126144),
    "Ottoman Siyaq Numbers": range(126208, 126288),
    "Arabic Mathematical Alphabetic Symbols": range(126464, 126720),
    "Mahjong Tiles": range(126976, 127024),
    "Domino Tiles": range(127024, 127136),
    "Playing Cards": range(127136, 127232),
    "Enclosed Alphanumeric Supplement": range(127232, 127488),
    "Enclosed Ideographic Supplement": range(127488, 127744),
    "Miscellaneous Symbols and Pictographs": range(127744, 128512),
    "Emoticons range(Emoji)": range(128512, 128592),
    "Ornamental Dingbats": range(128592, 128640),
    "Transport and Map Symbols": range(128640, 128768),
    "Alchemical Symbols": range(128768, 128896),
    "Geometric Shapes Extended": range(128896, 129024),
    "Supplemental Arrows-C": range(129024, 129280),
    "Supplemental Symbols and Pictographs": range(129280, 129536),
    "Chess Symbols": range(129536, 129648),
    "Symbols and Pictographs Extended-A": range(129648, 129792),
    "Symbols for Legacy Computing": range(129792, 130048),
    "CJK Unified Ideographs Extension B": range(131072, 173792),
    "CJK Unified Ideographs Extension C": range(173824, 177984),
    "CJK Unified Ideographs Extension D": range(177984, 178208),
    "CJK Unified Ideographs Extension E": range(178208, 183984),
    "CJK Unified Ideographs Extension F": range(183984, 191472),
    "CJK Compatibility Ideographs Supplement": range(194560, 195104),
    "CJK Unified Ideographs Extension G": range(196608, 201552),
    "CJK Unified Ideographs Extension H": range(201552, 205744),
    "Tags": range(917504, 917632),
    "Variation Selectors Supplement": range(917760, 918000),
    "Supplementary Private Use Area-A": range(983040, 1048576),
    "Supplementary Private Use Area-B": range(1048576, 1114112),
}


UNICODE_SECONDARY_RANGE_KEYWORD: list[str] = [
    "Supplement",
    "Extended",
    "Extensions",
    "Modifier",
    "Marks",
    "Punctuation",
    "Symbols",
    "Forms",
    "Operators",
    "Miscellaneous",
    "Drawing",
    "Block",
    "Shapes",
    "Supplemental",
    "Tags",
]

RE_POSSIBLE_ENCODING_INDICATION = re_compile(
    r"(?:(?:encoding)|(?:charset)|(?:coding))(?:[\:= ]{1,10})(?:[\"\']?)([a-zA-Z0-9\-_]+)(?:[\"\']?)",
    IGNORECASE,
)

IANA_NO_ALIASES = [
    "cp720",
    "cp737",
    "cp856",
    "cp874",
    "cp875",
    "cp1006",
    "koi8_r",
    "koi8_t",
    "koi8_u",
]

IANA_SUPPORTED: list[str] = sorted(
    filter(
        lambda x: x.endswith("_codec") is False
        and x not in {"rot_13", "tactis", "mbcs"},
        list(set(aliases.values())) + IANA_NO_ALIASES,
    )
)

IANA_SUPPORTED_COUNT: int = len(IANA_SUPPORTED)

# pre-computed code page that are similar using the function cp_similarity.
IANA_SUPPORTED_SIMILAR: dict[str, list[str]] = {
    "cp037": ["cp1026", "cp1140", "cp273", "cp500"],
    "cp1026": ["cp037", "cp1140", "cp273", "cp500"],
    "cp1125": ["cp866"],
    "cp1140": ["cp037", "cp1026", "cp273", "cp500"],
    "cp1250": ["iso8859_2"],
    "cp1251": ["kz1048", "ptcp154"],
    "cp1252": ["iso8859_15", "iso8859_9", "latin_1"],
    "cp1253": ["iso8859_7"],
    "cp1254": ["iso8859_15", "iso8859_9", "latin_1"],
    "cp1257": ["iso8859_13"],
    "cp273": ["cp037", "cp1026", "cp1140", "cp500"],
    "cp437": ["cp850", "cp858", "cp860", "cp861", "cp862", "cp863", "cp865"],
    "cp500": ["cp037", "cp1026", "cp1140", "cp273"],
    "cp850": ["cp437", "cp857", "cp858", "cp865"],
    "cp857": ["cp850", "cp858", "cp865"],
    "cp858": ["cp437", "cp850", "cp857", "cp865"],
    "cp860": ["cp437", "cp861", "cp862", "cp863", "cp865"],
    "cp861": ["cp437", "cp860", "cp862", "cp863", "cp865"],
    "cp862": ["cp437", "cp860", "cp861", "cp863", "cp865"],
    "cp863": ["cp437", "cp860", "cp861", "cp862", "cp865"],
    "cp865": ["cp437", "cp850", "cp857", "cp858", "cp860", "cp861", "cp862", "cp863"],
    "cp866": ["cp1125"],
    "iso8859_10": ["iso8859_14", "iso8859_15", "iso8859_4", "iso8859_9", "latin_1"],
    "iso8859_11": ["tis_620"],
    "iso8859_13": ["cp1257"],
    "iso8859_14": [
        "iso8859_10",
        "iso8859_15",
        "iso8859_16",
        "iso8859_3",
        "iso8859_9",
        "latin_1",
    ],
    "iso8859_15": [
        "cp1252",
        "cp1254",
        "iso8859_10",
        "iso8859_14",
        "iso8859_16",
        "iso8859_3",
        "iso8859_9",
        "latin_1",
    ],
    "iso8859_16": [
        "iso8859_14",
        "iso8859_15",
        "iso8859_2",
        "iso8859_3",
        "iso8859_9",
        "latin_1",
    ],
    "iso8859_2": ["cp1250", "iso8859_16", "iso8859_4"],
    "iso8859_3": ["iso8859_14", "iso8859_15", "iso8859_16", "iso8859_9", "latin_1"],
    "iso8859_4": ["iso8859_10", "iso8859_2", "iso8859_9", "latin_1"],
    "iso8859_7": ["cp1253"],
    "iso8859_9": [
        "cp1252",
        "cp1254",
        "cp1258",
        "iso8859_10",
        "iso8859_14",
        "iso8859_15",
        "iso8859_16",
        "iso8859_3",
        "iso8859_4",
        "latin_1",
    ],
    "kz1048": ["cp1251", "ptcp154"],
    "latin_1": [
        "cp1252",
        "cp1254",
        "cp1258",
        "iso8859_10",
        "iso8859_14",
        "iso8859_15",
        "iso8859_16",
        "iso8859_3",
        "iso8859_4",
        "iso8859_9",
    ],
    "mac_iceland": ["mac_roman", "mac_turkish"],
    "mac_roman": ["mac_iceland", "mac_turkish"],
    "mac_turkish": ["mac_iceland", "mac_roman"],
    "ptcp154": ["cp1251", "kz1048"],
    "tis_620": ["iso8859_11"],
}


CHARDET_CORRESPONDENCE: dict[str, str] = {
    "iso2022_kr": "ISO-2022-KR",
    "iso2022_jp": "ISO-2022-JP",
    "euc_kr": "EUC-KR",
    "tis_620": "TIS-620",
    "utf_32": "UTF-32",
    "euc_jp": "EUC-JP",
    "koi8_r": "KOI8-R",
    "iso8859_1": "ISO-8859-1",
    "iso8859_2": "ISO-8859-2",
    "iso8859_5": "ISO-8859-5",
    "iso8859_6": "ISO-8859-6",
    "iso8859_7": "ISO-8859-7",
    "iso8859_8": "ISO-8859-8",
    "utf_16": "UTF-16",
    "cp855": "IBM855",
    "mac_cyrillic": "MacCyrillic",
    "gb2312": "GB2312",
    "gb18030": "GB18030",
    "cp932": "CP932",
    "cp866": "IBM866",
    "utf_8": "utf-8",
    "utf_8_sig": "UTF-8-SIG",
    "shift_jis": "SHIFT_JIS",
    "big5": "Big5",
    "cp1250": "windows-1250",
    "cp1251": "windows-1251",
    "cp1252": "Windows-1252",
    "cp1253": "windows-1253",
    "cp1255": "windows-1255",
    "cp1256": "windows-1256",
    "cp1254": "Windows-1254",
    "cp949": "CP949",
}


COMMON_SAFE_ASCII_CHARACTERS: set[str] = {
    "<",
    ">",
    "=",
    ":",
    "/",
    "&",
    ";",
    "{",
    "}",
    "[",
    "]",
    ",",
    "|",
    '"',
    "-",
    "(",
    ")",
}

# Sample character sets — replace with full lists if needed
COMMON_CHINESE_CHARACTERS = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"

COMMON_JAPANESE_CHARACTERS = "日一国年大十二本中長出三時行見月分後前生五間上東四今金九入学高円子外八六下来気小七山話女北午百書先名川千水半男西電校語土木聞食車何南万毎白天母火右読友左休父雨"

COMMON_KOREAN_CHARACTERS = "一二三四五六七八九十百千萬上下左右中人女子大小山川日月火水木金土父母天地國名年時文校學生"

# Combine all into a set
COMMON_CJK_CHARACTERS = set(
    "".join(
        [
            COMMON_CHINESE_CHARACTERS,
            COMMON_JAPANESE_CHARACTERS,
            COMMON_KOREAN_CHARACTERS,
        ]
    )
)

KO_NAMES: set[str] = {"johab", "cp949", "euc_kr"}
ZH_NAMES: set[str] = {"big5", "cp950", "big5hkscs", "hz"}

# Logging LEVEL below DEBUG
TRACE: int = 5


# Language label that contain the em dash "—"
# character are to be considered alternative seq to origin
FREQUENCIES: dict[str, list[str]] = {
    "English": [
        "e",
        "a",
        "t",
        "i",
        "o",
        "n",
        "s",
        "r",
        "h",
        "l",
        "d",
        "c",
        "u",
        "m",
        "f",
        "p",
        "g",
        "w",
        "y",
        "b",
        "v",
        "k",
        "x",
        "j",
        "z",
        "q",
    ],
    "English—": [
        "e",
        "a",
        "t",
        "i",
        "o",
        "n",
        "s",
        "r",
        "h",
        "l",
        "d",
        "c",
        "m",
        "u",
        "f",
        "p",
        "g",
        "w",
        "b",
        "y",
        "v",
        "k",
        "j",
        "x",
        "z",
        "q",
    ],
    "German": [
        "e",
        "n",
        "i",
        "r",
        "s",
        "t",
        "a",
        "d",
        "h",
        "u",
        "l",
        "g",
        "o",
        "c",
        "m",
        "b",
        "f",
        "k",
        "w",
        "z",
        "p",
        "v",
        "ü",
        "ä",
        "ö",
        "j",
    ],
    "French": [
        "e",
        "a",
        "s",
        "n",
        "i",
        "t",
        "r",
        "l",
        "u",
        "o",
        "d",
        "c",
        "p",
        "m",
        "é",
        "v",
        "g",
        "f",
        "b",
        "h",
        "q",
        "à",
        "x",
        "è",
        "y",
        "j",
    ],
    "Dutch": [
        "e",
        "n",
        "a",
        "i",
        "r",
        "t",
        "o",
        "d",
        "s",
        "l",
        "g",
        "h",
        "v",
        "m",
        "u",
        "k",
        "c",
        "p",
        "b",
        "w",
        "j",
        "z",
        "f",
        "y",
        "x",
        "ë",
    ],
    "Italian": [
        "e",
        "i",
        "a",
        "o",
        "n",
        "l",
        "t",
        "r",
        "s",
        "c",
        "d",
        "u",
        "p",
        "m",
        "g",
        "v",
        "f",
        "b",
        "z",
        "h",
        "q",
        "è",
        "à",
        "k",
        "y",
        "ò",
    ],
    "Polish": [
        "a",
        "i",
        "o",
        "e",
        "n",
        "r",
        "z",
        "w",
        "s",
        "c",
        "t",
        "k",
        "y",
        "d",
        "p",
        "m",
        "u",
        "l",
        "j",
        "ł",
        "g",
        "b",
        "h",
        "ą",
        "ę",
        "ó",
    ],
    "Spanish": [
        "e",
        "a",
        "o",
        "n",
        "s",
        "r",
        "i",
        "l",
        "d",
        "t",
        "c",
        "u",
        "m",
        "p",
        "b",
        "g",
        "v",
        "f",
        "y",
        "ó",
        "h",
        "q",
        "í",
        "j",
        "z",
        "á",
    ],
    "Russian": [
        "о",
        "а",
        "е",
        "и",
        "н",
        "с",
        "т",
        "р",
        "в",
        "л",
        "к",
        "м",
        "д",
        "п",
        "у",
        "г",
        "я",
        "ы",
        "з",
        "б",
        "й",
        "ь",
        "ч",
        "х",
        "ж",
        "ц",
    ],
    # Jap-Kanji
    "Japanese": [
        "人",
        "一",
        "大",
        "亅",
        "丁",
        "丨",
        "竹",
        "笑",
        "口",
        "日",
        "今",
        "二",
        "彳",
        "行",
        "十",
        "土",
        "丶",
        "寸",
        "寺",
        "時",
        "乙",
        "丿",
        "乂",
        "气",
        "気",
        "冂",
        "巾",
        "亠",
        "市",
        "目",
        "儿",
        "見",
        "八",
        "小",
        "凵",
        "県",
        "月",
        "彐",
        "門",
        "間",
        "木",
        "東",
        "山",
        "出",
        "本",
        "中",
        "刀",
        "分",
        "耳",
        "又",
        "取",
        "最",
        "言",
        "田",
        "心",
        "思",
        "刂",
        "前",
        "京",
        "尹",
        "事",
        "生",
        "厶",
        "云",
        "会",
        "未",
        "来",
        "白",
        "冫",
        "楽",
        "灬",
        "馬",
        "尸",
        "尺",
        "駅",
        "明",
        "耂",
        "者",
        "了",
        "阝",
        "都",
        "高",
        "卜",
        "占",
        "厂",
        "广",
        "店",
        "子",
        "申",
        "奄",
        "亻",
        "俺",
        "上",
        "方",
        "冖",
        "学",
        "衣",
        "艮",
        "食",
        "自",
    ],
    # Jap-Katakana
    "Japanese—": [
        "ー",
        "ン",
        "ス",
        "・",
        "ル",
        "ト",
        "リ",
        "イ",
        "ア",
        "ラ",
        "ッ",
        "ク",
        "ド",
        "シ",
        "レ",
        "ジ",
        "タ",
        "フ",
        "ロ",
        "カ",
        "テ",
        "マ",
        "ィ",
        "グ",
        "バ",
        "ム",
        "プ",
        "オ",
        "コ",
        "デ",
        "ニ",
        "ウ",
        "メ",
        "サ",
        "ビ",
        "ナ",
        "ブ",
        "ャ",
        "エ",
        "ュ",
        "チ",
        "キ",
        "ズ",
        "ダ",
        "パ",
        "ミ",
        "ェ",
        "ョ",
        "ハ",
        "セ",
        "ベ",
        "ガ",
        "モ",
        "ツ",
        "ネ",
        "ボ",
        "ソ",
        "ノ",
        "ァ",
        "ヴ",
        "ワ",
        "ポ",
        "ペ",
        "ピ",
        "ケ",
        "ゴ",
        "ギ",
        "ザ",
        "ホ",
        "ゲ",
        "ォ",
        "ヤ",
        "ヒ",
        "ユ",
        "ヨ",
        "ヘ",
        "ゼ",
        "ヌ",
        "ゥ",
        "ゾ",
        "ヶ",
        "ヂ",
        "ヲ",
        "ヅ",
        "ヵ",
        "ヱ",
        "ヰ",
        "ヮ",
        "ヽ",
        "゠",
        "ヾ",
        "ヷ",
        "ヿ",
        "ヸ",
        "ヹ",
        "ヺ",
    ],
    # Jap-Hiragana
    "Japanese——": [
        "の",
        "に",
        "る",
        "た",
        "と",
        "は",
        "し",
        "い",
        "を",
        "で",
        "て",
        "が",
        "な",
        "れ",
        "か",
        "ら",
        "さ",
        "っ",
        "り",
        "す",
        "あ",
        "も",
        "こ",
        "ま",
        "う",
        "く",
        "よ",
        "き",
        "ん",
        "め",
        "お",
        "け",
        "そ",
        "つ",
        "だ",
        "や",
        "え",
        "ど",
        "わ",
        "ち",
        "み",
        "せ",
        "じ",
        "ば",
        "へ",
        "び",
        "ず",
        "ろ",
        "ほ",
        "げ",
        "む",
        "べ",
        "ひ",
        "ょ",
        "ゆ",
        "ぶ",
        "ご",
        "ゃ",
        "ね",
        "ふ",
        "ぐ",
        "ぎ",
        "ぼ",
        "ゅ",
        "づ",
        "ざ",
        "ぞ",
        "ぬ",
        "ぜ",
        "ぱ",
        "ぽ",
        "ぷ",
        "ぴ",
        "ぃ",
        "ぁ",
        "ぇ",
        "ぺ",
        "ゞ",
        "ぢ",
        "ぉ",
        "ぅ",
        "ゐ",
        "ゝ",
        "ゑ",
        "゛",
        "゜",
        "ゎ",
        "ゔ",
        "゚",
        "ゟ",
        "゙",
        "ゕ",
        "ゖ",
    ],
    "Portuguese": [
        "a",
        "e",
        "o",
        "s",
        "i",
        "r",
        "d",
        "n",
        "t",
        "m",
        "u",
        "c",
        "l",
        "p",
        "g",
        "v",
        "b",
        "f",
        "h",
        "ã",
        "q",
        "é",
        "ç",
        "á",
        "z",
        "í",
    ],
    "Swedish": [
        "e",
        "a",
        "n",
        "r",
        "t",
        "s",
        "i",
        "l",
        "d",
        "o",
        "m",
        "k",
        "g",
        "v",
        "h",
        "f",
        "u",
        "p",
        "ä",
        "c",
        "b",
        "ö",
        "å",
        "y",
        "j",
        "x",
    ],
    "Chinese": [
        "的",
        "一",
        "是",
        "不",
        "了",
        "在",
        "人",
        "有",
        "我",
        "他",
        "这",
        "个",
        "们",
        "中",
        "来",
        "上",
        "大",
        "为",
        "和",
        "国",
        "地",
        "到",
        "以",
        "说",
        "时",
        "要",
        "就",
        "出",
        "会",
        "可",
        "也",
        "你",
        "对",
        "生",
        "能",
        "而",
        "子",
        "那",
        "得",
        "于",
        "着",
        "下",
        "自",
        "之",
        "年",
        "过",
        "发",
        "后",
        "作",
        "里",
        "用",
        "道",
        "行",
        "所",
        "然",
        "家",
        "种",
        "事",
        "成",
        "方",
        "多",
        "经",
        "么",
        "去",
        "法",
        "学",
        "如",
        "都",
        "同",
        "现",
        "当",
        "没",
        "动",
        "面",
        "起",
        "看",
        "定",
        "天",
        "分",
        "还",
        "进",
        "好",
        "小",
        "部",
        "其",
        "些",
        "主",
        "样",
        "理",
        "心",
        "她",
        "本",
        "前",
        "开",
        "但",
        "因",
        "只",
        "从",
        "想",
        "实",
    ],
    "Ukrainian": [
        "о",
        "а",
        "н",
        "і",
        "и",
        "р",
        "в",
        "т",
        "е",
        "с",
        "к",
        "л",
        "у",
        "д",
        "м",
        "п",
        "з",
        "я",
        "ь",
        "б",
        "г",
        "й",
        "ч",
        "х",
        "ц",
        "ї",
    ],
    "Norwegian": [
        "e",
        "r",
        "n",
        "t",
        "a",
        "s",
        "i",
        "o",
        "l",
        "d",
        "g",
        "k",
        "m",
        "v",
        "f",
        "p",
        "u",
        "b",
        "h",
        "å",
        "y",
        "j",
        "ø",
        "c",
        "æ",
        "w",
    ],
    "Finnish": [
        "a",
        "i",
        "n",
        "t",
        "e",
        "s",
        "l",
        "o",
        "u",
        "k",
        "ä",
        "m",
        "r",
        "v",
        "j",
        "h",
        "p",
        "y",
        "d",
        "ö",
        "g",
        "c",
        "b",
        "f",
        "w",
        "z",
    ],
    "Vietnamese": [
        "n",
        "h",
        "t",
        "i",
        "c",
        "g",
        "a",
        "o",
        "u",
        "m",
        "l",
        "r",
        "à",
        "đ",
        "s",
        "e",
        "v",
        "p",
        "b",
        "y",
        "ư",
        "d",
        "á",
        "k",
        "ộ",
        "ế",
    ],
    "Czech": [
        "o",
        "e",
        "a",
        "n",
        "t",
        "s",
        "i",
        "l",
        "v",
        "r",
        "k",
        "d",
        "u",
        "m",
        "p",
        "í",
        "c",
        "h",
        "z",
        "á",
        "y",
        "j",
        "b",
        "ě",
        "é",
        "ř",
    ],
    "Hungarian": [
        "e",
        "a",
        "t",
        "l",
        "s",
        "n",
        "k",
        "r",
        "i",
        "o",
        "z",
        "á",
        "é",
        "g",
        "m",
        "b",
        "y",
        "v",
        "d",
        "h",
        "u",
        "p",
        "j",
        "ö",
        "f",
        "c",
    ],
    "Korean": [
        "이",
        "다",
        "에",
        "의",
        "는",
        "로",
        "하",
        "을",
        "가",
        "고",
        "지",
        "서",
        "한",
        "은",
        "기",
        "으",
        "년",
        "대",
        "사",
        "시",
        "를",
        "리",
        "도",
        "인",
        "스",
        "일",
    ],
    "Indonesian": [
        "a",
        "n",
        "e",
        "i",
        "r",
        "t",
        "u",
        "s",
        "d",
        "k",
        "m",
        "l",
        "g",
        "p",
        "b",
        "o",
        "h",
        "y",
        "j",
        "c",
        "w",
        "f",
        "v",
        "z",
        "x",
        "q",
    ],
    "Turkish": [
        "a",
        "e",
        "i",
        "n",
        "r",
        "l",
        "ı",
        "k",
        "d",
        "t",
        "s",
        "m",
        "y",
        "u",
        "o",
        "b",
        "ü",
        "ş",
        "v",
        "g",
        "z",
        "h",
        "c",
        "p",
        "ç",
        "ğ",
    ],
    "Romanian": [
        "e",
        "i",
        "a",
        "r",
        "n",
        "t",
        "u",
        "l",
        "o",
        "c",
        "s",
        "d",
        "p",
        "m",
        "ă",
        "f",
        "v",
        "î",
        "g",
        "b",
        "ș",
        "ț",
        "z",
        "h",
        "â",
        "j",
    ],
    "Farsi": [
        "ا",
        "ی",
        "ر",
        "د",
        "ن",
        "ه",
        "و",
        "م",
        "ت",
        "ب",
        "س",
        "ل",
        "ک",
        "ش",
        "ز",
        "ف",
        "گ",
        "ع",
        "خ",
        "ق",
        "ج",
        "آ",
        "پ",
        "ح",
        "ط",
        "ص",
    ],
    "Arabic": [
        "ا",
        "ل",
        "ي",
        "م",
        "و",
        "ن",
        "ر",
        "ت",
        "ب",
        "ة",
        "ع",
        "د",
        "س",
        "ف",
        "ه",
        "ك",
        "ق",
        "أ",
        "ح",
        "ج",
        "ش",
        "ط",
        "ص",
        "ى",
        "خ",
        "إ",
    ],
    "Danish": [
        "e",
        "r",
        "n",
        "t",
        "a",
        "i",
        "s",
        "d",
        "l",
        "o",
        "g",
        "m",
        "k",
        "f",
        "v",
        "u",
        "b",
        "h",
        "p",
        "å",
        "y",
        "ø",
        "æ",
        "c",
        "j",
        "w",
    ],
    "Serbian": [
        "а",
        "и",
        "о",
        "е",
        "н",
        "р",
        "с",
        "у",
        "т",
        "к",
        "ј",
        "в",
        "д",
        "м",
        "п",
        "л",
        "г",
        "з",
        "б",
        "a",
        "i",
        "e",
        "o",
        "n",
        "ц",
        "ш",
    ],
    "Lithuanian": [
        "i",
        "a",
        "s",
        "o",
        "r",
        "e",
        "t",
        "n",
        "u",
        "k",
        "m",
        "l",
        "p",
        "v",
        "d",
        "j",
        "g",
        "ė",
        "b",
        "y",
        "ų",
        "š",
        "ž",
        "c",
        "ą",
        "į",
    ],
    "Slovene": [
        "e",
        "a",
        "i",
        "o",
        "n",
        "r",
        "s",
        "l",
        "t",
        "j",
        "v",
        "k",
        "d",
        "p",
        "m",
        "u",
        "z",
        "b",
        "g",
        "h",
        "č",
        "c",
        "š",
        "ž",
        "f",
        "y",
    ],
    "Slovak": [
        "o",
        "a",
        "e",
        "n",
        "i",
        "r",
        "v",
        "t",
        "s",
        "l",
        "k",
        "d",
        "m",
        "p",
        "u",
        "c",
        "h",
        "j",
        "b",
        "z",
        "á",
        "y",
        "ý",
        "í",
        "č",
        "é",
    ],
    "Hebrew": [
        "י",
        "ו",
        "ה",
        "ל",
        "ר",
        "ב",
        "ת",
        "מ",
        "א",
        "ש",
        "נ",
        "ע",
        "ם",
        "ד",
        "ק",
        "ח",
        "פ",
        "ס",
        "כ",
        "ג",
        "ט",
        "צ",
        "ן",
        "ז",
        "ך",
    ],
    "Bulgarian": [
        "а",
        "и",
        "о",
        "е",
        "н",
        "т",
        "р",
        "с",
        "в",
        "л",
        "к",
        "д",
        "п",
        "м",
        "з",
        "г",
        "я",
        "ъ",
        "у",
        "б",
        "ч",
        "ц",
        "й",
        "ж",
        "щ",
        "х",
    ],
    "Croatian": [
        "a",
        "i",
        "o",
        "e",
        "n",
        "r",
        "j",
        "s",
        "t",
        "u",
        "k",
        "l",
        "v",
        "d",
        "m",
        "p",
        "g",
        "z",
        "b",
        "c",
        "č",
        "h",
        "š",
        "ž",
        "ć",
        "f",
    ],
    "Hindi": [
        "क",
        "र",
        "स",
        "न",
        "त",
        "म",
        "ह",
        "प",
        "य",
        "ल",
        "व",
        "ज",
        "द",
        "ग",
        "ब",
        "श",
        "ट",
        "अ",
        "ए",
        "थ",
        "भ",
        "ड",
        "च",
        "ध",
        "ष",
        "इ",
    ],
    "Estonian": [
        "a",
        "i",
        "e",
        "s",
        "t",
        "l",
        "u",
        "n",
        "o",
        "k",
        "r",
        "d",
        "m",
        "v",
        "g",
        "p",
        "j",
        "h",
        "ä",
        "b",
        "õ",
        "ü",
        "f",
        "c",
        "ö",
        "y",
    ],
    "Thai": [
        "า",
        "น",
        "ร",
        "อ",
        "ก",
        "เ",
        "ง",
        "ม",
        "ย",
        "ล",
        "ว",
        "ด",
        "ท",
        "ส",
        "ต",
        "ะ",
        "ป",
        "บ",
        "ค",
        "ห",
        "แ",
        "จ",
        "พ",
        "ช",
        "ข",
        "ใ",
    ],
    "Greek": [
        "α",
        "τ",
        "ο",
        "ι",
        "ε",
        "ν",
        "ρ",
        "σ",
        "κ",
        "η",
        "π",
        "ς",
        "υ",
        "μ",
        "λ",
        "ί",
        "ό",
        "ά",
        "γ",
        "έ",
        "δ",
        "ή",
        "ω",
        "χ",
        "θ",
        "ύ",
    ],
    "Tamil": [
        "க",
        "த",
        "ப",
        "ட",
        "ர",
        "ம",
        "ல",
        "ன",
        "வ",
        "ற",
        "ய",
        "ள",
        "ச",
        "ந",
        "இ",
        "ண",
        "அ",
        "ஆ",
        "ழ",
        "ங",
        "எ",
        "உ",
        "ஒ",
        "ஸ",
    ],
    "Kazakh": [
        "а",
        "ы",
        "е",
        "н",
        "т",
        "р",
        "л",
        "і",
        "д",
        "с",
        "м",
        "қ",
        "к",
        "о",
        "б",
        "и",
        "у",
        "ғ",
        "ж",
        "ң",
        "з",
        "ш",
        "й",
        "п",
        "г",
        "ө",
    ],
}

LANGUAGE_SUPPORTED_COUNT: int = len(FREQUENCIES)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_array_utils_impl.py ===
"""
Miscellaneous utils.
"""
from numpy._core import asarray
from numpy._core.numeric import normalize_axis_index, normalize_axis_tuple
from numpy._utils import set_module

__all__ = ["byte_bounds", "normalize_axis_tuple", "normalize_axis_index"]


@set_module("numpy.lib.array_utils")
def byte_bounds(a):
    """
    Returns pointers to the end-points of an array.

    Parameters
    ----------
    a : ndarray
        Input array. It must conform to the Python-side of the array
        interface.

    Returns
    -------
    (low, high) : tuple of 2 integers
        The first integer is the first byte of the array, the second
        integer is just past the last byte of the array.  If `a` is not
        contiguous it will not use every byte between the (`low`, `high`)
        values.

    Examples
    --------
    >>> import numpy as np
    >>> I = np.eye(2, dtype='f'); I.dtype
    dtype('float32')
    >>> low, high = np.lib.array_utils.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True
    >>> I = np.eye(2); I.dtype
    dtype('float64')
    >>> low, high = np.lib.array_utils.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True

    """
    ai = a.__array_interface__
    a_data = ai['data'][0]
    astrides = ai['strides']
    ashape = ai['shape']
    bytes_a = asarray(a).dtype.itemsize

    a_low = a_high = a_data
    if astrides is None:
        # contiguous case
        a_high += a.size * bytes_a
    else:
        for shape, stride in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape - 1) * stride
            else:
                a_high += (shape - 1) * stride
        a_high += bytes_a
    return a_low, a_high

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\associationproxy.py ===
# ext/associationproxy.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Contain the ``AssociationProxy`` class.

The ``AssociationProxy`` is a Python property object which provides
transparent proxied access to the endpoint of an association object.

See the example ``examples/association/proxied_association.py``.

"""
from __future__ import annotations

import operator
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import ItemsView
from typing import Iterable
from typing import Iterator
from typing import KeysView
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import MutableSet
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import ValuesView

from .. import ColumnElement
from .. import exc
from .. import inspect
from .. import orm
from .. import util
from ..orm import collections
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.base import SQLORMOperations
from ..orm.interfaces import _AttributeOptions
from ..orm.interfaces import _DCAttributeOptions
from ..orm.interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from ..sql import operators
from ..sql import or_
from ..sql.base import _NoArg
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import SupportsIndex
from ..util.typing import SupportsKeysAndGetItem

if typing.TYPE_CHECKING:
    from ..orm.interfaces import MapperProperty
    from ..orm.interfaces import PropComparator
    from ..orm.mapper import Mapper
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _InfoType


_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)
_T_con = TypeVar("_T_con", bound=Any, contravariant=True)
_S = TypeVar("_S", bound=Any)
_KT = TypeVar("_KT", bound=Any)
_VT = TypeVar("_VT", bound=Any)


def association_proxy(
    target_collection: str,
    attr: str,
    *,
    creator: Optional[_CreatorProtocol] = None,
    getset_factory: Optional[_GetSetFactoryProtocol] = None,
    proxy_factory: Optional[_ProxyFactoryProtocol] = None,
    proxy_bulk_set: Optional[_ProxyBulkSetProtocol] = None,
    info: Optional[_InfoType] = None,
    cascade_scalar_deletes: bool = False,
    create_on_none_assignment: bool = False,
    init: Union[_NoArg, bool] = _NoArg.NO_ARG,
    repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
    default: Optional[Any] = _NoArg.NO_ARG,
    default_factory: Union[_NoArg, Callable[[], _T]] = _NoArg.NO_ARG,
    compare: Union[_NoArg, bool] = _NoArg.NO_ARG,
    kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
    hash: Union[_NoArg, bool, None] = _NoArg.NO_ARG,  # noqa: A002
) -> AssociationProxy[Any]:
    r"""Return a Python property implementing a view of a target
    attribute which references an attribute on members of the
    target.

    The returned value is an instance of :class:`.AssociationProxy`.

    Implements a Python property representing a relationship as a collection
    of simpler values, or a scalar value.  The proxied property will mimic
    the collection type of the target (list, dict or set), or, in the case of
    a one to one relationship, a simple scalar value.

    :param target_collection: Name of the attribute that is the immediate
      target.  This attribute is typically mapped by
      :func:`~sqlalchemy.orm.relationship` to link to a target collection, but
      can also be a many-to-one or non-scalar relationship.

    :param attr: Attribute on the associated instance or instances that
      are available on instances of the target object.

    :param creator: optional.

      Defines custom behavior when new items are added to the proxied
      collection.

      By default, adding new items to the collection will trigger a
      construction of an instance of the target object, passing the given
      item as a positional argument to the target constructor.  For cases
      where this isn't sufficient, :paramref:`.association_proxy.creator`
      can supply a callable that will construct the object in the
      appropriate way, given the item that was passed.

      For list- and set- oriented collections, a single argument is
      passed to the callable. For dictionary oriented collections, two
      arguments are passed, corresponding to the key and value.

      The :paramref:`.association_proxy.creator` callable is also invoked
      for scalar (i.e. many-to-one, one-to-one) relationships. If the
      current value of the target relationship attribute is ``None``, the
      callable is used to construct a new object.  If an object value already
      exists, the given attribute value is populated onto that object.

      .. seealso::

        :ref:`associationproxy_creator`

    :param cascade_scalar_deletes: when True, indicates that setting
        the proxied value to ``None``, or deleting it via ``del``, should
        also remove the source object.  Only applies to scalar attributes.
        Normally, removing the proxied target will not remove the proxy
        source, as this object may have other state that is still to be
        kept.

        .. versionadded:: 1.3

        .. seealso::

            :ref:`cascade_scalar_deletes` - complete usage example

    :param create_on_none_assignment: when True, indicates that setting
      the proxied value to ``None`` should **create** the source object
      if it does not exist, using the creator.  Only applies to scalar
      attributes.  This is mutually exclusive
      vs. the :paramref:`.assocation_proxy.cascade_scalar_deletes`.

      .. versionadded:: 2.0.18

    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__init__()``
     method as generated by the dataclass process.

     .. versionadded:: 2.0.0b4

    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the attribute established by this :class:`.AssociationProxy`
     should be part of the ``__repr__()`` method as generated by the dataclass
     process.

     .. versionadded:: 2.0.0b4

    :param default_factory: Specific to
     :ref:`orm_declarative_native_dataclasses`, specifies a default-value
     generation function that will take place as part of the ``__init__()``
     method as generated by the dataclass process.

     .. versionadded:: 2.0.0b4

    :param compare: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be included in comparison operations when generating the
     ``__eq__()`` and ``__ne__()`` methods for the mapped class.

     .. versionadded:: 2.0.0b4

    :param kw_only: Specific to :ref:`orm_declarative_native_dataclasses`,
     indicates if this field should be marked as keyword-only when generating
     the ``__init__()`` method as generated by the dataclass process.

     .. versionadded:: 2.0.0b4

    :param hash: Specific to
     :ref:`orm_declarative_native_dataclasses`, controls if this field
     is included when generating the ``__hash__()`` method for the mapped
     class.

     .. versionadded:: 2.0.36

    :param info: optional, will be assigned to
     :attr:`.AssociationProxy.info` if present.


    The following additional parameters involve injection of custom behaviors
    within the :class:`.AssociationProxy` object and are for advanced use
    only:

    :param getset_factory: Optional.  Proxied attribute access is
        automatically handled by routines that get and set values based on
        the `attr` argument for this proxy.

        If you would like to customize this behavior, you may supply a
        `getset_factory` callable that produces a tuple of `getter` and
        `setter` functions.  The factory is called with two arguments, the
        abstract type of the underlying collection and this proxy instance.

    :param proxy_factory: Optional.  The type of collection to emulate is
        determined by sniffing the target collection.  If your collection
        type can't be determined by duck typing or you'd like to use a
        different collection implementation, you may supply a factory
        function to produce those collections.  Only applicable to
        non-scalar relationships.

    :param proxy_bulk_set: Optional, use with proxy_factory.


    """
    return AssociationProxy(
        target_collection,
        attr,
        creator=creator,
        getset_factory=getset_factory,
        proxy_factory=proxy_factory,
        proxy_bulk_set=proxy_bulk_set,
        info=info,
        cascade_scalar_deletes=cascade_scalar_deletes,
        create_on_none_assignment=create_on_none_assignment,
        attribute_options=_AttributeOptions(
            init, repr, default, default_factory, compare, kw_only, hash
        ),
    )


class AssociationProxyExtensionType(InspectionAttrExtensionType):
    ASSOCIATION_PROXY = "ASSOCIATION_PROXY"
    """Symbol indicating an :class:`.InspectionAttr` that's
    of type :class:`.AssociationProxy`.

    Is assigned to the :attr:`.InspectionAttr.extension_type`
    attribute.

    """


class _GetterProtocol(Protocol[_T_co]):
    def __call__(self, instance: Any) -> _T_co: ...


# mypy 0.990 we are no longer allowed to make this Protocol[_T_con]
class _SetterProtocol(Protocol): ...


class _PlainSetterProtocol(_SetterProtocol, Protocol[_T_con]):
    def __call__(self, instance: Any, value: _T_con) -> None: ...


class _DictSetterProtocol(_SetterProtocol, Protocol[_T_con]):
    def __call__(self, instance: Any, key: Any, value: _T_con) -> None: ...


# mypy 0.990 we are no longer allowed to make this Protocol[_T_con]
class _CreatorProtocol(Protocol): ...


class _PlainCreatorProtocol(_CreatorProtocol, Protocol[_T_con]):
    def __call__(self, value: _T_con) -> Any: ...


class _KeyCreatorProtocol(_CreatorProtocol, Protocol[_T_con]):
    def __call__(self, key: Any, value: Optional[_T_con]) -> Any: ...


class _LazyCollectionProtocol(Protocol[_T]):
    def __call__(
        self,
    ) -> Union[
        MutableSet[_T], MutableMapping[Any, _T], MutableSequence[_T]
    ]: ...


class _GetSetFactoryProtocol(Protocol):
    def __call__(
        self,
        collection_class: Optional[Type[Any]],
        assoc_instance: AssociationProxyInstance[Any],
    ) -> Tuple[_GetterProtocol[Any], _SetterProtocol]: ...


class _ProxyFactoryProtocol(Protocol):
    def __call__(
        self,
        lazy_collection: _LazyCollectionProtocol[Any],
        creator: _CreatorProtocol,
        value_attr: str,
        parent: AssociationProxyInstance[Any],
    ) -> Any: ...


class _ProxyBulkSetProtocol(Protocol):
    def __call__(
        self, proxy: _AssociationCollection[Any], collection: Iterable[Any]
    ) -> None: ...


class _AssociationProxyProtocol(Protocol[_T]):
    """describes the interface of :class:`.AssociationProxy`
    without including descriptor methods in the interface."""

    creator: Optional[_CreatorProtocol]
    key: str
    target_collection: str
    value_attr: str
    cascade_scalar_deletes: bool
    create_on_none_assignment: bool
    getset_factory: Optional[_GetSetFactoryProtocol]
    proxy_factory: Optional[_ProxyFactoryProtocol]
    proxy_bulk_set: Optional[_ProxyBulkSetProtocol]

    @util.ro_memoized_property
    def info(self) -> _InfoType: ...

    def for_class(
        self, class_: Type[Any], obj: Optional[object] = None
    ) -> AssociationProxyInstance[_T]: ...

    def _default_getset(
        self, collection_class: Any
    ) -> Tuple[_GetterProtocol[Any], _SetterProtocol]: ...


class AssociationProxy(
    interfaces.InspectionAttrInfo,
    ORMDescriptor[_T],
    _DCAttributeOptions,
    _AssociationProxyProtocol[_T],
):
    """A descriptor that presents a read/write view of an object attribute."""

    is_attribute = True
    extension_type = AssociationProxyExtensionType.ASSOCIATION_PROXY

    def __init__(
        self,
        target_collection: str,
        attr: str,
        *,
        creator: Optional[_CreatorProtocol] = None,
        getset_factory: Optional[_GetSetFactoryProtocol] = None,
        proxy_factory: Optional[_ProxyFactoryProtocol] = None,
        proxy_bulk_set: Optional[_ProxyBulkSetProtocol] = None,
        info: Optional[_InfoType] = None,
        cascade_scalar_deletes: bool = False,
        create_on_none_assignment: bool = False,
        attribute_options: Optional[_AttributeOptions] = None,
    ):
        """Construct a new :class:`.AssociationProxy`.

        The :class:`.AssociationProxy` object is typically constructed using
        the :func:`.association_proxy` constructor function. See the
        description of :func:`.association_proxy` for a description of all
        parameters.


        """
        self.target_collection = target_collection
        self.value_attr = attr
        self.creator = creator
        self.getset_factory = getset_factory
        self.proxy_factory = proxy_factory
        self.proxy_bulk_set = proxy_bulk_set

        if cascade_scalar_deletes and create_on_none_assignment:
            raise exc.ArgumentError(
                "The cascade_scalar_deletes and create_on_none_assignment "
                "parameters are mutually exclusive."
            )
        self.cascade_scalar_deletes = cascade_scalar_deletes
        self.create_on_none_assignment = create_on_none_assignment

        self.key = "_%s_%s_%s" % (
            type(self).__name__,
            target_collection,
            id(self),
        )
        if info:
            self.info = info  # type: ignore

        if (
            attribute_options
            and attribute_options != _DEFAULT_ATTRIBUTE_OPTIONS
        ):
            self._has_dataclass_arguments = True
            self._attribute_options = attribute_options
        else:
            self._has_dataclass_arguments = False
            self._attribute_options = _DEFAULT_ATTRIBUTE_OPTIONS

    @overload
    def __get__(
        self, instance: Literal[None], owner: Literal[None]
    ) -> Self: ...

    @overload
    def __get__(
        self, instance: Literal[None], owner: Any
    ) -> AssociationProxyInstance[_T]: ...

    @overload
    def __get__(self, instance: object, owner: Any) -> _T: ...

    def __get__(
        self, instance: object, owner: Any
    ) -> Union[AssociationProxyInstance[_T], _T, AssociationProxy[_T]]:
        if owner is None:
            return self
        inst = self._as_instance(owner, instance)
        if inst:
            return inst.get(instance)

        assert instance is None

        return self

    def __set__(self, instance: object, values: _T) -> None:
        class_ = type(instance)
        self._as_instance(class_, instance).set(instance, values)

    def __delete__(self, instance: object) -> None:
        class_ = type(instance)
        self._as_instance(class_, instance).delete(instance)

    def for_class(
        self, class_: Type[Any], obj: Optional[object] = None
    ) -> AssociationProxyInstance[_T]:
        r"""Return the internal state local to a specific mapped class.

        E.g., given a class ``User``::

            class User(Base):
                # ...

                keywords = association_proxy("kws", "keyword")

        If we access this :class:`.AssociationProxy` from
        :attr:`_orm.Mapper.all_orm_descriptors`, and we want to view the
        target class for this proxy as mapped by ``User``::

            inspect(User).all_orm_descriptors["keywords"].for_class(User).target_class

        This returns an instance of :class:`.AssociationProxyInstance` that
        is specific to the ``User`` class.   The :class:`.AssociationProxy`
        object remains agnostic of its parent class.

        :param class\_: the class that we are returning state for.

        :param obj: optional, an instance of the class that is required
         if the attribute refers to a polymorphic target, e.g. where we have
         to look at the type of the actual destination object to get the
         complete path.

        .. versionadded:: 1.3 - :class:`.AssociationProxy` no longer stores
           any state specific to a particular parent class; the state is now
           stored in per-class :class:`.AssociationProxyInstance` objects.


        """
        return self._as_instance(class_, obj)

    def _as_instance(
        self, class_: Any, obj: Any
    ) -> AssociationProxyInstance[_T]:
        try:
            inst = class_.__dict__[self.key + "_inst"]
        except KeyError:
            inst = None

        # avoid exception context
        if inst is None:
            owner = self._calc_owner(class_)
            if owner is not None:
                inst = AssociationProxyInstance.for_proxy(self, owner, obj)
                setattr(class_, self.key + "_inst", inst)
            else:
                inst = None

        if inst is not None and not inst._is_canonical:
            # the AssociationProxyInstance can't be generalized
            # since the proxied attribute is not on the targeted
            # class, only on subclasses of it, which might be
            # different.  only return for the specific
            # object's current value
            return inst._non_canonical_get_for_object(obj)  # type: ignore
        else:
            return inst  # type: ignore  # TODO

    def _calc_owner(self, target_cls: Any) -> Any:
        # we might be getting invoked for a subclass
        # that is not mapped yet, in some declarative situations.
        # save until we are mapped
        try:
            insp = inspect(target_cls)
        except exc.NoInspectionAvailable:
            # can't find a mapper, don't set owner. if we are a not-yet-mapped
            # subclass, we can also scan through __mro__ to find a mapped
            # class, but instead just wait for us to be called again against a
            # mapped class normally.
            return None
        else:
            return insp.mapper.class_manager.class_

    def _default_getset(
        self, collection_class: Any
    ) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
        attr = self.value_attr
        _getter = operator.attrgetter(attr)

        def getter(instance: Any) -> Optional[Any]:
            return _getter(instance) if instance is not None else None

        if collection_class is dict:

            def dict_setter(instance: Any, k: Any, value: Any) -> None:
                setattr(instance, attr, value)

            return getter, dict_setter

        else:

            def plain_setter(o: Any, v: Any) -> None:
                setattr(o, attr, v)

            return getter, plain_setter

    def __repr__(self) -> str:
        return "AssociationProxy(%r, %r)" % (
            self.target_collection,
            self.value_attr,
        )


# the pep-673 Self type does not work in Mypy for a "hybrid"
# style method that returns type or Self, so for one specific case
# we still need to use the pre-pep-673 workaround.
_Self = TypeVar("_Self", bound="AssociationProxyInstance[Any]")


class AssociationProxyInstance(SQLORMOperations[_T]):
    """A per-class object that serves class- and object-specific results.

    This is used by :class:`.AssociationProxy` when it is invoked
    in terms of a specific class or instance of a class, i.e. when it is
    used as a regular Python descriptor.

    When referring to the :class:`.AssociationProxy` as a normal Python
    descriptor, the :class:`.AssociationProxyInstance` is the object that
    actually serves the information.   Under normal circumstances, its presence
    is transparent::

        >>> User.keywords.scalar
        False

    In the special case that the :class:`.AssociationProxy` object is being
    accessed directly, in order to get an explicit handle to the
    :class:`.AssociationProxyInstance`, use the
    :meth:`.AssociationProxy.for_class` method::

        proxy_state = inspect(User).all_orm_descriptors["keywords"].for_class(User)

        # view if proxy object is scalar or not
        >>> proxy_state.scalar
        False

    .. versionadded:: 1.3

    """  # noqa

    collection_class: Optional[Type[Any]]
    parent: _AssociationProxyProtocol[_T]

    def __init__(
        self,
        parent: _AssociationProxyProtocol[_T],
        owning_class: Type[Any],
        target_class: Type[Any],
        value_attr: str,
    ):
        self.parent = parent
        self.key = parent.key
        self.owning_class = owning_class
        self.target_collection = parent.target_collection
        self.collection_class = None
        self.target_class = target_class
        self.value_attr = value_attr

    target_class: Type[Any]
    """The intermediary class handled by this
    :class:`.AssociationProxyInstance`.

    Intercepted append/set/assignment events will result
    in the generation of new instances of this class.

    """

    @classmethod
    def for_proxy(
        cls,
        parent: AssociationProxy[_T],
        owning_class: Type[Any],
        parent_instance: Any,
    ) -> AssociationProxyInstance[_T]:
        target_collection = parent.target_collection
        value_attr = parent.value_attr
        prop = cast(
            "orm.RelationshipProperty[_T]",
            orm.class_mapper(owning_class).get_property(target_collection),
        )

        # this was never asserted before but this should be made clear.
        if not isinstance(prop, orm.RelationshipProperty):
            raise NotImplementedError(
                "association proxy to a non-relationship "
                "intermediary is not supported"
            ) from None

        target_class = prop.mapper.class_

        try:
            target_assoc = cast(
                "AssociationProxyInstance[_T]",
                cls._cls_unwrap_target_assoc_proxy(target_class, value_attr),
            )
        except AttributeError:
            # the proxied attribute doesn't exist on the target class;
            # return an "ambiguous" instance that will work on a per-object
            # basis
            return AmbiguousAssociationProxyInstance(
                parent, owning_class, target_class, value_attr
            )
        except Exception as err:
            raise exc.InvalidRequestError(
                f"Association proxy received an unexpected error when "
                f"trying to retreive attribute "
                f'"{target_class.__name__}.{parent.value_attr}" from '
                f'class "{target_class.__name__}": {err}'
            ) from err
        else:
            return cls._construct_for_assoc(
                target_assoc, parent, owning_class, target_class, value_attr
            )

    @classmethod
    def _construct_for_assoc(
        cls,
        target_assoc: Optional[AssociationProxyInstance[_T]],
        parent: _AssociationProxyProtocol[_T],
        owning_class: Type[Any],
        target_class: Type[Any],
        value_attr: str,
    ) -> AssociationProxyInstance[_T]:
        if target_assoc is not None:
            return ObjectAssociationProxyInstance(
                parent, owning_class, target_class, value_attr
            )

        attr = getattr(target_class, value_attr)
        if not hasattr(attr, "_is_internal_proxy"):
            return AmbiguousAssociationProxyInstance(
                parent, owning_class, target_class, value_attr
            )
        is_object = attr._impl_uses_objects
        if is_object:
            return ObjectAssociationProxyInstance(
                parent, owning_class, target_class, value_attr
            )
        else:
            return ColumnAssociationProxyInstance(
                parent, owning_class, target_class, value_attr
            )

    def _get_property(self) -> MapperProperty[Any]:
        return orm.class_mapper(self.owning_class).get_property(
            self.target_collection
        )

    @property
    def _comparator(self) -> PropComparator[Any]:
        return getattr(  # type: ignore
            self.owning_class, self.target_collection
        ).comparator

    def __clause_element__(self) -> NoReturn:
        raise NotImplementedError(
            "The association proxy can't be used as a plain column "
            "expression; it only works inside of a comparison expression"
        )

    @classmethod
    def _cls_unwrap_target_assoc_proxy(
        cls, target_class: Any, value_attr: str
    ) -> Optional[AssociationProxyInstance[_T]]:
        attr = getattr(target_class, value_attr)
        assert not isinstance(attr, AssociationProxy)
        if isinstance(attr, AssociationProxyInstance):
            return attr
        return None

    @util.memoized_property
    def _unwrap_target_assoc_proxy(
        self,
    ) -> Optional[AssociationProxyInstance[_T]]:
        return self._cls_unwrap_target_assoc_proxy(
            self.target_class, self.value_attr
        )

    @property
    def remote_attr(self) -> SQLORMOperations[_T]:
        """The 'remote' class attribute referenced by this
        :class:`.AssociationProxyInstance`.

        .. seealso::

            :attr:`.AssociationProxyInstance.attr`

            :attr:`.AssociationProxyInstance.local_attr`

        """
        return cast(
            "SQLORMOperations[_T]", getattr(self.target_class, self.value_attr)
        )

    @property
    def local_attr(self) -> SQLORMOperations[Any]:
        """The 'local' class attribute referenced by this
        :class:`.AssociationProxyInstance`.

        .. seealso::

            :attr:`.AssociationProxyInstance.attr`

            :attr:`.AssociationProxyInstance.remote_attr`

        """
        return cast(
            "SQLORMOperations[Any]",
            getattr(self.owning_class, self.target_collection),
        )

    @property
    def attr(self) -> Tuple[SQLORMOperations[Any], SQLORMOperations[_T]]:
        """Return a tuple of ``(local_attr, remote_attr)``.

        This attribute was originally intended to facilitate using the
        :meth:`_query.Query.join` method to join across the two relationships
        at once, however this makes use of a deprecated calling style.

        To use :meth:`_sql.select.join` or :meth:`_orm.Query.join` with
        an association proxy, the current method is to make use of the
        :attr:`.AssociationProxyInstance.local_attr` and
        :attr:`.AssociationProxyInstance.remote_attr` attributes separately::

            stmt = (
                select(Parent)
                .join(Parent.proxied.local_attr)
                .join(Parent.proxied.remote_attr)
            )

        A future release may seek to provide a more succinct join pattern
        for association proxy attributes.

        .. seealso::

            :attr:`.AssociationProxyInstance.local_attr`

            :attr:`.AssociationProxyInstance.remote_attr`

        """
        return (self.local_attr, self.remote_attr)

    @util.memoized_property
    def scalar(self) -> bool:
        """Return ``True`` if this :class:`.AssociationProxyInstance`
        proxies a scalar relationship on the local side."""

        scalar = not self._get_property().uselist
        if scalar:
            self._initialize_scalar_accessors()
        return scalar

    @util.memoized_property
    def _value_is_scalar(self) -> bool:
        return (
            not self._get_property()
            .mapper.get_property(self.value_attr)
            .uselist
        )

    @property
    def _target_is_object(self) -> bool:
        raise NotImplementedError()

    _scalar_get: _GetterProtocol[_T]
    _scalar_set: _PlainSetterProtocol[_T]

    def _initialize_scalar_accessors(self) -> None:
        if self.parent.getset_factory:
            get, set_ = self.parent.getset_factory(None, self)
        else:
            get, set_ = self.parent._default_getset(None)
        self._scalar_get, self._scalar_set = get, cast(
            "_PlainSetterProtocol[_T]", set_
        )

    def _default_getset(
        self, collection_class: Any
    ) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
        attr = self.value_attr
        _getter = operator.attrgetter(attr)

        def getter(instance: Any) -> Optional[_T]:
            return _getter(instance) if instance is not None else None

        if collection_class is dict:

            def dict_setter(instance: Any, k: Any, value: _T) -> None:
                setattr(instance, attr, value)

            return getter, dict_setter
        else:

            def plain_setter(o: Any, v: _T) -> None:
                setattr(o, attr, v)

            return getter, plain_setter

    @util.ro_non_memoized_property
    def info(self) -> _InfoType:
        return self.parent.info

    @overload
    def get(self: _Self, obj: Literal[None]) -> _Self: ...

    @overload
    def get(self, obj: Any) -> _T: ...

    def get(
        self, obj: Any
    ) -> Union[Optional[_T], AssociationProxyInstance[_T]]:
        if obj is None:
            return self

        proxy: _T

        if self.scalar:
            target = getattr(obj, self.target_collection)
            return self._scalar_get(target)
        else:
            try:
                # If the owning instance is reborn (orm session resurrect,
                # etc.), refresh the proxy cache.
                creator_id, self_id, proxy = cast(
                    "Tuple[int, int, _T]", getattr(obj, self.key)
                )
            except AttributeError:
                pass
            else:
                if id(obj) == creator_id and id(self) == self_id:
                    assert self.collection_class is not None
                    return proxy

            self.collection_class, proxy = self._new(
                _lazy_collection(obj, self.target_collection)
            )
            setattr(obj, self.key, (id(obj), id(self), proxy))
            return proxy

    def set(self, obj: Any, values: _T) -> None:
        if self.scalar:
            creator = cast(
                "_PlainCreatorProtocol[_T]",
                (
                    self.parent.creator
                    if self.parent.creator
                    else self.target_class
                ),
            )
            target = getattr(obj, self.target_collection)
            if target is None:
                if (
                    values is None
                    and not self.parent.create_on_none_assignment
                ):
                    return
                setattr(obj, self.target_collection, creator(values))
            else:
                self._scalar_set(target, values)
                if values is None and self.parent.cascade_scalar_deletes:
                    setattr(obj, self.target_collection, None)
        else:
            proxy = self.get(obj)
            assert self.collection_class is not None
            if proxy is not values:
                proxy._bulk_replace(self, values)

    def delete(self, obj: Any) -> None:
        if self.owning_class is None:
            self._calc_owner(obj, None)

        if self.scalar:
            target = getattr(obj, self.target_collection)
            if target is not None:
                delattr(target, self.value_attr)
        delattr(obj, self.target_collection)

    def _new(
        self, lazy_collection: _LazyCollectionProtocol[_T]
    ) -> Tuple[Type[Any], _T]:
        creator = (
            self.parent.creator
            if self.parent.creator is not None
            else cast("_CreatorProtocol", self.target_class)
        )
        collection_class = util.duck_type_collection(lazy_collection())

        if collection_class is None:
            raise exc.InvalidRequestError(
                f"lazy collection factory did not return a "
                f"valid collection type, got {collection_class}"
            )
        if self.parent.proxy_factory:
            return (
                collection_class,
                self.parent.proxy_factory(
                    lazy_collection, creator, self.value_attr, self
                ),
            )

        if self.parent.getset_factory:
            getter, setter = self.parent.getset_factory(collection_class, self)
        else:
            getter, setter = self.parent._default_getset(collection_class)

        if collection_class is list:
            return (
                collection_class,
                cast(
                    _T,
                    _AssociationList(
                        lazy_collection, creator, getter, setter, self
                    ),
                ),
            )
        elif collection_class is dict:
            return (
                collection_class,
                cast(
                    _T,
                    _AssociationDict(
                        lazy_collection, creator, getter, setter, self
                    ),
                ),
            )
        elif collection_class is set:
            return (
                collection_class,
                cast(
                    _T,
                    _AssociationSet(
                        lazy_collection, creator, getter, setter, self
                    ),
                ),
            )
        else:
            raise exc.ArgumentError(
                "could not guess which interface to use for "
                'collection_class "%s" backing "%s"; specify a '
                "proxy_factory and proxy_bulk_set manually"
                % (self.collection_class, self.target_collection)
            )

    def _set(
        self, proxy: _AssociationCollection[Any], values: Iterable[Any]
    ) -> None:
        if self.parent.proxy_bulk_set:
            self.parent.proxy_bulk_set(proxy, values)
        elif self.collection_class is list:
            cast("_AssociationList[Any]", proxy).extend(values)
        elif self.collection_class is dict:
            cast("_AssociationDict[Any, Any]", proxy).update(values)
        elif self.collection_class is set:
            cast("_AssociationSet[Any]", proxy).update(values)
        else:
            raise exc.ArgumentError(
                "no proxy_bulk_set supplied for custom "
                "collection_class implementation"
            )

    def _inflate(self, proxy: _AssociationCollection[Any]) -> None:
        creator = (
            self.parent.creator
            and self.parent.creator
            or cast(_CreatorProtocol, self.target_class)
        )

        if self.parent.getset_factory:
            getter, setter = self.parent.getset_factory(
                self.collection_class, self
            )
        else:
            getter, setter = self.parent._default_getset(self.collection_class)

        proxy.creator = creator
        proxy.getter = getter
        proxy.setter = setter

    def _criterion_exists(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> ColumnElement[bool]:
        is_has = kwargs.pop("is_has", None)

        target_assoc = self._unwrap_target_assoc_proxy
        if target_assoc is not None:
            inner = target_assoc._criterion_exists(
                criterion=criterion, **kwargs
            )
            return self._comparator._criterion_exists(inner)

        if self._target_is_object:
            attr = getattr(self.target_class, self.value_attr)
            value_expr = attr.comparator._criterion_exists(criterion, **kwargs)
        else:
            if kwargs:
                raise exc.ArgumentError(
                    "Can't apply keyword arguments to column-targeted "
                    "association proxy; use =="
                )
            elif is_has and criterion is not None:
                raise exc.ArgumentError(
                    "Non-empty has() not allowed for "
                    "column-targeted association proxy; use =="
                )

            value_expr = criterion

        return self._comparator._criterion_exists(value_expr)

    def any(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> ColumnElement[bool]:
        """Produce a proxied 'any' expression using EXISTS.

        This expression will be a composed product
        using the :meth:`.Relationship.Comparator.any`
        and/or :meth:`.Relationship.Comparator.has`
        operators of the underlying proxied attributes.

        """
        if self._unwrap_target_assoc_proxy is None and (
            self.scalar
            and (not self._target_is_object or self._value_is_scalar)
        ):
            raise exc.InvalidRequestError(
                "'any()' not implemented for scalar attributes. Use has()."
            )
        return self._criterion_exists(
            criterion=criterion, is_has=False, **kwargs
        )

    def has(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> ColumnElement[bool]:
        """Produce a proxied 'has' expression using EXISTS.

        This expression will be a composed product
        using the :meth:`.Relationship.Comparator.any`
        and/or :meth:`.Relationship.Comparator.has`
        operators of the underlying proxied attributes.

        """
        if self._unwrap_target_assoc_proxy is None and (
            not self.scalar
            or (self._target_is_object and not self._value_is_scalar)
        ):
            raise exc.InvalidRequestError(
                "'has()' not implemented for collections. Use any()."
            )
        return self._criterion_exists(
            criterion=criterion, is_has=True, **kwargs
        )

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, self.parent)


class AmbiguousAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` where we cannot determine
    the type of target object.
    """

    _is_canonical = False

    def _ambiguous(self) -> NoReturn:
        raise AttributeError(
            "Association proxy %s.%s refers to an attribute '%s' that is not "
            "directly mapped on class %s; therefore this operation cannot "
            "proceed since we don't know what type of object is referred "
            "towards"
            % (
                self.owning_class.__name__,
                self.target_collection,
                self.value_attr,
                self.target_class,
            )
        )

    def get(self, obj: Any) -> Any:
        if obj is None:
            return self
        else:
            return super().get(obj)

    def __eq__(self, obj: object) -> NoReturn:
        self._ambiguous()

    def __ne__(self, obj: object) -> NoReturn:
        self._ambiguous()

    def any(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        self._ambiguous()

    def has(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        self._ambiguous()

    @util.memoized_property
    def _lookup_cache(self) -> Dict[Type[Any], AssociationProxyInstance[_T]]:
        # mapping of <subclass>->AssociationProxyInstance.
        # e.g. proxy is A-> A.b -> B -> B.b_attr, but B.b_attr doesn't exist;
        # only B1(B) and B2(B) have "b_attr", keys in here would be B1, B2
        return {}

    def _non_canonical_get_for_object(
        self, parent_instance: Any
    ) -> AssociationProxyInstance[_T]:
        if parent_instance is not None:
            actual_obj = getattr(parent_instance, self.target_collection)
            if actual_obj is not None:
                try:
                    insp = inspect(actual_obj)
                except exc.NoInspectionAvailable:
                    pass
                else:
                    mapper = insp.mapper
                    instance_class = mapper.class_
                    if instance_class not in self._lookup_cache:
                        self._populate_cache(instance_class, mapper)

                    try:
                        return self._lookup_cache[instance_class]
                    except KeyError:
                        pass

        # no object or ambiguous object given, so return "self", which
        # is a proxy with generally only instance-level functionality
        return self

    def _populate_cache(
        self, instance_class: Any, mapper: Mapper[Any]
    ) -> None:
        prop = orm.class_mapper(self.owning_class).get_property(
            self.target_collection
        )

        if mapper.isa(prop.mapper):
            target_class = instance_class
            try:
                target_assoc = self._cls_unwrap_target_assoc_proxy(
                    target_class, self.value_attr
                )
            except AttributeError:
                pass
            else:
                self._lookup_cache[instance_class] = self._construct_for_assoc(
                    cast("AssociationProxyInstance[_T]", target_assoc),
                    self.parent,
                    self.owning_class,
                    target_class,
                    self.value_attr,
                )


class ObjectAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` that has an object as a target."""

    _target_is_object: bool = True
    _is_canonical = True

    def contains(self, other: Any, **kw: Any) -> ColumnElement[bool]:
        """Produce a proxied 'contains' expression using EXISTS.

        This expression will be a composed product
        using the :meth:`.Relationship.Comparator.any`,
        :meth:`.Relationship.Comparator.has`,
        and/or :meth:`.Relationship.Comparator.contains`
        operators of the underlying proxied attributes.
        """

        target_assoc = self._unwrap_target_assoc_proxy
        if target_assoc is not None:
            return self._comparator._criterion_exists(
                target_assoc.contains(other)
                if not target_assoc.scalar
                else target_assoc == other
            )
        elif (
            self._target_is_object
            and self.scalar
            and not self._value_is_scalar
        ):
            return self._comparator.has(
                getattr(self.target_class, self.value_attr).contains(other)
            )
        elif self._target_is_object and self.scalar and self._value_is_scalar:
            raise exc.InvalidRequestError(
                "contains() doesn't apply to a scalar object endpoint; use =="
            )
        else:
            return self._comparator._criterion_exists(
                **{self.value_attr: other}
            )

    def __eq__(self, obj: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
        # note the has() here will fail for collections; eq_()
        # is only allowed with a scalar.
        if obj is None:
            return or_(
                self._comparator.has(**{self.value_attr: obj}),
                self._comparator == None,
            )
        else:
            return self._comparator.has(**{self.value_attr: obj})

    def __ne__(self, obj: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
        # note the has() here will fail for collections; eq_()
        # is only allowed with a scalar.
        return self._comparator.has(
            getattr(self.target_class, self.value_attr) != obj
        )


class ColumnAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` that has a database column as a
    target.
    """

    _target_is_object: bool = False
    _is_canonical = True

    def __eq__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
        # special case "is None" to check for no related row as well
        expr = self._criterion_exists(
            self.remote_attr.operate(operators.eq, other)
        )
        if other is None:
            return or_(expr, self._comparator == None)
        else:
            return expr

    def operate(
        self, op: operators.OperatorType, *other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        return self._criterion_exists(
            self.remote_attr.operate(op, *other, **kwargs)
        )


class _lazy_collection(_LazyCollectionProtocol[_T]):
    def __init__(self, obj: Any, target: str):
        self.parent = obj
        self.target = target

    def __call__(
        self,
    ) -> Union[MutableSet[_T], MutableMapping[Any, _T], MutableSequence[_T]]:
        return getattr(self.parent, self.target)  # type: ignore[no-any-return]

    def __getstate__(self) -> Any:
        return {"obj": self.parent, "target": self.target}

    def __setstate__(self, state: Any) -> None:
        self.parent = state["obj"]
        self.target = state["target"]


_IT = TypeVar("_IT", bound="Any")
"""instance type - this is the type of object inside a collection.

this is not the same as the _T of AssociationProxy and
AssociationProxyInstance itself, which will often refer to the
collection[_IT] type.

"""


class _AssociationCollection(Generic[_IT]):
    getter: _GetterProtocol[_IT]
    """A function.  Given an associated object, return the 'value'."""

    creator: _CreatorProtocol
    """
    A function that creates new target entities.  Given one parameter:
    value.  This assertion is assumed::

    obj = creator(somevalue)
    assert getter(obj) == somevalue
    """

    parent: AssociationProxyInstance[_IT]
    setter: _SetterProtocol
    """A function.  Given an associated object and a value, store that
        value on the object.
    """

    lazy_collection: _LazyCollectionProtocol[_IT]
    """A callable returning a list-based collection of entities (usually an
          object attribute managed by a SQLAlchemy relationship())"""

    def __init__(
        self,
        lazy_collection: _LazyCollectionProtocol[_IT],
        creator: _CreatorProtocol,
        getter: _GetterProtocol[_IT],
        setter: _SetterProtocol,
        parent: AssociationProxyInstance[_IT],
    ):
        """Constructs an _AssociationCollection.

        This will always be a subclass of either _AssociationList,
        _AssociationSet, or _AssociationDict.

        """
        self.lazy_collection = lazy_collection
        self.creator = creator
        self.getter = getter
        self.setter = setter
        self.parent = parent

    if typing.TYPE_CHECKING:
        col: Collection[_IT]
    else:
        col = property(lambda self: self.lazy_collection())

    def __len__(self) -> int:
        return len(self.col)

    def __bool__(self) -> bool:
        return bool(self.col)

    def __getstate__(self) -> Any:
        return {"parent": self.parent, "lazy_collection": self.lazy_collection}

    def __setstate__(self, state: Any) -> None:
        self.parent = state["parent"]
        self.lazy_collection = state["lazy_collection"]
        self.parent._inflate(self)

    def clear(self) -> None:
        raise NotImplementedError()


class _AssociationSingleItem(_AssociationCollection[_T]):
    setter: _PlainSetterProtocol[_T]
    creator: _PlainCreatorProtocol[_T]

    def _create(self, value: _T) -> Any:
        return self.creator(value)

    def _get(self, object_: Any) -> _T:
        return self.getter(object_)

    def _bulk_replace(
        self, assoc_proxy: AssociationProxyInstance[Any], values: Iterable[_IT]
    ) -> None:
        self.clear()
        assoc_proxy._set(self, values)


class _AssociationList(_AssociationSingleItem[_T], MutableSequence[_T]):
    """Generic, converting, list-to-list proxy."""

    col: MutableSequence[_T]

    def _set(self, object_: Any, value: _T) -> None:
        self.setter(object_, value)

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[_T]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[_T, MutableSequence[_T]]:
        if not isinstance(index, slice):
            return self._get(self.col[index])
        else:
            return [self._get(member) for member in self.col[index]]

    @overload
    def __setitem__(self, index: int, value: _T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[_T]) -> None: ...

    def __setitem__(
        self, index: Union[int, slice], value: Union[_T, Iterable[_T]]
    ) -> None:
        if not isinstance(index, slice):
            self._set(self.col[index], cast("_T", value))
        else:
            if index.stop is None:
                stop = len(self)
            elif index.stop < 0:
                stop = len(self) + index.stop
            else:
                stop = index.stop
            step = index.step or 1

            start = index.start or 0
            rng = list(range(index.start or 0, stop, step))

            sized_value = list(value)

            if step == 1:
                for i in rng:
                    del self[start]
                i = start
                for item in sized_value:
                    self.insert(i, item)
                    i += 1
            else:
                if len(sized_value) != len(rng):
                    raise ValueError(
                        "attempt to assign sequence of size %s to "
                        "extended slice of size %s"
                        % (len(sized_value), len(rng))
                    )
                for i, item in zip(rng, value):
                    self._set(self.col[i], item)

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: Union[slice, int]) -> None:
        del self.col[index]

    def __contains__(self, value: object) -> bool:
        for member in self.col:
            # testlib.pragma exempt:__eq__
            if self._get(member) == value:
                return True
        return False

    def __iter__(self) -> Iterator[_T]:
        """Iterate over proxied values.

        For the actual domain objects, iterate over .col instead or
        just use the underlying collection directly from its property
        on the parent.
        """

        for member in self.col:
            yield self._get(member)
        return

    def append(self, value: _T) -> None:
        col = self.col
        item = self._create(value)
        col.append(item)

    def count(self, value: Any) -> int:
        count = 0
        for v in self:
            if v == value:
                count += 1
        return count

    def extend(self, values: Iterable[_T]) -> None:
        for v in values:
            self.append(v)

    def insert(self, index: int, value: _T) -> None:
        self.col[index:index] = [self._create(value)]

    def pop(self, index: int = -1) -> _T:
        return self.getter(self.col.pop(index))

    def remove(self, value: _T) -> None:
        for i, val in enumerate(self):
            if val == value:
                del self.col[i]
                return
        raise ValueError("value not in list")

    def reverse(self) -> NoReturn:
        """Not supported, use reversed(mylist)"""

        raise NotImplementedError()

    def sort(self) -> NoReturn:
        """Not supported, use sorted(mylist)"""

        raise NotImplementedError()

    def clear(self) -> None:
        del self.col[0 : len(self.col)]

    def __eq__(self, other: object) -> bool:
        return list(self) == other

    def __ne__(self, other: object) -> bool:
        return list(self) != other

    def __lt__(self, other: List[_T]) -> bool:
        return list(self) < other

    def __le__(self, other: List[_T]) -> bool:
        return list(self) <= other

    def __gt__(self, other: List[_T]) -> bool:
        return list(self) > other

    def __ge__(self, other: List[_T]) -> bool:
        return list(self) >= other

    def __add__(self, other: List[_T]) -> List[_T]:
        try:
            other = list(other)
        except TypeError:
            return NotImplemented
        return list(self) + other

    def __radd__(self, other: List[_T]) -> List[_T]:
        try:
            other = list(other)
        except TypeError:
            return NotImplemented
        return other + list(self)

    def __mul__(self, n: SupportsIndex) -> List[_T]:
        if not isinstance(n, int):
            return NotImplemented
        return list(self) * n

    def __rmul__(self, n: SupportsIndex) -> List[_T]:
        if not isinstance(n, int):
            return NotImplemented
        return n * list(self)

    def __iadd__(self, iterable: Iterable[_T]) -> Self:
        self.extend(iterable)
        return self

    def __imul__(self, n: SupportsIndex) -> Self:
        # unlike a regular list *=, proxied __imul__ will generate unique
        # backing objects for each copy.  *= on proxied lists is a bit of
        # a stretch anyhow, and this interpretation of the __imul__ contract
        # is more plausibly useful than copying the backing objects.
        if not isinstance(n, int):
            raise NotImplementedError()
        if n == 0:
            self.clear()
        elif n > 1:
            self.extend(list(self) * (n - 1))
        return self

    if typing.TYPE_CHECKING:
        # TODO: no idea how to do this without separate "stub"
        def index(
            self, value: Any, start: int = ..., stop: int = ...
        ) -> int: ...

    else:

        def index(self, value: Any, *arg) -> int:
            ls = list(self)
            return ls.index(value, *arg)

    def copy(self) -> List[_T]:
        return list(self)

    def __repr__(self) -> str:
        return repr(list(self))

    def __hash__(self) -> NoReturn:
        raise TypeError("%s objects are unhashable" % type(self).__name__)

    if not typing.TYPE_CHECKING:
        for func_name, func in list(locals().items()):
            if (
                callable(func)
                and func.__name__ == func_name
                and not func.__doc__
                and hasattr(list, func_name)
            ):
                func.__doc__ = getattr(list, func_name).__doc__
        del func_name, func


class _AssociationDict(_AssociationCollection[_VT], MutableMapping[_KT, _VT]):
    """Generic, converting, dict-to-dict proxy."""

    setter: _DictSetterProtocol[_VT]
    creator: _KeyCreatorProtocol[_VT]
    col: MutableMapping[_KT, Optional[_VT]]

    def _create(self, key: _KT, value: Optional[_VT]) -> Any:
        return self.creator(key, value)

    def _get(self, object_: Any) -> _VT:
        return self.getter(object_)

    def _set(self, object_: Any, key: _KT, value: _VT) -> None:
        return self.setter(object_, key, value)

    def __getitem__(self, key: _KT) -> _VT:
        return self._get(self.col[key])

    def __setitem__(self, key: _KT, value: _VT) -> None:
        if key in self.col:
            self._set(self.col[key], key, value)
        else:
            self.col[key] = self._create(key, value)

    def __delitem__(self, key: _KT) -> None:
        del self.col[key]

    def __contains__(self, key: object) -> bool:
        return key in self.col

    def __iter__(self) -> Iterator[_KT]:
        return iter(self.col.keys())

    def clear(self) -> None:
        self.col.clear()

    def __eq__(self, other: object) -> bool:
        return dict(self) == other

    def __ne__(self, other: object) -> bool:
        return dict(self) != other

    def __repr__(self) -> str:
        return repr(dict(self))

    @overload
    def get(self, __key: _KT) -> Optional[_VT]: ...

    @overload
    def get(self, __key: _KT, default: Union[_VT, _T]) -> Union[_VT, _T]: ...

    def get(
        self, key: _KT, default: Optional[Union[_VT, _T]] = None
    ) -> Union[_VT, _T, None]:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: _KT, default: Optional[_VT] = None) -> _VT:
        # TODO: again, no idea how to create an actual MutableMapping.
        # default must allow None, return type can't include None,
        # the stub explicitly allows for default of None with a cryptic message
        # "This overload should be allowed only if the value type is
        # compatible with None.".
        if key not in self.col:
            self.col[key] = self._create(key, default)
            return default  # type: ignore
        else:
            return self[key]

    def keys(self) -> KeysView[_KT]:
        return self.col.keys()

    def items(self) -> ItemsView[_KT, _VT]:
        return ItemsView(self)

    def values(self) -> ValuesView[_VT]:
        return ValuesView(self)

    @overload
    def pop(self, __key: _KT) -> _VT: ...

    @overload
    def pop(
        self, __key: _KT, default: Union[_VT, _T] = ...
    ) -> Union[_VT, _T]: ...

    def pop(self, __key: _KT, *arg: Any, **kw: Any) -> Union[_VT, _T]:
        member = self.col.pop(__key, *arg, **kw)
        return self._get(member)

    def popitem(self) -> Tuple[_KT, _VT]:
        item = self.col.popitem()
        return (item[0], self._get(item[1]))

    @overload
    def update(
        self, __m: SupportsKeysAndGetItem[_KT, _VT], **kwargs: _VT
    ) -> None: ...

    @overload
    def update(
        self, __m: Iterable[tuple[_KT, _VT]], **kwargs: _VT
    ) -> None: ...

    @overload
    def update(self, **kwargs: _VT) -> None: ...

    def update(self, *a: Any, **kw: Any) -> None:
        up: Dict[_KT, _VT] = {}
        up.update(*a, **kw)

        for key, value in up.items():
            self[key] = value

    def _bulk_replace(
        self,
        assoc_proxy: AssociationProxyInstance[Any],
        values: Mapping[_KT, _VT],
    ) -> None:
        existing = set(self)
        constants = existing.intersection(values or ())
        additions = set(values or ()).difference(constants)
        removals = existing.difference(constants)

        for key, member in values.items() or ():
            if key in additions:
                self[key] = member
            elif key in constants:
                self[key] = member

        for key in removals:
            del self[key]

    def copy(self) -> Dict[_KT, _VT]:
        return dict(self.items())

    def __hash__(self) -> NoReturn:
        raise TypeError("%s objects are unhashable" % type(self).__name__)

    if not typing.TYPE_CHECKING:
        for func_name, func in list(locals().items()):
            if (
                callable(func)
                and func.__name__ == func_name
                and not func.__doc__
                and hasattr(dict, func_name)
            ):
                func.__doc__ = getattr(dict, func_name).__doc__
        del func_name, func


class _AssociationSet(_AssociationSingleItem[_T], MutableSet[_T]):
    """Generic, converting, set-to-set proxy."""

    col: MutableSet[_T]

    def __len__(self) -> int:
        return len(self.col)

    def __bool__(self) -> bool:
        if self.col:
            return True
        else:
            return False

    def __contains__(self, __o: object) -> bool:
        for member in self.col:
            if self._get(member) == __o:
                return True
        return False

    def __iter__(self) -> Iterator[_T]:
        """Iterate over proxied values.

        For the actual domain objects, iterate over .col instead or just use
        the underlying collection directly from its property on the parent.

        """
        for member in self.col:
            yield self._get(member)
        return

    def add(self, __element: _T) -> None:
        if __element not in self:
            self.col.add(self._create(__element))

    # for discard and remove, choosing a more expensive check strategy rather
    # than call self.creator()
    def discard(self, __element: _T) -> None:
        for member in self.col:
            if self._get(member) == __element:
                self.col.discard(member)
                break

    def remove(self, __element: _T) -> None:
        for member in self.col:
            if self._get(member) == __element:
                self.col.discard(member)
                return
        raise KeyError(__element)

    def pop(self) -> _T:
        if not self.col:
            raise KeyError("pop from an empty set")
        member = self.col.pop()
        return self._get(member)

    def update(self, *s: Iterable[_T]) -> None:
        for iterable in s:
            for value in iterable:
                self.add(value)

    def _bulk_replace(self, assoc_proxy: Any, values: Iterable[_T]) -> None:
        existing = set(self)
        constants = existing.intersection(values or ())
        additions = set(values or ()).difference(constants)
        removals = existing.difference(constants)

        appender = self.add
        remover = self.remove

        for member in values or ():
            if member in additions:
                appender(member)
            elif member in constants:
                appender(member)

        for member in removals:
            remover(member)

    def __ior__(  # type: ignore
        self, other: AbstractSet[_S]
    ) -> MutableSet[Union[_T, _S]]:
        if not collections._set_binops_check_strict(self, other):
            raise NotImplementedError()
        for value in other:
            self.add(value)
        return self

    def _set(self) -> Set[_T]:
        return set(iter(self))

    def union(self, *s: Iterable[_S]) -> MutableSet[Union[_T, _S]]:
        return set(self).union(*s)

    def __or__(self, __s: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        return self.union(__s)

    def difference(self, *s: Iterable[Any]) -> MutableSet[_T]:
        return set(self).difference(*s)

    def __sub__(self, s: AbstractSet[Any]) -> MutableSet[_T]:
        return self.difference(s)

    def difference_update(self, *s: Iterable[Any]) -> None:
        for other in s:
            for value in other:
                self.discard(value)

    def __isub__(self, s: AbstractSet[Any]) -> Self:
        if not collections._set_binops_check_strict(self, s):
            raise NotImplementedError()
        for value in s:
            self.discard(value)
        return self

    def intersection(self, *s: Iterable[Any]) -> MutableSet[_T]:
        return set(self).intersection(*s)

    def __and__(self, s: AbstractSet[Any]) -> MutableSet[_T]:
        return self.intersection(s)

    def intersection_update(self, *s: Iterable[Any]) -> None:
        for other in s:
            want, have = self.intersection(other), set(self)

            remove, add = have - want, want - have

            for value in remove:
                self.remove(value)
            for value in add:
                self.add(value)

    def __iand__(self, s: AbstractSet[Any]) -> Self:
        if not collections._set_binops_check_strict(self, s):
            raise NotImplementedError()
        want = self.intersection(s)
        have: Set[_T] = set(self)

        remove, add = have - want, want - have

        for value in remove:
            self.remove(value)
        for value in add:
            self.add(value)
        return self

    def symmetric_difference(self, __s: Iterable[_T]) -> MutableSet[_T]:
        return set(self).symmetric_difference(__s)

    def __xor__(self, s: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        return self.symmetric_difference(s)

    def symmetric_difference_update(self, other: Iterable[Any]) -> None:
        want, have = self.symmetric_difference(other), set(self)

        remove, add = have - want, want - have

        for value in remove:
            self.remove(value)
        for value in add:
            self.add(value)

    def __ixor__(self, other: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:  # type: ignore  # noqa: E501
        if not collections._set_binops_check_strict(self, other):
            raise NotImplementedError()

        self.symmetric_difference_update(other)
        return self

    def issubset(self, __s: Iterable[Any]) -> bool:
        return set(self).issubset(__s)

    def issuperset(self, __s: Iterable[Any]) -> bool:
        return set(self).issuperset(__s)

    def clear(self) -> None:
        self.col.clear()

    def copy(self) -> AbstractSet[_T]:
        return set(self)

    def __eq__(self, other: object) -> bool:
        return set(self) == other

    def __ne__(self, other: object) -> bool:
        return set(self) != other

    def __lt__(self, other: AbstractSet[Any]) -> bool:
        return set(self) < other

    def __le__(self, other: AbstractSet[Any]) -> bool:
        return set(self) <= other

    def __gt__(self, other: AbstractSet[Any]) -> bool:
        return set(self) > other

    def __ge__(self, other: AbstractSet[Any]) -> bool:
        return set(self) >= other

    def __repr__(self) -> str:
        return repr(set(self))

    def __hash__(self) -> NoReturn:
        raise TypeError("%s objects are unhashable" % type(self).__name__)

    if not typing.TYPE_CHECKING:
        for func_name, func in list(locals().items()):
            if (
                callable(func)
                and func.__name__ == func_name
                and not func.__doc__
                and hasattr(set, func_name)
            ):
                func.__doc__ = getattr(set, func_name).__doc__
        del func_name, func

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_select.py ===
# testing/suite/test_select.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

import collections.abc as collections_abc
import itertools

from .. import AssertsCompiledSQL
from .. import AssertsExecutionResults
from .. import config
from .. import fixtures
from ..assertions import assert_raises
from ..assertions import eq_
from ..assertions import in_
from ..assertsql import CursorSQL
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import case
from ... import column
from ... import Computed
from ... import exists
from ... import false
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import null
from ... import select
from ... import String
from ... import table
from ... import testing
from ... import text
from ... import true
from ... import tuple_
from ... import TupleType
from ... import union
from ... import values
from ...exc import DatabaseError
from ...exc import ProgrammingError


class CollateTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(100)),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "data": "collate data1"},
                {"id": 2, "data": "collate data2"},
            ],
        )

    def _assert_result(self, select, result):
        with config.db.connect() as conn:
            eq_(conn.execute(select).fetchall(), result)

    @testing.requires.order_by_collation
    def test_collate_order_by(self):
        collation = testing.requires.get_order_by_collation(testing.config)

        self._assert_result(
            select(self.tables.some_table).order_by(
                self.tables.some_table.c.data.collate(collation).asc()
            ),
            [(1, "collate data1"), (2, "collate data2")],
        )


class OrderByLabelTest(fixtures.TablesTest):
    """Test the dialect sends appropriate ORDER BY expressions when
    labels are used.

    This essentially exercises the "supports_simple_order_by_label"
    setting.

    """

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x", Integer),
            Column("y", Integer),
            Column("q", String(50)),
            Column("p", String(50)),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "x": 1, "y": 2, "q": "q1", "p": "p3"},
                {"id": 2, "x": 2, "y": 3, "q": "q2", "p": "p2"},
                {"id": 3, "x": 3, "y": 4, "q": "q3", "p": "p1"},
            ],
        )

    def _assert_result(self, select, result):
        with config.db.connect() as conn:
            eq_(conn.execute(select).fetchall(), result)

    def test_plain(self):
        table = self.tables.some_table
        lx = table.c.x.label("lx")
        self._assert_result(select(lx).order_by(lx), [(1,), (2,), (3,)])

    def test_composed_int(self):
        table = self.tables.some_table
        lx = (table.c.x + table.c.y).label("lx")
        self._assert_result(select(lx).order_by(lx), [(3,), (5,), (7,)])

    def test_composed_multiple(self):
        table = self.tables.some_table
        lx = (table.c.x + table.c.y).label("lx")
        ly = (func.lower(table.c.q) + table.c.p).label("ly")
        self._assert_result(
            select(lx, ly).order_by(lx, ly.desc()),
            [(3, "q1p3"), (5, "q2p2"), (7, "q3p1")],
        )

    def test_plain_desc(self):
        table = self.tables.some_table
        lx = table.c.x.label("lx")
        self._assert_result(select(lx).order_by(lx.desc()), [(3,), (2,), (1,)])

    def test_composed_int_desc(self):
        table = self.tables.some_table
        lx = (table.c.x + table.c.y).label("lx")
        self._assert_result(select(lx).order_by(lx.desc()), [(7,), (5,), (3,)])

    @testing.requires.group_by_complex_expression
    def test_group_by_composed(self):
        table = self.tables.some_table
        expr = (table.c.x + table.c.y).label("lx")
        stmt = (
            select(func.count(table.c.id), expr).group_by(expr).order_by(expr)
        )
        self._assert_result(stmt, [(1, 3), (1, 5), (1, 7)])


class ValuesExpressionTest(fixtures.TestBase):
    __requires__ = ("table_value_constructor",)

    __backend__ = True

    def test_tuples(self, connection):
        value_expr = values(
            column("id", Integer), column("name", String), name="my_values"
        ).data([(1, "name1"), (2, "name2"), (3, "name3")])

        eq_(
            connection.execute(select(value_expr)).all(),
            [(1, "name1"), (2, "name2"), (3, "name3")],
        )


class FetchLimitOffsetTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x", Integer),
            Column("y", Integer),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "x": 1, "y": 2},
                {"id": 2, "x": 2, "y": 3},
                {"id": 3, "x": 3, "y": 4},
                {"id": 4, "x": 4, "y": 5},
                {"id": 5, "x": 4, "y": 6},
            ],
        )

    def _assert_result(
        self, connection, select, result, params=(), set_=False
    ):
        if set_:
            query_res = connection.execute(select, params).fetchall()
            eq_(len(query_res), len(result))
            eq_(set(query_res), set(result))

        else:
            eq_(connection.execute(select, params).fetchall(), result)

    def _assert_result_str(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.exec_driver_sql(select, params).fetchall(), result)

    def test_simple_limit(self, connection):
        table = self.tables.some_table
        stmt = select(table).order_by(table.c.id)
        self._assert_result(
            connection,
            stmt.limit(2),
            [(1, 1, 2), (2, 2, 3)],
        )
        self._assert_result(
            connection,
            stmt.limit(3),
            [(1, 1, 2), (2, 2, 3), (3, 3, 4)],
        )

    def test_limit_render_multiple_times(self, connection):
        table = self.tables.some_table
        stmt = select(table.c.id).limit(1).scalar_subquery()

        u = union(select(stmt), select(stmt)).subquery().select()

        self._assert_result(
            connection,
            u,
            [
                (1,),
            ],
        )

    @testing.requires.fetch_first
    def test_simple_fetch(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).fetch(2),
            [(1, 1, 2), (2, 2, 3)],
        )
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).fetch(3),
            [(1, 1, 2), (2, 2, 3), (3, 3, 4)],
        )

    @testing.requires.offset
    def test_simple_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).offset(2),
            [(3, 3, 4), (4, 4, 5), (5, 4, 6)],
        )
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).offset(3),
            [(4, 4, 5), (5, 4, 6)],
        )

    @testing.combinations(
        ([(2, 0), (2, 1), (3, 2)]),
        ([(2, 1), (2, 0), (3, 2)]),
        ([(3, 1), (2, 1), (3, 1)]),
        argnames="cases",
    )
    @testing.requires.offset
    def test_simple_limit_offset(self, connection, cases):
        table = self.tables.some_table
        connection = connection.execution_options(compiled_cache={})

        assert_data = [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)]

        for limit, offset in cases:
            expected = assert_data[offset : offset + limit]
            self._assert_result(
                connection,
                select(table).order_by(table.c.id).limit(limit).offset(offset),
                expected,
            )

    @testing.requires.fetch_first
    def test_simple_fetch_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).fetch(2).offset(1),
            [(2, 2, 3), (3, 3, 4)],
        )

        self._assert_result(
            connection,
            select(table).order_by(table.c.id).fetch(3).offset(2),
            [(3, 3, 4), (4, 4, 5), (5, 4, 6)],
        )

    @testing.requires.fetch_no_order_by
    def test_fetch_offset_no_order(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).fetch(10),
            [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)],
            set_=True,
        )

    @testing.requires.offset
    def test_simple_offset_zero(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).offset(0),
            [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)],
        )

        self._assert_result(
            connection,
            select(table).order_by(table.c.id).offset(1),
            [(2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)],
        )

    @testing.requires.offset
    def test_limit_offset_nobinds(self):
        """test that 'literal binds' mode works - no bound params."""

        table = self.tables.some_table
        stmt = select(table).order_by(table.c.id).limit(2).offset(1)
        sql = stmt.compile(
            dialect=config.db.dialect, compile_kwargs={"literal_binds": True}
        )
        sql = str(sql)

        self._assert_result_str(sql, [(2, 2, 3), (3, 3, 4)])

    @testing.requires.fetch_first
    def test_fetch_offset_nobinds(self):
        """test that 'literal binds' mode works - no bound params."""

        table = self.tables.some_table
        stmt = select(table).order_by(table.c.id).fetch(2).offset(1)
        sql = stmt.compile(
            dialect=config.db.dialect, compile_kwargs={"literal_binds": True}
        )
        sql = str(sql)

        self._assert_result_str(sql, [(2, 2, 3), (3, 3, 4)])

    @testing.requires.bound_limit_offset
    def test_bound_limit(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).limit(bindparam("l")),
            [(1, 1, 2), (2, 2, 3)],
            params={"l": 2},
        )

        self._assert_result(
            connection,
            select(table).order_by(table.c.id).limit(bindparam("l")),
            [(1, 1, 2), (2, 2, 3), (3, 3, 4)],
            params={"l": 3},
        )

    @testing.requires.bound_limit_offset
    def test_bound_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).offset(bindparam("o")),
            [(3, 3, 4), (4, 4, 5), (5, 4, 6)],
            params={"o": 2},
        )

        self._assert_result(
            connection,
            select(table).order_by(table.c.id).offset(bindparam("o")),
            [(2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)],
            params={"o": 1},
        )

    @testing.requires.bound_limit_offset
    def test_bound_limit_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(bindparam("l"))
            .offset(bindparam("o")),
            [(2, 2, 3), (3, 3, 4)],
            params={"l": 2, "o": 1},
        )

        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(bindparam("l"))
            .offset(bindparam("o")),
            [(3, 3, 4), (4, 4, 5), (5, 4, 6)],
            params={"l": 3, "o": 2},
        )

    @testing.requires.fetch_first
    def test_bound_fetch_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .fetch(bindparam("f"))
            .offset(bindparam("o")),
            [(2, 2, 3), (3, 3, 4)],
            params={"f": 2, "o": 1},
        )

        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .fetch(bindparam("f"))
            .offset(bindparam("o")),
            [(3, 3, 4), (4, 4, 5), (5, 4, 6)],
            params={"f": 3, "o": 2},
        )

    @testing.requires.sql_expression_limit_offset
    def test_expr_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .offset(literal_column("1") + literal_column("2")),
            [(4, 4, 5), (5, 4, 6)],
        )

    @testing.requires.sql_expression_limit_offset
    def test_expr_limit(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(literal_column("1") + literal_column("2")),
            [(1, 1, 2), (2, 2, 3), (3, 3, 4)],
        )

    @testing.requires.sql_expression_limit_offset
    def test_expr_limit_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(literal_column("1") + literal_column("1"))
            .offset(literal_column("1") + literal_column("1")),
            [(3, 3, 4), (4, 4, 5)],
        )

    @testing.requires.fetch_first
    @testing.requires.fetch_expression
    def test_expr_fetch_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .fetch(literal_column("1") + literal_column("1"))
            .offset(literal_column("1") + literal_column("1")),
            [(3, 3, 4), (4, 4, 5)],
        )

    @testing.requires.sql_expression_limit_offset
    def test_simple_limit_expr_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(2)
            .offset(literal_column("1") + literal_column("1")),
            [(3, 3, 4), (4, 4, 5)],
        )

        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(3)
            .offset(literal_column("1") + literal_column("1")),
            [(3, 3, 4), (4, 4, 5), (5, 4, 6)],
        )

    @testing.requires.sql_expression_limit_offset
    def test_expr_limit_simple_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(literal_column("1") + literal_column("1"))
            .offset(2),
            [(3, 3, 4), (4, 4, 5)],
        )

        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .limit(literal_column("1") + literal_column("1"))
            .offset(1),
            [(2, 2, 3), (3, 3, 4)],
        )

    @testing.requires.fetch_ties
    def test_simple_fetch_ties(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.x.desc()).fetch(1, with_ties=True),
            [(4, 4, 5), (5, 4, 6)],
            set_=True,
        )

        self._assert_result(
            connection,
            select(table).order_by(table.c.x.desc()).fetch(3, with_ties=True),
            [(3, 3, 4), (4, 4, 5), (5, 4, 6)],
            set_=True,
        )

    @testing.requires.fetch_ties
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_ties(self, connection):
        table = self.tables.some_table
        fa = connection.execute(
            select(table)
            .order_by(table.c.x)
            .fetch(2, with_ties=True)
            .offset(2)
        ).fetchall()
        eq_(fa[0], (3, 3, 4))
        eq_(set(fa), {(3, 3, 4), (4, 4, 5), (5, 4, 6)})

    @testing.requires.fetch_ties
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_ties_exact_number(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.x)
            .fetch(2, with_ties=True)
            .offset(1),
            [(2, 2, 3), (3, 3, 4)],
        )

        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.x)
            .fetch(3, with_ties=True)
            .offset(3),
            [(4, 4, 5), (5, 4, 6)],
        )

    @testing.requires.fetch_percent
    def test_simple_fetch_percent(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table).order_by(table.c.id).fetch(20, percent=True),
            [(1, 1, 2)],
        )

    @testing.requires.fetch_percent
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_percent(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.id)
            .fetch(40, percent=True)
            .offset(1),
            [(2, 2, 3), (3, 3, 4)],
        )

    @testing.requires.fetch_ties
    @testing.requires.fetch_percent
    def test_simple_fetch_percent_ties(self, connection):
        table = self.tables.some_table
        self._assert_result(
            connection,
            select(table)
            .order_by(table.c.x.desc())
            .fetch(20, percent=True, with_ties=True),
            [(4, 4, 5), (5, 4, 6)],
            set_=True,
        )

    @testing.requires.fetch_ties
    @testing.requires.fetch_percent
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_percent_ties(self, connection):
        table = self.tables.some_table
        fa = connection.execute(
            select(table)
            .order_by(table.c.x)
            .fetch(40, percent=True, with_ties=True)
            .offset(2)
        ).fetchall()
        eq_(fa[0], (3, 3, 4))
        eq_(set(fa), {(3, 3, 4), (4, 4, 5), (5, 4, 6)})


class SameNamedSchemaTableTest(fixtures.TablesTest):
    """tests for #7471"""

    __backend__ = True

    __requires__ = ("schemas",)

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            schema=config.test_schema,
        )
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column(
                "some_table_id",
                Integer,
                # ForeignKey("%s.some_table.id" % config.test_schema),
                nullable=False,
            ),
        )

    @classmethod
    def insert_data(cls, connection):
        some_table, some_table_schema = cls.tables(
            "some_table", "%s.some_table" % config.test_schema
        )
        connection.execute(some_table_schema.insert(), {"id": 1})
        connection.execute(some_table.insert(), {"id": 1, "some_table_id": 1})

    def test_simple_join_both_tables(self, connection):
        some_table, some_table_schema = self.tables(
            "some_table", "%s.some_table" % config.test_schema
        )

        eq_(
            connection.execute(
                select(some_table, some_table_schema).join_from(
                    some_table,
                    some_table_schema,
                    some_table.c.some_table_id == some_table_schema.c.id,
                )
            ).first(),
            (1, 1, 1),
        )

    def test_simple_join_whereclause_only(self, connection):
        some_table, some_table_schema = self.tables(
            "some_table", "%s.some_table" % config.test_schema
        )

        eq_(
            connection.execute(
                select(some_table)
                .join_from(
                    some_table,
                    some_table_schema,
                    some_table.c.some_table_id == some_table_schema.c.id,
                )
                .where(some_table.c.id == 1)
            ).first(),
            (1, 1),
        )

    def test_subquery(self, connection):
        some_table, some_table_schema = self.tables(
            "some_table", "%s.some_table" % config.test_schema
        )

        subq = (
            select(some_table)
            .join_from(
                some_table,
                some_table_schema,
                some_table.c.some_table_id == some_table_schema.c.id,
            )
            .where(some_table.c.id == 1)
            .subquery()
        )

        eq_(
            connection.execute(
                select(some_table, subq.c.id)
                .join_from(
                    some_table,
                    subq,
                    some_table.c.some_table_id == subq.c.id,
                )
                .where(some_table.c.id == 1)
            ).first(),
            (1, 1, 1),
        )


class JoinTest(fixtures.TablesTest):
    __backend__ = True

    def _assert_result(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.execute(select, params).fetchall(), result)

    @classmethod
    def define_tables(cls, metadata):
        Table("a", metadata, Column("id", Integer, primary_key=True))
        Table(
            "b",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("a_id", ForeignKey("a.id"), nullable=False),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.a.insert(),
            [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}],
        )

        connection.execute(
            cls.tables.b.insert(),
            [
                {"id": 1, "a_id": 1},
                {"id": 2, "a_id": 1},
                {"id": 4, "a_id": 2},
                {"id": 5, "a_id": 3},
            ],
        )

    def test_inner_join_fk(self):
        a, b = self.tables("a", "b")

        stmt = select(a, b).select_from(a.join(b)).order_by(a.c.id, b.c.id)

        self._assert_result(stmt, [(1, 1, 1), (1, 2, 1), (2, 4, 2), (3, 5, 3)])

    def test_inner_join_true(self):
        a, b = self.tables("a", "b")

        stmt = (
            select(a, b)
            .select_from(a.join(b, true()))
            .order_by(a.c.id, b.c.id)
        )

        self._assert_result(
            stmt,
            [
                (a, b, c)
                for (a,), (b, c) in itertools.product(
                    [(1,), (2,), (3,), (4,), (5,)],
                    [(1, 1), (2, 1), (4, 2), (5, 3)],
                )
            ],
        )

    def test_inner_join_false(self):
        a, b = self.tables("a", "b")

        stmt = (
            select(a, b)
            .select_from(a.join(b, false()))
            .order_by(a.c.id, b.c.id)
        )

        self._assert_result(stmt, [])

    def test_outer_join_false(self):
        a, b = self.tables("a", "b")

        stmt = (
            select(a, b)
            .select_from(a.outerjoin(b, false()))
            .order_by(a.c.id, b.c.id)
        )

        self._assert_result(
            stmt,
            [
                (1, None, None),
                (2, None, None),
                (3, None, None),
                (4, None, None),
                (5, None, None),
            ],
        )

    def test_outer_join_fk(self):
        a, b = self.tables("a", "b")

        stmt = select(a, b).select_from(a.join(b)).order_by(a.c.id, b.c.id)

        self._assert_result(stmt, [(1, 1, 1), (1, 2, 1), (2, 4, 2), (3, 5, 3)])


class CompoundSelectTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x", Integer),
            Column("y", Integer),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "x": 1, "y": 2},
                {"id": 2, "x": 2, "y": 3},
                {"id": 3, "x": 3, "y": 4},
                {"id": 4, "x": 4, "y": 5},
            ],
        )

    def _assert_result(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.execute(select, params).fetchall(), result)

    def test_plain_union(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2)
        s2 = select(table).where(table.c.id == 3)

        u1 = union(s1, s2)
        self._assert_result(
            u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)]
        )

    def test_select_from_plain_union(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2)
        s2 = select(table).where(table.c.id == 3)

        u1 = union(s1, s2).alias().select()
        self._assert_result(
            u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)]
        )

    @testing.requires.order_by_col_from_union
    @testing.requires.parens_in_union_contained_select_w_limit_offset
    def test_limit_offset_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).limit(1).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).limit(1).order_by(table.c.id)

        u1 = union(s1, s2).limit(2)
        self._assert_result(
            u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)]
        )

    @testing.requires.parens_in_union_contained_select_wo_limit_offset
    def test_order_by_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).order_by(table.c.id)

        u1 = union(s1, s2).limit(2)
        self._assert_result(
            u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)]
        )

    def test_distinct_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).distinct()
        s2 = select(table).where(table.c.id == 3).distinct()

        u1 = union(s1, s2).limit(2)
        self._assert_result(
            u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)]
        )

    @testing.requires.parens_in_union_contained_select_w_limit_offset
    def test_limit_offset_in_unions_from_alias(self):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).limit(1).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).limit(1).order_by(table.c.id)

        # this necessarily has double parens
        u1 = union(s1, s2).alias()
        self._assert_result(
            u1.select().limit(2).order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)]
        )

    def test_limit_offset_aliased_selectable_in_unions(self):
        table = self.tables.some_table
        s1 = (
            select(table)
            .where(table.c.id == 2)
            .limit(1)
            .order_by(table.c.id)
            .alias()
            .select()
        )
        s2 = (
            select(table)
            .where(table.c.id == 3)
            .limit(1)
            .order_by(table.c.id)
            .alias()
            .select()
        )

        u1 = union(s1, s2).limit(2)
        self._assert_result(
            u1.order_by(u1.selected_columns.id), [(2, 2, 3), (3, 3, 4)]
        )


class PostCompileParamsTest(
    AssertsExecutionResults, AssertsCompiledSQL, fixtures.TablesTest
):
    __backend__ = True

    __requires__ = ("standard_cursor_sql",)

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x", Integer),
            Column("y", Integer),
            Column("z", String(50)),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "x": 1, "y": 2, "z": "z1"},
                {"id": 2, "x": 2, "y": 3, "z": "z2"},
                {"id": 3, "x": 3, "y": 4, "z": "z3"},
                {"id": 4, "x": 4, "y": 5, "z": "z4"},
            ],
        )

    def test_compile(self):
        table = self.tables.some_table

        stmt = select(table.c.id).where(
            table.c.x == bindparam("q", literal_execute=True)
        )

        self.assert_compile(
            stmt,
            "SELECT some_table.id FROM some_table "
            "WHERE some_table.x = __[POSTCOMPILE_q]",
            {},
        )

    def test_compile_literal_binds(self):
        table = self.tables.some_table

        stmt = select(table.c.id).where(
            table.c.x == bindparam("q", 10, literal_execute=True)
        )

        self.assert_compile(
            stmt,
            "SELECT some_table.id FROM some_table WHERE some_table.x = 10",
            {},
            literal_binds=True,
        )

    def test_execute(self):
        table = self.tables.some_table

        stmt = select(table.c.id).where(
            table.c.x == bindparam("q", literal_execute=True)
        )

        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=10))

        asserter.assert_(
            CursorSQL(
                "SELECT some_table.id \nFROM some_table "
                "\nWHERE some_table.x = 10",
                () if config.db.dialect.positional else {},
            )
        )

    def test_execute_expanding_plus_literal_execute(self):
        table = self.tables.some_table

        stmt = select(table.c.id).where(
            table.c.x.in_(bindparam("q", expanding=True, literal_execute=True))
        )

        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=[5, 6, 7]))

        asserter.assert_(
            CursorSQL(
                "SELECT some_table.id \nFROM some_table "
                "\nWHERE some_table.x IN (5, 6, 7)",
                () if config.db.dialect.positional else {},
            )
        )

    @testing.requires.tuple_in
    def test_execute_tuple_expanding_plus_literal_execute(self):
        table = self.tables.some_table

        stmt = select(table.c.id).where(
            tuple_(table.c.x, table.c.y).in_(
                bindparam("q", expanding=True, literal_execute=True)
            )
        )

        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=[(5, 10), (12, 18)]))

        asserter.assert_(
            CursorSQL(
                "SELECT some_table.id \nFROM some_table "
                "\nWHERE (some_table.x, some_table.y) "
                "IN (%s(5, 10), (12, 18))"
                % ("VALUES " if config.db.dialect.tuple_in_values else ""),
                () if config.db.dialect.positional else {},
            )
        )

    @testing.requires.tuple_in
    def test_execute_tuple_expanding_plus_literal_heterogeneous_execute(self):
        table = self.tables.some_table

        stmt = select(table.c.id).where(
            tuple_(table.c.x, table.c.z).in_(
                bindparam("q", expanding=True, literal_execute=True)
            )
        )

        with self.sql_execution_asserter() as asserter:
            with config.db.connect() as conn:
                conn.execute(stmt, dict(q=[(5, "z1"), (12, "z3")]))

        asserter.assert_(
            CursorSQL(
                "SELECT some_table.id \nFROM some_table "
                "\nWHERE (some_table.x, some_table.z) "
                "IN (%s(5, 'z1'), (12, 'z3'))"
                % ("VALUES " if config.db.dialect.tuple_in_values else ""),
                () if config.db.dialect.positional else {},
            )
        )


class ExpandingBoundInTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x", Integer),
            Column("y", Integer),
            Column("z", String(50)),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "x": 1, "y": 2, "z": "z1"},
                {"id": 2, "x": 2, "y": 3, "z": "z2"},
                {"id": 3, "x": 3, "y": 4, "z": "z3"},
                {"id": 4, "x": 4, "y": 5, "z": "z4"},
            ],
        )

    def _assert_result(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.execute(select, params).fetchall(), result)

    def test_multiple_empty_sets_bindparam(self):
        # test that any anonymous aliasing used by the dialect
        # is fine with duplicates
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.in_(bindparam("q")))
            .where(table.c.y.in_(bindparam("p")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [], params={"q": [], "p": []})

    def test_multiple_empty_sets_direct(self):
        # test that any anonymous aliasing used by the dialect
        # is fine with duplicates
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.in_([]))
            .where(table.c.y.in_([]))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [])

    @testing.requires.tuple_in_w_empty
    def test_empty_heterogeneous_tuples_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(tuple_(table.c.x, table.c.z).in_(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [], params={"q": []})

    @testing.requires.tuple_in_w_empty
    def test_empty_heterogeneous_tuples_direct(self):
        table = self.tables.some_table

        def go(val, expected):
            stmt = (
                select(table.c.id)
                .where(tuple_(table.c.x, table.c.z).in_(val))
                .order_by(table.c.id)
            )
            self._assert_result(stmt, expected)

        go([], [])
        go([(2, "z2"), (3, "z3"), (4, "z4")], [(2,), (3,), (4,)])
        go([], [])

    @testing.requires.tuple_in_w_empty
    def test_empty_homogeneous_tuples_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(tuple_(table.c.x, table.c.y).in_(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [], params={"q": []})

    @testing.requires.tuple_in_w_empty
    def test_empty_homogeneous_tuples_direct(self):
        table = self.tables.some_table

        def go(val, expected):
            stmt = (
                select(table.c.id)
                .where(tuple_(table.c.x, table.c.y).in_(val))
                .order_by(table.c.id)
            )
            self._assert_result(stmt, expected)

        go([], [])
        go([(1, 2), (2, 3), (3, 4)], [(1,), (2,), (3,)])
        go([], [])

    def test_bound_in_scalar_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.in_(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [(2,), (3,), (4,)], params={"q": [2, 3, 4]})

    def test_bound_in_scalar_direct(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.in_([2, 3, 4]))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [(2,), (3,), (4,)])

    def test_nonempty_in_plus_empty_notin(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.in_([2, 3]))
            .where(table.c.id.not_in([]))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [(2,), (3,)])

    def test_empty_in_plus_notempty_notin(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.in_([]))
            .where(table.c.id.not_in([2, 3]))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [])

    def test_typed_str_in(self):
        """test related to #7292.

        as a type is given to the bound param, there is no ambiguity
        to the type of element.

        """

        stmt = text(
            "select id FROM some_table WHERE z IN :q ORDER BY id"
        ).bindparams(bindparam("q", type_=String, expanding=True))
        self._assert_result(
            stmt,
            [(2,), (3,), (4,)],
            params={"q": ["z2", "z3", "z4"]},
        )

    def test_untyped_str_in(self):
        """test related to #7292.

        for untyped expression, we look at the types of elements.
        Test for Sequence to detect tuple in.  but not strings or bytes!
        as always....

        """

        stmt = text(
            "select id FROM some_table WHERE z IN :q ORDER BY id"
        ).bindparams(bindparam("q", expanding=True))
        self._assert_result(
            stmt,
            [(2,), (3,), (4,)],
            params={"q": ["z2", "z3", "z4"]},
        )

    @testing.requires.tuple_in
    def test_bound_in_two_tuple_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(tuple_(table.c.x, table.c.y).in_(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(
            stmt, [(2,), (3,), (4,)], params={"q": [(2, 3), (3, 4), (4, 5)]}
        )

    @testing.requires.tuple_in
    def test_bound_in_two_tuple_direct(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(tuple_(table.c.x, table.c.y).in_([(2, 3), (3, 4), (4, 5)]))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [(2,), (3,), (4,)])

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(tuple_(table.c.x, table.c.z).in_(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(
            stmt,
            [(2,), (3,), (4,)],
            params={"q": [(2, "z2"), (3, "z3"), (4, "z4")]},
        )

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_direct(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(
                tuple_(table.c.x, table.c.z).in_(
                    [(2, "z2"), (3, "z3"), (4, "z4")]
                )
            )
            .order_by(table.c.id)
        )
        self._assert_result(
            stmt,
            [(2,), (3,), (4,)],
        )

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_text_bindparam(self):
        # note this becomes ARRAY if we dont use expanding
        # explicitly right now
        stmt = text(
            "select id FROM some_table WHERE (x, z) IN :q ORDER BY id"
        ).bindparams(bindparam("q", expanding=True))
        self._assert_result(
            stmt,
            [(2,), (3,), (4,)],
            params={"q": [(2, "z2"), (3, "z3"), (4, "z4")]},
        )

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_typed_bindparam_non_tuple(self):
        class LikeATuple(collections_abc.Sequence):
            def __init__(self, *data):
                self._data = data

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)

        stmt = text(
            "select id FROM some_table WHERE (x, z) IN :q ORDER BY id"
        ).bindparams(
            bindparam(
                "q", type_=TupleType(Integer(), String()), expanding=True
            )
        )
        self._assert_result(
            stmt,
            [(2,), (3,), (4,)],
            params={
                "q": [
                    LikeATuple(2, "z2"),
                    LikeATuple(3, "z3"),
                    LikeATuple(4, "z4"),
                ]
            },
        )

    @testing.requires.tuple_in
    def test_bound_in_heterogeneous_two_tuple_text_bindparam_non_tuple(self):
        # note this becomes ARRAY if we dont use expanding
        # explicitly right now

        class LikeATuple(collections_abc.Sequence):
            def __init__(self, *data):
                self._data = data

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)

        stmt = text(
            "select id FROM some_table WHERE (x, z) IN :q ORDER BY id"
        ).bindparams(bindparam("q", expanding=True))
        self._assert_result(
            stmt,
            [(2,), (3,), (4,)],
            params={
                "q": [
                    LikeATuple(2, "z2"),
                    LikeATuple(3, "z3"),
                    LikeATuple(4, "z4"),
                ]
            },
        )

    def test_empty_set_against_integer_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.in_(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [], params={"q": []})

    def test_empty_set_against_integer_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.x.in_([])).order_by(table.c.id)
        self._assert_result(stmt, [])

    def test_empty_set_against_integer_negation_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.x.not_in(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)], params={"q": []})

    def test_empty_set_against_integer_negation_direct(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id).where(table.c.x.not_in([])).order_by(table.c.id)
        )
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)])

    def test_empty_set_against_string_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.z.in_(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [], params={"q": []})

    def test_empty_set_against_string_direct(self):
        table = self.tables.some_table
        stmt = select(table.c.id).where(table.c.z.in_([])).order_by(table.c.id)
        self._assert_result(stmt, [])

    def test_empty_set_against_string_negation_bindparam(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id)
            .where(table.c.z.not_in(bindparam("q")))
            .order_by(table.c.id)
        )
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)], params={"q": []})

    def test_empty_set_against_string_negation_direct(self):
        table = self.tables.some_table
        stmt = (
            select(table.c.id).where(table.c.z.not_in([])).order_by(table.c.id)
        )
        self._assert_result(stmt, [(1,), (2,), (3,), (4,)])

    def test_null_in_empty_set_is_false_bindparam(self, connection):
        stmt = select(
            case(
                (
                    null().in_(bindparam("foo", value=())),
                    true(),
                ),
                else_=false(),
            )
        )
        in_(connection.execute(stmt).fetchone()[0], (False, 0))

    def test_null_in_empty_set_is_false_direct(self, connection):
        stmt = select(
            case(
                (
                    null().in_([]),
                    true(),
                ),
                else_=false(),
            )
        )
        in_(connection.execute(stmt).fetchone()[0], (False, 0))


class LikeFunctionsTest(fixtures.TablesTest):
    __backend__ = True

    run_inserts = "once"
    run_deletes = None

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(50)),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "data": "abcdefg"},
                {"id": 2, "data": "ab/cdefg"},
                {"id": 3, "data": "ab%cdefg"},
                {"id": 4, "data": "ab_cdefg"},
                {"id": 5, "data": "abcde/fg"},
                {"id": 6, "data": "abcde%fg"},
                {"id": 7, "data": "ab#cdefg"},
                {"id": 8, "data": "ab9cdefg"},
                {"id": 9, "data": "abcde#fg"},
                {"id": 10, "data": "abcd9fg"},
                {"id": 11, "data": None},
            ],
        )

    def _test(self, expr, expected):
        some_table = self.tables.some_table

        with config.db.connect() as conn:
            rows = {
                value
                for value, in conn.execute(select(some_table.c.id).where(expr))
            }

        eq_(rows, expected)

    def test_startswith_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab%c"), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

    @testing.requires.like_escapes
    def test_startswith_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab%c", autoescape=True), {3})

    def test_startswith_sqlexpr(self):
        col = self.tables.some_table.c.data
        self._test(
            col.startswith(literal_column("'ab%c'")),
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        )

    @testing.requires.like_escapes
    def test_startswith_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab##c", escape="#"), {7})

    @testing.requires.like_escapes
    def test_startswith_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab%c", autoescape=True, escape="#"), {3})
        self._test(col.startswith("ab#c", autoescape=True, escape="#"), {7})

    def test_endswith_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e%fg"), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_endswith_sqlexpr(self):
        col = self.tables.some_table.c.data
        self._test(
            col.endswith(literal_column("'e%fg'")), {1, 2, 3, 4, 5, 6, 7, 8, 9}
        )

    @testing.requires.like_escapes
    def test_endswith_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e%fg", autoescape=True), {6})

    @testing.requires.like_escapes
    def test_endswith_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e##fg", escape="#"), {9})

    @testing.requires.like_escapes
    def test_endswith_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e%fg", autoescape=True, escape="#"), {6})
        self._test(col.endswith("e#fg", autoescape=True, escape="#"), {9})

    def test_contains_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b%cde"), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    @testing.requires.like_escapes
    def test_contains_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b%cde", autoescape=True), {3})

    @testing.requires.like_escapes
    def test_contains_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b##cde", escape="#"), {7})

    @testing.requires.like_escapes
    def test_contains_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b%cd", autoescape=True, escape="#"), {3})
        self._test(col.contains("b#cd", autoescape=True, escape="#"), {7})

    @testing.requires.regexp_match
    def test_not_regexp_match(self):
        col = self.tables.some_table.c.data
        self._test(~col.regexp_match("a.cde"), {2, 3, 4, 7, 8, 10})

    @testing.requires.regexp_replace
    def test_regexp_replace(self):
        col = self.tables.some_table.c.data
        self._test(
            col.regexp_replace("a.cde", "FOO").contains("FOO"), {1, 5, 6, 9}
        )

    @testing.requires.regexp_match
    @testing.combinations(
        ("a.cde", {1, 5, 6, 9}),
        ("abc", {1, 5, 6, 9, 10}),
        ("^abc", {1, 5, 6, 9, 10}),
        ("9cde", {8}),
        ("^a", set(range(1, 11))),
        ("(b|c)", set(range(1, 11))),
        ("^(b|c)", set()),
    )
    def test_regexp_match(self, text, expected):
        col = self.tables.some_table.c.data
        self._test(col.regexp_match(text), expected)


class ComputedColumnTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ("computed_columns",)

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "square",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("side", Integer),
            Column("area", Integer, Computed("side * side")),
            Column("perimeter", Integer, Computed("4 * side")),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.square.insert(),
            [{"id": 1, "side": 10}, {"id": 10, "side": 42}],
        )

    def test_select_all(self):
        with config.db.connect() as conn:
            res = conn.execute(
                select(text("*"))
                .select_from(self.tables.square)
                .order_by(self.tables.square.c.id)
            ).fetchall()
            eq_(res, [(1, 10, 100, 40), (10, 42, 1764, 168)])

    def test_select_columns(self):
        with config.db.connect() as conn:
            res = conn.execute(
                select(
                    self.tables.square.c.area, self.tables.square.c.perimeter
                )
                .select_from(self.tables.square)
                .order_by(self.tables.square.c.id)
            ).fetchall()
            eq_(res, [(100, 40), (1764, 168)])


class IdentityColumnTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ("identity_columns",)
    run_inserts = "once"
    run_deletes = "once"

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "tbl_a",
            metadata,
            Column(
                "id",
                Integer,
                Identity(
                    always=True, start=42, nominvalue=True, nomaxvalue=True
                ),
                primary_key=True,
            ),
            Column("desc", String(100)),
        )
        Table(
            "tbl_b",
            metadata,
            Column(
                "id",
                Integer,
                Identity(increment=-5, start=0, minvalue=-1000, maxvalue=0),
                primary_key=True,
            ),
            Column("desc", String(100)),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.tbl_a.insert(),
            [{"desc": "a"}, {"desc": "b"}],
        )
        connection.execute(
            cls.tables.tbl_b.insert(),
            [{"desc": "a"}, {"desc": "b"}],
        )
        connection.execute(
            cls.tables.tbl_b.insert(),
            [{"id": 42, "desc": "c"}],
        )

    def test_select_all(self, connection):
        res = connection.execute(
            select(text("*"))
            .select_from(self.tables.tbl_a)
            .order_by(self.tables.tbl_a.c.id)
        ).fetchall()
        eq_(res, [(42, "a"), (43, "b")])

        res = connection.execute(
            select(text("*"))
            .select_from(self.tables.tbl_b)
            .order_by(self.tables.tbl_b.c.id)
        ).fetchall()
        eq_(res, [(-5, "b"), (0, "a"), (42, "c")])

    def test_select_columns(self, connection):
        res = connection.execute(
            select(self.tables.tbl_a.c.id).order_by(self.tables.tbl_a.c.id)
        ).fetchall()
        eq_(res, [(42,), (43,)])

    @testing.requires.identity_columns_standard
    def test_insert_always_error(self, connection):
        def fn():
            connection.execute(
                self.tables.tbl_a.insert(),
                [{"id": 200, "desc": "a"}],
            )

        assert_raises((DatabaseError, ProgrammingError), fn)


class IdentityAutoincrementTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ("autoincrement_without_sequence",)

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "tbl",
            metadata,
            Column(
                "id",
                Integer,
                Identity(),
                primary_key=True,
                autoincrement=True,
            ),
            Column("desc", String(100)),
        )

    def test_autoincrement_with_identity(self, connection):
        connection.execute(self.tables.tbl.insert(), {"desc": "row"})
        res = connection.execute(self.tables.tbl.select()).first()
        eq_(res, (1, "row"))


class ExistsTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "stuff",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(50)),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.stuff.insert(),
            [
                {"id": 1, "data": "some data"},
                {"id": 2, "data": "some data"},
                {"id": 3, "data": "some data"},
                {"id": 4, "data": "some other data"},
            ],
        )

    def test_select_exists(self, connection):
        stuff = self.tables.stuff
        eq_(
            connection.execute(
                select(literal(1)).where(
                    exists().where(stuff.c.data == "some data")
                )
            ).fetchall(),
            [(1,)],
        )

    def test_select_exists_false(self, connection):
        stuff = self.tables.stuff
        eq_(
            connection.execute(
                select(literal(1)).where(
                    exists().where(stuff.c.data == "no data")
                )
            ).fetchall(),
            [],
        )


class DistinctOnTest(AssertsCompiledSQL, fixtures.TablesTest):
    __backend__ = True

    @testing.fails_if(testing.requires.supports_distinct_on)
    def test_distinct_on(self):
        stm = select("*").distinct(column("q")).select_from(table("foo"))
        with testing.expect_deprecated(
            "DISTINCT ON is currently supported only by the PostgreSQL "
        ):
            self.assert_compile(stm, "SELECT DISTINCT * FROM foo")


class IsOrIsNotDistinctFromTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ("supports_is_distinct_from",)

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "is_distinct_test",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("col_a", Integer, nullable=True),
            Column("col_b", Integer, nullable=True),
        )

    @testing.combinations(
        ("both_int_different", 0, 1, 1),
        ("both_int_same", 1, 1, 0),
        ("one_null_first", None, 1, 1),
        ("one_null_second", 0, None, 1),
        ("both_null", None, None, 0),
        id_="iaaa",
        argnames="col_a_value, col_b_value, expected_row_count_for_is",
    )
    def test_is_or_is_not_distinct_from(
        self, col_a_value, col_b_value, expected_row_count_for_is, connection
    ):
        tbl = self.tables.is_distinct_test

        connection.execute(
            tbl.insert(),
            [{"id": 1, "col_a": col_a_value, "col_b": col_b_value}],
        )

        result = connection.execute(
            tbl.select().where(tbl.c.col_a.is_distinct_from(tbl.c.col_b))
        ).fetchall()
        eq_(
            len(result),
            expected_row_count_for_is,
        )

        expected_row_count_for_is_not = (
            1 if expected_row_count_for_is == 0 else 0
        )
        result = connection.execute(
            tbl.select().where(tbl.c.col_a.is_not_distinct_from(tbl.c.col_b))
        ).fetchall()
        eq_(
            len(result),
            expected_row_count_for_is_not,
        )


class WindowFunctionTest(fixtures.TablesTest):
    __requires__ = ("window_functions",)

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("col1", Integer),
            Column("col2", Integer),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [{"id": i, "col1": i, "col2": i * 5} for i in range(1, 50)],
        )

    def test_window(self, connection):
        some_table = self.tables.some_table
        rows = connection.execute(
            select(
                func.max(some_table.c.col2).over(
                    order_by=[some_table.c.col1.desc()]
                )
            ).where(some_table.c.col1 < 20)
        ).all()

        eq_(rows, [(95,) for i in range(19)])

    def test_window_rows_between(self, connection):
        some_table = self.tables.some_table

        # note the rows are part of the cache key right now, not handled
        # as binds.  this is issue #11515
        rows = connection.execute(
            select(
                func.max(some_table.c.col2).over(
                    order_by=[some_table.c.col1],
                    rows=(-5, 0),
                )
            )
        ).all()

        eq_(rows, [(i,) for i in range(5, 250, 5)])


class BitwiseTest(fixtures.TablesTest):
    __backend__ = True
    run_inserts = run_deletes = "once"

    inserted_data = [{"a": i, "b": i + 1} for i in range(10)]

    @classmethod
    def define_tables(cls, metadata):
        Table("bitwise", metadata, Column("a", Integer), Column("b", Integer))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.bitwise.insert(), cls.inserted_data)

    @testing.combinations(
        (
            lambda a: a.bitwise_xor(5),
            [i for i in range(10) if i != 5],
            testing.requires.supports_bitwise_xor,
        ),
        (
            lambda a: a.bitwise_or(1),
            list(range(10)),
            testing.requires.supports_bitwise_or,
        ),
        (
            lambda a: a.bitwise_and(4),
            list(range(4, 8)),
            testing.requires.supports_bitwise_and,
        ),
        (
            lambda a: (a - 2).bitwise_not(),
            [0],
            testing.requires.supports_bitwise_not,
        ),
        (
            lambda a: a.bitwise_lshift(1),
            list(range(1, 10)),
            testing.requires.supports_bitwise_shift,
        ),
        (
            lambda a: a.bitwise_rshift(2),
            list(range(4, 10)),
            testing.requires.supports_bitwise_shift,
        ),
        argnames="case, expected",
    )
    def test_bitwise(self, case, expected, connection):
        tbl = self.tables.bitwise

        a = tbl.c.a

        op = testing.resolve_lambda(case, a=a)

        stmt = select(tbl).where(op > 0).order_by(a)

        res = connection.execute(stmt).mappings().all()
        eq_(res, [self.inserted_data[i] for i in expected])