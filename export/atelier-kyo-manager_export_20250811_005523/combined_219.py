
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\generate_legacy_storage_files.py ===
"""
self-contained to write legacy storage pickle files

To use this script. Create an environment where you want
generate pickles, say its for 0.20.3, with your pandas clone
in ~/pandas

. activate pandas_0.20.3
cd ~/pandas/pandas

$ python -m tests.io.generate_legacy_storage_files \
    tests/io/data/legacy_pickle/0.20.3/ pickle

This script generates a storage file for the current arch, system,
and python version
  pandas version: 0.20.3
  output dir    : pandas/pandas/tests/io/data/legacy_pickle/0.20.3/
  storage format: pickle
created pickle file: 0.20.3_x86_64_darwin_3.5.2.pickle

The idea here is you are using the *current* version of the
generate_legacy_storage_files with an *older* version of pandas to
generate a pickle file. We will then check this file into a current
branch, and test using test_pickle.py. This will load the *older*
pickles and test versus the current data that is generated
(with main). These are then compared.

If we have cases where we changed the signature (e.g. we renamed
offset -> freq in Timestamp). Then we have to conditionally execute
in the generate_legacy_storage_files.py to make it
run under the older AND the newer version.

"""

from datetime import timedelta
import os
import pickle
import platform as pl
import sys

# Remove script directory from path, otherwise Python will try to
# import the JSON test directory as the json module
sys.path.pop(0)

import numpy as np

import pandas
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Period,
    RangeIndex,
    Series,
    Timestamp,
    bdate_range,
    date_range,
    interval_range,
    period_range,
    timedelta_range,
)
from pandas.arrays import SparseArray

from pandas.tseries.offsets import (
    FY5253,
    BusinessDay,
    BusinessHour,
    CustomBusinessDay,
    DateOffset,
    Day,
    Easter,
    Hour,
    LastWeekOfMonth,
    Minute,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)


def _create_sp_series():
    nan = np.nan

    # nan-based
    arr = np.arange(15, dtype=np.float64)
    arr[7:12] = nan
    arr[-1:] = nan

    bseries = Series(SparseArray(arr, kind="block"))
    bseries.name = "bseries"
    return bseries


def _create_sp_tsseries():
    nan = np.nan

    # nan-based
    arr = np.arange(15, dtype=np.float64)
    arr[7:12] = nan
    arr[-1:] = nan

    date_index = bdate_range("1/1/2011", periods=len(arr))
    bseries = Series(SparseArray(arr, kind="block"), index=date_index)
    bseries.name = "btsseries"
    return bseries


def _create_sp_frame():
    nan = np.nan

    data = {
        "A": [nan, nan, nan, 0, 1, 2, 3, 4, 5, 6],
        "B": [0, 1, 2, nan, nan, nan, 3, 4, 5, 6],
        "C": np.arange(10).astype(np.int64),
        "D": [0, 1, 2, 3, 4, 5, nan, nan, nan, nan],
    }

    dates = bdate_range("1/1/2011", periods=10)
    return DataFrame(data, index=dates).apply(SparseArray)


def create_pickle_data():
    """create the pickle data"""
    data = {
        "A": [0.0, 1.0, 2.0, 3.0, np.nan],
        "B": [0, 1, 0, 1, 0],
        "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
        "D": date_range("1/1/2009", periods=5),
        "E": [0.0, 1, Timestamp("20100101"), "foo", 2.0],
    }

    scalars = {"timestamp": Timestamp("20130101"), "period": Period("2012", "M")}

    index = {
        "int": Index(np.arange(10)),
        "date": date_range("20130101", periods=10),
        "period": period_range("2013-01-01", freq="M", periods=10),
        "float": Index(np.arange(10, dtype=np.float64)),
        "uint": Index(np.arange(10, dtype=np.uint64)),
        "timedelta": timedelta_range("00:00:00", freq="30min", periods=10),
    }

    index["range"] = RangeIndex(10)

    index["interval"] = interval_range(0, periods=10)

    mi = {
        "reg2": MultiIndex.from_tuples(
            tuple(
                zip(
                    *[
                        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                        ["one", "two", "one", "two", "one", "two", "one", "two"],
                    ]
                )
            ),
            names=["first", "second"],
        )
    }

    series = {
        "float": Series(data["A"]),
        "int": Series(data["B"]),
        "mixed": Series(data["E"]),
        "ts": Series(
            np.arange(10).astype(np.int64), index=date_range("20130101", periods=10)
        ),
        "mi": Series(
            np.arange(5).astype(np.float64),
            index=MultiIndex.from_tuples(
                tuple(zip(*[[1, 1, 2, 2, 2], [3, 4, 3, 4, 5]])), names=["one", "two"]
            ),
        ),
        "dup": Series(np.arange(5).astype(np.float64), index=["A", "B", "C", "D", "A"]),
        "cat": Series(Categorical(["foo", "bar", "baz"])),
        "dt": Series(date_range("20130101", periods=5)),
        "dt_tz": Series(date_range("20130101", periods=5, tz="US/Eastern")),
        "period": Series([Period("2000Q1")] * 5),
    }

    mixed_dup_df = DataFrame(data)
    mixed_dup_df.columns = list("ABCDA")
    frame = {
        "float": DataFrame({"A": series["float"], "B": series["float"] + 1}),
        "int": DataFrame({"A": series["int"], "B": series["int"] + 1}),
        "mixed": DataFrame({k: data[k] for k in ["A", "B", "C", "D"]}),
        "mi": DataFrame(
            {"A": np.arange(5).astype(np.float64), "B": np.arange(5).astype(np.int64)},
            index=MultiIndex.from_tuples(
                tuple(
                    zip(
                        *[
                            ["bar", "bar", "baz", "baz", "baz"],
                            ["one", "two", "one", "two", "three"],
                        ]
                    )
                ),
                names=["first", "second"],
            ),
        ),
        "dup": DataFrame(
            np.arange(15).reshape(5, 3).astype(np.float64), columns=["A", "B", "A"]
        ),
        "cat_onecol": DataFrame({"A": Categorical(["foo", "bar"])}),
        "cat_and_float": DataFrame(
            {
                "A": Categorical(["foo", "bar", "baz"]),
                "B": np.arange(3).astype(np.int64),
            }
        ),
        "mixed_dup": mixed_dup_df,
        "dt_mixed_tzs": DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        ),
        "dt_mixed2_tzs": DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
                "C": Timestamp("20130603", tz="UTC"),
            },
            index=range(5),
        ),
    }

    cat = {
        "int8": Categorical(list("abcdefg")),
        "int16": Categorical(np.arange(1000)),
        "int32": Categorical(np.arange(10000)),
    }

    timestamp = {
        "normal": Timestamp("2011-01-01"),
        "nat": NaT,
        "tz": Timestamp("2011-01-01", tz="US/Eastern"),
    }

    off = {
        "DateOffset": DateOffset(years=1),
        "DateOffset_h_ns": DateOffset(hour=6, nanoseconds=5824),
        "BusinessDay": BusinessDay(offset=timedelta(seconds=9)),
        "BusinessHour": BusinessHour(normalize=True, n=6, end="15:14"),
        "CustomBusinessDay": CustomBusinessDay(weekmask="Mon Fri"),
        "SemiMonthBegin": SemiMonthBegin(day_of_month=9),
        "SemiMonthEnd": SemiMonthEnd(day_of_month=24),
        "MonthBegin": MonthBegin(1),
        "MonthEnd": MonthEnd(1),
        "QuarterBegin": QuarterBegin(1),
        "QuarterEnd": QuarterEnd(1),
        "Day": Day(1),
        "YearBegin": YearBegin(1),
        "YearEnd": YearEnd(1),
        "Week": Week(1),
        "Week_Tues": Week(2, normalize=False, weekday=1),
        "WeekOfMonth": WeekOfMonth(week=3, weekday=4),
        "LastWeekOfMonth": LastWeekOfMonth(n=1, weekday=3),
        "FY5253": FY5253(n=2, weekday=6, startingMonth=7, variation="last"),
        "Easter": Easter(),
        "Hour": Hour(1),
        "Minute": Minute(1),
    }

    return {
        "series": series,
        "frame": frame,
        "index": index,
        "scalars": scalars,
        "mi": mi,
        "sp_series": {"float": _create_sp_series(), "ts": _create_sp_tsseries()},
        "sp_frame": {"float": _create_sp_frame()},
        "cat": cat,
        "timestamp": timestamp,
        "offsets": off,
    }


def platform_name():
    return "_".join(
        [
            str(pandas.__version__),
            str(pl.machine()),
            str(pl.system().lower()),
            str(pl.python_version()),
        ]
    )


def write_legacy_pickles(output_dir):
    version = pandas.__version__

    print(
        "This script generates a storage file for the current arch, system, "
        "and python version"
    )
    print(f"  pandas version: {version}")
    print(f"  output dir    : {output_dir}")
    print("  storage format: pickle")

    pth = f"{platform_name()}.pickle"

    with open(os.path.join(output_dir, pth), "wb") as fh:
        pickle.dump(create_pickle_data(), fh, pickle.DEFAULT_PROTOCOL)

    print(f"created pickle file: {pth}")


def write_legacy_file():
    # force our cwd to be the first searched
    sys.path.insert(0, "")

    if not 3 <= len(sys.argv) <= 4:
        sys.exit(
            "Specify output directory and storage type: generate_legacy_"
            "storage_files.py <output_dir> <storage_type> "
        )

    output_dir = str(sys.argv[1])
    storage_type = str(sys.argv[2])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if storage_type == "pickle":
        write_legacy_pickles(output_dir=output_dir)
    else:
        sys.exit("storage_type must be one of {'pickle'}")


if __name__ == "__main__":
    write_legacy_file()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_textreader.py ===
"""
Tests the TextReader class in parsers.pyx, which
is integral to the C engine in parsers.py
"""
from io import (
    BytesIO,
    StringIO,
)

import numpy as np
import pytest

import pandas._libs.parsers as parser
from pandas._libs.parsers import TextReader
from pandas.errors import ParserWarning

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.parsers import (
    TextFileReader,
    read_csv,
)
from pandas.io.parsers.c_parser_wrapper import ensure_dtype_objs


class TestTextReader:
    @pytest.fixture
    def csv_path(self, datapath):
        return datapath("io", "data", "csv", "test1.csv")

    def test_file_handle(self, csv_path):
        with open(csv_path, "rb") as f:
            reader = TextReader(f)
            reader.read()

    def test_file_handle_mmap(self, csv_path):
        # this was never using memory_map=True
        with open(csv_path, "rb") as f:
            reader = TextReader(f, header=None)
            reader.read()

    def test_StringIO(self, csv_path):
        with open(csv_path, "rb") as f:
            text = f.read()
        src = BytesIO(text)
        reader = TextReader(src, header=None)
        reader.read()

    def test_string_factorize(self):
        # should this be optional?
        data = "a\nb\na\nb\na"
        reader = TextReader(StringIO(data), header=None)
        result = reader.read()
        assert len(set(map(id, result[0]))) == 2

    def test_skipinitialspace(self):
        data = "a,   b\na,   b\na,   b\na,   b"

        reader = TextReader(StringIO(data), skipinitialspace=True, header=None)
        result = reader.read()

        tm.assert_numpy_array_equal(
            result[0], np.array(["a", "a", "a", "a"], dtype=np.object_)
        )
        tm.assert_numpy_array_equal(
            result[1], np.array(["b", "b", "b", "b"], dtype=np.object_)
        )

    def test_parse_booleans(self):
        data = "True\nFalse\nTrue\nTrue"

        reader = TextReader(StringIO(data), header=None)
        result = reader.read()

        assert result[0].dtype == np.bool_

    def test_delimit_whitespace(self):
        data = 'a  b\na\t\t "b"\n"a"\t \t b'

        reader = TextReader(StringIO(data), delim_whitespace=True, header=None)
        result = reader.read()

        tm.assert_numpy_array_equal(
            result[0], np.array(["a", "a", "a"], dtype=np.object_)
        )
        tm.assert_numpy_array_equal(
            result[1], np.array(["b", "b", "b"], dtype=np.object_)
        )

    def test_embedded_newline(self):
        data = 'a\n"hello\nthere"\nthis'

        reader = TextReader(StringIO(data), header=None)
        result = reader.read()

        expected = np.array(["a", "hello\nthere", "this"], dtype=np.object_)
        tm.assert_numpy_array_equal(result[0], expected)

    def test_euro_decimal(self):
        data = "12345,67\n345,678"

        reader = TextReader(StringIO(data), delimiter=":", decimal=",", header=None)
        result = reader.read()

        expected = np.array([12345.67, 345.678])
        tm.assert_almost_equal(result[0], expected)

    def test_integer_thousands(self):
        data = "123,456\n12,500"

        reader = TextReader(StringIO(data), delimiter=":", thousands=",", header=None)
        result = reader.read()

        expected = np.array([123456, 12500], dtype=np.int64)
        tm.assert_almost_equal(result[0], expected)

    def test_integer_thousands_alt(self):
        data = "123.456\n12.500"

        reader = TextFileReader(
            StringIO(data), delimiter=":", thousands=".", header=None
        )
        result = reader.read()

        expected = DataFrame([123456, 12500])
        tm.assert_frame_equal(result, expected)

    def test_skip_bad_lines(self):
        # too many lines, see #2430 for why
        data = "a:b:c\nd:e:f\ng:h:i\nj:k:l:m\nl:m:n\no:p:q:r"

        reader = TextReader(StringIO(data), delimiter=":", header=None)
        msg = r"Error tokenizing data\. C error: Expected 3 fields in line 4, saw 4"
        with pytest.raises(parser.ParserError, match=msg):
            reader.read()

        reader = TextReader(
            StringIO(data), delimiter=":", header=None, on_bad_lines=2  # Skip
        )
        result = reader.read()
        expected = {
            0: np.array(["a", "d", "g", "l"], dtype=object),
            1: np.array(["b", "e", "h", "m"], dtype=object),
            2: np.array(["c", "f", "i", "n"], dtype=object),
        }
        assert_array_dicts_equal(result, expected)

        with tm.assert_produces_warning(ParserWarning, match="Skipping line"):
            reader = TextReader(
                StringIO(data), delimiter=":", header=None, on_bad_lines=1  # Warn
            )
            reader.read()

    def test_header_not_enough_lines(self):
        data = "skip this\nskip this\na,b,c\n1,2,3\n4,5,6"

        reader = TextReader(StringIO(data), delimiter=",", header=2)
        header = reader.header
        expected = [["a", "b", "c"]]
        assert header == expected

        recs = reader.read()
        expected = {
            0: np.array([1, 4], dtype=np.int64),
            1: np.array([2, 5], dtype=np.int64),
            2: np.array([3, 6], dtype=np.int64),
        }
        assert_array_dicts_equal(recs, expected)

    def test_escapechar(self):
        data = '\\"hello world"\n\\"hello world"\n\\"hello world"'

        reader = TextReader(StringIO(data), delimiter=",", header=None, escapechar="\\")
        result = reader.read()
        expected = {0: np.array(['"hello world"'] * 3, dtype=object)}
        assert_array_dicts_equal(result, expected)

    def test_eof_has_eol(self):
        # handling of new line at EOF
        pass

    def test_na_substitution(self):
        pass

    def test_numpy_string_dtype(self):
        data = """\
a,1
aa,2
aaa,3
aaaa,4
aaaaa,5"""

        def _make_reader(**kwds):
            if "dtype" in kwds:
                kwds["dtype"] = ensure_dtype_objs(kwds["dtype"])
            return TextReader(StringIO(data), delimiter=",", header=None, **kwds)

        reader = _make_reader(dtype="S5,i4")
        result = reader.read()

        assert result[0].dtype == "S5"

        ex_values = np.array(["a", "aa", "aaa", "aaaa", "aaaaa"], dtype="S5")
        assert (result[0] == ex_values).all()
        assert result[1].dtype == "i4"

        reader = _make_reader(dtype="S4")
        result = reader.read()
        assert result[0].dtype == "S4"
        ex_values = np.array(["a", "aa", "aaa", "aaaa", "aaaa"], dtype="S4")
        assert (result[0] == ex_values).all()
        assert result[1].dtype == "S4"

    def test_pass_dtype(self):
        data = """\
one,two
1,a
2,b
3,c
4,d"""

        def _make_reader(**kwds):
            if "dtype" in kwds:
                kwds["dtype"] = ensure_dtype_objs(kwds["dtype"])
            return TextReader(StringIO(data), delimiter=",", **kwds)

        reader = _make_reader(dtype={"one": "u1", 1: "S1"})
        result = reader.read()
        assert result[0].dtype == "u1"
        assert result[1].dtype == "S1"

        reader = _make_reader(dtype={"one": np.uint8, 1: object})
        result = reader.read()
        assert result[0].dtype == "u1"
        assert result[1].dtype == "O"

        reader = _make_reader(dtype={"one": np.dtype("u1"), 1: np.dtype("O")})
        result = reader.read()
        assert result[0].dtype == "u1"
        assert result[1].dtype == "O"

    def test_usecols(self):
        data = """\
a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""

        def _make_reader(**kwds):
            return TextReader(StringIO(data), delimiter=",", **kwds)

        reader = _make_reader(usecols=(1, 2))
        result = reader.read()

        exp = _make_reader().read()
        assert len(result) == 2
        assert (result[1] == exp[1]).all()
        assert (result[2] == exp[2]).all()

    @pytest.mark.parametrize(
        "text, kwargs",
        [
            ("a,b,c\r1,2,3\r4,5,6\r7,8,9\r10,11,12", {"delimiter": ","}),
            (
                "a  b  c\r1  2  3\r4  5  6\r7  8  9\r10  11  12",
                {"delim_whitespace": True},
            ),
            ("a,b,c\r1,2,3\r4,5,6\r,88,9\r10,11,12", {"delimiter": ","}),
            (
                (
                    "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O\r"
                    "AAAAA,BBBBB,0,0,0,0,0,0,0,0,0,0,0,0,0\r"
                    ",BBBBB,0,0,0,0,0,0,0,0,0,0,0,0,0"
                ),
                {"delimiter": ","},
            ),
            ("A  B  C\r  2  3\r4  5  6", {"delim_whitespace": True}),
            ("A B C\r2 3\r4 5 6", {"delim_whitespace": True}),
        ],
    )
    def test_cr_delimited(self, text, kwargs):
        nice_text = text.replace("\r", "\r\n")
        result = TextReader(StringIO(text), **kwargs).read()
        expected = TextReader(StringIO(nice_text), **kwargs).read()
        assert_array_dicts_equal(result, expected)

    def test_empty_field_eof(self):
        data = "a,b,c\n1,2,3\n4,,"

        result = TextReader(StringIO(data), delimiter=",").read()

        expected = {
            0: np.array([1, 4], dtype=np.int64),
            1: np.array(["2", ""], dtype=object),
            2: np.array(["3", ""], dtype=object),
        }
        assert_array_dicts_equal(result, expected)

    @pytest.mark.parametrize("repeat", range(10))
    def test_empty_field_eof_mem_access_bug(self, repeat):
        # GH5664
        a = DataFrame([["b"], [np.nan]], columns=["a"], index=["a", "c"])
        b = DataFrame([[1, 1, 1, 0], [1, 1, 1, 0]], columns=list("abcd"), index=[1, 1])
        c = DataFrame(
            [
                [1, 2, 3, 4],
                [6, np.nan, np.nan, np.nan],
                [8, 9, 10, 11],
                [13, 14, np.nan, np.nan],
            ],
            columns=list("abcd"),
            index=[0, 5, 7, 12],
        )

        df = read_csv(StringIO("a,b\nc\n"), skiprows=0, names=["a"], engine="c")
        tm.assert_frame_equal(df, a)

        df = read_csv(
            StringIO("1,1,1,1,0\n" * 2 + "\n" * 2), names=list("abcd"), engine="c"
        )
        tm.assert_frame_equal(df, b)

        df = read_csv(
            StringIO("0,1,2,3,4\n5,6\n7,8,9,10,11\n12,13,14"),
            names=list("abcd"),
            engine="c",
        )
        tm.assert_frame_equal(df, c)

    def test_empty_csv_input(self):
        # GH14867
        with read_csv(
            StringIO(), chunksize=20, header=None, names=["a", "b", "c"]
        ) as df:
            assert isinstance(df, TextFileReader)


def assert_array_dicts_equal(left, right):
    for k, v in left.items():
        tm.assert_numpy_array_equal(np.asarray(v), np.asarray(right[k]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\frame\test_hist_box_by.py ===
import re

import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_plot_works,
    get_x_axis,
    get_y_axis,
)

pytest.importorskip("matplotlib")


@pytest.fixture
def hist_df():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 2)), columns=["A", "B"]
    )
    df["C"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    df["D"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    return df


class TestHistWithBy:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [
            ("C", "A", ["a", "b", "c"], [["A"]] * 3),
            ("C", ["A", "B"], ["a", "b", "c"], [["A", "B"]] * 3),
            ("C", None, ["a", "b", "c"], [["A", "B"]] * 3),
            (
                ["C", "D"],
                "A",
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A"]] * 3,
            ),
            (
                ["C", "D"],
                ["A", "B"],
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A", "B"]] * 3,
            ),
            (
                ["C", "D"],
                None,
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A", "B"]] * 3,
            ),
        ],
    )
    def test_hist_plot_by_argument(self, by, column, titles, legends, hist_df):
        # GH 15079
        axes = _check_plot_works(
            hist_df.plot.hist, column=column, by=by, default_axes=True
        )
        result_titles = [ax.get_title() for ax in axes]
        result_legends = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]

        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [
            (0, "A", ["a", "b", "c"], [["A"]] * 3),
            (0, None, ["a", "b", "c"], [["A", "B"]] * 3),
            (
                [0, "D"],
                "A",
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A"]] * 3,
            ),
        ],
    )
    def test_hist_plot_by_0(self, by, column, titles, legends, hist_df):
        # GH 15079
        df = hist_df.copy()
        df = df.rename(columns={"C": 0})

        axes = _check_plot_works(df.plot.hist, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_legends = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]

        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),
            ([], ["A", "B"]),
            ((), None),
            ((), ["A", "B"]),
        ],
    )
    def test_hist_plot_empty_list_string_tuple_by(self, by, column, hist_df):
        # GH 15079
        msg = "No group keys passed"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(
                hist_df.plot.hist, default_axes=True, column=column, by=by
            )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [
            (["C"], "A", (2, 2), 3),
            ("C", "A", (2, 2), 3),
            (["C"], ["A"], (1, 3), 3),
            ("C", None, (3, 1), 3),
            ("C", ["A", "B"], (3, 1), 3),
            (["C", "D"], "A", (9, 1), 3),
            (["C", "D"], "A", (3, 3), 3),
            (["C", "D"], ["A"], (5, 2), 3),
            (["C", "D"], ["A", "B"], (9, 1), 3),
            (["C", "D"], None, (9, 1), 3),
            (["C", "D"], ["A", "B"], (5, 2), 3),
        ],
    )
    def test_hist_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
        # GH 15079
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(
                hist_df.plot.hist, column=column, by=by, layout=layout
            )
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [
            ("larger than required size", ["C", "D"], (1, 1)),
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),
            ("At least one dimension of layout must be positive", "C", (-1, -1)),
        ],
    )
    def test_hist_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
        # GH 15079, test if error is raised when invalid layout is given

        with pytest.raises(ValueError, match=msg):
            hist_df.plot.hist(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.slow
    def test_axis_share_x_with_by(self, hist_df):
        # GH 15079
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharex=True)

        # share x
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)
        assert get_x_axis(ax3).joined(ax1, ax3)
        assert get_x_axis(ax3).joined(ax2, ax3)

        # don't share y
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)
        assert not get_y_axis(ax3).joined(ax1, ax3)
        assert not get_y_axis(ax3).joined(ax2, ax3)

    @pytest.mark.slow
    def test_axis_share_y_with_by(self, hist_df):
        # GH 15079
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharey=True)

        # share y
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)
        assert get_y_axis(ax3).joined(ax1, ax3)
        assert get_y_axis(ax3).joined(ax2, ax3)

        # don't share x
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)
        assert not get_x_axis(ax3).joined(ax1, ax3)
        assert not get_x_axis(ax3).joined(ax2, ax3)

    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(self, figsize, hist_df):
        # GH 15079
        axes = hist_df.plot.hist(column="A", by="C", figsize=figsize)
        _check_axes_shape(axes, axes_num=3, figsize=figsize)


class TestBoxWithBy:
    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            ("C", "A", ["A"], [["a", "b", "c"]]),
            (
                ["C", "D"],
                "A",
                ["A"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ],
            ),
            ("C", ["A", "B"], ["A", "B"], [["a", "b", "c"]] * 2),
            (
                ["C", "D"],
                ["A", "B"],
                ["A", "B"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ]
                * 2,
            ),
            (["C"], None, ["A", "B"], [["a", "b", "c"]] * 2),
        ],
    )
    def test_box_plot_by_argument(self, by, column, titles, xticklabels, hist_df):
        # GH 15079
        axes = _check_plot_works(
            hist_df.plot.box, default_axes=True, column=column, by=by
        )
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]

        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            (0, "A", ["A"], [["a", "b", "c"]]),
            (
                [0, "D"],
                "A",
                ["A"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ],
            ),
            (0, None, ["A", "B"], [["a", "b", "c"]] * 2),
        ],
    )
    def test_box_plot_by_0(self, by, column, titles, xticklabels, hist_df):
        # GH 15079
        df = hist_df.copy()
        df = df.rename(columns={"C": 0})

        axes = _check_plot_works(df.plot.box, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]

        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),
            ((), "A"),
            ([], None),
            ((), ["A", "B"]),
        ],
    )
    def test_box_plot_with_none_empty_list_by(self, by, column, hist_df):
        # GH 15079
        msg = "No group keys passed"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [
            (["C"], "A", (1, 1), 1),
            ("C", "A", (1, 1), 1),
            ("C", None, (2, 1), 2),
            ("C", ["A", "B"], (1, 2), 2),
            (["C", "D"], "A", (1, 1), 1),
            (["C", "D"], None, (1, 2), 2),
        ],
    )
    def test_box_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
        # GH 15079
        axes = _check_plot_works(
            hist_df.plot.box, default_axes=True, column=column, by=by, layout=layout
        )
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [
            ("larger than required size", ["C", "D"], (1, 1)),
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),
            ("At least one dimension of layout must be positive", "C", (-1, -1)),
        ],
    )
    def test_box_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
        # GH 15079, test if error is raised when invalid layout is given

        with pytest.raises(ValueError, match=msg):
            hist_df.plot.box(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(self, figsize, hist_df):
        # GH 15079
        axes = hist_df.plot.box(column="A", by="C", figsize=figsize)
        _check_axes_shape(axes, axes_num=1, figsize=figsize)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\diffgeom\tests\test_diffgeom.py ===
from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
        WedgeProduct, BaseCovarDerivativeOp, CovarDerivativeOp, LieDerivative,
        covariant_order, contravariant_order, twoform_to_matrix, metric_to_Christoffel_1st,
        metric_to_Christoffel_2nd, metric_to_Riemann_components,
        metric_to_Ricci_components, intcurve_diffequ, intcurve_series)
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy

TP = TensorProduct


def test_coordsys_transform():
    # test inverse transforms
    p, q, r, s = symbols('p q r s')
    rel = {('first', 'second'): [(p, q), (q, -p)]}
    R2_pq = CoordSystem('first', R2_origin, [p, q], rel)
    R2_rs = CoordSystem('second', R2_origin, [r, s], rel)
    r, s = R2_rs.symbols
    assert R2_rs.transform(R2_pq) == Matrix([[-s], [r]])

    # inverse transform impossible case
    a, b = symbols('a b', positive=True)
    rel = {('first', 'second'): [(a,), (-a,)]}
    R2_a = CoordSystem('first', R2_origin, [a], rel)
    R2_b = CoordSystem('second', R2_origin, [b], rel)
    # This transformation is uninvertible because there is no positive a, b satisfying a = -b
    with raises(NotImplementedError):
        R2_b.transform(R2_a)

    # inverse transform ambiguous case
    c, d = symbols('c d')
    rel = {('first', 'second'): [(c,), (c**2,)]}
    R2_c = CoordSystem('first', R2_origin, [c], rel)
    R2_d = CoordSystem('second', R2_origin, [d], rel)
    # The transform method should throw if it finds multiple inverses for a coordinate transformation.
    with raises(ValueError):
        R2_d.transform(R2_c)

    # test indirect transformation
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2*a, 3*b)],
        ('C2', 'C3'): [(c, d), (3*c, 2*d)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, d/3])
    assert C1.transform(C3) == Matrix([6*a, 6*b])
    assert C3.transform(C1) == Matrix([e/6, f/6])
    assert C3.transform(C2) == Matrix([e/3, f/2])

    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2*a, 3*b + 1)],
        ('C3', 'C2'): [(e, f), (-e - 2, 2*f)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, (d - 1)/3])
    assert C1.transform(C3) == Matrix([-2*a - 2, (3*b + 1)/2])
    assert C3.transform(C1) == Matrix([-e/2 - 1, (2*f - 1)/3])
    assert C3.transform(C2) == Matrix([-e - 2, 2*f])

    # old signature uses Lambda
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): Lambda((a, b), (2*a, 3*b + 1)),
        ('C3', 'C2'): Lambda((e, f), (-e - 2, 2*f))}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, (d - 1)/3])
    assert C1.transform(C3) == Matrix([-2*a - 2, (3*b + 1)/2])
    assert C3.transform(C1) == Matrix([-e/2 - 1, (2*f - 1)/3])
    assert C3.transform(C2) == Matrix([-e - 2, 2*f])


def test_R2():
    x0, y0, r0, theta0 = symbols('x0, y0, r0, theta0', real=True)
    point_r = R2_r.point([x0, y0])
    point_p = R2_p.point([r0, theta0])

    # r**2 = x**2 + y**2
    assert (R2.r**2 - R2.x**2 - R2.y**2).rcall(point_r) == 0
    assert trigsimp( (R2.r**2 - R2.x**2 - R2.y**2).rcall(point_p) ) == 0
    assert trigsimp(R2.e_r(R2.x**2 + R2.y**2).rcall(point_p).doit()) == 2*r0

    # polar->rect->polar == Id
    a, b = symbols('a b', positive=True)
    m = Matrix([[a], [b]])

    #TODO assert m == R2_r.transform(R2_p, R2_p.transform(R2_r, [a, b])).applyfunc(simplify)
    assert m == R2_p.transform(R2_r, R2_r.transform(R2_p, m)).applyfunc(simplify)

    # deprecated method
    with warns_deprecated_sympy():
        assert m == R2_p.coord_tuple_transform_to(
            R2_r, R2_r.coord_tuple_transform_to(R2_p, m)).applyfunc(simplify)


def test_R3():
    a, b, c = symbols('a b c', positive=True)
    m = Matrix([[a], [b], [c]])

    assert m == R3_c.transform(R3_r, R3_r.transform(R3_c, m)).applyfunc(simplify)
    #TODO assert m == R3_r.transform(R3_c, R3_c.transform(R3_r, m)).applyfunc(simplify)
    assert m == R3_s.transform(
        R3_r, R3_r.transform(R3_s, m)).applyfunc(simplify)
    #TODO assert m == R3_r.transform(R3_s, R3_s.transform(R3_r, m)).applyfunc(simplify)
    assert m == R3_s.transform(
        R3_c, R3_c.transform(R3_s, m)).applyfunc(simplify)
    #TODO assert m == R3_c.transform(R3_s, R3_s.transform(R3_c, m)).applyfunc(simplify)

    with warns_deprecated_sympy():
        assert m == R3_c.coord_tuple_transform_to(
            R3_r, R3_r.coord_tuple_transform_to(R3_c, m)).applyfunc(simplify)
        #TODO assert m == R3_r.coord_tuple_transform_to(R3_c, R3_c.coord_tuple_transform_to(R3_r, m)).applyfunc(simplify)
        assert m == R3_s.coord_tuple_transform_to(
            R3_r, R3_r.coord_tuple_transform_to(R3_s, m)).applyfunc(simplify)
        #TODO assert m == R3_r.coord_tuple_transform_to(R3_s, R3_s.coord_tuple_transform_to(R3_r, m)).applyfunc(simplify)
        assert m == R3_s.coord_tuple_transform_to(
            R3_c, R3_c.coord_tuple_transform_to(R3_s, m)).applyfunc(simplify)
        #TODO assert m == R3_c.coord_tuple_transform_to(R3_s, R3_s.coord_tuple_transform_to(R3_c, m)).applyfunc(simplify)


def test_CoordinateSymbol():
    x, y = R2_r.symbols
    r, theta = R2_p.symbols
    assert y.rewrite(R2_p) == r*sin(theta)


def test_point():
    x, y = symbols('x, y')
    p = R2_r.point([x, y])
    assert p.free_symbols == {x, y}
    assert p.coords(R2_r) == p.coords() == Matrix([x, y])
    assert p.coords(R2_p) == Matrix([sqrt(x**2 + y**2), atan2(y, x)])


def test_commutator():
    assert Commutator(R2.e_x, R2.e_y) == 0
    assert Commutator(R2.x*R2.e_x, R2.x*R2.e_x) == 0
    assert Commutator(R2.x*R2.e_x, R2.x*R2.e_y) == R2.x*R2.e_y
    c = Commutator(R2.e_x, R2.e_r)
    assert c(R2.x) == R2.y*(R2.x**2 + R2.y**2)**(-1)*sin(R2.theta)


def test_differential():
    xdy = R2.x*R2.dy
    dxdy = Differential(xdy)
    assert xdy.rcall(None) == xdy
    assert dxdy(R2.e_x, R2.e_y) == 1
    assert dxdy(R2.e_x, R2.x*R2.e_y) == R2.x
    assert Differential(dxdy) == 0


def test_products():
    assert TensorProduct(
        R2.dx, R2.dy)(R2.e_x, R2.e_y) == R2.dx(R2.e_x)*R2.dy(R2.e_y) == 1
    assert TensorProduct(R2.dx, R2.dy)(None, R2.e_y) == R2.dx
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x, None) == R2.dy
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x) == R2.dy
    assert TensorProduct(R2.x, R2.dx) == R2.x*R2.dx
    assert TensorProduct(
        R2.e_x, R2.e_y)(R2.x, R2.y) == R2.e_x(R2.x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.e_x, R2.e_y)(None, R2.y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x, None) == R2.e_y
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(
        R2.dx, R2.e_y)(R2.e_x, R2.y) == R2.dx(R2.e_x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.dx, R2.e_y)(None, R2.y) == R2.dx
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x, None) == R2.e_y
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(
        R2.e_x, R2.dy)(R2.x, R2.e_y) == R2.e_x(R2.x) * R2.dy(R2.e_y) == 1
    assert TensorProduct(R2.e_x, R2.dy)(None, R2.e_y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.dy)(R2.x, None) == R2.dy
    assert TensorProduct(R2.e_x, R2.dy)(R2.x) == R2.dy
    assert TensorProduct(R2.e_y,R2.e_x)(R2.x**2 + R2.y**2,R2.x**2 + R2.y**2) == 4*R2.x*R2.y

    assert WedgeProduct(R2.dx, R2.dy)(R2.e_x, R2.e_y) == 1
    assert WedgeProduct(R2.e_x, R2.e_y)(R2.x, R2.y) == 1


def test_lie_derivative():
    assert LieDerivative(R2.e_x, R2.y) == R2.e_x(R2.y) == 0
    assert LieDerivative(R2.e_x, R2.x) == R2.e_x(R2.x) == 1
    assert LieDerivative(R2.e_x, R2.e_x) == Commutator(R2.e_x, R2.e_x) == 0
    assert LieDerivative(R2.e_x, R2.e_r) == Commutator(R2.e_x, R2.e_r)
    assert LieDerivative(R2.e_x + R2.e_y, R2.x) == 1
    assert LieDerivative(
        R2.e_x, TensorProduct(R2.dx, R2.dy))(R2.e_x, R2.e_y) == 0


@nocache_fail
def test_covar_deriv():
    ch = metric_to_Christoffel_2nd(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    assert cvd(R2.x) == 1
    # This line fails if the cache is disabled:
    assert cvd(R2.x*R2.e_x) == R2.e_x
    cvd = CovarDerivativeOp(R2.x*R2.e_x, ch)
    assert cvd(R2.x) == R2.x
    assert cvd(R2.x*R2.e_x) == R2.x*R2.e_x


def test_intcurve_diffequ():
    t = symbols('t')
    start_point = R2_r.point([1, 0])
    vector_field = -R2.y*R2.e_x + R2.x*R2.e_y
    equations, init_cond = intcurve_diffequ(vector_field, t, start_point)
    assert str(equations) == '[f_1(t) + Derivative(f_0(t), t), -f_0(t) + Derivative(f_1(t), t)]'
    assert str(init_cond) == '[f_0(0) - 1, f_1(0)]'
    equations, init_cond = intcurve_diffequ(vector_field, t, start_point, R2_p)
    assert str(
        equations) == '[Derivative(f_0(t), t), Derivative(f_1(t), t) - 1]'
    assert str(init_cond) == '[f_0(0) - 1, f_1(0)]'


def test_helpers_and_coordinate_dependent():
    one_form = R2.dr + R2.dx
    two_form = Differential(R2.x*R2.dr + R2.r*R2.dx)
    three_form = Differential(
        R2.y*two_form) + Differential(R2.x*Differential(R2.r*R2.dr))
    metric = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dy, R2.dy)
    metric_ambig = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dr, R2.dr)
    misform_a = TensorProduct(R2.dr, R2.dr) + R2.dr
    misform_b = R2.dr**4
    misform_c = R2.dx*R2.dy
    twoform_not_sym = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dx, R2.dy)
    twoform_not_TP = WedgeProduct(R2.dx, R2.dy)

    one_vector = R2.e_x + R2.e_y
    two_vector = TensorProduct(R2.e_x, R2.e_y)
    three_vector = TensorProduct(R2.e_x, R2.e_y, R2.e_x)
    two_wp = WedgeProduct(R2.e_x,R2.e_y)

    assert covariant_order(one_form) == 1
    assert covariant_order(two_form) == 2
    assert covariant_order(three_form) == 3
    assert covariant_order(two_form + metric) == 2
    assert covariant_order(two_form + metric_ambig) == 2
    assert covariant_order(two_form + twoform_not_sym) == 2
    assert covariant_order(two_form + twoform_not_TP) == 2

    assert contravariant_order(one_vector) == 1
    assert contravariant_order(two_vector) == 2
    assert contravariant_order(three_vector) == 3
    assert contravariant_order(two_vector + two_wp) == 2

    raises(ValueError, lambda: covariant_order(misform_a))
    raises(ValueError, lambda: covariant_order(misform_b))
    raises(ValueError, lambda: covariant_order(misform_c))

    assert twoform_to_matrix(metric) == Matrix([[1, 0], [0, 1]])
    assert twoform_to_matrix(twoform_not_sym) == Matrix([[1, 0], [1, 0]])
    assert twoform_to_matrix(twoform_not_TP) == Matrix([[0, -1], [1, 0]])

    raises(ValueError, lambda: twoform_to_matrix(one_form))
    raises(ValueError, lambda: twoform_to_matrix(three_form))
    raises(ValueError, lambda: twoform_to_matrix(metric_ambig))

    raises(ValueError, lambda: metric_to_Christoffel_1st(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Christoffel_2nd(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Riemann_components(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Ricci_components(twoform_not_sym))


def test_correct_arguments():
    raises(ValueError, lambda: R2.e_x(R2.e_x))
    raises(ValueError, lambda: R2.e_x(R2.dx))

    raises(ValueError, lambda: Commutator(R2.e_x, R2.x))
    raises(ValueError, lambda: Commutator(R2.dx, R2.e_x))

    raises(ValueError, lambda: Differential(Differential(R2.e_x)))

    raises(ValueError, lambda: R2.dx(R2.x))

    raises(ValueError, lambda: LieDerivative(R2.dx, R2.dx))
    raises(ValueError, lambda: LieDerivative(R2.x, R2.dx))

    raises(ValueError, lambda: CovarDerivativeOp(R2.dx, []))
    raises(ValueError, lambda: CovarDerivativeOp(R2.x, []))

    a = Symbol('a')
    raises(ValueError, lambda: intcurve_series(R2.dx, a, R2_r.point([1, 2])))
    raises(ValueError, lambda: intcurve_series(R2.x, a, R2_r.point([1, 2])))

    raises(ValueError, lambda: intcurve_diffequ(R2.dx, a, R2_r.point([1, 2])))
    raises(ValueError, lambda: intcurve_diffequ(R2.x, a, R2_r.point([1, 2])))

    raises(ValueError, lambda: contravariant_order(R2.e_x + R2.dx))
    raises(ValueError, lambda: covariant_order(R2.e_x + R2.dx))

    raises(ValueError, lambda: contravariant_order(R2.e_x*R2.e_y))
    raises(ValueError, lambda: covariant_order(R2.dx*R2.dy))

def test_simplify():
    x, y = R2_r.coord_functions()
    dx, dy = R2_r.base_oneforms()
    ex, ey = R2_r.base_vectors()
    assert simplify(x) == x
    assert simplify(x*y) == x*y
    assert simplify(dx*dy) == dx*dy
    assert simplify(ex*ey) == ex*ey
    assert ((1-x)*dx)/(1-x)**2 == dx/(1-x)


def test_issue_17917():
    X = R2.x*R2.e_x - R2.y*R2.e_y
    Y = (R2.x**2 + R2.y**2)*R2.e_x - R2.x*R2.y*R2.e_y
    assert LieDerivative(X, Y).expand() == (
        R2.x**2*R2.e_x - 3*R2.y**2*R2.e_x - R2.x*R2.y*R2.e_y)

def test_deprecations():
    m = Manifold('M', 2)
    p = Patch('P', m)
    with warns_deprecated_sympy():
        CoordSystem('Car2d', p, names=['x', 'y'])

    with warns_deprecated_sympy():
        c = CoordSystem('Car2d', p, ['x', 'y'])

    with warns_deprecated_sympy():
        list(m.patches)

    with warns_deprecated_sympy():
        list(c.transforms)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_vector.py ===
from sympy.core import Rational, S, Add, Mul, I
from sympy.simplify import simplify, trigsimp
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.vector import Vector, BaseVector, VectorAdd, \
     VectorMul, VectorZero
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Cross, Dot, cross
from sympy.testing.pytest import raises
from sympy.vector.kind import VectorKind
from sympy.core.kind import NumberKind
from sympy.testing.pytest import XFAIL


C = CoordSys3D('C')

i, j, k = C.base_vectors()
a, b, c = symbols('a b c')


def test_cross():
    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert Cross(v1, v2) == Cross(C.x*C.i + C.z**2*C.j, C.x*C.i + C.y*C.j + C.z*C.k)
    assert Cross(v1, v2).doit() == C.z**3*C.i + (-C.x*C.z)*C.j + (C.x*C.y - C.x*C.z**2)*C.k
    assert cross(v1, v2) == C.z**3*C.i + (-C.x*C.z)*C.j + (C.x*C.y - C.x*C.z**2)*C.k
    assert Cross(v1, v2) == -Cross(v2, v1)
    # XXX: Cannot use Cross here. See XFAIL test below:
    assert cross(v1, v2) + cross(v2, v1) == Vector.zero


@XFAIL
def test_cross_xfail():
    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert Cross(v1, v2) + Cross(v2, v1) == Vector.zero


def test_dot():
    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert Dot(v1, v2) == Dot(C.x*C.i + C.z**2*C.j, C.x*C.i + C.y*C.j + C.z*C.k)
    assert Dot(v1, v2).doit() == C.x**2 + C.y*C.z**2
    assert Dot(v2, v1).doit() == C.x**2 + C.y*C.z**2
    assert Dot(v1, v2) == Dot(v2, v1)


def test_vector_sympy():
    """
    Test whether the Vector framework confirms to the hashing
    and equality testing properties of SymPy.
    """
    v1 = 3*j
    assert v1 == j*3
    assert v1.components == {j: 3}
    v2 = 3*i + 4*j + 5*k
    v3 = 2*i + 4*j + i + 4*k + k
    assert v3 == v2
    assert v3.__hash__() == v2.__hash__()


def test_kind():
    assert C.i.kind is VectorKind(NumberKind)
    assert C.j.kind is VectorKind(NumberKind)
    assert C.k.kind is VectorKind(NumberKind)

    assert C.x.kind is NumberKind
    assert C.y.kind is NumberKind
    assert C.z.kind is NumberKind

    assert Mul._kind_dispatcher(NumberKind, VectorKind(NumberKind)) is VectorKind(NumberKind)
    assert Mul(2, C.i).kind is VectorKind(NumberKind)

    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert v1.kind is VectorKind(NumberKind)
    assert v2.kind is VectorKind(NumberKind)

    assert (v1 + v2).kind is VectorKind(NumberKind)
    assert Add(v1, v2).kind is VectorKind(NumberKind)
    assert Cross(v1, v2).doit().kind is VectorKind(NumberKind)
    assert VectorAdd(v1, v2).kind is VectorKind(NumberKind)
    assert VectorMul(2, v1).kind is VectorKind(NumberKind)
    assert VectorZero().kind is VectorKind(NumberKind)

    assert v1.projection(v2).kind is VectorKind(NumberKind)
    assert v2.projection(v1).kind is VectorKind(NumberKind)


def test_vectoradd():
    assert isinstance(Add(C.i, C.j), VectorAdd)
    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert isinstance(Add(v1, v2), VectorAdd)

    # https://github.com/sympy/sympy/issues/26121

    E = Matrix([C.i, C.j, C.k]).T
    a = Matrix([1, 2, 3])
    av = E*a

    assert av[0].kind == VectorKind()
    assert isinstance(av[0], VectorAdd)


def test_vector():
    assert isinstance(i, BaseVector)
    assert i != j
    assert j != k
    assert k != i
    assert i - i == Vector.zero
    assert i + Vector.zero == i
    assert i - Vector.zero == i
    assert Vector.zero != 0
    assert -Vector.zero == Vector.zero

    v1 = a*i + b*j + c*k
    v2 = a**2*i + b**2*j + c**2*k
    v3 = v1 + v2
    v4 = 2 * v1
    v5 = a * i

    assert isinstance(v1, VectorAdd)
    assert v1 - v1 == Vector.zero
    assert v1 + Vector.zero == v1
    assert v1.dot(i) == a
    assert v1.dot(j) == b
    assert v1.dot(k) == c
    assert i.dot(v2) == a**2
    assert j.dot(v2) == b**2
    assert k.dot(v2) == c**2
    assert v3.dot(i) == a**2 + a
    assert v3.dot(j) == b**2 + b
    assert v3.dot(k) == c**2 + c

    assert v1 + v2 == v2 + v1
    assert v1 - v2 == -1 * (v2 - v1)
    assert a * v1 == v1 * a

    assert isinstance(v5, VectorMul)
    assert v5.base_vector == i
    assert v5.measure_number == a
    assert isinstance(v4, Vector)
    assert isinstance(v4, VectorAdd)
    assert isinstance(v4, Vector)
    assert isinstance(Vector.zero, VectorZero)
    assert isinstance(Vector.zero, Vector)
    assert isinstance(v1 * 0, VectorZero)

    assert v1.to_matrix(C) == Matrix([[a], [b], [c]])

    assert i.components == {i: 1}
    assert v5.components == {i: a}
    assert v1.components == {i: a, j: b, k: c}

    assert VectorAdd(v1, Vector.zero) == v1
    assert VectorMul(a, v1) == v1*a
    assert VectorMul(1, i) == i
    assert VectorAdd(v1, Vector.zero) == v1
    assert VectorMul(0, Vector.zero) == Vector.zero
    raises(TypeError, lambda: v1.outer(1))
    raises(TypeError, lambda: v1.dot(1))


def test_vector_magnitude_normalize():
    assert Vector.zero.magnitude() == 0
    assert Vector.zero.normalize() == Vector.zero

    assert i.magnitude() == 1
    assert j.magnitude() == 1
    assert k.magnitude() == 1
    assert i.normalize() == i
    assert j.normalize() == j
    assert k.normalize() == k

    v1 = a * i
    assert v1.normalize() == (a/sqrt(a**2))*i
    assert v1.magnitude() == sqrt(a**2)

    v2 = a*i + b*j + c*k
    assert v2.magnitude() == sqrt(a**2 + b**2 + c**2)
    assert v2.normalize() == v2 / v2.magnitude()

    v3 = i + j
    assert v3.normalize() == (sqrt(2)/2)*C.i + (sqrt(2)/2)*C.j


def test_vector_simplify():
    A, s, k, m = symbols('A, s, k, m')

    test1 = (1 / a + 1 / b) * i
    assert (test1 & i) != (a + b) / (a * b)
    test1 = simplify(test1)
    assert (test1 & i) == (a + b) / (a * b)
    assert test1.simplify() == simplify(test1)

    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * i
    test2 = simplify(test2)
    assert (test2 & i) == (A**2 * s**4 / (4 * pi * k * m**3))

    test3 = ((4 + 4 * a - 2 * (2 + 2 * a)) / (2 + 2 * a)) * i
    test3 = simplify(test3)
    assert (test3 & i) == 0

    test4 = ((-4 * a * b**2 - 2 * b**3 - 2 * a**2 * b) / (a + b)**2) * i
    test4 = simplify(test4)
    assert (test4 & i) == -2 * b

    v = (sin(a)+cos(a))**2*i - j
    assert trigsimp(v) == (2*sin(a + pi/4)**2)*i + (-1)*j
    assert trigsimp(v) == v.trigsimp()

    assert simplify(Vector.zero) == Vector.zero


def test_vector_equals():
    assert (2*i).equals(j) is False
    assert i.equals(i) is True

    # https://github.com/sympy/sympy/issues/25915
    A = (sqrt(2) + sqrt(6)) / sqrt(sqrt(3) + 2)
    assert (A*i).equals(2*i) is True
    assert (A*i).equals(3*i) is False

    # Test comparing vectors in different coordinate systems
    D = C.orient_new_axis('D', pi/2, C.k)
    assert (D.i).equals(C.j) is True
    assert (D.i).equals(C.i) is False


def test_vector_conjugate():
    # https://github.com/sympy/sympy/issues/27094
    assert (I*i + (1 + I)*j + 2*k).conjugate() == -I*i + (1 - I)*j + 2*k


def test_vector_dot():
    assert i.dot(Vector.zero) == 0
    assert Vector.zero.dot(i) == 0
    assert i & Vector.zero == 0

    assert i.dot(i) == 1
    assert i.dot(j) == 0
    assert i.dot(k) == 0
    assert i & i == 1
    assert i & j == 0
    assert i & k == 0

    assert j.dot(i) == 0
    assert j.dot(j) == 1
    assert j.dot(k) == 0
    assert j & i == 0
    assert j & j == 1
    assert j & k == 0

    assert k.dot(i) == 0
    assert k.dot(j) == 0
    assert k.dot(k) == 1
    assert k & i == 0
    assert k & j == 0
    assert k & k == 1

    raises(TypeError, lambda: k.dot(1))


def test_vector_cross():
    assert i.cross(Vector.zero) == Vector.zero
    assert Vector.zero.cross(i) == Vector.zero

    assert i.cross(i) == Vector.zero
    assert i.cross(j) == k
    assert i.cross(k) == -j
    assert i ^ i == Vector.zero
    assert i ^ j == k
    assert i ^ k == -j

    assert j.cross(i) == -k
    assert j.cross(j) == Vector.zero
    assert j.cross(k) == i
    assert j ^ i == -k
    assert j ^ j == Vector.zero
    assert j ^ k == i

    assert k.cross(i) == j
    assert k.cross(j) == -i
    assert k.cross(k) == Vector.zero
    assert k ^ i == j
    assert k ^ j == -i
    assert k ^ k == Vector.zero

    assert k.cross(1) == Cross(k, 1)


def test_projection():
    v1 = i + j + k
    v2 = 3*i + 4*j
    v3 = 0*i + 0*j
    assert v1.projection(v1) == i + j + k
    assert v1.projection(v2) == Rational(7, 3)*C.i + Rational(7, 3)*C.j + Rational(7, 3)*C.k
    assert v1.projection(v1, scalar=True) == S.One
    assert v1.projection(v2, scalar=True) == Rational(7, 3)
    assert v3.projection(v1) == Vector.zero
    assert v3.projection(v1, scalar=True) == S.Zero


def test_vector_diff_integrate():
    f = Function('f')
    v = f(a)*C.i + a**2*C.j - C.k
    assert Derivative(v, a) == Derivative((f(a))*C.i +
                                          a**2*C.j + (-1)*C.k, a)
    assert (diff(v, a) == v.diff(a) == Derivative(v, a).doit() ==
            (Derivative(f(a), a))*C.i + 2*a*C.j)
    assert (Integral(v, a) == (Integral(f(a), a))*C.i +
            (Integral(a**2, a))*C.j + (Integral(-1, a))*C.k)


def test_vector_args():
    raises(ValueError, lambda: BaseVector(3, C))
    raises(TypeError, lambda: BaseVector(0, Vector.zero))


def test_srepr():
    from sympy.printing.repr import srepr
    res = "CoordSys3D(Str('C'), Tuple(ImmutableDenseMatrix([[Integer(1), "\
            "Integer(0), Integer(0)], [Integer(0), Integer(1), Integer(0)], "\
            "[Integer(0), Integer(0), Integer(1)]]), VectorZero())).i"
    assert srepr(C.i) == res


def test_scalar():
    from sympy.vector import CoordSys3D
    C = CoordSys3D('C')
    v1 = 3*C.i + 4*C.j + 5*C.k
    v2 = 3*C.i - 4*C.j + 5*C.k
    assert v1.is_Vector is True
    assert v1.is_scalar is False
    assert (v1.dot(v2)).is_scalar is True
    assert (v1.cross(v2)).is_scalar is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tools\test_to_timedelta.py ===
from datetime import (
    time,
    timedelta,
)

import numpy as np
import pytest

from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta

import pandas as pd
from pandas import (
    Series,
    TimedeltaIndex,
    isna,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestTimedeltas:
    def test_to_timedelta_dt64_raises(self):
        # Passing datetime64-dtype data to TimedeltaIndex is no longer
        #  supported GH#29794
        msg = r"dtype datetime64\[ns\] cannot be converted to timedelta64\[ns\]"

        ser = Series([pd.NaT])
        with pytest.raises(TypeError, match=msg):
            to_timedelta(ser)
        with pytest.raises(TypeError, match=msg):
            ser.to_frame().apply(to_timedelta)

    @pytest.mark.parametrize("readonly", [True, False])
    def test_to_timedelta_readonly(self, readonly):
        # GH#34857
        arr = np.array([], dtype=object)
        if readonly:
            arr.setflags(write=False)
        result = to_timedelta(arr)
        expected = to_timedelta([])
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_null(self):
        result = to_timedelta(["", ""])
        assert isna(result).all()

    def test_to_timedelta_same_np_timedelta64(self):
        # pass thru
        result = to_timedelta(np.array([np.timedelta64(1, "s")]))
        expected = pd.Index(np.array([np.timedelta64(1, "s")]))
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_series(self):
        # Series
        expected = Series([timedelta(days=1), timedelta(days=1, seconds=1)])
        result = to_timedelta(Series(["1d", "1days 00:00:01"]))
        tm.assert_series_equal(result, expected)

    def test_to_timedelta_units(self):
        # with units
        result = TimedeltaIndex(
            [np.timedelta64(0, "ns"), np.timedelta64(10, "s").astype("m8[ns]")]
        )
        expected = to_timedelta([0, 10], unit="s")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, unit",
        [
            ["int64", "s"],
            ["int64", "m"],
            ["int64", "h"],
            ["timedelta64[s]", "s"],
            ["timedelta64[D]", "D"],
        ],
    )
    def test_to_timedelta_units_dtypes(self, dtype, unit):
        # arrays of various dtypes
        arr = np.array([1] * 5, dtype=dtype)
        result = to_timedelta(arr, unit=unit)
        exp_dtype = "m8[ns]" if dtype == "int64" else "m8[s]"
        expected = TimedeltaIndex([np.timedelta64(1, unit)] * 5, dtype=exp_dtype)
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_oob_non_nano(self):
        arr = np.array([pd.NaT._value + 1], dtype="timedelta64[m]")

        msg = (
            "Cannot convert -9223372036854775807 minutes to "
            r"timedelta64\[s\] without overflow"
        )
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            to_timedelta(arr)

        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            TimedeltaIndex(arr)

        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            TimedeltaArray._from_sequence(arr, dtype="m8[s]")

    @pytest.mark.parametrize(
        "arg", [np.arange(10).reshape(2, 5), pd.DataFrame(np.arange(10).reshape(2, 5))]
    )
    @pytest.mark.parametrize("errors", ["ignore", "raise", "coerce"])
    @pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
    def test_to_timedelta_dataframe(self, arg, errors):
        # GH 11776
        with pytest.raises(TypeError, match="1-d array"):
            to_timedelta(arg, errors=errors)

    def test_to_timedelta_invalid_errors(self):
        # bad value for errors parameter
        msg = "errors must be one of"
        with pytest.raises(ValueError, match=msg):
            to_timedelta(["foo"], errors="never")

    @pytest.mark.parametrize("arg", [[1, 2], 1])
    def test_to_timedelta_invalid_unit(self, arg):
        # these will error
        msg = "invalid unit abbreviation: foo"
        with pytest.raises(ValueError, match=msg):
            to_timedelta(arg, unit="foo")

    def test_to_timedelta_time(self):
        # time not supported ATM
        msg = (
            "Value must be Timedelta, string, integer, float, timedelta or convertible"
        )
        with pytest.raises(ValueError, match=msg):
            to_timedelta(time(second=1))
        assert to_timedelta(time(second=1), errors="coerce") is pd.NaT

    def test_to_timedelta_bad_value(self):
        msg = "Could not convert 'foo' to NumPy timedelta"
        with pytest.raises(ValueError, match=msg):
            to_timedelta(["foo", "bar"])

    def test_to_timedelta_bad_value_coerce(self):
        tm.assert_index_equal(
            TimedeltaIndex([pd.NaT, pd.NaT]),
            to_timedelta(["foo", "bar"], errors="coerce"),
        )

        tm.assert_index_equal(
            TimedeltaIndex(["1 day", pd.NaT, "1 min"]),
            to_timedelta(["1 day", "bar", "1 min"], errors="coerce"),
        )

    def test_to_timedelta_invalid_errors_ignore(self):
        # gh-13613: these should not error because errors='ignore'
        msg = "errors='ignore' is deprecated"
        invalid_data = "apple"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_timedelta(invalid_data, errors="ignore")
        assert invalid_data == result

        invalid_data = ["apple", "1 days"]
        expected = np.array(invalid_data, dtype=object)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_timedelta(invalid_data, errors="ignore")
        tm.assert_numpy_array_equal(expected, result)

        invalid_data = pd.Index(["apple", "1 days"])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_timedelta(invalid_data, errors="ignore")
        tm.assert_index_equal(invalid_data, result)

        invalid_data = Series(["apple", "1 days"])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_timedelta(invalid_data, errors="ignore")
        tm.assert_series_equal(invalid_data, result)

    @pytest.mark.parametrize(
        "val, errors",
        [
            ("1M", True),
            ("1 M", True),
            ("1Y", True),
            ("1 Y", True),
            ("1y", True),
            ("1 y", True),
            ("1m", False),
            ("1 m", False),
            ("1 day", False),
            ("2day", False),
        ],
    )
    def test_unambiguous_timedelta_values(self, val, errors):
        # GH36666 Deprecate use of strings denoting units with 'M', 'Y', 'm' or 'y'
        # in pd.to_timedelta
        msg = "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta"
        if errors:
            with pytest.raises(ValueError, match=msg):
                to_timedelta(val)
        else:
            # check it doesn't raise
            to_timedelta(val)

    def test_to_timedelta_via_apply(self):
        # GH 5458
        expected = Series([np.timedelta64(1, "s")])
        result = Series(["00:00:01"]).apply(to_timedelta)
        tm.assert_series_equal(result, expected)

        result = Series([to_timedelta("00:00:01")])
        tm.assert_series_equal(result, expected)

    def test_to_timedelta_inference_without_warning(self):
        # GH#41731 inference produces a warning in the Series constructor,
        #  but _not_ in to_timedelta
        vals = ["00:00:01", pd.NaT]
        with tm.assert_produces_warning(None):
            result = to_timedelta(vals)

        expected = TimedeltaIndex([pd.Timedelta(seconds=1), pd.NaT])
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_on_missing_values(self):
        # GH5438
        timedelta_NaT = np.timedelta64("NaT")

        actual = to_timedelta(Series(["00:00:01", np.nan]))
        expected = Series(
            [np.timedelta64(1000000000, "ns"), timedelta_NaT],
            dtype=f"{tm.ENDIAN}m8[ns]",
        )
        tm.assert_series_equal(actual, expected)

        ser = Series(["00:00:01", pd.NaT], dtype="m8[ns]")
        actual = to_timedelta(ser)
        tm.assert_series_equal(actual, expected)

    @pytest.mark.parametrize("val", [np.nan, pd.NaT, pd.NA])
    def test_to_timedelta_on_missing_values_scalar(self, val):
        actual = to_timedelta(val)
        assert actual._value == np.timedelta64("NaT").astype("int64")

    @pytest.mark.parametrize("val", [np.nan, pd.NaT, pd.NA])
    def test_to_timedelta_on_missing_values_list(self, val):
        actual = to_timedelta([val])
        assert actual[0]._value == np.timedelta64("NaT").astype("int64")

    @pytest.mark.xfail(not IS64, reason="Floating point error")
    def test_to_timedelta_float(self):
        # https://github.com/pandas-dev/pandas/issues/25077
        arr = np.arange(0, 1, 1e-6)[-10:]
        result = to_timedelta(arr, unit="s")
        expected_asi8 = np.arange(999990000, 10**9, 1000, dtype="int64")
        tm.assert_numpy_array_equal(result.asi8, expected_asi8)

    def test_to_timedelta_coerce_strings_unit(self):
        arr = np.array([1, 2, "error"], dtype=object)
        result = to_timedelta(arr, unit="ns", errors="coerce")
        expected = to_timedelta([1, 2, pd.NaT], unit="ns")
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_ignore_strings_unit(self):
        arr = np.array([1, 2, "error"], dtype=object)
        msg = "errors='ignore' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_timedelta(arr, unit="ns", errors="ignore")
        tm.assert_numpy_array_equal(result, arr)

    @pytest.mark.parametrize(
        "expected_val, result_val", [[timedelta(days=2), 2], [None, None]]
    )
    def test_to_timedelta_nullable_int64_dtype(self, expected_val, result_val):
        # GH 35574
        expected = Series([timedelta(days=1), expected_val])
        result = to_timedelta(Series([1, result_val], dtype="Int64"), unit="days")

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        ("input", "expected"),
        [
            ("8:53:08.71800000001", "8:53:08.718"),
            ("8:53:08.718001", "8:53:08.718001"),
            ("8:53:08.7180000001", "8:53:08.7180000001"),
            ("-8:53:08.71800000001", "-8:53:08.718"),
            ("8:53:08.7180000089", "8:53:08.718000008"),
        ],
    )
    @pytest.mark.parametrize("func", [pd.Timedelta, to_timedelta])
    def test_to_timedelta_precision_over_nanos(self, input, expected, func):
        # GH: 36738
        expected = pd.Timedelta(expected)
        result = func(input)
        assert result == expected

    def test_to_timedelta_zerodim(self, fixed_now_ts):
        # ndarray.item() incorrectly returns int for dt64[ns] and td64[ns]
        dt64 = fixed_now_ts.to_datetime64()
        arg = np.array(dt64)

        msg = (
            "Value must be Timedelta, string, integer, float, timedelta "
            "or convertible, not datetime64"
        )
        with pytest.raises(ValueError, match=msg):
            to_timedelta(arg)

        arg2 = arg.view("m8[ns]")
        result = to_timedelta(arg2)
        assert isinstance(result, pd.Timedelta)
        assert result._value == dt64.view("i8")

    def test_to_timedelta_numeric_ea(self, any_numeric_ea_dtype):
        # GH#48796
        ser = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
        result = to_timedelta(ser)
        expected = Series([pd.Timedelta(1, unit="ns"), pd.NaT])
        tm.assert_series_equal(result, expected)

    def test_to_timedelta_fraction(self):
        result = to_timedelta(1.0 / 3, unit="h")
        expected = pd.Timedelta("0 days 00:19:59.999999998")
        assert result == expected


def test_from_numeric_arrow_dtype(any_numeric_ea_dtype):
    # GH 52425
    pytest.importorskip("pyarrow")
    ser = Series([1, 2], dtype=f"{any_numeric_ea_dtype.lower()}[pyarrow]")
    result = to_timedelta(ser)
    expected = Series([1, 2], dtype="timedelta64[ns]")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["ns", "ms"])
def test_from_timedelta_arrow_dtype(unit):
    # GH 54298
    pytest.importorskip("pyarrow")
    expected = Series([timedelta(1)], dtype=f"duration[{unit}][pyarrow]")
    result = to_timedelta(expected)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_body.py ===
from sympy import (Symbol, symbols, sin, cos, Matrix, zeros,
                                simplify)
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_default():
    with warns_deprecated_sympy():
        body = Body('body')
    assert body.name == 'body'
    assert body.loads == []
    point = Point('body_masscenter')
    point.set_vel(body.frame, 0)
    com = body.masscenter
    frame = body.frame
    assert com.vel(frame) == point.vel(frame)
    assert body.mass == Symbol('body_mass')
    ixx, iyy, izz = symbols('body_ixx body_iyy body_izz')
    ixy, iyz, izx = symbols('body_ixy body_iyz body_izx')
    assert body.inertia == (inertia(body.frame, ixx, iyy, izz, ixy, iyz, izx),
                            body.masscenter)


def test_custom_rigid_body():
    # Body with RigidBody.
    rigidbody_masscenter = Point('rigidbody_masscenter')
    rigidbody_mass = Symbol('rigidbody_mass')
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    with warns_deprecated_sympy():
        rigid_body = Body('rigidbody_body', rigidbody_masscenter,
                          rigidbody_mass, rigidbody_frame, body_inertia)
    com = rigid_body.masscenter
    frame = rigid_body.frame
    rigidbody_masscenter.set_vel(rigidbody_frame, 0)
    assert com.vel(frame) == rigidbody_masscenter.vel(frame)
    assert com.pos_from(com) == rigidbody_masscenter.pos_from(com)

    assert rigid_body.mass == rigidbody_mass
    assert rigid_body.inertia == (body_inertia, rigidbody_masscenter)

    assert rigid_body.is_rigidbody

    assert hasattr(rigid_body, 'masscenter')
    assert hasattr(rigid_body, 'mass')
    assert hasattr(rigid_body, 'frame')
    assert hasattr(rigid_body, 'inertia')


def test_particle_body():
    #  Body with Particle
    particle_masscenter = Point('particle_masscenter')
    particle_mass = Symbol('particle_mass')
    particle_frame = ReferenceFrame('particle_frame')
    with warns_deprecated_sympy():
        particle_body = Body('particle_body', particle_masscenter,
                             particle_mass, particle_frame)
    com = particle_body.masscenter
    frame = particle_body.frame
    particle_masscenter.set_vel(particle_frame, 0)
    assert com.vel(frame) == particle_masscenter.vel(frame)
    assert com.pos_from(com) == particle_masscenter.pos_from(com)

    assert particle_body.mass == particle_mass
    assert not hasattr(particle_body, "_inertia")
    assert hasattr(particle_body, 'frame')
    assert hasattr(particle_body, 'masscenter')
    assert hasattr(particle_body, 'mass')
    assert particle_body.inertia == (Dyadic(0), particle_body.masscenter)
    assert particle_body.central_inertia == Dyadic(0)
    assert not particle_body.is_rigidbody

    particle_body.central_inertia = inertia(particle_frame, 1, 1, 1)
    assert particle_body.central_inertia == inertia(particle_frame, 1, 1, 1)
    assert particle_body.is_rigidbody

    with warns_deprecated_sympy():
        particle_body = Body('particle_body', mass=particle_mass)
    assert not particle_body.is_rigidbody
    point = particle_body.masscenter.locatenew('point', particle_body.x)
    point_inertia = particle_mass * inertia(particle_body.frame, 0, 1, 1)
    particle_body.inertia = (point_inertia, point)
    assert particle_body.inertia == (point_inertia, point)
    assert particle_body.central_inertia == Dyadic(0)
    assert particle_body.is_rigidbody


def test_particle_body_add_force():
    #  Body with Particle
    particle_masscenter = Point('particle_masscenter')
    particle_mass = Symbol('particle_mass')
    particle_frame = ReferenceFrame('particle_frame')
    with warns_deprecated_sympy():
        particle_body = Body('particle_body', particle_masscenter,
                             particle_mass, particle_frame)

    a = Symbol('a')
    force_vector = a * particle_body.frame.x
    particle_body.apply_force(force_vector, particle_body.masscenter)
    assert len(particle_body.loads) == 1
    point = particle_body.masscenter.locatenew(
        particle_body._name + '_point0', 0)
    point.set_vel(particle_body.frame, 0)
    force_point = particle_body.loads[0][0]

    frame = particle_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    assert force_point.pos_from(force_point) == point.pos_from(force_point)

    assert particle_body.loads[0][1] == force_vector


def test_body_add_force():
    # Body with RigidBody.
    rigidbody_masscenter = Point('rigidbody_masscenter')
    rigidbody_mass = Symbol('rigidbody_mass')
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    with warns_deprecated_sympy():
        rigid_body = Body('rigidbody_body', rigidbody_masscenter,
                          rigidbody_mass, rigidbody_frame, body_inertia)

    l = Symbol('l')
    Fa = Symbol('Fa')
    point = rigid_body.masscenter.locatenew(
        'rigidbody_body_point0',
        l * rigid_body.frame.x)
    point.set_vel(rigid_body.frame, 0)
    force_vector = Fa * rigid_body.frame.z
    # apply_force with point
    rigid_body.apply_force(force_vector, point)
    assert len(rigid_body.loads) == 1
    force_point = rigid_body.loads[0][0]
    frame = rigid_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    assert force_point.pos_from(force_point) == point.pos_from(force_point)
    assert rigid_body.loads[0][1] == force_vector
    # apply_force without point
    rigid_body.apply_force(force_vector)
    assert len(rigid_body.loads) == 2
    assert rigid_body.loads[1][1] == force_vector
    # passing something else than point
    raises(TypeError, lambda: rigid_body.apply_force(force_vector,  0))
    raises(TypeError, lambda: rigid_body.apply_force(0))

def test_body_add_torque():
    with warns_deprecated_sympy():
        body = Body('body')
    torque_vector = body.frame.x
    body.apply_torque(torque_vector)

    assert len(body.loads) == 1
    assert body.loads[0] == (body.frame, torque_vector)
    raises(TypeError, lambda: body.apply_torque(0))

def test_body_masscenter_vel():
    with warns_deprecated_sympy():
        A = Body('A')
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        B = Body('B', frame=N)
    A.masscenter.set_vel(N, N.z)
    assert A.masscenter_vel(B) == N.z
    assert A.masscenter_vel(N) == N.z

def test_body_ang_vel():
    with warns_deprecated_sympy():
        A = Body('A')
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        B = Body('B', frame=N)
    A.frame.set_ang_vel(N, N.y)
    assert A.ang_vel_in(B) == N.y
    assert B.ang_vel_in(A) == -N.y
    assert A.ang_vel_in(N) == N.y

def test_body_dcm():
    with warns_deprecated_sympy():
        A = Body('A')
        B = Body('B')
    A.frame.orient_axis(B.frame, B.frame.z, 10)
    assert A.dcm(B) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])
    assert A.dcm(B.frame) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])

def test_body_axis():
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        B = Body('B', frame=N)
    assert B.x == N.x
    assert B.y == N.y
    assert B.z == N.z

def test_apply_force_multiple_one_point():
    a, b = symbols('a b')
    P = Point('P')
    with warns_deprecated_sympy():
        B = Body('B')
    f1 = a*B.x
    f2 = b*B.y
    B.apply_force(f1, P)
    assert B.loads == [(P, f1)]
    B.apply_force(f2, P)
    assert B.loads == [(P, f1+f2)]

def test_apply_force():
    f, g = symbols('f g')
    q, x, v1, v2 = dynamicsymbols('q x v1 v2')
    P1 = Point('P1')
    P2 = Point('P2')
    with warns_deprecated_sympy():
        B1 = Body('B1')
        B2 = Body('B2')
    N = ReferenceFrame('N')

    P1.set_vel(B1.frame, v1*B1.x)
    P2.set_vel(B2.frame, v2*B2.x)
    force = f*q*N.z # time varying force

    B1.apply_force(force, P1, B2, P2) #applying equal and opposite force on moving points
    assert B1.loads == [(P1, force)]
    assert B2.loads == [(P2, -force)]

    g1 = B1.mass*g*N.y
    g2 = B2.mass*g*N.y

    B1.apply_force(g1) #applying gravity on B1 masscenter
    B2.apply_force(g2) #applying gravity on B2 masscenter

    assert B1.loads == [(P1,force), (B1.masscenter, g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, g2)]

    force2 = x*N.x

    B1.apply_force(force2, reaction_body=B2) #Applying time varying force on masscenter

    assert B1.loads == [(P1, force), (B1.masscenter, force2+g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, -force2+g2)]

def test_apply_torque():
    t = symbols('t')
    q = dynamicsymbols('q')
    with warns_deprecated_sympy():
        B1 = Body('B1')
        B2 = Body('B2')
    N = ReferenceFrame('N')
    torque = t*q*N.x

    B1.apply_torque(torque, B2) #Applying equal and opposite torque
    assert B1.loads == [(B1.frame, torque)]
    assert B2.loads == [(B2.frame, -torque)]

    torque2 = t*N.y
    B1.apply_torque(torque2)
    assert B1.loads == [(B1.frame, torque+torque2)]

def test_clear_load():
    a = symbols('a')
    P = Point('P')
    with warns_deprecated_sympy():
        B = Body('B')
    force = a*B.z
    B.apply_force(force, P)
    assert B.loads == [(P, force)]
    B.clear_loads()
    assert B.loads == []

def test_remove_load():
    P1 = Point('P1')
    P2 = Point('P2')
    with warns_deprecated_sympy():
        B = Body('B')
    f1 = B.x
    f2 = B.y
    B.apply_force(f1, P1)
    B.apply_force(f2, P2)
    assert B.loads == [(P1, f1), (P2, f2)]
    B.remove_load(P2)
    assert B.loads == [(P1, f1)]
    B.apply_torque(f1.cross(f2))
    assert B.loads == [(P1, f1), (B.frame, f1.cross(f2))]
    B.remove_load()
    assert B.loads == [(P1, f1)]

def test_apply_loads_on_multi_degree_freedom_holonomic_system():
    """Example based on: https://pydy.readthedocs.io/en/latest/examples/multidof-holonomic.html"""
    with warns_deprecated_sympy():
        W = Body('W') #Wall
        B = Body('B') #Block
        P = Body('P') #Pendulum
        b = Body('b') #bob
    q1, q2 = dynamicsymbols('q1 q2') #generalized coordinates
    k, c, g, kT = symbols('k c g kT') #constants
    F, T = dynamicsymbols('F T') #Specified forces

    #Applying forces
    B.apply_force(F*W.x)
    W.apply_force(k*q1*W.x, reaction_body=B) #Spring force
    W.apply_force(c*q1.diff()*W.x, reaction_body=B) #dampner
    P.apply_force(P.mass*g*W.y)
    b.apply_force(b.mass*g*W.y)

    #Applying torques
    P.apply_torque(kT*q2*W.z, reaction_body=b)
    P.apply_torque(T*W.z)

    assert B.loads == [(B.masscenter, (F - k*q1 - c*q1.diff())*W.x)]
    assert P.loads == [(P.masscenter, P.mass*g*W.y), (P.frame, (T + kT*q2)*W.z)]
    assert b.loads == [(b.masscenter, b.mass*g*W.y), (b.frame, -kT*q2*W.z)]
    assert W.loads == [(W.masscenter, (c*q1.diff() + k*q1)*W.x)]


def test_parallel_axis():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    # Test RigidBody
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    with warns_deprecated_sympy():
        R = Body('R', masscenter=o, frame=N, mass=m, central_inertia=Io)
    Ip = R.parallel_axis(p)
    Ip_expected = inertia(N, Ix + m * b**2, Iy + m * a**2,
                          Iz + m * (a**2 + b**2), ixy=-m * a * b)
    assert Ip == Ip_expected
    # Reference frame from which the parallel axis is viewed should not matter
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, 1)
    assert simplify(
        (R.parallel_axis(p, A) - Ip_expected).to_matrix(A)) == zeros(3, 3)
    # Test Particle
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    with warns_deprecated_sympy():
        P = Body('P', masscenter=o, mass=m, frame=N)
    Ip = P.parallel_axis(p, N)
    Ip_expected = inertia(N, m * b ** 2, m * a ** 2, m * (a ** 2 + b ** 2),
                          ixy=-m * a * b)
    assert not P.is_rigidbody
    assert Ip == Ip_expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_year.py ===
"""
Tests for the following offsets:
- YearBegin
- YearEnd
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    YearBegin,
    YearEnd,
)


class TestYearBegin:
    def test_misspecified(self):
        with pytest.raises(ValueError, match="Month must go from 1 to 12"):
            YearBegin(month=13)

    offset_cases = []
    offset_cases.append(
        (
            YearBegin(),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 1),
                datetime(2005, 12, 31): datetime(2006, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 1),
                datetime(2005, 12, 31): datetime(2006, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(3),
            {
                datetime(2008, 1, 1): datetime(2011, 1, 1),
                datetime(2008, 6, 30): datetime(2011, 1, 1),
                datetime(2008, 12, 31): datetime(2011, 1, 1),
                datetime(2005, 12, 30): datetime(2008, 1, 1),
                datetime(2005, 12, 31): datetime(2008, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 1, 1),
                datetime(2007, 1, 15): datetime(2007, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 1, 1),
                datetime(2008, 12, 31): datetime(2008, 1, 1),
                datetime(2006, 12, 29): datetime(2006, 1, 1),
                datetime(2006, 12, 30): datetime(2006, 1, 1),
                datetime(2007, 1, 1): datetime(2006, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 1, 1),
                datetime(2008, 6, 30): datetime(2007, 1, 1),
                datetime(2008, 12, 31): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(month=4),
            {
                datetime(2007, 4, 1): datetime(2008, 4, 1),
                datetime(2007, 4, 15): datetime(2008, 4, 1),
                datetime(2007, 3, 1): datetime(2007, 4, 1),
                datetime(2007, 12, 15): datetime(2008, 4, 1),
                datetime(2012, 1, 31): datetime(2012, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(0, month=4),
            {
                datetime(2007, 4, 1): datetime(2007, 4, 1),
                datetime(2007, 3, 1): datetime(2007, 4, 1),
                datetime(2007, 12, 15): datetime(2008, 4, 1),
                datetime(2012, 1, 31): datetime(2012, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(4, month=4),
            {
                datetime(2007, 4, 1): datetime(2011, 4, 1),
                datetime(2007, 4, 15): datetime(2011, 4, 1),
                datetime(2007, 3, 1): datetime(2010, 4, 1),
                datetime(2007, 12, 15): datetime(2011, 4, 1),
                datetime(2012, 1, 31): datetime(2015, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-1, month=4),
            {
                datetime(2007, 4, 1): datetime(2006, 4, 1),
                datetime(2007, 3, 1): datetime(2006, 4, 1),
                datetime(2007, 12, 15): datetime(2007, 4, 1),
                datetime(2012, 1, 31): datetime(2011, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-3, month=4),
            {
                datetime(2007, 4, 1): datetime(2004, 4, 1),
                datetime(2007, 3, 1): datetime(2004, 4, 1),
                datetime(2007, 12, 15): datetime(2005, 4, 1),
                datetime(2012, 1, 31): datetime(2009, 4, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (YearBegin(), datetime(2007, 1, 3), False),
        (YearBegin(), datetime(2008, 1, 1), True),
        (YearBegin(), datetime(2006, 12, 31), False),
        (YearBegin(), datetime(2006, 1, 2), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestYearEnd:
    def test_misspecified(self):
        with pytest.raises(ValueError, match="Month must go from 1 to 12"):
            YearEnd(month=13)

    offset_cases = []
    offset_cases.append(
        (
            YearEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2009, 12, 31),
                datetime(2005, 12, 30): datetime(2005, 12, 31),
                datetime(2005, 12, 31): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2008, 12, 31),
                datetime(2005, 12, 30): datetime(2005, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2007, 12, 31),
                datetime(2008, 12, 31): datetime(2007, 12, 31),
                datetime(2006, 12, 29): datetime(2005, 12, 31),
                datetime(2006, 12, 30): datetime(2005, 12, 31),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 12, 31),
                datetime(2008, 6, 30): datetime(2006, 12, 31),
                datetime(2008, 12, 31): datetime(2006, 12, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (YearEnd(), datetime(2007, 12, 31), True),
        (YearEnd(), datetime(2008, 1, 1), False),
        (YearEnd(), datetime(2006, 12, 31), True),
        (YearEnd(), datetime(2006, 12, 29), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestYearEndDiffMonth:
    offset_cases = []
    offset_cases.append(
        (
            YearEnd(month=3),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 31),
                datetime(2008, 2, 15): datetime(2008, 3, 31),
                datetime(2008, 3, 31): datetime(2009, 3, 31),
                datetime(2008, 3, 30): datetime(2008, 3, 31),
                datetime(2005, 3, 31): datetime(2006, 3, 31),
                datetime(2006, 7, 30): datetime(2007, 3, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(0, month=3),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 31),
                datetime(2008, 2, 28): datetime(2008, 3, 31),
                datetime(2008, 3, 31): datetime(2008, 3, 31),
                datetime(2005, 3, 30): datetime(2005, 3, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-1, month=3),
            {
                datetime(2007, 1, 1): datetime(2006, 3, 31),
                datetime(2008, 2, 28): datetime(2007, 3, 31),
                datetime(2008, 3, 31): datetime(2007, 3, 31),
                datetime(2006, 3, 29): datetime(2005, 3, 31),
                datetime(2006, 3, 30): datetime(2005, 3, 31),
                datetime(2007, 3, 1): datetime(2006, 3, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-2, month=3),
            {
                datetime(2007, 1, 1): datetime(2005, 3, 31),
                datetime(2008, 6, 30): datetime(2007, 3, 31),
                datetime(2008, 3, 31): datetime(2006, 3, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (YearEnd(month=3), datetime(2007, 3, 31), True),
        (YearEnd(month=3), datetime(2008, 1, 1), False),
        (YearEnd(month=3), datetime(2006, 3, 31), True),
        (YearEnd(month=3), datetime(2006, 3, 29), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


def test_add_out_of_pydatetime_range():
    # GH#50348 don't raise in Timestamp.replace
    ts = Timestamp(np.datetime64("-20000-12-31"))
    off = YearEnd()

    result = ts + off
    # TODO(cython3): "arg: datetime" annotation will impose
    # datetime limitations on Timestamp. The fused type below works in cy3
    # ctypedef fused datetimelike:
    #     _Timestamp
    #     datetime
    # expected = Timestamp(np.datetime64("-19999-12-31"))
    # assert result == expected
    assert result.year in (-19999, 1973)
    assert result.month == 12
    assert result.day == 31

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_astype.py ===
from datetime import datetime

import dateutil
import numpy as np
import pytest
import pytz

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    PeriodIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndex:
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_astype_asobject_around_dst_transition(self, tzstr):
        # GH#1345

        # dates around a dst transition
        rng = date_range("2/13/2010", "5/6/2010", tz=tzstr)

        objs = rng.astype(object)
        for i, x in enumerate(objs):
            exval = rng[i]
            assert x == exval
            assert x.tzinfo == exval.tzinfo

        objs = rng.astype(object)
        for i, x in enumerate(objs):
            exval = rng[i]
            assert x == exval
            assert x.tzinfo == exval.tzinfo

    def test_astype(self):
        # GH 13149, GH 13209
        idx = DatetimeIndex(
            ["2016-05-16", "NaT", NaT, np.nan], dtype="M8[ns]", name="idx"
        )

        result = idx.astype(object)
        expected = Index(
            [Timestamp("2016-05-16")] + [NaT] * 3, dtype=object, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(np.int64)
        expected = Index(
            [1463356800000000000] + [-9223372036854775808] * 3,
            dtype=np.int64,
            name="idx",
        )
        tm.assert_index_equal(result, expected)

    def test_astype2(self):
        rng = date_range("1/1/2000", periods=10, name="idx")
        result = rng.astype("i8")
        tm.assert_index_equal(result, Index(rng.asi8, name="idx"))
        tm.assert_numpy_array_equal(result.values, rng.asi8)

    def test_astype_uint(self):
        arr = date_range("2000", periods=2, name="idx")

        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint64")
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint32")

    def test_astype_with_tz(self):
        # with tz
        rng = date_range("1/1/2000", periods=10, tz="US/Eastern")
        msg = "Cannot use .astype to convert from timezone-aware"
        with pytest.raises(TypeError, match=msg):
            # deprecated
            rng.astype("datetime64[ns]")
        with pytest.raises(TypeError, match=msg):
            # check DatetimeArray while we're here deprecated
            rng._data.astype("datetime64[ns]")

    def test_astype_tzaware_to_tzaware(self):
        # GH 18951: tz-aware to tz-aware
        idx = date_range("20170101", periods=4, tz="US/Pacific")
        result = idx.astype("datetime64[ns, US/Eastern]")
        expected = date_range("20170101 03:00:00", periods=4, tz="US/Eastern")
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_astype_tznaive_to_tzaware(self):
        # GH 18951: tz-naive to tz-aware
        idx = date_range("20170101", periods=4)
        idx = idx._with_freq(None)  # tz_localize does not preserve freq
        msg = "Cannot use .astype to convert from timezone-naive"
        with pytest.raises(TypeError, match=msg):
            # dt64->dt64tz deprecated
            idx.astype("datetime64[ns, US/Eastern]")
        with pytest.raises(TypeError, match=msg):
            # dt64->dt64tz deprecated
            idx._data.astype("datetime64[ns, US/Eastern]")

    def test_astype_str_nat(self, using_infer_string):
        # GH 13149, GH 13209
        # verify that we are returning NaT as a string (and not unicode)

        idx = DatetimeIndex(["2016-05-16", "NaT", NaT, np.nan])
        result = idx.astype(str)
        if using_infer_string:
            expected = Index(["2016-05-16", None, None, None], dtype="str")
        else:
            expected = Index(["2016-05-16", "NaT", "NaT", "NaT"], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_astype_str(self):
        # test astype string - #10442
        dti = date_range("2012-01-01", periods=4, name="test_name")
        result = dti.astype(str)
        expected = Index(
            ["2012-01-01", "2012-01-02", "2012-01-03", "2012-01-04"],
            name="test_name",
            dtype="str",
        )
        tm.assert_index_equal(result, expected)

    def test_astype_str_tz_and_name(self):
        # test astype string with tz and name
        dti = date_range("2012-01-01", periods=3, name="test_name", tz="US/Eastern")
        result = dti.astype(str)
        expected = Index(
            [
                "2012-01-01 00:00:00-05:00",
                "2012-01-02 00:00:00-05:00",
                "2012-01-03 00:00:00-05:00",
            ],
            name="test_name",
            dtype="str",
        )
        tm.assert_index_equal(result, expected)

    def test_astype_str_freq_and_name(self):
        # test astype string with freqH and name
        dti = date_range("1/1/2011", periods=3, freq="h", name="test_name")
        result = dti.astype(str)
        expected = Index(
            ["2011-01-01 00:00:00", "2011-01-01 01:00:00", "2011-01-01 02:00:00"],
            name="test_name",
            dtype="str",
        )
        tm.assert_index_equal(result, expected)

    def test_astype_str_freq_and_tz(self):
        # test astype string with freqH and timezone
        dti = date_range(
            "3/6/2012 00:00", periods=2, freq="h", tz="Europe/London", name="test_name"
        )
        result = dti.astype(str)
        expected = Index(
            ["2012-03-06 00:00:00+00:00", "2012-03-06 01:00:00+00:00"],
            dtype="str",
            name="test_name",
        )
        tm.assert_index_equal(result, expected)

    def test_astype_datetime64(self):
        # GH 13149, GH 13209
        idx = DatetimeIndex(
            ["2016-05-16", "NaT", NaT, np.nan], dtype="M8[ns]", name="idx"
        )

        result = idx.astype("datetime64[ns]")
        tm.assert_index_equal(result, idx)
        assert result is not idx

        result = idx.astype("datetime64[ns]", copy=False)
        tm.assert_index_equal(result, idx)
        assert result is idx

        idx_tz = DatetimeIndex(["2016-05-16", "NaT", NaT, np.nan], tz="EST", name="idx")
        msg = "Cannot use .astype to convert from timezone-aware"
        with pytest.raises(TypeError, match=msg):
            # dt64tz->dt64 deprecated
            result = idx_tz.astype("datetime64[ns]")

    def test_astype_object(self):
        rng = date_range("1/1/2000", periods=20)

        casted = rng.astype("O")
        exp_values = list(rng)

        tm.assert_index_equal(casted, Index(exp_values, dtype=np.object_))
        assert casted.tolist() == exp_values

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo"])
    def test_astype_object_tz(self, tz):
        idx = date_range(start="2013-01-01", periods=4, freq="ME", name="idx", tz=tz)
        expected_list = [
            Timestamp("2013-01-31", tz=tz),
            Timestamp("2013-02-28", tz=tz),
            Timestamp("2013-03-31", tz=tz),
            Timestamp("2013-04-30", tz=tz),
        ]
        expected = Index(expected_list, dtype=object, name="idx")
        result = idx.astype(object)
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    def test_astype_object_with_nat(self):
        idx = DatetimeIndex(
            [datetime(2013, 1, 1), datetime(2013, 1, 2), NaT, datetime(2013, 1, 4)],
            name="idx",
        )
        expected_list = [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-02"),
            NaT,
            Timestamp("2013-01-04"),
        ]
        expected = Index(expected_list, dtype=object, name="idx")
        result = idx.astype(object)
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    @pytest.mark.parametrize(
        "dtype",
        [float, "timedelta64", "timedelta64[ns]", "datetime64", "datetime64[D]"],
    )
    def test_astype_raises(self, dtype):
        # GH 13149, GH 13209
        idx = DatetimeIndex(["2016-05-16", "NaT", NaT, np.nan])
        msg = "Cannot cast DatetimeIndex to dtype"
        if dtype == "datetime64":
            msg = "Casting to unit-less dtype 'datetime64' is not supported"
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)

    def test_index_convert_to_datetime_array(self):
        def _check_rng(rng):
            converted = rng.to_pydatetime()
            assert isinstance(converted, np.ndarray)
            for x, stamp in zip(converted, rng):
                assert isinstance(x, datetime)
                assert x == stamp.to_pydatetime()
                assert x.tzinfo == stamp.tzinfo

        rng = date_range("20090415", "20090519")
        rng_eastern = date_range("20090415", "20090519", tz="US/Eastern")
        rng_utc = date_range("20090415", "20090519", tz="utc")

        _check_rng(rng)
        _check_rng(rng_eastern)
        _check_rng(rng_utc)

    def test_index_convert_to_datetime_array_explicit_pytz(self):
        def _check_rng(rng):
            converted = rng.to_pydatetime()
            assert isinstance(converted, np.ndarray)
            for x, stamp in zip(converted, rng):
                assert isinstance(x, datetime)
                assert x == stamp.to_pydatetime()
                assert x.tzinfo == stamp.tzinfo

        rng = date_range("20090415", "20090519")
        rng_eastern = date_range("20090415", "20090519", tz=pytz.timezone("US/Eastern"))
        rng_utc = date_range("20090415", "20090519", tz=pytz.utc)

        _check_rng(rng)
        _check_rng(rng_eastern)
        _check_rng(rng_utc)

    def test_index_convert_to_datetime_array_dateutil(self):
        def _check_rng(rng):
            converted = rng.to_pydatetime()
            assert isinstance(converted, np.ndarray)
            for x, stamp in zip(converted, rng):
                assert isinstance(x, datetime)
                assert x == stamp.to_pydatetime()
                assert x.tzinfo == stamp.tzinfo

        rng = date_range("20090415", "20090519")
        rng_eastern = date_range("20090415", "20090519", tz="dateutil/US/Eastern")
        rng_utc = date_range("20090415", "20090519", tz=dateutil.tz.tzutc())

        _check_rng(rng)
        _check_rng(rng_eastern)
        _check_rng(rng_utc)

    @pytest.mark.parametrize(
        "tz, dtype",
        [["US/Pacific", "datetime64[ns, US/Pacific]"], [None, "datetime64[ns]"]],
    )
    def test_integer_index_astype_datetime(self, tz, dtype):
        # GH 20997, 20964, 24559
        val = [Timestamp("2018-01-01", tz=tz).as_unit("ns")._value]
        result = Index(val, name="idx").astype(dtype)
        expected = DatetimeIndex(["2018-01-01"], tz=tz, name="idx").as_unit("ns")
        tm.assert_index_equal(result, expected)

    def test_dti_astype_period(self):
        idx = DatetimeIndex([NaT, "2011-01-01", "2011-02-01"], name="idx")

        res = idx.astype("period[M]")
        exp = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="M", name="idx")
        tm.assert_index_equal(res, exp)

        res = idx.astype("period[3M]")
        exp = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="3M", name="idx")
        tm.assert_index_equal(res, exp)


class TestAstype:
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_astype_category(self, tz):
        obj = date_range("2000", periods=2, tz=tz, name="idx")
        result = obj.astype("category")
        dti = DatetimeIndex(["2000-01-01", "2000-01-02"], tz=tz).as_unit("ns")
        expected = pd.CategoricalIndex(
            dti,
            name="idx",
        )
        tm.assert_index_equal(result, expected)

        result = obj._data.astype("category")
        expected = expected.values
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_astype_array_fallback(self, tz):
        obj = date_range("2000", periods=2, tz=tz, name="idx")
        result = obj.astype(bool)
        expected = Index(np.array([True, True]), name="idx")
        tm.assert_index_equal(result, expected)

        result = obj._data.astype(bool)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_nonunique_indexes.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameNonuniqueIndexes:
    def test_setattr_columns_vs_construct_with_columns(self):
        # assignment
        # GH 3687
        arr = np.random.default_rng(2).standard_normal((3, 2))
        idx = list(range(2))
        df = DataFrame(arr, columns=["A", "A"])
        df.columns = idx
        expected = DataFrame(arr, columns=idx)
        tm.assert_frame_equal(df, expected)

    def test_setattr_columns_vs_construct_with_columns_datetimeindx(self):
        idx = date_range("20130101", periods=4, freq="QE-NOV")
        df = DataFrame(
            [[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=["a", "a", "a", "a"]
        )
        df.columns = idx
        expected = DataFrame([[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=idx)
        tm.assert_frame_equal(df, expected)

    def test_insert_with_duplicate_columns(self):
        # insert
        df = DataFrame(
            [[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]],
            columns=["foo", "bar", "foo", "hello"],
        )
        df["string"] = "bah"
        expected = DataFrame(
            [[1, 1, 1, 5, "bah"], [1, 1, 2, 5, "bah"], [2, 1, 3, 5, "bah"]],
            columns=["foo", "bar", "foo", "hello", "string"],
        )
        tm.assert_frame_equal(df, expected)
        with pytest.raises(ValueError, match="Length of value"):
            df.insert(0, "AnotherColumn", range(len(df.index) - 1))

        # insert same dtype
        df["foo2"] = 3
        expected = DataFrame(
            [[1, 1, 1, 5, "bah", 3], [1, 1, 2, 5, "bah", 3], [2, 1, 3, 5, "bah", 3]],
            columns=["foo", "bar", "foo", "hello", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # set (non-dup)
        df["foo2"] = 4
        expected = DataFrame(
            [[1, 1, 1, 5, "bah", 4], [1, 1, 2, 5, "bah", 4], [2, 1, 3, 5, "bah", 4]],
            columns=["foo", "bar", "foo", "hello", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)
        df["foo2"] = 3

        # delete (non dup)
        del df["bar"]
        expected = DataFrame(
            [[1, 1, 5, "bah", 3], [1, 2, 5, "bah", 3], [2, 3, 5, "bah", 3]],
            columns=["foo", "foo", "hello", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # try to delete again (its not consolidated)
        del df["hello"]
        expected = DataFrame(
            [[1, 1, "bah", 3], [1, 2, "bah", 3], [2, 3, "bah", 3]],
            columns=["foo", "foo", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # consolidate
        df = df._consolidate()
        expected = DataFrame(
            [[1, 1, "bah", 3], [1, 2, "bah", 3], [2, 3, "bah", 3]],
            columns=["foo", "foo", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # insert
        df.insert(2, "new_col", 5.0)
        expected = DataFrame(
            [[1, 1, 5.0, "bah", 3], [1, 2, 5.0, "bah", 3], [2, 3, 5.0, "bah", 3]],
            columns=["foo", "foo", "new_col", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # insert a dup
        with pytest.raises(ValueError, match="cannot insert"):
            df.insert(2, "new_col", 4.0)

        df.insert(2, "new_col", 4.0, allow_duplicates=True)
        expected = DataFrame(
            [
                [1, 1, 4.0, 5.0, "bah", 3],
                [1, 2, 4.0, 5.0, "bah", 3],
                [2, 3, 4.0, 5.0, "bah", 3],
            ],
            columns=["foo", "foo", "new_col", "new_col", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        # delete (dup)
        del df["foo"]
        expected = DataFrame(
            [[4.0, 5.0, "bah", 3], [4.0, 5.0, "bah", 3], [4.0, 5.0, "bah", 3]],
            columns=["new_col", "new_col", "string", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

    def test_dup_across_dtypes(self):
        # dup across dtypes
        df = DataFrame(
            [[1, 1, 1.0, 5], [1, 1, 2.0, 5], [2, 1, 3.0, 5]],
            columns=["foo", "bar", "foo", "hello"],
        )

        df["foo2"] = 7.0
        expected = DataFrame(
            [[1, 1, 1.0, 5, 7.0], [1, 1, 2.0, 5, 7.0], [2, 1, 3.0, 5, 7.0]],
            columns=["foo", "bar", "foo", "hello", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        result = df["foo"]
        expected = DataFrame([[1, 1.0], [1, 2.0], [2, 3.0]], columns=["foo", "foo"])
        tm.assert_frame_equal(result, expected)

        # multiple replacements
        df["foo"] = "string"
        expected = DataFrame(
            [
                ["string", 1, "string", 5, 7.0],
                ["string", 1, "string", 5, 7.0],
                ["string", 1, "string", 5, 7.0],
            ],
            columns=["foo", "bar", "foo", "hello", "foo2"],
        )
        tm.assert_frame_equal(df, expected)

        del df["foo"]
        expected = DataFrame(
            [[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=["bar", "hello", "foo2"]
        )
        tm.assert_frame_equal(df, expected)

    def test_column_dups_indexes(self):
        # check column dups with index equal and not equal to df's index
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "A"],
        )
        for index in [df.index, pd.Index(list("edcba"))]:
            this_df = df.copy()
            expected_ser = Series(index.values, index=this_df.index)
            expected_df = DataFrame(
                {"A": expected_ser, "B": this_df["B"]},
                columns=["A", "B", "A"],
            )
            this_df["A"] = index
            tm.assert_frame_equal(this_df, expected_df)

    def test_changing_dtypes_with_duplicate_columns(self):
        # multiple assignments that change dtypes
        # the location indexer is a slice
        # GH 6120
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["that", "that"]
        )
        expected = DataFrame(1.0, index=range(5), columns=["that", "that"])

        df["that"] = 1.0
        tm.assert_frame_equal(df, expected)

        df = DataFrame(
            np.random.default_rng(2).random((5, 2)), columns=["that", "that"]
        )
        expected = DataFrame(1, index=range(5), columns=["that", "that"])

        df["that"] = 1
        tm.assert_frame_equal(df, expected)

    def test_dup_columns_comparisons(self):
        # equality
        df1 = DataFrame([[1, 2], [2, np.nan], [3, 4], [4, 4]], columns=["A", "B"])
        df2 = DataFrame([[0, 1], [2, 4], [2, np.nan], [4, 5]], columns=["A", "A"])

        # not-comparing like-labelled
        msg = (
            r"Can only compare identically-labeled \(both index and columns\) "
            "DataFrame objects"
        )
        with pytest.raises(ValueError, match=msg):
            df1 == df2

        df1r = df1.reindex_like(df2)
        result = df1r == df2
        expected = DataFrame(
            [[False, True], [True, False], [False, False], [True, False]],
            columns=["A", "A"],
        )
        tm.assert_frame_equal(result, expected)

    def test_mixed_column_selection(self):
        # mixed column selection
        # GH 5639
        dfbool = DataFrame(
            {
                "one": Series([True, True, False], index=["a", "b", "c"]),
                "two": Series([False, False, True, False], index=["a", "b", "c", "d"]),
                "three": Series([False, True, True, True], index=["a", "b", "c", "d"]),
            }
        )
        expected = pd.concat([dfbool["one"], dfbool["three"], dfbool["one"]], axis=1)
        result = dfbool[["one", "three", "one"]]
        tm.assert_frame_equal(result, expected)

    def test_multi_axis_dups(self):
        # multi-axis dups
        # GH 6121
        df = DataFrame(
            np.arange(25.0).reshape(5, 5),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "C", "D", "E"],
        )
        z = df[["A", "C", "A"]].copy()
        expected = z.loc[["a", "c", "a"]]

        df = DataFrame(
            np.arange(25.0).reshape(5, 5),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "C", "D", "E"],
        )
        z = df[["A", "C", "A"]]
        result = z.loc[["a", "c", "a"]]
        tm.assert_frame_equal(result, expected)

    def test_columns_with_dups(self):
        # GH 3468 related

        # basic
        df = DataFrame([[1, 2]], columns=["a", "a"])
        df.columns = ["a", "a.1"]
        expected = DataFrame([[1, 2]], columns=["a", "a.1"])
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[1, 2, 3]], columns=["b", "a", "a"])
        df.columns = ["b", "a", "a.1"]
        expected = DataFrame([[1, 2, 3]], columns=["b", "a", "a.1"])
        tm.assert_frame_equal(df, expected)

    def test_columns_with_dup_index(self):
        # with a dup index
        df = DataFrame([[1, 2]], columns=["a", "a"])
        df.columns = ["b", "b"]
        expected = DataFrame([[1, 2]], columns=["b", "b"])
        tm.assert_frame_equal(df, expected)

    def test_multi_dtype(self):
        # multi-dtype
        df = DataFrame(
            [[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]],
            columns=["a", "a", "b", "b", "d", "c", "c"],
        )
        df.columns = list("ABCDEFG")
        expected = DataFrame(
            [[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]], columns=list("ABCDEFG")
        )
        tm.assert_frame_equal(df, expected)

    def test_multi_dtype2(self):
        df = DataFrame([[1, 2, "foo", "bar"]], columns=["a", "a", "a", "a"])
        df.columns = ["a", "a.1", "a.2", "a.3"]
        expected = DataFrame([[1, 2, "foo", "bar"]], columns=["a", "a.1", "a.2", "a.3"])
        tm.assert_frame_equal(df, expected)

    def test_dups_across_blocks(self, using_array_manager):
        # dups across blocks
        df_float = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), dtype="float64"
        )
        df_int = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)).astype("int64")
        )
        df_bool = DataFrame(True, index=df_float.index, columns=df_float.columns)
        df_object = DataFrame("foo", index=df_float.index, columns=df_float.columns)
        df_dt = DataFrame(
            pd.Timestamp("20010101"), index=df_float.index, columns=df_float.columns
        )
        df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1)

        if not using_array_manager:
            assert len(df._mgr.blknos) == len(df.columns)
            assert len(df._mgr.blklocs) == len(df.columns)

        # testing iloc
        for i in range(len(df.columns)):
            df.iloc[:, i]

    def test_dup_columns_across_dtype(self):
        # dup columns across dtype GH 2079/2194
        vals = [[1, -1, 2.0], [2, -2, 3.0]]
        rs = DataFrame(vals, columns=["A", "A", "B"])
        xp = DataFrame(vals)
        xp.columns = ["A", "A", "B"]
        tm.assert_frame_equal(rs, xp)

    def test_set_value_by_index(self):
        # See gh-12344
        warn = None
        msg = "will attempt to set the values inplace"

        df = DataFrame(np.arange(9).reshape(3, 3).T)
        df.columns = list("AAA")
        expected = df.iloc[:, 2].copy()

        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[:, 0] = 3
        tm.assert_series_equal(df.iloc[:, 2], expected)

        df = DataFrame(np.arange(9).reshape(3, 3).T)
        df.columns = [2, float(2), str(2)]
        expected = df.iloc[:, 1].copy()

        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[:, 0] = 3
        tm.assert_series_equal(df.iloc[:, 1], expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\conftest.py ===
from __future__ import annotations

import os

import pytest

from pandas.compat import HAS_PYARROW
from pandas.compat._optional import VERSIONS

from pandas import (
    read_csv,
    read_table,
)
import pandas._testing as tm


class BaseParser:
    engine: str | None = None
    low_memory = True
    float_precision_choices: list[str | None] = []

    def update_kwargs(self, kwargs):
        kwargs = kwargs.copy()
        kwargs.update({"engine": self.engine, "low_memory": self.low_memory})

        return kwargs

    def read_csv(self, *args, **kwargs):
        kwargs = self.update_kwargs(kwargs)
        return read_csv(*args, **kwargs)

    def read_csv_check_warnings(
        self,
        warn_type: type[Warning],
        warn_msg: str,
        *args,
        raise_on_extra_warnings=True,
        check_stacklevel: bool = True,
        **kwargs,
    ):
        # We need to check the stacklevel here instead of in the tests
        # since this is where read_csv is called and where the warning
        # should point to.
        kwargs = self.update_kwargs(kwargs)
        with tm.assert_produces_warning(
            warn_type,
            match=warn_msg,
            raise_on_extra_warnings=raise_on_extra_warnings,
            check_stacklevel=check_stacklevel,
        ):
            return read_csv(*args, **kwargs)

    def read_table(self, *args, **kwargs):
        kwargs = self.update_kwargs(kwargs)
        return read_table(*args, **kwargs)

    def read_table_check_warnings(
        self,
        warn_type: type[Warning],
        warn_msg: str,
        *args,
        raise_on_extra_warnings=True,
        **kwargs,
    ):
        # We need to check the stacklevel here instead of in the tests
        # since this is where read_table is called and where the warning
        # should point to.
        kwargs = self.update_kwargs(kwargs)
        with tm.assert_produces_warning(
            warn_type, match=warn_msg, raise_on_extra_warnings=raise_on_extra_warnings
        ):
            return read_table(*args, **kwargs)


class CParser(BaseParser):
    engine = "c"
    float_precision_choices = [None, "high", "round_trip"]


class CParserHighMemory(CParser):
    low_memory = False


class CParserLowMemory(CParser):
    low_memory = True


class PythonParser(BaseParser):
    engine = "python"
    float_precision_choices = [None]


class PyArrowParser(BaseParser):
    engine = "pyarrow"
    float_precision_choices = [None]


@pytest.fixture
def csv_dir_path(datapath):
    """
    The directory path to the data files needed for parser tests.
    """
    return datapath("io", "parser", "data")


@pytest.fixture
def csv1(datapath):
    """
    The path to the data file "test1.csv" needed for parser tests.
    """
    return os.path.join(datapath("io", "data", "csv"), "test1.csv")


_cParserHighMemory = CParserHighMemory
_cParserLowMemory = CParserLowMemory
_pythonParser = PythonParser
_pyarrowParser = PyArrowParser

_py_parsers_only = [_pythonParser]
_c_parsers_only = [_cParserHighMemory, _cParserLowMemory]
_pyarrow_parsers_only = [
    pytest.param(
        _pyarrowParser,
        marks=[
            pytest.mark.single_cpu,
            pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow is not installed"),
        ],
    )
]

_all_parsers = [*_c_parsers_only, *_py_parsers_only, *_pyarrow_parsers_only]

_py_parser_ids = ["python"]
_c_parser_ids = ["c_high", "c_low"]
_pyarrow_parsers_ids = ["pyarrow"]

_all_parser_ids = [*_c_parser_ids, *_py_parser_ids, *_pyarrow_parsers_ids]


@pytest.fixture(params=_all_parsers, ids=_all_parser_ids)
def all_parsers(request):
    """
    Fixture all of the CSV parsers.
    """
    parser = request.param()
    if parser.engine == "pyarrow":
        pytest.importorskip("pyarrow", VERSIONS["pyarrow"])
        # Try finding a way to disable threads all together
        # for more stable CI runs
        import pyarrow

        pyarrow.set_cpu_count(1)
    return parser


@pytest.fixture(params=_c_parsers_only, ids=_c_parser_ids)
def c_parser_only(request):
    """
    Fixture all of the CSV parsers using the C engine.
    """
    return request.param()


@pytest.fixture(params=_py_parsers_only, ids=_py_parser_ids)
def python_parser_only(request):
    """
    Fixture all of the CSV parsers using the Python engine.
    """
    return request.param()


@pytest.fixture(params=_pyarrow_parsers_only, ids=_pyarrow_parsers_ids)
def pyarrow_parser_only(request):
    """
    Fixture all of the CSV parsers using the Pyarrow engine.
    """
    return request.param()


def _get_all_parser_float_precision_combinations():
    """
    Return all allowable parser and float precision
    combinations and corresponding ids.
    """
    params = []
    ids = []
    for parser, parser_id in zip(_all_parsers, _all_parser_ids):
        if hasattr(parser, "values"):
            # Wrapped in pytest.param, get the actual parser back
            parser = parser.values[0]
        for precision in parser.float_precision_choices:
            # Re-wrap in pytest.param for pyarrow
            mark = (
                [
                    pytest.mark.single_cpu,
                    pytest.mark.skipif(
                        not HAS_PYARROW, reason="pyarrow is not installed"
                    ),
                ]
                if parser.engine == "pyarrow"
                else ()
            )
            param = pytest.param((parser(), precision), marks=mark)
            params.append(param)
            ids.append(f"{parser_id}-{precision}")

    return {"params": params, "ids": ids}


@pytest.fixture(
    params=_get_all_parser_float_precision_combinations()["params"],
    ids=_get_all_parser_float_precision_combinations()["ids"],
)
def all_parsers_all_precisions(request):
    """
    Fixture for all allowable combinations of parser
    and float precision
    """
    return request.param


_utf_values = [8, 16, 32]

_encoding_seps = ["", "-", "_"]
_encoding_prefixes = ["utf", "UTF"]

_encoding_fmts = [
    f"{prefix}{sep}{{0}}" for sep in _encoding_seps for prefix in _encoding_prefixes
]


@pytest.fixture(params=_utf_values)
def utf_value(request):
    """
    Fixture for all possible integer values for a UTF encoding.
    """
    return request.param


@pytest.fixture(params=_encoding_fmts)
def encoding_fmt(request):
    """
    Fixture for all possible string formats of a UTF encoding.
    """
    return request.param


@pytest.fixture(
    params=[
        ("-1,0", -1.0),
        ("-1,2e0", -1.2),
        ("-1e0", -1.0),
        ("+1e0", 1.0),
        ("+1e+0", 1.0),
        ("+1e-1", 0.1),
        ("+,1e1", 1.0),
        ("+1,e0", 1.0),
        ("-,1e1", -1.0),
        ("-1,e0", -1.0),
        ("0,1", 0.1),
        ("1,", 1.0),
        (",1", 0.1),
        ("-,1", -0.1),
        ("1_,", 1.0),
        ("1_234,56", 1234.56),
        ("1_234,56e0", 1234.56),
        # negative cases; must not parse as float
        ("_", "_"),
        ("-_", "-_"),
        ("-_1", "-_1"),
        ("-_1e0", "-_1e0"),
        ("_1", "_1"),
        ("_1,", "_1,"),
        ("_1,_", "_1,_"),
        ("_1e0", "_1e0"),
        ("1,2e_1", "1,2e_1"),
        ("1,2e1_0", "1,2e1_0"),
        ("1,_2", "1,_2"),
        (",1__2", ",1__2"),
        (",1e", ",1e"),
        ("-,1e", "-,1e"),
        ("1_000,000_000", "1_000,000_000"),
        ("1,e1_2", "1,e1_2"),
        ("e11,2", "e11,2"),
        ("1e11,2", "1e11,2"),
        ("1,2,2", "1,2,2"),
        ("1,2_1", "1,2_1"),
        ("1,2e-10e1", "1,2e-10e1"),
        ("--1,2", "--1,2"),
        ("1a_2,1", "1a_2,1"),
        ("1,2E-1", 0.12),
        ("1,2E1", 12.0),
    ]
)
def numeric_decimal(request):
    """
    Fixture for all numeric formats which should get recognized. The first entry
    represents the value to read while the second represents the expected result.
    """
    return request.param


@pytest.fixture
def pyarrow_xfail(request):
    """
    Fixture that xfails a test if the engine is pyarrow.

    Use if failure is do to unsupported keywords or inconsistent results.
    """
    if "all_parsers" in request.fixturenames:
        parser = request.getfixturevalue("all_parsers")
    elif "all_parsers_all_precisions" in request.fixturenames:
        # Return value is tuple of (engine, precision)
        parser = request.getfixturevalue("all_parsers_all_precisions")[0]
    else:
        return
    if parser.engine == "pyarrow":
        mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
        request.applymarker(mark)


@pytest.fixture
def pyarrow_skip(request):
    """
    Fixture that skips a test if the engine is pyarrow.

    Use if failure is do a parsing failure from pyarrow.csv.read_csv
    """
    if "all_parsers" in request.fixturenames:
        parser = request.getfixturevalue("all_parsers")
    elif "all_parsers_all_precisions" in request.fixturenames:
        # Return value is tuple of (engine, precision)
        parser = request.getfixturevalue("all_parsers_all_precisions")[0]
    else:
        return
    if parser.engine == "pyarrow":
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_encoding.py ===
"""
Tests encoding functionality during parsing
for all of the parsers defined in parsers.py
"""
from io import (
    BytesIO,
    TextIOWrapper,
)
import os
import tempfile
import uuid

import numpy as np
import pytest

from pandas import (
    DataFrame,
    read_csv,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_bytes_io_input(all_parsers):
    encoding = "cp1255"
    parser = all_parsers

    data = BytesIO("שלום:1234\n562:123".encode(encoding))
    result = parser.read_csv(data, sep=":", encoding=encoding)

    expected = DataFrame([[562, 123]], columns=["שלום", "1234"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_read_csv_unicode(all_parsers):
    parser = all_parsers
    data = BytesIO("\u0141aski, Jan;1".encode())

    result = parser.read_csv(data, sep=";", encoding="utf-8", header=None)
    expected = DataFrame([["\u0141aski, Jan", 1]])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("sep", [",", "\t"])
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16le", "utf-16be"])
def test_utf16_bom_skiprows(all_parsers, sep, encoding):
    # see gh-2298
    parser = all_parsers
    data = """skip this
skip this too
A,B,C
1,2,3
4,5,6""".replace(
        ",", sep
    )
    path = f"__{uuid.uuid4()}__.csv"
    kwargs = {"sep": sep, "skiprows": 2}
    utf8 = "utf-8"

    with tm.ensure_clean(path) as path:
        bytes_data = data.encode(encoding)

        with open(path, "wb") as f:
            f.write(bytes_data)

        with TextIOWrapper(BytesIO(data.encode(utf8)), encoding=utf8) as bytes_buffer:
            result = parser.read_csv(path, encoding=encoding, **kwargs)
            expected = parser.read_csv(bytes_buffer, encoding=utf8, **kwargs)
        tm.assert_frame_equal(result, expected)


def test_utf16_example(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, "utf16_ex.txt")
    parser = all_parsers
    result = parser.read_csv(path, encoding="utf-16", sep="\t")
    assert len(result) == 50


def test_unicode_encoding(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, "unicode_series.csv")
    parser = all_parsers

    result = parser.read_csv(path, header=None, encoding="latin-1")
    result = result.set_index(0)
    got = result[1][1632]

    expected = "\xc1 k\xf6ldum klaka (Cold Fever) (1994)"
    assert got == expected


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # Basic test
        ("a\n1", {}, DataFrame({"a": [1]})),
        # "Regular" quoting
        ('"a"\n1', {"quotechar": '"'}, DataFrame({"a": [1]})),
        # Test in a data row instead of header
        ("b\n1", {"names": ["a"]}, DataFrame({"a": ["b", "1"]})),
        # Test in empty data row with skipping
        ("\n1", {"names": ["a"], "skip_blank_lines": True}, DataFrame({"a": [1]})),
        # Test in empty data row without skipping
        (
            "\n1",
            {"names": ["a"], "skip_blank_lines": False},
            DataFrame({"a": [np.nan, 1]}),
        ),
    ],
)
def test_utf8_bom(all_parsers, data, kwargs, expected, request):
    # see gh-4793
    parser = all_parsers
    bom = "\ufeff"
    utf8 = "utf-8"

    def _encode_data_with_bom(_data):
        bom_data = (bom + _data).encode(utf8)
        return BytesIO(bom_data)

    if (
        parser.engine == "pyarrow"
        and data == "\n1"
        and kwargs.get("skip_blank_lines", True)
    ):
        # CSV parse error: Empty CSV file or block: cannot infer number of columns
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    result = parser.read_csv(_encode_data_with_bom(data), encoding=utf8, **kwargs)
    tm.assert_frame_equal(result, expected)


def test_read_csv_utf_aliases(all_parsers, utf_value, encoding_fmt):
    # see gh-13549
    expected = DataFrame({"mb_num": [4.8], "multibyte": ["test"]})
    parser = all_parsers

    encoding = encoding_fmt.format(utf_value)
    data = "mb_num,multibyte\n4.8,test".encode(encoding)

    result = parser.read_csv(BytesIO(data), encoding=encoding)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "file_path,encoding",
    [
        (("io", "data", "csv", "test1.csv"), "utf-8"),
        (("io", "parser", "data", "unicode_series.csv"), "latin-1"),
        (("io", "parser", "data", "sauron.SHIFT_JIS.csv"), "shiftjis"),
    ],
)
def test_binary_mode_file_buffers(all_parsers, file_path, encoding, datapath):
    # gh-23779: Python csv engine shouldn't error on files opened in binary.
    # gh-31575: Python csv engine shouldn't error on files opened in raw binary.
    parser = all_parsers

    fpath = datapath(*file_path)
    expected = parser.read_csv(fpath, encoding=encoding)

    with open(fpath, encoding=encoding) as fa:
        result = parser.read_csv(fa)
        assert not fa.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb") as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb", buffering=0) as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("pass_encoding", [True, False])
def test_encoding_temp_file(all_parsers, utf_value, encoding_fmt, pass_encoding):
    # see gh-24130
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)

    if parser.engine == "pyarrow" and pass_encoding is True and utf_value in [16, 32]:
        # FIXME: this is bad!
        pytest.skip("These cases freeze")

    expected = DataFrame({"foo": ["bar"]})

    with tm.ensure_clean(mode="w+", encoding=encoding, return_filelike=True) as f:
        f.write("foo\nbar")
        f.seek(0)

        result = parser.read_csv(f, encoding=encoding if pass_encoding else None)
        tm.assert_frame_equal(result, expected)


def test_encoding_named_temp_file(all_parsers):
    # see gh-31819
    parser = all_parsers
    encoding = "shift-jis"

    title = "てすと"
    data = "こむ"

    expected = DataFrame({title: [data]})

    with tempfile.NamedTemporaryFile() as f:
        f.write(f"{title}\n{data}".encode(encoding))

        f.seek(0)

        result = parser.read_csv(f, encoding=encoding)
        tm.assert_frame_equal(result, expected)
        assert not f.closed


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-16", "utf-16-be", "utf-16-le", "utf-32"]
)
def test_parse_encoded_special_characters(encoding):
    # GH16218 Verify parsing of data with encoded special characters
    # Data contains a Unicode 'FULLWIDTH COLON' (U+FF1A) at position (0,"a")
    data = "a\tb\n：foo\t0\nbar\t1\nbaz\t2"  # noqa: RUF001
    encoded_data = BytesIO(data.encode(encoding))
    result = read_csv(encoded_data, delimiter="\t", encoding=encoding)

    expected = DataFrame(
        data=[["：foo", 0], ["bar", 1], ["baz", 2]],  # noqa: RUF001
        columns=["a", "b"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", ["utf-8", None, "utf-16", "cp1255", "latin-1"])
def test_encoding_memory_map(all_parsers, encoding):
    # GH40986
    parser = all_parsers
    expected = DataFrame(
        {
            "name": ["Raphael", "Donatello", "Miguel Angel", "Leonardo"],
            "mask": ["red", "purple", "orange", "blue"],
            "weapon": ["sai", "bo staff", "nunchunk", "katana"],
        }
    )
    with tm.ensure_clean() as file:
        expected.to_csv(file, index=False, encoding=encoding)

        if parser.engine == "pyarrow":
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(file, encoding=encoding, memory_map=True)
            return

        df = parser.read_csv(file, encoding=encoding, memory_map=True)
    tm.assert_frame_equal(df, expected)


def test_chunk_splits_multibyte_char(all_parsers):
    """
    Chunk splits a multibyte character with memory_map=True

    GH 43540
    """
    parser = all_parsers
    # DEFAULT_CHUNKSIZE = 262144, defined in parsers.pyx
    df = DataFrame(data=["a" * 127] * 2048)

    # Put two-bytes utf-8 encoded character "ą" at the end of chunk
    # utf-8 encoding of "ą" is b'\xc4\x85'
    df.iloc[2047] = "a" * 127 + "ą"
    with tm.ensure_clean("bug-gh43540.csv") as fname:
        df.to_csv(fname, index=False, header=False, encoding="utf-8")

        if parser.engine == "pyarrow":
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(fname, header=None, memory_map=True)
            return

        dfr = parser.read_csv(fname, header=None, memory_map=True)
    tm.assert_frame_equal(dfr, df)


def test_readcsv_memmap_utf8(all_parsers):
    """
    GH 43787

    Test correct handling of UTF-8 chars when memory_map=True and encoding is UTF-8
    """
    lines = []
    line_length = 128
    start_char = " "
    end_char = "\U00010080"
    # This for loop creates a list of 128-char strings
    # consisting of consecutive Unicode chars
    for lnum in range(ord(start_char), ord(end_char), line_length):
        line = "".join([chr(c) for c in range(lnum, lnum + 0x80)]) + "\n"
        try:
            line.encode("utf-8")
        except UnicodeEncodeError:
            continue
        lines.append(line)
    parser = all_parsers
    df = DataFrame(lines)
    with tm.ensure_clean("utf8test.csv") as fname:
        df.to_csv(fname, index=False, header=False, encoding="utf-8")

        if parser.engine == "pyarrow":
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(fname, header=None, memory_map=True, encoding="utf-8")
            return

        dfr = parser.read_csv(fname, header=None, memory_map=True, encoding="utf-8")
    tm.assert_frame_equal(df, dfr)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("mode", ["w+b", "w+t"])
def test_not_readable(all_parsers, mode):
    # GH43439
    parser = all_parsers
    content = b"abcd"
    if "t" in mode:
        content = "abcd"
    with tempfile.SpooledTemporaryFile(mode=mode, encoding="utf-8") as handle:
        handle.write(content)
        handle.seek(0)
        df = parser.read_csv(handle)
    expected = DataFrame([], columns=["abcd"])
    tm.assert_frame_equal(df, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_sort_index.py ===
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    IntervalIndex,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.fixture(params=["quicksort", "mergesort", "heapsort", "stable"])
def sort_kind(request):
    return request.param


class TestSeriesSortIndex:
    def test_sort_index_name(self, datetime_series):
        result = datetime_series.sort_index(ascending=False)
        assert result.name == datetime_series.name

    def test_sort_index(self, datetime_series):
        datetime_series.index = datetime_series.index._with_freq(None)

        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)

        random_order = datetime_series.reindex(rindex)
        sorted_series = random_order.sort_index()
        tm.assert_series_equal(sorted_series, datetime_series)

        # descending
        sorted_series = random_order.sort_index(ascending=False)
        tm.assert_series_equal(
            sorted_series, datetime_series.reindex(datetime_series.index[::-1])
        )

        # compat on level
        sorted_series = random_order.sort_index(level=0)
        tm.assert_series_equal(sorted_series, datetime_series)

        # compat on axis
        sorted_series = random_order.sort_index(axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)

        msg = "No axis named 1 for object type Series"
        with pytest.raises(ValueError, match=msg):
            random_order.sort_values(axis=1)

        sorted_series = random_order.sort_index(level=0, axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)

        with pytest.raises(ValueError, match=msg):
            random_order.sort_index(level=0, axis=1)

    def test_sort_index_inplace(self, datetime_series):
        datetime_series.index = datetime_series.index._with_freq(None)

        # For GH#11402
        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)

        # descending
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=False, inplace=True)

        assert result is None
        expected = datetime_series.reindex(datetime_series.index[::-1])
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)

        # ascending
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=True, inplace=True)

        assert result is None
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)

    def test_sort_index_level(self):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        res = s.sort_index(level="A")
        tm.assert_series_equal(backwards, res)

        res = s.sort_index(level=["A", "B"])
        tm.assert_series_equal(backwards, res)

        res = s.sort_index(level="A", sort_remaining=False)
        tm.assert_series_equal(s, res)

        res = s.sort_index(level=["A", "B"], sort_remaining=False)
        tm.assert_series_equal(s, res)

    @pytest.mark.parametrize("level", ["A", 0])  # GH#21052
    def test_sort_index_multiindex(self, level):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        # implicit sort_remaining=True
        res = s.sort_index(level=level)
        tm.assert_series_equal(backwards, res)

        # GH#13496
        # sort has no effect without remaining lvls
        res = s.sort_index(level=level, sort_remaining=False)
        tm.assert_series_equal(s, res)

    def test_sort_index_kind(self, sort_kind):
        # GH#14444 & GH#13589:  Add support for sort algo choosing
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(kind=sort_kind)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position(self):
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(na_position="first")
        tm.assert_series_equal(expected_series_first, index_sorted_series)

        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)

        index_sorted_series = series.sort_index(na_position="last")
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_sort_index_intervals(self):
        s = Series(
            [np.nan, 1, 2, 3], IntervalIndex.from_arrays([0, 1, 2, 3], [1, 2, 3, 4])
        )

        result = s.sort_index()
        expected = s
        tm.assert_series_equal(result, expected)

        result = s.sort_index(ascending=False)
        expected = Series(
            [3, 2, 1, np.nan], IntervalIndex.from_arrays([3, 2, 1, 0], [4, 3, 2, 1])
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_list, sorted_list, ascending, ignore_index, output_index",
        [
            ([2, 3, 6, 1], [2, 3, 6, 1], True, True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [2, 3, 6, 1], True, False, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [1, 6, 3, 2], False, True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [1, 6, 3, 2], False, False, [3, 2, 1, 0]),
        ],
    )
    def test_sort_index_ignore_index(
        self, inplace, original_list, sorted_list, ascending, ignore_index, output_index
    ):
        # GH 30114
        ser = Series(original_list)
        expected = Series(sorted_list, index=output_index)
        kwargs = {
            "ascending": ascending,
            "ignore_index": ignore_index,
            "inplace": inplace,
        }

        if inplace:
            result_ser = ser.copy()
            result_ser.sort_index(**kwargs)
        else:
            result_ser = ser.sort_index(**kwargs)

        tm.assert_series_equal(result_ser, expected)
        tm.assert_series_equal(ser, Series(original_list))

    def test_sort_index_ascending_list(self):
        # GH#16934

        # Set up a Series with a three level MultiIndex
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
            [4, 3, 2, 1, 4, 3, 2, 1],
        ]
        tuples = zip(*arrays)
        mi = MultiIndex.from_tuples(tuples, names=["first", "second", "third"])
        ser = Series(range(8), index=mi)

        # Sort with boolean ascending
        result = ser.sort_index(level=["third", "first"], ascending=False)
        expected = ser.iloc[[4, 0, 5, 1, 6, 2, 7, 3]]
        tm.assert_series_equal(result, expected)

        # Sort with list of boolean ascending
        result = ser.sort_index(level=["third", "first"], ascending=[False, True])
        expected = ser.iloc[[0, 4, 1, 5, 2, 6, 3, 7]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending",
        [
            None,
            (True, None),
            (False, "True"),
        ],
    )
    def test_sort_index_ascending_bad_value_raises(self, ascending):
        ser = Series(range(10), index=[0, 3, 2, 1, 4, 5, 7, 6, 8, 9])
        match = 'For argument "ascending" expected type bool'
        with pytest.raises(ValueError, match=match):
            ser.sort_index(ascending=ascending)


class TestSeriesSortIndexKey:
    def test_sort_index_multiindex_key(self):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        result = s.sort_index(level="C", key=lambda x: -x)
        tm.assert_series_equal(s, result)

        result = s.sort_index(level="C", key=lambda x: x)  # nothing happens
        tm.assert_series_equal(backwards, result)

    def test_sort_index_multiindex_key_multi_level(self):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        result = s.sort_index(level=["A", "C"], key=lambda x: -x)
        tm.assert_series_equal(s, result)

        result = s.sort_index(level=["A", "C"], key=lambda x: x)  # nothing happens
        tm.assert_series_equal(backwards, result)

    def test_sort_index_key(self):
        series = Series(np.arange(6, dtype="int64"), index=list("aaBBca"))

        result = series.sort_index()
        expected = series.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_series_equal(result, expected)

        result = series.sort_index(key=lambda x: x.str.lower())
        expected = series.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_series_equal(result, expected)

        result = series.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = series.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_key_int(self):
        series = Series(np.arange(6, dtype="int64"), index=np.arange(6, dtype="int64"))

        result = series.sort_index()
        tm.assert_series_equal(result, series)

        result = series.sort_index(key=lambda x: -x)
        expected = series.sort_index(ascending=False)
        tm.assert_series_equal(result, expected)

        result = series.sort_index(key=lambda x: 2 * x)
        tm.assert_series_equal(result, series)

    def test_sort_index_kind_key(self, sort_kind, sort_by_key):
        # GH #14444 & #13589:  Add support for sort algo choosing
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(kind=sort_kind, key=sort_by_key)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_kind_neg_key(self, sort_kind):
        # GH #14444 & #13589:  Add support for sort algo choosing
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[4, 3, 3, 2, 1], dtype=object)

        index_sorted_series = series.sort_index(kind=sort_kind, key=lambda x: -x)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position_key(self, sort_by_key):
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(na_position="first", key=sort_by_key)
        tm.assert_series_equal(expected_series_first, index_sorted_series)

        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)

        index_sorted_series = series.sort_index(na_position="last", key=sort_by_key)
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_changes_length_raises(self):
        s = Series([1, 2, 3])
        with pytest.raises(ValueError, match="change the shape"):
            s.sort_index(key=lambda x: x[:1])

    def test_sort_values_key_type(self):
        s = Series([1, 2, 3], DatetimeIndex(["2008-10-24", "2008-11-23", "2007-12-22"]))

        result = s.sort_index(key=lambda x: x.month)
        expected = s.iloc[[0, 1, 2]]
        tm.assert_series_equal(result, expected)

        result = s.sort_index(key=lambda x: x.day)
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

        result = s.sort_index(key=lambda x: x.year)
        expected = s.iloc[[2, 0, 1]]
        tm.assert_series_equal(result, expected)

        result = s.sort_index(key=lambda x: x.month_name())
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending",
        [
            [True, False],
            [False, True],
        ],
    )
    def test_sort_index_multi_already_monotonic(self, ascending):
        # GH 56049
        mi = MultiIndex.from_product([[1, 2], [3, 4]])
        ser = Series(range(len(mi)), index=mi)
        result = ser.sort_index(ascending=ascending)
        if ascending == [True, False]:
            expected = ser.take([1, 0, 3, 2])
        elif ascending == [False, True]:
            expected = ser.take([2, 3, 0, 1])
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_array_to_datetime.py ===
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)

from dateutil.tz.tz import tzoffset
import numpy as np
import pytest

from pandas._libs import (
    NaT,
    iNaT,
    tslib,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

from pandas import Timestamp
import pandas._testing as tm

creso_infer = NpyDatetimeUnit.NPY_FR_GENERIC.value


class TestArrayToDatetimeResolutionInference:
    # TODO: tests that include tzs, ints

    def test_infer_all_nat(self):
        arr = np.array([NaT, np.nan], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        assert result.dtype == "M8[s]"

    def test_infer_homogeoneous_datetimes(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        arr = np.array([dt, dt, dt], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([dt, dt, dt], dtype="M8[us]")
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_date_objects(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt2 = dt.date()
        arr = np.array([None, dt2, dt2, dt2], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64("NaT"), dt2, dt2, dt2], dtype="M8[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_dt64(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt64 = np.datetime64(dt, "ms")
        arr = np.array([None, dt64, dt64, dt64], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64("NaT"), dt64, dt64, dt64], dtype="M8[ms]")
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_timestamps(self):
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        ts = Timestamp(dt).as_unit("ns")
        arr = np.array([None, ts, ts, ts], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64("NaT")] + [ts.asm8] * 3, dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_datetimes_strings(self):
        item = "2023-10-27 18:03:05.678000"
        arr = np.array([None, item, item, item], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64("NaT"), item, item, item], dtype="M8[us]")
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_heterogeneous(self):
        dtstr = "2023-10-27 18:03:05.678000"

        arr = np.array([dtstr, dtstr[:-3], dtstr[:-7], None], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array(arr, dtype="M8[us]")
        tm.assert_numpy_array_equal(result, expected)

        result, tz = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        assert tz is None
        tm.assert_numpy_array_equal(result, expected[::-1])

    @pytest.mark.parametrize(
        "item", [float("nan"), NaT.value, float(NaT.value), "NaT", ""]
    )
    def test_infer_with_nat_int_float_str(self, item):
        # floats/ints get inferred to nanos *unless* they are NaN/iNaT,
        # similar NaT string gets treated like NaT scalar (ignored for resolution)
        dt = datetime(2023, 11, 15, 15, 5, 6)

        arr = np.array([dt, item], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([dt, np.datetime64("NaT")], dtype="M8[us]")
        tm.assert_numpy_array_equal(result, expected)

        result2, tz2 = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        assert tz2 is None
        tm.assert_numpy_array_equal(result2, expected[::-1])


class TestArrayToDatetimeWithTZResolutionInference:
    def test_array_to_datetime_with_tz_resolution(self):
        tz = tzoffset("custom", 3600)
        vals = np.array(["2016-01-01 02:03:04.567", NaT], dtype=object)
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        assert res.dtype == "M8[ms]"

        vals2 = np.array([datetime(2016, 1, 1, 2, 3, 4), NaT], dtype=object)
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        assert res2.dtype == "M8[us]"

        vals3 = np.array([NaT, np.datetime64(12345, "s")], dtype=object)
        res3 = tslib.array_to_datetime_with_tz(vals3, tz, False, False, creso_infer)
        assert res3.dtype == "M8[s]"

    def test_array_to_datetime_with_tz_resolution_all_nat(self):
        tz = tzoffset("custom", 3600)
        vals = np.array(["NaT"], dtype=object)
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        assert res.dtype == "M8[s]"

        vals2 = np.array([NaT, NaT], dtype=object)
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        assert res2.dtype == "M8[s]"


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            ["01-01-2013", "01-02-2013"],
            [
                "2013-01-01T00:00:00.000000000",
                "2013-01-02T00:00:00.000000000",
            ],
        ),
        (
            ["Mon Sep 16 2013", "Tue Sep 17 2013"],
            [
                "2013-09-16T00:00:00.000000000",
                "2013-09-17T00:00:00.000000000",
            ],
        ),
    ],
)
def test_parsing_valid_dates(data, expected):
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr)

    expected = np.array(expected, dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "dt_string, expected_tz",
    [
        ["01-01-2013 08:00:00+08:00", 480],
        ["2013-01-01T08:00:00.000000000+0800", 480],
        ["2012-12-31T16:00:00.000000000-0800", -480],
        ["12-31-2012 23:00:00-01:00", -60],
    ],
)
def test_parsing_timezone_offsets(dt_string, expected_tz):
    # All of these datetime strings with offsets are equivalent
    # to the same datetime after the timezone offset is added.
    arr = np.array(["01-01-2013 00:00:00"], dtype=object)
    expected, _ = tslib.array_to_datetime(arr)

    arr = np.array([dt_string], dtype=object)
    result, result_tz = tslib.array_to_datetime(arr)

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz == timezone(timedelta(minutes=expected_tz))


def test_parsing_non_iso_timezone_offset():
    dt_string = "01-01-2013T00:00:00.000000000+0000"
    arr = np.array([dt_string], dtype=object)

    with tm.assert_produces_warning(None):
        # GH#50949 should not get tzlocal-deprecation warning here
        result, result_tz = tslib.array_to_datetime(arr)
    expected = np.array([np.datetime64("2013-01-01 00:00:00.000000000")])

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is timezone.utc


def test_parsing_different_timezone_offsets():
    # see gh-17697
    data = ["2015-11-18 15:30:00+05:30", "2015-11-18 15:30:00+06:30"]
    data = np.array(data, dtype=object)

    msg = "parsing datetimes with mixed time zones will raise an error"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result, result_tz = tslib.array_to_datetime(data)
    expected = np.array(
        [
            datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 19800)),
            datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 23400)),
        ],
        dtype=object,
    )

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is None


@pytest.mark.parametrize(
    "data", [["-352.737091", "183.575577"], ["1", "2", "3", "4", "5"]]
)
def test_number_looking_strings_not_into_datetime(data):
    # see gh-4601
    #
    # These strings don't look like datetimes, so
    # they shouldn't be attempted to be converted.
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr, errors="ignore")

    tm.assert_numpy_array_equal(result, arr)


@pytest.mark.parametrize(
    "invalid_date",
    [
        date(1000, 1, 1),
        datetime(1000, 1, 1),
        "1000-01-01",
        "Jan 1, 1000",
        np.datetime64("1000-01-01"),
    ],
)
@pytest.mark.parametrize("errors", ["coerce", "raise"])
def test_coerce_outside_ns_bounds(invalid_date, errors):
    arr = np.array([invalid_date], dtype="object")
    kwargs = {"values": arr, "errors": errors}

    if errors == "raise":
        msg = "^Out of bounds nanosecond timestamp: .*, at position 0$"

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            tslib.array_to_datetime(**kwargs)
    else:  # coerce.
        result, _ = tslib.array_to_datetime(**kwargs)
        expected = np.array([iNaT], dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)


def test_coerce_outside_ns_bounds_one_valid():
    arr = np.array(["1/1/1000", "1/1/2000"], dtype=object)
    result, _ = tslib.array_to_datetime(arr, errors="coerce")

    expected = [iNaT, "2000-01-01T00:00:00.000000000"]
    expected = np.array(expected, dtype="M8[ns]")

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("errors", ["ignore", "coerce"])
def test_coerce_of_invalid_datetimes(errors):
    arr = np.array(["01-01-2013", "not_a_date", "1"], dtype=object)
    kwargs = {"values": arr, "errors": errors}

    if errors == "ignore":
        # Without coercing, the presence of any invalid
        # dates prevents any values from being converted.
        result, _ = tslib.array_to_datetime(**kwargs)
        tm.assert_numpy_array_equal(result, arr)
    else:  # coerce.
        # With coercing, the invalid dates becomes iNaT
        result, _ = tslib.array_to_datetime(arr, errors="coerce")
        expected = ["2013-01-01T00:00:00.000000000", iNaT, iNaT]

        tm.assert_numpy_array_equal(result, np.array(expected, dtype="M8[ns]"))


def test_to_datetime_barely_out_of_bounds():
    # see gh-19382, gh-19529
    #
    # Close enough to bounds that dropping nanos
    # would result in an in-bounds datetime.
    arr = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)
    msg = "^Out of bounds nanosecond timestamp: 2262-04-11 23:47:16, at position 0$"

    with pytest.raises(tslib.OutOfBoundsDatetime, match=msg):
        tslib.array_to_datetime(arr)


@pytest.mark.parametrize(
    "timestamp",
    [
        # Close enough to bounds that scaling micros to nanos overflows
        # but adding nanos would result in an in-bounds datetime.
        "1677-09-21T00:12:43.145224193",
        "1677-09-21T00:12:43.145224999",
        # this always worked
        "1677-09-21T00:12:43.145225000",
    ],
)
def test_to_datetime_barely_inside_bounds(timestamp):
    # see gh-57150
    result, _ = tslib.array_to_datetime(np.array([timestamp], dtype=object))
    tm.assert_numpy_array_equal(result, np.array([timestamp], dtype="M8[ns]"))


class SubDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "data,expected",
    [
        ([SubDatetime(2000, 1, 1)], ["2000-01-01T00:00:00.000000000"]),
        ([datetime(2000, 1, 1)], ["2000-01-01T00:00:00.000000000"]),
        ([Timestamp(2000, 1, 1)], ["2000-01-01T00:00:00.000000000"]),
    ],
)
def test_datetime_subclass(data, expected):
    # GH 25851
    # ensure that subclassed datetime works with
    # array_to_datetime

    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr)

    expected = np.array(expected, dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\test_label_or_level_utils.py ===
import pytest

from pandas.core.dtypes.missing import array_equivalent

import pandas as pd


# Fixtures
# ========
@pytest.fixture
def df():
    """DataFrame with columns 'L1', 'L2', and 'L3'"""
    return pd.DataFrame({"L1": [1, 2, 3], "L2": [11, 12, 13], "L3": ["A", "B", "C"]})


@pytest.fixture(params=[[], ["L1"], ["L1", "L2"], ["L1", "L2", "L3"]])
def df_levels(request, df):
    """DataFrame with columns or index levels 'L1', 'L2', and 'L3'"""
    levels = request.param

    if levels:
        df = df.set_index(levels)

    return df


@pytest.fixture
def df_ambig(df):
    """DataFrame with levels 'L1' and 'L2' and labels 'L1' and 'L3'"""
    df = df.set_index(["L1", "L2"])

    df["L1"] = df["L3"]

    return df


@pytest.fixture
def df_duplabels(df):
    """DataFrame with level 'L1' and labels 'L2', 'L3', and 'L2'"""
    df = df.set_index(["L1"])
    df = pd.concat([df, df["L2"]], axis=1)

    return df


# Test is label/level reference
# =============================
def get_labels_levels(df_levels):
    expected_labels = list(df_levels.columns)
    expected_levels = [name for name in df_levels.index.names if name is not None]
    return expected_labels, expected_levels


def assert_label_reference(frame, labels, axis):
    for label in labels:
        assert frame._is_label_reference(label, axis=axis)
        assert not frame._is_level_reference(label, axis=axis)
        assert frame._is_label_or_level_reference(label, axis=axis)


def assert_level_reference(frame, levels, axis):
    for level in levels:
        assert frame._is_level_reference(level, axis=axis)
        assert not frame._is_label_reference(level, axis=axis)
        assert frame._is_label_or_level_reference(level, axis=axis)


# DataFrame
# ---------
def test_is_level_or_label_reference_df_simple(df_levels, axis):
    axis = df_levels._get_axis_number(axis)
    # Compute expected labels and levels
    expected_labels, expected_levels = get_labels_levels(df_levels)

    # Transpose frame if axis == 1
    if axis == 1:
        df_levels = df_levels.T

    # Perform checks
    assert_level_reference(df_levels, expected_levels, axis=axis)
    assert_label_reference(df_levels, expected_labels, axis=axis)


def test_is_level_reference_df_ambig(df_ambig, axis):
    axis = df_ambig._get_axis_number(axis)

    # Transpose frame if axis == 1
    if axis == 1:
        df_ambig = df_ambig.T

    # df has both an on-axis level and off-axis label named L1
    # Therefore L1 should reference the label, not the level
    assert_label_reference(df_ambig, ["L1"], axis=axis)

    # df has an on-axis level named L2 and it is not ambiguous
    # Therefore L2 is an level reference
    assert_level_reference(df_ambig, ["L2"], axis=axis)

    # df has a column named L3 and it not an level reference
    assert_label_reference(df_ambig, ["L3"], axis=axis)


# Series
# ------
def test_is_level_reference_series_simple_axis0(df):
    # Make series with L1 as index
    s = df.set_index("L1").L2
    assert_level_reference(s, ["L1"], axis=0)
    assert not s._is_level_reference("L2")

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    assert_level_reference(s, ["L1", "L2"], axis=0)
    assert not s._is_level_reference("L3")


def test_is_level_reference_series_axis1_error(df):
    # Make series with L1 as index
    s = df.set_index("L1").L2

    with pytest.raises(ValueError, match="No axis named 1"):
        s._is_level_reference("L1", axis=1)


# Test _check_label_or_level_ambiguity_df
# =======================================


# DataFrame
# ---------
def test_check_label_or_level_ambiguity_df(df_ambig, axis):
    axis = df_ambig._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_ambig = df_ambig.T
        msg = "'L1' is both a column level and an index label"

    else:
        msg = "'L1' is both an index level and a column label"
    # df_ambig has both an on-axis level and off-axis label named L1
    # Therefore, L1 is ambiguous.
    with pytest.raises(ValueError, match=msg):
        df_ambig._check_label_or_level_ambiguity("L1", axis=axis)

    # df_ambig has an on-axis level named L2,, and it is not ambiguous.
    df_ambig._check_label_or_level_ambiguity("L2", axis=axis)

    # df_ambig has an off-axis label named L3, and it is not ambiguous
    assert not df_ambig._check_label_or_level_ambiguity("L3", axis=axis)


# Series
# ------
def test_check_label_or_level_ambiguity_series(df):
    # A series has no columns and therefore references are never ambiguous

    # Make series with L1 as index
    s = df.set_index("L1").L2
    s._check_label_or_level_ambiguity("L1", axis=0)
    s._check_label_or_level_ambiguity("L2", axis=0)

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    s._check_label_or_level_ambiguity("L1", axis=0)
    s._check_label_or_level_ambiguity("L2", axis=0)
    s._check_label_or_level_ambiguity("L3", axis=0)


def test_check_label_or_level_ambiguity_series_axis1_error(df):
    # Make series with L1 as index
    s = df.set_index("L1").L2

    with pytest.raises(ValueError, match="No axis named 1"):
        s._check_label_or_level_ambiguity("L1", axis=1)


# Test _get_label_or_level_values
# ===============================
def assert_label_values(frame, labels, axis):
    axis = frame._get_axis_number(axis)
    for label in labels:
        if axis == 0:
            expected = frame[label]._values
        else:
            expected = frame.loc[label]._values

        result = frame._get_label_or_level_values(label, axis=axis)
        assert array_equivalent(expected, result)


def assert_level_values(frame, levels, axis):
    axis = frame._get_axis_number(axis)
    for level in levels:
        if axis == 0:
            expected = frame.index.get_level_values(level=level)._values
        else:
            expected = frame.columns.get_level_values(level=level)._values

        result = frame._get_label_or_level_values(level, axis=axis)
        assert array_equivalent(expected, result)


# DataFrame
# ---------
def test_get_label_or_level_values_df_simple(df_levels, axis):
    # Compute expected labels and levels
    expected_labels, expected_levels = get_labels_levels(df_levels)

    axis = df_levels._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_levels = df_levels.T

    # Perform checks
    assert_label_values(df_levels, expected_labels, axis=axis)
    assert_level_values(df_levels, expected_levels, axis=axis)


def test_get_label_or_level_values_df_ambig(df_ambig, axis):
    axis = df_ambig._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_ambig = df_ambig.T

    # df has an on-axis level named L2, and it is not ambiguous.
    assert_level_values(df_ambig, ["L2"], axis=axis)

    # df has an off-axis label named L3, and it is not ambiguous.
    assert_label_values(df_ambig, ["L3"], axis=axis)


def test_get_label_or_level_values_df_duplabels(df_duplabels, axis):
    axis = df_duplabels._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_duplabels = df_duplabels.T

    # df has unambiguous level 'L1'
    assert_level_values(df_duplabels, ["L1"], axis=axis)

    # df has unique label 'L3'
    assert_label_values(df_duplabels, ["L3"], axis=axis)

    # df has duplicate labels 'L2'
    if axis == 0:
        expected_msg = "The column label 'L2' is not unique"
    else:
        expected_msg = "The index label 'L2' is not unique"

    with pytest.raises(ValueError, match=expected_msg):
        assert_label_values(df_duplabels, ["L2"], axis=axis)


# Series
# ------
def test_get_label_or_level_values_series_axis0(df):
    # Make series with L1 as index
    s = df.set_index("L1").L2
    assert_level_values(s, ["L1"], axis=0)

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    assert_level_values(s, ["L1", "L2"], axis=0)


def test_get_label_or_level_values_series_axis1_error(df):
    # Make series with L1 as index
    s = df.set_index("L1").L2

    with pytest.raises(ValueError, match="No axis named 1"):
        s._get_label_or_level_values("L1", axis=1)


# Test _drop_labels_or_levels
# ===========================
def assert_labels_dropped(frame, labels, axis):
    axis = frame._get_axis_number(axis)
    for label in labels:
        df_dropped = frame._drop_labels_or_levels(label, axis=axis)

        if axis == 0:
            assert label in frame.columns
            assert label not in df_dropped.columns
        else:
            assert label in frame.index
            assert label not in df_dropped.index


def assert_levels_dropped(frame, levels, axis):
    axis = frame._get_axis_number(axis)
    for level in levels:
        df_dropped = frame._drop_labels_or_levels(level, axis=axis)

        if axis == 0:
            assert level in frame.index.names
            assert level not in df_dropped.index.names
        else:
            assert level in frame.columns.names
            assert level not in df_dropped.columns.names


# DataFrame
# ---------
def test_drop_labels_or_levels_df(df_levels, axis):
    # Compute expected labels and levels
    expected_labels, expected_levels = get_labels_levels(df_levels)

    axis = df_levels._get_axis_number(axis)
    # Transpose frame if axis == 1
    if axis == 1:
        df_levels = df_levels.T

    # Perform checks
    assert_labels_dropped(df_levels, expected_labels, axis=axis)
    assert_levels_dropped(df_levels, expected_levels, axis=axis)

    with pytest.raises(ValueError, match="not valid labels or levels"):
        df_levels._drop_labels_or_levels("L4", axis=axis)


# Series
# ------
def test_drop_labels_or_levels_series(df):
    # Make series with L1 as index
    s = df.set_index("L1").L2
    assert_levels_dropped(s, ["L1"], axis=0)

    with pytest.raises(ValueError, match="not valid labels or levels"):
        s._drop_labels_or_levels("L4", axis=0)

    # Make series with L1 and L2 as index
    s = df.set_index(["L1", "L2"]).L3
    assert_levels_dropped(s, ["L1", "L2"], axis=0)

    with pytest.raises(ValueError, match="not valid labels or levels"):
        s._drop_labels_or_levels("L4", axis=0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\tests\test_accumulationbounds.py ===
from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x

a = Symbol('a', real=True)
B = AccumBounds


def test_AccumBounds():
    assert B(1, 2).args == (1, 2)
    assert B(1, 2).delta is S.One
    assert B(1, 2).mid == Rational(3, 2)
    assert B(1, 3).is_real == True

    assert B(1, 1) is S.One

    assert B(1, 2) + 1 == B(2, 3)
    assert 1 + B(1, 2) == B(2, 3)
    assert B(1, 2) + B(2, 3) == B(3, 5)

    assert -B(1, 2) == B(-2, -1)

    assert B(1, 2) - 1 == B(0, 1)
    assert 1 - B(1, 2) == B(-1, 0)
    assert B(2, 3) - B(1, 2) == B(0, 2)

    assert x + B(1, 2) == Add(B(1, 2), x)
    assert a + B(1, 2) == B(1 + a, 2 + a)
    assert B(1, 2) - x == Add(B(1, 2), -x)

    assert B(-oo, 1) + oo == B(-oo, oo)
    assert B(1, oo) + oo is oo
    assert B(1, oo) - oo == B(-oo, oo)
    assert (-oo - B(-1, oo)) is -oo
    assert B(-oo, 1) - oo is -oo

    assert B(1, oo) - oo == B(-oo, oo)
    assert B(-oo, 1) - (-oo) == B(-oo, oo)
    assert (oo - B(1, oo)) == B(-oo, oo)
    assert (-oo - B(1, oo)) is -oo

    assert B(1, 2)/2 == B(S.Half, 1)
    assert 2/B(2, 3) == B(Rational(2, 3), 1)
    assert 1/B(-1, 1) == B(-oo, oo)

    assert abs(B(1, 2)) == B(1, 2)
    assert abs(B(-2, -1)) == B(1, 2)
    assert abs(B(-2, 1)) == B(0, 2)
    assert abs(B(-1, 2)) == B(0, 2)
    c = Symbol('c')
    raises(ValueError, lambda: B(0, c))
    raises(ValueError, lambda: B(1, -1))
    r = Symbol('r', real=True)
    raises(ValueError, lambda: B(r, r - 1))


def test_AccumBounds_mul():
    assert B(1, 2)*2 == B(2, 4)
    assert 2*B(1, 2) == B(2, 4)
    assert B(1, 2)*B(2, 3) == B(2, 6)
    assert B(0, 2)*B(2, oo) == B(0, oo)
    l, r = B(-oo, oo), B(-a, a)
    assert l*r == B(-oo, oo)
    assert r*l == B(-oo, oo)
    l, r = B(1, oo), B(-3, -2)
    assert l*r == B(-oo, -2)
    assert r*l == B(-oo, -2)
    assert B(1, 2)*0 == 0
    assert B(1, oo)*0 == B(0, oo)
    assert B(-oo, 1)*0 == B(-oo, 0)
    assert B(-oo, oo)*0 == B(-oo, oo)

    assert B(1, 2)*x == Mul(B(1, 2), x, evaluate=False)

    assert B(0, 2)*oo == B(0, oo)
    assert B(-2, 0)*oo == B(-oo, 0)
    assert B(0, 2)*(-oo) == B(-oo, 0)
    assert B(-2, 0)*(-oo) == B(0, oo)
    assert B(-1, 1)*oo == B(-oo, oo)
    assert B(-1, 1)*(-oo) == B(-oo, oo)
    assert B(-oo, oo)*oo == B(-oo, oo)


def test_AccumBounds_div():
    assert B(-1, 3)/B(3, 4) == B(Rational(-1, 3), 1)
    assert B(-2, 4)/B(-3, 4) == B(-oo, oo)
    assert B(-3, -2)/B(-4, 0) == B(S.Half, oo)

    # these two tests can have a better answer
    # after Union of B is improved
    assert B(-3, -2)/B(-2, 1) == B(-oo, oo)
    assert B(2, 3)/B(-2, 2) == B(-oo, oo)

    assert B(-3, -2)/B(0, 4) == B(-oo, Rational(-1, 2))
    assert B(2, 4)/B(-3, 0) == B(-oo, Rational(-2, 3))
    assert B(2, 4)/B(0, 3) == B(Rational(2, 3), oo)

    assert B(0, 1)/B(0, 1) == B(0, oo)
    assert B(-1, 0)/B(0, 1) == B(-oo, 0)
    assert B(-1, 2)/B(-2, 2) == B(-oo, oo)

    assert 1/B(-1, 2) == B(-oo, oo)
    assert 1/B(0, 2) == B(S.Half, oo)
    assert (-1)/B(0, 2) == B(-oo, Rational(-1, 2))
    assert 1/B(-oo, 0) == B(-oo, 0)
    assert 1/B(-1, 0) == B(-oo, -1)
    assert (-2)/B(-oo, 0) == B(0, oo)
    assert 1/B(-oo, -1) == B(-1, 0)

    assert B(1, 2)/a == Mul(B(1, 2), 1/a, evaluate=False)

    assert B(1, 2)/0 == B(1, 2)*zoo
    assert B(1, oo)/oo == B(0, oo)
    assert B(1, oo)/(-oo) == B(-oo, 0)
    assert B(-oo, -1)/oo == B(-oo, 0)
    assert B(-oo, -1)/(-oo) == B(0, oo)
    assert B(-oo, oo)/oo == B(-oo, oo)
    assert B(-oo, oo)/(-oo) == B(-oo, oo)
    assert B(-1, oo)/oo == B(0, oo)
    assert B(-1, oo)/(-oo) == B(-oo, 0)
    assert B(-oo, 1)/oo == B(-oo, 0)
    assert B(-oo, 1)/(-oo) == B(0, oo)


def test_issue_18795():
    r = Symbol('r', real=True)
    a = B(-1,1)
    c = B(7, oo)
    b = B(-oo, oo)
    assert c - tan(r) == B(7-tan(r), oo)
    assert b + tan(r) == B(-oo, oo)
    assert (a + r)/a == B(-oo, oo)*B(r - 1, r + 1)
    assert (b + a)/a == B(-oo, oo)


def test_AccumBounds_func():
    assert (x**2 + 2*x + 1).subs(x, B(-1, 1)) == B(-1, 4)
    assert exp(B(0, 1)) == B(1, E)
    assert exp(B(-oo, oo)) == B(0, oo)
    assert log(B(3, 6)) == B(log(3), log(6))


@XFAIL
def test_AccumBounds_powf():
    nn = Symbol('nn', nonnegative=True)
    assert B(1 + nn, 2 + nn)**B(1, 2) == B(1 + nn, (2 + nn)**2)
    i = Symbol('i', integer=True, negative=True)
    assert B(1, 2)**i == B(2**i, 1)


def test_AccumBounds_pow():
    assert B(0, 2)**2 == B(0, 4)
    assert B(-1, 1)**2 == B(0, 1)
    assert B(1, 2)**2 == B(1, 4)
    assert B(-1, 2)**3 == B(-1, 8)
    assert B(-1, 1)**0 == 1

    assert B(1, 2)**Rational(5, 2) == B(1, 4*sqrt(2))
    assert B(0, 2)**S.Half == B(0, sqrt(2))

    neg = Symbol('neg', negative=True)
    assert unchanged(Pow, B(neg, 1), S.Half)
    nn = Symbol('nn', nonnegative=True)
    assert B(nn, nn + 1)**S.Half == B(sqrt(nn), sqrt(nn + 1))
    assert B(nn, nn + 1)**nn == B(nn**nn, (nn + 1)**nn)
    assert unchanged(Pow, B(nn, nn + 1), x)
    i = Symbol('i', integer=True)
    assert B(1, 2)**i == B(Min(1, 2**i), Max(1, 2**i))
    i = Symbol('i', integer=True, nonnegative=True)
    assert B(1, 2)**i == B(1, 2**i)
    assert B(0, 1)**i == B(0**i, 1)

    assert B(1, 5)**(-2) == B(Rational(1, 25), 1)
    assert B(-1, 3)**(-2) == B(0, oo)
    assert B(0, 2)**(-3) == B(Rational(1, 8), oo)
    assert B(-2, 0)**(-3) == B(-oo, -Rational(1, 8))
    assert B(0, 2)**(-2) == B(Rational(1, 4), oo)
    assert B(-1, 2)**(-3) == B(-oo, oo)
    assert B(-3, -2)**(-3) == B(Rational(-1, 8), Rational(-1, 27))
    assert B(-3, -2)**(-2) == B(Rational(1, 9), Rational(1, 4))
    assert B(0, oo)**S.Half == B(0, oo)
    assert B(-oo, 0)**(-2) == B(0, oo)
    assert B(-2, 0)**(-2) == B(Rational(1, 4), oo)

    assert B(Rational(1, 3), S.Half)**oo is S.Zero
    assert B(0, S.Half)**oo is S.Zero
    assert B(S.Half, 1)**oo == B(0, oo)
    assert B(0, 1)**oo == B(0, oo)
    assert B(2, 3)**oo is oo
    assert B(1, 2)**oo == B(0, oo)
    assert B(S.Half, 3)**oo == B(0, oo)
    assert B(Rational(-1, 3), Rational(-1, 4))**oo is S.Zero
    assert B(-1, Rational(-1, 2))**oo is S.NaN
    assert B(-3, -2)**oo is zoo
    assert B(-2, -1)**oo is S.NaN
    assert B(-2, Rational(-1, 2))**oo is S.NaN
    assert B(Rational(-1, 2), S.Half)**oo is S.Zero
    assert B(Rational(-1, 2), 1)**oo == B(0, oo)
    assert B(Rational(-2, 3), 2)**oo == B(0, oo)
    assert B(-1, 1)**oo == B(-oo, oo)
    assert B(-1, S.Half)**oo == B(-oo, oo)
    assert B(-1, 2)**oo == B(-oo, oo)
    assert B(-2, S.Half)**oo == B(-oo, oo)

    assert B(1, 2)**x == Pow(B(1, 2), x, evaluate=False)

    assert B(2, 3)**(-oo) is S.Zero
    assert B(0, 2)**(-oo) == B(0, oo)
    assert B(-1, 2)**(-oo) == B(-oo, oo)

    assert (tan(x)**sin(2*x)).subs(x, B(0, pi/2)) == \
        Pow(B(-oo, oo), B(0, 1))


def test_AccumBounds_exponent():
    # base is 0
    z = 0**B(a, a + S.Half)
    assert z.subs(a, 0) == B(0, 1)
    assert z.subs(a, 1) == 0
    p = z.subs(a, -1)
    assert p.is_Pow and p.args == (0, B(-1, -S.Half))
    # base > 0
    #   when base is 1 the type of bounds does not matter
    assert 1**B(a, a + 1) == 1
    #  otherwise we need to know if 0 is in the bounds
    assert S.Half**B(-2, 2) == B(S(1)/4, 4)
    assert 2**B(-2, 2) == B(S(1)/4, 4)

    # +eps may introduce +oo
    # if there is a negative integer exponent
    assert B(0, 1)**B(S(1)/2, 1) == B(0, 1)
    assert B(0, 1)**B(0, 1) == B(0, 1)

    # positive bases have positive bounds
    assert B(2, 3)**B(-3, -2) == B(S(1)/27, S(1)/4)
    assert B(2, 3)**B(-3, 2) == B(S(1)/27, 9)

    # bounds generating imaginary parts unevaluated
    assert unchanged(Pow, B(-1, 1), B(1, 2))
    assert B(0, S(1)/2)**B(1, oo) == B(0, S(1)/2)
    assert B(0, 1)**B(1, oo) == B(0, oo)
    assert B(0, 2)**B(1, oo) == B(0, oo)
    assert B(0, oo)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 1)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 1)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, 1)**B(-oo, oo) == B(0, oo)
    assert B(S(1)/2, 2)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 2)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, 2)**B(-oo, oo) == B(0, oo)
    assert B(S(1)/2, oo)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, oo)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, oo)**B(-oo, oo) == B(0, oo)
    assert B(1, 2)**B(1, oo) == B(0, oo)
    assert B(1, 2)**B(-oo, -1) == B(0, oo)
    assert B(1, 2)**B(-oo, oo) == B(0, oo)
    assert B(1, oo)**B(1, oo) == B(0, oo)
    assert B(1, oo)**B(-oo, -1) == B(0, oo)
    assert B(1, oo)**B(-oo, oo) == B(0, oo)
    assert B(2, oo)**B(1, oo) == B(2, oo)
    assert B(2, oo)**B(-oo, -1) == B(0, S(1)/2)
    assert B(2, oo)**B(-oo, oo) == B(0, oo)


def test_comparison_AccumBounds():
    assert (B(1, 3) < 4) == S.true
    assert (B(1, 3) < -1) == S.false
    assert (B(1, 3) < 2).rel_op == '<'
    assert (B(1, 3) <= 2).rel_op == '<='

    assert (B(1, 3) > 4) == S.false
    assert (B(1, 3) > -1) == S.true
    assert (B(1, 3) > 2).rel_op == '>'
    assert (B(1, 3) >= 2).rel_op == '>='

    assert (B(1, 3) < B(4, 6)) == S.true
    assert (B(1, 3) < B(2, 4)).rel_op == '<'
    assert (B(1, 3) < B(-2, 0)) == S.false

    assert (B(1, 3) <= B(4, 6)) == S.true
    assert (B(1, 3) <= B(-2, 0)) == S.false

    assert (B(1, 3) > B(4, 6)) == S.false
    assert (B(1, 3) > B(-2, 0)) == S.true

    assert (B(1, 3) >= B(4, 6)) == S.false
    assert (B(1, 3) >= B(-2, 0)) == S.true

    # issue 13499
    assert (cos(x) > 0).subs(x, oo) == (B(-1, 1) > 0)

    c = Symbol('c')
    raises(TypeError, lambda: (B(0, 1) < c))
    raises(TypeError, lambda: (B(0, 1) <= c))
    raises(TypeError, lambda: (B(0, 1) > c))
    raises(TypeError, lambda: (B(0, 1) >= c))


def test_contains_AccumBounds():
    assert (1 in B(1, 2)) == S.true
    raises(TypeError, lambda: a in B(1, 2))
    assert 0 in B(-1, 0)
    raises(TypeError, lambda:
        (cos(1)**2 + sin(1)**2 - 1) in B(-1, 0))
    assert (-oo in B(1, oo)) == S.true
    assert (oo in B(-oo, 0)) == S.true

    # issue 13159
    assert Mul(0, B(-1, 1)) == Mul(B(-1, 1), 0) == 0
    import itertools
    for perm in itertools.permutations([0, B(-1, 1), x]):
        assert Mul(*perm) == 0


def test_intersection_AccumBounds():
    assert B(0, 3).intersection(B(1, 2)) == B(1, 2)
    assert B(0, 3).intersection(B(1, 4)) == B(1, 3)
    assert B(0, 3).intersection(B(-1, 2)) == B(0, 2)
    assert B(0, 3).intersection(B(-1, 4)) == B(0, 3)
    assert B(0, 1).intersection(B(2, 3)) == S.EmptySet
    raises(TypeError, lambda: B(0, 3).intersection(1))


def test_union_AccumBounds():
    assert B(0, 3).union(B(1, 2)) == B(0, 3)
    assert B(0, 3).union(B(1, 4)) == B(0, 4)
    assert B(0, 3).union(B(-1, 2)) == B(-1, 3)
    assert B(0, 3).union(B(-1, 4)) == B(-1, 4)
    raises(TypeError, lambda: B(0, 3).union(1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_matplotlib.py ===
import gc

import numpy as np
import pytest

from pandas import (
    DataFrame,
    IndexSlice,
    Series,
)

pytest.importorskip("matplotlib")
pytest.importorskip("jinja2")

import matplotlib as mpl

from pandas.io.formats.style import Styler


@pytest.fixture(autouse=True)
def mpl_cleanup():
    # matplotlib/testing/decorators.py#L24
    # 1) Resets units registry
    # 2) Resets rc_context
    # 3) Closes all figures
    mpl = pytest.importorskip("matplotlib")
    mpl_units = pytest.importorskip("matplotlib.units")
    plt = pytest.importorskip("matplotlib.pyplot")
    orig_units_registry = mpl_units.registry.copy()
    with mpl.rc_context():
        mpl.use("template")
        yield
    mpl_units.registry.clear()
    mpl_units.registry.update(orig_units_registry)
    plt.close("all")
    # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.6.0.html#garbage-collection-is-no-longer-run-on-figure-close  # noqa: E501
    gc.collect(1)


@pytest.fixture
def df():
    return DataFrame([[1, 2], [2, 4]], columns=["A", "B"])


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)


@pytest.fixture
def df_blank():
    return DataFrame([[0, 0], [0, 0]], columns=["A", "B"], index=["X", "Y"])


@pytest.fixture
def styler_blank(df_blank):
    return Styler(df_blank, uuid_len=0)


@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])
def test_function_gradient(styler, f):
    for c_map in [None, "YlOrRd"]:
        result = getattr(styler, f)(cmap=c_map)._compute().ctx
        assert all("#" in x[0][1] for x in result.values())
        assert result[(0, 0)] == result[(0, 1)]
        assert result[(1, 0)] == result[(1, 1)]


@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])
def test_background_gradient_color(styler, f):
    result = getattr(styler, f)(subset=IndexSlice[1, "A"])._compute().ctx
    if f == "background_gradient":
        assert result[(1, 0)] == [("background-color", "#fff7fb"), ("color", "#000000")]
    elif f == "text_gradient":
        assert result[(1, 0)] == [("color", "#fff7fb")]


@pytest.mark.parametrize(
    "axis, expected",
    [
        (0, ["low", "low", "high", "high"]),
        (1, ["low", "high", "low", "high"]),
        (None, ["low", "mid", "mid", "high"]),
    ],
)
@pytest.mark.parametrize("f", ["background_gradient", "text_gradient"])
def test_background_gradient_axis(styler, axis, expected, f):
    if f == "background_gradient":
        colors = {
            "low": [("background-color", "#f7fbff"), ("color", "#000000")],
            "mid": [("background-color", "#abd0e6"), ("color", "#000000")],
            "high": [("background-color", "#08306b"), ("color", "#f1f1f1")],
        }
    elif f == "text_gradient":
        colors = {
            "low": [("color", "#f7fbff")],
            "mid": [("color", "#abd0e6")],
            "high": [("color", "#08306b")],
        }
    result = getattr(styler, f)(cmap="Blues", axis=axis)._compute().ctx
    for i, cell in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        assert result[cell] == colors[expected[i]]


@pytest.mark.parametrize(
    "cmap, expected",
    [
        (
            "PuBu",
            {
                (4, 5): [("background-color", "#86b0d3"), ("color", "#000000")],
                (4, 6): [("background-color", "#83afd3"), ("color", "#f1f1f1")],
            },
        ),
        (
            "YlOrRd",
            {
                (4, 8): [("background-color", "#fd913e"), ("color", "#000000")],
                (4, 9): [("background-color", "#fd8f3d"), ("color", "#f1f1f1")],
            },
        ),
        (
            None,
            {
                (7, 0): [("background-color", "#48c16e"), ("color", "#f1f1f1")],
                (7, 1): [("background-color", "#4cc26c"), ("color", "#000000")],
            },
        ),
    ],
)
def test_text_color_threshold(cmap, expected):
    # GH 39888
    df = DataFrame(np.arange(100).reshape(10, 10))
    result = df.style.background_gradient(cmap=cmap, axis=None)._compute().ctx
    for k in expected.keys():
        assert result[k] == expected[k]


def test_background_gradient_vmin_vmax():
    # GH 12145
    df = DataFrame(range(5))
    ctx = df.style.background_gradient(vmin=1, vmax=3)._compute().ctx
    assert ctx[(0, 0)] == ctx[(1, 0)]
    assert ctx[(4, 0)] == ctx[(3, 0)]


def test_background_gradient_int64():
    # GH 28869
    df1 = Series(range(3)).to_frame()
    df2 = Series(range(3), dtype="Int64").to_frame()
    ctx1 = df1.style.background_gradient()._compute().ctx
    ctx2 = df2.style.background_gradient()._compute().ctx
    assert ctx2[(0, 0)] == ctx1[(0, 0)]
    assert ctx2[(1, 0)] == ctx1[(1, 0)]
    assert ctx2[(2, 0)] == ctx1[(2, 0)]


@pytest.mark.parametrize(
    "axis, gmap, expected",
    [
        (
            0,
            [1, 2],
            {
                (0, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 0): [("background-color", "#023858"), ("color", "#f1f1f1")],
                (0, 1): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
            },
        ),
        (
            1,
            [1, 2],
            {
                (0, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (0, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
            },
        ),
        (
            None,
            np.array([[2, 1], [1, 2]]),
            {
                (0, 0): [("background-color", "#023858"), ("color", "#f1f1f1")],
                (1, 0): [("background-color", "#fff7fb"), ("color", "#000000")],
                (0, 1): [("background-color", "#fff7fb"), ("color", "#000000")],
                (1, 1): [("background-color", "#023858"), ("color", "#f1f1f1")],
            },
        ),
    ],
)
def test_background_gradient_gmap_array(styler_blank, axis, gmap, expected):
    # tests when gmap is given as a sequence and converted to ndarray
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "gmap, axis", [([1, 2, 3], 0), ([1, 2], 1), (np.array([[1, 2], [1, 2]]), None)]
)
def test_background_gradient_gmap_array_raises(gmap, axis):
    # test when gmap as converted ndarray is bad shape
    df = DataFrame([[0, 0, 0], [0, 0, 0]])
    msg = "supplied 'gmap' is not correct shape"
    with pytest.raises(ValueError, match=msg):
        df.style.background_gradient(gmap=gmap, axis=axis)._compute()


@pytest.mark.parametrize(
    "gmap",
    [
        DataFrame(  # reverse the columns
            [[2, 1], [1, 2]], columns=["B", "A"], index=["X", "Y"]
        ),
        DataFrame(  # reverse the index
            [[2, 1], [1, 2]], columns=["A", "B"], index=["Y", "X"]
        ),
        DataFrame(  # reverse the index and columns
            [[1, 2], [2, 1]], columns=["B", "A"], index=["Y", "X"]
        ),
        DataFrame(  # add unnecessary columns
            [[1, 2, 3], [2, 1, 3]], columns=["A", "B", "C"], index=["X", "Y"]
        ),
        DataFrame(  # add unnecessary index
            [[1, 2], [2, 1], [3, 3]], columns=["A", "B"], index=["X", "Y", "Z"]
        ),
    ],
)
@pytest.mark.parametrize(
    "subset, exp_gmap",  # exp_gmap is underlying map DataFrame should conform to
    [
        (None, [[1, 2], [2, 1]]),
        (["A"], [[1], [2]]),  # slice only column "A" in data and gmap
        (["B", "A"], [[2, 1], [1, 2]]),  # reverse the columns in data
        (IndexSlice["X", :], [[1, 2]]),  # slice only index "X" in data and gmap
        (IndexSlice[["Y", "X"], :], [[2, 1], [1, 2]]),  # reverse the index in data
    ],
)
def test_background_gradient_gmap_dataframe_align(styler_blank, gmap, subset, exp_gmap):
    # test gmap given as DataFrame that it aligns to the data including subset
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap, subset=subset)
    result = styler_blank.background_gradient(axis=None, gmap=gmap, subset=subset)
    assert expected._compute().ctx == result._compute().ctx


@pytest.mark.parametrize(
    "gmap, axis, exp_gmap",
    [
        (Series([2, 1], index=["Y", "X"]), 0, [[1, 1], [2, 2]]),  # revrse the index
        (Series([2, 1], index=["B", "A"]), 1, [[1, 2], [1, 2]]),  # revrse the cols
        (Series([1, 2, 3], index=["X", "Y", "Z"]), 0, [[1, 1], [2, 2]]),  # add idx
        (Series([1, 2, 3], index=["A", "B", "C"]), 1, [[1, 2], [1, 2]]),  # add col
    ],
)
def test_background_gradient_gmap_series_align(styler_blank, gmap, axis, exp_gmap):
    # test gmap given as Series that it aligns to the data including subset
    expected = styler_blank.background_gradient(axis=None, gmap=exp_gmap)._compute()
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute()
    assert expected.ctx == result.ctx


@pytest.mark.parametrize(
    "gmap, axis",
    [
        (DataFrame([[1, 2], [2, 1]], columns=["A", "B"], index=["X", "Y"]), 1),
        (DataFrame([[1, 2], [2, 1]], columns=["A", "B"], index=["X", "Y"]), 0),
    ],
)
def test_background_gradient_gmap_wrong_dataframe(styler_blank, gmap, axis):
    # test giving a gmap in DataFrame but with wrong axis
    msg = "'gmap' is a DataFrame but underlying data for operations is a Series"
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=axis)._compute()


def test_background_gradient_gmap_wrong_series(styler_blank):
    # test giving a gmap in Series form but with wrong axis
    msg = "'gmap' is a Series but underlying data for operations is a DataFrame"
    gmap = Series([1, 2], index=["X", "Y"])
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=None)._compute()


def test_background_gradient_nullable_dtypes():
    # GH 50712
    df1 = DataFrame([[1], [0], [np.nan]], dtype=float)
    df2 = DataFrame([[1], [0], [None]], dtype="Int64")

    ctx1 = df1.style.background_gradient()._compute().ctx
    ctx2 = df2.style.background_gradient()._compute().ctx
    assert ctx1 == ctx2


@pytest.mark.parametrize(
    "cmap",
    ["PuBu", mpl.colormaps["PuBu"]],
)
def test_bar_colormap(cmap):
    data = DataFrame([[1, 2], [3, 4]])
    ctx = data.style.bar(cmap=cmap, axis=None)._compute().ctx
    pubu_colors = {
        (0, 0): "#d0d1e6",
        (1, 0): "#056faf",
        (0, 1): "#73a9cf",
        (1, 1): "#023858",
    }
    for k, v in pubu_colors.items():
        assert v in ctx[k][1][1]


def test_bar_color_raises(df):
    msg = "`color` must be string or list or tuple of 2 strings"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color={"a", "b"}).to_html()
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=["a", "b", "c"]).to_html()

    msg = "`color` and `cmap` cannot both be given"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color="something", cmap="something else").to_html()


@pytest.mark.parametrize(
    "plot_method",
    ["scatter", "hexbin"],
)
def test_pass_colormap_instance(df, plot_method):
    # https://github.com/pandas-dev/pandas/issues/49374
    cmap = mpl.colors.ListedColormap([[1, 1, 1], [0, 0, 0]])
    df["c"] = df.A + df.B
    kwargs = {"x": "A", "y": "B", "c": "c", "colormap": cmap}
    if plot_method == "hexbin":
        kwargs["C"] = kwargs.pop("c")
    getattr(df.plot, plot_method)(**kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_numpy.py ===
# This testfile tests SymPy <-> NumPy compatibility

# Don't test any SymPy features here. Just pure interaction with NumPy.
# Always write regular SymPy tests for anything, that can be tested in pure
# Python (without numpy). Here we test everything, that a user may need when
# using SymPy with NumPy
from sympy.external.importtools import version_tuple
from sympy.external import import_module

numpy = import_module('numpy')
if numpy:
    array, matrix, ndarray = numpy.array, numpy.matrix, numpy.ndarray
else:
    #bin/test will not execute any tests now
    disabled = True


from sympy.core.numbers import (Float, Integer, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import (Matrix, list2numpy, matrix2numpy, symarray)
from sympy.utilities.lambdify import lambdify
import sympy

import mpmath
from sympy.abc import x, y, z
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.testing.pytest import raises


# first, systematically check, that all operations are implemented and don't
# raise an exception


def test_systematic_basic():
    def s(sympy_object, numpy_array):
        _ = [sympy_object + numpy_array,
        numpy_array + sympy_object,
        sympy_object - numpy_array,
        numpy_array - sympy_object,
        sympy_object * numpy_array,
        numpy_array * sympy_object,
        sympy_object / numpy_array,
        numpy_array / sympy_object,
        sympy_object ** numpy_array,
        numpy_array ** sympy_object]
    x = Symbol("x")
    y = Symbol("y")
    sympy_objs = [
        Rational(2, 3),
        Float("1.3"),
        x,
        y,
        pow(x, y)*y,
        Integer(5),
        Float(5.5),
    ]
    numpy_objs = [
        array([1]),
        array([3, 8, -1]),
        array([x, x**2, Rational(5)]),
        array([x/y*sin(y), 5, Rational(5)]),
    ]
    for x in sympy_objs:
        for y in numpy_objs:
            s(x, y)


# now some random tests, that test particular problems and that also
# check that the results of the operations are correct

def test_basics():
    one = Rational(1)
    zero = Rational(0)
    assert array(1) == array(one)
    assert array([one]) == array([one])
    assert array([x]) == array([x])
    assert array(x) == array(Symbol("x"))
    assert array(one + x) == array(1 + x)

    X = array([one, zero, zero])
    assert (X == array([one, zero, zero])).all()
    assert (X == array([one, 0, 0])).all()


def test_arrays():
    one = Rational(1)
    zero = Rational(0)
    X = array([one, zero, zero])
    Y = one*X
    X = array([Symbol("a") + Rational(1, 2)])
    Y = X + X
    assert Y == array([1 + 2*Symbol("a")])
    Y = Y + 1
    assert Y == array([2 + 2*Symbol("a")])
    Y = X - X
    assert Y == array([0])


def test_conversion1():
    a = list2numpy([x**2, x])
    #looks like an array?
    assert isinstance(a, ndarray)
    assert a[0] == x**2
    assert a[1] == x
    assert len(a) == 2
    #yes, it's the array


def test_conversion2():
    a = 2*list2numpy([x**2, x])
    b = list2numpy([2*x**2, 2*x])
    assert (a == b).all()

    one = Rational(1)
    zero = Rational(0)
    X = list2numpy([one, zero, zero])
    Y = one*X
    X = list2numpy([Symbol("a") + Rational(1, 2)])
    Y = X + X
    assert Y == array([1 + 2*Symbol("a")])
    Y = Y + 1
    assert Y == array([2 + 2*Symbol("a")])
    Y = X - X
    assert Y == array([0])


def test_list2numpy():
    assert (array([x**2, x]) == list2numpy([x**2, x])).all()


def test_Matrix1():
    m = Matrix([[x, x**2], [5, 2/x]])
    assert (array(m.subs(x, 2)) == array([[2, 4], [5, 1]])).all()
    m = Matrix([[sin(x), x**2], [5, 2/x]])
    assert (array(m.subs(x, 2)) == array([[sin(2), 4], [5, 1]])).all()


def test_Matrix2():
    m = Matrix([[x, x**2], [5, 2/x]])
    with ignore_warnings(PendingDeprecationWarning):
        assert (matrix(m.subs(x, 2)) == matrix([[2, 4], [5, 1]])).all()
    m = Matrix([[sin(x), x**2], [5, 2/x]])
    with ignore_warnings(PendingDeprecationWarning):
        assert (matrix(m.subs(x, 2)) == matrix([[sin(2), 4], [5, 1]])).all()


def test_Matrix3():
    a = array([[2, 4], [5, 1]])
    assert Matrix(a) == Matrix([[2, 4], [5, 1]])
    assert Matrix(a) != Matrix([[2, 4], [5, 2]])
    a = array([[sin(2), 4], [5, 1]])
    assert Matrix(a) == Matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) != Matrix([[sin(0), 4], [5, 1]])


def test_Matrix4():
    with ignore_warnings(PendingDeprecationWarning):
        a = matrix([[2, 4], [5, 1]])
    assert Matrix(a) == Matrix([[2, 4], [5, 1]])
    assert Matrix(a) != Matrix([[2, 4], [5, 2]])
    with ignore_warnings(PendingDeprecationWarning):
        a = matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) == Matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) != Matrix([[sin(0), 4], [5, 1]])


def test_Matrix_sum():
    M = Matrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 3, 4], [x, 5, 6], [x, y, z**2]])
    assert M + m == Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    assert m + M == Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    assert M + m == M.add(m)


def test_Matrix_mul():
    M = Matrix([[1, 2, 3], [x, y, x]])
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 4], [x, 6], [x, z**2]])
    assert M*m == Matrix([
        [         2 + 5*x,        16 + 3*z**2],
        [2*x + x*y + x**2, 4*x + 6*y + x*z**2],
    ])

    assert m*M == Matrix([
        [   2 + 4*x,      4 + 4*y,      6 + 4*x],
        [       7*x,    2*x + 6*y,          9*x],
        [x + x*z**2, 2*x + y*z**2, 3*x + x*z**2],
    ])
    a = array([2])
    assert a[0] * M == 2 * M
    assert M * a[0] == 2 * M


def test_Matrix_array():
    class matarray:
        def __array__(self, dtype=object, copy=None):
            if copy is not None and not copy:
                raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
            from numpy import array
            return array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matarr = matarray()
    assert Matrix(matarr) == Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_matrix2numpy():
    a = matrix2numpy(Matrix([[1, x**2], [3*sin(x), 0]]))
    assert isinstance(a, ndarray)
    assert a.shape == (2, 2)
    assert a[0, 0] == 1
    assert a[0, 1] == x**2
    assert a[1, 0] == 3*sin(x)
    assert a[1, 1] == 0


def test_matrix2numpy_conversion():
    a = Matrix([[1, 2, sin(x)], [x**2, x, Rational(1, 2)]])
    b = array([[1, 2, sin(x)], [x**2, x, Rational(1, 2)]])
    assert (matrix2numpy(a) == b).all()
    assert matrix2numpy(a).dtype == numpy.dtype('object')

    c = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='int8')
    d = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='float64')
    assert c.dtype == numpy.dtype('int8')
    assert d.dtype == numpy.dtype('float64')


def test_issue_3728():
    assert (Rational(1, 2)*array([2*x, 0]) == array([x, 0])).all()
    assert (Rational(1, 2) + array(
        [2*x, 0]) == array([2*x + Rational(1, 2), Rational(1, 2)])).all()
    assert (Float("0.5")*array([2*x, 0]) == array([Float("1.0")*x, 0])).all()
    assert (Float("0.5") + array(
        [2*x, 0]) == array([2*x + Float("0.5"), Float("0.5")])).all()


@conserve_mpmath_dps
def test_lambdify():
    mpmath.mp.dps = 16
    sin02 = mpmath.mpf("0.198669330795061215459412627")
    f = lambdify(x, sin(x), "numpy")
    prec = 1e-15
    assert -prec < f(0.2) - sin02 < prec

    # if this succeeds, it can't be a numpy function

    if version_tuple(numpy.__version__) >= version_tuple('1.17'):
        with raises(TypeError):
            f(x)
    else:
        with raises(AttributeError):
            f(x)


def test_lambdify_matrix():
    f = lambdify(x, Matrix([[x, 2*x], [1, 2]]), [{'ImmutableMatrix': numpy.array}, "numpy"])
    assert (f(1) == array([[1, 2], [1, 2]])).all()


def test_lambdify_matrix_multi_input():
    M = sympy.Matrix([[x**2, x*y, x*z],
                      [y*x, y**2, y*z],
                      [z*x, z*y, z**2]])
    f = lambdify((x, y, z), M, [{'ImmutableMatrix': numpy.array}, "numpy"])

    xh, yh, zh = 1.0, 2.0, 3.0
    expected = array([[xh**2, xh*yh, xh*zh],
                      [yh*xh, yh**2, yh*zh],
                      [zh*xh, zh*yh, zh**2]])
    actual = f(xh, yh, zh)
    assert numpy.allclose(actual, expected)


def test_lambdify_matrix_vec_input():
    X = sympy.DeferredVector('X')
    M = Matrix([
        [X[0]**2, X[0]*X[1], X[0]*X[2]],
        [X[1]*X[0], X[1]**2, X[1]*X[2]],
        [X[2]*X[0], X[2]*X[1], X[2]**2]])
    f = lambdify(X, M, [{'ImmutableMatrix': numpy.array}, "numpy"])

    Xh = array([1.0, 2.0, 3.0])
    expected = array([[Xh[0]**2, Xh[0]*Xh[1], Xh[0]*Xh[2]],
                      [Xh[1]*Xh[0], Xh[1]**2, Xh[1]*Xh[2]],
                      [Xh[2]*Xh[0], Xh[2]*Xh[1], Xh[2]**2]])
    actual = f(Xh)
    assert numpy.allclose(actual, expected)


def test_lambdify_transl():
    from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
    for sym, mat in NUMPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert mat in numpy.__dict__


def test_symarray():
    """Test creation of numpy arrays of SymPy symbols."""

    import numpy as np
    import numpy.testing as npt

    syms = symbols('_0,_1,_2')
    s1 = symarray("", 3)
    s2 = symarray("", 3)
    npt.assert_array_equal(s1, np.array(syms, dtype=object))
    assert s1[0] == s2[0]

    a = symarray('a', 3)
    b = symarray('b', 3)
    assert not(a[0] == b[0])

    asyms = symbols('a_0,a_1,a_2')
    npt.assert_array_equal(a, np.array(asyms, dtype=object))

    # Multidimensional checks
    a2d = symarray('a', (2, 3))
    assert a2d.shape == (2, 3)
    a00, a12 = symbols('a_0_0,a_1_2')
    assert a2d[0, 0] == a00
    assert a2d[1, 2] == a12

    a3d = symarray('a', (2, 3, 2))
    assert a3d.shape == (2, 3, 2)
    a000, a120, a121 = symbols('a_0_0_0,a_1_2_0,a_1_2_1')
    assert a3d[0, 0, 0] == a000
    assert a3d[1, 2, 0] == a120
    assert a3d[1, 2, 1] == a121


def test_vectorize():
    assert (numpy.vectorize(
        sin)([1, 2, 3]) == numpy.array([sin(1), sin(2), sin(3)])).all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_skiprows.py ===
"""
Tests that skipped rows are properly handled during
parsing for all of the parsers defined in parsers.py
"""

from datetime import datetime
from io import StringIO

import numpy as np
import pytest

from pandas.errors import EmptyDataError

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
@pytest.mark.parametrize("skiprows", [list(range(6)), 6])
def test_skip_rows_bug(all_parsers, skiprows):
    # see gh-505
    parser = all_parsers
    text = """#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
1/1/2000,1.,2.,3.
1/2/2000,4,5,6
1/3/2000,7,8,9
"""
    result = parser.read_csv(
        StringIO(text), skiprows=skiprows, header=None, index_col=0, parse_dates=True
    )
    index = Index(
        [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], name=0
    )

    expected = DataFrame(
        np.arange(1.0, 10.0).reshape((3, 3)), columns=[1, 2, 3], index=index
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_deep_skip_rows(all_parsers):
    # see gh-4382
    parser = all_parsers
    data = "a,b,c\n" + "\n".join(
        [",".join([str(i), str(i + 1), str(i + 2)]) for i in range(10)]
    )
    condensed_data = "a,b,c\n" + "\n".join(
        [",".join([str(i), str(i + 1), str(i + 2)]) for i in [0, 1, 2, 3, 4, 6, 8, 9]]
    )

    result = parser.read_csv(StringIO(data), skiprows=[6, 8])
    condensed_result = parser.read_csv(StringIO(condensed_data))
    tm.assert_frame_equal(result, condensed_result)


@xfail_pyarrow  # AssertionError: DataFrame are different
def test_skip_rows_blank(all_parsers):
    # see gh-9832
    parser = all_parsers
    text = """#foo,a,b,c
#foo,a,b,c

#foo,a,b,c
#foo,a,b,c

1/1/2000,1.,2.,3.
1/2/2000,4,5,6
1/3/2000,7,8,9
"""
    data = parser.read_csv(
        StringIO(text), skiprows=6, header=None, index_col=0, parse_dates=True
    )
    index = Index(
        [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], name=0
    )

    expected = DataFrame(
        np.arange(1.0, 10.0).reshape((3, 3)), columns=[1, 2, 3], index=index
    )
    tm.assert_frame_equal(data, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            """id,text,num_lines
1,"line 11
line 12",2
2,"line 21
line 22",2
3,"line 31",1""",
            {"skiprows": [1]},
            DataFrame(
                [[2, "line 21\nline 22", 2], [3, "line 31", 1]],
                columns=["id", "text", "num_lines"],
            ),
        ),
        (
            "a,b,c\n~a\n b~,~e\n d~,~f\n f~\n1,2,~12\n 13\n 14~",
            {"quotechar": "~", "skiprows": [2]},
            DataFrame([["a\n b", "e\n d", "f\n f"]], columns=["a", "b", "c"]),
        ),
        (
            (
                "Text,url\n~example\n "
                "sentence\n one~,url1\n~"
                "example\n sentence\n two~,url2\n~"
                "example\n sentence\n three~,url3"
            ),
            {"quotechar": "~", "skiprows": [1, 3]},
            DataFrame([["example\n sentence\n two", "url2"]], columns=["Text", "url"]),
        ),
    ],
)
@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_skip_row_with_newline(all_parsers, data, kwargs, expected):
    # see gh-12775 and gh-10911
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_skip_row_with_quote(all_parsers):
    # see gh-12775 and gh-10911
    parser = all_parsers
    data = """id,text,num_lines
1,"line '11' line 12",2
2,"line '21' line 22",2
3,"line '31' line 32",1"""

    exp_data = [[2, "line '21' line 22", 2], [3, "line '31' line 32", 1]]
    expected = DataFrame(exp_data, columns=["id", "text", "num_lines"])

    result = parser.read_csv(StringIO(data), skiprows=[1])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,exp_data",
    [
        (
            """id,text,num_lines
1,"line \n'11' line 12",2
2,"line \n'21' line 22",2
3,"line \n'31' line 32",1""",
            [[2, "line \n'21' line 22", 2], [3, "line \n'31' line 32", 1]],
        ),
        (
            """id,text,num_lines
1,"line '11\n' line 12",2
2,"line '21\n' line 22",2
3,"line '31\n' line 32",1""",
            [[2, "line '21\n' line 22", 2], [3, "line '31\n' line 32", 1]],
        ),
        (
            """id,text,num_lines
1,"line '11\n' \r\tline 12",2
2,"line '21\n' \r\tline 22",2
3,"line '31\n' \r\tline 32",1""",
            [[2, "line '21\n' \r\tline 22", 2], [3, "line '31\n' \r\tline 32", 1]],
        ),
    ],
)
@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_skip_row_with_newline_and_quote(all_parsers, data, exp_data):
    # see gh-12775 and gh-10911
    parser = all_parsers
    result = parser.read_csv(StringIO(data), skiprows=[1])

    expected = DataFrame(exp_data, columns=["id", "text", "num_lines"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: The 'delim_whitespace' option is not supported
@pytest.mark.parametrize(
    "lineterminator", ["\n", "\r\n", "\r"]  # "LF"  # "CRLF"  # "CR"
)
def test_skiprows_lineterminator(all_parsers, lineterminator, request):
    # see gh-9079
    parser = all_parsers
    data = "\n".join(
        [
            "SMOSMANIA ThetaProbe-ML2X ",
            "2007/01/01 01:00   0.2140 U M ",
            "2007/01/01 02:00   0.2141 M O ",
            "2007/01/01 04:00   0.2142 D M ",
        ]
    )
    expected = DataFrame(
        [
            ["2007/01/01", "01:00", 0.2140, "U", "M"],
            ["2007/01/01", "02:00", 0.2141, "M", "O"],
            ["2007/01/01", "04:00", 0.2142, "D", "M"],
        ],
        columns=["date", "time", "var", "flag", "oflag"],
    )

    if parser.engine == "python" and lineterminator == "\r":
        mark = pytest.mark.xfail(reason="'CR' not respect with the Python parser yet")
        request.applymarker(mark)

    data = data.replace("\n", lineterminator)

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data),
            skiprows=1,
            delim_whitespace=True,
            names=["date", "time", "var", "flag", "oflag"],
        )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # AssertionError: DataFrame are different
def test_skiprows_infield_quote(all_parsers):
    # see gh-14459
    parser = all_parsers
    data = 'a"\nb"\na\n1'
    expected = DataFrame({"a": [1]})

    result = parser.read_csv(StringIO(data), skiprows=2)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, DataFrame({"1": [3, 5]})),
        ({"header": 0, "names": ["foo"]}, DataFrame({"foo": [3, 5]})),
    ],
)
def test_skip_rows_callable(all_parsers, kwargs, expected):
    parser = all_parsers
    data = "a\n1\n2\n3\n4\n5"

    result = parser.read_csv(StringIO(data), skiprows=lambda x: x % 2 == 0, **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_skip_rows_callable_not_in(all_parsers):
    parser = all_parsers
    data = "0,a\n1,b\n2,c\n3,d\n4,e"
    expected = DataFrame([[1, "b"], [3, "d"]])

    result = parser.read_csv(
        StringIO(data), header=None, skiprows=lambda x: x not in [1, 3]
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_skip_rows_skip_all(all_parsers):
    parser = all_parsers
    data = "a\n1\n2\n3\n4\n5"
    msg = "No columns to parse from file"

    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data), skiprows=lambda x: True)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_skip_rows_bad_callable(all_parsers):
    msg = "by zero"
    parser = all_parsers
    data = "a\n1\n2\n3\n4\n5"

    with pytest.raises(ZeroDivisionError, match=msg):
        parser.read_csv(StringIO(data), skiprows=lambda x: 1 / 0)


@xfail_pyarrow  # ValueError: skiprows argument must be an integer
def test_skip_rows_and_n_rows(all_parsers):
    # GH#44021
    data = """a,b
1,a
2,b
3,c
4,d
5,e
6,f
7,g
8,h
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), nrows=5, skiprows=[2, 4, 6])
    expected = DataFrame({"a": [1, 3, 5, 7, 8], "b": ["a", "c", "e", "g", "h"]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_skip_rows_with_chunks(all_parsers):
    # GH 55677
    data = """col_a
10
20
30
40
50
60
70
80
90
100
"""
    parser = all_parsers
    reader = parser.read_csv(
        StringIO(data), engine=parser, skiprows=lambda x: x in [1, 4, 5], chunksize=4
    )
    df1 = next(reader)
    df2 = next(reader)

    tm.assert_frame_equal(df1, DataFrame({"col_a": [20, 30, 60, 70]}))
    tm.assert_frame_equal(df2, DataFrame({"col_a": [80, 90, 100]}, index=[4, 5, 6]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\dtypes\test_categorical.py ===
"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO
import os

import numpy as np
import pytest

from pandas._libs import parsers as libparsers

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Timestamp,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
@pytest.mark.parametrize(
    "dtype",
    [
        "category",
        CategoricalDtype(),
        {"a": "category", "b": "category", "c": CategoricalDtype()},
    ],
)
def test_categorical_dtype(all_parsers, dtype):
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,a,3.4
1,a,3.4
2,b,4.5"""
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["a", "a", "b"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("dtype", [{"b": "category"}, {1: "category"}])
def test_categorical_dtype_single(all_parsers, dtype, request):
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,a,3.4
1,a,3.4
2,b,4.5"""
    expected = DataFrame(
        {"a": [1, 1, 2], "b": Categorical(["a", "a", "b"]), "c": [3.4, 3.4, 4.5]}
    )
    if parser.engine == "pyarrow":
        mark = pytest.mark.xfail(
            strict=False,
            reason="Flaky test sometimes gives object dtype instead of Categorical",
        )
        request.applymarker(mark)

    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
def test_categorical_dtype_unsorted(all_parsers):
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,b,3.4
1,b,3.4
2,a,4.5"""
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["b", "b", "a"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype="category")
    tm.assert_frame_equal(actual, expected)


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
def test_categorical_dtype_missing(all_parsers):
    # see gh-10153
    parser = all_parsers
    data = """a,b,c
1,b,3.4
1,nan,3.4
2,a,4.5"""
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["b", np.nan, "a"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    actual = parser.read_csv(StringIO(data), dtype="category")
    tm.assert_frame_equal(actual, expected)


@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
@pytest.mark.slow
def test_categorical_dtype_high_cardinality_numeric(all_parsers, monkeypatch):
    # see gh-18186
    # was an issue with C parser, due to DEFAULT_BUFFER_HEURISTIC
    parser = all_parsers
    heuristic = 2**5
    data = np.sort([str(i) for i in range(heuristic + 1)])
    expected = DataFrame({"a": Categorical(data, ordered=True)})
    with monkeypatch.context() as m:
        m.setattr(libparsers, "DEFAULT_BUFFER_HEURISTIC", heuristic)
        actual = parser.read_csv(StringIO("a\n" + "\n".join(data)), dtype="category")
    actual["a"] = actual["a"].cat.reorder_categories(
        np.sort(actual.a.cat.categories), ordered=True
    )
    tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_utf16(all_parsers, csv_dir_path):
    # see gh-10153
    pth = os.path.join(csv_dir_path, "utf16_ex.txt")
    parser = all_parsers
    encoding = "utf-16"
    sep = "\t"

    expected = parser.read_csv(pth, sep=sep, encoding=encoding)
    expected = expected.apply(Categorical)

    actual = parser.read_csv(pth, sep=sep, encoding=encoding, dtype="category")
    tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_chunksize_infer_categories(all_parsers):
    # see gh-10153
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    expecteds = [
        DataFrame({"a": [1, 1], "b": Categorical(["a", "b"])}),
        DataFrame({"a": [1, 2], "b": Categorical(["b", "c"])}, index=[2, 3]),
    ]

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={"b": "category"}, chunksize=2)
        return

    with parser.read_csv(
        StringIO(data), dtype={"b": "category"}, chunksize=2
    ) as actuals:
        for actual, expected in zip(actuals, expecteds):
            tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_chunksize_explicit_categories(all_parsers):
    # see gh-10153
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    cats = ["a", "b", "c"]
    expecteds = [
        DataFrame({"a": [1, 1], "b": Categorical(["a", "b"], categories=cats)}),
        DataFrame(
            {"a": [1, 2], "b": Categorical(["b", "c"], categories=cats)},
            index=[2, 3],
        ),
    ]
    dtype = CategoricalDtype(cats)

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={"b": dtype}, chunksize=2)
        return

    with parser.read_csv(StringIO(data), dtype={"b": dtype}, chunksize=2) as actuals:
        for actual, expected in zip(actuals, expecteds):
            tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_latin1(all_parsers, csv_dir_path):
    # see gh-10153
    pth = os.path.join(csv_dir_path, "unicode_series.csv")
    parser = all_parsers
    encoding = "latin-1"

    expected = parser.read_csv(pth, header=None, encoding=encoding)
    expected[1] = Categorical(expected[1])

    actual = parser.read_csv(pth, header=None, encoding=encoding, dtype={1: "category"})
    tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "categories",
    [["a", "b", "c"], ["a", "c", "b"], ["a", "b", "c", "d"], ["c", "b", "a"]],
)
def test_categorical_category_dtype(all_parsers, categories, ordered):
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    expected = DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": Categorical(
                ["a", "b", "b", "c"], categories=categories, ordered=ordered
            ),
        }
    )

    dtype = {"b": CategoricalDtype(categories=categories, ordered=ordered)}
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_category_dtype_unsorted(all_parsers):
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    dtype = CategoricalDtype(["c", "b", "a"])
    expected = DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": Categorical(["a", "b", "b", "c"], categories=["c", "b", "a"]),
        }
    )

    result = parser.read_csv(StringIO(data), dtype={"b": dtype})
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_numeric(all_parsers):
    parser = all_parsers
    dtype = {"b": CategoricalDtype([1, 2, 3])}

    data = "b\n1\n1\n2\n3"
    expected = DataFrame({"b": Categorical([1, 1, 2, 3])})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_datetime(all_parsers):
    parser = all_parsers
    dti = pd.DatetimeIndex(["2017-01-01", "2018-01-01", "2019-01-01"], freq=None)
    dtype = {"b": CategoricalDtype(dti)}

    data = "b\n2017-01-01\n2018-01-01\n2019-01-01"
    expected = DataFrame({"b": Categorical(dtype["b"].categories)})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_timestamp(all_parsers):
    parser = all_parsers
    dtype = {"b": CategoricalDtype([Timestamp("2014")])}

    data = "b\n2014-01-01\n2014-01-01"
    expected = DataFrame({"b": Categorical([Timestamp("2014")] * 2)})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_coerces_timedelta(all_parsers):
    parser = all_parsers
    dtype = {"b": CategoricalDtype(pd.to_timedelta(["1h", "2h", "3h"]))}

    data = "b\n1h\n2h\n3h"
    expected = DataFrame({"b": Categorical(dtype["b"].categories)})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        "b\nTrue\nFalse\nNA\nFalse",
        "b\ntrue\nfalse\nNA\nfalse",
        "b\nTRUE\nFALSE\nNA\nFALSE",
        "b\nTrue\nFalse\nNA\nFALSE",
    ],
)
def test_categorical_dtype_coerces_boolean(all_parsers, data):
    # see gh-20498
    parser = all_parsers
    dtype = {"b": CategoricalDtype([False, True])}
    expected = DataFrame({"b": Categorical([True, False, None, False])})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


def test_categorical_unexpected_categories(all_parsers):
    parser = all_parsers
    dtype = {"b": CategoricalDtype(["a", "b", "d", "e"])}

    data = "b\nd\na\nc\nd"  # Unexpected c
    expected = DataFrame({"b": Categorical(list("dacd"), dtype=dtype["b"])})

    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\test_arithmetic.py ===
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from dateutil.tz import gettz
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    Timedelta,
    Timestamp,
    offsets,
    to_offset,
)

import pandas._testing as tm


class TestTimestampArithmetic:
    def test_overflow_offset(self):
        # no overflow expected

        stamp = Timestamp("2000/1/1")
        offset_no_overflow = to_offset("D") * 100

        expected = Timestamp("2000/04/10")
        assert stamp + offset_no_overflow == expected

        assert offset_no_overflow + stamp == expected

        expected = Timestamp("1999/09/23")
        assert stamp - offset_no_overflow == expected

    def test_overflow_offset_raises(self):
        # xref https://github.com/statsmodels/statsmodels/issues/3374
        # ends up multiplying really large numbers which overflow

        stamp = Timestamp("2017-01-13 00:00:00").as_unit("ns")
        offset_overflow = 20169940 * offsets.Day(1)
        lmsg2 = r"Cannot cast -?20169940 days \+?00:00:00 to unit='ns' without overflow"

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):
            stamp + offset_overflow

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):
            offset_overflow + stamp

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):
            stamp - offset_overflow

        # xref https://github.com/pandas-dev/pandas/issues/14080
        # used to crash, so check for proper overflow exception

        stamp = Timestamp("2000/1/1").as_unit("ns")
        offset_overflow = to_offset("D") * 100**5

        lmsg3 = (
            r"Cannot cast -?10000000000 days \+?00:00:00 to unit='ns' without overflow"
        )
        with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):
            stamp + offset_overflow

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):
            offset_overflow + stamp

        with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):
            stamp - offset_overflow

    def test_overflow_timestamp_raises(self):
        # https://github.com/pandas-dev/pandas/issues/31774
        msg = "Result is too large"
        a = Timestamp("2101-01-01 00:00:00").as_unit("ns")
        b = Timestamp("1688-01-01 00:00:00").as_unit("ns")

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            a - b

        # but we're OK for timestamp and datetime.datetime
        assert (a - b.to_pydatetime()) == (a.to_pydatetime() - b)

    def test_delta_preserve_nanos(self):
        val = Timestamp(1337299200000000123)
        result = val + timedelta(1)
        assert result.nanosecond == val.nanosecond

    def test_rsub_dtscalars(self, tz_naive_fixture):
        # In particular, check that datetime64 - Timestamp works GH#28286
        td = Timedelta(1235345642000)
        ts = Timestamp("2021-01-01", tz=tz_naive_fixture)
        other = ts + td

        assert other - ts == td
        assert other.to_pydatetime() - ts == td
        if tz_naive_fixture is None:
            assert other.to_datetime64() - ts == td
        else:
            msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
            with pytest.raises(TypeError, match=msg):
                other.to_datetime64() - ts

    def test_timestamp_sub_datetime(self):
        dt = datetime(2013, 10, 12)
        ts = Timestamp(datetime(2013, 10, 13))
        assert (ts - dt).days == 1
        assert (dt - ts).days == -1

    def test_subtract_tzaware_datetime(self):
        t1 = Timestamp("2020-10-22T22:00:00+00:00")
        t2 = datetime(2020, 10, 22, 22, tzinfo=timezone.utc)

        result = t1 - t2

        assert isinstance(result, Timedelta)
        assert result == Timedelta("0 days")

    def test_subtract_timestamp_from_different_timezone(self):
        t1 = Timestamp("20130101").tz_localize("US/Eastern")
        t2 = Timestamp("20130101").tz_localize("CET")

        result = t1 - t2

        assert isinstance(result, Timedelta)
        assert result == Timedelta("0 days 06:00:00")

    def test_subtracting_involving_datetime_with_different_tz(self):
        t1 = datetime(2013, 1, 1, tzinfo=timezone(timedelta(hours=-5)))
        t2 = Timestamp("20130101").tz_localize("CET")

        result = t1 - t2

        assert isinstance(result, Timedelta)
        assert result == Timedelta("0 days 06:00:00")

        result = t2 - t1
        assert isinstance(result, Timedelta)
        assert result == Timedelta("-1 days +18:00:00")

    def test_subtracting_different_timezones(self, tz_aware_fixture):
        t_raw = Timestamp("20130101")
        t_UTC = t_raw.tz_localize("UTC")
        t_diff = t_UTC.tz_convert(tz_aware_fixture) + Timedelta("0 days 05:00:00")

        result = t_diff - t_UTC

        assert isinstance(result, Timedelta)
        assert result == Timedelta("0 days 05:00:00")

    def test_addition_subtraction_types(self):
        # Assert on the types resulting from Timestamp +/- various date/time
        # objects
        dt = datetime(2014, 3, 4)
        td = timedelta(seconds=1)
        ts = Timestamp(dt)

        msg = "Addition/subtraction of integers"
        with pytest.raises(TypeError, match=msg):
            # GH#22535 add/sub with integers is deprecated
            ts + 1
        with pytest.raises(TypeError, match=msg):
            ts - 1

        # Timestamp + datetime not supported, though subtraction is supported
        # and yields timedelta more tests in tseries/base/tests/test_base.py
        assert type(ts - dt) == Timedelta
        assert type(ts + td) == Timestamp
        assert type(ts - td) == Timestamp

        # Timestamp +/- datetime64 not supported, so not tested (could possibly
        # assert error raised?)
        td64 = np.timedelta64(1, "D")
        assert type(ts + td64) == Timestamp
        assert type(ts - td64) == Timestamp

    @pytest.mark.parametrize(
        "td", [Timedelta(hours=3), np.timedelta64(3, "h"), timedelta(hours=3)]
    )
    def test_radd_tdscalar(self, td, fixed_now_ts):
        # GH#24775 timedelta64+Timestamp should not raise
        ts = fixed_now_ts
        assert td + ts == ts + td

    @pytest.mark.parametrize(
        "other,expected_difference",
        [
            (np.timedelta64(-123, "ns"), -123),
            (np.timedelta64(1234567898, "ns"), 1234567898),
            (np.timedelta64(-123, "us"), -123000),
            (np.timedelta64(-123, "ms"), -123000000),
        ],
    )
    def test_timestamp_add_timedelta64_unit(self, other, expected_difference):
        now = datetime.now(timezone.utc)
        ts = Timestamp(now).as_unit("ns")
        result = ts + other
        valdiff = result._value - ts._value
        assert valdiff == expected_difference

        ts2 = Timestamp(now)
        assert ts2 + other == result

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("1776-07-04"),
            Timestamp("1776-07-04", tz="UTC"),
        ],
    )
    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.uint64),
        ],
    )
    def test_add_int_with_freq(self, ts, other):
        msg = "Addition/subtraction of integers and integer-arrays"
        with pytest.raises(TypeError, match=msg):
            ts + other
        with pytest.raises(TypeError, match=msg):
            other + ts

        with pytest.raises(TypeError, match=msg):
            ts - other

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            other - ts

    @pytest.mark.parametrize("shape", [(6,), (2, 3)])
    def test_addsub_m8ndarray(self, shape):
        # GH#33296
        ts = Timestamp("2020-04-04 15:45").as_unit("ns")
        other = np.arange(6).astype("m8[h]").reshape(shape)

        result = ts + other

        ex_stamps = [ts + Timedelta(hours=n) for n in range(6)]
        expected = np.array([x.asm8 for x in ex_stamps], dtype="M8[ns]").reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        result = other + ts
        tm.assert_numpy_array_equal(result, expected)

        result = ts - other
        ex_stamps = [ts - Timedelta(hours=n) for n in range(6)]
        expected = np.array([x.asm8 for x in ex_stamps], dtype="M8[ns]").reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            other - ts

    @pytest.mark.parametrize("shape", [(6,), (2, 3)])
    def test_addsub_m8ndarray_tzaware(self, shape):
        # GH#33296
        ts = Timestamp("2020-04-04 15:45", tz="US/Pacific")

        other = np.arange(6).astype("m8[h]").reshape(shape)

        result = ts + other

        ex_stamps = [ts + Timedelta(hours=n) for n in range(6)]
        expected = np.array(ex_stamps).reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        result = other + ts
        tm.assert_numpy_array_equal(result, expected)

        result = ts - other
        ex_stamps = [ts - Timedelta(hours=n) for n in range(6)]
        expected = np.array(ex_stamps).reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            other - ts

    def test_subtract_different_utc_objects(self, utc_fixture, utc_fixture2):
        # GH 32619
        dt = datetime(2021, 1, 1)
        ts1 = Timestamp(dt, tz=utc_fixture)
        ts2 = Timestamp(dt, tz=utc_fixture2)
        result = ts1 - ts2
        expected = Timedelta(0)
        assert result == expected

    @pytest.mark.parametrize(
        "tz",
        [
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_add_timedelta_push_over_dst_boundary(self, tz):
        # GH#1389

        # 4 hours before DST transition
        stamp = Timestamp("3/10/2012 22:00", tz=tz)

        result = stamp + timedelta(hours=6)

        # spring forward, + "7" hours
        expected = Timestamp("3/11/2012 05:00", tz=tz)

        assert result == expected


class SubDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "lh,rh",
    [
        (SubDatetime(2000, 1, 1), Timedelta(hours=1)),
        (Timedelta(hours=1), SubDatetime(2000, 1, 1)),
    ],
)
def test_dt_subclass_add_timedelta(lh, rh):
    # GH#25851
    # ensure that subclassed datetime works for
    # Timedelta operations
    result = lh + rh
    expected = SubDatetime(2000, 1, 1, 1)
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_indexing.py ===
# Test GroupBy._positional_selector positional grouped indexing GH#42864

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [0, [0, 1, 4]],
        [2, [5]],
        [5, []],
        [-1, [3, 4, 7]],
        [-2, [1, 6]],
        [-6, []],
    ],
)
def test_int(slice_test_df, slice_test_grouped, arg, expected_rows):
    # Test single integer
    result = slice_test_grouped._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_slice(slice_test_df, slice_test_grouped):
    # Test single slice
    result = slice_test_grouped._positional_selector[0:3:2]
    expected = slice_test_df.iloc[[0, 1, 4, 5]]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [[0, 2], [0, 1, 4, 5]],
        [[0, 2, -1], [0, 1, 3, 4, 5, 7]],
        [range(0, 3, 2), [0, 1, 4, 5]],
        [{0, 2}, [0, 1, 4, 5]],
    ],
    ids=[
        "list",
        "negative",
        "range",
        "set",
    ],
)
def test_list(slice_test_df, slice_test_grouped, arg, expected_rows):
    # Test lists of integers and integer valued iterables
    result = slice_test_grouped._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_ints(slice_test_df, slice_test_grouped):
    # Test tuple of ints
    result = slice_test_grouped._positional_selector[0, 2, -1]
    expected = slice_test_df.iloc[[0, 1, 3, 4, 5, 7]]

    tm.assert_frame_equal(result, expected)


def test_slices(slice_test_df, slice_test_grouped):
    # Test tuple of slices
    result = slice_test_grouped._positional_selector[:2, -2:]
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    tm.assert_frame_equal(result, expected)


def test_mix(slice_test_df, slice_test_grouped):
    # Test mixed tuple of ints and slices
    result = slice_test_grouped._positional_selector[0, 1, -2:]
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [0, [0, 1, 4]],
        [[0, 2, -1], [0, 1, 3, 4, 5, 7]],
        [(slice(None, 2), slice(-2, None)), [0, 1, 2, 3, 4, 6, 7]],
    ],
)
def test_as_index(slice_test_df, arg, expected_rows):
    # Test the default as_index behaviour
    result = slice_test_df.groupby("Group", sort=False)._positional_selector[arg]
    expected = slice_test_df.iloc[expected_rows]

    tm.assert_frame_equal(result, expected)


def test_doc_examples():
    # Test the examples in the documentation
    df = pd.DataFrame(
        [["a", 1], ["a", 2], ["a", 3], ["b", 4], ["b", 5]], columns=["A", "B"]
    )

    grouped = df.groupby("A", as_index=False)

    result = grouped._positional_selector[1:2]
    expected = pd.DataFrame([["a", 2], ["b", 5]], columns=["A", "B"], index=[1, 4])

    tm.assert_frame_equal(result, expected)

    result = grouped._positional_selector[1, -1]
    expected = pd.DataFrame(
        [["a", 2], ["a", 3], ["b", 5]], columns=["A", "B"], index=[1, 2, 4]
    )

    tm.assert_frame_equal(result, expected)


@pytest.fixture()
def multiindex_data():
    rng = np.random.default_rng(2)
    ndates = 100
    nitems = 20
    dates = pd.date_range("20130101", periods=ndates, freq="D")
    items = [f"item {i}" for i in range(nitems)]

    data = {}
    for date in dates:
        nitems_for_date = nitems - rng.integers(0, 12)
        levels = [
            (item, rng.integers(0, 10000) / 100, rng.integers(0, 10000) / 100)
            for item in items[:nitems_for_date]
        ]
        levels.sort(key=lambda x: x[1])
        data[date] = levels

    return data


def _make_df_from_data(data):
    rows = {}
    for date in data:
        for level in data[date]:
            rows[(date, level[0])] = {"A": level[1], "B": level[2]}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.names = ("Date", "Item")
    return df


def test_multiindex(multiindex_data):
    # Test the multiindex mentioned as the use-case in the documentation
    df = _make_df_from_data(multiindex_data)
    result = df.groupby("Date", as_index=False).nth(slice(3, -3))

    sliced = {date: multiindex_data[date][3:-3] for date in multiindex_data}
    expected = _make_df_from_data(sliced)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("arg", [1, 5, 30, 1000, -1, -5, -30, -1000])
@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("simulated", [True, False])
def test_against_head_and_tail(arg, method, simulated):
    # Test gives the same results as grouped head and tail
    n_groups = 100
    n_rows_per_group = 30

    data = {
        "group": [
            f"group {g}" for j in range(n_rows_per_group) for g in range(n_groups)
        ],
        "value": [
            f"group {g} row {j}"
            for j in range(n_rows_per_group)
            for g in range(n_groups)
        ],
    }
    df = pd.DataFrame(data)
    grouped = df.groupby("group", as_index=False)
    size = arg if arg >= 0 else n_rows_per_group + arg

    if method == "head":
        result = grouped._positional_selector[:arg]

        if simulated:
            indices = [
                j * n_groups + i
                for j in range(size)
                for i in range(n_groups)
                if j * n_groups + i < n_groups * n_rows_per_group
            ]
            expected = df.iloc[indices]

        else:
            expected = grouped.head(arg)

    else:
        result = grouped._positional_selector[-arg:]

        if simulated:
            indices = [
                (n_rows_per_group + j - size) * n_groups + i
                for j in range(size)
                for i in range(n_groups)
                if (n_rows_per_group + j - size) * n_groups + i >= 0
            ]
            expected = df.iloc[indices]

        else:
            expected = grouped.tail(arg)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("start", [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize("stop", [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize("step", [None, 1, 5])
def test_against_df_iloc(start, stop, step):
    # Test that a single group gives the same results as DataFrame.iloc
    n_rows = 30

    data = {
        "group": ["group 0"] * n_rows,
        "value": list(range(n_rows)),
    }
    df = pd.DataFrame(data)
    grouped = df.groupby("group", as_index=False)

    result = grouped._positional_selector[start:stop:step]
    expected = df.iloc[start:stop:step]

    tm.assert_frame_equal(result, expected)


def test_series():
    # Test grouped Series
    ser = pd.Series([1, 2, 3, 4, 5], index=["a", "a", "a", "b", "b"])
    grouped = ser.groupby(level=0)
    result = grouped._positional_selector[1:2]
    expected = pd.Series([2, 5], index=["a", "b"])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_step(step):
    # Test slice with various step values
    data = [["x", f"x{i}"] for i in range(5)]
    data += [["y", f"y{i}"] for i in range(4)]
    data += [["z", f"z{i}"] for i in range(3)]
    df = pd.DataFrame(data, columns=["A", "B"])

    grouped = df.groupby("A", as_index=False)

    result = grouped._positional_selector[::step]

    data = [["x", f"x{i}"] for i in range(0, 5, step)]
    data += [["y", f"y{i}"] for i in range(0, 4, step)]
    data += [["z", f"z{i}"] for i in range(0, 3, step)]

    index = [0 + i for i in range(0, 5, step)]
    index += [5 + i for i in range(0, 4, step)]
    index += [9 + i for i in range(0, 3, step)]

    expected = pd.DataFrame(data, columns=["A", "B"], index=index)

    tm.assert_frame_equal(result, expected)


@pytest.fixture()
def column_group_df():
    return pd.DataFrame(
        [[0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 0, 1, 0, 2]],
        columns=["A", "B", "C", "D", "E", "F", "G"],
    )


def test_column_axis(column_group_df):
    msg = "DataFrame.groupby with axis=1"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        g = column_group_df.groupby(column_group_df.iloc[1], axis=1)
    result = g._positional_selector[1:-1]
    expected = column_group_df.iloc[:, [1, 3]]

    tm.assert_frame_equal(result, expected)


def test_columns_on_iter():
    # GitHub issue #44821
    df = pd.DataFrame({k: range(10) for k in "ABC"})

    # Group-by and select columns
    cols = ["A", "B"]
    for _, dg in df.groupby(df.A < 4)[cols]:
        tm.assert_index_equal(dg.columns, pd.Index(cols))
        assert "C" not in dg.columns


@pytest.mark.parametrize("func", [list, pd.Index, pd.Series, np.array])
def test_groupby_duplicated_columns(func):
    # GH#44924
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 3],
            "C": ["G", "G"],
        }
    )
    result = df.groupby("C")[func(["A", "B", "A"])].mean()
    expected = pd.DataFrame(
        [[1.5, 3.0, 1.5]], columns=["A", "B", "A"], index=pd.Index(["G"], name="C")
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_get_nonexisting_groups():
    # GH#32492
    df = pd.DataFrame(
        data={
            "A": ["a1", "a2", None],
            "B": ["b1", "b2", "b1"],
            "val": [1, 2, 3],
        }
    )
    grps = df.groupby(by=["A", "B"])

    msg = "('a2', 'b1')"
    with pytest.raises(KeyError, match=msg):
        grps.get_group(("a2", "b1"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_linalg.py ===
# TODO: don't use round

from __future__ import division

import pytest
from mpmath import *
xrange = libmp.backend.xrange

# XXX: these shouldn't be visible(?)
LU_decomp = mp.LU_decomp
L_solve = mp.L_solve
U_solve = mp.U_solve
householder = mp.householder
improve_solution = mp.improve_solution

A1 = matrix([[3, 1, 6],
             [2, 1, 3],
             [1, 1, 1]])
b1 = [2, 7, 4]

A2 = matrix([[ 2, -1, -1,  2],
             [ 6, -2,  3, -1],
             [-4,  2,  3, -2],
             [ 2,  0,  4, -3]])
b2 = [3, -3, -2, -1]

A3 = matrix([[ 1,  0, -1, -1,  0],
             [ 0,  1,  1,  0, -1],
             [ 4, -5,  2,  0,  0],
             [ 0,  0, -2,  9,-12],
             [ 0,  5,  0,  0, 12]])
b3 = [0, 0, 0, 0, 50]

A4 = matrix([[10.235, -4.56,   0.,   -0.035,  5.67],
             [-2.463,  1.27,   3.97, -8.63,   1.08],
             [-6.58,   0.86,  -0.257, 9.32, -43.6 ],
             [ 9.83,   7.39, -17.25,  0.036, 24.86],
             [-9.31,  34.9,   78.56,  1.07,  65.8 ]])
b4 = [8.95, 20.54, 7.42, 5.60, 58.43]

A5 = matrix([[ 1,  2, -4],
             [-2, -3,  5],
             [ 3,  5, -8]])

A6 = matrix([[ 1.377360,  2.481400,   5.359190],
             [ 2.679280, -1.229560,  25.560210],
             [-1.225280+1.e6,  9.910180, -35.049900-1.e6]])
b6 = [23.500000, -15.760000, 2.340000]

A7 = matrix([[1, -0.5],
             [2, 1],
             [-2, 6]])
b7 = [3, 2, -4]

A8 = matrix([[1, 2, 3],
             [-1, 0, 1],
             [-1, -2, -1],
             [1, 0, -1]])
b8 = [1, 2, 3, 4]

A9 = matrix([[ 4,  2, -2],
             [ 2,  5, -4],
             [-2, -4, 5.5]])
b9 = [10, 16, -15.5]

A10 = matrix([[1.0 + 1.0j, 2.0, 2.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]])
b10 = [1.0, 1.0 + 1.0j, 1.0]


def test_LU_decomp():
    A = A3.copy()
    b = b3
    A, p = LU_decomp(A)
    y = L_solve(A, b, p)
    x = U_solve(A, y)
    assert p == [2, 1, 2, 3]
    assert [round(i, 14) for i in x] == [3.78953107960742, 2.9989094874591098,
            -0.081788440567070006, 3.8713195201744801, 2.9171210468920399]
    A = A4.copy()
    b = b4
    A, p = LU_decomp(A)
    y = L_solve(A, b, p)
    x = U_solve(A, y)
    assert p == [0, 3, 4, 3]
    assert [round(i, 14) for i in x] == [2.6383625899619201, 2.6643834462368399,
            0.79208015947958998, -2.5088376454101899, -1.0567657691375001]
    A = randmatrix(3)
    bak = A.copy()
    LU_decomp(A, overwrite=1)
    assert A != bak

def test_inverse():
    for A in [A1, A2, A5]:
        inv = inverse(A)
        assert mnorm(A*inv - eye(A.rows), 1) < 1.e-14

def test_householder():
    mp.dps = 15
    A, b = A8, b8
    H, p, x, r = householder(extend(A, b))
    assert H == matrix(
    [[mpf('3.0'), mpf('-2.0'), mpf('-1.0'), 0],
     [-1.0,mpf('3.333333333333333'),mpf('-2.9999999999999991'),mpf('2.0')],
     [-1.0, mpf('-0.66666666666666674'),mpf('2.8142135623730948'),
      mpf('-2.8284271247461898')],
     [1.0, mpf('-1.3333333333333333'),mpf('-0.20000000000000018'),
      mpf('4.2426406871192857')]])
    assert p == [-2, -2, mpf('-1.4142135623730949')]
    assert round(norm(r, 2), 10) == 4.2426406870999998

    y = [102.102, 58.344, 36.463, 24.310, 17.017, 12.376, 9.282, 7.140, 5.610,
         4.488, 3.6465, 3.003]

    def coeff(n):
        # similiar to Hilbert matrix
        A = []
        for i in range(1, 13):
            A.append([1. / (i + j - 1) for j in range(1, n + 1)])
        return matrix(A)

    residuals = []
    refres = []
    for n in range(2, 7):
        A = coeff(n)
        H, p, x, r = householder(extend(A, y))
        x = matrix(x)
        y = matrix(y)
        residuals.append(norm(r, 2))
        refres.append(norm(residual(A, x, y), 2))
    assert [round(res, 10) for res in residuals] == [15.1733888877,
           0.82378073210000002, 0.302645887, 0.0260109244,
           0.00058653999999999998]
    assert norm(matrix(residuals) - matrix(refres), inf) < 1.e-13

    def hilbert_cmplx(n):
        # Complexified  Hilbert matrix
        A = hilbert(2*n,n)
        v = randmatrix(2*n, 2, min=-1, max=1)
        v = v.apply(lambda x: exp(1J*pi()*x))
        A = diag(v[:,0])*A*diag(v[:n,1])
        return A

    residuals_cmplx = []
    refres_cmplx = []
    for n in range(2, 10):
        A = hilbert_cmplx(n)
        H, p, x, r = householder(A.copy())
        residuals_cmplx.append(norm(r, 2))
        refres_cmplx.append(norm(residual(A[:,:n-1], x, A[:,n-1]), 2))
    assert norm(matrix(residuals_cmplx) - matrix(refres_cmplx), inf) < 1.e-13

def test_factorization():
    A = randmatrix(5)
    P, L, U = lu(A)
    assert mnorm(P*A - L*U, 1) < 1.e-15

def test_solve():
    assert norm(residual(A6, lu_solve(A6, b6), b6), inf) < 1.e-10
    assert norm(residual(A7, lu_solve(A7, b7), b7), inf) < 1.5
    assert norm(residual(A8, lu_solve(A8, b8), b8), inf) <= 3 + 1.e-10
    assert norm(residual(A6, qr_solve(A6, b6)[0], b6), inf) < 1.e-10
    assert norm(residual(A7, qr_solve(A7, b7)[0], b7), inf) < 1.5
    assert norm(residual(A8, qr_solve(A8, b8)[0], b8), 2) <= 4.3
    assert norm(residual(A10, lu_solve(A10, b10), b10), 2) < 1.e-10
    assert norm(residual(A10, qr_solve(A10, b10)[0], b10), 2) < 1.e-10

def test_solve_overdet_complex():
    A = matrix([[1, 2j], [3, 4j], [5, 6]])
    b = matrix([1 + j, 2, -j])
    assert norm(residual(A, lu_solve(A, b), b)) < 1.0208

def test_singular():
    mp.dps = 15
    A = [[5.6, 1.2], [7./15, .1]]
    B = repr(zeros(2))
    b = [1, 2]
    for i in ['lu_solve(%s, %s)' % (A, b), 'lu_solve(%s, %s)' % (B, b),
              'qr_solve(%s, %s)' % (A, b), 'qr_solve(%s, %s)' % (B, b)]:
        pytest.raises((ZeroDivisionError, ValueError), lambda: eval(i))

def test_cholesky():
    assert fp.cholesky(fp.matrix(A9)) == fp.matrix([[2, 0, 0], [1, 2, 0], [-1, -3/2, 3/2]])
    x = fp.cholesky_solve(A9, b9)
    assert fp.norm(fp.residual(A9, x, b9), fp.inf) == 0

def test_det():
    assert det(A1) == 1
    assert round(det(A2), 14) == 8
    assert round(det(A3)) == 1834
    assert round(det(A4)) == 4443376
    assert det(A5) == 1
    assert round(det(A6)) == 78356463
    assert det(zeros(3)) == 0

def test_cond():
    mp.dps = 15
    A = matrix([[1.2969, 0.8648], [0.2161, 0.1441]])
    assert cond(A, lambda x: mnorm(x,1)) == mpf('327065209.73817754')
    assert cond(A, lambda x: mnorm(x,inf)) == mpf('327065209.73817754')
    assert cond(A, lambda x: mnorm(x,'F')) == mpf('249729266.80008656')

@extradps(50)
def test_precision():
    A = randmatrix(10, 10)
    assert mnorm(inverse(inverse(A)) - A, 1) < 1.e-45

def test_interval_matrix():
    mp.dps = 15
    iv.dps = 15
    a = iv.matrix([['0.1','0.3','1.0'],['7.1','5.5','4.8'],['3.2','4.4','5.6']])
    b = iv.matrix(['4','0.6','0.5'])
    c = iv.lu_solve(a, b)
    assert c[0].delta < 1e-13
    assert c[1].delta < 1e-13
    assert c[2].delta < 1e-13
    assert 5.25823271130625686059275 in c[0]
    assert -13.155049396267837541163 in c[1]
    assert 7.42069154774972557628979 in c[2]

def test_LU_cache():
    A = randmatrix(3)
    LU = LU_decomp(A)
    assert A._LU == LU_decomp(A)
    A[0,0] = -1000
    assert A._LU is None

def test_improve_solution():
    A = randmatrix(5, min=1e-20, max=1e20)
    b = randmatrix(5, 1, min=-1000, max=1000)
    x1 = lu_solve(A, b) + randmatrix(5, 1, min=-1e-5, max=1.e-5)
    x2 = improve_solution(A, x1, b)
    assert norm(residual(A, x2, b), 2) < norm(residual(A, x1, b), 2)

def test_exp_pade():
    for i in range(3):
        dps = 15
        extra = 15
        mp.dps = dps + extra
        dm = 0
        N = 3
        dg = range(1,N+1)
        a = diag(dg)
        expa = diag([exp(x) for x in dg])
        # choose a random matrix not close to be singular
        # to avoid adding too much extra precision in computing
        # m**-1 * M * m
        while abs(dm) < 0.01:
            m = randmatrix(N)
            dm = det(m)
        m = m/dm
        a1 = m**-1 * a * m
        e2 = m**-1 * expa * m
        mp.dps = dps
        e1 = expm(a1, method='pade')
        mp.dps = dps + extra
        d = e2 - e1
        #print d
        mp.dps = dps
        assert norm(d, inf).ae(0)
    mp.dps = 15

def test_qr():
    mp.dps = 15                     # used default value for dps
    lowlimit = -9                   # lower limit of matrix element value
    uplimit = 9                     # uppter limit of matrix element value
    maxm = 4                        # max matrix size
    flg = False                     # toggle to create real vs complex matrix
    zero = mpf('0.0')

    for k in xrange(0,10):
        exdps = 0
        mode = 'full'
        flg = bool(k % 2)

        # generate arbitrary matrix size (2 to maxm)
        num1 = nint(maxm*rand())
        num2 = nint(maxm*rand())
        m = int(max(num1, num2))
        n = int(min(num1, num2))

        # create matrix
        A = mp.matrix(m,n)

        # populate matrix values with arbitrary integers
        if flg:
            flg = False
            dtype = 'complex'
            for j in xrange(0,n):
                for i in xrange(0,m):
                    val = nint(lowlimit + (uplimit-lowlimit)*rand())
                    val2 = nint(lowlimit + (uplimit-lowlimit)*rand())
                    A[i,j] = mpc(val, val2)
        else:
            flg = True
            dtype = 'real'
            for j in xrange(0,n):
                for i in xrange(0,m):
                    val = nint(lowlimit + (uplimit-lowlimit)*rand())
                    A[i,j] = mpf(val)

        # perform A -> QR decomposition
        Q, R = qr(A, mode, edps = exdps)

        #print('\n\n A = \n', nstr(A, 4))
        #print('\n Q = \n', nstr(Q, 4))
        #print('\n R = \n', nstr(R, 4))
        #print('\n Q*R = \n', nstr(Q*R, 4))

        maxnorm = mpf('1.0E-11')
        n1 = norm(A - Q * R)
        #print '\n Norm of A - Q * R = ', n1
        assert n1 <= maxnorm

        if dtype == 'real':
            n1 = norm(eye(m) - Q.T * Q)
            #print ' Norm of I - Q.T * Q = ', n1
            assert n1 <= maxnorm

            n1 = norm(eye(m) - Q * Q.T)
            #print ' Norm of I - Q * Q.T = ', n1
            assert n1 <= maxnorm

        if dtype == 'complex':
            n1 = norm(eye(m) - Q.T * Q.conjugate())
            #print ' Norm of I - Q.T * Q.conjugate() = ', n1
            assert n1 <= maxnorm

            n1 = norm(eye(m) - Q.conjugate() * Q.T)
            #print ' Norm of I - Q.conjugate() * Q.T = ', n1
            assert n1 <= maxnorm

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\holiday\test_holiday.py ===
from datetime import datetime

import pytest
from pytz import utc

from pandas import (
    DatetimeIndex,
    Series,
)
import pandas._testing as tm

from pandas.tseries.holiday import (
    MO,
    SA,
    AbstractHolidayCalendar,
    DateOffset,
    EasterMonday,
    GoodFriday,
    Holiday,
    HolidayCalendarFactory,
    Timestamp,
    USColumbusDay,
    USFederalHolidayCalendar,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    get_calendar,
    next_monday,
)


@pytest.mark.parametrize(
    "holiday,start_date,end_date,expected",
    [
        (
            USMemorialDay,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                datetime(2011, 5, 30),
                datetime(2012, 5, 28),
                datetime(2013, 5, 27),
                datetime(2014, 5, 26),
                datetime(2015, 5, 25),
                datetime(2016, 5, 30),
                datetime(2017, 5, 29),
                datetime(2018, 5, 28),
                datetime(2019, 5, 27),
                datetime(2020, 5, 25),
            ],
        ),
        (
            Holiday("July 4th Eve", month=7, day=3),
            "2001-01-01",
            "2003-03-03",
            [Timestamp("2001-07-03 00:00:00"), Timestamp("2002-07-03 00:00:00")],
        ),
        (
            Holiday("July 4th Eve", month=7, day=3, days_of_week=(0, 1, 2, 3)),
            "2001-01-01",
            "2008-03-03",
            [
                Timestamp("2001-07-03 00:00:00"),
                Timestamp("2002-07-03 00:00:00"),
                Timestamp("2003-07-03 00:00:00"),
                Timestamp("2006-07-03 00:00:00"),
                Timestamp("2007-07-03 00:00:00"),
            ],
        ),
        (
            EasterMonday,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                Timestamp("2011-04-25 00:00:00"),
                Timestamp("2012-04-09 00:00:00"),
                Timestamp("2013-04-01 00:00:00"),
                Timestamp("2014-04-21 00:00:00"),
                Timestamp("2015-04-06 00:00:00"),
                Timestamp("2016-03-28 00:00:00"),
                Timestamp("2017-04-17 00:00:00"),
                Timestamp("2018-04-02 00:00:00"),
                Timestamp("2019-04-22 00:00:00"),
                Timestamp("2020-04-13 00:00:00"),
            ],
        ),
        (
            GoodFriday,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                Timestamp("2011-04-22 00:00:00"),
                Timestamp("2012-04-06 00:00:00"),
                Timestamp("2013-03-29 00:00:00"),
                Timestamp("2014-04-18 00:00:00"),
                Timestamp("2015-04-03 00:00:00"),
                Timestamp("2016-03-25 00:00:00"),
                Timestamp("2017-04-14 00:00:00"),
                Timestamp("2018-03-30 00:00:00"),
                Timestamp("2019-04-19 00:00:00"),
                Timestamp("2020-04-10 00:00:00"),
            ],
        ),
        (
            USThanksgivingDay,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                datetime(2011, 11, 24),
                datetime(2012, 11, 22),
                datetime(2013, 11, 28),
                datetime(2014, 11, 27),
                datetime(2015, 11, 26),
                datetime(2016, 11, 24),
                datetime(2017, 11, 23),
                datetime(2018, 11, 22),
                datetime(2019, 11, 28),
                datetime(2020, 11, 26),
            ],
        ),
    ],
)
def test_holiday_dates(holiday, start_date, end_date, expected):
    assert list(holiday.dates(start_date, end_date)) == expected

    # Verify that timezone info is preserved.
    assert list(
        holiday.dates(
            utc.localize(Timestamp(start_date)), utc.localize(Timestamp(end_date))
        )
    ) == [utc.localize(dt) for dt in expected]


@pytest.mark.parametrize(
    "holiday,start,expected",
    [
        (USMemorialDay, datetime(2015, 7, 1), []),
        (USMemorialDay, "2015-05-25", [Timestamp("2015-05-25")]),
        (USLaborDay, datetime(2015, 7, 1), []),
        (USLaborDay, "2015-09-07", [Timestamp("2015-09-07")]),
        (USColumbusDay, datetime(2015, 7, 1), []),
        (USColumbusDay, "2015-10-12", [Timestamp("2015-10-12")]),
        (USThanksgivingDay, datetime(2015, 7, 1), []),
        (USThanksgivingDay, "2015-11-26", [Timestamp("2015-11-26")]),
        (USMartinLutherKingJr, datetime(2015, 7, 1), []),
        (USMartinLutherKingJr, "2015-01-19", [Timestamp("2015-01-19")]),
        (USPresidentsDay, datetime(2015, 7, 1), []),
        (USPresidentsDay, "2015-02-16", [Timestamp("2015-02-16")]),
        (GoodFriday, datetime(2015, 7, 1), []),
        (GoodFriday, "2015-04-03", [Timestamp("2015-04-03")]),
        (EasterMonday, "2015-04-06", [Timestamp("2015-04-06")]),
        (EasterMonday, datetime(2015, 7, 1), []),
        (EasterMonday, "2015-04-05", []),
        ("New Year's Day", "2015-01-01", [Timestamp("2015-01-01")]),
        ("New Year's Day", "2010-12-31", [Timestamp("2010-12-31")]),
        ("New Year's Day", datetime(2015, 7, 1), []),
        ("New Year's Day", "2011-01-01", []),
        ("Independence Day", "2015-07-03", [Timestamp("2015-07-03")]),
        ("Independence Day", datetime(2015, 7, 1), []),
        ("Independence Day", "2015-07-04", []),
        ("Veterans Day", "2012-11-12", [Timestamp("2012-11-12")]),
        ("Veterans Day", datetime(2015, 7, 1), []),
        ("Veterans Day", "2012-11-11", []),
        ("Christmas Day", "2011-12-26", [Timestamp("2011-12-26")]),
        ("Christmas Day", datetime(2015, 7, 1), []),
        ("Christmas Day", "2011-12-25", []),
        ("Juneteenth National Independence Day", "2020-06-19", []),
        (
            "Juneteenth National Independence Day",
            "2021-06-18",
            [Timestamp("2021-06-18")],
        ),
        ("Juneteenth National Independence Day", "2022-06-19", []),
        (
            "Juneteenth National Independence Day",
            "2022-06-20",
            [Timestamp("2022-06-20")],
        ),
    ],
)
def test_holidays_within_dates(holiday, start, expected):
    # see gh-11477
    #
    # Fix holiday behavior where holiday.dates returned dates outside
    # start/end date, or observed rules could not be applied because the
    # holiday was not in the original date range (e.g., 7/4/2015 -> 7/3/2015).
    if isinstance(holiday, str):
        calendar = get_calendar("USFederalHolidayCalendar")
        holiday = calendar.rule_from_name(holiday)

    assert list(holiday.dates(start, start)) == expected

    # Verify that timezone info is preserved.
    assert list(
        holiday.dates(utc.localize(Timestamp(start)), utc.localize(Timestamp(start)))
    ) == [utc.localize(dt) for dt in expected]


@pytest.mark.parametrize(
    "transform", [lambda x: x.strftime("%Y-%m-%d"), lambda x: Timestamp(x)]
)
def test_argument_types(transform):
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)

    holidays = USThanksgivingDay.dates(start_date, end_date)
    holidays2 = USThanksgivingDay.dates(transform(start_date), transform(end_date))
    tm.assert_index_equal(holidays, holidays2)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("One-Time", {"year": 2012, "month": 5, "day": 28}),
        (
            "Range",
            {
                "month": 5,
                "day": 28,
                "start_date": datetime(2012, 1, 1),
                "end_date": datetime(2012, 12, 31),
                "offset": DateOffset(weekday=MO(1)),
            },
        ),
    ],
)
def test_special_holidays(name, kwargs):
    base_date = [datetime(2012, 5, 28)]
    holiday = Holiday(name, **kwargs)

    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)

    assert base_date == holiday.dates(start_date, end_date)


def test_get_calendar():
    class TestCalendar(AbstractHolidayCalendar):
        rules = []

    calendar = get_calendar("TestCalendar")
    assert TestCalendar == type(calendar)


def test_factory():
    class_1 = HolidayCalendarFactory(
        "MemorialDay", AbstractHolidayCalendar, USMemorialDay
    )
    class_2 = HolidayCalendarFactory(
        "Thanksgiving", AbstractHolidayCalendar, USThanksgivingDay
    )
    class_3 = HolidayCalendarFactory("Combined", class_1, class_2)

    assert len(class_1.rules) == 1
    assert len(class_2.rules) == 1
    assert len(class_3.rules) == 2


def test_both_offset_observance_raises():
    # see gh-10217
    msg = "Cannot use both offset and observance"
    with pytest.raises(NotImplementedError, match=msg):
        Holiday(
            "Cyber Monday",
            month=11,
            day=1,
            offset=[DateOffset(weekday=SA(4))],
            observance=next_monday,
        )


def test_half_open_interval_with_observance():
    # Prompted by GH 49075
    # Check for holidays that have a half-open date interval where
    # they have either a start_date or end_date defined along
    # with a defined observance pattern to make sure that the return type
    # for Holiday.dates() remains consistent before & after the year that
    # marks the 'edge' of the half-open date interval.

    holiday_1 = Holiday(
        "Arbitrary Holiday - start 2022-03-14",
        start_date=datetime(2022, 3, 14),
        month=3,
        day=14,
        observance=next_monday,
    )
    holiday_2 = Holiday(
        "Arbitrary Holiday 2 - end 2022-03-20",
        end_date=datetime(2022, 3, 20),
        month=3,
        day=20,
        observance=next_monday,
    )

    class TestHolidayCalendar(AbstractHolidayCalendar):
        rules = [
            USMartinLutherKingJr,
            holiday_1,
            holiday_2,
            USLaborDay,
        ]

    start = Timestamp("2022-08-01")
    end = Timestamp("2022-08-31")
    year_offset = DateOffset(years=5)
    expected_results = DatetimeIndex([], dtype="datetime64[ns]", freq=None)
    test_cal = TestHolidayCalendar()

    date_interval_low = test_cal.holidays(start - year_offset, end - year_offset)
    date_window_edge = test_cal.holidays(start, end)
    date_interval_high = test_cal.holidays(start + year_offset, end + year_offset)

    tm.assert_index_equal(date_interval_low, expected_results)
    tm.assert_index_equal(date_window_edge, expected_results)
    tm.assert_index_equal(date_interval_high, expected_results)


def test_holidays_with_timezone_specified_but_no_occurences():
    # GH 54580
    # _apply_rule() in holiday.py was silently dropping timezones if you passed it
    # an empty list of holiday dates that had timezone information
    start_date = Timestamp("2018-01-01", tz="America/Chicago")
    end_date = Timestamp("2018-01-11", tz="America/Chicago")
    test_case = USFederalHolidayCalendar().holidays(
        start_date, end_date, return_name=True
    )
    expected_results = Series("New Year's Day", index=[start_date])
    expected_results.index = expected_results.index.as_unit("ns")

    tm.assert_equal(test_case, expected_results)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\control\tests\test_control_plots.py ===
from math import isclose
from sympy.core.numbers import I, all_close
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import (Abs, arg)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.abc import s, p, a
from sympy import pi
from sympy.external import import_module
from sympy.physics.control.control_plots import \
    (pole_zero_numerical_data, pole_zero_plot, step_response_numerical_data,
    step_response_plot, impulse_response_numerical_data,
    impulse_response_plot, ramp_response_numerical_data,
    ramp_response_plot, bode_magnitude_numerical_data,
    bode_phase_numerical_data, bode_plot, nyquist_plot_expr,
    nichols_plot_expr)

from sympy.physics.control.lti import (TransferFunction,
    Series, Parallel, TransferFunctionMatrix)
from sympy.testing.pytest import raises, skip

matplotlib = import_module(
        'matplotlib', import_kwargs={'fromlist': ['pyplot']},
        catch=(RuntimeError,))

numpy = import_module('numpy')

tf1 = TransferFunction(1, p**2 + 0.5*p + 2, p)
tf2 = TransferFunction(p, 6*p**2 + 3*p + 1, p)
tf3 = TransferFunction(p, p**3 - 1, p)
tf4 = TransferFunction(10, p**3, p)
tf5 = TransferFunction(5, s**2 + 2*s + 10, s)
tf6 = TransferFunction(1, 1, s)
tf7 = TransferFunction(4*s*3 + 9*s**2 + 0.1*s + 11, 8*s**6 + 9*s**4 + 11, s)
tf8 = TransferFunction(5, s**2 + (2+I)*s + 10, s)

ser1 = Series(tf4, TransferFunction(1, p - 5, p))
ser2 = Series(tf3, TransferFunction(p, p + 2, p))

par1 = Parallel(tf1, tf2)


def _to_tuple(a, b):
    return tuple(a), tuple(b)

def _trim_tuple(a, b):
    a, b = _to_tuple(a, b)
    return tuple(a[0: 2] + a[len(a)//2 : len(a)//2 + 1] + a[-2:]), \
        tuple(b[0: 2] + b[len(b)//2 : len(b)//2 + 1] + b[-2:])

def y_coordinate_equality(plot_data_func, evalf_func, system):
    """Checks whether the y-coordinate value of the plotted
    data point is equal to the value of the function at a
    particular x."""
    x, y = plot_data_func(system)
    x, y = _trim_tuple(x, y)
    y_exp = tuple(evalf_func(system, x_i) for x_i in x)
    return all(Abs(y_exp_i - y_i) < 1e-8 for y_exp_i, y_i in zip(y_exp, y))


def test_errors():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # Invalid `system` check
    tfm = TransferFunctionMatrix([[tf6, tf5], [tf5, tf6]])
    expr = 1/(s**2 - 1)
    raises(NotImplementedError, lambda: pole_zero_plot(tfm))
    raises(NotImplementedError, lambda: pole_zero_numerical_data(expr))
    raises(NotImplementedError, lambda: impulse_response_plot(expr))
    raises(NotImplementedError, lambda: impulse_response_numerical_data(tfm))
    raises(NotImplementedError, lambda: step_response_plot(tfm))
    raises(NotImplementedError, lambda: step_response_numerical_data(expr))
    raises(NotImplementedError, lambda: ramp_response_plot(expr))
    raises(NotImplementedError, lambda: ramp_response_numerical_data(tfm))
    raises(NotImplementedError, lambda: bode_plot(tfm))

    # More than 1 variables
    tf_a = TransferFunction(a, s + 1, s)
    raises(ValueError, lambda: pole_zero_plot(tf_a))
    raises(ValueError, lambda: pole_zero_numerical_data(tf_a))
    raises(ValueError, lambda: impulse_response_plot(tf_a))
    raises(ValueError, lambda: impulse_response_numerical_data(tf_a))
    raises(ValueError, lambda: step_response_plot(tf_a))
    raises(ValueError, lambda: step_response_numerical_data(tf_a))
    raises(ValueError, lambda: ramp_response_plot(tf_a))
    raises(ValueError, lambda: ramp_response_numerical_data(tf_a))
    raises(ValueError, lambda: bode_plot(tf_a))

    # lower_limit > 0 for response plots
    raises(ValueError, lambda: impulse_response_plot(tf1, lower_limit=-1))
    raises(ValueError, lambda: step_response_plot(tf1, lower_limit=-0.1))
    raises(ValueError, lambda: ramp_response_plot(tf1, lower_limit=-4/3))

    # slope in ramp_response_plot() is negative
    raises(ValueError, lambda: ramp_response_plot(tf1, slope=-0.1))

    # incorrect frequency or phase unit
    raises(ValueError, lambda: bode_plot(tf1,freq_unit = 'hz'))
    raises(ValueError, lambda: bode_plot(tf1,phase_unit = 'degree'))


def test_pole_zero():

    def pz_tester(sys, expected_value):
        _z, _p = pole_zero_numerical_data(sys)
        z_check = all_close(_z, expected_value[0])
        p_check = all_close(_p, expected_value[1])
        return p_check and z_check

    exp1 = [[], [-0.24999999999999994-1.3919410907075054j, -0.24999999999999994+1.3919410907075054j]]
    exp2 = [[0.0], [-0.25-0.3227486121839514j, -0.25+0.3227486121839514j]]
    exp3 = [[0.0], [0.9999999999999998+0j, -0.5000000000000004-0.8660254037844395j,
        -0.5000000000000004+0.8660254037844395j]]
    exp4 = [[], [0.0, 0.0, 0.0, 5.0]]
    exp5 = [[-5.645751311064592, -0.5000000000000008, -0.3542486889354093],
        [-0.24999999999999986-0.322748612183951348j,
        -0.2499999999999998+0.32274861218395134j,
        -0.24999999999999986-1.3919410907075052j,
         -0.2499999999999998+1.3919410907075052j]]
    exp6 = [[], [-1.1641600331447917-3.545808351896439j,
          -0.8358399668552097+2.5458083518964383j]]

    assert pz_tester(tf1, exp1)
    assert pz_tester(tf2, exp2)
    assert pz_tester(tf3, exp3)
    assert pz_tester(ser1, exp4)
    assert pz_tester(par1, exp5)
    assert pz_tester(tf8, exp6)


def test_bode():
    if not numpy:
        skip("NumPy is required for this test")

    def bode_phase_evalf(system, point):
        expr = system.to_expr()
        _w = Dummy("w", real=True)
        w_expr = expr.subs({system.var: I*_w})
        return arg(w_expr).subs({_w: point}).evalf()

    def bode_mag_evalf(system, point):
        expr = system.to_expr()
        _w = Dummy("w", real=True)
        w_expr = expr.subs({system.var: I*_w})
        return 20*log(Abs(w_expr), 10).subs({_w: point}).evalf()

    def test_bode_data(sys):
        return y_coordinate_equality(bode_magnitude_numerical_data, bode_mag_evalf, sys) \
            and y_coordinate_equality(bode_phase_numerical_data, bode_phase_evalf, sys)

    assert test_bode_data(tf1)
    assert test_bode_data(tf2)
    assert test_bode_data(tf3)
    assert test_bode_data(tf4)
    assert test_bode_data(tf5)


def check_point_accuracy(a, b):
    return all(isclose(*_, rel_tol=1e-1, abs_tol=1e-6
        ) for _ in zip(a, b))


def test_impulse_response():
    if not numpy:
        skip("NumPy is required for this test")

    def impulse_res_tester(sys, expected_value):
        x, y = _to_tuple(*impulse_response_numerical_data(sys,
            adaptive=False, n=10))
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.544019738507865, 0.01993849743234938, -0.31140243360893216, -0.022852779906491996, 0.1778306498155759,
        0.01962941084328499, -0.1013115194573652, -0.014975541213105696, 0.0575789724730714))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.1666666675, 0.08389223412935855,
        0.02338051973475047, -0.014966807776379383, -0.034645954223054234, -0.040560075735512804,
        -0.037658628907103885, -0.030149507719590022, -0.021162090730736834, -0.012721292737437523))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (4.369893391586999e-09, 1.1750333000630964,
        3.2922404058312473, 9.432290008148343, 28.37098083007151, 86.18577464367974, 261.90356653762115,
        795.6538758627842, 2416.9920942096983, 7342.159505206647))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 6.17283950617284, 24.69135802469136,
        55.555555555555564, 98.76543209876544, 154.320987654321, 222.22222222222226, 302.46913580246917,
        395.0617283950618, 500.0))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, -0.10455606138085417,
        0.06757671513476461, -0.03234567568833768, 0.013582514927757873, -0.005273419510705473,
        0.0019364083003354075, -0.000680070134067832, 0.00022969845960406913, -7.476094359583917e-05))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-6.016699583000218e-09, 0.35039802056107394, 3.3728423827689884, 12.119846079276684,
        25.86101014293389, 29.352480635282088, -30.49475907497664, -273.8717189554019, -863.2381702029659,
        -1747.0262164682233))
    exp7 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335,
        4.444444444444445, 5.555555555555555, 6.666666666666667, 7.777777777777779,
        8.88888888888889, 10.0), (0.0, 18.934638095560974, 5346.93244680907, 1384609.8718249386,
        358161126.65801865, 92645770015.70108, 23964739753087.42, 6198974342083139.0, 1.603492601616059e+18,
        4.147764422869658e+20))

    assert impulse_res_tester(tf1, exp1)
    assert impulse_res_tester(tf2, exp2)
    assert impulse_res_tester(tf3, exp3)
    assert impulse_res_tester(tf4, exp4)
    assert impulse_res_tester(tf5, exp5)
    assert impulse_res_tester(tf7, exp6)
    assert impulse_res_tester(ser1, exp7)


def test_step_response():
    if not numpy:
        skip("NumPy is required for this test")

    def step_res_tester(sys, expected_value):
        x, y = _to_tuple(*step_response_numerical_data(sys,
            adaptive=False, n=10))
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-1.9193285738516863e-08, 0.42283495488246126, 0.7840485977945262, 0.5546841805655717,
        0.33903033806932087, 0.4627251747410237, 0.5909907598988051, 0.5247213989553071,
        0.4486997874319281, 0.4839358435839171))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.13728409095645816, 0.19474559355325086, 0.1974909129243011, 0.16841657696573073,
        0.12559777736159378, 0.08153828016664713, 0.04360471317348958, 0.015072994568868221,
        -0.003636420058445484))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.6314542141914303, 2.9356520038101035, 9.37731009663807, 28.452300356688376,
        86.25721933273988, 261.9236645044672, 795.6435410577224, 2416.9786984578764, 7342.154119725917))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 2.286236899862826, 18.28989519890261, 61.72839629629631, 146.31916159122088, 285.7796124828532,
        493.8271703703705, 784.1792566529494, 1170.553292729767, 1666.6667))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-3.999999997894577e-09, 0.6720357068882895, 0.4429938256137113, 0.5182010838004518,
        0.4944139147159695, 0.5016379853883338, 0.4995466896527733, 0.5001154784851325,
        0.49997448824584123, 0.5000039745919259))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-1.5433688493882158e-09, 0.3428705539937336, 1.1253619102202777, 3.1849962651016517,
        9.47532757182671, 28.727231099148135, 87.29426924860557, 265.2138681048606, 805.6636260007757,
        2447.387582370878))

    assert step_res_tester(tf1, exp1)
    assert step_res_tester(tf2, exp2)
    assert step_res_tester(tf3, exp3)
    assert step_res_tester(tf4, exp4)
    assert step_res_tester(tf5, exp5)
    assert step_res_tester(ser2, exp6)


def test_ramp_response():
    if not numpy:
        skip("NumPy is required for this test")

    def ramp_res_tester(sys, num_points, expected_value, slope=1):
        x, y = _to_tuple(*ramp_response_numerical_data(sys,
            slope=slope, adaptive=False, n=num_points))
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 2.0, 4.0, 6.0, 8.0, 10.0), (0.0, 0.7324667795033895, 1.9909720978650398,
        2.7956587704217783, 3.9224897567931514, 4.85022655284895))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (2.4360213402019326e-08, 0.10175320182493253, 0.33057612497658406, 0.5967937263298935,
        0.8431511866718248, 1.0398805391471613, 1.1776043125035738, 1.2600994825747305, 1.2981042689274653,
        1.304684417610106))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (-3.9329040468771836e-08,
        0.34686634635794555, 2.9998828170537903, 12.33303690737476, 40.993913948137795, 127.84145222317912,
        391.41713691996, 1192.0006858708389, 3623.9808672503405, 11011.728034546572))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 1.9051973784484078, 30.483158055174524,
        154.32098765432104, 487.7305288827924, 1190.7483615302544, 2469.1358024691367, 4574.3789056546275,
        7803.688462124678, 12500.0))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 3.8844361856975635, 9.141792069209865,
        14.096349157657231, 19.09783068994694, 24.10179770390321, 29.09907319114121, 34.10040420185154,
        39.09983919254265, 44.10006013058409))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 1.1111111111111112, 2.2222222222222223,
        3.3333333333333335, 4.444444444444445, 5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0))

    assert ramp_res_tester(tf1, 6, exp1)
    assert ramp_res_tester(tf2, 10, exp2, 1.2)
    assert ramp_res_tester(tf3, 10, exp3, 1.5)
    assert ramp_res_tester(tf4, 10, exp4, 3)
    assert ramp_res_tester(tf5, 10, exp5, 9)
    assert ramp_res_tester(tf6, 10, exp6)


def test_nyquist_plot_expr():
    r1, i1, w1 = nyquist_plot_expr(tf1)
    r2, i2, w2 = nyquist_plot_expr(tf2)
    r3, i3, w3 = nyquist_plot_expr(tf3)
    r4, i4, w4 = nyquist_plot_expr(tf4)
    assert r1 == (2 - w1**2)/(0.25*w1**2 + (2 - w1**2)**2)
    assert i1 == -0.5*w1/(0.25*w1**2 + (2 - w1**2)**2)
    assert r2 == 3*w2**2/(9*w2**2 + (1 - 6*w2**2)**2)
    assert i2 == w2*(1 - 6*w2**2)/(9*w2**2 + (1 - 6*w2**2)**2)
    assert r3 == -w3**4/(w3**6 + 1)
    assert i3 == -w3/(w3**6 + 1)
    assert r4 == 0
    assert i4 == 10/w4**3


def test_nichols_expr():
    m1, p1, w1 = nichols_plot_expr(tf1)
    m2, p2, w2 = nichols_plot_expr(tf2)
    m3, p3, w3 = nichols_plot_expr(tf3)
    m4, p4, w4 = nichols_plot_expr(tf4)
    assert m1 == 20*log(1/sqrt(w1**4 - 3.75*w1**2 + 4))/log(10)
    assert p1 == 180*arg(1/(-w1**2 + 0.5*w1*I + 2))/pi
    assert m2 == 20*log(Abs(w2)/sqrt(36*w2**4 - 3*w2**2 + 1))/log(10)
    assert p2 == 180*arg(w2*I/(-6*w2**2 + 3*w2*I + 1))/pi
    assert m3 == 20*log(Abs(w3)/sqrt(w3**6 + 1))/log(10)
    assert p3 == 180*arg(-w3*I/(w3**3*I + 1))/pi
    assert m4 == 20*log(10/(w4**2*Abs(w4)))/log(10)
    assert p4 == 180*arg(I/w4**3)/pi

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_libgroupby.py ===
import numpy as np
import pytest

from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
    group_cumprod,
    group_cumsum,
    group_mean,
    group_sum,
    group_var,
)

from pandas.core.dtypes.common import ensure_platform_int

from pandas import isna
import pandas._testing as tm


class GroupVarTestMixin:
    def test_group_var_generic_1d(self):
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((5, 1))).astype(self.dtype)
        counts = np.zeros(5, dtype="int64")
        values = 10 * prng.random((15, 1)).astype(self.dtype)
        labels = np.tile(np.arange(5), (3,)).astype("intp")

        expected_out = (
            np.squeeze(values).reshape((5, 3), order="F").std(axis=1, ddof=1) ** 2
        )[:, np.newaxis]
        expected_counts = counts + 3

        self.algo(out, counts, values, labels)
        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_1d_flat_labels(self):
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((1, 1))).astype(self.dtype)
        counts = np.zeros(1, dtype="int64")
        values = 10 * prng.random((5, 1)).astype(self.dtype)
        labels = np.zeros(5, dtype="intp")

        expected_out = np.array([[values.std(ddof=1) ** 2]])
        expected_counts = counts + 5

        self.algo(out, counts, values, labels)

        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_all_finite(self):
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((5, 2))).astype(self.dtype)
        counts = np.zeros(5, dtype="int64")
        values = 10 * prng.random((10, 2)).astype(self.dtype)
        labels = np.tile(np.arange(5), (2,)).astype("intp")

        expected_out = np.std(values.reshape(2, 5, 2), ddof=1, axis=0) ** 2
        expected_counts = counts + 2

        self.algo(out, counts, values, labels)
        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_some_nan(self):
        prng = np.random.default_rng(2)

        out = (np.nan * np.ones((5, 2))).astype(self.dtype)
        counts = np.zeros(5, dtype="int64")
        values = 10 * prng.random((10, 2)).astype(self.dtype)
        values[:, 1] = np.nan
        labels = np.tile(np.arange(5), (2,)).astype("intp")

        expected_out = np.vstack(
            [
                values[:, 0].reshape(5, 2, order="F").std(ddof=1, axis=1) ** 2,
                np.nan * np.ones(5),
            ]
        ).T.astype(self.dtype)
        expected_counts = counts + 2

        self.algo(out, counts, values, labels)
        tm.assert_almost_equal(out, expected_out, rtol=0.5e-06)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_constant(self):
        # Regression test from GH 10448.

        out = np.array([[np.nan]], dtype=self.dtype)
        counts = np.array([0], dtype="int64")
        values = 0.832845131556193 * np.ones((3, 1), dtype=self.dtype)
        labels = np.zeros(3, dtype="intp")

        self.algo(out, counts, values, labels)

        assert counts[0] == 3
        assert out[0, 0] >= 0
        tm.assert_almost_equal(out[0, 0], 0.0)


class TestGroupVarFloat64(GroupVarTestMixin):
    __test__ = True

    algo = staticmethod(group_var)
    dtype = np.float64
    rtol = 1e-5

    def test_group_var_large_inputs(self):
        prng = np.random.default_rng(2)

        out = np.array([[np.nan]], dtype=self.dtype)
        counts = np.array([0], dtype="int64")
        values = (prng.random(10**6) + 10**12).astype(self.dtype)
        values.shape = (10**6, 1)
        labels = np.zeros(10**6, dtype="intp")

        self.algo(out, counts, values, labels)

        assert counts[0] == 10**6
        tm.assert_almost_equal(out[0, 0], 1.0 / 12, rtol=0.5e-3)


class TestGroupVarFloat32(GroupVarTestMixin):
    __test__ = True

    algo = staticmethod(group_var)
    dtype = np.float32
    rtol = 1e-2


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_group_ohlc(dtype):
    obj = np.array(np.random.default_rng(2).standard_normal(20), dtype=dtype)

    bins = np.array([6, 12, 20])
    out = np.zeros((3, 4), dtype)
    counts = np.zeros(len(out), dtype=np.int64)
    labels = ensure_platform_int(np.repeat(np.arange(3), np.diff(np.r_[0, bins])))

    func = libgroupby.group_ohlc
    func(out, counts, obj[:, None], labels)

    def _ohlc(group):
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group[0], group.max(), group.min(), group[-1]]

    expected = np.array([_ohlc(obj[:6]), _ohlc(obj[6:12]), _ohlc(obj[12:])])

    tm.assert_almost_equal(out, expected)
    tm.assert_numpy_array_equal(counts, np.array([6, 6, 8], dtype=np.int64))

    obj[:6] = np.nan
    func(out, counts, obj[:, None], labels)
    expected[0] = np.nan
    tm.assert_almost_equal(out, expected)


def _check_cython_group_transform_cumulative(pd_op, np_op, dtype):
    """
    Check a group transform that executes a cumulative function.

    Parameters
    ----------
    pd_op : callable
        The pandas cumulative function.
    np_op : callable
        The analogous one in NumPy.
    dtype : type
        The specified dtype of the data.
    """
    is_datetimelike = False

    data = np.array([[1], [2], [3], [4]], dtype=dtype)
    answer = np.zeros_like(data)

    labels = np.array([0, 0, 0, 0], dtype=np.intp)
    ngroups = 1
    pd_op(answer, data, labels, ngroups, is_datetimelike)

    tm.assert_numpy_array_equal(np_op(data), answer[:, 0], check_dtype=False)


@pytest.mark.parametrize("np_dtype", ["int64", "uint64", "float32", "float64"])
def test_cython_group_transform_cumsum(np_dtype):
    # see gh-4095
    dtype = np.dtype(np_dtype).type
    pd_op, np_op = group_cumsum, np.cumsum
    _check_cython_group_transform_cumulative(pd_op, np_op, dtype)


def test_cython_group_transform_cumprod():
    # see gh-4095
    dtype = np.float64
    pd_op, np_op = group_cumprod, np.cumprod
    _check_cython_group_transform_cumulative(pd_op, np_op, dtype)


def test_cython_group_transform_algos():
    # see gh-4095
    is_datetimelike = False

    # with nans
    labels = np.array([0, 0, 0, 0, 0], dtype=np.intp)
    ngroups = 1

    data = np.array([[1], [2], [3], [np.nan], [4]], dtype="float64")
    actual = np.zeros_like(data)
    actual.fill(np.nan)
    group_cumprod(actual, data, labels, ngroups, is_datetimelike)
    expected = np.array([1, 2, 6, np.nan, 24], dtype="float64")
    tm.assert_numpy_array_equal(actual[:, 0], expected)

    actual = np.zeros_like(data)
    actual.fill(np.nan)
    group_cumsum(actual, data, labels, ngroups, is_datetimelike)
    expected = np.array([1, 3, 6, np.nan, 10], dtype="float64")
    tm.assert_numpy_array_equal(actual[:, 0], expected)

    # timedelta
    is_datetimelike = True
    data = np.array([np.timedelta64(1, "ns")] * 5, dtype="m8[ns]")[:, None]
    actual = np.zeros_like(data, dtype="int64")
    group_cumsum(actual, data.view("int64"), labels, ngroups, is_datetimelike)
    expected = np.array(
        [
            np.timedelta64(1, "ns"),
            np.timedelta64(2, "ns"),
            np.timedelta64(3, "ns"),
            np.timedelta64(4, "ns"),
            np.timedelta64(5, "ns"),
        ]
    )
    tm.assert_numpy_array_equal(actual[:, 0].view("m8[ns]"), expected)


def test_cython_group_mean_datetimelike():
    actual = np.zeros(shape=(1, 1), dtype="float64")
    counts = np.array([0], dtype="int64")
    data = (
        np.array(
            [np.timedelta64(2, "ns"), np.timedelta64(4, "ns"), np.timedelta64("NaT")],
            dtype="m8[ns]",
        )[:, None]
        .view("int64")
        .astype("float64")
    )
    labels = np.zeros(len(data), dtype=np.intp)

    group_mean(actual, counts, data, labels, is_datetimelike=True)

    tm.assert_numpy_array_equal(actual[:, 0], np.array([3], dtype="float64"))


def test_cython_group_mean_wrong_min_count():
    actual = np.zeros(shape=(1, 1), dtype="float64")
    counts = np.zeros(1, dtype="int64")
    data = np.zeros(1, dtype="float64")[:, None]
    labels = np.zeros(1, dtype=np.intp)

    with pytest.raises(AssertionError, match="min_count"):
        group_mean(actual, counts, data, labels, is_datetimelike=True, min_count=0)


def test_cython_group_mean_not_datetimelike_but_has_NaT_values():
    actual = np.zeros(shape=(1, 1), dtype="float64")
    counts = np.array([0], dtype="int64")
    data = (
        np.array(
            [np.timedelta64("NaT"), np.timedelta64("NaT")],
            dtype="m8[ns]",
        )[:, None]
        .view("int64")
        .astype("float64")
    )
    labels = np.zeros(len(data), dtype=np.intp)

    group_mean(actual, counts, data, labels, is_datetimelike=False)

    tm.assert_numpy_array_equal(
        actual[:, 0], np.array(np.divide(np.add(data[0], data[1]), 2), dtype="float64")
    )


def test_cython_group_mean_Inf_at_begining_and_end():
    # GH 50367
    actual = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype="float64")
    counts = np.array([0, 0], dtype="int64")
    data = np.array(
        [[np.inf, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5, np.inf]],
        dtype="float64",
    )
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)

    group_mean(actual, counts, data, labels, is_datetimelike=False)

    expected = np.array([[np.inf, 3], [3, np.inf]], dtype="float64")

    tm.assert_numpy_array_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "values, out",
    [
        ([[np.inf], [np.inf], [np.inf]], [[np.inf], [np.inf]]),
        ([[np.inf], [np.inf], [-np.inf]], [[np.inf], [np.nan]]),
        ([[np.inf], [-np.inf], [np.inf]], [[np.inf], [np.nan]]),
        ([[np.inf], [-np.inf], [-np.inf]], [[np.inf], [-np.inf]]),
    ],
)
def test_cython_group_sum_Inf_at_begining_and_end(values, out):
    # GH #53606
    actual = np.array([[np.nan], [np.nan]], dtype="float64")
    counts = np.array([0, 0], dtype="int64")
    data = np.array(values, dtype="float64")
    labels = np.array([0, 1, 1], dtype=np.intp)

    group_sum(actual, counts, data, labels, None, is_datetimelike=False)

    expected = np.array(out, dtype="float64")

    tm.assert_numpy_array_equal(
        actual,
        expected,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_scalar_compat.py ===
"""
Tests for DatetimeIndex methods behaving like their Timestamp counterparts
"""

import calendar
from datetime import (
    date,
    datetime,
    time,
)
import locale
import unicodedata

import numpy as np
import pytest

from pandas._libs.tslibs import timezones

from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestDatetimeIndexOps:
    def test_dti_no_millisecond_field(self):
        msg = "type object 'DatetimeIndex' has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex.millisecond

        msg = "'DatetimeIndex' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex([]).millisecond

    def test_dti_time(self):
        rng = date_range("1/1/2000", freq="12min", periods=10)
        result = Index(rng).time
        expected = [t.time() for t in rng]
        assert (result == expected).all()

    def test_dti_date(self):
        rng = date_range("1/1/2000", freq="12h", periods=10)
        result = Index(rng).date
        expected = [t.date() for t in rng]
        assert (result == expected).all()

    @pytest.mark.parametrize(
        "dtype",
        [None, "datetime64[ns, CET]", "datetime64[ns, EST]", "datetime64[ns, UTC]"],
    )
    def test_dti_date2(self, dtype):
        # Regression test for GH#21230
        expected = np.array([date(2018, 6, 4), NaT])

        index = DatetimeIndex(["2018-06-04 10:00:00", NaT], dtype=dtype)
        result = index.date

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [None, "datetime64[ns, CET]", "datetime64[ns, EST]", "datetime64[ns, UTC]"],
    )
    def test_dti_time2(self, dtype):
        # Regression test for GH#21267
        expected = np.array([time(10, 20, 30), NaT])

        index = DatetimeIndex(["2018-06-04 10:20:30", NaT], dtype=dtype)
        result = index.time

        tm.assert_numpy_array_equal(result, expected)

    def test_dti_timetz(self, tz_naive_fixture):
        # GH#21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)

        expected = np.array([time(10, 20, 30, tzinfo=tz), NaT])

        index = DatetimeIndex(["2018-06-04 10:20:30", NaT], tz=tz)
        result = index.timetz

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "field",
        [
            "dayofweek",
            "day_of_week",
            "dayofyear",
            "day_of_year",
            "quarter",
            "days_in_month",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ],
    )
    def test_dti_timestamp_fields(self, field):
        # extra fields from DatetimeIndex like quarter and week
        idx = date_range("2020-01-01", periods=10)
        expected = getattr(idx, field)[-1]

        result = getattr(Timestamp(idx[-1]), field)
        assert result == expected

    def test_dti_nanosecond(self):
        dti = DatetimeIndex(np.arange(10))
        expected = Index(np.arange(10, dtype=np.int32))

        tm.assert_index_equal(dti.nanosecond, expected)

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_hour_tzaware(self, prefix):
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]
        rng = DatetimeIndex(strdates, tz=prefix + "US/Eastern")
        assert (rng.hour == 0).all()

        # a more unusual time zone, GH#1946
        dr = date_range(
            "2011-10-02 00:00", freq="h", periods=10, tz=prefix + "America/Atikokan"
        )

        expected = Index(np.arange(10, dtype=np.int32))
        tm.assert_index_equal(dr.hour, expected)

    # GH#12806
    # error: Unsupported operand types for + ("List[None]" and "List[str]")
    @pytest.mark.parametrize(
        "time_locale", [None] + tm.get_locales()  # type: ignore[operator]
    )
    def test_day_name_month_name(self, time_locale):
        # Test Monday -> Sunday and January -> December, in that sequence
        if time_locale is None:
            # If the time_locale is None, day-name and month_name should
            # return the english attributes
            expected_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            expected_months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = calendar.day_name[:]
                expected_months = calendar.month_name[1:]

        # GH#11128
        dti = date_range(freq="D", start=datetime(1998, 1, 1), periods=365)
        english_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            name = name.capitalize()
            assert dti.day_name(locale=time_locale)[day] == name
            assert dti.day_name(locale=None)[day] == eng_name
            ts = Timestamp(datetime(2016, 4, day))
            assert ts.day_name(locale=time_locale) == name
        dti = dti.append(DatetimeIndex([NaT]))
        assert np.isnan(dti.day_name(locale=time_locale)[-1])
        ts = Timestamp(NaT)
        assert np.isnan(ts.day_name(locale=time_locale))

        # GH#12805
        dti = date_range(freq="ME", start="2012", end="2013")
        result = dti.month_name(locale=time_locale)
        expected = Index([month.capitalize() for month in expected_months])

        # work around different normalization schemes GH#22342
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        tm.assert_index_equal(result, expected)

        for item, expected in zip(dti, expected_months):
            result = item.month_name(locale=time_locale)
            expected = expected.capitalize()

            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", result)

            assert result == expected
        dti = dti.append(DatetimeIndex([NaT]))
        assert np.isnan(dti.month_name(locale=time_locale)[-1])

    def test_dti_week(self):
        # GH#6538: Check that DatetimeIndex and its TimeStamp elements
        # return the same weekofyear accessor close to new year w/ tz
        dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
        dates = DatetimeIndex(dates, tz="Europe/Brussels")
        expected = [52, 1, 1]
        assert dates.isocalendar().week.tolist() == expected
        assert [d.weekofyear for d in dates] == expected

    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_dti_fields(self, tz):
        # GH#13303
        dti = date_range(freq="D", start=datetime(1998, 1, 1), periods=365, tz=tz)
        assert dti.year[0] == 1998
        assert dti.month[0] == 1
        assert dti.day[0] == 1
        assert dti.hour[0] == 0
        assert dti.minute[0] == 0
        assert dti.second[0] == 0
        assert dti.microsecond[0] == 0
        assert dti.dayofweek[0] == 3

        assert dti.dayofyear[0] == 1
        assert dti.dayofyear[120] == 121

        assert dti.isocalendar().week.iloc[0] == 1
        assert dti.isocalendar().week.iloc[120] == 18

        assert dti.quarter[0] == 1
        assert dti.quarter[120] == 2

        assert dti.days_in_month[0] == 31
        assert dti.days_in_month[90] == 30

        assert dti.is_month_start[0]
        assert not dti.is_month_start[1]
        assert dti.is_month_start[31]
        assert dti.is_quarter_start[0]
        assert dti.is_quarter_start[90]
        assert dti.is_year_start[0]
        assert not dti.is_year_start[364]
        assert not dti.is_month_end[0]
        assert dti.is_month_end[30]
        assert not dti.is_month_end[31]
        assert dti.is_month_end[364]
        assert not dti.is_quarter_end[0]
        assert not dti.is_quarter_end[30]
        assert dti.is_quarter_end[89]
        assert dti.is_quarter_end[364]
        assert not dti.is_year_end[0]
        assert dti.is_year_end[364]

        assert len(dti.year) == 365
        assert len(dti.month) == 365
        assert len(dti.day) == 365
        assert len(dti.hour) == 365
        assert len(dti.minute) == 365
        assert len(dti.second) == 365
        assert len(dti.microsecond) == 365
        assert len(dti.dayofweek) == 365
        assert len(dti.dayofyear) == 365
        assert len(dti.isocalendar()) == 365
        assert len(dti.quarter) == 365
        assert len(dti.is_month_start) == 365
        assert len(dti.is_month_end) == 365
        assert len(dti.is_quarter_start) == 365
        assert len(dti.is_quarter_end) == 365
        assert len(dti.is_year_start) == 365
        assert len(dti.is_year_end) == 365

        dti.name = "name"

        # non boolean accessors -> return Index
        for accessor in DatetimeArray._field_ops:
            res = getattr(dti, accessor)
            assert len(res) == 365
            assert isinstance(res, Index)
            assert res.name == "name"

        # boolean accessors -> return array
        for accessor in DatetimeArray._bool_ops:
            res = getattr(dti, accessor)
            assert len(res) == 365
            assert isinstance(res, np.ndarray)

        # test boolean indexing
        res = dti[dti.is_quarter_start]
        exp = dti[[0, 90, 181, 273]]
        tm.assert_index_equal(res, exp)
        res = dti[dti.is_leap_year]
        exp = DatetimeIndex([], freq="D", tz=dti.tz, name="name").as_unit("ns")
        tm.assert_index_equal(res, exp)

    def test_dti_is_year_quarter_start(self):
        dti = date_range(freq="BQE-FEB", start=datetime(1998, 1, 1), periods=4)

        assert sum(dti.is_quarter_start) == 0
        assert sum(dti.is_quarter_end) == 4
        assert sum(dti.is_year_start) == 0
        assert sum(dti.is_year_end) == 1

    def test_dti_is_month_start(self):
        dti = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])

        assert dti.is_month_start[0] == 1

    def test_dti_is_month_start_custom(self):
        # Ensure is_start/end accessors throw ValueError for CustomBusinessDay,
        bday_egypt = offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")
        dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
        msg = "Custom business days is not supported by is_month_start"
        with pytest.raises(ValueError, match=msg):
            dti.is_month_start

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_custom_business_hour.py ===
"""
Tests for offsets.CustomBusinessHour
"""
from __future__ import annotations

from datetime import (
    datetime,
    time as dt_time,
)

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    BusinessHour,
    CustomBusinessHour,
    Nano,
)

from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries.holiday import USFederalHolidayCalendar

holidays = ["2014-06-27", datetime(2014, 6, 30), np.datetime64("2014-07-02")]


@pytest.fixture
def dt():
    return datetime(2014, 7, 1, 10, 00)


@pytest.fixture
def _offset():
    return CustomBusinessHour


# 2014 Calendar to check custom holidays
#   Sun Mon Tue Wed Thu Fri Sat
#  6/22  23  24  25  26  27  28
#    29  30 7/1   2   3   4   5
#     6   7   8   9  10  11  12
@pytest.fixture
def offset1():
    return CustomBusinessHour(weekmask="Tue Wed Thu Fri")


@pytest.fixture
def offset2():
    return CustomBusinessHour(holidays=holidays)


class TestCustomBusinessHour:
    def test_constructor_errors(self):
        msg = "time data must be specified only with hour and minute"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start=dt_time(11, 0, 5))
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start="AAA")
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start="14:00:05")

    def test_different_normalize_equals(self, _offset):
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(self, offset1, offset2):
        assert repr(offset1) == "<CustomBusinessHour: cbh=09:00-17:00>"
        assert repr(offset2) == "<CustomBusinessHour: cbh=09:00-17:00>"

    def test_with_offset(self, dt):
        expected = Timestamp("2014-07-01 13:00")

        assert dt + CustomBusinessHour() * 3 == expected
        assert dt + CustomBusinessHour(n=3) == expected

    def test_eq(self, offset1, offset2):
        for offset in [offset1, offset2]:
            assert offset == offset

        assert CustomBusinessHour() != CustomBusinessHour(-1)
        assert CustomBusinessHour(start="09:00") == CustomBusinessHour()
        assert CustomBusinessHour(start="09:00") != CustomBusinessHour(start="09:01")
        assert CustomBusinessHour(start="09:00", end="17:00") != CustomBusinessHour(
            start="17:00", end="09:01"
        )

        assert CustomBusinessHour(weekmask="Tue Wed Thu Fri") != CustomBusinessHour(
            weekmask="Mon Tue Wed Thu Fri"
        )
        assert CustomBusinessHour(holidays=["2014-06-27"]) != CustomBusinessHour(
            holidays=["2014-06-28"]
        )

    def test_hash(self, offset1, offset2):
        assert hash(offset1) == hash(offset1)
        assert hash(offset2) == hash(offset2)

    def test_add_dateime(self, dt, offset1, offset2):
        assert offset1 + dt == datetime(2014, 7, 1, 11)
        assert offset2 + dt == datetime(2014, 7, 1, 11)

    def testRollback1(self, dt, offset1, offset2):
        assert offset1.rollback(dt) == dt
        assert offset2.rollback(dt) == dt

        d = datetime(2014, 7, 1, 0)

        # 2014/07/01 is Tuesday, 06/30 is Monday(holiday)
        assert offset1.rollback(d) == datetime(2014, 6, 27, 17)

        # 2014/6/30 and 2014/6/27 are holidays
        assert offset2.rollback(d) == datetime(2014, 6, 26, 17)

    def testRollback2(self, _offset):
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    def testRollforward1(self, dt, offset1, offset2):
        assert offset1.rollforward(dt) == dt
        assert offset2.rollforward(dt) == dt

        d = datetime(2014, 7, 1, 0)
        assert offset1.rollforward(d) == datetime(2014, 7, 1, 9)
        assert offset2.rollforward(d) == datetime(2014, 7, 1, 9)

    def testRollforward2(self, _offset):
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    def test_roll_date_object(self):
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)

    normalize_cases = [
        (
            CustomBusinessHour(normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        ),
        (
            CustomBusinessHour(-1, normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 26),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        ),
        (
            CustomBusinessHour(
                1, normalize=True, start="17:00", end="04:00", holidays=holidays
            ),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 3),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        ),
    ]

    @pytest.mark.parametrize("norm_cases", normalize_cases)
    def test_normalize(self, norm_cases):
        offset, cases = norm_cases
        for dt, expected in cases.items():
            assert offset._apply(dt) == expected

    @pytest.mark.parametrize(
        "dt, expected",
        [
            [datetime(2014, 7, 1, 9), False],
            [datetime(2014, 7, 1, 10), True],
            [datetime(2014, 7, 1, 15), True],
            [datetime(2014, 7, 1, 15, 1), False],
            [datetime(2014, 7, 5, 12), False],
            [datetime(2014, 7, 6, 12), False],
        ],
    )
    def test_is_on_offset(self, dt, expected):
        offset = CustomBusinessHour(start="10:00", end="15:00", holidays=holidays)
        assert offset.is_on_offset(dt) == expected

    apply_cases = [
        (
            CustomBusinessHour(holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 3, 9, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 10),
                # out of business hours
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            CustomBusinessHour(4, holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 3, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 12, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 12, 30, 30),
            },
        ),
    ]

    @pytest.mark.parametrize("apply_case", apply_cases)
    def test_apply(self, apply_case):
        offset, cases = apply_case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    nano_cases = [
        (
            CustomBusinessHour(holidays=holidays),
            {
                Timestamp("2014-07-01 15:00")
                + Nano(5): Timestamp("2014-07-01 16:00")
                + Nano(5),
                Timestamp("2014-07-01 16:00")
                + Nano(5): Timestamp("2014-07-03 09:00")
                + Nano(5),
                Timestamp("2014-07-01 16:00")
                - Nano(5): Timestamp("2014-07-01 17:00")
                - Nano(5),
            },
        ),
        (
            CustomBusinessHour(-1, holidays=holidays),
            {
                Timestamp("2014-07-01 15:00")
                + Nano(5): Timestamp("2014-07-01 14:00")
                + Nano(5),
                Timestamp("2014-07-01 10:00")
                + Nano(5): Timestamp("2014-07-01 09:00")
                + Nano(5),
                Timestamp("2014-07-01 10:00")
                - Nano(5): Timestamp("2014-06-26 17:00")
                - Nano(5),
            },
        ),
    ]

    @pytest.mark.parametrize("nano_case", nano_cases)
    def test_apply_nanoseconds(self, nano_case):
        offset, cases = nano_case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_us_federal_holiday_with_datetime(self):
        # GH 16867
        bhour_us = CustomBusinessHour(calendar=USFederalHolidayCalendar())
        t0 = datetime(2014, 1, 17, 15)
        result = t0 + bhour_us * 8
        expected = Timestamp("2014-01-21 15:00:00")
        assert result == expected


@pytest.mark.parametrize(
    "weekmask, expected_time, mult",
    [
        ["Mon Tue Wed Thu Fri Sat", "2018-11-10 09:00:00", 10],
        ["Tue Wed Thu Fri Sat", "2018-11-13 08:00:00", 18],
    ],
)
def test_custom_businesshour_weekmask_and_holidays(weekmask, expected_time, mult):
    # GH 23542
    holidays = ["2018-11-09"]
    bh = CustomBusinessHour(
        start="08:00", end="17:00", weekmask=weekmask, holidays=holidays
    )
    result = Timestamp("2018-11-08 08:00") + mult * bh
    expected = Timestamp(expected_time)
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_apply.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    isna,
    notna,
)
import pandas._testing as tm

from pandas.tseries import offsets

# suppress warnings about empty slices, as we are deliberately testing
# with a 0-length Series
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*(empty slice|0 for slice).*:RuntimeWarning"
)


def f(x):
    return x[np.isfinite(x)].mean()


@pytest.mark.parametrize("bad_raw", [None, 1, 0])
def test_rolling_apply_invalid_raw(bad_raw):
    with pytest.raises(ValueError, match="raw parameter must be `True` or `False`"):
        Series(range(3)).rolling(1).apply(len, raw=bad_raw)


def test_rolling_apply_out_of_bounds(engine_and_raw):
    # gh-1850
    engine, raw = engine_and_raw

    vals = Series([1, 2, 3, 4])

    result = vals.rolling(10).apply(np.sum, engine=engine, raw=raw)
    assert result.isna().all()

    result = vals.rolling(10, min_periods=1).apply(np.sum, engine=engine, raw=raw)
    expected = Series([1, 3, 6, 10], dtype=float)
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize("window", [2, "2s"])
def test_rolling_apply_with_pandas_objects(window):
    # 5071
    df = DataFrame(
        {
            "A": np.random.default_rng(2).standard_normal(5),
            "B": np.random.default_rng(2).integers(0, 10, size=5),
        },
        index=date_range("20130101", periods=5, freq="s"),
    )

    # we have an equal spaced timeseries index
    # so simulate removing the first period
    def f(x):
        if x.index[0] == df.index[0]:
            return np.nan
        return x.iloc[-1]

    result = df.rolling(window).apply(f, raw=False)
    expected = df.iloc[2:].reindex_like(df)
    tm.assert_frame_equal(result, expected)

    with tm.external_error_raised(AttributeError):
        df.rolling(window).apply(f, raw=True)


def test_rolling_apply(engine_and_raw, step):
    engine, raw = engine_and_raw

    expected = Series([], dtype="float64")
    result = expected.rolling(10, step=step).apply(
        lambda x: x.mean(), engine=engine, raw=raw
    )
    tm.assert_series_equal(result, expected)

    # gh-8080
    s = Series([None, None, None])
    result = s.rolling(2, min_periods=0, step=step).apply(
        lambda x: len(x), engine=engine, raw=raw
    )
    expected = Series([1.0, 2.0, 2.0])[::step]
    tm.assert_series_equal(result, expected)

    result = s.rolling(2, min_periods=0, step=step).apply(len, engine=engine, raw=raw)
    tm.assert_series_equal(result, expected)


def test_all_apply(engine_and_raw):
    engine, raw = engine_and_raw

    df = (
        DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        ).set_index("A")
        * 2
    )
    er = df.rolling(window=1)
    r = df.rolling(window="1s")

    result = r.apply(lambda x: 1, engine=engine, raw=raw)
    expected = er.apply(lambda x: 1, engine=engine, raw=raw)
    tm.assert_frame_equal(result, expected)


def test_ragged_apply(engine_and_raw):
    engine, raw = engine_and_raw

    df = DataFrame({"B": range(5)})
    df.index = [
        Timestamp("20130101 09:00:00"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:03"),
        Timestamp("20130101 09:00:05"),
        Timestamp("20130101 09:00:06"),
    ]

    f = lambda x: 1
    result = df.rolling(window="1s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)

    result = df.rolling(window="2s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)

    result = df.rolling(window="5s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)


def test_invalid_engine():
    with pytest.raises(ValueError, match="engine must be either 'numba' or 'cython'"):
        Series(range(1)).rolling(1).apply(lambda x: x, engine="foo")


def test_invalid_engine_kwargs_cython():
    with pytest.raises(ValueError, match="cython engine does not accept engine_kwargs"):
        Series(range(1)).rolling(1).apply(
            lambda x: x, engine="cython", engine_kwargs={"nopython": False}
        )


def test_invalid_raw_numba():
    with pytest.raises(
        ValueError, match="raw must be `True` when using the numba engine"
    ):
        Series(range(1)).rolling(1).apply(lambda x: x, raw=False, engine="numba")


@pytest.mark.parametrize("args_kwargs", [[None, {"par": 10}], [(10,), None]])
def test_rolling_apply_args_kwargs(args_kwargs):
    # GH 33433
    def numpysum(x, par):
        return np.sum(x + par)

    df = DataFrame({"gr": [1, 1], "a": [1, 2]})

    idx = Index(["gr", "a"])
    expected = DataFrame([[11.0, 11.0], [11.0, 12.0]], columns=idx)

    result = df.rolling(1).apply(numpysum, args=args_kwargs[0], kwargs=args_kwargs[1])
    tm.assert_frame_equal(result, expected)

    midx = MultiIndex.from_tuples([(1, 0), (1, 1)], names=["gr", None])
    expected = Series([11.0, 12.0], index=midx, name="a")

    gb_rolling = df.groupby("gr")["a"].rolling(1)

    result = gb_rolling.apply(numpysum, args=args_kwargs[0], kwargs=args_kwargs[1])
    tm.assert_series_equal(result, expected)


def test_nans(raw):
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = obj.rolling(50, min_periods=30).apply(f, raw=raw)
    tm.assert_almost_equal(result.iloc[-1], np.mean(obj[10:-10]))

    # min_periods is working correctly
    result = obj.rolling(20, min_periods=15).apply(f, raw=raw)
    assert isna(result.iloc[23])
    assert not isna(result.iloc[24])

    assert not isna(result.iloc[-6])
    assert isna(result.iloc[-5])

    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    result = obj2.rolling(10, min_periods=5).apply(f, raw=raw)
    assert isna(result.iloc[3])
    assert notna(result.iloc[4])

    result0 = obj.rolling(20, min_periods=0).apply(f, raw=raw)
    result1 = obj.rolling(20, min_periods=1).apply(f, raw=raw)
    tm.assert_almost_equal(result0, result1)


def test_center(raw):
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = obj.rolling(20, min_periods=15, center=True).apply(f, raw=raw)
    expected = (
        concat([obj, Series([np.nan] * 9)])
        .rolling(20, min_periods=15)
        .apply(f, raw=raw)
        .iloc[9:]
        .reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


def test_series(raw, series):
    result = series.rolling(50).apply(f, raw=raw)
    assert isinstance(result, Series)
    tm.assert_almost_equal(result.iloc[-1], np.mean(series[-50:]))


def test_frame(raw, frame):
    result = frame.rolling(50).apply(f, raw=raw)
    assert isinstance(result, DataFrame)
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[-50:, :].apply(np.mean, axis=0, raw=raw),
        check_names=False,
    )


def test_time_rule_series(raw, series):
    win = 25
    minp = 10
    ser = series[::2].resample("B").mean()
    series_result = ser.rolling(window=win, min_periods=minp).apply(f, raw=raw)
    last_date = series_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_series = series[::2].truncate(prev_date, last_date)
    tm.assert_almost_equal(series_result.iloc[-1], np.mean(trunc_series))


def test_time_rule_frame(raw, frame):
    win = 25
    minp = 10
    frm = frame[::2].resample("B").mean()
    frame_result = frm.rolling(window=win, min_periods=minp).apply(f, raw=raw)
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(np.mean, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("minp", [0, 99, 100])
def test_min_periods(raw, series, minp, step):
    result = series.rolling(len(series) + 1, min_periods=minp, step=step).apply(
        f, raw=raw
    )
    expected = series.rolling(len(series), min_periods=minp, step=step).apply(
        f, raw=raw
    )
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


def test_center_reindex_series(raw, series):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]
    minp = 10

    series_xp = (
        series.reindex(list(series.index) + s)
        .rolling(window=25, min_periods=minp)
        .apply(f, raw=raw)
        .shift(-12)
        .reindex(series.index)
    )
    series_rs = series.rolling(window=25, min_periods=minp, center=True).apply(
        f, raw=raw
    )
    tm.assert_series_equal(series_xp, series_rs)


def test_center_reindex_frame(raw):
    # shifter index
    frame = DataFrame(range(100), index=date_range("2020-01-01", freq="D", periods=100))
    s = [f"x{x:d}" for x in range(12)]
    minp = 10

    frame_xp = (
        frame.reindex(list(frame.index) + s)
        .rolling(window=25, min_periods=minp)
        .apply(f, raw=raw)
        .shift(-12)
        .reindex(frame.index)
    )
    frame_rs = frame.rolling(window=25, min_periods=minp, center=True).apply(f, raw=raw)
    tm.assert_frame_equal(frame_xp, frame_rs)


def test_axis1(raw):
    # GH 45912
    df = DataFrame([1, 2])
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=1, axis=1).apply(np.sum, raw=raw)
    expected = DataFrame([1.0, 2.0])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_network.py ===
"""
Tests parsers ability to read and parse non-local files
and hence require a network connection to be read.
"""
from io import BytesIO
import logging
import re

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.mark.network
@pytest.mark.single_cpu
@pytest.mark.parametrize("mode", ["explicit", "infer"])
@pytest.mark.parametrize("engine", ["python", "c"])
def test_compressed_urls(
    httpserver,
    datapath,
    salaries_table,
    mode,
    engine,
    compression_only,
    compression_to_extension,
):
    # test reading compressed urls with various engines and
    # extension inference
    if compression_only == "tar":
        pytest.skip("TODO: Add tar salaraies.csv to pandas/io/parsers/data")

    extension = compression_to_extension[compression_only]
    with open(datapath("io", "parser", "data", "salaries.csv" + extension), "rb") as f:
        httpserver.serve_content(content=f.read())

    url = httpserver.url + "/salaries.csv" + extension

    if mode != "explicit":
        compression_only = mode

    url_table = read_csv(url, sep="\t", compression=compression_only, engine=engine)
    tm.assert_frame_equal(url_table, salaries_table)


@pytest.mark.network
@pytest.mark.single_cpu
def test_url_encoding_csv(httpserver, datapath):
    """
    read_csv should honor the requested encoding for URLs.

    GH 10424
    """
    with open(datapath("io", "parser", "data", "unicode_series.csv"), "rb") as f:
        httpserver.serve_content(content=f.read())
        df = read_csv(httpserver.url, encoding="latin-1", header=None)
    assert df.loc[15, 1] == "Á köldum klaka (Cold Fever) (1994)"


@pytest.fixture
def tips_df(datapath):
    """DataFrame with the tips dataset."""
    return read_csv(datapath("io", "data", "csv", "tips.csv"))


@pytest.mark.single_cpu
@pytest.mark.usefixtures("s3_resource")
@td.skip_if_not_us_locale()
class TestS3:
    def test_parse_public_s3_bucket(self, s3_public_bucket_with_data, tips_df, s3so):
        # more of an integration test due to the not-public contents portion
        # can probably mock this though.
        pytest.importorskip("s3fs")
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                compression=comp,
                storage_options=s3so,
            )
            assert isinstance(df, DataFrame)
            assert not df.empty
            tm.assert_frame_equal(df, tips_df)

    def test_parse_private_s3_bucket(self, s3_private_bucket_with_data, tips_df, s3so):
        # Read public file from bucket with not-public contents
        pytest.importorskip("s3fs")
        df = read_csv(
            f"s3://{s3_private_bucket_with_data.name}/tips.csv", storage_options=s3so
        )
        assert isinstance(df, DataFrame)
        assert not df.empty
        tm.assert_frame_equal(df, tips_df)

    def test_parse_public_s3n_bucket(self, s3_public_bucket_with_data, tips_df, s3so):
        # Read from AWS s3 as "s3n" URL
        df = read_csv(
            f"s3n://{s3_public_bucket_with_data.name}/tips.csv",
            nrows=10,
            storage_options=s3so,
        )
        assert isinstance(df, DataFrame)
        assert not df.empty
        tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_parse_public_s3a_bucket(self, s3_public_bucket_with_data, tips_df, s3so):
        # Read from AWS s3 as "s3a" URL
        df = read_csv(
            f"s3a://{s3_public_bucket_with_data.name}/tips.csv",
            nrows=10,
            storage_options=s3so,
        )
        assert isinstance(df, DataFrame)
        assert not df.empty
        tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_parse_public_s3_bucket_nrows(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                nrows=10,
                compression=comp,
                storage_options=s3so,
            )
            assert isinstance(df, DataFrame)
            assert not df.empty
            tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_parse_public_s3_bucket_chunked(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        # Read with a chunksize
        chunksize = 5
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            with read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                chunksize=chunksize,
                compression=comp,
                storage_options=s3so,
            ) as df_reader:
                assert df_reader.chunksize == chunksize
                for i_chunk in [0, 1, 2]:
                    # Read a couple of chunks and make sure we see them
                    # properly.
                    df = df_reader.get_chunk()
                    assert isinstance(df, DataFrame)
                    assert not df.empty
                    true_df = tips_df.iloc[
                        chunksize * i_chunk : chunksize * (i_chunk + 1)
                    ]
                    tm.assert_frame_equal(true_df, df)

    def test_parse_public_s3_bucket_chunked_python(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        # Read with a chunksize using the Python parser
        chunksize = 5
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            with read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                chunksize=chunksize,
                compression=comp,
                engine="python",
                storage_options=s3so,
            ) as df_reader:
                assert df_reader.chunksize == chunksize
                for i_chunk in [0, 1, 2]:
                    # Read a couple of chunks and make sure we see them properly.
                    df = df_reader.get_chunk()
                    assert isinstance(df, DataFrame)
                    assert not df.empty
                    true_df = tips_df.iloc[
                        chunksize * i_chunk : chunksize * (i_chunk + 1)
                    ]
                    tm.assert_frame_equal(true_df, df)

    def test_parse_public_s3_bucket_python(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                engine="python",
                compression=comp,
                storage_options=s3so,
            )
            assert isinstance(df, DataFrame)
            assert not df.empty
            tm.assert_frame_equal(df, tips_df)

    def test_infer_s3_compression(self, s3_public_bucket_with_data, tips_df, s3so):
        for ext in ["", ".gz", ".bz2"]:
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                engine="python",
                compression="infer",
                storage_options=s3so,
            )
            assert isinstance(df, DataFrame)
            assert not df.empty
            tm.assert_frame_equal(df, tips_df)

    def test_parse_public_s3_bucket_nrows_python(
        self, s3_public_bucket_with_data, tips_df, s3so
    ):
        for ext, comp in [("", None), (".gz", "gzip"), (".bz2", "bz2")]:
            df = read_csv(
                f"s3://{s3_public_bucket_with_data.name}/tips.csv" + ext,
                engine="python",
                nrows=10,
                compression=comp,
                storage_options=s3so,
            )
            assert isinstance(df, DataFrame)
            assert not df.empty
            tm.assert_frame_equal(tips_df.iloc[:10], df)

    def test_read_s3_fails(self, s3so):
        msg = "The specified bucket does not exist"
        with pytest.raises(OSError, match=msg):
            read_csv("s3://nyqpug/asdf.csv", storage_options=s3so)

    def test_read_s3_fails_private(self, s3_private_bucket, s3so):
        msg = "The specified bucket does not exist"
        # Receive a permission error when trying to read a private bucket.
        # It's irrelevant here that this isn't actually a table.
        with pytest.raises(OSError, match=msg):
            read_csv(f"s3://{s3_private_bucket.name}/file.csv")

    @pytest.mark.xfail(reason="GH#39155 s3fs upgrade", strict=False)
    def test_write_s3_csv_fails(self, tips_df, s3so):
        # GH 32486
        # Attempting to write to an invalid S3 path should raise
        import botocore

        # GH 34087
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
        # Catch a ClientError since AWS Service Errors are defined dynamically
        error = (FileNotFoundError, botocore.exceptions.ClientError)

        with pytest.raises(error, match="The specified bucket does not exist"):
            tips_df.to_csv(
                "s3://an_s3_bucket_data_doesnt_exit/not_real.csv", storage_options=s3so
            )

    @pytest.mark.xfail(reason="GH#39155 s3fs upgrade", strict=False)
    def test_write_s3_parquet_fails(self, tips_df, s3so):
        # GH 27679
        # Attempting to write to an invalid S3 path should raise
        pytest.importorskip("pyarrow")
        import botocore

        # GH 34087
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
        # Catch a ClientError since AWS Service Errors are defined dynamically
        error = (FileNotFoundError, botocore.exceptions.ClientError)

        with pytest.raises(error, match="The specified bucket does not exist"):
            tips_df.to_parquet(
                "s3://an_s3_bucket_data_doesnt_exit/not_real.parquet",
                storage_options=s3so,
            )

    @pytest.mark.single_cpu
    def test_read_csv_handles_boto_s3_object(
        self, s3_public_bucket_with_data, tips_file
    ):
        # see gh-16135

        s3_object = s3_public_bucket_with_data.Object("tips.csv")

        with BytesIO(s3_object.get()["Body"].read()) as buffer:
            result = read_csv(buffer, encoding="utf8")
        assert isinstance(result, DataFrame)
        assert not result.empty

        expected = read_csv(tips_file)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    def test_read_csv_chunked_download(self, s3_public_bucket, caplog, s3so):
        # 8 MB, S3FS uses 5MB chunks
        df = DataFrame(np.zeros((100000, 4)), columns=list("abcd"))
        with BytesIO(df.to_csv().encode("utf-8")) as buf:
            s3_public_bucket.put_object(Key="large-file.csv", Body=buf)
            uri = f"{s3_public_bucket.name}/large-file.csv"
            match_re = re.compile(rf"^Fetch: {uri}, 0-(?P<stop>\d+)$")
            with caplog.at_level(logging.DEBUG, logger="s3fs"):
                read_csv(
                    f"s3://{uri}",
                    nrows=5,
                    storage_options=s3so,
                )
                for log in caplog.messages:
                    if match := re.match(match_re, log):
                        # Less than 8 MB
                        assert int(match.group("stop")) < 8000000

    def test_read_s3_with_hash_in_key(self, s3_public_bucket_with_data, tips_df, s3so):
        # GH 25945
        result = read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips#1.csv", storage_options=s3so
        )
        tm.assert_frame_equal(tips_df, result)

    def test_read_feather_s3_file_path(
        self, s3_public_bucket_with_data, feather_file, s3so
    ):
        # GH 29055
        pytest.importorskip("pyarrow")
        expected = read_feather(feather_file)
        res = read_feather(
            f"s3://{s3_public_bucket_with_data.name}/simple_dataset.feather",
            storage_options=s3so,
        )
        tm.assert_frame_equal(expected, res)