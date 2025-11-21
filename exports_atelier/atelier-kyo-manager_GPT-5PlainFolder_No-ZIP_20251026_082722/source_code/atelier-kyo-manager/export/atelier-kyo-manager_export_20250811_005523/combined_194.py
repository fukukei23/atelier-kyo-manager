
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_common_basic.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import HAS_PYARROW
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)

from pandas import (
    DataFrame,
    Index,
    Timestamp,
    compat,
)
import pandas._testing as tm

from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_override_set_noconvert_columns():
    # see gh-17351
    #
    # Usecols needs to be sorted in _set_noconvert_columns based
    # on the test_usecols_with_parse_dates test from test_usecols.py
    class MyTextFileReader(TextFileReader):
        def __init__(self) -> None:
            self._currow = 0
            self.squeeze = False

    class MyCParserWrapper(CParserWrapper):
        def _set_noconvert_columns(self):
            if self.usecols_dtype == "integer":
                # self.usecols is a set, which is documented as unordered
                # but in practice, a CPython set of integers is sorted.
                # In other implementations this assumption does not hold.
                # The following code simulates a different order, which
                # before GH 17351 would cause the wrong columns to be
                # converted via the parse_dates parameter
                self.usecols = list(self.usecols)
                self.usecols.reverse()
            return CParserWrapper._set_noconvert_columns(self)

    data = """a,b,c,d,e
0,1,2014-01-01,09:00,4
0,1,2014-01-02,10:00,4"""

    parse_dates = [[1, 2]]
    cols = {
        "a": [0, 0],
        "c_d": [Timestamp("2014-01-01 09:00:00"), Timestamp("2014-01-02 10:00:00")],
    }
    expected = DataFrame(cols, columns=["c_d", "a"])

    parser = MyTextFileReader()
    parser.options = {
        "usecols": [0, 2, 3],
        "parse_dates": parse_dates,
        "delimiter": ",",
    }
    parser.engine = "c"
    parser._engine = MyCParserWrapper(StringIO(data), **parser.options)

    result = parser.read()
    tm.assert_frame_equal(result, expected)


def test_read_csv_local(all_parsers, csv1):
    prefix = "file:///" if compat.is_platform_windows() else "file://"
    parser = all_parsers

    fname = prefix + str(os.path.abspath(csv1))
    result = parser.read_csv(fname, index_col=0, parse_dates=True)
    # TODO: make unit check more specific
    if parser.engine == "pyarrow":
        result.index = result.index.as_unit("ns")
    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            name="index",
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_1000_sep(all_parsers):
    parser = all_parsers
    data = """A|B|C
1|2,334|5
10|13|10.
"""
    expected = DataFrame({"A": [1, 10], "B": [2334, 13], "C": [5, 10.0]})

    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep="|", thousands=",")
        return

    result = parser.read_csv(StringIO(data), sep="|", thousands=",")
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_unnamed_columns(all_parsers):
    data = """A,B,C,,
1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=np.int64,
        columns=["A", "B", "C", "Unnamed: 3", "Unnamed: 4"],
    )
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_csv_mixed_type(all_parsers):
    data = """A,B,C
a,1,2
b,3,4
c,4,5
"""
    parser = all_parsers
    expected = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]})
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_read_csv_low_memory_no_rows_with_index(all_parsers):
    # see gh-21141
    parser = all_parsers

    if not parser.low_memory:
        pytest.skip("This is a low-memory specific test")

    data = """A,B,C
1,1,1,2
2,2,3,4
3,3,4,5
"""

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
        return

    result = parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
    expected = DataFrame(columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


def test_read_csv_dataframe(all_parsers, csv1):
    parser = all_parsers
    result = parser.read_csv(csv1, index_col=0, parse_dates=True)
    # TODO: make unit check more specific
    if parser.engine == "pyarrow":
        result.index = result.index.as_unit("ns")
    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            name="index",
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [3, 3.0])
def test_read_nrows(all_parsers, nrows):
    # see gh-10476
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    expected = DataFrame(
        [["foo", 2, 3, 4, 5], ["bar", 7, 8, 9, 10], ["baz", 12, 13, 14, 15]],
        columns=["index", "A", "B", "C", "D"],
    )
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), nrows=nrows)
        return

    result = parser.read_csv(StringIO(data), nrows=nrows)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1.2, "foo", -1])
def test_read_nrows_bad(all_parsers, nrows):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    msg = r"'nrows' must be an integer >=0"
    parser = all_parsers
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), nrows=nrows)


def test_nrows_skipfooter_errors(all_parsers):
    msg = "'skipfooter' not supported with 'nrows'"
    data = "a\n1\n2\n3\n4\n5\n6"
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=1, nrows=5)


@skip_pyarrow
def test_missing_trailing_delimiters(all_parsers):
    parser = all_parsers
    data = """A,B,C,D
1,2,3,4
1,3,3,
1,4,5"""

    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [[1, 2, 3, 4], [1, 3, 3, np.nan], [1, 4, 5, np.nan]],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)


def test_skip_initial_space(all_parsers):
    data = (
        '"09-Apr-2012", "01:10:18.300", 2456026.548822908, 12849, '
        "1.00361,  1.12551, 330.65659, 0355626618.16711,  73.48821, "
        "314.11625,  1917.09447,   179.71425,  80.000, 240.000, -350,  "
        "70.06056, 344.98370, 1,   1, -0.689265, -0.692787,  "
        "0.212036,    14.7674,   41.605,   -9999.0,   -9999.0,   "
        "-9999.0,   -9999.0,   -9999.0,  -9999.0, 000, 012, 128"
    )
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=list(range(33)),
                header=None,
                na_values=["-9999.0"],
                skipinitialspace=True,
            )
        return

    result = parser.read_csv(
        StringIO(data),
        names=list(range(33)),
        header=None,
        na_values=["-9999.0"],
        skipinitialspace=True,
    )
    expected = DataFrame(
        [
            [
                "09-Apr-2012",
                "01:10:18.300",
                2456026.548822908,
                12849,
                1.00361,
                1.12551,
                330.65659,
                355626618.16711,
                73.48821,
                314.11625,
                1917.09447,
                179.71425,
                80.0,
                240.0,
                -350,
                70.06056,
                344.9837,
                1,
                1,
                -0.689265,
                -0.692787,
                0.212036,
                14.7674,
                41.605,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                12,
                128,
            ]
        ]
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_trailing_delimiters(all_parsers):
    # see gh-2442
    data = """A,B,C
1,2,3,
4,5,6,
7,8,9,"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)

    expected = DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


def test_escapechar(all_parsers):
    # https://stackoverflow.com/questions/13824840/feature-request-for-
    # pandas-read-csv
    data = '''SEARCH_TERM,ACTUAL_URL
"bra tv board","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"tv p\xc3\xa5 hjul","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"SLAGBORD, \\"Bergslagen\\", IKEA:s 1700-tals series","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"'''

    parser = all_parsers
    result = parser.read_csv(
        StringIO(data), escapechar="\\", quotechar='"', encoding="utf-8"
    )

    assert result["SEARCH_TERM"][2] == 'SLAGBORD, "Bergslagen", IKEA:s 1700-tals series'

    tm.assert_index_equal(result.columns, Index(["SEARCH_TERM", "ACTUAL_URL"]))


def test_ignore_leading_whitespace(all_parsers):
    # see gh-3374, gh-6607
    parser = all_parsers
    data = " a b c\n 1 2 3\n 4 5 6\n 7 8 9"

    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+")
        return
    result = parser.read_csv(StringIO(data), sep=r"\s+")

    expected = DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("usecols", [None, [0, 1], ["a", "b"]])
def test_uneven_lines_with_usecols(all_parsers, usecols):
    # see gh-12203
    parser = all_parsers
    data = r"""a,b,c
0,1,2
3,4,5,6,7
8,9,10"""

    if usecols is None:
        # Make sure that an error is still raised
        # when the "usecols" parameter is not provided.
        msg = r"Expected \d+ fields in line \d+, saw \d+"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
    else:
        expected = DataFrame({"a": [0, 3, 8], "b": [1, 4, 9]})

        result = parser.read_csv(StringIO(data), usecols=usecols)
        tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # First, check to see that the response of parser when faced with no
        # provided columns raises the correct error, with or without usecols.
        ("", {}, None),
        ("", {"usecols": ["X"]}, None),
        (
            ",,",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"], index=[0], dtype=np.float64),
        ),
        (
            "",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"]),
        ),
    ],
)
def test_read_empty_with_usecols(all_parsers, data, kwargs, expected):
    # see gh-12493
    parser = all_parsers

    if expected is None:
        msg = "No columns to parse from file"
        with pytest.raises(EmptyDataError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # gh-8661, gh-8679: this should ignore six lines, including
        # lines with trailing whitespace and blank lines.
        (
            {
                "header": None,
                "delim_whitespace": True,
                "skiprows": [0, 1, 2, 3, 5, 6],
                "skip_blank_lines": True,
            },
            DataFrame([[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]]),
        ),
        # gh-8983: test skipping set of rows after a row with trailing spaces.
        (
            {
                "delim_whitespace": True,
                "skiprows": [1, 2, 3, 5, 6],
                "skip_blank_lines": True,
            },
            DataFrame({"A": [1.0, 5.1], "B": [2.0, np.nan], "C": [4.0, 10]}),
        ),
    ],
)
def test_trailing_spaces(all_parsers, kwargs, expected):
    data = "A B C  \nrandom line with trailing spaces    \nskip\n1,2,3\n1,2.,4.\nrandom line with trailing tabs\t\t\t\n   \n5.1,NaN,10.0\n"  # noqa: E501
    parser = all_parsers

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"

    if parser.engine == "pyarrow":
        msg = "The 'delim_whitespace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                FutureWarning, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
        return

    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
    tm.assert_frame_equal(result, expected)


def test_raise_on_sep_with_delim_whitespace(all_parsers):
    # see gh-6607
    data = "a b c\n1 2 3"
    parser = all_parsers

    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with pytest.raises(ValueError, match="you can only specify one"):
        with tm.assert_produces_warning(
            FutureWarning, match=depr_msg, check_stacklevel=False
        ):
            parser.read_csv(StringIO(data), sep=r"\s", delim_whitespace=True)


def test_read_filepath_or_buffer(all_parsers):
    # see gh-43366
    parser = all_parsers

    with pytest.raises(TypeError, match="Expected file path name or file-like"):
        parser.read_csv(filepath_or_buffer=b"input")


@pytest.mark.parametrize("delim_whitespace", [True, False])
def test_single_char_leading_whitespace(all_parsers, delim_whitespace):
    # see gh-9710
    parser = all_parsers
    data = """\
MyColumn
a
b
a
b\n"""

    expected = DataFrame({"MyColumn": list("abab")})
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"

    if parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                FutureWarning, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(
                    StringIO(data),
                    skipinitialspace=True,
                    delim_whitespace=delim_whitespace,
                )
        return

    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), skipinitialspace=True, delim_whitespace=delim_whitespace
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sep,skip_blank_lines,exp_data",
    [
        (",", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        (r"\s+", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        (
            ",",
            False,
            [
                [1.0, 2.0, 4.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [5.0, np.nan, 10.0],
                [np.nan, np.nan, np.nan],
                [-70.0, 0.4, 1.0],
            ],
        ),
    ],
)
def test_empty_lines(all_parsers, sep, skip_blank_lines, exp_data, request):
    parser = all_parsers
    data = """\
A,B,C
1,2.,4.


5.,NaN,10.0

-70,.4,1
"""

    if sep == r"\s+":
        data = data.replace(",", "  ")

        if parser.engine == "pyarrow":
            msg = "the 'pyarrow' engine does not support regex separators"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(
                    StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines
                )
            return

    result = parser.read_csv(StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines)
    expected = DataFrame(exp_data, columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_whitespace_lines(all_parsers):
    parser = all_parsers
    data = """

\t  \t\t
\t
A,B,C
\t    1,2.,4.
5.,NaN,10.0
"""
    expected = DataFrame([[1, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            """   A   B   C   D
a   1   2   3   4
b   1   2   3   4
c   1   2   3   4
""",
            DataFrame(
                [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                columns=["A", "B", "C", "D"],
                index=["a", "b", "c"],
            ),
        ),
        (
            "    a b c\n1 2 3 \n4 5  6\n 7 8 9",
            DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]),
        ),
    ],
)
def test_whitespace_regex_separator(all_parsers, data, expected):
    # see gh-6607
    parser = all_parsers
    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+")
        return

    result = parser.read_csv(StringIO(data), sep=r"\s+")
    tm.assert_frame_equal(result, expected)


def test_sub_character(all_parsers, csv_dir_path):
    # see gh-16893
    filename = os.path.join(csv_dir_path, "sub_char.csv")
    expected = DataFrame([[1, 2, 3]], columns=["a", "\x1ab", "c"])

    parser = all_parsers
    result = parser.read_csv(filename)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("filename", ["sé-es-vé.csv", "ru-sй.csv", "中文文件名.csv"])
def test_filename_with_special_chars(all_parsers, filename):
    # see gh-15086.
    parser = all_parsers
    df = DataFrame({"a": [1, 2, 3]})

    with tm.ensure_clean(filename) as path:
        df.to_csv(path, index=False)

        result = parser.read_csv(path)
        tm.assert_frame_equal(result, df)


def test_read_table_same_signature_as_read_csv(all_parsers):
    # GH-34976
    parser = all_parsers

    table_sign = signature(parser.read_table)
    csv_sign = signature(parser.read_csv)

    assert table_sign.parameters.keys() == csv_sign.parameters.keys()
    assert table_sign.return_annotation == csv_sign.return_annotation

    for key, csv_param in csv_sign.parameters.items():
        table_param = table_sign.parameters[key]
        if key == "sep":
            assert csv_param.default == ","
            assert table_param.default == "\t"
            assert table_param.annotation == csv_param.annotation
            assert table_param.kind == csv_param.kind
            continue

        assert table_param == csv_param


def test_read_table_equivalency_to_read_csv(all_parsers):
    # see gh-21948
    # As of 0.25.0, read_table is undeprecated
    parser = all_parsers
    data = "a\tb\n1\t2\n3\t4"
    expected = parser.read_csv(StringIO(data), sep="\t")
    result = parser.read_table(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("read_func", ["read_csv", "read_table"])
def test_read_csv_and_table_sys_setprofile(all_parsers, read_func):
    # GH#41069
    parser = all_parsers
    data = "a b\n0 1"

    sys.setprofile(lambda *a, **k: None)
    result = getattr(parser, read_func)(StringIO(data))
    sys.setprofile(None)

    expected = DataFrame({"a b": ["0 1"]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_first_row_bom(all_parsers):
    # see gh-26545
    parser = all_parsers
    data = '''\ufeff"Head1"\t"Head2"\t"Head3"'''

    result = parser.read_csv(StringIO(data), delimiter="\t")
    expected = DataFrame(columns=["Head1", "Head2", "Head3"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_first_row_bom_unquoted(all_parsers):
    # see gh-36343
    parser = all_parsers
    data = """\ufeffHead1\tHead2\tHead3"""

    result = parser.read_csv(StringIO(data), delimiter="\t")
    expected = DataFrame(columns=["Head1", "Head2", "Head3"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers, nrows):
    # GH 28071
    ref = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [1, 2], [np.nan, np.nan], [3, 4]],
        columns=list("ab"),
    )
    csv = "\nheader\n\na,b\n\n\n1,2\n\n3,4"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False
            )
        return

    df = parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
    tm.assert_frame_equal(df, ref[:nrows])


@skip_pyarrow
def test_no_header_two_extra_columns(all_parsers):
    # GH 26218
    column_names = ["one", "two", "three"]
    ref = DataFrame([["foo", "bar", "baz"]], columns=column_names)
    stream = StringIO("foo,bar,baz,bam,blah")
    parser = all_parsers
    df = parser.read_csv_check_warnings(
        ParserWarning,
        "Length of header or names does not match length of data. "
        "This leads to a loss of data with index_col=False.",
        stream,
        header=None,
        names=column_names,
        index_col=False,
    )
    tm.assert_frame_equal(df, ref)


def test_read_csv_names_not_accepting_sets(all_parsers):
    # GH 34946
    data = """\
    1,2,3
    4,5,6\n"""
    parser = all_parsers
    with pytest.raises(ValueError, match="Names should be an ordered collection."):
        parser.read_csv(StringIO(data), names=set("QAZ"))


def test_read_table_delim_whitespace_default_sep(all_parsers):
    # GH: 35958
    f = StringIO("a  b  c\n1 -2 -3\n4  5   6")
    parser = all_parsers

    depr_msg = "The 'delim_whitespace' keyword in pd.read_table is deprecated"

    if parser.engine == "pyarrow":
        msg = "The 'delim_whitespace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                FutureWarning, match=depr_msg, check_stacklevel=False
            ):
                parser.read_table(f, delim_whitespace=True)
        return
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_table(f, delim_whitespace=True)
    expected = DataFrame({"a": [1, 4], "b": [-2, 5], "c": [-3, 6]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("delimiter", [",", "\t"])
def test_read_csv_delim_whitespace_non_default_sep(all_parsers, delimiter):
    # GH: 35958
    f = StringIO("a  b  c\n1 -2 -3\n4  5   6")
    parser = all_parsers
    msg = (
        "Specified a delimiter with both sep and "
        "delim_whitespace=True; you can only specify one."
    )
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(f, delim_whitespace=True, sep=delimiter)

        with pytest.raises(ValueError, match=msg):
            parser.read_csv(f, delim_whitespace=True, delimiter=delimiter)


def test_read_csv_delimiter_and_sep_no_default(all_parsers):
    # GH#39823
    f = StringIO("a,b\n1,2")
    parser = all_parsers
    msg = "Specified a sep and a delimiter; you can only specify one."
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(f, sep=" ", delimiter=".")


@pytest.mark.parametrize("kwargs", [{"delimiter": "\n"}, {"sep": "\n"}])
def test_read_csv_line_break_as_separator(kwargs, all_parsers):
    # GH#43528
    parser = all_parsers
    data = """a,b,c
1,2,3
    """
    msg = (
        r"Specified \\n as separator or delimiter. This forces the python engine "
        r"which does not accept a line terminator. Hence it is not allowed to use "
        r"the line terminator as separator."
    )
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


@pytest.mark.parametrize("delimiter", [",", "\t"])
def test_read_table_delim_whitespace_non_default_sep(all_parsers, delimiter):
    # GH: 35958
    f = StringIO("a  b  c\n1 -2 -3\n4  5   6")
    parser = all_parsers
    msg = (
        "Specified a delimiter with both sep and "
        "delim_whitespace=True; you can only specify one."
    )
    depr_msg = "The 'delim_whitespace' keyword in pd.read_table is deprecated"
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        with pytest.raises(ValueError, match=msg):
            parser.read_table(f, delim_whitespace=True, sep=delimiter)

        with pytest.raises(ValueError, match=msg):
            parser.read_table(f, delim_whitespace=True, delimiter=delimiter)


@skip_pyarrow
def test_dict_keys_as_names(all_parsers):
    # GH: 36928
    data = "1,2"

    keys = {"a": int, "b": int}.keys()
    parser = all_parsers

    result = parser.read_csv(StringIO(data), names=keys)
    expected = DataFrame({"a": [1], "b": [2]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(using_string_dtype() and HAS_PYARROW, reason="TODO(infer_string)")
@xfail_pyarrow  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 0
def test_encoding_surrogatepass(all_parsers):
    # GH39017
    parser = all_parsers
    content = b"\xed\xbd\xbf"
    decoded = content.decode("utf-8", errors="surrogatepass")
    expected = DataFrame({decoded: [decoded]}, index=[decoded * 2])
    expected.index.name = decoded * 2

    with tm.ensure_clean() as path:
        Path(path).write_bytes(
            content * 2 + b"," + content + b"\n" + content * 2 + b"," + content
        )
        df = parser.read_csv(path, encoding_errors="surrogatepass", index_col=0)
        tm.assert_frame_equal(df, expected)
        with pytest.raises(UnicodeDecodeError, match="'utf-8' codec can't decode byte"):
            parser.read_csv(path)


def test_malformed_second_line(all_parsers):
    # see GH14782
    parser = all_parsers
    data = "\na\nb\n"
    result = parser.read_csv(StringIO(data), skip_blank_lines=False, header=1)
    expected = DataFrame({"a": ["b"]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_short_single_line(all_parsers):
    # GH 47566
    parser = all_parsers
    columns = ["a", "b", "c"]
    data = "1,2"
    result = parser.read_csv(StringIO(data), header=None, names=columns)
    expected = DataFrame({"a": [1], "b": [2], "c": [np.nan]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Length mismatch: Expected axis has 2 elements
def test_short_multi_line(all_parsers):
    # GH 47566
    parser = all_parsers
    columns = ["a", "b", "c"]
    data = "1,2\n1,2"
    result = parser.read_csv(StringIO(data), header=None, names=columns)
    expected = DataFrame({"a": [1, 1], "b": [2, 2], "c": [np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_read_seek(all_parsers):
    # GH48646
    parser = all_parsers
    prefix = "### DATA\n"
    content = "nkey,value\ntables,rectangular\n"
    with tm.ensure_clean() as path:
        Path(path).write_text(prefix + content, encoding="utf-8")
        with open(path, encoding="utf-8") as file:
            file.readline()
            actual = parser.read_csv(file)
        expected = parser.read_csv(StringIO(content))
    tm.assert_frame_equal(actual, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tools\test_to_numeric.py ===
import decimal

import numpy as np
from numpy import iinfo
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    Series,
    option_context,
    to_numeric,
)
import pandas._testing as tm


@pytest.fixture(params=[None, "ignore", "raise", "coerce"])
def errors(request):
    return request.param


@pytest.fixture(params=[True, False])
def signed(request):
    return request.param


@pytest.fixture(params=[lambda x: x, str], ids=["identity", "str"])
def transform(request):
    return request.param


@pytest.fixture(params=[47393996303418497800, 100000000000000000000])
def large_val(request):
    return request.param


@pytest.fixture(params=[True, False])
def multiple_elts(request):
    return request.param


@pytest.fixture(
    params=[
        (lambda x: Index(x, name="idx"), tm.assert_index_equal),
        (lambda x: Series(x, name="ser"), tm.assert_series_equal),
        (lambda x: np.array(Index(x).values), tm.assert_numpy_array_equal),
    ]
)
def transform_assert_equal(request):
    return request.param


@pytest.mark.parametrize(
    "input_kwargs,result_kwargs",
    [
        ({}, {"dtype": np.int64}),
        ({"errors": "coerce", "downcast": "integer"}, {"dtype": np.int8}),
    ],
)
def test_empty(input_kwargs, result_kwargs):
    # see gh-16302
    ser = Series([], dtype=object)
    result = to_numeric(ser, **input_kwargs)

    expected = Series([], **result_kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("last_val", ["7", 7])
def test_series(last_val, infer_string):
    with option_context("future.infer_string", infer_string):
        ser = Series(["1", "-3.14", last_val])
        result = to_numeric(ser)

    expected = Series([1, -3.14, 7])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [1, 3, 4, 5],
        [1.0, 3.0, 4.0, 5.0],
        # Bool is regarded as numeric.
        [True, False, True, True],
    ],
)
def test_series_numeric(data):
    ser = Series(data, index=list("ABCD"), name="EFG")

    result = to_numeric(ser)
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize(
    "data,msg",
    [
        ([1, -3.14, "apple"], 'Unable to parse string "apple" at position 2'),
        (
            ["orange", 1, -3.14, "apple"],
            'Unable to parse string "orange" at position 0',
        ),
    ],
)
def test_error(data, msg):
    ser = Series(data)

    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, errors="raise")


@pytest.mark.parametrize(
    "errors,exp_data", [("ignore", [1, -3.14, "apple"]), ("coerce", [1, -3.14, np.nan])]
)
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_ignore_error(errors, exp_data):
    ser = Series([1, -3.14, "apple"])
    result = to_numeric(ser, errors=errors)

    expected = Series(exp_data)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "errors,exp",
    [
        ("raise", 'Unable to parse string "apple" at position 2'),
        ("ignore", [True, False, "apple"]),
        # Coerces to float.
        ("coerce", [1.0, 0.0, np.nan]),
    ],
)
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_bool_handling(errors, exp):
    ser = Series([True, False, "apple"])

    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        expected = Series(exp)

        tm.assert_series_equal(result, expected)


def test_list():
    ser = ["1", "-3.14", "7"]
    res = to_numeric(ser)

    expected = np.array([1, -3.14, 7])
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "data,arr_kwargs",
    [
        ([1, 3, 4, 5], {"dtype": np.int64}),
        ([1.0, 3.0, 4.0, 5.0], {}),
        # Boolean is regarded as numeric.
        ([True, False, True, True], {}),
    ],
)
def test_list_numeric(data, arr_kwargs):
    result = to_numeric(data)
    expected = np.array(data, **arr_kwargs)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"dtype": "O"}, {}])
def test_numeric(kwargs):
    data = [1, -3.14, 7]

    ser = Series(data, **kwargs)
    result = to_numeric(ser)

    expected = Series(data)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "columns",
    [
        # One column.
        "a",
        # Multiple columns.
        ["a", "b"],
    ],
)
def test_numeric_df_columns(columns):
    # see gh-14827
    df = DataFrame(
        {
            "a": [1.2, decimal.Decimal(3.14), decimal.Decimal("infinity"), "0.1"],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )

    expected = DataFrame({"a": [1.2, 3.14, np.inf, 0.1], "b": [1.0, 2.0, 3.0, 4.0]})

    df_copy = df.copy()
    df_copy[columns] = df_copy[columns].apply(to_numeric)

    tm.assert_frame_equal(df_copy, expected)


@pytest.mark.parametrize(
    "data,exp_data",
    [
        (
            [[decimal.Decimal(3.14), 1.0], decimal.Decimal(1.6), 0.1],
            [[3.14, 1.0], 1.6, 0.1],
        ),
        ([np.array([decimal.Decimal(3.14), 1.0]), 0.1], [[3.14, 1.0], 0.1]),
    ],
)
def test_numeric_embedded_arr_likes(data, exp_data):
    # Test to_numeric with embedded lists and arrays
    df = DataFrame({"a": data})
    df["a"] = df["a"].apply(to_numeric)

    expected = DataFrame({"a": exp_data})
    tm.assert_frame_equal(df, expected)


def test_all_nan():
    ser = Series(["a", "b", "c"])
    result = to_numeric(ser, errors="coerce")

    expected = Series([np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_type_check(errors):
    # see gh-11776
    df = DataFrame({"a": [1, -3.14, 7], "b": ["4", "5", "6"]})
    kwargs = {"errors": errors} if errors is not None else {}
    with pytest.raises(TypeError, match="1-d array"):
        to_numeric(df, **kwargs)


@pytest.mark.parametrize("val", [1, 1.1, 20001])
def test_scalar(val, signed, transform):
    val = -val if signed else val
    assert to_numeric(transform(val)) == float(val)


@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_really_large_scalar(large_val, signed, transform, errors):
    # see gh-24910
    kwargs = {"errors": errors} if errors is not None else {}
    val = -large_val if signed else large_val

    val = transform(val)
    val_is_string = isinstance(val, str)

    if val_is_string and errors in (None, "raise"):
        msg = "Integer out of range. at position 0"
        with pytest.raises(ValueError, match=msg):
            to_numeric(val, **kwargs)
    else:
        expected = float(val) if (errors == "coerce" and val_is_string) else val
        tm.assert_almost_equal(to_numeric(val, **kwargs), expected)


@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_really_large_in_arr(large_val, signed, transform, multiple_elts, errors):
    # see gh-24910
    kwargs = {"errors": errors} if errors is not None else {}
    val = -large_val if signed else large_val
    val = transform(val)

    extra_elt = "string"
    arr = [val] + multiple_elts * [extra_elt]

    val_is_string = isinstance(val, str)
    coercing = errors == "coerce"

    if errors in (None, "raise") and (val_is_string or multiple_elts):
        if val_is_string:
            msg = "Integer out of range. at position 0"
        else:
            msg = 'Unable to parse string "string" at position 1'

        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result = to_numeric(arr, **kwargs)

        exp_val = float(val) if (coercing and val_is_string) else val
        expected = [exp_val]

        if multiple_elts:
            if coercing:
                expected.append(np.nan)
                exp_dtype = float
            else:
                expected.append(extra_elt)
                exp_dtype = object
        else:
            exp_dtype = float if isinstance(exp_val, (int, float)) else object

        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_really_large_in_arr_consistent(large_val, signed, multiple_elts, errors):
    # see gh-24910
    #
    # Even if we discover that we have to hold float, does not mean
    # we should be lenient on subsequent elements that fail to be integer.
    kwargs = {"errors": errors} if errors is not None else {}
    arr = [str(-large_val if signed else large_val)]

    if multiple_elts:
        arr.insert(0, large_val)

    if errors in (None, "raise"):
        index = int(multiple_elts)
        msg = f"Integer out of range. at position {index}"

        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result = to_numeric(arr, **kwargs)

        if errors == "coerce":
            expected = [float(i) for i in arr]
            exp_dtype = float
        else:
            expected = arr
            exp_dtype = object

        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


@pytest.mark.parametrize(
    "errors,checker",
    [
        ("raise", 'Unable to parse string "fail" at position 0'),
        ("ignore", lambda x: x == "fail"),
        ("coerce", lambda x: np.isnan(x)),
    ],
)
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_scalar_fail(errors, checker):
    scalar = "fail"

    if isinstance(checker, str):
        with pytest.raises(ValueError, match=checker):
            to_numeric(scalar, errors=errors)
    else:
        assert checker(to_numeric(scalar, errors=errors))


@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
def test_numeric_dtypes(data, transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    data = transform(data)

    result = to_numeric(data)
    assert_equal(result, data)


@pytest.mark.parametrize(
    "data,exp",
    [
        (["1", "2", "3"], np.array([1, 2, 3], dtype="int64")),
        (["1.5", "2.7", "3.4"], np.array([1.5, 2.7, 3.4])),
    ],
)
def test_str(data, exp, transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    result = to_numeric(transform(data))

    expected = transform(exp)
    assert_equal(result, expected)


def test_datetime_like(tz_naive_fixture, transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    idx = pd.date_range("20130101", periods=3, tz=tz_naive_fixture)

    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)


def test_timedelta(transform_assert_equal):
    transform, assert_equal = transform_assert_equal
    idx = pd.timedelta_range("1 days", periods=3, freq="D")

    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)


def test_period(request, transform_assert_equal):
    transform, assert_equal = transform_assert_equal

    idx = pd.period_range("2011-01", periods=3, freq="M", name="")
    inp = transform(idx)

    if not isinstance(inp, Index):
        request.applymarker(
            pytest.mark.xfail(reason="Missing PeriodDtype support in to_numeric")
        )
    result = to_numeric(inp)
    expected = transform(idx.asi8)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "errors,expected",
    [
        ("raise", "Invalid object type at position 0"),
        ("ignore", Series([[10.0, 2], 1.0, "apple"])),
        ("coerce", Series([np.nan, 1.0, np.nan])),
    ],
)
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_non_hashable(errors, expected):
    # see gh-13324
    ser = Series([[10.0, 2], 1.0, "apple"])

    if isinstance(expected, str):
        with pytest.raises(TypeError, match=expected):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, expected)


def test_downcast_invalid_cast():
    # see gh-13352
    data = ["1", 2, 3]
    invalid_downcast = "unsigned-integer"
    msg = "invalid downcasting method provided"

    with pytest.raises(ValueError, match=msg):
        to_numeric(data, downcast=invalid_downcast)


def test_errors_invalid_value():
    # see gh-26466
    data = ["1", 2, 3]
    invalid_error_value = "invalid"
    msg = "invalid error value specified"

    with pytest.raises(ValueError, match=msg):
        to_numeric(data, errors=invalid_error_value)


@pytest.mark.parametrize(
    "data",
    [
        ["1", 2, 3],
        [1, 2, 3],
        np.array(["1970-01-02", "1970-01-03", "1970-01-04"], dtype="datetime64[D]"),
    ],
)
@pytest.mark.parametrize(
    "kwargs,exp_dtype",
    [
        # Basic function tests.
        ({}, np.int64),
        ({"downcast": None}, np.int64),
        # Support below np.float32 is rare and far between.
        ({"downcast": "float"}, np.dtype(np.float32).char),
        # Basic dtype support.
        ({"downcast": "unsigned"}, np.dtype(np.typecodes["UnsignedInteger"][0])),
    ],
)
def test_downcast_basic(data, kwargs, exp_dtype):
    # see gh-13352
    result = to_numeric(data, **kwargs)
    expected = np.array([1, 2, 3], dtype=exp_dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("signed_downcast", ["integer", "signed"])
@pytest.mark.parametrize(
    "data",
    [
        ["1", 2, 3],
        [1, 2, 3],
        np.array(["1970-01-02", "1970-01-03", "1970-01-04"], dtype="datetime64[D]"),
    ],
)
def test_signed_downcast(data, signed_downcast):
    # see gh-13352
    smallest_int_dtype = np.dtype(np.typecodes["Integer"][0])
    expected = np.array([1, 2, 3], dtype=smallest_int_dtype)

    res = to_numeric(data, downcast=signed_downcast)
    tm.assert_numpy_array_equal(res, expected)


def test_ignore_downcast_invalid_data():
    # If we can't successfully cast the given
    # data to a numeric dtype, do not bother
    # with the downcast parameter.
    data = ["foo", 2, 3]
    expected = np.array(data, dtype=object)

    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = to_numeric(data, errors="ignore", downcast="unsigned")
    tm.assert_numpy_array_equal(res, expected)


def test_ignore_downcast_neg_to_unsigned():
    # Cannot cast to an unsigned integer
    # because we have a negative number.
    data = ["-1", 2, 3]
    expected = np.array([-1, 2, 3], dtype=np.int64)

    res = to_numeric(data, downcast="unsigned")
    tm.assert_numpy_array_equal(res, expected)


# Warning in 32 bit platforms
@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.parametrize("downcast", ["integer", "signed", "unsigned"])
@pytest.mark.parametrize(
    "data,expected",
    [
        (["1.1", 2, 3], np.array([1.1, 2, 3], dtype=np.float64)),
        (
            [10000.0, 20000, 3000, 40000.36, 50000, 50000.00],
            np.array(
                [10000.0, 20000, 3000, 40000.36, 50000, 50000.00], dtype=np.float64
            ),
        ),
    ],
)
def test_ignore_downcast_cannot_convert_float(data, expected, downcast):
    # Cannot cast to an integer (signed or unsigned)
    # because we have a float number.
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "downcast,expected_dtype",
    [("integer", np.int16), ("signed", np.int16), ("unsigned", np.uint16)],
)
def test_downcast_not8bit(downcast, expected_dtype):
    # the smallest integer dtype need not be np.(u)int8
    data = ["256", 257, 258]

    expected = np.array([256, 257, 258], dtype=expected_dtype)
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "dtype,downcast,min_max",
    [
        ("int8", "integer", [iinfo(np.int8).min, iinfo(np.int8).max]),
        ("int16", "integer", [iinfo(np.int16).min, iinfo(np.int16).max]),
        ("int32", "integer", [iinfo(np.int32).min, iinfo(np.int32).max]),
        ("int64", "integer", [iinfo(np.int64).min, iinfo(np.int64).max]),
        ("uint8", "unsigned", [iinfo(np.uint8).min, iinfo(np.uint8).max]),
        ("uint16", "unsigned", [iinfo(np.uint16).min, iinfo(np.uint16).max]),
        ("uint32", "unsigned", [iinfo(np.uint32).min, iinfo(np.uint32).max]),
        ("uint64", "unsigned", [iinfo(np.uint64).min, iinfo(np.uint64).max]),
        ("int16", "integer", [iinfo(np.int8).min, iinfo(np.int8).max + 1]),
        ("int32", "integer", [iinfo(np.int16).min, iinfo(np.int16).max + 1]),
        ("int64", "integer", [iinfo(np.int32).min, iinfo(np.int32).max + 1]),
        ("int16", "integer", [iinfo(np.int8).min - 1, iinfo(np.int16).max]),
        ("int32", "integer", [iinfo(np.int16).min - 1, iinfo(np.int32).max]),
        ("int64", "integer", [iinfo(np.int32).min - 1, iinfo(np.int64).max]),
        ("uint16", "unsigned", [iinfo(np.uint8).min, iinfo(np.uint8).max + 1]),
        ("uint32", "unsigned", [iinfo(np.uint16).min, iinfo(np.uint16).max + 1]),
        ("uint64", "unsigned", [iinfo(np.uint32).min, iinfo(np.uint32).max + 1]),
    ],
)
def test_downcast_limits(dtype, downcast, min_max):
    # see gh-14404: test the limits of each downcast.
    series = to_numeric(Series(min_max), downcast=downcast)
    assert series.dtype == dtype


def test_downcast_float64_to_float32():
    # GH-43693: Check float64 preservation when >= 16,777,217
    series = Series([16777217.0, np.finfo(np.float64).max, np.nan], dtype=np.float64)
    result = to_numeric(series, downcast="float")

    assert series.dtype == result.dtype


@pytest.mark.parametrize(
    "ser,expected",
    [
        (
            Series([0, 9223372036854775808]),
            Series([0, 9223372036854775808], dtype=np.uint64),
        )
    ],
)
def test_downcast_uint64(ser, expected):
    # see gh-14422:
    # BUG: to_numeric doesn't work uint64 numbers

    result = to_numeric(ser, downcast="unsigned")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data,exp_data",
    [
        (
            [200, 300, "", "NaN", 30000000000000000000],
            [200, 300, np.nan, np.nan, 30000000000000000000],
        ),
        (
            ["12345678901234567890", "1234567890", "ITEM"],
            [12345678901234567890, 1234567890, np.nan],
        ),
    ],
)
def test_coerce_uint64_conflict(data, exp_data):
    # see gh-17007 and gh-17125
    #
    # Still returns float despite the uint64-nan conflict,
    # which would normally force the casting to object.
    result = to_numeric(Series(data), errors="coerce")
    expected = Series(exp_data, dtype=float)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "errors,exp",
    [
        ("ignore", Series(["12345678901234567890", "1234567890", "ITEM"])),
        ("raise", "Unable to parse string"),
    ],
)
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_non_coerce_uint64_conflict(errors, exp):
    # see gh-17007 and gh-17125
    #
    # For completeness.
    ser = Series(["12345678901234567890", "1234567890", "ITEM"])

    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, ser)


@pytest.mark.parametrize("dc1", ["integer", "float", "unsigned"])
@pytest.mark.parametrize("dc2", ["integer", "float", "unsigned"])
def test_downcast_empty(dc1, dc2):
    # GH32493

    tm.assert_numpy_array_equal(
        to_numeric([], downcast=dc1),
        to_numeric([], downcast=dc2),
        check_dtype=False,
    )


def test_failure_to_convert_uint64_string_to_NaN():
    # GH 32394
    result = to_numeric("uint64", errors="coerce")
    assert np.isnan(result)

    ser = Series([32, 64, np.nan])
    result = to_numeric(Series(["32", "64", "uint64"]), errors="coerce")
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize(
    "strrep",
    [
        "243.164",
        "245.968",
        "249.585",
        "259.745",
        "265.742",
        "272.567",
        "279.196",
        "280.366",
        "275.034",
        "271.351",
        "272.889",
        "270.627",
        "280.828",
        "290.383",
        "308.153",
        "319.945",
        "336.0",
        "344.09",
        "351.385",
        "356.178",
        "359.82",
        "361.03",
        "367.701",
        "380.812",
        "387.98",
        "391.749",
        "391.171",
        "385.97",
        "385.345",
        "386.121",
        "390.996",
        "399.734",
        "413.073",
        "421.532",
        "430.221",
        "437.092",
        "439.746",
        "446.01",
        "451.191",
        "460.463",
        "469.779",
        "472.025",
        "479.49",
        "474.864",
        "467.54",
        "471.978",
    ],
)
def test_precision_float_conversion(strrep):
    # GH 31364
    result = to_numeric(strrep)

    assert result == float(strrep)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["1", "2", None], Series([1, 2, np.nan], dtype="Int64")),
        (["1", "2", "3"], Series([1, 2, 3], dtype="Int64")),
        (["1", "2", 3], Series([1, 2, 3], dtype="Int64")),
        (["1", "2", 3.5], Series([1, 2, 3.5], dtype="Float64")),
        (["1", None, 3.5], Series([1, np.nan, 3.5], dtype="Float64")),
        (["1", "2", "3.5"], Series([1, 2, 3.5], dtype="Float64")),
    ],
)
def test_to_numeric_from_nullable_string(values, nullable_string_dtype, expected):
    # https://github.com/pandas-dev/pandas/issues/37262
    s = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(s)
    tm.assert_series_equal(result, expected)


def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype):
    # GH#52146
    values = ["a", "1"]
    ser = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(ser, errors="coerce")
    expected = Series([pd.NA, 1], dtype="Int64")
    tm.assert_series_equal(result, expected)


def test_to_numeric_from_nullable_string_ignore(nullable_string_dtype):
    # GH#52146
    values = ["a", "1"]
    ser = Series(values, dtype=nullable_string_dtype)
    expected = ser.copy()
    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_numeric(ser, errors="ignore")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, input_dtype, downcast, expected_dtype",
    (
        ([1, 1], "Int64", "integer", "Int8"),
        ([1.0, pd.NA], "Float64", "integer", "Int8"),
        ([1.0, 1.1], "Float64", "integer", "Float64"),
        ([1, pd.NA], "Int64", "integer", "Int8"),
        ([450, 300], "Int64", "integer", "Int16"),
        ([1, 1], "Float64", "integer", "Int8"),
        ([np.iinfo(np.int64).max - 1, 1], "Int64", "integer", "Int64"),
        ([1, 1], "Int64", "signed", "Int8"),
        ([1.0, 1.0], "Float32", "signed", "Int8"),
        ([1.0, 1.1], "Float64", "signed", "Float64"),
        ([1, pd.NA], "Int64", "signed", "Int8"),
        ([450, -300], "Int64", "signed", "Int16"),
        ([np.iinfo(np.uint64).max - 1, 1], "UInt64", "signed", "UInt64"),
        ([1, 1], "Int64", "unsigned", "UInt8"),
        ([1.0, 1.0], "Float32", "unsigned", "UInt8"),
        ([1.0, 1.1], "Float64", "unsigned", "Float64"),
        ([1, pd.NA], "Int64", "unsigned", "UInt8"),
        ([450, -300], "Int64", "unsigned", "Int64"),
        ([-1, -1], "Int32", "unsigned", "Int32"),
        ([1, 1], "Float64", "float", "Float32"),
        ([1, 1.1], "Float64", "float", "Float32"),
        ([1, 1], "Float32", "float", "Float32"),
        ([1, 1.1], "Float32", "float", "Float32"),
    ),
)
def test_downcast_nullable_numeric(data, input_dtype, downcast, expected_dtype):
    arr = pd.array(data, dtype=input_dtype)
    result = to_numeric(arr, downcast=downcast)
    expected = pd.array(data, dtype=expected_dtype)
    tm.assert_extension_array_equal(result, expected)


def test_downcast_nullable_mask_is_copied():
    # GH38974

    arr = pd.array([1, 2, pd.NA], dtype="Int64")

    result = to_numeric(arr, downcast="integer")
    expected = pd.array([1, 2, pd.NA], dtype="Int8")
    tm.assert_extension_array_equal(result, expected)

    arr[1] = pd.NA  # should not modify result
    tm.assert_extension_array_equal(result, expected)


def test_to_numeric_scientific_notation():
    # GH 15898
    result = to_numeric("1.7e+308")
    expected = np.float64(1.7e308)
    assert result == expected


@pytest.mark.parametrize("val", [9876543210.0, 2.0**128])
def test_to_numeric_large_float_not_downcast_to_float_32(val):
    # GH 19729
    expected = Series([val])
    result = to_numeric(expected, downcast="float")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "val, dtype", [(1, "Int64"), (1.5, "Float64"), (True, "boolean")]
)
def test_to_numeric_dtype_backend(val, dtype):
    # GH#50505
    ser = Series([val], dtype=object)
    result = to_numeric(ser, dtype_backend="numpy_nullable")
    expected = Series([val], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "val, dtype",
    [
        (1, "Int64"),
        (1.5, "Float64"),
        (True, "boolean"),
        (1, "int64[pyarrow]"),
        (1.5, "float64[pyarrow]"),
        (True, "bool[pyarrow]"),
    ],
)
def test_to_numeric_dtype_backend_na(val, dtype):
    # GH#50505
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
        dtype_backend = "pyarrow"
    else:
        dtype_backend = "numpy_nullable"
    ser = Series([val, None], dtype=object)
    result = to_numeric(ser, dtype_backend=dtype_backend)
    expected = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "val, dtype, downcast",
    [
        (1, "Int8", "integer"),
        (1.5, "Float32", "float"),
        (1, "Int8", "signed"),
        (1, "int8[pyarrow]", "integer"),
        (1.5, "float[pyarrow]", "float"),
        (1, "int8[pyarrow]", "signed"),
    ],
)
def test_to_numeric_dtype_backend_downcasting(val, dtype, downcast):
    # GH#50505
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
        dtype_backend = "pyarrow"
    else:
        dtype_backend = "numpy_nullable"
    ser = Series([val, None], dtype=object)
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast=downcast)
    expected = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "smaller, dtype_backend",
    [["UInt8", "numpy_nullable"], ["uint8[pyarrow]", "pyarrow"]],
)
def test_to_numeric_dtype_backend_downcasting_uint(smaller, dtype_backend):
    # GH#50505
    if dtype_backend == "pyarrow":
        pytest.importorskip("pyarrow")
    ser = Series([1, pd.NA], dtype="UInt64")
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast="unsigned")
    expected = Series([1, pd.NA], dtype=smaller)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        "UInt64",
        "Float64",
        "boolean",
        "int64[pyarrow]",
        "uint64[pyarrow]",
        "float64[pyarrow]",
        "bool[pyarrow]",
    ],
)
def test_to_numeric_dtype_backend_already_nullable(dtype):
    # GH#50505
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    ser = Series([1, pd.NA], dtype=dtype)
    result = to_numeric(ser, dtype_backend="numpy_nullable")
    expected = Series([1, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_to_numeric_dtype_backend_error(dtype_backend):
    # GH#50505
    ser = Series(["a", "b", ""])
    expected = ser.copy()
    with pytest.raises(ValueError, match="Unable to parse string"):
        to_numeric(ser, dtype_backend=dtype_backend)

    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_numeric(ser, dtype_backend=dtype_backend, errors="ignore")
    tm.assert_series_equal(result, expected)

    result = to_numeric(ser, dtype_backend=dtype_backend, errors="coerce")
    if dtype_backend == "pyarrow":
        dtype = "double[pyarrow]"
    else:
        dtype = "Float64"
    expected = Series([np.nan, np.nan, np.nan], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_invalid_dtype_backend():
    ser = Series([1, 2, 3])
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, dtype_backend="numpy")


def test_coerce_pyarrow_backend():
    # GH 52588
    pa = pytest.importorskip("pyarrow")
    ser = Series(list("12x"), dtype=ArrowDtype(pa.string()))
    result = to_numeric(ser, errors="coerce", dtype_backend="pyarrow")
    expected = Series([1, 2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_quantile.py ===
import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm


@pytest.fixture(
    params=[["linear", "single"], ["nearest", "table"]], ids=lambda x: "-".join(x)
)
def interp_method(request):
    """(interpolation, method) arguments for quantile"""
    return request.param


class TestDataFrameQuantile:
    @pytest.mark.parametrize(
        "df,expected",
        [
            [
                DataFrame(
                    {
                        0: Series(pd.arrays.SparseArray([1, 2])),
                        1: Series(pd.arrays.SparseArray([3, 4])),
                    }
                ),
                Series([1.5, 3.5], name=0.5),
            ],
            [
                DataFrame(Series([0.0, None, 1.0, 2.0], dtype="Sparse[float]")),
                Series([1.0], name=0.5),
            ],
        ],
    )
    def test_quantile_sparse(self, df, expected):
        # GH#17198
        # GH#24600
        result = df.quantile()
        expected = expected.astype("Sparse[float]")
        tm.assert_series_equal(result, expected)

    def test_quantile(
        self, datetime_frame, interp_method, using_array_manager, request
    ):
        interpolation, method = interp_method
        df = datetime_frame
        result = df.quantile(
            0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series(
            [np.percentile(df[col], 10) for col in df.columns],
            index=df.columns,
            name=0.1,
        )
        if interpolation == "linear":
            # np.percentile values only comparable to linear interpolation
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            request.applymarker(
                pytest.mark.xfail(
                    using_array_manager, reason="Name set incorrectly for arraymanager"
                )
            )
            assert result.name == expected.name

        result = df.quantile(
            0.9, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series(
            [np.percentile(df.loc[date], 90) for date in df.index],
            index=df.index,
            name=0.9,
        )
        if interpolation == "linear":
            # np.percentile values only comparable to linear interpolation
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            request.applymarker(
                pytest.mark.xfail(
                    using_array_manager, reason="Name set incorrectly for arraymanager"
                )
            )
            assert result.name == expected.name

    def test_empty(self, interp_method):
        interpolation, method = interp_method
        q = DataFrame({"x": [], "y": []}).quantile(
            0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method
        )
        assert np.isnan(q["x"]) and np.isnan(q["y"])

    def test_non_numeric_exclusion(self, interp_method, request, using_array_manager):
        interpolation, method = interp_method
        df = DataFrame({"col1": ["A", "A", "B", "B"], "col2": [1, 2, 3, 4]})
        rs = df.quantile(
            0.5, numeric_only=True, interpolation=interpolation, method=method
        )
        xp = df.median(numeric_only=True).rename(0.5)
        if interpolation == "nearest":
            xp = (xp + 0.5).astype(np.int64)
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        tm.assert_series_equal(rs, xp)

    def test_axis(self, interp_method, request, using_array_manager):
        # axis
        interpolation, method = interp_method
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        tm.assert_series_equal(result, expected)

        result = df.quantile(
            [0.5, 0.75], axis=1, interpolation=interpolation, method=method
        )
        expected = DataFrame(
            {1: [1.5, 1.75], 2: [2.5, 2.75], 3: [3.5, 3.75]}, index=[0.5, 0.75]
        )
        if interpolation == "nearest":
            expected.iloc[0, :] -= 0.5
            expected.iloc[1, :] += 0.25
            expected = expected.astype(np.int64)
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_axis_numeric_only_true(self, interp_method, request, using_array_manager):
        # We may want to break API in the future to change this
        # so that we exclude non-numeric along the same axis
        # See GH #7312
        interpolation, method = interp_method
        df = DataFrame([[1, 2, 3], ["a", "b", 4]])
        result = df.quantile(
            0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series([3.0, 4.0], index=[0, 1], name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        tm.assert_series_equal(result, expected)

    def test_quantile_date_range(self, interp_method, request, using_array_manager):
        # GH 2460
        interpolation, method = interp_method
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        ser = Series(dti)
        df = DataFrame(ser)

        result = df.quantile(
            numeric_only=False, interpolation=interpolation, method=method
        )
        expected = Series(
            ["2016-01-02 00:00:00"], name=0.5, dtype="datetime64[ns, US/Pacific]"
        )
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))

        tm.assert_series_equal(result, expected)

    def test_quantile_axis_mixed(self, interp_method, request, using_array_manager):
        # mixed on axis=1
        interpolation, method = interp_method
        df = DataFrame(
            {
                "A": [1, 2, 3],
                "B": [2.0, 3.0, 4.0],
                "C": pd.date_range("20130101", periods=3),
                "D": ["foo", "bar", "baz"],
            }
        )
        result = df.quantile(
            0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series([1.5, 2.5, 3.5], name=0.5)
        if interpolation == "nearest":
            expected -= 0.5
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        tm.assert_series_equal(result, expected)

        # must raise
        msg = "'<' not supported between instances of 'Timestamp' and 'float'"
        with pytest.raises(TypeError, match=msg):
            df.quantile(0.5, axis=1, numeric_only=False)

    def test_quantile_axis_parameter(self, interp_method, request, using_array_manager):
        # GH 9543/9544
        interpolation, method = interp_method
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])

        result = df.quantile(0.5, axis=0, interpolation=interpolation, method=method)

        expected = Series([2.0, 3.0], index=["A", "B"], name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        expected = df.quantile(
            0.5, axis="index", interpolation=interpolation, method=method
        )
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)

        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        result = df.quantile(
            0.5, axis="columns", interpolation=interpolation, method=method
        )
        tm.assert_series_equal(result, expected)

        msg = "No axis named -1 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis=-1, interpolation=interpolation, method=method)
        msg = "No axis named column for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis="column")

    def test_quantile_interpolation(self):
        # see gh-10174

        # interpolation method other than default linear
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1, 2, 3], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)

        # cross-check interpolation=nearest results in original dtype
        exp = np.percentile(
            np.array([[1, 2, 3], [2, 3, 4]]),
            0.5,
            axis=0,
            method="nearest",
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="int64")
        tm.assert_series_equal(result, expected)

        # float
        df = DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1.0, 2.0, 3.0], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)
        exp = np.percentile(
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            0.5,
            axis=0,
            method="nearest",
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="float64")
        tm.assert_series_equal(result, expected)

        # axis
        result = df.quantile([0.5, 0.75], axis=1, interpolation="lower")
        expected = DataFrame(
            {1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}, index=[0.5, 0.75]
        )
        tm.assert_frame_equal(result, expected)

        # test degenerate case
        df = DataFrame({"x": [], "y": []})
        q = df.quantile(0.1, axis=0, interpolation="higher")
        assert np.isnan(q["x"]) and np.isnan(q["y"])

        # multi
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile([0.25, 0.5], interpolation="midpoint")

        # https://github.com/numpy/numpy/issues/7163
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        tm.assert_frame_equal(result, expected)

    def test_quantile_interpolation_datetime(self, datetime_frame):
        # see gh-10174

        # interpolation = linear (default case)
        df = datetime_frame
        q = df.quantile(0.1, axis=0, numeric_only=True, interpolation="linear")
        assert q["A"] == np.percentile(df["A"], 10)

    def test_quantile_interpolation_int(self, int_frame):
        # see gh-10174

        df = int_frame
        # interpolation = linear (default case)
        q = df.quantile(0.1)
        assert q["A"] == np.percentile(df["A"], 10)

        # test with and without interpolation keyword
        q1 = df.quantile(0.1, axis=0, interpolation="linear")
        assert q1["A"] == np.percentile(df["A"], 10)
        tm.assert_series_equal(q, q1)

    def test_quantile_multi(self, interp_method, request, using_array_manager):
        interpolation, method = interp_method
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile([0.25, 0.5], interpolation=interpolation, method=method)
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        tm.assert_frame_equal(result, expected)

    def test_quantile_multi_axis_1(self, interp_method, request, using_array_manager):
        interpolation, method = interp_method
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile(
            [0.25, 0.5], axis=1, interpolation=interpolation, method=method
        )
        expected = DataFrame(
            [[1.0, 2.0, 3.0]] * 2, index=[0.25, 0.5], columns=[0, 1, 2]
        )
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        tm.assert_frame_equal(result, expected)

    def test_quantile_multi_empty(self, interp_method):
        interpolation, method = interp_method
        result = DataFrame({"x": [], "y": []}).quantile(
            [0.1, 0.9], axis=0, interpolation=interpolation, method=method
        )
        expected = DataFrame(
            {"x": [np.nan, np.nan], "y": [np.nan, np.nan]}, index=[0.1, 0.9]
        )
        tm.assert_frame_equal(result, expected)

    def test_quantile_datetime(self, unit):
        dti = pd.to_datetime(["2010", "2011"]).as_unit(unit)
        df = DataFrame({"a": dti, "b": [0, 5]})

        # exclude datetime
        result = df.quantile(0.5, numeric_only=True)
        expected = Series([2.5], index=["b"], name=0.5)
        tm.assert_series_equal(result, expected)

        # datetime
        result = df.quantile(0.5, numeric_only=False)
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), 2.5], index=["a", "b"], name=0.5
        )
        tm.assert_series_equal(result, expected)

        # datetime w/ multi
        result = df.quantile([0.5], numeric_only=False)
        expected = DataFrame(
            {"a": Timestamp("2010-07-02 12:00:00").as_unit(unit), "b": 2.5},
            index=[0.5],
        )
        tm.assert_frame_equal(result, expected)

        # axis = 1
        df["c"] = pd.to_datetime(["2011", "2012"]).as_unit(unit)
        result = df[["a", "c"]].quantile(0.5, axis=1, numeric_only=False)
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), Timestamp("2011-07-02 12:00:00")],
            index=[0, 1],
            name=0.5,
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

        result = df[["a", "c"]].quantile([0.5], axis=1, numeric_only=False)
        expected = DataFrame(
            [[Timestamp("2010-07-02 12:00:00"), Timestamp("2011-07-02 12:00:00")]],
            index=[0.5],
            columns=[0, 1],
            dtype=f"M8[{unit}]",
        )
        tm.assert_frame_equal(result, expected)

        # empty when numeric_only=True
        result = df[["a", "c"]].quantile(0.5, numeric_only=True)
        expected = Series([], index=[], dtype=np.float64, name=0.5)
        tm.assert_series_equal(result, expected)

        result = df[["a", "c"]].quantile([0.5], numeric_only=True)
        expected = DataFrame(index=[0.5], columns=[])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[ns]",
            "datetime64[ns, US/Pacific]",
            "timedelta64[ns]",
            "Period[D]",
        ],
    )
    def test_quantile_dt64_empty(self, dtype, interp_method):
        # GH#41544
        interpolation, method = interp_method
        df = DataFrame(columns=["a", "b"], dtype=dtype)

        res = df.quantile(
            0.5, axis=1, numeric_only=False, interpolation=interpolation, method=method
        )
        expected = Series([], index=[], name=0.5, dtype=dtype)
        tm.assert_series_equal(res, expected)

        # no columns in result, so no dtype preservation
        res = df.quantile(
            [0.5],
            axis=1,
            numeric_only=False,
            interpolation=interpolation,
            method=method,
        )
        expected = DataFrame(index=[0.5], columns=[])
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize("invalid", [-1, 2, [0.5, -1], [0.5, 2]])
    def test_quantile_invalid(self, invalid, datetime_frame, interp_method):
        msg = "percentiles should all be in the interval \\[0, 1\\]"
        interpolation, method = interp_method
        with pytest.raises(ValueError, match=msg):
            datetime_frame.quantile(invalid, interpolation=interpolation, method=method)

    def test_quantile_box(self, interp_method, request, using_array_manager):
        interpolation, method = interp_method
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        df = DataFrame(
            {
                "A": [
                    Timestamp("2011-01-01"),
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-03"),
                ],
                "B": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "C": [
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                ],
            }
        )

        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )

        exp = Series(
            [
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timedelta("2 days"),
            ],
            name=0.5,
            index=["A", "B", "C"],
        )
        tm.assert_series_equal(res, exp)

        res = df.quantile(
            [0.5], numeric_only=False, interpolation=interpolation, method=method
        )
        exp = DataFrame(
            [
                [
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.Timedelta("2 days"),
                ]
            ],
            index=[0.5],
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(res, exp)

    def test_quantile_box_nat(self):
        # DatetimeLikeBlock may be consolidated and contain NaT in different loc
        df = DataFrame(
            {
                "A": [
                    Timestamp("2011-01-01"),
                    pd.NaT,
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-03"),
                ],
                "a": [
                    Timestamp("2011-01-01"),
                    Timestamp("2011-01-02"),
                    pd.NaT,
                    Timestamp("2011-01-03"),
                ],
                "B": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "b": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "C": [
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                    pd.NaT,
                ],
                "c": [
                    pd.NaT,
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                ],
            },
            columns=list("AaBbCc"),
        )

        res = df.quantile(0.5, numeric_only=False)
        exp = Series(
            [
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timedelta("2 days"),
                pd.Timedelta("2 days"),
            ],
            name=0.5,
            index=list("AaBbCc"),
        )
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5], numeric_only=False)
        exp = DataFrame(
            [
                [
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("2 days"),
                ]
            ],
            index=[0.5],
            columns=list("AaBbCc"),
        )
        tm.assert_frame_equal(res, exp)

    def test_quantile_nan(self, interp_method, request, using_array_manager):
        interpolation, method = interp_method
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        # GH 14357 - float block where some cols have missing values
        df = DataFrame({"a": np.arange(1, 6.0), "b": np.arange(1, 6.0)})
        df.iloc[-1, 1] = np.nan

        res = df.quantile(0.5, interpolation=interpolation, method=method)
        exp = Series(
            [3.0, 2.5 if interpolation == "linear" else 3.0], index=["a", "b"], name=0.5
        )
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5, 0.75], interpolation=interpolation, method=method)
        exp = DataFrame(
            {
                "a": [3.0, 4.0],
                "b": [2.5, 3.25] if interpolation == "linear" else [3.0, 4.0],
            },
            index=[0.5, 0.75],
        )
        tm.assert_frame_equal(res, exp)

        res = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        exp = Series(np.arange(1.0, 6.0), name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile(
            [0.5, 0.75], axis=1, interpolation=interpolation, method=method
        )
        exp = DataFrame([np.arange(1.0, 6.0)] * 2, index=[0.5, 0.75])
        if interpolation == "nearest":
            exp.iloc[1, -1] = np.nan
        tm.assert_frame_equal(res, exp)

        # full-nan column
        df["b"] = np.nan

        res = df.quantile(0.5, interpolation=interpolation, method=method)
        exp = Series([3.0, np.nan], index=["a", "b"], name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5, 0.75], interpolation=interpolation, method=method)
        exp = DataFrame({"a": [3.0, 4.0], "b": [np.nan, np.nan]}, index=[0.5, 0.75])
        tm.assert_frame_equal(res, exp)

    def test_quantile_nat(self, interp_method, request, using_array_manager, unit):
        interpolation, method = interp_method
        if method == "table" and using_array_manager:
            request.applymarker(pytest.mark.xfail(reason="Axis name incorrectly set."))
        # full NaT column
        df = DataFrame({"a": [pd.NaT, pd.NaT, pd.NaT]}, dtype=f"M8[{unit}]")

        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        exp = Series([pd.NaT], index=["a"], name=0.5, dtype=f"M8[{unit}]")
        tm.assert_series_equal(res, exp)

        res = df.quantile(
            [0.5], numeric_only=False, interpolation=interpolation, method=method
        )
        exp = DataFrame({"a": [pd.NaT]}, index=[0.5], dtype=f"M8[{unit}]")
        tm.assert_frame_equal(res, exp)

        # mixed non-null / full null column
        df = DataFrame(
            {
                "a": [
                    Timestamp("2012-01-01"),
                    Timestamp("2012-01-02"),
                    Timestamp("2012-01-03"),
                ],
                "b": [pd.NaT, pd.NaT, pd.NaT],
            },
            dtype=f"M8[{unit}]",
        )

        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        exp = Series(
            [Timestamp("2012-01-02"), pd.NaT],
            index=["a", "b"],
            name=0.5,
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(res, exp)

        res = df.quantile(
            [0.5], numeric_only=False, interpolation=interpolation, method=method
        )
        exp = DataFrame(
            [[Timestamp("2012-01-02"), pd.NaT]],
            index=[0.5],
            columns=["a", "b"],
            dtype=f"M8[{unit}]",
        )
        tm.assert_frame_equal(res, exp)

    def test_quantile_empty_no_rows_floats(self, interp_method):
        interpolation, method = interp_method

        df = DataFrame(columns=["a", "b"], dtype="float64")

        res = df.quantile(0.5, interpolation=interpolation, method=method)
        exp = Series([np.nan, np.nan], index=["a", "b"], name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5], interpolation=interpolation, method=method)
        exp = DataFrame([[np.nan, np.nan]], columns=["a", "b"], index=[0.5])
        tm.assert_frame_equal(res, exp)

        res = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        exp = Series([], index=[], dtype="float64", name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5], axis=1, interpolation=interpolation, method=method)
        exp = DataFrame(columns=[], index=[0.5])
        tm.assert_frame_equal(res, exp)

    def test_quantile_empty_no_rows_ints(self, interp_method):
        interpolation, method = interp_method
        df = DataFrame(columns=["a", "b"], dtype="int64")

        res = df.quantile(0.5, interpolation=interpolation, method=method)
        exp = Series([np.nan, np.nan], index=["a", "b"], name=0.5)
        tm.assert_series_equal(res, exp)

    def test_quantile_empty_no_rows_dt64(self, interp_method):
        interpolation, method = interp_method
        # datetimes
        df = DataFrame(columns=["a", "b"], dtype="datetime64[ns]")

        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        exp = Series(
            [pd.NaT, pd.NaT], index=["a", "b"], dtype="datetime64[ns]", name=0.5
        )
        tm.assert_series_equal(res, exp)

        # Mixed dt64/dt64tz
        df["a"] = df["a"].dt.tz_localize("US/Central")
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        exp = exp.astype(object)
        if interpolation == "nearest":
            # GH#18463 TODO: would we prefer NaTs here?
            msg = "The 'downcast' keyword in fillna is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                exp = exp.fillna(np.nan, downcast=False)
        tm.assert_series_equal(res, exp)

        # both dt64tz
        df["b"] = df["b"].dt.tz_localize("US/Central")
        res = df.quantile(
            0.5, numeric_only=False, interpolation=interpolation, method=method
        )
        exp = exp.astype(df["b"].dtype)
        tm.assert_series_equal(res, exp)

    def test_quantile_empty_no_columns(self, interp_method):
        # GH#23925 _get_numeric_data may drop all columns
        interpolation, method = interp_method
        df = DataFrame(pd.date_range("1/1/18", periods=5))
        df.columns.name = "captain tightpants"
        result = df.quantile(
            0.5, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series([], index=[], name=0.5, dtype=np.float64)
        expected.index.name = "captain tightpants"
        tm.assert_series_equal(result, expected)

        result = df.quantile(
            [0.5], numeric_only=True, interpolation=interpolation, method=method
        )
        expected = DataFrame([], index=[0.5], columns=[])
        expected.columns.name = "captain tightpants"
        tm.assert_frame_equal(result, expected)

    def test_quantile_item_cache(
        self, using_array_manager, interp_method, using_copy_on_write
    ):
        # previous behavior incorrect retained an invalid _item_cache entry
        interpolation, method = interp_method
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)), columns=["A", "B", "C"]
        )
        df["D"] = df["A"] * 2
        ser = df["A"]
        if not using_array_manager:
            assert len(df._mgr.blocks) == 2

        df.quantile(numeric_only=False, interpolation=interpolation, method=method)

        if using_copy_on_write:
            ser.iloc[0] = 99
            assert df.iloc[0, 0] == df["A"][0]
            assert df.iloc[0, 0] != 99
        else:
            ser.values[0] = 99
            assert df.iloc[0, 0] == df["A"][0]
            assert df.iloc[0, 0] == 99

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Invalid method: foo"):
            DataFrame(range(1)).quantile(0.5, method="foo")

    def test_table_invalid_interpolation(self):
        with pytest.raises(ValueError, match="Invalid interpolation: foo"):
            DataFrame(range(1)).quantile(0.5, method="table", interpolation="foo")


class TestQuantileExtensionDtype:
    # TODO: tests for axis=1?
    # TODO: empty case?

    @pytest.fixture(
        params=[
            pytest.param(
                pd.IntervalIndex.from_breaks(range(10)),
                marks=pytest.mark.xfail(reason="raises when trying to add Intervals"),
            ),
            pd.period_range("2016-01-01", periods=9, freq="D"),
            pd.date_range("2016-01-01", periods=9, tz="US/Pacific"),
            pd.timedelta_range("1 Day", periods=9),
            pd.array(np.arange(9), dtype="Int64"),
            pd.array(np.arange(9), dtype="Float64"),
        ],
        ids=lambda x: str(x.dtype),
    )
    def index(self, request):
        # NB: not actually an Index object
        idx = request.param
        idx.name = "A"
        return idx

    @pytest.fixture
    def obj(self, index, frame_or_series):
        # bc index is not always an Index (yet), we need to re-patch .name
        obj = frame_or_series(index).copy()

        if frame_or_series is Series:
            obj.name = "A"
        else:
            obj.columns = ["A"]
        return obj

    def compute_quantile(self, obj, qs):
        if isinstance(obj, Series):
            result = obj.quantile(qs)
        else:
            result = obj.quantile(qs, numeric_only=False)
        return result

    def test_quantile_ea(self, request, obj, index):
        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)

        exp_dtype = index.dtype
        if index.dtype == "Int64":
            # match non-nullable casting behavior
            exp_dtype = "Float64"

        # expected here assumes len(index) == 9
        expected = Series(
            [index[4], index[0], index[-1]], dtype=exp_dtype, index=qs, name="A"
        )
        expected = type(obj)(expected)

        tm.assert_equal(result, expected)

    def test_quantile_ea_with_na(self, obj, index):
        obj.iloc[0] = index._na_value
        obj.iloc[-1] = index._na_value

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)

        # expected here assumes len(index) == 9
        expected = Series(
            [index[4], index[1], index[-2]], dtype=index.dtype, index=qs, name="A"
        )
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_all_na(self, request, obj, index):
        obj.iloc[:] = index._na_value
        # Check dtypes were preserved; this was once a problem see GH#39763
        assert np.all(obj.dtypes == index.dtype)

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)

        expected = index.take([-1, -1, -1], allow_fill=True, fill_value=index._na_value)
        expected = Series(expected, index=qs, name="A")
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_scalar(self, request, obj, index):
        # scalar qs

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = 0.5
        result = self.compute_quantile(obj, qs)

        exp_dtype = index.dtype
        if index.dtype == "Int64":
            exp_dtype = "Float64"

        expected = Series({"A": index[4]}, dtype=exp_dtype, name=0.5)
        if isinstance(obj, Series):
            expected = expected["A"]
            assert result == expected
        else:
            tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)", strict=False)
    @pytest.mark.parametrize(
        "dtype, expected_data, expected_index, axis",
        [
            ["float64", [], [], 1],
            ["int64", [], [], 1],
            ["float64", [np.nan, np.nan], ["a", "b"], 0],
            ["int64", [np.nan, np.nan], ["a", "b"], 0],
        ],
    )
    def test_empty_numeric(self, dtype, expected_data, expected_index, axis):
        # GH 14564
        df = DataFrame(columns=["a", "b"], dtype=dtype)
        result = df.quantile(0.5, axis=axis)
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype="float64"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)", strict=False)
    @pytest.mark.parametrize(
        "dtype, expected_data, expected_index, axis, expected_dtype",
        [
            ["datetime64[ns]", [], [], 1, "datetime64[ns]"],
            ["datetime64[ns]", [pd.NaT, pd.NaT], ["a", "b"], 0, "datetime64[ns]"],
        ],
    )
    def test_empty_datelike(
        self, dtype, expected_data, expected_index, axis, expected_dtype
    ):
        # GH 14564
        df = DataFrame(columns=["a", "b"], dtype=dtype)
        result = df.quantile(0.5, axis=axis, numeric_only=False)
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype=expected_dtype
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)", strict=False)
    @pytest.mark.parametrize(
        "expected_data, expected_index, axis",
        [
            [[np.nan, np.nan], range(2), 1],
            [[], [], 0],
        ],
    )
    def test_datelike_numeric_only(self, expected_data, expected_index, axis):
        # GH 14564
        df = DataFrame(
            {
                "a": pd.to_datetime(["2010", "2011"]),
                "b": [0, 5],
                "c": pd.to_datetime(["2011", "2012"]),
            }
        )
        result = df[["a", "c"]].quantile(0.5, axis=axis, numeric_only=True)
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype=np.float64
        )
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_hist_method.py ===
""" Test cases for .hist method """
import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_colors,
    _check_legend_labels,
    _check_patches_all_filled,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    get_x_axis,
    get_y_axis,
)

mpl = pytest.importorskip("matplotlib")


@pytest.fixture
def ts():
    return Series(
        np.arange(30, dtype=np.float64),
        index=date_range("2020-01-01", periods=30, freq="B"),
        name="ts",
    )


class TestSeriesPlots:
    @pytest.mark.parametrize("kwargs", [{}, {"grid": False}, {"figsize": (8, 10)}])
    def test_hist_legacy_kwargs(self, ts, kwargs):
        _check_plot_works(ts.hist, **kwargs)

    @pytest.mark.parametrize("kwargs", [{}, {"bins": 5}])
    def test_hist_legacy_kwargs_warning(self, ts, kwargs):
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(ts.hist, by=ts.index.month, **kwargs)

    def test_hist_legacy_ax(self, ts):
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, default_axes=True)

    def test_hist_legacy_ax_and_fig(self, ts):
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, figure=fig, default_axes=True)

    def test_hist_legacy_fig(self, ts):
        fig, _ = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, figure=fig, default_axes=True)

    def test_hist_legacy_multi_ax(self, ts):
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2)
        _check_plot_works(ts.hist, figure=fig, ax=ax1, default_axes=True)
        _check_plot_works(ts.hist, figure=fig, ax=ax2, default_axes=True)

    def test_hist_legacy_by_fig_error(self, ts):
        fig, _ = mpl.pyplot.subplots(1, 1)
        msg = (
            "Cannot pass 'figure' when using the 'by' argument, since a new 'Figure' "
            "instance will be created"
        )
        with pytest.raises(ValueError, match=msg):
            ts.hist(by=ts.index, figure=fig)

    def test_hist_bins_legacy(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        ax = df.hist(bins=2)[0][0]
        assert len(ax.patches) == 2

    def test_hist_layout(self, hist_df):
        df = hist_df
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=(1, 1))

        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=[1, 1])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, layout, axes_num, res_layout",
        [
            ["gender", (2, 1), 2, (2, 1)],
            ["gender", (3, -1), 2, (3, 1)],
            ["category", (4, 1), 4, (4, 1)],
            ["category", (2, -1), 4, (2, 2)],
            ["category", (3, -1), 4, (3, 2)],
            ["category", (-1, 4), 4, (1, 4)],
            ["classroom", (2, 2), 3, (2, 2)],
        ],
    )
    def test_hist_layout_with_by(self, hist_df, by, layout, axes_num, res_layout):
        df = hist_df

        # _check_plot_works adds an `ax` kwarg to the method call
        # so we get a warning about an axis being cleared, even
        # though we don't explicing pass one, see GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.height.hist, by=getattr(df, by), layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=res_layout)

    def test_hist_layout_with_by_shape(self, hist_df):
        df = hist_df

        axes = df.height.hist(by=df.category, layout=(4, 2), figsize=(12, 7))
        _check_axes_shape(axes, axes_num=4, layout=(4, 2), figsize=(12, 7))

    def test_hist_no_overlap(self):
        from matplotlib.pyplot import (
            gcf,
            subplot,
        )

        x = Series(np.random.default_rng(2).standard_normal(2))
        y = Series(np.random.default_rng(2).standard_normal(2))
        subplot(121)
        x.hist()
        subplot(122)
        y.hist()
        fig = gcf()
        axes = fig.axes
        assert len(axes) == 2

    def test_hist_by_no_extra_plots(self, hist_df):
        df = hist_df
        df.height.hist(by=df.gender)
        assert len(mpl.pyplot.get_fignums()) == 1

    def test_plot_fails_when_ax_differs_from_figure(self, ts):
        from pylab import figure

        fig1 = figure()
        fig2 = figure()
        ax1 = fig1.add_subplot(111)
        msg = "passed axis not bound to passed figure"
        with pytest.raises(AssertionError, match=msg):
            ts.hist(ax=ax1, figure=fig2)

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 Verify functioning of histtype argument
        ser = Series(np.random.default_rng(2).integers(1, 10))
        ax = ser.hist(histtype=histtype)
        _check_patches_all_filled(ax, filled=expected)

    @pytest.mark.parametrize(
        "by, expected_axes_num, expected_layout", [(None, 1, (1, 1)), ("b", 2, (1, 2))]
    )
    def test_hist_with_legend(self, by, expected_axes_num, expected_layout):
        # GH 6279 - Series histogram can have a legend
        index = 15 * ["1"] + 15 * ["2"]
        s = Series(np.random.default_rng(2).standard_normal(30), index=index, name="a")
        s.index.name = "b"

        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(s.hist, default_axes=True, legend=True, by=by)
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        _check_legend_labels(axes, "a")

    @pytest.mark.parametrize("by", [None, "b"])
    def test_hist_with_legend_raises(self, by):
        # GH 6279 - Series histogram with legend and label raises
        index = 15 * ["1"] + 15 * ["2"]
        s = Series(np.random.default_rng(2).standard_normal(30), index=index, name="a")
        s.index.name = "b"

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            s.hist(legend=True, by=by, label="c")

    def test_hist_kwargs(self, ts):
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 5
        _check_text_labels(ax.yaxis.get_label(), "Frequency")

    def test_hist_kwargs_horizontal(self, ts):
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(orientation="horizontal", ax=ax)
        _check_text_labels(ax.xaxis.get_label(), "Frequency")

    def test_hist_kwargs_align(self, ts):
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(align="left", stacked=True, ax=ax)

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_hist_kde(self, ts):
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        # ticks are values, thus ticklabels are blank
        _check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [""] * len(ylabels))

    def test_hist_kde_plot_works(self, ts):
        pytest.importorskip("scipy")
        _check_plot_works(ts.plot.kde)

    def test_hist_kde_density_works(self, ts):
        pytest.importorskip("scipy")
        _check_plot_works(ts.plot.density)

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_hist_kde_logy(self, ts):
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.kde(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [""] * len(ylabels))

    def test_hist_kde_color_bins(self, ts):
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.hist(logy=True, bins=10, color="b", ax=ax)
        _check_ax_scales(ax, yaxis="log")
        assert len(ax.patches) == 10
        _check_colors(ax.patches, facecolors=["b"] * 10)

    def test_hist_kde_color(self, ts):
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.kde(logy=True, color="r", ax=ax)
        _check_ax_scales(ax, yaxis="log")
        lines = ax.get_lines()
        assert len(lines) == 1
        _check_colors(lines, ["r"])


class TestDataFramePlots:
    @pytest.mark.slow
    def test_hist_df_legacy(self, hist_df):
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(hist_df.hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout(self):
        # make sure layout is handled
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, grid=False)
        _check_axes_shape(axes, axes_num=3, layout=(2, 2))
        assert not axes[1, 1].get_visible()

        _check_plot_works(df[[2]].hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout2(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
        _check_plot_works(df.hist)

    @pytest.mark.slow
    def test_hist_df_legacy_layout3(self):
        # make sure layout is handled
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, layout=(4, 2))
        _check_axes_shape(axes, axes_num=6, layout=(4, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs", [{"sharex": True, "sharey": True}, {"figsize": (8, 10)}, {"bins": 5}]
    )
    def test_hist_df_legacy_layout_kwargs(self, kwargs):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # make sure sharex, sharey is handled
        # handle figsize arg
        # check bins argument
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.hist, **kwargs)

    @pytest.mark.slow
    def test_hist_df_legacy_layout_labelsize_rot(self, frame_or_series):
        # make sure xlabelsize and xrot are handled
        obj = frame_or_series(range(10))
        xf, yf = 20, 18
        xrot, yrot = 30, 40
        axes = obj.hist(xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
        _check_ticks_props(axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)

    @pytest.mark.slow
    def test_hist_df_legacy_rectangles(self):
        from matplotlib.patches import Rectangle

        ser = Series(range(10))
        ax = ser.hist(cumulative=True, bins=4, density=True)
        # height of last bin (index 5) must be 1.0
        rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

    @pytest.mark.slow
    def test_hist_df_legacy_scale(self):
        ser = Series(range(10))
        ax = ser.hist(log=True)
        # scale of y must be 'log'
        _check_ax_scales(ax, yaxis="log")

    @pytest.mark.slow
    def test_hist_df_legacy_external_error(self):
        ser = Series(range(10))
        # propagate attr exception from matplotlib.Axes.hist
        with tm.external_error_raised(AttributeError):
            ser.hist(foo="bar")

    def test_hist_non_numerical_or_datetime_raises(self):
        # gh-10444, GH32590
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(10),
                "b": np.random.default_rng(2).integers(0, 10, 10),
                "c": to_datetime(
                    np.random.default_rng(2).integers(
                        1582800000000000000, 1583500000000000000, 10, dtype=np.int64
                    )
                ),
                "d": to_datetime(
                    np.random.default_rng(2).integers(
                        1582800000000000000, 1583500000000000000, 10, dtype=np.int64
                    ),
                    utc=True,
                ),
            }
        )
        df_o = df.astype(object)

        msg = "hist method requires numerical or datetime columns, nothing to plot."
        with pytest.raises(ValueError, match=msg):
            df_o.hist()

    @pytest.mark.parametrize(
        "layout_test",
        (
            {"layout": None, "expected_size": (2, 2)},  # default is 2x2
            {"layout": (2, 2), "expected_size": (2, 2)},
            {"layout": (4, 1), "expected_size": (4, 1)},
            {"layout": (1, 4), "expected_size": (1, 4)},
            {"layout": (3, 3), "expected_size": (3, 3)},
            {"layout": (-1, 4), "expected_size": (1, 4)},
            {"layout": (4, -1), "expected_size": (4, 1)},
            {"layout": (-1, 2), "expected_size": (2, 2)},
            {"layout": (2, -1), "expected_size": (2, 2)},
        ),
    )
    def test_hist_layout(self, layout_test):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        axes = df.hist(layout=layout_test["layout"])
        expected = layout_test["expected_size"]
        _check_axes_shape(axes, axes_num=3, layout=expected)

    def test_hist_layout_error(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # layout too small for all 4 plots
        msg = "Layout of 1x1 must be larger than required size 3"
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(1, 1))

        # invalid format for layout
        msg = re.escape("Layout must be a tuple of (rows, columns)")
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(1,))
        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(-1, -1))

    # GH 9351
    def test_tight_layout(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((100, 2)))
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=100,
                dtype=np.int64,
            )
        )
        # Use default_axes=True when plotting method generate subplots itself
        _check_plot_works(df.hist, default_axes=True)
        mpl.pyplot.tight_layout()

    def test_hist_subplot_xrot(self):
        # GH 30288
        df = DataFrame(
            {
                "length": [1.5, 0.5, 1.2, 0.9, 3],
                "animal": ["pig", "rabbit", "pig", "pig", "rabbit"],
            }
        )
        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            column="length",
            by="animal",
            bins=5,
            xrot=0,
        )
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize(
        "column, expected",
        [
            (None, ["width", "length", "height"]),
            (["length", "width", "height"], ["length", "width", "height"]),
        ],
    )
    def test_hist_column_order_unchanged(self, column, expected):
        # GH29235

        df = DataFrame(
            {
                "width": [0.7, 0.2, 0.15, 0.2, 1.1],
                "length": [1.5, 0.5, 1.2, 0.9, 3],
                "height": [3, 0.5, 3.4, 2, 1],
            },
            index=["pig", "rabbit", "duck", "chicken", "horse"],
        )

        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            column=column,
            layout=(1, 3),
        )
        result = [axes[0, i].get_title() for i in range(3)]
        assert result == expected

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 Verify functioning of histtype argument
        df = DataFrame(
            np.random.default_rng(2).integers(1, 10, size=(100, 2)), columns=["a", "b"]
        )
        ax = df.hist(histtype=histtype)
        _check_patches_all_filled(ax, filled=expected)

    @pytest.mark.parametrize("by", [None, "c"])
    @pytest.mark.parametrize("column", [None, "b"])
    def test_hist_with_legend(self, by, column):
        # GH 6279 - DataFrame histogram can have a legend
        expected_axes_num = 1 if by is None and column is not None else 2
        expected_layout = (1, expected_axes_num)
        expected_labels = column or ["a", "b"]
        if by is not None:
            expected_labels = [expected_labels] * 2

        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )

        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            legend=True,
            by=by,
            column=column,
        )

        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        if by is None and column is None:
            axes = axes[0]
        for expected_label, ax in zip(expected_labels, axes):
            _check_legend_labels(ax, expected_label)

    @pytest.mark.parametrize("by", [None, "c"])
    @pytest.mark.parametrize("column", [None, "b"])
    def test_hist_with_legend_raises(self, by, column):
        # GH 6279 - DataFrame histogram with legend and label raises
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            df.hist(legend=True, by=by, column=column, label="d")

    def test_hist_df_kwargs(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 10

    def test_hist_df_with_nonnumerics(self):
        # GH 9853
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=["A", "B", "C", "D"],
        )
        df["E"] = ["x", "y"] * 5
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 20

    def test_hist_df_with_nonnumerics_no_bins(self):
        # GH 9853
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=["A", "B", "C", "D"],
        )
        df["E"] = ["x", "y"] * 5
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(ax=ax)  # bins=10
        assert len(ax.patches) == 40

    def test_hist_secondary_legend(self):
        # GH 9610
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)), columns=list("abcd")
        )

        # primary -> secondary
        _, ax = mpl.pyplot.subplots()
        ax = df["a"].plot.hist(legend=True, ax=ax)
        df["b"].plot.hist(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left and right axis must be visible
        _check_legend_labels(ax, labels=["a", "b (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_hist_secondary_secondary(self):
        # GH 9610
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)), columns=list("abcd")
        )
        # secondary -> secondary
        _, ax = mpl.pyplot.subplots()
        ax = df["a"].plot.hist(legend=True, secondary_y=True, ax=ax)
        df["b"].plot.hist(ax=ax, legend=True, secondary_y=True)
        # both legends are draw on left ax
        # left axis must be invisible, right axis must be visible
        _check_legend_labels(ax.left_ax, labels=["a (right)", "b (right)"])
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_hist_secondary_primary(self):
        # GH 9610
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)), columns=list("abcd")
        )
        # secondary -> primary
        _, ax = mpl.pyplot.subplots()
        ax = df["a"].plot.hist(legend=True, secondary_y=True, ax=ax)
        # right axes is returned
        df["b"].plot.hist(ax=ax, legend=True)
        # both legends are draw on left ax
        # left and right axis must be visible
        _check_legend_labels(ax.left_ax, labels=["a (right)", "b"])
        assert ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_hist_with_nans_and_weights(self):
        # GH 48884
        mpl_patches = pytest.importorskip("matplotlib.patches")
        df = DataFrame(
            [[np.nan, 0.2, 0.3], [0.4, np.nan, np.nan], [0.7, 0.8, 0.9]],
            columns=list("abc"),
        )
        weights = np.array([0.25, 0.3, 0.45])
        no_nan_df = DataFrame([[0.4, 0.2, 0.3], [0.7, 0.8, 0.9]], columns=list("abc"))
        no_nan_weights = np.array([[0.3, 0.25, 0.25], [0.45, 0.45, 0.45]])

        _, ax0 = mpl.pyplot.subplots()
        df.plot.hist(ax=ax0, weights=weights)
        rects = [x for x in ax0.get_children() if isinstance(x, mpl_patches.Rectangle)]
        heights = [rect.get_height() for rect in rects]
        _, ax1 = mpl.pyplot.subplots()
        no_nan_df.plot.hist(ax=ax1, weights=no_nan_weights)
        no_nan_rects = [
            x for x in ax1.get_children() if isinstance(x, mpl_patches.Rectangle)
        ]
        no_nan_heights = [rect.get_height() for rect in no_nan_rects]
        assert all(h0 == h1 for h0, h1 in zip(heights, no_nan_heights))

        idxerror_weights = np.array([[0.3, 0.25], [0.45, 0.45]])

        msg = "weights must have the same shape as data, or be a single column"
        with pytest.raises(ValueError, match=msg):
            _, ax2 = mpl.pyplot.subplots()
            no_nan_df.plot.hist(ax=ax2, weights=idxerror_weights)


class TestDataFrameGroupByPlots:
    def test_grouped_hist_legacy(self):
        from pandas.plotting._matplotlib.hist import _grouped_hist

        rs = np.random.default_rng(10)
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        df["C"] = rs.integers(0, 4, 10)
        df["D"] = ["X"] * 10

        axes = _grouped_hist(df.A, by=df.C)
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    def test_grouped_hist_legacy_axes_shape_no_col(self):
        rs = np.random.default_rng(10)
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        df["C"] = rs.integers(0, 4, 10)
        df["D"] = ["X"] * 10
        axes = df.hist(by=df.C)
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    def test_grouped_hist_legacy_single_key(self):
        rs = np.random.default_rng(2)
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        df["C"] = rs.integers(0, 4, 10)
        df["D"] = ["X"] * 10
        # group by a key with single value
        axes = df.hist(by="D", rot=30)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        _check_ticks_props(axes, xrot=30)

    def test_grouped_hist_legacy_grouped_hist_kwargs(self):
        from matplotlib.patches import Rectangle

        from pandas.plotting._matplotlib.hist import _grouped_hist

        rs = np.random.default_rng(2)
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        df["C"] = rs.integers(0, 4, 10)
        # make sure kwargs to hist are handled
        xf, yf = 20, 18
        xrot, yrot = 30, 40

        axes = _grouped_hist(
            df.A,
            by=df.C,
            cumulative=True,
            bins=4,
            xlabelsize=xf,
            xrot=xrot,
            ylabelsize=yf,
            yrot=yrot,
            density=True,
        )
        # height of last bin (index 5) must be 1.0
        for ax in axes.ravel():
            rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]
            height = rects[-1].get_height()
            tm.assert_almost_equal(height, 1.0)
        _check_ticks_props(axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)

    def test_grouped_hist_legacy_grouped_hist(self):
        from pandas.plotting._matplotlib.hist import _grouped_hist

        rs = np.random.default_rng(2)
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        df["C"] = rs.integers(0, 4, 10)
        df["D"] = ["X"] * 10
        axes = _grouped_hist(df.A, by=df.C, log=True)
        # scale of y must be 'log'
        _check_ax_scales(axes, yaxis="log")

    def test_grouped_hist_legacy_external_err(self):
        from pandas.plotting._matplotlib.hist import _grouped_hist

        rs = np.random.default_rng(2)
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        df["C"] = rs.integers(0, 4, 10)
        df["D"] = ["X"] * 10
        # propagate attr exception from matplotlib.Axes.hist
        with tm.external_error_raised(AttributeError):
            _grouped_hist(df.A, by=df.C, foo="bar")

    def test_grouped_hist_legacy_figsize_err(self):
        rs = np.random.default_rng(2)
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        df["C"] = rs.integers(0, 4, 10)
        df["D"] = ["X"] * 10
        msg = "Specify figure size by tuple instead"
        with pytest.raises(ValueError, match=msg):
            df.hist(by="C", figsize="default")

    def test_grouped_hist_legacy2(self):
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender_int = np.random.default_rng(2).choice([0, 1], size=n)
        df_int = DataFrame({"height": height, "weight": weight, "gender": gender_int})
        gb = df_int.groupby("gender")
        axes = gb.hist()
        assert len(axes) == 2
        assert len(mpl.pyplot.get_fignums()) == 2

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "msg, plot_col, by_col, layout",
        [
            [
                "Layout of 1x1 must be larger than required size 2",
                "weight",
                "gender",
                (1, 1),
            ],
            [
                "Layout of 1x3 must be larger than required size 4",
                "height",
                "category",
                (1, 3),
            ],
            [
                "At least one dimension of layout must be positive",
                "height",
                "category",
                (-1, -1),
            ],
        ],
    )
    def test_grouped_hist_layout_error(self, hist_df, msg, plot_col, by_col, layout):
        df = hist_df
        with pytest.raises(ValueError, match=msg):
            df.hist(column=plot_col, by=getattr(df, by_col), layout=layout)

    @pytest.mark.slow
    def test_grouped_hist_layout_warning(self, hist_df):
        df = hist_df
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(
                df.hist, column="height", by=df.gender, layout=(2, 1)
            )
        _check_axes_shape(axes, axes_num=2, layout=(2, 1))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "layout, check_layout, figsize",
        [[(4, 1), (4, 1), None], [(-1, 1), (4, 1), None], [(4, 2), (4, 2), (12, 8)]],
    )
    def test_grouped_hist_layout_figsize(self, hist_df, layout, check_layout, figsize):
        df = hist_df
        axes = df.hist(column="height", by=df.category, layout=layout, figsize=figsize)
        _check_axes_shape(axes, axes_num=4, layout=check_layout, figsize=figsize)

    @pytest.mark.slow
    @pytest.mark.parametrize("kwargs", [{}, {"column": "height", "layout": (2, 2)}])
    def test_grouped_hist_layout_by_warning(self, hist_df, kwargs):
        df = hist_df
        # GH 6769
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.hist, by="classroom", **kwargs)
        _check_axes_shape(axes, axes_num=3, layout=(2, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs, axes_num, layout",
        [
            [{"by": "gender", "layout": (3, 5)}, 2, (3, 5)],
            [{"column": ["height", "weight", "category"]}, 3, (2, 2)],
        ],
    )
    def test_grouped_hist_layout_axes(self, hist_df, kwargs, axes_num, layout):
        df = hist_df
        axes = df.hist(**kwargs)
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    def test_grouped_hist_multiple_axes(self, hist_df):
        # GH 6970, GH 7069
        df = hist_df

        fig, axes = mpl.pyplot.subplots(2, 3)
        returned = df.hist(column=["height", "weight", "category"], ax=axes[0])
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[0])
        assert returned[0].figure is fig

    def test_grouped_hist_multiple_axes_no_cols(self, hist_df):
        # GH 6970, GH 7069
        df = hist_df

        fig, axes = mpl.pyplot.subplots(2, 3)
        returned = df.hist(by="classroom", ax=axes[1])
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[1])
        assert returned[0].figure is fig

    def test_grouped_hist_multiple_axes_error(self, hist_df):
        # GH 6970, GH 7069
        df = hist_df
        fig, axes = mpl.pyplot.subplots(2, 3)
        # pass different number of axes from required
        msg = "The number of passed axes must be 1, the same as the output plot"
        with pytest.raises(ValueError, match=msg):
            axes = df.hist(column="height", ax=axes)

    def test_axis_share_x(self, hist_df):
        df = hist_df
        # GH4089
        ax1, ax2 = df.hist(column="height", by=df.gender, sharex=True)

        # share x
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)

        # don't share y
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)

    def test_axis_share_y(self, hist_df):
        df = hist_df
        ax1, ax2 = df.hist(column="height", by=df.gender, sharey=True)

        # share y
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

        # don't share x
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)

    def test_axis_share_xy(self, hist_df):
        df = hist_df
        ax1, ax2 = df.hist(column="height", by=df.gender, sharex=True, sharey=True)

        # share both x and y
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)

        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 Verify functioning of histtype argument
        df = DataFrame(
            np.random.default_rng(2).integers(1, 10, size=(10, 2)), columns=["a", "b"]
        )
        ax = df.hist(by="a", histtype=histtype)
        _check_patches_all_filled(ax, filled=expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_arithmetic.py ===
from datetime import (
    date,
    timedelta,
    timezone,
)
from decimal import Decimal
import operator

import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency

import pandas as pd
from pandas import (
    Categorical,
    DatetimeTZDtype,
    Index,
    Series,
    Timedelta,
    bdate_range,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED


@pytest.fixture(autouse=True, params=[0, 1000000], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield


def _permute(obj):
    return obj.take(np.random.default_rng(2).permutation(len(obj)))


class TestSeriesFlexArithmetic:
    @pytest.mark.parametrize(
        "ts",
        [
            (lambda x: x, lambda x: x * 2, False),
            (lambda x: x, lambda x: x[::2], False),
            (lambda x: x, lambda x: 5, True),
            (
                lambda x: Series(range(10), dtype=np.float64),
                lambda x: Series(range(10), dtype=np.float64),
                True,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "opname", ["add", "sub", "mul", "floordiv", "truediv", "pow"]
    )
    def test_flex_method_equivalence(self, opname, ts):
        # check that Series.{opname} behaves like Series.__{opname}__,
        tser = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20),
            name="ts",
        )

        series = ts[0](tser)
        other = ts[1](tser)
        check_reverse = ts[2]

        op = getattr(Series, opname)
        alt = getattr(operator, opname)

        result = op(series, other)
        expected = alt(series, other)
        tm.assert_almost_equal(result, expected)
        if check_reverse:
            rop = getattr(Series, "r" + opname)
            result = rop(series, other)
            expected = alt(other, series)
            tm.assert_almost_equal(result, expected)

    def test_flex_method_subclass_metadata_preservation(self, all_arithmetic_operators):
        # GH 13208
        class MySeries(Series):
            _metadata = ["x"]

            @property
            def _constructor(self):
                return MySeries

        opname = all_arithmetic_operators
        op = getattr(Series, opname)
        m = MySeries([1, 2, 3], name="test")
        m.x = 42
        result = op(m, 1)
        assert result.x == 42

    def test_flex_add_scalar_fill_value(self):
        # GH12723
        ser = Series([0, 1, np.nan, 3, 4, 5])

        exp = ser.fillna(0).add(2)
        res = ser.add(2, fill_value=0)
        tm.assert_series_equal(res, exp)

    pairings = [(Series.div, operator.truediv, 1), (Series.rdiv, ops.rtruediv, 1)]
    for op in ["add", "sub", "mul", "pow", "truediv", "floordiv"]:
        fv = 0
        lop = getattr(Series, op)
        lequiv = getattr(operator, op)
        rop = getattr(Series, "r" + op)
        # bind op at definition time...
        requiv = lambda x, y, op=op: getattr(operator, op)(y, x)
        pairings.append((lop, lequiv, fv))
        pairings.append((rop, requiv, fv))

    @pytest.mark.parametrize("op, equiv_op, fv", pairings)
    def test_operators_combine(self, op, equiv_op, fv):
        def _check_fill(meth, op, a, b, fill_value=0):
            exp_index = a.index.union(b.index)
            a = a.reindex(exp_index)
            b = b.reindex(exp_index)

            amask = isna(a)
            bmask = isna(b)

            exp_values = []
            for i in range(len(exp_index)):
                with np.errstate(all="ignore"):
                    if amask[i]:
                        if bmask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(fill_value, b[i]))
                    elif bmask[i]:
                        if amask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(a[i], fill_value))
                    else:
                        exp_values.append(op(a[i], b[i]))

            result = meth(a, b, fill_value=fill_value)
            expected = Series(exp_values, exp_index)
            tm.assert_series_equal(result, expected)

        a = Series([np.nan, 1.0, 2.0, 3.0, np.nan], index=np.arange(5))
        b = Series([np.nan, 1, np.nan, 3, np.nan, 4.0], index=np.arange(6))

        result = op(a, b)
        exp = equiv_op(a, b)
        tm.assert_series_equal(result, exp)
        _check_fill(op, equiv_op, a, b, fill_value=fv)
        # should accept axis=0 or axis='rows'
        op(a, b, axis=0)


class TestSeriesArithmetic:
    # Some of these may end up in tests/arithmetic, but are not yet sorted

    def test_add_series_with_period_index(self):
        rng = pd.period_range("1/1/2000", "1/1/2010", freq="Y")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        result = ts + ts[::2]
        expected = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_series_equal(result, expected)

        result = ts + _permute(ts[::2])
        tm.assert_series_equal(result, expected)

        msg = "Input has different freq=D from Period\\(freq=Y-DEC\\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts + ts.asfreq("D", how="end")

    @pytest.mark.parametrize(
        "target_add,input_value,expected_value",
        [
            ("!", ["hello", "world"], ["hello!", "world!"]),
            ("m", ["hello", "world"], ["hellom", "worldm"]),
        ],
    )
    def test_string_addition(self, target_add, input_value, expected_value):
        # GH28658 - ensure adding 'm' does not raise an error
        a = Series(input_value)

        result = a + target_add
        expected = Series(expected_value)
        tm.assert_series_equal(result, expected)

    def test_divmod(self):
        # GH#25557
        a = Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
        b = Series([2, np.nan, 1, np.nan], index=["a", "b", "d", "e"])

        result = a.divmod(b)
        expected = divmod(a, b)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        result = a.rdivmod(b)
        expected = divmod(b, a)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    @pytest.mark.parametrize("index", [None, range(9)])
    def test_series_integer_mod(self, index):
        # GH#24396
        s1 = Series(range(1, 10))
        s2 = Series("foo", index=index)

        msg = "not all arguments converted during string formatting|'mod' not supported"

        with pytest.raises(TypeError, match=msg):
            s2 % s1

    def test_add_with_duplicate_index(self):
        # GH14227
        s1 = Series([1, 2], index=[1, 1])
        s2 = Series([10, 10], index=[1, 2])
        result = s1 + s2
        expected = Series([11, 12, np.nan], index=[1, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_add_na_handling(self):
        ser = Series(
            [Decimal("1.3"), Decimal("2.3")], index=[date(2012, 1, 1), date(2012, 1, 2)]
        )

        result = ser + ser.shift(1)
        result2 = ser.shift(1) + ser
        assert isna(result.iloc[0])
        assert isna(result2.iloc[0])

    def test_add_corner_cases(self, datetime_series):
        empty = Series([], index=Index([]), dtype=np.float64)

        result = datetime_series + empty
        assert np.isnan(result).all()

        result = empty + empty.copy()
        assert len(result) == 0

    def test_add_float_plus_int(self, datetime_series):
        # float + int
        int_ts = datetime_series.astype(int)[:-5]
        added = datetime_series + int_ts
        expected = Series(
            datetime_series.values[:-5] + int_ts.values,
            index=datetime_series.index[:-5],
            name="ts",
        )
        tm.assert_series_equal(added[:-5], expected)

    def test_mul_empty_int_corner_case(self):
        s1 = Series([], [], dtype=np.int32)
        s2 = Series({"x": 0.0})
        tm.assert_series_equal(s1 * s2, Series([np.nan], index=["x"]))

    def test_sub_datetimelike_align(self):
        # GH#7500
        # datetimelike ops need to align
        dt = Series(date_range("2012-1-1", periods=3, freq="D"))
        dt.iloc[2] = np.nan
        dt2 = dt[::-1]

        expected = Series([timedelta(0), timedelta(0), pd.NaT])
        # name is reset
        result = dt2 - dt
        tm.assert_series_equal(result, expected)

        expected = Series(expected, name=0)
        result = (dt2.to_frame() - dt.to_frame())[0]
        tm.assert_series_equal(result, expected)

    def test_alignment_doesnt_change_tz(self):
        # GH#33671
        dti = date_range("2016-01-01", periods=10, tz="CET")
        dti_utc = dti.tz_convert("UTC")
        ser = Series(10, index=dti)
        ser_utc = Series(10, index=dti_utc)

        # we don't care about the result, just that original indexes are unchanged
        ser * ser_utc

        assert ser.index is dti
        assert ser_utc.index is dti_utc

    def test_alignment_categorical(self):
        # GH13365
        cat = Categorical(["3z53", "3z53", "LoJG", "LoJG", "LoJG", "N503"])
        ser1 = Series(2, index=cat)
        ser2 = Series(2, index=cat[:-1])
        result = ser1 * ser2

        exp_index = ["3z53"] * 4 + ["LoJG"] * 9 + ["N503"]
        exp_index = pd.CategoricalIndex(exp_index, categories=cat.categories)
        exp_values = [4.0] * 13 + [np.nan]
        expected = Series(exp_values, exp_index)

        tm.assert_series_equal(result, expected)

    def test_arithmetic_with_duplicate_index(self):
        # GH#8363
        # integer ops with a non-unique index
        index = [2, 2, 3, 3, 4]
        ser = Series(np.arange(1, 6, dtype="int64"), index=index)
        other = Series(np.arange(5, dtype="int64"), index=index)
        result = ser - other
        expected = Series(1, index=[2, 2, 3, 3, 4])
        tm.assert_series_equal(result, expected)

        # GH#8363
        # datetime ops with a non-unique index
        ser = Series(date_range("20130101 09:00:00", periods=5), index=index)
        other = Series(date_range("20130101", periods=5), index=index)
        result = ser - other
        expected = Series(Timedelta("9 hours"), index=[2, 2, 3, 3, 4])
        tm.assert_series_equal(result, expected)

    def test_masked_and_non_masked_propagate_na(self):
        # GH#45810
        ser1 = Series([0, np.nan], dtype="float")
        ser2 = Series([0, 1], dtype="Int64")
        result = ser1 * ser2
        expected = Series([0, pd.NA], dtype="Float64")
        tm.assert_series_equal(result, expected)

    def test_mask_div_propagate_na_for_non_na_dtype(self):
        # GH#42630
        ser1 = Series([15, pd.NA, 5, 4], dtype="Int64")
        ser2 = Series([15, 5, np.nan, 4])
        result = ser1 / ser2
        expected = Series([1.0, pd.NA, pd.NA, 1.0], dtype="Float64")
        tm.assert_series_equal(result, expected)

        result = ser2 / ser1
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val, dtype", [(3, "Int64"), (3.5, "Float64")])
    def test_add_list_to_masked_array(self, val, dtype):
        # GH#22962
        ser = Series([1, None, 3], dtype="Int64")
        result = ser + [1, None, val]
        expected = Series([2, None, 3 + val], dtype=dtype)
        tm.assert_series_equal(result, expected)

        result = [1, None, val] + ser
        tm.assert_series_equal(result, expected)

    def test_add_list_to_masked_array_boolean(self, request):
        # GH#22962
        warning = (
            UserWarning
            if request.node.callspec.id == "numexpr" and NUMEXPR_INSTALLED
            else None
        )
        ser = Series([True, None, False], dtype="boolean")
        with tm.assert_produces_warning(warning):
            result = ser + [True, None, True]
        expected = Series([True, None, True], dtype="boolean")
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(warning):
            result = [True, None, True] + ser
        tm.assert_series_equal(result, expected)


# ------------------------------------------------------------------
# Comparisons


class TestSeriesFlexComparison:
    @pytest.mark.parametrize("axis", [0, None, "index"])
    def test_comparison_flex_basic(self, axis, comparison_op):
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))
        result = getattr(left, comparison_op.__name__)(right, axis=axis)
        expected = comparison_op(left, right)
        tm.assert_series_equal(result, expected)

    def test_comparison_bad_axis(self, comparison_op):
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))

        msg = "No axis named 1 for object type"
        with pytest.raises(ValueError, match=msg):
            getattr(left, comparison_op.__name__)(right, axis=1)

    @pytest.mark.parametrize(
        "values, op",
        [
            ([False, False, True, False], "eq"),
            ([True, True, False, True], "ne"),
            ([False, False, True, False], "le"),
            ([False, False, False, False], "lt"),
            ([False, True, True, False], "ge"),
            ([False, True, False, False], "gt"),
        ],
    )
    def test_comparison_flex_alignment(self, values, op):
        left = Series([1, 3, 2], index=list("abc"))
        right = Series([2, 2, 2], index=list("bcd"))
        result = getattr(left, op)(right)
        expected = Series(values, index=list("abcd"))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values, op, fill_value",
        [
            ([False, False, True, True], "eq", 2),
            ([True, True, False, False], "ne", 2),
            ([False, False, True, True], "le", 0),
            ([False, False, False, True], "lt", 0),
            ([True, True, True, False], "ge", 0),
            ([True, True, False, False], "gt", 0),
        ],
    )
    def test_comparison_flex_alignment_fill(self, values, op, fill_value):
        left = Series([1, 3, 2], index=list("abc"))
        right = Series([2, 2, 2], index=list("bcd"))
        result = getattr(left, op)(right, fill_value=fill_value)
        expected = Series(values, index=list("abcd"))
        tm.assert_series_equal(result, expected)


class TestSeriesComparison:
    def test_comparison_different_length(self):
        a = Series(["a", "b", "c"])
        b = Series(["b", "a"])
        msg = "only compare identically-labeled Series"
        with pytest.raises(ValueError, match=msg):
            a < b

        a = Series([1, 2])
        b = Series([2, 3, 4])
        with pytest.raises(ValueError, match=msg):
            a == b

    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    def test_ser_flex_cmp_return_dtypes(self, opname):
        # GH#15115
        ser = Series([1, 3, 2], index=range(3))
        const = 2
        result = getattr(ser, opname)(const).dtypes
        expected = np.dtype("bool")
        assert result == expected

    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    def test_ser_flex_cmp_return_dtypes_empty(self, opname):
        # GH#15115 empty Series case
        ser = Series([1, 3, 2], index=range(3))
        empty = ser.iloc[:0]
        const = 2
        result = getattr(empty, opname)(const).dtypes
        expected = np.dtype("bool")
        assert result == expected

    @pytest.mark.parametrize(
        "names", [(None, None, None), ("foo", "bar", None), ("baz", "baz", "baz")]
    )
    def test_ser_cmp_result_names(self, names, comparison_op):
        # datetime64 dtype
        op = comparison_op
        dti = date_range("1949-06-07 03:00:00", freq="h", periods=5, name=names[0])
        ser = Series(dti).rename(names[1])
        result = op(ser, dti)
        assert result.name == names[2]

        # datetime64tz dtype
        dti = dti.tz_localize("US/Central")
        dti = pd.DatetimeIndex(dti, freq="infer")  # freq not preserved by tz_localize
        ser = Series(dti).rename(names[1])
        result = op(ser, dti)
        assert result.name == names[2]

        # timedelta64 dtype
        tdi = dti - dti.shift(1)
        ser = Series(tdi).rename(names[1])
        result = op(ser, tdi)
        assert result.name == names[2]

        # interval dtype
        if op in [operator.eq, operator.ne]:
            # interval dtype comparisons not yet implemented
            ii = pd.interval_range(start=0, periods=5, name=names[0])
            ser = Series(ii).rename(names[1])
            result = op(ser, ii)
            assert result.name == names[2]

        # categorical
        if op in [operator.eq, operator.ne]:
            # categorical dtype comparisons raise for inequalities
            cidx = tdi.astype("category")
            ser = Series(cidx).rename(names[1])
            result = op(ser, cidx)
            assert result.name == names[2]

    def test_comparisons(self):
        s = Series(["a", "b", "c"])
        s2 = Series([False, True, False])

        # it works!
        exp = Series([False, False, False])
        tm.assert_series_equal(s == s2, exp)
        tm.assert_series_equal(s2 == s, exp)

    # -----------------------------------------------------------------
    # Categorical Dtype Comparisons

    def test_categorical_comparisons(self):
        # GH#8938
        # allow equality comparisons
        a = Series(list("abc"), dtype="category")
        b = Series(list("abc"), dtype="object")
        c = Series(["a", "b", "cc"], dtype="object")
        d = Series(list("acb"), dtype="object")
        e = Categorical(list("abc"))
        f = Categorical(list("acb"))

        # vs scalar
        assert not (a == "a").all()
        assert ((a != "a") == ~(a == "a")).all()

        assert not ("a" == a).all()
        assert (a == "a")[0]
        assert ("a" == a)[0]
        assert not ("a" != a)[0]

        # vs list-like
        assert (a == a).all()
        assert not (a != a).all()

        assert (a == list(a)).all()
        assert (a == b).all()
        assert (b == a).all()
        assert ((~(a == b)) == (a != b)).all()
        assert ((~(b == a)) == (b != a)).all()

        assert not (a == c).all()
        assert not (c == a).all()
        assert not (a == d).all()
        assert not (d == a).all()

        # vs a cat-like
        assert (a == e).all()
        assert (e == a).all()
        assert not (a == f).all()
        assert not (f == a).all()

        assert (~(a == e) == (a != e)).all()
        assert (~(e == a) == (e != a)).all()
        assert (~(a == f) == (a != f)).all()
        assert (~(f == a) == (f != a)).all()

        # non-equality is not comparable
        msg = "can only compare equality or not"
        with pytest.raises(TypeError, match=msg):
            a < b
        with pytest.raises(TypeError, match=msg):
            b < a
        with pytest.raises(TypeError, match=msg):
            a > b
        with pytest.raises(TypeError, match=msg):
            b > a

    def test_unequal_categorical_comparison_raises_type_error(self):
        # unequal comparison should raise for unordered cats
        cat = Series(Categorical(list("abc")))
        msg = "can only compare equality or not"
        with pytest.raises(TypeError, match=msg):
            cat > "b"

        cat = Series(Categorical(list("abc"), ordered=False))
        with pytest.raises(TypeError, match=msg):
            cat > "b"

        # https://github.com/pandas-dev/pandas/issues/9836#issuecomment-92123057
        # and following comparisons with scalars not in categories should raise
        # for unequal comps, but not for equal/not equal
        cat = Series(Categorical(list("abc"), ordered=True))

        msg = "Invalid comparison between dtype=category and str"
        with pytest.raises(TypeError, match=msg):
            cat < "d"
        with pytest.raises(TypeError, match=msg):
            cat > "d"
        with pytest.raises(TypeError, match=msg):
            "d" < cat
        with pytest.raises(TypeError, match=msg):
            "d" > cat

        tm.assert_series_equal(cat == "d", Series([False, False, False]))
        tm.assert_series_equal(cat != "d", Series([True, True, True]))

    # -----------------------------------------------------------------

    def test_comparison_tuples(self):
        # GH#11339
        # comparisons vs tuple
        s = Series([(1, 1), (1, 2)])

        result = s == (1, 2)
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)

        result = s != (1, 2)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

        result = s == (0, 0)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

        result = s != (0, 0)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        s = Series([(1, 1), (1, 1)])

        result = s == (1, 1)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        result = s != (1, 1)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_frozenset(self):
        ser = Series([frozenset([1]), frozenset([1, 2])])

        result = ser == frozenset([1])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_operators_with_nas(self, comparison_op):
        ser = Series(bdate_range("1/1/2000", periods=10), dtype=object)
        ser[::2] = np.nan

        # test that comparisons work
        val = ser[5]

        result = comparison_op(ser, val)
        expected = comparison_op(ser.dropna(), val).reindex(ser.index)

        msg = "Downcasting object dtype arrays"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            if comparison_op is operator.ne:
                expected = expected.fillna(True).astype(bool)
            else:
                expected = expected.fillna(False).astype(bool)

        tm.assert_series_equal(result, expected)

    def test_ne(self):
        ts = Series([3, 4, 5, 6, 7], [3, 4, 5, 6, 7], dtype=float)
        expected = np.array([True, True, False, True, True])
        tm.assert_numpy_array_equal(ts.index != 5, expected)
        tm.assert_numpy_array_equal(~(ts.index == 5), expected)

    @pytest.mark.parametrize(
        "left, right",
        [
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2], index=list("ABD"), name="x"),
            ),
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2, 2], index=list("ABCD"), name="x"),
            ),
        ],
    )
    def test_comp_ops_df_compat(self, left, right, frame_or_series):
        # GH 1134
        # GH 50083 to clarify that index and columns must be identically labeled
        if frame_or_series is not Series:
            msg = (
                rf"Can only compare identically-labeled \(both index and columns\) "
                f"{frame_or_series.__name__} objects"
            )
            left = left.to_frame()
            right = right.to_frame()
        else:
            msg = (
                f"Can only compare identically-labeled {frame_or_series.__name__} "
                f"objects"
            )

        with pytest.raises(ValueError, match=msg):
            left == right
        with pytest.raises(ValueError, match=msg):
            right == left

        with pytest.raises(ValueError, match=msg):
            left != right
        with pytest.raises(ValueError, match=msg):
            right != left

        with pytest.raises(ValueError, match=msg):
            left < right
        with pytest.raises(ValueError, match=msg):
            right < left

    def test_compare_series_interval_keyword(self):
        # GH#25338
        ser = Series(["IntervalA", "IntervalB", "IntervalC"])
        result = ser == "IntervalA"
        expected = Series([True, False, False])
        tm.assert_series_equal(result, expected)


# ------------------------------------------------------------------
# Unsorted
#  These arithmetic tests were previously in other files, eventually
#  should be parametrized and put into tests.arithmetic


class TestTimeSeriesArithmetic:
    def test_series_add_tz_mismatch_converts_to_utc(self):
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        perm = np.random.default_rng(2).permutation(100)[:90]
        ser1 = Series(
            np.random.default_rng(2).standard_normal(90),
            index=rng.take(perm).tz_convert("US/Eastern"),
        )

        perm = np.random.default_rng(2).permutation(100)[:90]
        ser2 = Series(
            np.random.default_rng(2).standard_normal(90),
            index=rng.take(perm).tz_convert("Europe/Berlin"),
        )

        result = ser1 + ser2

        uts1 = ser1.tz_convert("utc")
        uts2 = ser2.tz_convert("utc")
        expected = uts1 + uts2

        # sort since input indexes are not equal
        expected = expected.sort_index()

        assert result.index.tz is timezone.utc
        tm.assert_series_equal(result, expected)

    def test_series_add_aware_naive_raises(self):
        rng = date_range("1/1/2011", periods=10, freq="h")
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        ser_utc = ser.tz_localize("utc")

        msg = "Cannot join tz-naive with tz-aware DatetimeIndex"
        with pytest.raises(Exception, match=msg):
            ser + ser_utc

        with pytest.raises(Exception, match=msg):
            ser_utc + ser

    # TODO: belongs in tests/arithmetic?
    def test_datetime_understood(self, unit):
        # Ensures it doesn't fail to create the right series
        # reported in issue#16726
        series = Series(date_range("2012-01-01", periods=3, unit=unit))
        offset = pd.offsets.DateOffset(days=6)
        result = series - offset
        exp_dti = pd.to_datetime(["2011-12-26", "2011-12-27", "2011-12-28"]).as_unit(
            unit
        )
        expected = Series(exp_dti)
        tm.assert_series_equal(result, expected)

    def test_align_date_objects_with_datetimeindex(self):
        rng = date_range("1/1/2000", periods=20)
        ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)

        ts_slice = ts[5:]
        ts2 = ts_slice.copy()
        ts2.index = [x.date() for x in ts2.index]

        result = ts + ts2
        result2 = ts2 + ts
        expected = ts + ts[5:]
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)


class TestNamePreservation:
    @pytest.mark.parametrize("box", [list, tuple, np.array, Index, Series, pd.array])
    @pytest.mark.parametrize("flex", [True, False])
    def test_series_ops_name_retention(self, flex, box, names, all_binary_operators):
        # GH#33930 consistent name-retention
        op = all_binary_operators

        left = Series(range(10), name=names[0])
        right = Series(range(10), name=names[1])

        name = op.__name__.strip("_")
        is_logical = name in ["and", "rand", "xor", "rxor", "or", "ror"]

        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )
        warn = None
        if box in [list, tuple] and is_logical:
            warn = FutureWarning

        right = box(right)
        if flex:
            if is_logical:
                # Series doesn't have these as flex methods
                return
            result = getattr(left, name)(right)
        else:
            # GH#37374 logical ops behaving as set ops deprecated
            with tm.assert_produces_warning(warn, match=msg):
                result = op(left, right)

        assert isinstance(result, Series)
        if box in [Index, Series]:
            assert result.name is names[2] or result.name == names[2]
        else:
            assert result.name is names[0] or result.name == names[0]

    def test_binop_maybe_preserve_name(self, datetime_series):
        # names match, preserve
        result = datetime_series * datetime_series
        assert result.name == datetime_series.name
        result = datetime_series.mul(datetime_series)
        assert result.name == datetime_series.name

        result = datetime_series * datetime_series[:-2]
        assert result.name == datetime_series.name

        # names don't match, don't preserve
        cp = datetime_series.copy()
        cp.name = "something else"
        result = datetime_series + cp
        assert result.name is None
        result = datetime_series.add(cp)
        assert result.name is None

        ops = ["add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow"]
        ops = ops + ["r" + op for op in ops]
        for op in ops:
            # names match, preserve
            ser = datetime_series.copy()
            result = getattr(ser, op)(ser)
            assert result.name == datetime_series.name

            # names don't match, don't preserve
            cp = datetime_series.copy()
            cp.name = "changed"
            result = getattr(ser, op)(cp)
            assert result.name is None

    def test_scalarop_preserve_name(self, datetime_series):
        result = datetime_series * 2
        assert result.name == datetime_series.name


class TestInplaceOperations:
    @pytest.mark.parametrize(
        "dtype1, dtype2, dtype_expected, dtype_mul",
        (
            ("Int64", "Int64", "Int64", "Int64"),
            ("float", "float", "float", "float"),
            ("Int64", "float", "Float64", "Float64"),
            ("Int64", "Float64", "Float64", "Float64"),
        ),
    )
    def test_series_inplace_ops(self, dtype1, dtype2, dtype_expected, dtype_mul):
        # GH 37910

        ser1 = Series([1], dtype=dtype1)
        ser2 = Series([2], dtype=dtype2)
        ser1 += ser2
        expected = Series([3], dtype=dtype_expected)
        tm.assert_series_equal(ser1, expected)

        ser1 -= ser2
        expected = Series([1], dtype=dtype_expected)
        tm.assert_series_equal(ser1, expected)

        ser1 *= ser2
        expected = Series([2], dtype=dtype_mul)
        tm.assert_series_equal(ser1, expected)


def test_none_comparison(request, series_with_simple_index):
    series = series_with_simple_index

    if len(series) < 1:
        request.applymarker(
            pytest.mark.xfail(reason="Test doesn't make sense on empty data")
        )

    # bug brought up by #1079
    # changed from TypeError in 0.17.0
    series.iloc[0] = np.nan

    # noinspection PyComparisonWithNone
    result = series == None  # noqa: E711
    assert not result.iat[0]
    assert not result.iat[1]

    # noinspection PyComparisonWithNone
    result = series != None  # noqa: E711
    assert result.iat[0]
    assert result.iat[1]

    result = None == series  # noqa: E711
    assert not result.iat[0]
    assert not result.iat[1]

    result = None != series  # noqa: E711
    assert result.iat[0]
    assert result.iat[1]

    if lib.is_np_dtype(series.dtype, "M") or isinstance(series.dtype, DatetimeTZDtype):
        # Following DatetimeIndex (and Timestamp) convention,
        # inequality comparisons with Series[datetime64] raise
        msg = "Invalid comparison"
        with pytest.raises(TypeError, match=msg):
            None > series
        with pytest.raises(TypeError, match=msg):
            series > None
    else:
        result = None > series
        assert not result.iat[0]
        assert not result.iat[1]

        result = series < None
        assert not result.iat[0]
        assert not result.iat[1]


def test_series_varied_multiindex_alignment():
    # GH 20414
    s1 = Series(
        range(8),
        index=pd.MultiIndex.from_product(
            [list("ab"), list("xy"), [1, 2]], names=["ab", "xy", "num"]
        ),
    )
    s2 = Series(
        [1000 * i for i in range(1, 5)],
        index=pd.MultiIndex.from_product([list("xy"), [1, 2]], names=["xy", "num"]),
    )
    result = s1.loc[pd.IndexSlice[["a"], :, :]] + s2
    expected = Series(
        [1000, 2001, 3002, 4003],
        index=pd.MultiIndex.from_tuples(
            [("a", "x", 1), ("a", "x", 2), ("a", "y", 1), ("a", "y", 2)],
            names=["ab", "xy", "num"],
        ),
    )
    tm.assert_series_equal(result, expected)


def test_rmod_consistent_large_series():
    # GH 29602
    result = Series([2] * 10001).rmod(-1)
    expected = Series([1] * 10001)

    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_timegrouper.py ===
"""
test with the TimeGrouper / grouping with datetimes
"""
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest
import pytz

from pandas._config import using_string_dtype

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper


@pytest.fixture
def frame_for_truncated_bingrouper():
    """
    DataFrame used by groupby_with_truncated_bingrouper, made into
    a separate fixture for easier reuse in
    test_groupby_apply_timegrouper_with_nat_apply_squeeze
    """
    df = DataFrame(
        {
            "Quantity": [18, 3, 5, 1, 9, 3],
            "Date": [
                Timestamp(2013, 9, 1, 13, 0),
                Timestamp(2013, 9, 1, 13, 5),
                Timestamp(2013, 10, 1, 20, 0),
                Timestamp(2013, 10, 3, 10, 0),
                pd.NaT,
                Timestamp(2013, 9, 2, 14, 0),
            ],
        }
    )
    return df


@pytest.fixture
def groupby_with_truncated_bingrouper(frame_for_truncated_bingrouper):
    """
    GroupBy object such that gb._grouper is a BinGrouper and
    len(gb._grouper.result_index) < len(gb._grouper.group_keys_seq)

    Aggregations on this groupby should have

        dti = date_range("2013-09-01", "2013-10-01", freq="5D", name="Date")

    As either the index or an index level.
    """
    df = frame_for_truncated_bingrouper

    tdg = Grouper(key="Date", freq="5D")
    gb = df.groupby(tdg)

    # check we're testing the case we're interested in
    assert len(gb._grouper.result_index) != len(gb._grouper.group_keys_seq)

    return gb


class TestGroupBy:
    # TODO(infer_string) resample sum introduces 0's
    # https://github.com/pandas-dev/pandas/issues/60229
    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_groupby_with_timegrouper(self):
        # GH 4161
        # TimeGrouper requires a sorted index
        # also verifies that the resultant index has the correct name
        df_original = DataFrame(
            {
                "Buyer": "Carl Carl Carl Carl Joe Carl".split(),
                "Quantity": [18, 3, 5, 1, 9, 3],
                "Date": [
                    datetime(2013, 9, 1, 13, 0),
                    datetime(2013, 9, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 3, 10, 0),
                    datetime(2013, 12, 2, 12, 0),
                    datetime(2013, 9, 2, 14, 0),
                ],
            }
        )

        # GH 6908 change target column's order
        df_reordered = df_original.sort_values(by="Quantity")

        for df in [df_original, df_reordered]:
            df = df.set_index(["Date"])

            exp_dti = date_range(
                "20130901",
                "20131205",
                freq="5D",
                name="Date",
                inclusive="left",
                unit=df.index.unit,
            )
            expected = DataFrame(
                {"Buyer": 0, "Quantity": 0},
                index=exp_dti,
            )
            # Cast to object to avoid implicit cast when setting entry to "CarlCarlCarl"
            expected = expected.astype({"Buyer": object})
            expected.iloc[0, 0] = "CarlCarlCarl"
            expected.iloc[6, 0] = "CarlCarl"
            expected.iloc[18, 0] = "Joe"
            expected.iloc[[0, 6, 18], 1] = np.array([24, 6, 9], dtype="int64")

            result1 = df.resample("5D").sum()
            tm.assert_frame_equal(result1, expected)

            df_sorted = df.sort_index()
            result2 = df_sorted.groupby(Grouper(freq="5D")).sum()
            tm.assert_frame_equal(result2, expected)

            result3 = df.groupby(Grouper(freq="5D")).sum()
            tm.assert_frame_equal(result3, expected)

    @pytest.mark.parametrize("should_sort", [True, False])
    def test_groupby_with_timegrouper_methods(self, should_sort):
        # GH 3881
        # make sure API of timegrouper conforms

        df = DataFrame(
            {
                "Branch": "A A A A A B".split(),
                "Buyer": "Carl Mark Carl Joe Joe Carl".split(),
                "Quantity": [1, 3, 5, 8, 9, 3],
                "Date": [
                    datetime(2013, 1, 1, 13, 0),
                    datetime(2013, 1, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 12, 2, 12, 0),
                    datetime(2013, 12, 2, 14, 0),
                ],
            }
        )

        if should_sort:
            df = df.sort_values(by="Quantity", ascending=False)

        df = df.set_index("Date", drop=False)
        g = df.groupby(Grouper(freq="6ME"))
        assert g.group_keys

        assert isinstance(g._grouper, BinGrouper)
        groups = g.groups
        assert isinstance(groups, dict)
        assert len(groups) == 3

    def test_timegrouper_with_reg_groups(self):
        # GH 3794
        # allow combination of timegrouper/reg groups

        df_original = DataFrame(
            {
                "Branch": "A A A A A A A B".split(),
                "Buyer": "Carl Mark Carl Carl Joe Joe Joe Carl".split(),
                "Quantity": [1, 3, 5, 1, 8, 1, 9, 3],
                "Date": [
                    datetime(2013, 1, 1, 13, 0),
                    datetime(2013, 1, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 12, 2, 12, 0),
                    datetime(2013, 12, 2, 14, 0),
                ],
            }
        ).set_index("Date")

        df_sorted = df_original.sort_values(by="Quantity", ascending=False)

        for df in [df_original, df_sorted]:
            expected = DataFrame(
                {
                    "Buyer": "Carl Joe Mark".split(),
                    "Quantity": [10, 18, 3],
                    "Date": [
                        datetime(2013, 12, 31, 0, 0),
                        datetime(2013, 12, 31, 0, 0),
                        datetime(2013, 12, 31, 0, 0),
                    ],
                }
            ).set_index(["Date", "Buyer"])

            msg = "The default value of numeric_only"
            result = df.groupby([Grouper(freq="YE"), "Buyer"]).sum(numeric_only=True)
            tm.assert_frame_equal(result, expected)

            expected = DataFrame(
                {
                    "Buyer": "Carl Mark Carl Joe".split(),
                    "Quantity": [1, 3, 9, 18],
                    "Date": [
                        datetime(2013, 1, 1, 0, 0),
                        datetime(2013, 1, 1, 0, 0),
                        datetime(2013, 7, 1, 0, 0),
                        datetime(2013, 7, 1, 0, 0),
                    ],
                }
            ).set_index(["Date", "Buyer"])
            result = df.groupby([Grouper(freq="6MS"), "Buyer"]).sum(numeric_only=True)
            tm.assert_frame_equal(result, expected)

        df_original = DataFrame(
            {
                "Branch": "A A A A A A A B".split(),
                "Buyer": "Carl Mark Carl Carl Joe Joe Joe Carl".split(),
                "Quantity": [1, 3, 5, 1, 8, 1, 9, 3],
                "Date": [
                    datetime(2013, 10, 1, 13, 0),
                    datetime(2013, 10, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 10, 2, 12, 0),
                    datetime(2013, 10, 2, 14, 0),
                ],
            }
        ).set_index("Date")

        df_sorted = df_original.sort_values(by="Quantity", ascending=False)
        for df in [df_original, df_sorted]:
            expected = DataFrame(
                {
                    "Buyer": "Carl Joe Mark Carl Joe".split(),
                    "Quantity": [6, 8, 3, 4, 10],
                    "Date": [
                        datetime(2013, 10, 1, 0, 0),
                        datetime(2013, 10, 1, 0, 0),
                        datetime(2013, 10, 1, 0, 0),
                        datetime(2013, 10, 2, 0, 0),
                        datetime(2013, 10, 2, 0, 0),
                    ],
                }
            ).set_index(["Date", "Buyer"])

            result = df.groupby([Grouper(freq="1D"), "Buyer"]).sum(numeric_only=True)
            tm.assert_frame_equal(result, expected)

            result = df.groupby([Grouper(freq="1ME"), "Buyer"]).sum(numeric_only=True)
            expected = DataFrame(
                {
                    "Buyer": "Carl Joe Mark".split(),
                    "Quantity": [10, 18, 3],
                    "Date": [
                        datetime(2013, 10, 31, 0, 0),
                        datetime(2013, 10, 31, 0, 0),
                        datetime(2013, 10, 31, 0, 0),
                    ],
                }
            ).set_index(["Date", "Buyer"])
            tm.assert_frame_equal(result, expected)

            # passing the name
            df = df.reset_index()
            result = df.groupby([Grouper(freq="1ME", key="Date"), "Buyer"]).sum(
                numeric_only=True
            )
            tm.assert_frame_equal(result, expected)

            with pytest.raises(KeyError, match="'The grouper name foo is not found'"):
                df.groupby([Grouper(freq="1ME", key="foo"), "Buyer"]).sum()

            # passing the level
            df = df.set_index("Date")
            result = df.groupby([Grouper(freq="1ME", level="Date"), "Buyer"]).sum(
                numeric_only=True
            )
            tm.assert_frame_equal(result, expected)
            result = df.groupby([Grouper(freq="1ME", level=0), "Buyer"]).sum(
                numeric_only=True
            )
            tm.assert_frame_equal(result, expected)

            with pytest.raises(ValueError, match="The level foo is not valid"):
                df.groupby([Grouper(freq="1ME", level="foo"), "Buyer"]).sum()

            # multi names
            df = df.copy()
            df["Date"] = df.index + offsets.MonthEnd(2)
            result = df.groupby([Grouper(freq="1ME", key="Date"), "Buyer"]).sum(
                numeric_only=True
            )
            expected = DataFrame(
                {
                    "Buyer": "Carl Joe Mark".split(),
                    "Quantity": [10, 18, 3],
                    "Date": [
                        datetime(2013, 11, 30, 0, 0),
                        datetime(2013, 11, 30, 0, 0),
                        datetime(2013, 11, 30, 0, 0),
                    ],
                }
            ).set_index(["Date", "Buyer"])
            tm.assert_frame_equal(result, expected)

            # error as we have both a level and a name!
            msg = "The Grouper cannot specify both a key and a level!"
            with pytest.raises(ValueError, match=msg):
                df.groupby(
                    [Grouper(freq="1ME", key="Date", level="Date"), "Buyer"]
                ).sum()

            # single groupers
            expected = DataFrame(
                [[31]],
                columns=["Quantity"],
                index=DatetimeIndex(
                    [datetime(2013, 10, 31, 0, 0)], freq=offsets.MonthEnd(), name="Date"
                ),
            )
            result = df.groupby(Grouper(freq="1ME")).sum(numeric_only=True)
            tm.assert_frame_equal(result, expected)

            result = df.groupby([Grouper(freq="1ME")]).sum(numeric_only=True)
            tm.assert_frame_equal(result, expected)

            expected.index = expected.index.shift(1)
            assert expected.index.freq == offsets.MonthEnd()
            result = df.groupby(Grouper(freq="1ME", key="Date")).sum(numeric_only=True)
            tm.assert_frame_equal(result, expected)

            result = df.groupby([Grouper(freq="1ME", key="Date")]).sum(
                numeric_only=True
            )
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("freq", ["D", "ME", "YE", "QE-APR"])
    def test_timegrouper_with_reg_groups_freq(self, freq):
        # GH 6764 multiple grouping with/without sort
        df = DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "20121002",
                        "20121007",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20121002",
                        "20121207",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20130202",
                        "20130305",
                    ]
                ),
                "user_id": [1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5],
                "whole_cost": [
                    1790,
                    364,
                    280,
                    259,
                    201,
                    623,
                    90,
                    312,
                    359,
                    301,
                    359,
                    801,
                ],
                "cost1": [12, 15, 10, 24, 39, 1, 0, 90, 45, 34, 1, 12],
            }
        ).set_index("date")

        expected = (
            df.groupby("user_id")["whole_cost"]
            .resample(freq)
            .sum(min_count=1)  # XXX
            .dropna()
            .reorder_levels(["date", "user_id"])
            .sort_index()
            .astype("int64")
        )
        expected.name = "whole_cost"

        result1 = (
            df.sort_index().groupby([Grouper(freq=freq), "user_id"])["whole_cost"].sum()
        )
        tm.assert_series_equal(result1, expected)

        result2 = df.groupby([Grouper(freq=freq), "user_id"])["whole_cost"].sum()
        tm.assert_series_equal(result2, expected)

    def test_timegrouper_get_group(self):
        # GH 6914

        df_original = DataFrame(
            {
                "Buyer": "Carl Joe Joe Carl Joe Carl".split(),
                "Quantity": [18, 3, 5, 1, 9, 3],
                "Date": [
                    datetime(2013, 9, 1, 13, 0),
                    datetime(2013, 9, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 3, 10, 0),
                    datetime(2013, 12, 2, 12, 0),
                    datetime(2013, 9, 2, 14, 0),
                ],
            }
        )
        df_reordered = df_original.sort_values(by="Quantity")

        # single grouping
        expected_list = [
            df_original.iloc[[0, 1, 5]],
            df_original.iloc[[2, 3]],
            df_original.iloc[[4]],
        ]
        dt_list = ["2013-09-30", "2013-10-31", "2013-12-31"]

        for df in [df_original, df_reordered]:
            grouped = df.groupby(Grouper(freq="ME", key="Date"))
            for t, expected in zip(dt_list, expected_list):
                dt = Timestamp(t)
                result = grouped.get_group(dt)
                tm.assert_frame_equal(result, expected)

        # multiple grouping
        expected_list = [
            df_original.iloc[[1]],
            df_original.iloc[[3]],
            df_original.iloc[[4]],
        ]
        g_list = [("Joe", "2013-09-30"), ("Carl", "2013-10-31"), ("Joe", "2013-12-31")]

        for df in [df_original, df_reordered]:
            grouped = df.groupby(["Buyer", Grouper(freq="ME", key="Date")])
            for (b, t), expected in zip(g_list, expected_list):
                dt = Timestamp(t)
                result = grouped.get_group((b, dt))
                tm.assert_frame_equal(result, expected)

        # with index
        df_original = df_original.set_index("Date")
        df_reordered = df_original.sort_values(by="Quantity")

        expected_list = [
            df_original.iloc[[0, 1, 5]],
            df_original.iloc[[2, 3]],
            df_original.iloc[[4]],
        ]

        for df in [df_original, df_reordered]:
            grouped = df.groupby(Grouper(freq="ME"))
            for t, expected in zip(dt_list, expected_list):
                dt = Timestamp(t)
                result = grouped.get_group(dt)
                tm.assert_frame_equal(result, expected)

    def test_timegrouper_apply_return_type_series(self):
        # Using `apply` with the `TimeGrouper` should give the
        # same return type as an `apply` with a `Grouper`.
        # Issue #11742
        df = DataFrame({"date": ["10/10/2000", "11/10/2000"], "value": [10, 13]})
        df_dt = df.copy()
        df_dt["date"] = pd.to_datetime(df_dt["date"])

        def sumfunc_series(x):
            return Series([x["value"].sum()], ("sum",))

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df.groupby(Grouper(key="date")).apply(sumfunc_series)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df_dt.groupby(Grouper(freq="ME", key="date")).apply(sumfunc_series)
        tm.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_timegrouper_apply_return_type_value(self):
        # Using `apply` with the `TimeGrouper` should give the
        # same return type as an `apply` with a `Grouper`.
        # Issue #11742
        df = DataFrame({"date": ["10/10/2000", "11/10/2000"], "value": [10, 13]})
        df_dt = df.copy()
        df_dt["date"] = pd.to_datetime(df_dt["date"])

        def sumfunc_value(x):
            return x.value.sum()

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df.groupby(Grouper(key="date")).apply(sumfunc_value)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df_dt.groupby(Grouper(freq="ME", key="date")).apply(sumfunc_value)
        tm.assert_series_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_groupby_groups_datetimeindex(self):
        # GH#1430
        periods = 1000
        ind = date_range(start="2012/1/1", freq="5min", periods=periods)
        df = DataFrame(
            {"high": np.arange(periods), "low": np.arange(periods)}, index=ind
        )
        grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))

        # it works!
        groups = grouped.groups
        assert isinstance(next(iter(groups.keys())), datetime)

    def test_groupby_groups_datetimeindex2(self):
        # GH#11442
        index = date_range("2015/01/01", periods=5, name="date")
        df = DataFrame({"A": [5, 6, 7, 8, 9], "B": [1, 2, 3, 4, 5]}, index=index)
        result = df.groupby(level="date").groups
        dates = ["2015-01-05", "2015-01-04", "2015-01-03", "2015-01-02", "2015-01-01"]
        expected = {
            Timestamp(date): DatetimeIndex([date], name="date") for date in dates
        }
        tm.assert_dict_equal(result, expected)

        grouped = df.groupby(level="date")
        for date in dates:
            result = grouped.get_group(date)
            data = [[df.loc[date, "A"], df.loc[date, "B"]]]
            expected_index = DatetimeIndex(
                [date], name="date", freq="D", dtype=index.dtype
            )
            expected = DataFrame(data, columns=list("AB"), index=expected_index)
            tm.assert_frame_equal(result, expected)

    def test_groupby_groups_datetimeindex_tz(self):
        # GH 3950
        dates = [
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
        ]
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "datetime": dates,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )
        df["datetime"] = df["datetime"].apply(lambda d: Timestamp(d, tz="US/Pacific"))

        exp_idx1 = DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 09:00:00",
            ],
            tz="US/Pacific",
            name="datetime",
        )
        exp_idx2 = Index(["a", "b"] * 3, name="label")
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        expected = DataFrame(
            {"value1": [0, 3, 1, 4, 2, 5], "value2": [1, 2, 2, 1, 1, 2]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        result = df.groupby(["datetime", "label"]).sum()
        tm.assert_frame_equal(result, expected)

        # by level
        didx = DatetimeIndex(dates, tz="Asia/Tokyo")
        df = DataFrame(
            {"value1": np.arange(6, dtype="int64"), "value2": [1, 2, 3, 1, 2, 3]},
            index=didx,
        )

        exp_idx = DatetimeIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            tz="Asia/Tokyo",
        )
        expected = DataFrame(
            {"value1": [3, 5, 7], "value2": [2, 4, 6]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        result = df.groupby(level=0).sum()
        tm.assert_frame_equal(result, expected)

    def test_frame_datetime64_handling_groupby(self):
        # it works!
        df = DataFrame(
            [(3, np.datetime64("2012-07-03")), (3, np.datetime64("2012-07-04"))],
            columns=["a", "date"],
        )
        result = df.groupby("a").first()
        assert result["date"][3] == Timestamp("2012-07-03")

    def test_groupby_multi_timezone(self):
        # combining multiple / different timezones yields UTC
        df = DataFrame(
            {
                "value": range(5),
                "date": [
                    "2000-01-28 16:47:00",
                    "2000-01-29 16:48:00",
                    "2000-01-30 16:49:00",
                    "2000-01-31 16:50:00",
                    "2000-01-01 16:50:00",
                ],
                "tz": [
                    "America/Chicago",
                    "America/Chicago",
                    "America/Los_Angeles",
                    "America/Chicago",
                    "America/New_York",
                ],
            }
        )

        result = df.groupby("tz", group_keys=False).date.apply(
            lambda x: pd.to_datetime(x).dt.tz_localize(x.name)
        )

        expected = Series(
            [
                Timestamp("2000-01-28 16:47:00-0600", tz="America/Chicago"),
                Timestamp("2000-01-29 16:48:00-0600", tz="America/Chicago"),
                Timestamp("2000-01-30 16:49:00-0800", tz="America/Los_Angeles"),
                Timestamp("2000-01-31 16:50:00-0600", tz="America/Chicago"),
                Timestamp("2000-01-01 16:50:00-0500", tz="America/New_York"),
            ],
            name="date",
            dtype=object,
        )
        tm.assert_series_equal(result, expected)

        tz = "America/Chicago"
        res_values = df.groupby("tz").date.get_group(tz)
        result = pd.to_datetime(res_values).dt.tz_localize(tz)
        exp_values = Series(
            ["2000-01-28 16:47:00", "2000-01-29 16:48:00", "2000-01-31 16:50:00"],
            index=[0, 1, 3],
            name="date",
        )
        expected = pd.to_datetime(exp_values).dt.tz_localize(tz)
        tm.assert_series_equal(result, expected)

    def test_groupby_groups_periods(self):
        dates = [
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
        ]
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "period": [pd.Period(d, freq="h") for d in dates],
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )

        exp_idx1 = pd.PeriodIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 09:00:00",
            ],
            freq="h",
            name="period",
        )
        exp_idx2 = Index(["a", "b"] * 3, name="label")
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        expected = DataFrame(
            {"value1": [0, 3, 1, 4, 2, 5], "value2": [1, 2, 2, 1, 1, 2]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        result = df.groupby(["period", "label"]).sum()
        tm.assert_frame_equal(result, expected)

        # by level
        didx = pd.PeriodIndex(dates, freq="h")
        df = DataFrame(
            {"value1": np.arange(6, dtype="int64"), "value2": [1, 2, 3, 1, 2, 3]},
            index=didx,
        )

        exp_idx = pd.PeriodIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            freq="h",
        )
        expected = DataFrame(
            {"value1": [3, 5, 7], "value2": [2, 4, 6]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        result = df.groupby(level=0).sum()
        tm.assert_frame_equal(result, expected)

    def test_groupby_first_datetime64(self):
        df = DataFrame([(1, 1351036800000000000), (2, 1351036800000000000)])
        df[1] = df[1].astype("M8[ns]")

        assert issubclass(df[1].dtype.type, np.datetime64)

        result = df.groupby(level=0).first()
        got_dt = result[1].dtype
        assert issubclass(got_dt.type, np.datetime64)

        result = df[1].groupby(level=0).first()
        got_dt = result.dtype
        assert issubclass(got_dt.type, np.datetime64)

    def test_groupby_max_datetime64(self):
        # GH 5869
        # datetimelike dtype conversion from int
        df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
        # TODO: can we retain second reso in .apply here?
        expected = df.groupby("A")["A"].apply(lambda x: x.max()).astype("M8[s]")
        result = df.groupby("A")["A"].max()
        tm.assert_series_equal(result, expected)

    def test_groupby_datetime64_32_bit(self):
        # GH 6410 / numpy 4328
        # 32-bit under 1.9-dev indexing issue

        df = DataFrame({"A": range(2), "B": [Timestamp("2000-01-1")] * 2})
        result = df.groupby("A")["B"].transform("min")
        expected = Series([Timestamp("2000-01-1")] * 2, name="B")
        tm.assert_series_equal(result, expected)

    def test_groupby_with_timezone_selection(self):
        # GH 11616
        # Test that column selection returns output in correct timezone.

        df = DataFrame(
            {
                "factor": np.random.default_rng(2).integers(0, 3, size=60),
                "time": date_range("01/01/2000 00:00", periods=60, freq="s", tz="UTC"),
            }
        )
        df1 = df.groupby("factor").max()["time"]
        df2 = df.groupby("factor")["time"].max()
        tm.assert_series_equal(df1, df2)

    def test_timezone_info(self):
        # see gh-11682: Timezone info lost when broadcasting
        # scalar datetime to DataFrame

        df = DataFrame({"a": [1], "b": [datetime.now(pytz.utc)]})
        assert df["b"][0].tzinfo == pytz.utc
        df = DataFrame({"a": [1, 2, 3]})
        df["b"] = datetime.now(pytz.utc)
        assert df["b"][0].tzinfo == pytz.utc

    def test_datetime_count(self):
        df = DataFrame(
            {"a": [1, 2, 3] * 2, "dates": date_range("now", periods=6, freq="min")}
        )
        result = df.groupby("a").dates.count()
        expected = Series([2, 2, 2], index=Index([1, 2, 3], name="a"), name="dates")
        tm.assert_series_equal(result, expected)

    def test_first_last_max_min_on_time_data(self):
        # GH 10295
        # Verify that NaT is not in the result of max, min, first and last on
        # Dataframe with datetime or timedelta values.
        df_test = DataFrame(
            {
                "dt": [
                    np.nan,
                    "2015-07-24 10:10",
                    "2015-07-25 11:11",
                    "2015-07-23 12:12",
                    np.nan,
                ],
                "td": [
                    np.nan,
                    timedelta(days=1),
                    timedelta(days=2),
                    timedelta(days=3),
                    np.nan,
                ],
            }
        )
        df_test.dt = pd.to_datetime(df_test.dt)
        df_test["group"] = "A"
        df_ref = df_test[df_test.dt.notna()]

        grouped_test = df_test.groupby("group")
        grouped_ref = df_ref.groupby("group")

        tm.assert_frame_equal(grouped_ref.max(), grouped_test.max())
        tm.assert_frame_equal(grouped_ref.min(), grouped_test.min())
        tm.assert_frame_equal(grouped_ref.first(), grouped_test.first())
        tm.assert_frame_equal(grouped_ref.last(), grouped_test.last())

    def test_nunique_with_timegrouper_and_nat(self):
        # GH 17575
        test = DataFrame(
            {
                "time": [
                    Timestamp("2016-06-28 09:35:35"),
                    pd.NaT,
                    Timestamp("2016-06-28 16:46:28"),
                ],
                "data": ["1", "2", "3"],
            }
        )

        grouper = Grouper(key="time", freq="h")
        result = test.groupby(grouper)["data"].nunique()
        expected = test[test.time.notnull()].groupby(grouper)["data"].nunique()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    def test_scalar_call_versus_list_call(self):
        # Issue: 17530
        data_frame = {
            "location": ["shanghai", "beijing", "shanghai"],
            "time": Series(
                ["2017-08-09 13:32:23", "2017-08-11 23:23:15", "2017-08-11 22:23:15"],
                dtype="datetime64[ns]",
            ),
            "value": [1, 2, 3],
        }
        data_frame = DataFrame(data_frame).set_index("time")
        grouper = Grouper(freq="D")

        grouped = data_frame.groupby(grouper)
        result = grouped.count()
        grouped = data_frame.groupby([grouper])
        expected = grouped.count()

        tm.assert_frame_equal(result, expected)

    def test_grouper_period_index(self):
        # GH 32108
        periods = 2
        index = pd.period_range(
            start="2018-01", periods=periods, freq="M", name="Month"
        )
        period_series = Series(range(periods), index=index)
        result = period_series.groupby(period_series.index.month).sum()

        expected = Series(
            range(periods), index=Index(range(1, periods + 1), name=index.name)
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_apply_timegrouper_with_nat_dict_returns(
        self, groupby_with_truncated_bingrouper
    ):
        # GH#43500 case where gb._grouper.result_index and gb._grouper.group_keys_seq
        #  have different lengths that goes through the `isinstance(values[0], dict)`
        #  path
        gb = groupby_with_truncated_bingrouper

        res = gb["Quantity"].apply(lambda x: {"foo": len(x)})

        df = gb.obj
        unit = df["Date"]._values.unit
        dti = date_range("2013-09-01", "2013-10-01", freq="5D", name="Date", unit=unit)
        mi = MultiIndex.from_arrays([dti, ["foo"] * len(dti)])
        expected = Series([3, 0, 0, 0, 0, 0, 2], index=mi, name="Quantity")
        tm.assert_series_equal(res, expected)

    def test_groupby_apply_timegrouper_with_nat_scalar_returns(
        self, groupby_with_truncated_bingrouper
    ):
        # GH#43500 Previously raised ValueError bc used index with incorrect
        #  length in wrap_applied_result
        gb = groupby_with_truncated_bingrouper

        res = gb["Quantity"].apply(lambda x: x.iloc[0] if len(x) else np.nan)

        df = gb.obj
        unit = df["Date"]._values.unit
        dti = date_range("2013-09-01", "2013-10-01", freq="5D", name="Date", unit=unit)
        expected = Series(
            [18, np.nan, np.nan, np.nan, np.nan, np.nan, 5],
            index=dti._with_freq(None),
            name="Quantity",
        )

        tm.assert_series_equal(res, expected)

    def test_groupby_apply_timegrouper_with_nat_apply_squeeze(
        self, frame_for_truncated_bingrouper
    ):
        df = frame_for_truncated_bingrouper

        # We need to create a GroupBy object with only one non-NaT group,
        #  so use a huge freq so that all non-NaT dates will be grouped together
        tdg = Grouper(key="Date", freq="100YE")
        gb = df.groupby(tdg)

        # check that we will go through the singular_series path
        #  in _wrap_applied_output_series
        assert gb.ngroups == 1
        assert gb._selected_obj._get_axis(gb.axis).nlevels == 1

        # function that returns a Series
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = gb.apply(lambda x: x["Quantity"] * 2)

        dti = Index([Timestamp("2013-12-31")], dtype=df["Date"].dtype, name="Date")
        expected = DataFrame(
            [[36, 6, 6, 10, 2]],
            index=dti,
            columns=Index([0, 1, 5, 2, 3], name="Quantity"),
        )
        tm.assert_frame_equal(res, expected)

    @pytest.mark.single_cpu
    def test_groupby_agg_numba_timegrouper_with_nat(
        self, groupby_with_truncated_bingrouper
    ):
        pytest.importorskip("numba")

        # See discussion in GH#43487
        gb = groupby_with_truncated_bingrouper

        result = gb["Quantity"].aggregate(
            lambda values, index: np.nanmean(values), engine="numba"
        )

        expected = gb["Quantity"].aggregate("mean")
        tm.assert_series_equal(result, expected)

        result_df = gb[["Quantity"]].aggregate(
            lambda values, index: np.nanmean(values), engine="numba"
        )
        expected_df = gb[["Quantity"]].aggregate("mean")
        tm.assert_frame_equal(result_df, expected_df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_f2py2e.py ===
import platform
import re
import shlex
import subprocess
import sys
import textwrap
from collections import namedtuple
from pathlib import Path

import pytest

from numpy.f2py.f2py2e import main as f2pycli
from numpy.testing._private.utils import NOGIL_BUILD

from . import util

#######################
# F2PY Test utilities #
######################

# Tests for CLI commands which call meson will fail if no compilers are present, these are to be skipped

def compiler_check_f2pycli():
    if not util.has_fortran_compiler():
        pytest.skip("CLI command needs a Fortran compiler")
    else:
        f2pycli()

#########################
# CLI utils and classes #
#########################


PPaths = namedtuple("PPaths", "finp, f90inp, pyf, wrap77, wrap90, cmodf")


def get_io_paths(fname_inp, mname="untitled"):
    """Takes in a temporary file for testing and returns the expected output and input paths

    Here expected output is essentially one of any of the possible generated
    files.

    ..note::

         Since this does not actually run f2py, none of these are guaranteed to
         exist, and module names are typically incorrect

    Parameters
    ----------
    fname_inp : str
                The input filename
    mname : str, optional
                The name of the module, untitled by default

    Returns
    -------
    genp : NamedTuple PPaths
            The possible paths which are generated, not all of which exist
    """
    bpath = Path(fname_inp)
    return PPaths(
        finp=bpath.with_suffix(".f"),
        f90inp=bpath.with_suffix(".f90"),
        pyf=bpath.with_suffix(".pyf"),
        wrap77=bpath.with_name(f"{mname}-f2pywrappers.f"),
        wrap90=bpath.with_name(f"{mname}-f2pywrappers2.f90"),
        cmodf=bpath.with_name(f"{mname}module.c"),
    )


################
# CLI Fixtures #
################


@pytest.fixture(scope="session")
def hello_world_f90(tmpdir_factory):
    """Generates a single f90 file for testing"""
    fdat = util.getpath("tests", "src", "cli", "hiworld.f90").read_text()
    fn = tmpdir_factory.getbasetemp() / "hello.f90"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def gh23598_warn(tmpdir_factory):
    """F90 file for testing warnings in gh23598"""
    fdat = util.getpath("tests", "src", "crackfortran", "gh23598Warn.f90").read_text()
    fn = tmpdir_factory.getbasetemp() / "gh23598Warn.f90"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def gh22819_cli(tmpdir_factory):
    """F90 file for testing disallowed CLI arguments in ghff819"""
    fdat = util.getpath("tests", "src", "cli", "gh_22819.pyf").read_text()
    fn = tmpdir_factory.getbasetemp() / "gh_22819.pyf"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def hello_world_f77(tmpdir_factory):
    """Generates a single f77 file for testing"""
    fdat = util.getpath("tests", "src", "cli", "hi77.f").read_text()
    fn = tmpdir_factory.getbasetemp() / "hello.f"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def retreal_f77(tmpdir_factory):
    """Generates a single f77 file for testing"""
    fdat = util.getpath("tests", "src", "return_real", "foo77.f").read_text()
    fn = tmpdir_factory.getbasetemp() / "foo.f"
    fn.write_text(fdat, encoding="ascii")
    return fn

@pytest.fixture(scope="session")
def f2cmap_f90(tmpdir_factory):
    """Generates a single f90 file for testing"""
    fdat = util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90").read_text()
    f2cmap = util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap").read_text()
    fn = tmpdir_factory.getbasetemp() / "f2cmap.f90"
    fmap = tmpdir_factory.getbasetemp() / "mapfile"
    fn.write_text(fdat, encoding="ascii")
    fmap.write_text(f2cmap, encoding="ascii")
    return fn

#########
# Tests #
#########

def test_gh22819_cli(capfd, gh22819_cli, monkeypatch):
    """Check that module names are handled correctly
    gh-22819
    Essentially, the -m name cannot be used to import the module, so the module
    named in the .pyf needs to be used instead

    CLI :: -m and a .pyf file
    """
    ipath = Path(gh22819_cli)
    monkeypatch.setattr(sys, "argv", f"f2py -m blah {ipath}".split())
    with util.switchdir(ipath.parent):
        f2pycli()
        gen_paths = [item.name for item in ipath.parent.rglob("*") if item.is_file()]
        assert "blahmodule.c" not in gen_paths  # shouldn't be generated
        assert "blah-f2pywrappers.f" not in gen_paths
        assert "test_22819-f2pywrappers.f" in gen_paths
        assert "test_22819module.c" in gen_paths


def test_gh22819_many_pyf(capfd, gh22819_cli, monkeypatch):
    """Only one .pyf file allowed
    gh-22819
    CLI :: .pyf files
    """
    ipath = Path(gh22819_cli)
    monkeypatch.setattr(sys, "argv", f"f2py -m blah {ipath} hello.pyf".split())
    with util.switchdir(ipath.parent):
        with pytest.raises(ValueError, match="Only one .pyf file per call"):
            f2pycli()


def test_gh23598_warn(capfd, gh23598_warn, monkeypatch):
    foutl = get_io_paths(gh23598_warn, mname="test")
    ipath = foutl.f90inp
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test'.split())

    with util.switchdir(ipath.parent):
        f2pycli()  # Generate files
        wrapper = foutl.wrap90.read_text()
        assert "intproductf2pywrap, intpr" not in wrapper


def test_gen_pyf(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file is generated via the CLI
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    opath = Path(hello_world_f90).stem + ".pyf"
    monkeypatch.setattr(sys, "argv", f'f2py -h {opath} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()  # Generate wrappers
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out
        assert Path(f'{opath}').exists()


def test_gen_pyf_stdout(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file can be dumped to stdout
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -h stdout {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out
        assert "function hi() ! in " in out


def test_gen_pyf_no_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the CLI refuses to overwrite signature files
    CLI :: -h without --overwrite-signature
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -h faker.pyf {ipath}'.split())

    with util.switchdir(ipath.parent):
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        with pytest.raises(SystemExit):
            f2pycli()  # Refuse to overwrite
            _, err = capfd.readouterr()
            assert "Use --overwrite-signature to overwrite" in err


@pytest.mark.skipif(sys.version_info <= (3, 12), reason="Python 3.12 required")
def test_untitled_cli(capfd, hello_world_f90, monkeypatch):
    """Check that modules are named correctly

    CLI :: defaults
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f"f2py --backend meson -c {ipath}".split())
    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        out, _ = capfd.readouterr()
        assert "untitledmodule.c" in out


@pytest.mark.skipif((platform.system() != 'Linux') or (sys.version_info <= (3, 12)), reason='Compiler and 3.12 required')
def test_no_py312_distutils_fcompiler(capfd, hello_world_f90, monkeypatch):
    """Check that no distutils imports are performed on 3.12
    CLI :: --fcompiler --help-link --backend distutils
    """
    MNAME = "hi"
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    monkeypatch.setattr(
        sys, "argv", f"f2py {ipath} -c --fcompiler=gfortran -m {MNAME}".split()
    )
    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        out, _ = capfd.readouterr()
        assert "--fcompiler cannot be used with meson" in out
    monkeypatch.setattr(
        sys, "argv", ["f2py", "--help-link"]
    )
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Use --dep for meson builds" in out
    MNAME = "hi2"  # Needs to be different for a new -c
    monkeypatch.setattr(
        sys, "argv", f"f2py {ipath} -c -m {MNAME} --backend distutils".split()
    )
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Cannot use distutils backend with Python>=3.12" in out


@pytest.mark.xfail
def test_f2py_skip(capfd, retreal_f77, monkeypatch):
    """Tests that functions can be skipped
    CLI :: skip:
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    remaining = "td s0"
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test skip: {toskip}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not found the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in remaining.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_f2py_only(capfd, retreal_f77, monkeypatch):
    """Test that functions can be kept by only:
    CLI :: only:
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    tokeep = "td s0"
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test only: {tokeep}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_file_processing_switch(capfd, hello_world_f90, retreal_f77,
                                monkeypatch):
    """Tests that it is possible to return to file processing mode
    CLI :: :
    BUG: numpy-gh #20520
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    ipath2 = Path(hello_world_f90)
    tokeep = "td s0 hi"  # hi is in ipath2
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -m {mname} only: {tokeep} : {ipath2}'.split(
        ),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_mod_gen_f77(capfd, hello_world_f90, monkeypatch):
    """Checks the generation of files based on a module name
    CLI :: -m
    """
    MNAME = "hi"
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m {MNAME}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()

    # Always generate C module
    assert Path.exists(foutl.cmodf)
    # File contains a function, check for F77 wrappers
    assert Path.exists(foutl.wrap77)


def test_mod_gen_gh25263(capfd, hello_world_f77, monkeypatch):
    """Check that pyf files are correctly generated with module structure
    CLI :: -m <name> -h pyf_file
    BUG: numpy-gh #20520
    """
    MNAME = "hi"
    foutl = get_io_paths(hello_world_f77, mname=MNAME)
    ipath = foutl.finp
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m {MNAME} -h hi.pyf'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        with Path('hi.pyf').open() as hipyf:
            pyfdat = hipyf.read()
            assert "python module hi" in pyfdat


def test_lower_cmod(capfd, hello_world_f77, monkeypatch):
    """Lowers cases by flag or when -h is present

    CLI :: --[no-]lower
    """
    foutl = get_io_paths(hello_world_f77, mname="test")
    ipath = foutl.finp
    capshi = re.compile(r"HI\(\)")
    capslo = re.compile(r"hi\(\)")
    # Case I: --lower is passed
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m test --lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None
    # Case II: --no-lower is passed
    monkeypatch.setattr(sys, "argv",
                        f'f2py {ipath} -m test --no-lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None


def test_lower_sig(capfd, hello_world_f77, monkeypatch):
    """Lowers cases in signature files by flag or when -h is present

    CLI :: --[no-]lower -h
    """
    foutl = get_io_paths(hello_world_f77, mname="test")
    ipath = foutl.finp
    # Signature files
    capshi = re.compile(r"Block: HI")
    capslo = re.compile(r"Block: hi")
    # Case I: --lower is implied by -h
    # TODO: Clean up to prevent passing --overwrite-signature
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature'.split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None

    # Case II: --no-lower overrides -h
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature --no-lower'
        .split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None


def test_build_dir(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --build-dir
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    odir = "tttmp"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --build-dir {odir}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert f"Wrote C/API module \"{mname}\"" in out


def test_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --overwrite-signature
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(
        sys, "argv",
        f'f2py -h faker.pyf {ipath} --overwrite-signature'.split())

    with util.switchdir(ipath.parent):
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out


def test_latexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --latex-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --latex-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" in out
        with Path(f"{mname}module.tex").open() as otex:
            assert "\\documentclass" in otex.read()


def test_nolatexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-latex-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-latex-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" not in out


def test_shortlatex(capfd, hello_world_f90, monkeypatch):
    """Ensures that truncated documentation is written out

    TODO: Test to ensure this has no effect without --latex-doc
    CLI :: --latex-doc --short-latex
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} --latex-doc --short-latex'.split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" in out
        with Path(f"./{mname}module.tex").open() as otex:
            assert "\\documentclass" not in otex.read()


def test_restdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that RsT documentation is written out

    CLI :: --rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --rest-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "ReST Documentation is saved to file" in out
        with Path(f"./{mname}module.rest").open() as orst:
            assert r".. -*- rest -*-" in orst.read()


def test_norestexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-rest-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "ReST Documentation is saved to file" not in out


def test_debugcapi(capfd, hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers are written

    CLI :: --debug-capi
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --debug-capi'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f"./{mname}module.c").open() as ocmod:
            assert r"#define DEBUGCFUNCS" in ocmod.read()


@pytest.mark.skip(reason="Consistently fails on CI; noisy so skip not xfail.")
def test_debugcapi_bld(hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers work

    CLI :: --debug-capi -c
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} -c --debug-capi'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split(f"{sys.executable} -c \"import blah; blah.hi()\"")
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        eerr = textwrap.dedent("""\
debug-capi:Python C/API function blah.hi()
debug-capi:float hi=:output,hidden,scalar
debug-capi:hi=0
debug-capi:Fortran subroutine `f2pywraphi(&hi)'
debug-capi:hi=0
debug-capi:Building return value.
debug-capi:Python C/API function blah.hi: successful.
debug-capi:Freeing memory.
        """)
        assert rout.stdout == eout
        assert rout.stderr == eerr


def test_wrapfunc_def(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 are included by default

    CLI :: --[no]-wrap-functions
    """
    # Implied
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv", f'f2py -m {mname} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
    out, _ = capfd.readouterr()
    assert r"Fortran 77 wrappers are saved to" in out

    # Explicit
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" in out


def test_nowrapfunc(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 can be disabled

    CLI :: --no-wrap-functions
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" not in out


def test_inclheader(capfd, hello_world_f90, monkeypatch):
    """Add to the include directories

    CLI :: -include
    TODO: Document this in the help string
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} -include<stdbool.h> -include<stdio.h> '.
        split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f"./{mname}module.c").open() as ocmod:
            ocmr = ocmod.read()
            assert "#include <stdbool.h>" in ocmr
            assert "#include <stdio.h>" in ocmr


def test_inclpath():
    """Add to the include directories

    CLI :: --include-paths
    """
    # TODO: populate
    pass


def test_hlink():
    """Add to the include directories

    CLI :: --help-link
    """
    # TODO: populate
    pass


def test_f2cmap(capfd, f2cmap_f90, monkeypatch):
    """Check that Fortran-to-Python KIND specs can be passed

    CLI :: --f2cmap
    """
    ipath = Path(f2cmap_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --f2cmap mapfile'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Reading f2cmap from 'mapfile' ..." in out
        assert "Mapping \"real(kind=real32)\" to \"float\"" in out
        assert "Mapping \"real(kind=real64)\" to \"double\"" in out
        assert "Mapping \"integer(kind=int64)\" to \"long_long\"" in out
        assert "Successfully applied user defined f2cmap changes" in out


def test_quiet(capfd, hello_world_f90, monkeypatch):
    """Reduce verbosity

    CLI :: --quiet
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --quiet'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert len(out) == 0


def test_verbose(capfd, hello_world_f90, monkeypatch):
    """Increase verbosity

    CLI :: --verbose
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --verbose'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "analyzeline" in out


def test_version(capfd, monkeypatch):
    """Ensure version

    CLI :: -v
    """
    monkeypatch.setattr(sys, "argv", ["f2py", "-v"])
    # TODO: f2py2e should not call sys.exit() after printing the version
    with pytest.raises(SystemExit):
        f2pycli()
        out, _ = capfd.readouterr()
        import numpy as np
        assert np.__version__ == out.strip()


@pytest.mark.skip(reason="Consistently fails on CI; noisy so skip not xfail.")
def test_npdistop(hello_world_f90, monkeypatch):
    """
    CLI :: -c
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split(f"{sys.executable} -c \"import blah; blah.hi()\"")
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout


@pytest.mark.skipif((platform.system() != 'Linux') or sys.version_info <= (3, 12),
                    reason='Compiler and Python 3.12 or newer required')
def test_no_freethreading_compatible(hello_world_f90, monkeypatch):
    """
    CLI :: --no-freethreading-compatible
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c --no-freethreading-compatible'.split())

    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        cmd = f"{sys.executable} -c \"import blah; blah.hi();"
        if NOGIL_BUILD:
            cmd += "import sys; assert sys._is_gil_enabled() is True\""
        else:
            cmd += "\""
        cmd_run = shlex.split(cmd)
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout
        if NOGIL_BUILD:
            assert "The global interpreter lock (GIL) has been enabled to load module 'blah'" in rout.stderr
        assert rout.returncode == 0


@pytest.mark.skipif((platform.system() != 'Linux') or sys.version_info <= (3, 12),
                    reason='Compiler and Python 3.12 or newer required')
def test_freethreading_compatible(hello_world_f90, monkeypatch):
    """
    CLI :: --freethreading_compatible
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c --freethreading-compatible'.split())

    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        cmd = f"{sys.executable} -c \"import blah; blah.hi();"
        if NOGIL_BUILD:
            cmd += "import sys; assert sys._is_gil_enabled() is False\""
        else:
            cmd += "\""
        cmd_run = shlex.split(cmd)
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout
        assert rout.stderr == ""
        assert rout.returncode == 0


# Numpy distutils flags
# TODO: These should be tested separately

def test_npd_fcompiler():
    """
    CLI :: -c --fcompiler
    """
    # TODO: populate
    pass


def test_npd_compiler():
    """
    CLI :: -c --compiler
    """
    # TODO: populate
    pass


def test_npd_help_fcompiler():
    """
    CLI :: -c --help-fcompiler
    """
    # TODO: populate
    pass


def test_npd_f77exec():
    """
    CLI :: -c --f77exec
    """
    # TODO: populate
    pass


def test_npd_f90exec():
    """
    CLI :: -c --f90exec
    """
    # TODO: populate
    pass


def test_npd_f77flags():
    """
    CLI :: -c --f77flags
    """
    # TODO: populate
    pass


def test_npd_f90flags():
    """
    CLI :: -c --f90flags
    """
    # TODO: populate
    pass


def test_npd_opt():
    """
    CLI :: -c --opt
    """
    # TODO: populate
    pass


def test_npd_arch():
    """
    CLI :: -c --arch
    """
    # TODO: populate
    pass


def test_npd_noopt():
    """
    CLI :: -c --noopt
    """
    # TODO: populate
    pass


def test_npd_noarch():
    """
    CLI :: -c --noarch
    """
    # TODO: populate
    pass


def test_npd_debug():
    """
    CLI :: -c --debug
    """
    # TODO: populate
    pass


def test_npd_link_auto():
    """
    CLI :: -c --link-<resource>
    """
    # TODO: populate
    pass


def test_npd_lib():
    """
    CLI :: -c -L/path/to/lib/ -l<libname>
    """
    # TODO: populate
    pass


def test_npd_define():
    """
    CLI :: -D<define>
    """
    # TODO: populate
    pass


def test_npd_undefine():
    """
    CLI :: -U<name>
    """
    # TODO: populate
    pass


def test_npd_incl():
    """
    CLI :: -I/path/to/include/
    """
    # TODO: populate
    pass


def test_npd_linker():
    """
    CLI :: <filename>.o <filename>.so <filename>.a
    """
    # TODO: populate
    pass

# === atelier-kyo-manager/test_rembg.py ===
# test_rembg.py  ── そのまま保存して実行してください ───────────────
# rembg を使い 1 枚だけ背景除去 → REMBG_*.png を同じフォルダに書き出します
# ---------------------------------------------------------------
# 事前準備:  pip install rembg pillow

from pathlib import Path
from PIL import Image
from rembg import remove

# ★★ テストしたい JPEG フルパスをここに貼り付ける ★★
TARGET = Path(r"D:\catalog_images\GUCCI\80910\catalog_80910\93b42ae4-302b-4efb-bd8e-712e7c78b6e5_20250516022114122640.jpg")

# ---------------------------------------------------------------
if not TARGET.exists():
    raise FileNotFoundError(f"ファイルが見つかりません:\n{TARGET}")

print(f"[INFO] 入力: {TARGET}")
print(f"[INFO] サイズ: {TARGET.stat().st_size // 1024} KB")

# 背景除去
img_in   = Image.open(TARGET).convert("RGBA")   # CMYK 画像対策で convert
img_out  = remove(img_in)

# 出力ファイル名 例: REMBG_93b42ae4-....png
OUT = TARGET.parent / f"REMBG_{TARGET.stem}.png"
img_out.save(OUT)

print(f"[OK] 出力: {OUT}")
print(f"[OK] サイズ: {OUT.stat().st_size // 1024} KB")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\_numba\kernels\shared.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

import numba

if TYPE_CHECKING:
    import numpy as np


@numba.jit(
    # error: Any? not callable
    numba.boolean(numba.int64[:]),  # type: ignore[misc]
    nopython=True,
    nogil=True,
    parallel=False,
)
def is_monotonic_increasing(bounds: np.ndarray) -> bool:
    """Check if int64 values are monotonically increasing."""
    n = len(bounds)
    if n < 2:
        return True
    prev = bounds[0]
    for i in range(1, n):
        cur = bounds[i]
        if cur < prev:
            return False
        prev = cur
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\__init__.py ===
def __getattr__(key: str):
    # These imports need to be lazy to avoid circular import errors
    if key == "hash_array":
        from pandas.core.util.hashing import hash_array

        return hash_array
    if key == "hash_pandas_object":
        from pandas.core.util.hashing import hash_pandas_object

        return hash_pandas_object
    if key == "Appender":
        from pandas.util._decorators import Appender

        return Appender
    if key == "Substitution":
        from pandas.util._decorators import Substitution

        return Substitution

    if key == "cache_readonly":
        from pandas.util._decorators import cache_readonly

        return cache_readonly

    raise AttributeError(f"module 'pandas.util' has no attribute '{key}'")


def capitalize_first_letter(s):
    return s[:1].upper() + s[1:]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_testing\compat.py ===
"""
Helpers for sharing tests between DataFrame/Series
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from pandas._typing import DtypeObj


def get_dtype(obj) -> DtypeObj:
    if isinstance(obj, DataFrame):
        # Note: we are assuming only one column
        return obj.dtypes.iat[0]
    else:
        return obj.dtype


def get_obj(df: DataFrame, klass):
    """
    For sharing tests using frame_or_series, either return the DataFrame
    unchanged or return it's first column as a Series.
    """
    if klass is DataFrame:
        return df
    return df._ixs(0, axis=1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\distributions\installed.py ===
from typing import Optional

from pip._internal.distributions.base import AbstractDistribution
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution


class InstalledDistribution(AbstractDistribution):
    """Represents an installed package.

    This does not need any preparation as the required information has already
    been computed.
    """

    @property
    def build_tracker_id(self) -> Optional[str]:
        return None

    def get_metadata_distribution(self) -> BaseDistribution:
        assert self.req.satisfied_by is not None, "not actually installed"
        return self.req.satisfied_by

    def prepare_distribution_metadata(
        self,
        finder: PackageFinder,
        build_isolation: bool,
        check_build_deps: bool,
    ) -> None:
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\cachecontrol\__init__.py ===
# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0

"""CacheControl import Interface.

Make it easy to import from cachecontrol without long namespaces.
"""

__author__ = "Eric Larson"
__email__ = "eric@ionrock.org"
__version__ = "0.14.2"

from pip._vendor.cachecontrol.adapter import CacheControlAdapter
from pip._vendor.cachecontrol.controller import CacheController
from pip._vendor.cachecontrol.wrapper import CacheControl

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "CacheControlAdapter",
    "CacheController",
    "CacheControl",
]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\locator_converter.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from selenium.webdriver.common.by import By


class LocatorConverter:
    def convert(self, by, value):
        # Default conversion logic
        if by == By.ID:
            return By.CSS_SELECTOR, f'[id="{value}"]'
        elif by == By.CLASS_NAME:
            return By.CSS_SELECTOR, f".{value}"
        elif by == By.NAME:
            return By.CSS_SELECTOR, f'[name="{value}"]'
        return by, value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\type_tests\task_status.py ===
"""Check that started() can only be called for TaskStatus[None]."""

from trio import TaskStatus
from typing_extensions import assert_type


def check_status(
    none_status_explicit: TaskStatus[None],
    none_status_implicit: TaskStatus,
    int_status: TaskStatus[int],
) -> None:
    assert_type(none_status_explicit, TaskStatus[None])
    assert_type(none_status_implicit, TaskStatus[None])  # Default typevar
    assert_type(int_status, TaskStatus[int])

    # Omitting the parameter is only allowed for None.
    none_status_explicit.started()
    none_status_implicit.started()
    int_status.started()  # type: ignore

    # Explicit None is allowed.
    none_status_explicit.started(None)
    none_status_implicit.started(None)
    int_status.started(None)  # type: ignore

    none_status_explicit.started(42)  # type: ignore
    none_status_implicit.started(42)  # type: ignore
    int_status.started(42)
    int_status.started(True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_setops.py ===
"""
The tests in this package are to ensure the proper resultant dtypes of
set operations.
"""
from datetime import datetime
import operator

import numpy as np
import pytest

from pandas._libs import lib

from pandas.core.dtypes.cast import find_common_type

from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DatetimeTZDtype,
    Index,
    MultiIndex,
    PeriodDtype,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.api.types import (
    is_signed_integer_dtype,
    pandas_dtype,
)


def equal_contents(arr1, arr2) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)


@pytest.fixture(
    params=tm.ALL_REAL_NUMPY_DTYPES
    + [
        "object",
        "category",
        "datetime64[ns]",
        "timedelta64[ns]",
    ]
)
def any_dtype_for_small_pos_integer_indexes(request):
    """
    Dtypes that can be given to an Index with small positive integers.

    This means that for any dtype `x` in the params list, `Index([1, 2, 3], dtype=x)` is
    valid and gives the correct Index (sub-)class.
    """
    return request.param


def test_union_same_types(index):
    # Union with a non-unique, non-monotonic index raises error
    # Only needed for bool index factory
    idx1 = index.sort_values()
    idx2 = index.sort_values()
    assert idx1.union(idx2).dtype == idx1.dtype


def test_union_different_types(index_flat, index_flat2, request):
    # This test only considers combinations of indices
    # GH 23525
    idx1 = index_flat
    idx2 = index_flat2

    if (
        not idx1.is_unique
        and not idx2.is_unique
        and idx1.dtype.kind == "i"
        and idx2.dtype.kind == "b"
    ) or (
        not idx2.is_unique
        and not idx1.is_unique
        and idx2.dtype.kind == "i"
        and idx1.dtype.kind == "b"
    ):
        # Each condition had idx[1|2].is_monotonic_decreasing
        # but failed when e.g.
        # idx1 = Index(
        # [True, True, True, True, True, True, True, True, False, False], dtype='bool'
        # )
        # idx2 = Index([0, 0, 1, 1, 2, 2], dtype='int64')
        mark = pytest.mark.xfail(
            reason="GH#44000 True==1", raises=ValueError, strict=False
        )
        request.applymarker(mark)

    common_dtype = find_common_type([idx1.dtype, idx2.dtype])

    warn = None
    msg = "'<' not supported between"
    if not len(idx1) or not len(idx2):
        pass
    elif (idx1.dtype.kind == "c" and (not lib.is_np_dtype(idx2.dtype, "iufc"))) or (
        idx2.dtype.kind == "c" and (not lib.is_np_dtype(idx1.dtype, "iufc"))
    ):
        # complex objects non-sortable
        warn = RuntimeWarning
    elif (
        isinstance(idx1.dtype, PeriodDtype) and isinstance(idx2.dtype, CategoricalDtype)
    ) or (
        isinstance(idx2.dtype, PeriodDtype) and isinstance(idx1.dtype, CategoricalDtype)
    ):
        warn = FutureWarning
        msg = r"PeriodDtype\[B\] is deprecated"
        mark = pytest.mark.xfail(
            reason="Warning not produced on all builds",
            raises=AssertionError,
            strict=False,
        )
        request.applymarker(mark)

    any_uint64 = np.uint64 in (idx1.dtype, idx2.dtype)
    idx1_signed = is_signed_integer_dtype(idx1.dtype)
    idx2_signed = is_signed_integer_dtype(idx2.dtype)

    # Union with a non-unique, non-monotonic index raises error
    # This applies to the boolean index
    idx1 = idx1.sort_values()
    idx2 = idx2.sort_values()

    with tm.assert_produces_warning(warn, match=msg):
        res1 = idx1.union(idx2)
        res2 = idx2.union(idx1)

    if any_uint64 and (idx1_signed or idx2_signed):
        assert res1.dtype == np.dtype("O")
        assert res2.dtype == np.dtype("O")
    else:
        assert res1.dtype == common_dtype
        assert res2.dtype == common_dtype


@pytest.mark.parametrize(
    "idx1,idx2",
    [
        (Index(np.arange(5), dtype=np.int64), RangeIndex(5)),
        (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.int64)),
        (Index(np.arange(5), dtype=np.float64), RangeIndex(5)),
        (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.uint64)),
    ],
)
def test_compatible_inconsistent_pairs(idx1, idx2):
    # GH 23525
    res1 = idx1.union(idx2)
    res2 = idx2.union(idx1)

    assert res1.dtype in (idx1.dtype, idx2.dtype)
    assert res2.dtype in (idx1.dtype, idx2.dtype)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("int64", "int64", "int64"),
        ("int64", "uint64", "object"),
        ("int64", "float64", "float64"),
        ("uint64", "float64", "float64"),
        ("uint64", "uint64", "uint64"),
        ("float64", "float64", "float64"),
        ("datetime64[ns]", "int64", "object"),
        ("datetime64[ns]", "uint64", "object"),
        ("datetime64[ns]", "float64", "object"),
        ("datetime64[ns, CET]", "int64", "object"),
        ("datetime64[ns, CET]", "uint64", "object"),
        ("datetime64[ns, CET]", "float64", "object"),
        ("Period[D]", "int64", "object"),
        ("Period[D]", "uint64", "object"),
        ("Period[D]", "float64", "object"),
    ],
)
@pytest.mark.parametrize("names", [("foo", "foo", "foo"), ("foo", "bar", None)])
def test_union_dtypes(left, right, expected, names):
    left = pandas_dtype(left)
    right = pandas_dtype(right)
    a = Index([], dtype=left, name=names[0])
    b = Index([], dtype=right, name=names[1])
    result = a.union(b)
    assert result.dtype == expected
    assert result.name == names[2]

    # Testing name retention
    # TODO: pin down desired dtype; do we want it to be commutative?
    result = a.intersection(b)
    assert result.name == names[2]


@pytest.mark.parametrize("values", [[1, 2, 2, 3], [3, 3]])
def test_intersection_duplicates(values):
    # GH#31326
    a = Index(values)
    b = Index([3, 3])
    result = a.intersection(b)
    expected = Index([3])
    tm.assert_index_equal(result, expected)


class TestSetOps:
    # Set operation tests shared by all indexes in the `index` fixture
    @pytest.mark.parametrize("case", [0.5, "xxx"])
    @pytest.mark.parametrize(
        "method", ["intersection", "union", "difference", "symmetric_difference"]
    )
    def test_set_ops_error_cases(self, case, method, index):
        # non-iterable input
        msg = "Input must be Index or array-like"
        with pytest.raises(TypeError, match=msg):
            getattr(index, method)(case)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_intersection_base(self, index):
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"Not relevant for {type(index).__name__}")

        first = index[:5].unique()
        second = index[:3].unique()
        intersect = first.intersection(second)
        tm.assert_index_equal(intersect, second)

        if isinstance(index.dtype, DatetimeTZDtype):
            # The second.values below will drop tz, so the rest of this test
            #  is not applicable.
            return

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.intersection(case)
            assert equal_contents(result, second)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.intersection([1, 2, 3])

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_union_base(self, index):
        index = index.unique()
        first = index[3:]
        second = index[:5]
        everything = index

        union = first.union(second)
        tm.assert_index_equal(union.sort_values(), everything.sort_values())

        if isinstance(index.dtype, DatetimeTZDtype):
            # The second.values below will drop tz, so the rest of this test
            #  is not applicable.
            return

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.union(case)
            assert equal_contents(result, everything)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.union([1, 2, 3])

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_difference_base(self, sort, index):
        first = index[2:]
        second = index[:4]
        if index.inferred_type == "boolean":
            # i think (TODO: be sure) there assumptions baked in about
            #  the index fixture that don't hold here?
            answer = set(first).difference(set(second))
        elif isinstance(index, CategoricalIndex):
            answer = []
        else:
            answer = index[4:]
        result = first.difference(second, sort)
        assert equal_contents(result, answer)

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.difference(case, sort)
            assert equal_contents(result, answer)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.difference([1, 2, 3], sort)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_symmetric_difference(self, index, using_infer_string, request):
        if (
            using_infer_string
            and index.dtype == "object"
            and index.inferred_type == "string"
        ):
            request.applymarker(pytest.mark.xfail(reason="TODO: infer_string"))
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"Not relevant for {type(index).__name__}")
        if len(index) < 2:
            pytest.skip("Too few values for test")
        if index[0] in index[1:] or index[-1] in index[:-1]:
            # index fixture has e.g. an index of bools that does not satisfy this,
            #  another with [0, 0, 1, 1, 2, 2]
            pytest.skip("Index values no not satisfy test condition.")

        first = index[1:]
        second = index[:-1]
        answer = index[[0, -1]]
        result = first.symmetric_difference(second)
        tm.assert_index_equal(result.sort_values(), answer.sort_values())

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.symmetric_difference(case)
            assert equal_contents(result, answer)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.symmetric_difference([1, 2, 3])

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_corner_union(self, index_flat, fname, sname, expected_name):
        # GH#9943, GH#9862
        # Test unions with various name combinations
        # Do not test MultiIndex or repeats
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # Test copy.union(copy)
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test copy.union(empty)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(copy)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(empty)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_union_unequal(self, index_flat, fname, sname, expected_name):
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # test copy.union(subset) - need sort for unicode and string
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        union = first.union(second).sort_values()
        expected = index.set_names(expected_name).sort_values()
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_corner_intersect(self, index_flat, fname, sname, expected_name):
        # GH#35847
        # Test intersections with various name combinations
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # Test copy.intersection(copy)
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test copy.intersection(empty)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(copy)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(empty)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_intersect_unequal(self, index_flat, fname, sname, expected_name):
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # test copy.intersection(subset) - need sort for unicode and string
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        intersect = first.intersection(second).sort_values()
        expected = index[1:].set_names(expected_name).sort_values()
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_intersection_name_retention_with_nameless(self, index):
        if isinstance(index, MultiIndex):
            index = index.rename(list(range(index.nlevels)))
        else:
            index = index.rename("foo")

        other = np.asarray(index)

        result = index.intersection(other)
        assert result.name == index.name

        # empty other, same dtype
        result = index.intersection(other[:0])
        assert result.name == index.name

        # empty `self`
        result = index[:0].intersection(other)
        assert result.name == index.name

    def test_difference_preserves_type_empty(self, index, sort):
        # GH#20040
        # If taking difference of a set and itself, it
        # needs to preserve the type of the index
        if not index.is_unique:
            pytest.skip("Not relevant since index is not unique")
        result = index.difference(index, sort=sort)
        expected = index[:0]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_name_retention_equals(self, index, names):
        if isinstance(index, MultiIndex):
            names = [[x] * index.nlevels for x in names]
        index = index.rename(names[0])
        other = index.rename(names[1])

        assert index.equals(other)

        result = index.difference(other)
        expected = index[:0].rename(names[2])
        tm.assert_index_equal(result, expected)

    def test_intersection_difference_match_empty(self, index, sort):
        # GH#20040
        # Test that the intersection of an index with an
        # empty index produces the same index as the difference
        # of an index with itself.  Test for all types
        if not index.is_unique:
            pytest.skip("Not relevant because index is not unique")
        inter = index.intersection(index[:0])
        diff = index.difference(index, sort=sort)
        tm.assert_index_equal(inter, diff, exact=True)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(index_flat, sort, method):
    # MultiIndex tested separately in tests.indexes.multi.test_setops
    index = index_flat

    other = index.astype("category")
    exact = "equiv" if isinstance(index, RangeIndex) else True

    result = getattr(index, method)(other, sort=sort)
    expected = getattr(index, method)(index, sort=sort)
    tm.assert_index_equal(result, expected, exact=exact)

    result = getattr(index, method)(other[:5], sort=sort)
    expected = getattr(index, method)(index[:5], sort=sort)
    tm.assert_index_equal(result, expected, exact=exact)


def test_intersection_duplicates_all_indexes(index):
    # GH#38743
    if index.empty:
        # No duplicates in empty indexes
        pytest.skip("Not relevant for empty Index")

    idx = index
    idx_non_unique = idx[[0, 0, 1, 2]]

    assert idx.intersection(idx_non_unique).equals(idx_non_unique.intersection(idx))
    assert idx.intersection(idx_non_unique).is_unique


def test_union_duplicate_index_subsets_of_each_other(
    any_dtype_for_small_pos_integer_indexes,
):
    # GH#31326
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 2, 2, 3], dtype=dtype)
    b = Index([3, 3, 4], dtype=dtype)

    expected = Index([1, 2, 2, 3, 3, 4], dtype=dtype)
    if isinstance(a, CategoricalIndex):
        expected = Index([1, 2, 2, 3, 3, 4])
    result = a.union(b)
    tm.assert_index_equal(result, expected)
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)


def test_union_with_duplicate_index_and_non_monotonic(
    any_dtype_for_small_pos_integer_indexes,
):
    # GH#36289
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 0, 0], dtype=dtype)
    b = Index([0, 1], dtype=dtype)
    expected = Index([0, 0, 1], dtype=dtype)

    result = a.union(b)
    tm.assert_index_equal(result, expected)

    result = b.union(a)
    tm.assert_index_equal(result, expected)


def test_union_duplicate_index_different_dtypes():
    # GH#36289
    a = Index([1, 2, 2, 3])
    b = Index(["1", "0", "0"])
    expected = Index([1, 2, 2, 3, "1", "0", "0"])
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)


def test_union_same_value_duplicated_in_both():
    # GH#36289
    a = Index([0, 0, 1])
    b = Index([0, 0, 1, 2])
    result = a.union(b)
    expected = Index([0, 0, 1, 2])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dup", [1, np.nan])
def test_union_nan_in_both(dup):
    # GH#36289
    a = Index([np.nan, 1, 2, 2])
    b = Index([np.nan, dup, 1, 2])
    result = a.union(b, sort=False)
    expected = Index([np.nan, dup, 1.0, 2.0, 2.0])
    tm.assert_index_equal(result, expected)


def test_union_rangeindex_sort_true():
    # GH 53490
    idx1 = RangeIndex(1, 100, 6)
    idx2 = RangeIndex(1, 50, 3)
    result = idx1.union(idx2, sort=True)
    expected = Index(
        [
            1,
            4,
            7,
            10,
            13,
            16,
            19,
            22,
            25,
            28,
            31,
            34,
            37,
            40,
            43,
            46,
            49,
            55,
            61,
            67,
            73,
            79,
            85,
            91,
            97,
        ]
    )
    tm.assert_index_equal(result, expected)


def test_union_with_duplicate_index_not_subset_and_non_monotonic(
    any_dtype_for_small_pos_integer_indexes,
):
    # GH#36289
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 0, 2], dtype=dtype)
    b = Index([0, 0, 1], dtype=dtype)
    expected = Index([0, 0, 1, 2], dtype=dtype)
    if isinstance(a, CategoricalIndex):
        expected = Index([0, 0, 1, 2])

    result = a.union(b)
    tm.assert_index_equal(result, expected)

    result = b.union(a)
    tm.assert_index_equal(result, expected)


def test_union_int_categorical_with_nan():
    ci = CategoricalIndex([1, 2, np.nan])
    assert ci.categories.dtype.kind == "i"

    idx = Index([1, 2])

    result = idx.union(ci)
    expected = Index([1, 2, np.nan], dtype=np.float64)
    tm.assert_index_equal(result, expected)

    result = ci.union(idx)
    tm.assert_index_equal(result, expected)


class TestSetOpsUnsorted:
    # These may eventually belong in a dtype-specific test_setops, or
    #  parametrized over a more general fixture
    def test_intersect_str_dates(self):
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        index1 = Index(dt_dates, dtype=object)
        index2 = Index(["aa"], dtype=object)
        result = index2.intersection(index1)

        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_intersection(self, index, sort):
        first = index[:20]
        second = index[:10]
        intersect = first.intersection(second, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(intersect.sort_values(), second.sort_values())
        else:
            tm.assert_index_equal(intersect, second)

        # Corner cases
        inter = first.intersection(first, sort=sort)
        assert inter is first

    @pytest.mark.parametrize(
        "index2,keeps_name",
        [
            (Index([3, 4, 5, 6, 7], name="index"), True),  # preserve same name
            (Index([3, 4, 5, 6, 7], name="other"), False),  # drop diff names
            (Index([3, 4, 5, 6, 7]), False),
        ],
    )
    def test_intersection_name_preservation(self, index2, keeps_name, sort):
        index1 = Index([1, 2, 3, 4, 5], name="index")
        expected = Index([3, 4, 5])
        result = index1.intersection(index2, sort)

        if keeps_name:
            expected.name = "index"

        assert result.name == expected.name
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize(
        "first_name,second_name,expected_name",
        [("A", "A", "A"), ("A", "B", None), (None, "B", None)],
    )
    def test_intersection_name_preservation2(
        self, index, first_name, second_name, expected_name, sort
    ):
        first = index[5:20]
        second = index[:10]
        first.name = first_name
        second.name = second_name
        intersect = first.intersection(second, sort=sort)
        assert intersect.name == expected_name

    def test_chained_union(self, sort):
        # Chained unions handles names correctly
        i1 = Index([1, 2], name="i1")
        i2 = Index([5, 6], name="i2")
        i3 = Index([3, 4], name="i3")
        union = i1.union(i2.union(i3, sort=sort), sort=sort)
        expected = i1.union(i2, sort=sort).union(i3, sort=sort)
        tm.assert_index_equal(union, expected)

        j1 = Index([1, 2], name="j1")
        j2 = Index([], name="j2")
        j3 = Index([], name="j3")
        union = j1.union(j2.union(j3, sort=sort), sort=sort)
        expected = j1.union(j2, sort=sort).union(j3, sort=sort)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union(self, index, sort):
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        union = first.union(second, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(union.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union_from_iterables(self, index, klass, sort):
        # GH#10149
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        case = klass(second.values)
        result = first.union(case, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(result.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(result, everything)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union_identity(self, index, sort):
        first = index[5:20]

        union = first.union(first, sort=sort)
        # i.e. identity is not preserved when sort is True
        assert (union is first) is (not sort)

        # This should no longer be the same object, since [] is not consistent,
        # both objects will be recast to dtype('O')
        union = first.union(Index([], dtype=first.dtype), sort=sort)
        assert (union is first) is (not sort)

        union = Index([], dtype=first.dtype).union(first, sort=sort)
        assert (union is first) is (not sort)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("second_name,expected", [(None, None), ("name", "name")])
    def test_difference_name_preservation(self, index, second_name, expected, sort):
        first = index[5:20]
        second = index[:10]
        answer = index[10:20]

        first.name = "name"
        second.name = second_name
        result = first.difference(second, sort=sort)

        if sort is True:
            tm.assert_index_equal(result, answer)
        else:
            answer.name = second_name
            tm.assert_index_equal(result.sort_values(), answer.sort_values())

        if expected is None:
            assert result.name is None
        else:
            assert result.name == expected

    def test_difference_empty_arg(self, index, sort):
        first = index.copy()
        first = first[5:20]
        first.name = "name"
        result = first.difference([], sort)
        expected = index[5:20].unique()
        expected.name = "name"
        tm.assert_index_equal(result, expected)

    def test_difference_should_not_compare(self):
        # GH 55113
        left = Index([1, 1])
        right = Index([True])
        result = left.difference(right)
        expected = Index([1])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_difference_identity(self, index, sort):
        first = index[5:20]
        first.name = "name"
        result = first.difference(first, sort)

        assert len(result) == 0
        assert result.name == first.name

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_difference_sort(self, index, sort):
        first = index[5:20]
        second = index[:10]

        result = first.difference(second, sort)
        expected = index[10:20]

        if sort is None:
            expected = expected.sort_values()

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    def test_difference_incomparable(self, opname):
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        op = operator.methodcaller(opname, b)

        with tm.assert_produces_warning(RuntimeWarning):
            # sort=None, the default
            result = op(a)
        expected = Index([3, Timestamp("2000"), 2, Timestamp("1999")])
        if opname == "difference":
            expected = expected[:2]
        tm.assert_index_equal(result, expected)

        # sort=False
        op = operator.methodcaller(opname, b, sort=False)
        result = op(a)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    def test_difference_incomparable_true(self, opname):
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        op = operator.methodcaller(opname, b, sort=True)

        msg = "'<' not supported between instances of 'Timestamp' and 'int'"
        with pytest.raises(TypeError, match=msg):
            op(a)

    def test_symmetric_difference_mi(self, sort):
        index1 = MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3]))
        index2 = MultiIndex.from_tuples([("foo", 1), ("bar", 3)])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = MultiIndex.from_tuples([("bar", 2), ("baz", 3), ("bar", 3)])
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index2,expected",
        [
            (Index([0, 1, np.nan]), Index([2.0, 3.0, 0.0])),
            (Index([0, 1]), Index([np.nan, 2.0, 3.0, 0.0])),
        ],
    )
    def test_symmetric_difference_missing(self, index2, expected, sort):
        # GH#13514 change: {nan} - {nan} == {}
        # (GH#6444, sorting of nans, is no longer an issue)
        index1 = Index([1, np.nan, 2, 3])

        result = index1.symmetric_difference(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference_non_index(self, sort):
        index1 = Index([1, 2, 3, 4], name="index1")
        index2 = np.array([2, 3, 4, 5])
        expected = Index([1, 5], name="index1")
        result = index1.symmetric_difference(index2, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)
        assert result.name == "index1"

        result = index1.symmetric_difference(index2, result_name="new_name", sort=sort)
        expected.name = "new_name"
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)
        assert result.name == "new_name"

    def test_union_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        # GH#51365
        idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        idx2 = Index([3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.union(idx2)
        expected = Index([1, 2, 3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        tm.assert_index_equal(result, expected)

    def test_union_string_array(self, any_string_dtype):
        idx1 = Index(["a"], dtype=any_string_dtype)
        idx2 = Index(["b"], dtype=any_string_dtype)
        result = idx1.union(idx2)
        expected = Index(["a", "b"], dtype=any_string_dtype)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_iterables.py ===
from textwrap import dedent
from itertools import islice, product

from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices.dense import Matrix
from sympy.combinatorics import RGS_enum, RGS_unrank, Permutation
from sympy.utilities.iterables import (
    _partition, _set_partitions, binary_partitions, bracelets, capture,
    cartes, common_prefix, common_suffix, connected_components, dict_merge,
    filter_symbols, flatten, generate_bell, generate_derangements,
    generate_involutions, generate_oriented_forest, group, has_dups, ibin,
    iproduct, kbins, minlex, multiset, multiset_combinations,
    multiset_partitions, multiset_permutations, necklaces, numbered_symbols,
    partitions, permutations, postfixes,
    prefixes, reshape, rotate_left, rotate_right, runs, sift,
    strongly_connected_components, subsets, take, topological_sort, unflatten,
    uniq, variations, ordered_partitions, rotations, is_palindromic, iterable,
    NotIterable, multiset_derangements, signed_permutations,
    sequence_partitions, sequence_partitions_empty)
from sympy.utilities.enumerative import (
    factoring_visitor, multiset_partitions_taocp )

from sympy.core.singleton import S
from sympy.testing.pytest import raises, warns_deprecated_sympy

w, x, y, z = symbols('w,x,y,z')


def test_deprecated_iterables():
    from sympy.utilities.iterables import default_sort_key, ordered
    with warns_deprecated_sympy():
        assert list(ordered([y, x])) == [x, y]
    with warns_deprecated_sympy():
        assert sorted([y, x], key=default_sort_key) == [x, y]


def test_is_palindromic():
    assert is_palindromic('')
    assert is_palindromic('x')
    assert is_palindromic('xx')
    assert is_palindromic('xyx')
    assert not is_palindromic('xy')
    assert not is_palindromic('xyzx')
    assert is_palindromic('xxyzzyx', 1)
    assert not is_palindromic('xxyzzyx', 2)
    assert is_palindromic('xxyzzyx', 2, -1)
    assert is_palindromic('xxyzzyx', 2, 6)
    assert is_palindromic('xxyzyx', 1)
    assert not is_palindromic('xxyzyx', 2)
    assert is_palindromic('xxyzyx', 2, 2 + 3)


def test_flatten():
    assert flatten((1, (1,))) == [1, 1]
    assert flatten((x, (x,))) == [x, x]

    ls = [[(-2, -1), (1, 2)], [(0, 0)]]

    assert flatten(ls, levels=0) == ls
    assert flatten(ls, levels=1) == [(-2, -1), (1, 2), (0, 0)]
    assert flatten(ls, levels=2) == [-2, -1, 1, 2, 0, 0]
    assert flatten(ls, levels=3) == [-2, -1, 1, 2, 0, 0]

    raises(ValueError, lambda: flatten(ls, levels=-1))

    class MyOp(Basic):
        pass

    assert flatten([MyOp(x, y), z]) == [MyOp(x, y), z]
    assert flatten([MyOp(x, y), z], cls=MyOp) == [x, y, z]

    assert flatten({1, 11, 2}) == list({1, 11, 2})


def test_iproduct():
    assert list(iproduct()) == [()]
    assert list(iproduct([])) == []
    assert list(iproduct([1,2,3])) == [(1,),(2,),(3,)]
    assert sorted(iproduct([1, 2], [3, 4, 5])) == [
        (1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
    assert sorted(iproduct([0,1],[0,1],[0,1])) == [
        (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    assert iterable(iproduct(S.Integers)) is True
    assert iterable(iproduct(S.Integers, S.Integers)) is True
    assert (3,) in iproduct(S.Integers)
    assert (4, 5) in iproduct(S.Integers, S.Integers)
    assert (1, 2, 3) in iproduct(S.Integers, S.Integers, S.Integers)
    triples  = set(islice(iproduct(S.Integers, S.Integers, S.Integers), 1000))
    for n1, n2, n3 in triples:
        assert isinstance(n1, Integer)
        assert isinstance(n2, Integer)
        assert isinstance(n3, Integer)
    for t in set(product(*([range(-2, 3)]*3))):
        assert t in iproduct(S.Integers, S.Integers, S.Integers)


def test_group():
    assert group([]) == []
    assert group([], multiple=False) == []

    assert group([1]) == [[1]]
    assert group([1], multiple=False) == [(1, 1)]

    assert group([1, 1]) == [[1, 1]]
    assert group([1, 1], multiple=False) == [(1, 2)]

    assert group([1, 1, 1]) == [[1, 1, 1]]
    assert group([1, 1, 1], multiple=False) == [(1, 3)]

    assert group([1, 2, 1]) == [[1], [2], [1]]
    assert group([1, 2, 1], multiple=False) == [(1, 1), (2, 1), (1, 1)]

    assert group([1, 1, 2, 2, 2, 1, 3, 3]) == [[1, 1], [2, 2, 2], [1], [3, 3]]
    assert group([1, 1, 2, 2, 2, 1, 3, 3], multiple=False) == [(1, 2),
                 (2, 3), (1, 1), (3, 2)]


def test_subsets():
    # combinations
    assert list(subsets([1, 2, 3], 0)) == [()]
    assert list(subsets([1, 2, 3], 1)) == [(1,), (2,), (3,)]
    assert list(subsets([1, 2, 3], 2)) == [(1, 2), (1, 3), (2, 3)]
    assert list(subsets([1, 2, 3], 3)) == [(1, 2, 3)]
    l = list(range(4))
    assert list(subsets(l, 0, repetition=True)) == [()]
    assert list(subsets(l, 1, repetition=True)) == [(0,), (1,), (2,), (3,)]
    assert list(subsets(l, 2, repetition=True)) == [(0, 0), (0, 1), (0, 2),
                                                    (0, 3), (1, 1), (1, 2),
                                                    (1, 3), (2, 2), (2, 3),
                                                    (3, 3)]
    assert list(subsets(l, 3, repetition=True)) == [(0, 0, 0), (0, 0, 1),
                                                    (0, 0, 2), (0, 0, 3),
                                                    (0, 1, 1), (0, 1, 2),
                                                    (0, 1, 3), (0, 2, 2),
                                                    (0, 2, 3), (0, 3, 3),
                                                    (1, 1, 1), (1, 1, 2),
                                                    (1, 1, 3), (1, 2, 2),
                                                    (1, 2, 3), (1, 3, 3),
                                                    (2, 2, 2), (2, 2, 3),
                                                    (2, 3, 3), (3, 3, 3)]
    assert len(list(subsets(l, 4, repetition=True))) == 35

    assert list(subsets(l[:2], 3, repetition=False)) == []
    assert list(subsets(l[:2], 3, repetition=True)) == [(0, 0, 0),
                                                        (0, 0, 1),
                                                        (0, 1, 1),
                                                        (1, 1, 1)]
    assert list(subsets([1, 2], repetition=True)) == \
        [(), (1,), (2,), (1, 1), (1, 2), (2, 2)]
    assert list(subsets([1, 2], repetition=False)) == \
        [(), (1,), (2,), (1, 2)]
    assert list(subsets([1, 2, 3], 2)) == \
        [(1, 2), (1, 3), (2, 3)]
    assert list(subsets([1, 2, 3], 2, repetition=True)) == \
        [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


def test_variations():
    # permutations
    l = list(range(4))
    assert list(variations(l, 0, repetition=False)) == [()]
    assert list(variations(l, 1, repetition=False)) == [(0,), (1,), (2,), (3,)]
    assert list(variations(l, 2, repetition=False)) == [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]
    assert list(variations(l, 3, repetition=False)) == [(0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2), (1, 0, 2), (1, 0, 3), (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 0, 1), (2, 0, 3), (2, 1, 0), (2, 1, 3), (2, 3, 0), (2, 3, 1), (3, 0, 1), (3, 0, 2), (3, 1, 0), (3, 1, 2), (3, 2, 0), (3, 2, 1)]
    assert list(variations(l, 0, repetition=True)) == [()]
    assert list(variations(l, 1, repetition=True)) == [(0,), (1,), (2,), (3,)]
    assert list(variations(l, 2, repetition=True)) == [(0, 0), (0, 1), (0, 2),
                                                       (0, 3), (1, 0), (1, 1),
                                                       (1, 2), (1, 3), (2, 0),
                                                       (2, 1), (2, 2), (2, 3),
                                                       (3, 0), (3, 1), (3, 2),
                                                       (3, 3)]
    assert len(list(variations(l, 3, repetition=True))) == 64
    assert len(list(variations(l, 4, repetition=True))) == 256
    assert list(variations(l[:2], 3, repetition=False)) == []
    assert list(variations(l[:2], 3, repetition=True)) == [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    ]


def test_cartes():
    assert list(cartes([1, 2], [3, 4, 5])) == \
        [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    assert list(cartes()) == [()]
    assert list(cartes('a')) == [('a',)]
    assert list(cartes('a', repeat=2)) == [('a', 'a')]
    assert list(cartes(list(range(2)))) == [(0,), (1,)]


def test_filter_symbols():
    s = numbered_symbols()
    filtered = filter_symbols(s, symbols("x0 x2 x3"))
    assert take(filtered, 3) == list(symbols("x1 x4 x5"))


def test_numbered_symbols():
    s = numbered_symbols(cls=Dummy)
    assert isinstance(next(s), Dummy)
    assert next(numbered_symbols('C', start=1, exclude=[symbols('C1')])) == \
        symbols('C2')


def test_sift():
    assert sift(list(range(5)), lambda _: _ % 2) == {1: [1, 3], 0: [0, 2, 4]}
    assert sift([x, y], lambda _: _.has(x)) == {False: [y], True: [x]}
    assert sift([S.One], lambda _: _.has(x)) == {False: [1]}
    assert sift([0, 1, 2, 3], lambda x: x % 2, binary=True) == (
        [1, 3], [0, 2])
    assert sift([0, 1, 2, 3], lambda x: x % 3 == 1, binary=True) == (
        [1], [0, 2, 3])
    raises(ValueError, lambda:
        sift([0, 1, 2, 3], lambda x: x % 3, binary=True))


def test_take():
    X = numbered_symbols()

    assert take(X, 5) == list(symbols('x0:5'))
    assert take(X, 5) == list(symbols('x5:10'))

    assert take([1, 2, 3, 4, 5], 5) == [1, 2, 3, 4, 5]


def test_dict_merge():
    assert dict_merge({}, {1: x, y: z}) == {1: x, y: z}
    assert dict_merge({1: x, y: z}, {}) == {1: x, y: z}

    assert dict_merge({2: z}, {1: x, y: z}) == {1: x, 2: z, y: z}
    assert dict_merge({1: x, y: z}, {2: z}) == {1: x, 2: z, y: z}

    assert dict_merge({1: y, 2: z}, {1: x, y: z}) == {1: x, 2: z, y: z}
    assert dict_merge({1: x, y: z}, {1: y, 2: z}) == {1: y, 2: z, y: z}


def test_prefixes():
    assert list(prefixes([])) == []
    assert list(prefixes([1])) == [[1]]
    assert list(prefixes([1, 2])) == [[1], [1, 2]]

    assert list(prefixes([1, 2, 3, 4, 5])) == \
        [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]


def test_postfixes():
    assert list(postfixes([])) == []
    assert list(postfixes([1])) == [[1]]
    assert list(postfixes([1, 2])) == [[2], [1, 2]]

    assert list(postfixes([1, 2, 3, 4, 5])) == \
        [[5], [4, 5], [3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5]]


def test_topological_sort():
    V = [2, 3, 5, 7, 8, 9, 10, 11]
    E = [(7, 11), (7, 8), (5, 11),
         (3, 8), (3, 10), (11, 2),
         (11, 9), (11, 10), (8, 9)]

    assert topological_sort((V, E)) == [3, 5, 7, 8, 11, 2, 9, 10]
    assert topological_sort((V, E), key=lambda v: -v) == \
        [7, 5, 11, 3, 10, 8, 9, 2]

    raises(ValueError, lambda: topological_sort((V, E + [(10, 7)])))


def test_strongly_connected_components():
    assert strongly_connected_components(([], [])) == []
    assert strongly_connected_components(([1, 2, 3], [])) == [[1], [2], [3]]

    V = [1, 2, 3]
    E = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1)]
    assert strongly_connected_components((V, E)) == [[1, 2, 3]]

    V = [1, 2, 3, 4]
    E = [(1, 2), (2, 3), (3, 2), (3, 4)]
    assert strongly_connected_components((V, E)) == [[4], [2, 3], [1]]

    V = [1, 2, 3, 4]
    E = [(1, 2), (2, 1), (3, 4), (4, 3)]
    assert strongly_connected_components((V, E)) == [[1, 2], [3, 4]]


def test_connected_components():
    assert connected_components(([], [])) == []
    assert connected_components(([1, 2, 3], [])) == [[1], [2], [3]]

    V = [1, 2, 3]
    E = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1)]
    assert connected_components((V, E)) == [[1, 2, 3]]

    V = [1, 2, 3, 4]
    E = [(1, 2), (2, 3), (3, 2), (3, 4)]
    assert connected_components((V, E)) == [[1, 2, 3, 4]]

    V = [1, 2, 3, 4]
    E = [(1, 2), (3, 4)]
    assert connected_components((V, E)) == [[1, 2], [3, 4]]


def test_rotate():
    A = [0, 1, 2, 3, 4]

    assert rotate_left(A, 2) == [2, 3, 4, 0, 1]
    assert rotate_right(A, 1) == [4, 0, 1, 2, 3]
    A = []
    B = rotate_right(A, 1)
    assert B == []
    B.append(1)
    assert A == []
    B = rotate_left(A, 1)
    assert B == []
    B.append(1)
    assert A == []


def test_multiset_partitions():
    A = [0, 1, 2, 3, 4]

    assert list(multiset_partitions(A, 5)) == [[[0], [1], [2], [3], [4]]]
    assert len(list(multiset_partitions(A, 4))) == 10
    assert len(list(multiset_partitions(A, 3))) == 25

    assert list(multiset_partitions([1, 1, 1, 2, 2], 2)) == [
        [[1, 1, 1, 2], [2]], [[1, 1, 1], [2, 2]], [[1, 1, 2, 2], [1]],
        [[1, 1, 2], [1, 2]], [[1, 1], [1, 2, 2]]]

    assert list(multiset_partitions([1, 1, 2, 2], 2)) == [
        [[1, 1, 2], [2]], [[1, 1], [2, 2]], [[1, 2, 2], [1]],
        [[1, 2], [1, 2]]]

    assert list(multiset_partitions([1, 2, 3, 4], 2)) == [
        [[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],
        [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],
        [[1], [2, 3, 4]]]

    assert list(multiset_partitions([1, 2, 2], 2)) == [
        [[1, 2], [2]], [[1], [2, 2]]]

    assert list(multiset_partitions(3)) == [
        [[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]],
        [[0], [1], [2]]]
    assert list(multiset_partitions(3, 2)) == [
        [[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]]]
    assert list(multiset_partitions([1] * 3, 2)) == [[[1], [1, 1]]]
    assert list(multiset_partitions([1] * 3)) == [
        [[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]
    a = [3, 2, 1]
    assert list(multiset_partitions(a)) == \
        list(multiset_partitions(sorted(a)))
    assert list(multiset_partitions(a, 5)) == []
    assert list(multiset_partitions(a, 1)) == [[[1, 2, 3]]]
    assert list(multiset_partitions(a + [4], 5)) == []
    assert list(multiset_partitions(a + [4], 1)) == [[[1, 2, 3, 4]]]
    assert list(multiset_partitions(2, 5)) == []
    assert list(multiset_partitions(2, 1)) == [[[0, 1]]]
    assert list(multiset_partitions('a')) == [[['a']]]
    assert list(multiset_partitions('a', 2)) == []
    assert list(multiset_partitions('ab')) == [[['a', 'b']], [['a'], ['b']]]
    assert list(multiset_partitions('ab', 1)) == [[['a', 'b']]]
    assert list(multiset_partitions('aaa', 1)) == [['aaa']]
    assert list(multiset_partitions([1, 1], 1)) == [[[1, 1]]]
    ans = [('mpsyy',), ('mpsy', 'y'), ('mps', 'yy'), ('mps', 'y', 'y'),
           ('mpyy', 's'), ('mpy', 'sy'), ('mpy', 's', 'y'), ('mp', 'syy'),
           ('mp', 'sy', 'y'), ('mp', 's', 'yy'), ('mp', 's', 'y', 'y'),
           ('msyy', 'p'), ('msy', 'py'), ('msy', 'p', 'y'), ('ms', 'pyy'),
           ('ms', 'py', 'y'), ('ms', 'p', 'yy'), ('ms', 'p', 'y', 'y'),
           ('myy', 'ps'), ('myy', 'p', 's'), ('my', 'psy'), ('my', 'ps', 'y'),
           ('my', 'py', 's'), ('my', 'p', 'sy'), ('my', 'p', 's', 'y'),
           ('m', 'psyy'), ('m', 'psy', 'y'), ('m', 'ps', 'yy'),
           ('m', 'ps', 'y', 'y'), ('m', 'pyy', 's'), ('m', 'py', 'sy'),
           ('m', 'py', 's', 'y'), ('m', 'p', 'syy'),
           ('m', 'p', 'sy', 'y'), ('m', 'p', 's', 'yy'),
           ('m', 'p', 's', 'y', 'y')]
    assert [tuple("".join(part) for part in p)
                for p in multiset_partitions('sympy')] == ans
    factorings = [[24], [8, 3], [12, 2], [4, 6], [4, 2, 3],
                  [6, 2, 2], [2, 2, 2, 3]]
    assert [factoring_visitor(p, [2,3]) for
                p in multiset_partitions_taocp([3, 1])] == factorings


def test_multiset_combinations():
    ans = ['iii', 'iim', 'iip', 'iis', 'imp', 'ims', 'ipp', 'ips',
           'iss', 'mpp', 'mps', 'mss', 'pps', 'pss', 'sss']
    assert [''.join(i) for i in
            list(multiset_combinations('mississippi', 3))] == ans
    M = multiset('mississippi')
    assert [''.join(i) for i in
            list(multiset_combinations(M, 3))] == ans
    assert [''.join(i) for i in multiset_combinations(M, 30)] == []
    assert list(multiset_combinations([[1], [2, 3]], 2)) == [[[1], [2, 3]]]
    assert len(list(multiset_combinations('a', 3))) == 0
    assert len(list(multiset_combinations('a', 0))) == 1
    assert list(multiset_combinations('abc', 1)) == [['a'], ['b'], ['c']]
    raises(ValueError, lambda: list(multiset_combinations({0: 3, 1: -1}, 2)))


def test_multiset_permutations():
    ans = ['abby', 'abyb', 'aybb', 'baby', 'bayb', 'bbay', 'bbya', 'byab',
           'byba', 'yabb', 'ybab', 'ybba']
    assert [''.join(i) for i in multiset_permutations('baby')] == ans
    assert [''.join(i) for i in multiset_permutations(multiset('baby'))] == ans
    assert list(multiset_permutations([0, 0, 0], 2)) == [[0, 0]]
    assert list(multiset_permutations([0, 2, 1], 2)) == [
        [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    assert len(list(multiset_permutations('a', 0))) == 1
    assert len(list(multiset_permutations('a', 3))) == 0
    for nul in ([], {}, ''):
        assert list(multiset_permutations(nul)) == [[]]
    assert list(multiset_permutations(nul, 0)) == [[]]
    # impossible requests give no result
    assert list(multiset_permutations(nul, 1)) == []
    assert list(multiset_permutations(nul, -1)) == []

    def test():
        for i in range(1, 7):
            print(i)
            for p in multiset_permutations([0, 0, 1, 0, 1], i):
                print(p)
    assert capture(lambda: test()) == dedent('''\
        1
        [0]
        [1]
        2
        [0, 0]
        [0, 1]
        [1, 0]
        [1, 1]
        3
        [0, 0, 0]
        [0, 0, 1]
        [0, 1, 0]
        [0, 1, 1]
        [1, 0, 0]
        [1, 0, 1]
        [1, 1, 0]
        4
        [0, 0, 0, 1]
        [0, 0, 1, 0]
        [0, 0, 1, 1]
        [0, 1, 0, 0]
        [0, 1, 0, 1]
        [0, 1, 1, 0]
        [1, 0, 0, 0]
        [1, 0, 0, 1]
        [1, 0, 1, 0]
        [1, 1, 0, 0]
        5
        [0, 0, 0, 1, 1]
        [0, 0, 1, 0, 1]
        [0, 0, 1, 1, 0]
        [0, 1, 0, 0, 1]
        [0, 1, 0, 1, 0]
        [0, 1, 1, 0, 0]
        [1, 0, 0, 0, 1]
        [1, 0, 0, 1, 0]
        [1, 0, 1, 0, 0]
        [1, 1, 0, 0, 0]
        6\n''')
    raises(ValueError, lambda: list(multiset_permutations({0: 3, 1: -1})))


def test_partitions():
    ans = [[{}], [(0, {})]]
    for i in range(2):
        assert list(partitions(0, size=i)) == ans[i]
        assert list(partitions(1, 0, size=i)) == ans[i]
        assert list(partitions(6, 2, 2, size=i)) == ans[i]
        assert list(partitions(6, 2, None, size=i)) != ans[i]
        assert list(partitions(6, None, 2, size=i)) != ans[i]
        assert list(partitions(6, 2, 0, size=i)) == ans[i]

    assert list(partitions(6, k=2)) == [
        {2: 3}, {1: 2, 2: 2}, {1: 4, 2: 1}, {1: 6}]

    assert list(partitions(6, k=3)) == [
        {3: 2}, {1: 1, 2: 1, 3: 1}, {1: 3, 3: 1}, {2: 3}, {1: 2, 2: 2},
        {1: 4, 2: 1}, {1: 6}]

    assert list(partitions(8, k=4, m=3)) == [
        {4: 2}, {1: 1, 3: 1, 4: 1}, {2: 2, 4: 1}, {2: 1, 3: 2}] == [
        i for i in partitions(8, k=4, m=3) if all(k <= 4 for k in i)
        and sum(i.values()) <=3]

    assert list(partitions(S(3), m=2)) == [
        {3: 1}, {1: 1, 2: 1}]

    assert list(partitions(4, k=3)) == [
        {1: 1, 3: 1}, {2: 2}, {1: 2, 2: 1}, {1: 4}] == [
        i for i in partitions(4) if all(k <= 3 for k in i)]


    # Consistency check on output of _partitions and RGS_unrank.
    # This provides a sanity test on both routines.  Also verifies that
    # the total number of partitions is the same in each case.
    #    (from pkrathmann2)

    for n in range(2, 6):
        i  = 0
        for m, q  in _set_partitions(n):
            assert  q == RGS_unrank(i, n)
            i += 1
        assert i == RGS_enum(n)


def test_binary_partitions():
    assert [i[:] for i in binary_partitions(10)] == [[8, 2], [8, 1, 1],
        [4, 4, 2], [4, 4, 1, 1], [4, 2, 2, 2], [4, 2, 2, 1, 1],
        [4, 2, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1, 1], [2, 2, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    assert len([j[:] for j in binary_partitions(16)]) == 36


def test_bell_perm():
    assert [len(set(generate_bell(i))) for i in range(1, 7)] == [
        factorial(i) for i in range(1, 7)]
    assert list(generate_bell(3)) == [
        (0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    # generate_bell and trotterjohnson are advertised to return the same
    # permutations; this is not technically necessary so this test could
    # be removed
    for n in range(1, 5):
        p = Permutation(range(n))
        b = generate_bell(n)
        for bi in b:
            assert bi == tuple(p.array_form)
            p = p.next_trotterjohnson()
    raises(ValueError, lambda: list(generate_bell(0)))  # XXX is this consistent with other permutation algorithms?


def test_involutions():
    lengths = [1, 2, 4, 10, 26, 76]
    for n, N in enumerate(lengths):
        i = list(generate_involutions(n + 1))
        assert len(i) == N
        assert len({Permutation(j)**2 for j in i}) == 1


def test_derangements():
    assert len(list(generate_derangements(list(range(6))))) == 265
    assert ''.join(''.join(i) for i in generate_derangements('abcde')) == (
    'badecbaecdbcaedbcdeabceadbdaecbdeacbdecabeacdbedacbedcacabedcadebcaebd'
    'cdaebcdbeacdeabcdebaceabdcebadcedabcedbadabecdaebcdaecbdcaebdcbeadceab'
    'dcebadeabcdeacbdebacdebcaeabcdeadbceadcbecabdecbadecdabecdbaedabcedacb'
    'edbacedbca')
    assert list(generate_derangements([0, 1, 2, 3])) == [
        [1, 0, 3, 2], [1, 2, 3, 0], [1, 3, 0, 2], [2, 0, 3, 1],
        [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 2, 0, 1], [3, 2, 1, 0]]
    assert list(generate_derangements([0, 1, 2, 2])) == [
        [2, 2, 0, 1], [2, 2, 1, 0]]
    assert list(generate_derangements('ba')) == [list('ab')]
    # multiset_derangements
    D = multiset_derangements
    assert list(D('abb')) == []
    assert [''.join(i) for i in D('ab')] == ['ba']
    assert [''.join(i) for i in D('abc')] == ['bca', 'cab']
    assert [''.join(i) for i in D('aabb')] == ['bbaa']
    assert [''.join(i) for i in D('aabbcccc')] == [
        'ccccaabb', 'ccccabab', 'ccccabba', 'ccccbaab', 'ccccbaba',
        'ccccbbaa']
    assert [''.join(i) for i in D('aabbccc')] == [
        'cccabba', 'cccabab', 'cccaabb', 'ccacbba', 'ccacbab',
        'ccacabb', 'cbccbaa', 'cbccaba', 'cbccaab', 'bcccbaa',
        'bcccaba', 'bcccaab']
    assert [''.join(i) for i in D('books')] == ['kbsoo', 'ksboo',
        'sbkoo', 'skboo', 'oksbo', 'oskbo', 'okbso', 'obkso', 'oskob',
        'oksob', 'osbok', 'obsok']
    assert list(generate_derangements([[3], [2], [2], [1]])) == [
        [[2], [1], [3], [2]], [[2], [3], [1], [2]]]


def test_necklaces():
    def count(n, k, f):
        return len(list(necklaces(n, k, f)))
    m = []
    for i in range(1, 8):
        m.append((
        i, count(i, 2, 0), count(i, 2, 1), count(i, 3, 1)))
    assert Matrix(m) == Matrix([
        [1,   2,   2,   3],
        [2,   3,   3,   6],
        [3,   4,   4,  10],
        [4,   6,   6,  21],
        [5,   8,   8,  39],
        [6,  14,  13,  92],
        [7,  20,  18, 198]])


def test_bracelets():
    bc = list(bracelets(2, 4))
    assert Matrix(bc) == Matrix([
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 2],
        [2, 3],
        [3, 3]
        ])
    bc = list(bracelets(4, 2))
    assert Matrix(bc) == Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ])


def test_generate_oriented_forest():
    assert list(generate_oriented_forest(5)) == [[0, 1, 2, 3, 4],
        [0, 1, 2, 3, 3], [0, 1, 2, 3, 2], [0, 1, 2, 3, 1], [0, 1, 2, 3, 0],
        [0, 1, 2, 2, 2], [0, 1, 2, 2, 1], [0, 1, 2, 2, 0], [0, 1, 2, 1, 2],
        [0, 1, 2, 1, 1], [0, 1, 2, 1, 0], [0, 1, 2, 0, 1], [0, 1, 2, 0, 0],
        [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    assert len(list(generate_oriented_forest(10))) == 1842


def test_unflatten():
    r = list(range(10))
    assert unflatten(r) == list(zip(r[::2], r[1::2]))
    assert unflatten(r, 5) == [tuple(r[:5]), tuple(r[5:])]
    raises(ValueError, lambda: unflatten(list(range(10)), 3))
    raises(ValueError, lambda: unflatten(list(range(10)), -2))


def test_common_prefix_suffix():
    assert common_prefix([], [1]) == []
    assert common_prefix(list(range(3))) == [0, 1, 2]
    assert common_prefix(list(range(3)), list(range(4))) == [0, 1, 2]
    assert common_prefix([1, 2, 3], [1, 2, 5]) == [1, 2]
    assert common_prefix([1, 2, 3], [1, 3, 5]) == [1]

    assert common_suffix([], [1]) == []
    assert common_suffix(list(range(3))) == [0, 1, 2]
    assert common_suffix(list(range(3)), list(range(3))) == [0, 1, 2]
    assert common_suffix(list(range(3)), list(range(4))) == []
    assert common_suffix([1, 2, 3], [9, 2, 3]) == [2, 3]
    assert common_suffix([1, 2, 3], [9, 7, 3]) == [3]


def test_minlex():
    assert minlex([1, 2, 0]) == (0, 1, 2)
    assert minlex((1, 2, 0)) == (0, 1, 2)
    assert minlex((1, 0, 2)) == (0, 2, 1)
    assert minlex((1, 0, 2), directed=False) == (0, 1, 2)
    assert minlex('aba') == 'aab'
    assert minlex(('bb', 'aaa', 'c', 'a'), key=len) == ('c', 'a', 'bb', 'aaa')


def test_ordered():
    assert list(ordered((x, y), hash, default=False)) in [[x, y], [y, x]]
    assert list(ordered((x, y), hash, default=False)) == \
        list(ordered((y, x), hash, default=False))
    assert list(ordered((x, y))) == [x, y]

    seq, keys = [[[1, 2, 1], [0, 3, 1], [1, 1, 3], [2], [1]],
                 (lambda x: len(x), lambda x: sum(x))]
    assert list(ordered(seq, keys, default=False, warn=False)) == \
        [[1], [2], [1, 2, 1], [0, 3, 1], [1, 1, 3]]
    raises(ValueError, lambda:
           list(ordered(seq, keys, default=False, warn=True)))


def test_runs():
    assert runs([]) == []
    assert runs([1]) == [[1]]
    assert runs([1, 1]) == [[1], [1]]
    assert runs([1, 1, 2]) == [[1], [1, 2]]
    assert runs([1, 2, 1]) == [[1, 2], [1]]
    assert runs([2, 1, 1]) == [[2], [1], [1]]
    from operator import lt
    assert runs([2, 1, 1], lt) == [[2, 1], [1]]


def test_reshape():
    seq = list(range(1, 9))
    assert reshape(seq, [4]) == \
        [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert reshape(seq, (4,)) == \
        [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert reshape(seq, (2, 2)) == \
        [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert reshape(seq, (2, [2])) == \
        [(1, 2, [3, 4]), (5, 6, [7, 8])]
    assert reshape(seq, ((2,), [2])) == \
        [((1, 2), [3, 4]), ((5, 6), [7, 8])]
    assert reshape(seq, (1, [2], 1)) == \
        [(1, [2, 3], 4), (5, [6, 7], 8)]
    assert reshape(tuple(seq), ([[1], 1, (2,)],)) == \
        (([[1], 2, (3, 4)],), ([[5], 6, (7, 8)],))
    assert reshape(tuple(seq), ([1], 1, (2,))) == \
        (([1], 2, (3, 4)), ([5], 6, (7, 8)))
    assert reshape(list(range(12)), [2, [3], {2}, (1, (3,), 1)]) == \
        [[0, 1, [2, 3, 4], {5, 6}, (7, (8, 9, 10), 11)]]
    raises(ValueError, lambda: reshape([0, 1], [-1]))
    raises(ValueError, lambda: reshape([0, 1], [3]))


def test_uniq():
    assert list(uniq(p for p in partitions(4))) == \
        [{4: 1}, {1: 1, 3: 1}, {2: 2}, {1: 2, 2: 1}, {1: 4}]
    assert list(uniq(x % 2 for x in range(5))) == [0, 1]
    assert list(uniq('a')) == ['a']
    assert list(uniq('ababc')) == list('abc')
    assert list(uniq([[1], [2, 1], [1]])) == [[1], [2, 1]]
    assert list(uniq(permutations(i for i in [[1], 2, 2]))) == \
        [([1], 2, 2), (2, [1], 2), (2, 2, [1])]
    assert list(uniq([2, 3, 2, 4, [2], [1], [2], [3], [1]])) == \
        [2, 3, 4, [2], [1], [3]]
    f = [1]
    raises(RuntimeError, lambda: [f.remove(i) for i in uniq(f)])
    f = [[1]]
    raises(RuntimeError, lambda: [f.remove(i) for i in uniq(f)])


def test_kbins():
    assert len(list(kbins('1123', 2, ordered=1))) == 24
    assert len(list(kbins('1123', 2, ordered=11))) == 36
    assert len(list(kbins('1123', 2, ordered=10))) == 10
    assert len(list(kbins('1123', 2, ordered=0))) == 5
    assert len(list(kbins('1123', 2, ordered=None))) == 3

    def test1():
        for orderedval in [None, 0, 1, 10, 11]:
            print('ordered =', orderedval)
            for p in kbins([0, 0, 1], 2, ordered=orderedval):
                print('   ', p)
    assert capture(lambda : test1()) == dedent('''\
        ordered = None
            [[0], [0, 1]]
            [[0, 0], [1]]
        ordered = 0
            [[0, 0], [1]]
            [[0, 1], [0]]
        ordered = 1
            [[0], [0, 1]]
            [[0], [1, 0]]
            [[1], [0, 0]]
        ordered = 10
            [[0, 0], [1]]
            [[1], [0, 0]]
            [[0, 1], [0]]
            [[0], [0, 1]]
        ordered = 11
            [[0], [0, 1]]
            [[0, 0], [1]]
            [[0], [1, 0]]
            [[0, 1], [0]]
            [[1], [0, 0]]
            [[1, 0], [0]]\n''')

    def test2():
        for orderedval in [None, 0, 1, 10, 11]:
            print('ordered =', orderedval)
            for p in kbins(list(range(3)), 2, ordered=orderedval):
                print('   ', p)
    assert capture(lambda : test2()) == dedent('''\
        ordered = None
            [[0], [1, 2]]
            [[0, 1], [2]]
        ordered = 0
            [[0, 1], [2]]
            [[0, 2], [1]]
            [[0], [1, 2]]
        ordered = 1
            [[0], [1, 2]]
            [[0], [2, 1]]
            [[1], [0, 2]]
            [[1], [2, 0]]
            [[2], [0, 1]]
            [[2], [1, 0]]
        ordered = 10
            [[0, 1], [2]]
            [[2], [0, 1]]
            [[0, 2], [1]]
            [[1], [0, 2]]
            [[0], [1, 2]]
            [[1, 2], [0]]
        ordered = 11
            [[0], [1, 2]]
            [[0, 1], [2]]
            [[0], [2, 1]]
            [[0, 2], [1]]
            [[1], [0, 2]]
            [[1, 0], [2]]
            [[1], [2, 0]]
            [[1, 2], [0]]
            [[2], [0, 1]]
            [[2, 0], [1]]
            [[2], [1, 0]]
            [[2, 1], [0]]\n''')


def test_has_dups():
    assert has_dups(set()) is False
    assert has_dups(list(range(3))) is False
    assert has_dups([1, 2, 1]) is True
    assert has_dups([[1], [1]]) is True
    assert has_dups([[1], [2]]) is False


def test__partition():
    assert _partition('abcde', [1, 0, 1, 2, 0]) == [
        ['b', 'e'], ['a', 'c'], ['d']]
    assert _partition('abcde', [1, 0, 1, 2, 0], 3) == [
        ['b', 'e'], ['a', 'c'], ['d']]
    output = (3, [1, 0, 1, 2, 0])
    assert _partition('abcde', *output) == [['b', 'e'], ['a', 'c'], ['d']]


def test_ordered_partitions():
    from sympy.functions.combinatorial.numbers import nT
    f = ordered_partitions
    assert list(f(0, 1)) == [[]]
    assert list(f(1, 0)) == [[]]
    for i in range(1, 7):
        for j in [None] + list(range(1, i)):
            assert (
                sum(1 for p in f(i, j, 1)) ==
                sum(1 for p in f(i, j, 0)) ==
                nT(i, j))


def test_rotations():
    assert list(rotations('ab')) == [['a', 'b'], ['b', 'a']]
    assert list(rotations(range(3))) == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert list(rotations(range(3), dir=-1)) == [[0, 1, 2], [2, 0, 1], [1, 2, 0]]


def test_ibin():
    assert ibin(3) == [1, 1]
    assert ibin(3, 3) == [0, 1, 1]
    assert ibin(3, str=True) == '11'
    assert ibin(3, 3, str=True) == '011'
    assert list(ibin(2, 'all')) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert list(ibin(2, '', str=True)) == ['00', '01', '10', '11']
    raises(ValueError, lambda: ibin(-.5))
    raises(ValueError, lambda: ibin(2, 1))


def test_iterable():
    assert iterable(0) is False
    assert iterable(1) is False
    assert iterable(None) is False

    class Test1(NotIterable):
        pass

    assert iterable(Test1()) is False

    class Test2(NotIterable):
        _iterable = True

    assert iterable(Test2()) is True

    class Test3:
        pass

    assert iterable(Test3()) is False

    class Test4:
        _iterable = True

    assert iterable(Test4()) is True

    class Test5:
        def __iter__(self):
            yield 1

    assert iterable(Test5()) is True

    class Test6(Test5):
        _iterable = False

    assert iterable(Test6()) is False


def test_sequence_partitions():
    assert list(sequence_partitions([1], 1)) == [[[1]]]
    assert list(sequence_partitions([1, 2], 1)) == [[[1, 2]]]
    assert list(sequence_partitions([1, 2], 2)) == [[[1], [2]]]
    assert list(sequence_partitions([1, 2, 3], 1)) == [[[1, 2, 3]]]
    assert list(sequence_partitions([1, 2, 3], 2)) == \
        [[[1], [2, 3]], [[1, 2], [3]]]
    assert list(sequence_partitions([1, 2, 3], 3)) == [[[1], [2], [3]]]

    # Exceptional cases
    assert list(sequence_partitions([], 0)) == []
    assert list(sequence_partitions([], 1)) == []
    assert list(sequence_partitions([1, 2], 0)) == []
    assert list(sequence_partitions([1, 2], 3)) == []


def test_sequence_partitions_empty():
    assert list(sequence_partitions_empty([], 1)) == [[[]]]
    assert list(sequence_partitions_empty([], 2)) == [[[], []]]
    assert list(sequence_partitions_empty([], 3)) == [[[], [], []]]
    assert list(sequence_partitions_empty([1], 1)) == [[[1]]]
    assert list(sequence_partitions_empty([1], 2)) == [[[], [1]], [[1], []]]
    assert list(sequence_partitions_empty([1], 3)) == \
        [[[], [], [1]], [[], [1], []], [[1], [], []]]
    assert list(sequence_partitions_empty([1, 2], 1)) == [[[1, 2]]]
    assert list(sequence_partitions_empty([1, 2], 2)) == \
        [[[], [1, 2]], [[1], [2]], [[1, 2], []]]
    assert list(sequence_partitions_empty([1, 2], 3)) == [
        [[], [], [1, 2]], [[], [1], [2]], [[], [1, 2], []],
        [[1], [], [2]], [[1], [2], []], [[1, 2], [], []]
    ]
    assert list(sequence_partitions_empty([1, 2, 3], 1)) == [[[1, 2, 3]]]
    assert list(sequence_partitions_empty([1, 2, 3], 2)) == \
        [[[], [1, 2, 3]], [[1], [2, 3]], [[1, 2], [3]], [[1, 2, 3], []]]
    assert list(sequence_partitions_empty([1, 2, 3], 3)) == [
        [[], [], [1, 2, 3]], [[], [1], [2, 3]],
        [[], [1, 2], [3]], [[], [1, 2, 3], []],
        [[1], [], [2, 3]], [[1], [2], [3]],
        [[1], [2, 3], []], [[1, 2], [], [3]],
        [[1, 2], [3], []], [[1, 2, 3], [], []]
    ]

    # Exceptional cases
    assert list(sequence_partitions([], 0)) == []
    assert list(sequence_partitions([1], 0)) == []
    assert list(sequence_partitions([1, 2], 0)) == []


def test_signed_permutations():
    ans = [(0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0)]
    assert list(signed_permutations((0, 1, 1))) == ans
    assert list(signed_permutations((1, 0, 1))) == ans
    assert list(signed_permutations((1, 1, 0))) == ans

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\test_old_ma.py ===
import pickle
from functools import reduce

import pytest

import numpy as np
import numpy._core.fromnumeric as fromnumeric
import numpy._core.umath as umath
from numpy.ma import (
    MaskedArray,
    MaskType,
    absolute,
    add,
    all,
    allclose,
    allequal,
    alltrue,
    arange,
    arccos,
    arcsin,
    arctan,
    arctan2,
    array,
    average,
    choose,
    concatenate,
    conjugate,
    cos,
    cosh,
    count,
    divide,
    equal,
    exp,
    filled,
    getmask,
    greater,
    greater_equal,
    inner,
    isMaskedArray,
    less,
    less_equal,
    log,
    log10,
    make_mask,
    masked,
    masked_array,
    masked_equal,
    masked_greater,
    masked_greater_equal,
    masked_inside,
    masked_less,
    masked_less_equal,
    masked_not_equal,
    masked_outside,
    masked_print_option,
    masked_values,
    masked_where,
    maximum,
    minimum,
    multiply,
    nomask,
    nonzero,
    not_equal,
    ones,
    outer,
    product,
    put,
    ravel,
    repeat,
    resize,
    shape,
    sin,
    sinh,
    sometrue,
    sort,
    sqrt,
    subtract,
    sum,
    take,
    tan,
    tanh,
    transpose,
    where,
    zeros,
)
from numpy.testing import (
    assert_,
    assert_equal,
    assert_raises,
)

pi = np.pi


def eq(v, w, msg=''):
    result = allclose(v, w)
    if not result:
        print(f'Not eq:{msg}\n{v}\n----{w}')
    return result


class TestMa:

    def setup_method(self):
        x = np.array([1., 1., 1., -2., pi / 2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = array(x, mask=m1)
        ym = array(y, mask=m2)
        z = np.array([-.5, 0., .5, .8])
        zm = array(z, mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        s = x.shape
        xm.set_fill_value(1e+20)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf, s)

    def test_testBasic1d(self):
        # Test of basic array creation and properties in 1 dimension.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        assert_(not isMaskedArray(x))
        assert_(isMaskedArray(xm))
        assert_equal(shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.dtype, x.dtype)
        assert_equal(xm.size, reduce(lambda x, y: x * y, s))
        assert_equal(count(xm), len(m1) - reduce(lambda x, y: x + y, m1))
        assert_(eq(xm, xf))
        assert_(eq(filled(xm, 1.e20), xf))
        assert_(eq(x, xm))

    @pytest.mark.parametrize("s", [(4, 3), (6, 2)])
    def test_testBasic2d(self, s):
        # Test of basic array creation and properties in 2 dimensions.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        x.shape = s
        y.shape = s
        xm.shape = s
        ym.shape = s
        xf.shape = s

        assert_(not isMaskedArray(x))
        assert_(isMaskedArray(xm))
        assert_equal(shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.size, reduce(lambda x, y: x * y, s))
        assert_equal(count(xm), len(m1) - reduce(lambda x, y: x + y, m1))
        assert_(eq(xm, xf))
        assert_(eq(filled(xm, 1.e20), xf))
        assert_(eq(x, xm))

    def test_testArithmetic(self):
        # Test of basic arithmetic.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        a2d = array([[1, 2], [0, 4]])
        a2dm = masked_array(a2d, [[0, 0], [1, 0]])
        assert_(eq(a2d * a2d, a2d * a2dm))
        assert_(eq(a2d + a2d, a2d + a2dm))
        assert_(eq(a2d - a2d, a2d - a2dm))
        for s in [(12,), (4, 3), (2, 6)]:
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            assert_(eq(-x, -xm))
            assert_(eq(x + y, xm + ym))
            assert_(eq(x - y, xm - ym))
            assert_(eq(x * y, xm * ym))
            with np.errstate(divide='ignore', invalid='ignore'):
                assert_(eq(x / y, xm / ym))
            assert_(eq(a10 + y, a10 + ym))
            assert_(eq(a10 - y, a10 - ym))
            assert_(eq(a10 * y, a10 * ym))
            with np.errstate(divide='ignore', invalid='ignore'):
                assert_(eq(a10 / y, a10 / ym))
            assert_(eq(x + a10, xm + a10))
            assert_(eq(x - a10, xm - a10))
            assert_(eq(x * a10, xm * a10))
            assert_(eq(x / a10, xm / a10))
            assert_(eq(x ** 2, xm ** 2))
            assert_(eq(abs(x) ** 2.5, abs(xm) ** 2.5))
            assert_(eq(x ** y, xm ** ym))
            assert_(eq(np.add(x, y), add(xm, ym)))
            assert_(eq(np.subtract(x, y), subtract(xm, ym)))
            assert_(eq(np.multiply(x, y), multiply(xm, ym)))
            with np.errstate(divide='ignore', invalid='ignore'):
                assert_(eq(np.divide(x, y), divide(xm, ym)))

    def test_testMixedArithmetic(self):
        na = np.array([1])
        ma = array([1])
        assert_(isinstance(na + ma, MaskedArray))
        assert_(isinstance(ma + na, MaskedArray))

    def test_testUfuncs1(self):
        # Test various functions such as sin, cos.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        assert_(eq(np.cos(x), cos(xm)))
        assert_(eq(np.cosh(x), cosh(xm)))
        assert_(eq(np.sin(x), sin(xm)))
        assert_(eq(np.sinh(x), sinh(xm)))
        assert_(eq(np.tan(x), tan(xm)))
        assert_(eq(np.tanh(x), tanh(xm)))
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_(eq(np.sqrt(abs(x)), sqrt(xm)))
            assert_(eq(np.log(abs(x)), log(xm)))
            assert_(eq(np.log10(abs(x)), log10(xm)))
        assert_(eq(np.exp(x), exp(xm)))
        assert_(eq(np.arcsin(z), arcsin(zm)))
        assert_(eq(np.arccos(z), arccos(zm)))
        assert_(eq(np.arctan(z), arctan(zm)))
        assert_(eq(np.arctan2(x, y), arctan2(xm, ym)))
        assert_(eq(np.absolute(x), absolute(xm)))
        assert_(eq(np.equal(x, y), equal(xm, ym)))
        assert_(eq(np.not_equal(x, y), not_equal(xm, ym)))
        assert_(eq(np.less(x, y), less(xm, ym)))
        assert_(eq(np.greater(x, y), greater(xm, ym)))
        assert_(eq(np.less_equal(x, y), less_equal(xm, ym)))
        assert_(eq(np.greater_equal(x, y), greater_equal(xm, ym)))
        assert_(eq(np.conjugate(x), conjugate(xm)))
        assert_(eq(np.concatenate((x, y)), concatenate((xm, ym))))
        assert_(eq(np.concatenate((x, y)), concatenate((x, y))))
        assert_(eq(np.concatenate((x, y)), concatenate((xm, y))))
        assert_(eq(np.concatenate((x, y, x)), concatenate((x, ym, x))))

    def test_xtestCount(self):
        # Test count
        ott = array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        assert_(count(ott).dtype.type is np.intp)
        assert_equal(3, count(ott))
        assert_equal(1, count(1))
        assert_(eq(0, array(1, mask=[1])))
        ott = ott.reshape((2, 2))
        assert_(count(ott).dtype.type is np.intp)
        assert_(isinstance(count(ott, 0), np.ndarray))
        assert_(count(ott).dtype.type is np.intp)
        assert_(eq(3, count(ott)))
        assert_(getmask(count(ott, 0)) is nomask)
        assert_(eq([1, 2], count(ott, 0)))

    def test_testMinMax(self):
        # Test minimum and maximum.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        xr = np.ravel(x)  # max doesn't work if shaped
        xmr = ravel(xm)

        # true because of careful selection of data
        assert_(eq(max(xr), maximum.reduce(xmr)))
        assert_(eq(min(xr), minimum.reduce(xmr)))

    def test_testAddSumProd(self):
        # Test add, sum, product.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        assert_(eq(np.add.reduce(x), add.reduce(x)))
        assert_(eq(np.add.accumulate(x), add.accumulate(x)))
        assert_(eq(4, sum(array(4), axis=0)))
        assert_(eq(4, sum(array(4), axis=0)))
        assert_(eq(np.sum(x, axis=0), sum(x, axis=0)))
        assert_(eq(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0)))
        assert_(eq(np.sum(x, 0), sum(x, 0)))
        assert_(eq(np.prod(x, axis=0), product(x, axis=0)))
        assert_(eq(np.prod(x, 0), product(x, 0)))
        assert_(eq(np.prod(filled(xm, 1), axis=0),
                           product(xm, axis=0)))
        if len(s) > 1:
            assert_(eq(np.concatenate((x, y), 1),
                               concatenate((xm, ym), 1)))
            assert_(eq(np.add.reduce(x, 1), add.reduce(x, 1)))
            assert_(eq(np.sum(x, 1), sum(x, 1)))
            assert_(eq(np.prod(x, 1), product(x, 1)))

    def test_testCI(self):
        # Test of conversions and indexing
        x1 = np.array([1, 2, 4, 3])
        x2 = array(x1, mask=[1, 0, 0, 0])
        x3 = array(x1, mask=[0, 1, 0, 1])
        x4 = array(x1)
        # test conversion to strings
        str(x2)  # raises?
        repr(x2)  # raises?
        assert_(eq(np.sort(x1), sort(x2, fill_value=0)))
        # tests of indexing
        assert_(type(x2[1]) is type(x1[1]))
        assert_(x1[1] == x2[1])
        assert_(x2[0] is masked)
        assert_(eq(x1[2], x2[2]))
        assert_(eq(x1[2:5], x2[2:5]))
        assert_(eq(x1[:], x2[:]))
        assert_(eq(x1[1:], x3[1:]))
        x1[2] = 9
        x2[2] = 9
        assert_(eq(x1, x2))
        x1[1:3] = 99
        x2[1:3] = 99
        assert_(eq(x1, x2))
        x2[1] = masked
        assert_(eq(x1, x2))
        x2[1:3] = masked
        assert_(eq(x1, x2))
        x2[:] = x1
        x2[1] = masked
        assert_(allequal(getmask(x2), array([0, 1, 0, 0])))
        x3[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x3), array([0, 1, 1, 0])))
        x4[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x4), array([0, 1, 1, 0])))
        assert_(allequal(x4, array([1, 2, 3, 4])))
        x1 = np.arange(5) * 1.0
        x2 = masked_values(x1, 3.0)
        assert_(eq(x1, x2))
        assert_(allequal(array([0, 0, 0, 1, 0], MaskType), x2.mask))
        assert_(eq(3.0, x2.fill_value))
        x1 = array([1, 'hello', 2, 3], object)
        x2 = np.array([1, 'hello', 2, 3], object)
        s1 = x1[1]
        s2 = x2[1]
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        assert_equal(s1, s2)
        assert_(x1[1:1].shape == (0,))

    def test_testCopySize(self):
        # Tests of some subtle points of copying and sizing.
        n = [0, 0, 1, 0, 0]
        m = make_mask(n)
        m2 = make_mask(m)
        assert_(m is m2)
        m3 = make_mask(m, copy=True)
        assert_(m is not m3)

        x1 = np.arange(5)
        y1 = array(x1, mask=m)
        assert_(y1._data is not x1)
        assert_(allequal(x1, y1._data))
        assert_(y1._mask is m)

        y1a = array(y1, copy=0)
        # For copy=False, one might expect that the array would just
        # passed on, i.e., that it would be "is" instead of "==".
        # See gh-4043 for discussion.
        assert_(y1a._mask.__array_interface__ ==
                y1._mask.__array_interface__)

        y2 = array(x1, mask=m3, copy=0)
        assert_(y2._mask is m3)
        assert_(y2[2] is masked)
        y2[2] = 9
        assert_(y2[2] is not masked)
        assert_(y2._mask is m3)
        assert_(allequal(y2.mask, 0))

        y2a = array(x1, mask=m, copy=1)
        assert_(y2a._mask is not m)
        assert_(y2a[2] is masked)
        y2a[2] = 9
        assert_(y2a[2] is not masked)
        assert_(y2a._mask is not m)
        assert_(allequal(y2a.mask, 0))

        y3 = array(x1 * 1.0, mask=m)
        assert_(filled(y3).dtype is (x1 * 1.0).dtype)

        x4 = arange(4)
        x4[2] = masked
        y4 = resize(x4, (8,))
        assert_(eq(concatenate([x4, x4]), y4))
        assert_(eq(getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0]))
        y5 = repeat(x4, (2, 2, 2, 2), axis=0)
        assert_(eq(y5, [0, 0, 1, 1, 2, 2, 3, 3]))
        y6 = repeat(x4, 2, axis=0)
        assert_(eq(y5, y6))

    def test_testPut(self):
        # Test of put
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        m2 = m.copy()
        x = array(d, mask=m)
        assert_(x[3] is masked)
        assert_(x[4] is masked)
        x[[1, 4]] = [10, 40]
        assert_(x._mask is m)
        assert_(x[3] is masked)
        assert_(x[4] is not masked)
        assert_(eq(x, [0, 10, 2, -1, 40]))

        x = array(d, mask=m2, copy=True)
        x.put([0, 1, 2], [-1, 100, 200])
        assert_(x._mask is not m2)
        assert_(x[3] is masked)
        assert_(x[4] is masked)
        assert_(eq(x, [-1, 100, 200, 0, 0]))

    def test_testPut2(self):
        # Test of put
        d = arange(5)
        x = array(d, mask=[0, 0, 0, 0, 0])
        z = array([10, 40], mask=[1, 0])
        assert_(x[2] is not masked)
        assert_(x[3] is not masked)
        x[2:4] = z
        assert_(x[2] is masked)
        assert_(x[3] is not masked)
        assert_(eq(x, [0, 1, 10, 40, 4]))

        d = arange(5)
        x = array(d, mask=[0, 0, 0, 0, 0])
        y = x[2:4]
        z = array([10, 40], mask=[1, 0])
        assert_(x[2] is not masked)
        assert_(x[3] is not masked)
        y[:] = z
        assert_(y[0] is masked)
        assert_(y[1] is not masked)
        assert_(eq(y, [10, 40]))
        assert_(x[2] is masked)
        assert_(x[3] is not masked)
        assert_(eq(x, [0, 1, 10, 40, 4]))

    def test_testMaPut(self):
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
        i = np.nonzero(m)[0]
        put(ym, i, zm)
        assert_(all(take(ym, i, axis=0) == zm))

    def test_testOddFeatures(self):
        # Test of other odd features
        x = arange(20)
        x = x.reshape(4, 5)
        x.flat[5] = 12
        assert_(x[1, 0] == 12)
        z = x + 10j * x
        assert_(eq(z.real, x))
        assert_(eq(z.imag, 10 * x))
        assert_(eq((z * conjugate(z)).real, 101 * x * x))
        z.imag[...] = 0.0

        x = arange(10)
        x[3] = masked
        assert_(str(x[3]) == str(masked))
        c = x >= 8
        assert_(count(where(c, masked, masked)) == 0)
        assert_(shape(where(c, masked, masked)) == c.shape)
        z = where(c, x, masked)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is masked)
        assert_(z[7] is masked)
        assert_(z[8] is not masked)
        assert_(z[9] is not masked)
        assert_(eq(x, z))
        z = where(c, masked, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is not masked)
        assert_(z[7] is not masked)
        assert_(z[8] is masked)
        assert_(z[9] is masked)
        z = masked_where(c, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is not masked)
        assert_(z[7] is not masked)
        assert_(z[8] is masked)
        assert_(z[9] is masked)
        assert_(eq(x, z))
        x = array([1., 2., 3., 4., 5.])
        c = array([1, 1, 1, 0, 0])
        x[2] = masked
        z = where(c, x, -x)
        assert_(eq(z, [1., 2., 0., -4., -5]))
        c[0] = masked
        z = where(c, x, -x)
        assert_(eq(z, [1., 2., 0., -4., -5]))
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)
        assert_(eq(masked_where(greater(x, 2), x), masked_greater(x, 2)))
        assert_(eq(masked_where(greater_equal(x, 2), x),
                   masked_greater_equal(x, 2)))
        assert_(eq(masked_where(less(x, 2), x), masked_less(x, 2)))
        assert_(eq(masked_where(less_equal(x, 2), x), masked_less_equal(x, 2)))
        assert_(eq(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2)))
        assert_(eq(masked_where(equal(x, 2), x), masked_equal(x, 2)))
        assert_(eq(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2)))
        assert_(eq(masked_inside(list(range(5)), 1, 3), [0, 199, 199, 199, 4]))
        assert_(eq(masked_outside(list(range(5)), 1, 3), [199, 1, 2, 3, 199]))
        assert_(eq(masked_inside(array(list(range(5)),
                                       mask=[1, 0, 0, 0, 0]), 1, 3).mask,
                   [1, 1, 1, 1, 0]))
        assert_(eq(masked_outside(array(list(range(5)),
                                        mask=[0, 1, 0, 0, 0]), 1, 3).mask,
                   [1, 1, 0, 0, 1]))
        assert_(eq(masked_equal(array(list(range(5)),
                                      mask=[1, 0, 0, 0, 0]), 2).mask,
                   [1, 0, 1, 0, 0]))
        assert_(eq(masked_not_equal(array([2, 2, 1, 2, 1],
                                          mask=[1, 0, 0, 0, 0]), 2).mask,
                   [1, 0, 1, 0, 1]))
        assert_(eq(masked_where([1, 1, 0, 0, 0], [1, 2, 3, 4, 5]),
                   [99, 99, 3, 4, 5]))
        atest = ones((10, 10, 10), dtype=np.float32)
        btest = zeros(atest.shape, MaskType)
        ctest = masked_where(btest, atest)
        assert_(eq(atest, ctest))
        z = choose(c, (-x, x))
        assert_(eq(z, [1., 2., 0., -4., -5]))
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)
        x = arange(6)
        x[5] = masked
        y = arange(6) * 10
        y[2] = masked
        c = array([1, 1, 1, 0, 0, 0], mask=[1, 0, 0, 0, 0, 0])
        cm = c.filled(1)
        z = where(c, x, y)
        zm = where(cm, x, y)
        assert_(eq(z, zm))
        assert_(getmask(zm) is nomask)
        assert_(eq(zm, [0, 1, 2, 30, 40, 50]))
        z = where(c, masked, 1)
        assert_(eq(z, [99, 99, 99, 1, 1, 1]))
        z = where(c, 1, masked)
        assert_(eq(z, [99, 1, 1, 99, 99, 99]))

    def test_testMinMax2(self):
        # Test of minimum, maximum.
        assert_(eq(minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3]))
        assert_(eq(maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9]))
        x = arange(5)
        y = arange(5) - 2
        x[3] = masked
        y[0] = masked
        assert_(eq(minimum(x, y), where(less(x, y), x, y)))
        assert_(eq(maximum(x, y), where(greater(x, y), x, y)))
        assert_(minimum.reduce(x) == 0)
        assert_(maximum.reduce(x) == 4)

    def test_testTakeTransposeInnerOuter(self):
        # Test of take, transpose, inner, outer products
        x = arange(24)
        y = np.arange(24)
        x[5:6] = masked
        x = x.reshape(2, 3, 4)
        y = y.reshape(2, 3, 4)
        assert_(eq(np.transpose(y, (2, 0, 1)), transpose(x, (2, 0, 1))))
        assert_(eq(np.take(y, (2, 0, 1), 1), take(x, (2, 0, 1), 1)))
        assert_(eq(np.inner(filled(x, 0), filled(y, 0)),
                   inner(x, y)))
        assert_(eq(np.outer(filled(x, 0), filled(y, 0)),
                   outer(x, y)))
        y = array(['abc', 1, 'def', 2, 3], object)
        y[2] = masked
        t = take(y, [0, 3, 4])
        assert_(t[0] == 'abc')
        assert_(t[1] == 2)
        assert_(t[2] == 3)

    def test_testInplace(self):
        # Test of inplace operations and rich comparisons
        y = arange(10)

        x = arange(10)
        xm = arange(10)
        xm[2] = masked
        x += 1
        assert_(eq(x, y + 1))
        xm += 1
        assert_(eq(x, y + 1))

        x = arange(10)
        xm = arange(10)
        xm[2] = masked
        x -= 1
        assert_(eq(x, y - 1))
        xm -= 1
        assert_(eq(xm, y - 1))

        x = arange(10) * 1.0
        xm = arange(10) * 1.0
        xm[2] = masked
        x *= 2.0
        assert_(eq(x, y * 2))
        xm *= 2.0
        assert_(eq(xm, y * 2))

        x = arange(10) * 2
        xm = arange(10)
        xm[2] = masked
        x //= 2
        assert_(eq(x, y))
        xm //= 2
        assert_(eq(x, y))

        x = arange(10) * 1.0
        xm = arange(10) * 1.0
        xm[2] = masked
        x /= 2.0
        assert_(eq(x, y / 2.0))
        xm /= arange(10)
        assert_(eq(xm, ones((10,))))

        x = arange(10).astype(np.float32)
        xm = arange(10)
        xm[2] = masked
        x += 1.
        assert_(eq(x, y + 1.))

    def test_testPickle(self):
        # Test of pickling
        x = arange(12)
        x[4:10:2] = masked
        x = x.reshape(4, 3)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            s = pickle.dumps(x, protocol=proto)
            y = pickle.loads(s)
            assert_(eq(x, y))

    def test_testMasked(self):
        # Test of masked element
        xx = arange(6)
        xx[1] = masked
        assert_(str(masked) == '--')
        assert_(xx[1] is masked)
        assert_equal(filled(xx[1], 0), 0)

    def test_testAverage1(self):
        # Test of average.
        ott = array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        assert_(eq(2.0, average(ott, axis=0)))
        assert_(eq(2.0, average(ott, weights=[1., 1., 2., 1.])))
        result, wts = average(ott, weights=[1., 1., 2., 1.], returned=True)
        assert_(eq(2.0, result))
        assert_(wts == 4.0)
        ott[:] = masked
        assert_(average(ott, axis=0) is masked)
        ott = array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        ott = ott.reshape(2, 2)
        ott[:, 1] = masked
        assert_(eq(average(ott, axis=0), [2.0, 0.0]))
        assert_(average(ott, axis=1)[0] is masked)
        assert_(eq([2., 0.], average(ott, axis=0)))
        result, wts = average(ott, axis=0, returned=True)
        assert_(eq(wts, [1., 0.]))

    def test_testAverage2(self):
        # More tests of average.
        w1 = [0, 1, 1, 1, 1, 0]
        w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
        x = arange(6)
        assert_(allclose(average(x, axis=0), 2.5))
        assert_(allclose(average(x, axis=0, weights=w1), 2.5))
        y = array([arange(6), 2.0 * arange(6)])
        assert_(allclose(average(y, None),
                                 np.add.reduce(np.arange(6)) * 3. / 12.))
        assert_(allclose(average(y, axis=0), np.arange(6) * 3. / 2.))
        assert_(allclose(average(y, axis=1),
                                 [average(x, axis=0), average(x, axis=0) * 2.0]))
        assert_(allclose(average(y, None, weights=w2), 20. / 6.))
        assert_(allclose(average(y, axis=0, weights=w2),
                                 [0., 1., 2., 3., 4., 10.]))
        assert_(allclose(average(y, axis=1),
                                 [average(x, axis=0), average(x, axis=0) * 2.0]))
        m1 = zeros(6)
        m2 = [0, 0, 1, 1, 0, 0]
        m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
        m4 = ones(6)
        m5 = [0, 1, 1, 1, 1, 1]
        assert_(allclose(average(masked_array(x, m1), axis=0), 2.5))
        assert_(allclose(average(masked_array(x, m2), axis=0), 2.5))
        assert_(average(masked_array(x, m4), axis=0) is masked)
        assert_equal(average(masked_array(x, m5), axis=0), 0.0)
        assert_equal(count(average(masked_array(x, m4), axis=0)), 0)
        z = masked_array(y, m3)
        assert_(allclose(average(z, None), 20. / 6.))
        assert_(allclose(average(z, axis=0),
                                 [0., 1., 99., 99., 4.0, 7.5]))
        assert_(allclose(average(z, axis=1), [2.5, 5.0]))
        assert_(allclose(average(z, axis=0, weights=w2),
                                 [0., 1., 99., 99., 4.0, 10.0]))

        a = arange(6)
        b = arange(6) * 3
        r1, w1 = average([[a, b], [b, a]], axis=1, returned=True)
        assert_equal(shape(r1), shape(w1))
        assert_equal(r1.shape, w1.shape)
        r2, w2 = average(ones((2, 2, 3)), axis=0, weights=[3, 1], returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), weights=ones((2, 2, 3)), returned=True)
        assert_(shape(w2) == shape(r2))
        a2d = array([[1, 2], [0, 4]], float)
        a2dm = masked_array(a2d, [[0, 0], [1, 0]])
        a2da = average(a2d, axis=0)
        assert_(eq(a2da, [0.5, 3.0]))
        a2dma = average(a2dm, axis=0)
        assert_(eq(a2dma, [1.0, 3.0]))
        a2dma = average(a2dm, axis=None)
        assert_(eq(a2dma, 7. / 3.))
        a2dma = average(a2dm, axis=1)
        assert_(eq(a2dma, [1.5, 4.0]))

    def test_testToPython(self):
        assert_equal(1, int(array(1)))
        assert_equal(1.0, float(array(1)))
        assert_equal(1, int(array([[[1]]])))
        assert_equal(1.0, float(array([[1]])))
        assert_raises(TypeError, float, array([1, 1]))
        assert_raises(ValueError, bool, array([0, 1]))
        assert_raises(ValueError, bool, array([0, 0], mask=[0, 1]))

    def test_testScalarArithmetic(self):
        xm = array(0, mask=1)
        # TODO FIXME: Find out what the following raises a warning in r8247
        with np.errstate(divide='ignore'):
            assert_((1 / array(0)).mask)
        assert_((1 + xm).mask)
        assert_((-xm).mask)
        assert_((-xm).mask)
        assert_(maximum(xm, xm).mask)
        assert_(minimum(xm, xm).mask)
        assert_(xm.filled().dtype is xm._data.dtype)
        x = array(0, mask=0)
        assert_(x.filled() == x._data)
        assert_equal(str(xm), str(masked_print_option))

    def test_testArrayMethods(self):
        a = array([1, 3, 2])
        assert_(eq(a.any(), a._data.any()))
        assert_(eq(a.all(), a._data.all()))
        assert_(eq(a.argmax(), a._data.argmax()))
        assert_(eq(a.argmin(), a._data.argmin()))
        assert_(eq(a.choose(0, 1, 2, 3, 4),
                           a._data.choose(0, 1, 2, 3, 4)))
        assert_(eq(a.compress([1, 0, 1]), a._data.compress([1, 0, 1])))
        assert_(eq(a.conj(), a._data.conj()))
        assert_(eq(a.conjugate(), a._data.conjugate()))
        m = array([[1, 2], [3, 4]])
        assert_(eq(m.diagonal(), m._data.diagonal()))
        assert_(eq(a.sum(), a._data.sum()))
        assert_(eq(a.take([1, 2]), a._data.take([1, 2])))
        assert_(eq(m.transpose(), m._data.transpose()))

    def test_testArrayAttributes(self):
        a = array([1, 3, 2])
        assert_equal(a.ndim, 1)

    def test_testAPI(self):
        assert_(not [m for m in dir(np.ndarray)
                     if m not in dir(MaskedArray) and
                     not m.startswith('_')])

    def test_testSingleElementSubscript(self):
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a[0].shape, ())
        assert_equal(b[0].shape, ())
        assert_equal(b[1].shape, ())

    def test_assignment_by_condition(self):
        # Test for gh-18951
        a = array([1, 2, 3, 4], mask=[1, 0, 1, 0])
        c = a >= 3
        a[c] = 5
        assert_(a[2] is masked)

    def test_assignment_by_condition_2(self):
        # gh-19721
        a = masked_array([0, 1], mask=[False, False])
        b = masked_array([0, 1], mask=[True, True])
        mask = a < 1
        b[mask] = a[mask]
        expected_mask = [False, True]
        assert_equal(b.mask, expected_mask)


class TestUfuncs:
    def setup_method(self):
        self.d = (array([1.0, 0, -1, pi / 2] * 2, mask=[0, 1] + [0] * 6),
                  array([1.0, 0, -1, pi / 2] * 2, mask=[1, 0] + [0] * 6),)

    def test_testUfuncRegression(self):
        f_invalid_ignore = [
            'sqrt', 'arctanh', 'arcsin', 'arccos',
            'arccosh', 'arctanh', 'log', 'log10', 'divide',
            'true_divide', 'floor_divide', 'remainder', 'fmod']
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
                  'sin', 'cos', 'tan',
                  'arcsin', 'arccos', 'arctan',
                  'sinh', 'cosh', 'tanh',
                  'arcsinh',
                  'arccosh',
                  'arctanh',
                  'absolute', 'fabs', 'negative',
                  'floor', 'ceil',
                  'logical_not',
                  'add', 'subtract', 'multiply',
                  'divide', 'true_divide', 'floor_divide',
                  'remainder', 'fmod', 'hypot', 'arctan2',
                  'equal', 'not_equal', 'less_equal', 'greater_equal',
                  'less', 'greater',
                  'logical_and', 'logical_or', 'logical_xor']:
            try:
                uf = getattr(umath, f)
            except AttributeError:
                uf = getattr(fromnumeric, f)
            mf = getattr(np.ma, f)
            args = self.d[:uf.nin]
            with np.errstate():
                if f in f_invalid_ignore:
                    np.seterr(invalid='ignore')
                if f in ['arctanh', 'log', 'log10']:
                    np.seterr(divide='ignore')
                ur = uf(*args)
                mr = mf(*args)
            assert_(eq(ur.filled(0), mr.filled(0), f))
            assert_(eqmask(ur.mask, mr.mask))

    def test_reduce(self):
        a = self.d[0]
        assert_(not alltrue(a, axis=0))
        assert_(sometrue(a, axis=0))
        assert_equal(sum(a[:3], axis=0), 0)
        assert_equal(product(a, axis=0), 0)

    def test_minmax(self):
        a = arange(1, 13).reshape(3, 4)
        amask = masked_where(a < 5, a)
        assert_equal(amask.max(), a.max())
        assert_equal(amask.min(), 5)
        assert_((amask.max(0) == a.max(0)).all())
        assert_((amask.min(0) == [5, 6, 7, 8]).all())
        assert_(amask.max(1)[0].mask)
        assert_(amask.min(1)[0].mask)

    def test_nonzero(self):
        for t in "?bhilqpBHILQPfdgFDGO":
            x = array([1, 0, 2, 0], mask=[0, 0, 1, 1])
            assert_(eq(nonzero(x), [0]))


class TestArrayMethods:

    def setup_method(self):
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0])
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        self.d = (x, X, XX, m, mx, mX, mXX)

    def test_trace(self):
        (x, X, XX, m, mx, mX, mXX,) = self.d
        mXdiag = mX.diagonal()
        assert_equal(mX.trace(), mX.diagonal().compressed().sum())
        assert_(eq(mX.trace(),
                           X.trace() - sum(mXdiag.mask * X.diagonal(),
                                           axis=0)))

    def test_clip(self):
        (x, X, XX, m, mx, mX, mXX,) = self.d
        clipped = mx.clip(2, 8)
        assert_(eq(clipped.mask, mx.mask))
        assert_(eq(clipped._data, x.clip(2, 8)))
        assert_(eq(clipped._data, mx._data.clip(2, 8)))

    def test_ptp(self):
        (x, X, XX, m, mx, mX, mXX,) = self.d
        (n, m) = X.shape
        # print(type(mx), mx.compressed())
        # raise Exception()
        assert_equal(mx.ptp(), np.ptp(mx.compressed()))
        rows = np.zeros(n, np.float64)
        cols = np.zeros(m, np.float64)
        for k in range(m):
            cols[k] = np.ptp(mX[:, k].compressed())
        for k in range(n):
            rows[k] = np.ptp(mX[k].compressed())
        assert_(eq(mX.ptp(0), cols))
        assert_(eq(mX.ptp(1), rows))

    def test_swapaxes(self):
        (x, X, XX, m, mx, mX, mXX,) = self.d
        mXswapped = mX.swapaxes(0, 1)
        assert_(eq(mXswapped[-1], mX[:, -1]))
        mXXswapped = mXX.swapaxes(0, 2)
        assert_equal(mXXswapped.shape, (2, 2, 3, 3))

    def test_cumprod(self):
        (x, X, XX, m, mx, mX, mXX,) = self.d
        mXcp = mX.cumprod(0)
        assert_(eq(mXcp._data, mX.filled(1).cumprod(0)))
        mXcp = mX.cumprod(1)
        assert_(eq(mXcp._data, mX.filled(1).cumprod(1)))

    def test_cumsum(self):
        (x, X, XX, m, mx, mX, mXX,) = self.d
        mXcp = mX.cumsum(0)
        assert_(eq(mXcp._data, mX.filled(0).cumsum(0)))
        mXcp = mX.cumsum(1)
        assert_(eq(mXcp._data, mX.filled(0).cumsum(1)))

    def test_varstd(self):
        (x, X, XX, m, mx, mX, mXX,) = self.d
        assert_(eq(mX.var(axis=None), mX.compressed().var()))
        assert_(eq(mX.std(axis=None), mX.compressed().std()))
        assert_(eq(mXX.var(axis=3).shape, XX.var(axis=3).shape))
        assert_(eq(mX.var().shape, X.var().shape))
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))
        for k in range(6):
            assert_(eq(mXvar1[k], mX[k].compressed().var()))
            assert_(eq(mXvar0[k], mX[:, k].compressed().var()))
            assert_(eq(np.sqrt(mXvar0[k]),
                               mX[:, k].compressed().std()))


def eqmask(m1, m2):
    if m1 is nomask:
        return m2 is nomask
    if m2 is nomask:
        return m1 is nomask
    return (m1 == m2).all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_coercion.py ===
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
import itertools

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2

import pandas as pd
import pandas._testing as tm

###############################################################
# Index / Series common tests which may trigger dtype coercions
###############################################################


@pytest.fixture(autouse=True, scope="class")
def check_comprehensiveness(request):
    # Iterate over combination of dtype, method and klass
    # and ensure that each are contained within a collected test
    cls = request.cls
    combos = itertools.product(cls.klasses, cls.dtypes, [cls.method])

    def has_test(combo):
        klass, dtype, method = combo
        cls_funcs = request.node.session.items
        return any(
            klass in x.name and dtype in x.name and method in x.name for x in cls_funcs
        )

    opts = request.config.option
    if opts.lf or opts.keyword:
        # If we are running with "last-failed" or -k foo, we expect to only
        #  run a subset of tests.
        yield

    else:
        for combo in combos:
            if not has_test(combo):
                raise AssertionError(
                    f"test method is not defined: {cls.__name__}, {combo}"
                )

        yield


class CoercionBase:
    klasses = ["index", "series"]
    dtypes = [
        "object",
        "int64",
        "float64",
        "complex128",
        "bool",
        "datetime64",
        "datetime64tz",
        "timedelta64",
        "period",
    ]

    @property
    def method(self):
        raise NotImplementedError(self)


class TestSetitemCoercion(CoercionBase):
    method = "setitem"

    # disable comprehensiveness tests, as most of these have been moved to
    #  tests.series.indexing.test_setitem in SetitemCastingEquivalents subclasses.
    klasses: list[str] = []

    def test_setitem_series_no_coercion_from_values_list(self):
        # GH35865 - int casted to str when internally calling np.array(ser.values)
        ser = pd.Series(["a", 1])
        ser[:] = list(ser.values)

        expected = pd.Series(["a", 1])

        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(
        self, original_series, loc_key, expected_index, expected_dtype
    ):
        """test index's coercion triggered by assign key"""
        temp = original_series.copy()
        # GH#33469 pre-2.0 with int loc_key and temp.index.dtype == np.float64
        #  `temp[loc_key] = 5` treated loc_key as positional
        temp[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

        temp = original_series.copy()
        temp.loc[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize(
        "val,exp_dtype", [("x", object), (5, IndexError), (1.1, object)]
    )
    def test_setitem_index_object(self, val, exp_dtype):
        obj = pd.Series([1, 2, 3, 4], index=pd.Index(list("abcd"), dtype=object))
        assert obj.index.dtype == object

        if exp_dtype is IndexError:
            temp = obj.copy()
            warn_msg = "Series.__setitem__ treating keys as positions is deprecated"
            msg = "index 5 is out of bounds for axis 0 with size 4"
            with pytest.raises(exp_dtype, match=msg):
                with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                    temp[5] = 5
        else:
            exp_index = pd.Index(list("abcd") + [val], dtype=object)
            self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.int64), (1.1, np.float64), ("x", object)]
    )
    def test_setitem_index_int64(self, val, exp_dtype):
        obj = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64

        exp_index = pd.Index([0, 1, 2, 3, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.float64), (5.1, np.float64), ("x", object)]
    )
    def test_setitem_index_float64(self, val, exp_dtype, request):
        obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64

        exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_series_period(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64tz(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_period(self):
        raise NotImplementedError


class TestInsertIndexCoercion(CoercionBase):
    klasses = ["index"]
    method = "insert"

    def _assert_insert_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by insert"""
        target = original.copy()
        res = target.insert(1, value)
        tm.assert_index_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, object),
            (1.1, 1.1, object),
            (False, False, object),
            ("x", "x", object),
        ],
    )
    def test_insert_index_object(self, insert, coerced_val, coerced_dtype):
        obj = pd.Index(list("abcd"), dtype=object)
        assert obj.dtype == object

        exp = pd.Index(["a", coerced_val, "b", "c", "d"], dtype=object)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, None),
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_int_index(
        self, any_int_numpy_dtype, insert, coerced_val, coerced_dtype
    ):
        dtype = any_int_numpy_dtype
        obj = pd.Index([1, 2, 3, 4], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype

        exp = pd.Index([1, coerced_val, 2, 3, 4], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1.0, None),
            # When float_numpy_dtype=float32, this is not the case
            # see the correction below
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_float_index(
        self, float_numpy_dtype, insert, coerced_val, coerced_dtype
    ):
        dtype = float_numpy_dtype
        obj = pd.Index([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype

        if np_version_gt2 and dtype == "float32" and coerced_val == 1.1:
            # Hack, in the 2nd test case, since 1.1 can be losslessly cast to float32
            # the expected dtype will be float32 if the original dtype was float32
            coerced_dtype = np.float32
        exp = pd.Index([1.0, coerced_val, 2.0, 3.0, 4.0], dtype=coerced_dtype)
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
        ],
        ids=["datetime64", "datetime64tz"],
    )
    @pytest.mark.parametrize(
        "insert_value",
        [pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01", tz="Asia/Tokyo"), 1],
    )
    def test_insert_index_datetimes(self, fill_val, exp_dtype, insert_value):
        obj = pd.DatetimeIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], tz=fill_val.tz
        ).as_unit("ns")
        assert obj.dtype == exp_dtype

        exp = pd.DatetimeIndex(
            ["2011-01-01", fill_val.date(), "2011-01-02", "2011-01-03", "2011-01-04"],
            tz=fill_val.tz,
        ).as_unit("ns")
        self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)

        if fill_val.tz:
            # mismatched tzawareness
            ts = pd.Timestamp("2012-01-01")
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo")
            result = obj.insert(1, ts)
            # once deprecation is enforced:
            expected = obj.insert(1, ts.tz_convert(obj.dtype.tz))
            assert expected.dtype == obj.dtype
            tm.assert_index_equal(result, expected)

        else:
            # mismatched tzawareness
            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo")
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

        item = 1
        result = obj.insert(1, item)
        expected = obj.astype(object).insert(1, item)
        assert expected[1] == item
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)

    def test_insert_index_timedelta64(self):
        obj = pd.TimedeltaIndex(["1 day", "2 day", "3 day", "4 day"])
        assert obj.dtype == "timedelta64[ns]"

        # timedelta64 + timedelta64 => timedelta64
        exp = pd.TimedeltaIndex(["1 day", "10 day", "2 day", "3 day", "4 day"])
        self._assert_insert_conversion(
            obj, pd.Timedelta("10 day"), exp, "timedelta64[ns]"
        )

        for item in [pd.Timestamp("2012-01-01"), 1]:
            result = obj.insert(1, item)
            expected = obj.astype(object).insert(1, item)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (pd.Period("2012-01", freq="M"), "2012-01", "period[M]"),
            (pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01"), object),
            (1, 1, object),
            ("x", "x", object),
        ],
    )
    def test_insert_index_period(self, insert, coerced_val, coerced_dtype):
        obj = pd.PeriodIndex(["2011-01", "2011-02", "2011-03", "2011-04"], freq="M")
        assert obj.dtype == "period[M]"

        data = [
            pd.Period("2011-01", freq="M"),
            coerced_val,
            pd.Period("2011-02", freq="M"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-04", freq="M"),
        ]
        if isinstance(insert, pd.Period):
            exp = pd.PeriodIndex(data, freq="M")
            self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

            # string that can be parsed to appropriate PeriodDtype
            self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)

        else:
            result = obj.insert(0, insert)
            expected = obj.astype(object).insert(0, insert)
            tm.assert_index_equal(result, expected)

            # TODO: ATM inserting '2012-01-01 00:00:00' when we have obj.freq=="M"
            #  casts that string to Period[M], not clear that is desirable
            if not isinstance(insert, pd.Timestamp):
                # non-castable string
                result = obj.insert(0, str(insert))
                expected = obj.astype(object).insert(0, str(insert))
                tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_bool(self):
        raise NotImplementedError


class TestWhereCoercion(CoercionBase):
    method = "where"
    _cond = np.array([True, False, True, False])

    def _assert_where_conversion(
        self, original, cond, values, expected, expected_dtype
    ):
        """test coercion triggered by where"""
        target = original.copy()
        res = target.where(cond, values)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    def _construct_exp(self, obj, klass, fill_val, exp_dtype):
        if fill_val is True:
            values = klass([True, False, True, True])
        elif isinstance(fill_val, (datetime, np.datetime64)):
            values = pd.date_range(fill_val, periods=4)
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])

        exp = klass([obj[0], values[1], obj[2], values[3]], dtype=exp_dtype)
        return values, exp

    def _run_test(self, obj, fill_val, klass, exp_dtype):
        cond = klass(self._cond)

        exp = klass([obj[0], fill_val, obj[2], fill_val], dtype=exp_dtype)
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        values, exp = self._construct_exp(obj, klass, fill_val, exp_dtype)
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_where_object(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        obj = klass(list("abcd"), dtype=object)
        assert obj.dtype == object
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_int64(self, index_or_series, fill_val, exp_dtype, request):
        klass = index_or_series

        obj = klass([1, 2, 3, 4])
        assert obj.dtype == np.int64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val, exp_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_float64(self, index_or_series, fill_val, exp_dtype, request):
        klass = index_or_series

        obj = klass([1.1, 2.2, 3.3, 4.4])
        assert obj.dtype == np.float64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_where_complex128(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        obj = klass([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, np.bool_)],
    )
    def test_where_series_bool(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series

        obj = klass([True, False, True, False])
        assert obj.dtype == np.bool_
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), object),
        ],
        ids=["datetime64", "datetime64tz"],
    )
    def test_where_datetime64(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series

        obj = klass(pd.date_range("2011-01-01", periods=4, freq="D")._with_freq(None))
        assert obj.dtype == "datetime64[ns]"

        fv = fill_val
        # do the check with each of the available datetime scalars
        if exp_dtype == "datetime64[ns]":
            for scalar in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
                self._run_test(obj, scalar, klass, exp_dtype)
        else:
            for scalar in [fv, fv.to_pydatetime()]:
                self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_series_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_series_period(self):
        raise NotImplementedError

    @pytest.mark.parametrize(
        "value", [pd.Timedelta(days=9), timedelta(days=9), np.timedelta64(9, "D")]
    )
    def test_where_index_timedelta64(self, value):
        tdi = pd.timedelta_range("1 Day", periods=4)
        cond = np.array([True, False, False, True])

        expected = pd.TimedeltaIndex(["1 Day", value, value, "4 Days"])
        result = tdi.where(cond, value)
        tm.assert_index_equal(result, expected)

        # wrong-dtyped NaT
        dtnat = np.datetime64("NaT", "ns")
        expected = pd.Index([tdi[0], dtnat, dtnat, tdi[3]], dtype=object)
        assert expected[1] is dtnat

        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)

    def test_where_index_period(self):
        dti = pd.date_range("2016-01-01", periods=3, freq="QS")
        pi = dti.to_period("Q")

        cond = np.array([False, True, False])

        # Passing a valid scalar
        value = pi[-1] + pi.freq * 10
        expected = pd.PeriodIndex([value, pi[1], value])
        result = pi.where(cond, value)
        tm.assert_index_equal(result, expected)

        # Case passing ndarray[object] of Periods
        other = np.asarray(pi + pi.freq * 10, dtype=object)
        result = pi.where(cond, other)
        expected = pd.PeriodIndex([other[0], pi[1], other[2]])
        tm.assert_index_equal(result, expected)

        # Passing a mismatched scalar -> casts to object
        td = pd.Timedelta(days=4)
        expected = pd.Index([td, pi[1], td], dtype=object)
        result = pi.where(cond, td)
        tm.assert_index_equal(result, expected)

        per = pd.Period("2020-04-21", "D")
        expected = pd.Index([per, pi[1], per], dtype=object)
        result = pi.where(cond, per)
        tm.assert_index_equal(result, expected)


class TestFillnaSeriesCoercion(CoercionBase):
    # not indexing, but place here for consistency

    method = "fillna"

    @pytest.mark.xfail(reason="Test not implemented")
    def test_has_comprehensive_tests(self):
        raise NotImplementedError

    def _assert_fillna_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by fillna"""
        target = original.copy()
        res = target.fillna(value)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize(
        "fill_val, fill_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_fillna_object(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass(["a", np.nan, "c", "d"], dtype=object)
        assert obj.dtype == object

        exp = klass(["a", fill_val, "c", "d"], dtype=object)
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_fillna_float64(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass([1.1, np.nan, 3.3, 4.4])
        assert obj.dtype == np.float64

        exp = klass([1.1, fill_val, 3.3, 4.4])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_fillna_complex128(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass([1 + 1j, np.nan, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128

        exp = klass([1 + 1j, fill_val, 3 + 3j, 4 + 4j])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), object),
            (1, object),
            ("x", object),
        ],
        ids=["datetime64", "datetime64tz", "object", "object"],
    )
    def test_fillna_datetime(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass(
            [
                pd.Timestamp("2011-01-01"),
                pd.NaT,
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        assert obj.dtype == "datetime64[ns]"

        exp = klass(
            [
                pd.Timestamp("2011-01-01"),
                fill_val,
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
            (pd.Timestamp("2012-01-01"), object),
            # pre-2.0 with a mismatched tz we would get object result
            (pd.Timestamp("2012-01-01", tz="Asia/Tokyo"), "datetime64[ns, US/Eastern]"),
            (1, object),
            ("x", object),
        ],
    )
    def test_fillna_datetime64tz(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        tz = "US/Eastern"

        obj = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.NaT,
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        assert obj.dtype == "datetime64[ns, US/Eastern]"

        if getattr(fill_val, "tz", None) is None:
            fv = fill_val
        else:
            fv = fill_val.tz_convert(tz)
        exp = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                fv,
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val",
        [
            1,
            1.1,
            1 + 1j,
            True,
            pd.Interval(1, 2, closed="left"),
            pd.Timestamp("2012-01-01", tz="US/Eastern"),
            pd.Timestamp("2012-01-01"),
            pd.Timedelta(days=1),
            pd.Period("2016-01-01", "D"),
        ],
    )
    def test_fillna_interval(self, index_or_series, fill_val):
        ii = pd.interval_range(1.0, 5.0, closed="right").insert(1, np.nan)
        assert isinstance(ii.dtype, pd.IntervalDtype)
        obj = index_or_series(ii)

        exp = index_or_series([ii[0], fill_val, ii[2], ii[3], ii[4]], dtype=object)

        fill_dtype = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_int64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_int64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.parametrize(
        "fill_val",
        [
            1,
            1.1,
            1 + 1j,
            True,
            pd.Interval(1, 2, closed="left"),
            pd.Timestamp("2012-01-01", tz="US/Eastern"),
            pd.Timestamp("2012-01-01"),
            pd.Timedelta(days=1),
            pd.Period("2016-01-01", "W"),
        ],
    )
    def test_fillna_series_period(self, index_or_series, fill_val):
        pi = pd.period_range("2016-01-01", periods=4, freq="D").insert(1, pd.NaT)
        assert isinstance(pi.dtype, pd.PeriodDtype)
        obj = index_or_series(pi)

        exp = index_or_series([pi[0], fill_val, pi[2], pi[3], pi[4]], dtype=object)

        fill_dtype = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_period(self):
        raise NotImplementedError


class TestReplaceSeriesCoercion(CoercionBase):
    klasses = ["series"]
    method = "replace"

    rep: dict[str, list] = {}
    rep["object"] = ["a", "b"]
    rep["int64"] = [4, 5]
    rep["float64"] = [1.1, 2.2]
    rep["complex128"] = [1 + 1j, 2 + 2j]
    rep["bool"] = [True, False]
    rep["datetime64[ns]"] = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-03")]

    for tz in ["UTC", "US/Eastern"]:
        # to test tz => different tz replacement
        key = f"datetime64[ns, {tz}]"
        rep[key] = [
            pd.Timestamp("2011-01-01", tz=tz),
            pd.Timestamp("2011-01-03", tz=tz),
        ]

    rep["timedelta64[ns]"] = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]

    @pytest.fixture(params=["dict", "series"])
    def how(self, request):
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        ]
    )
    def from_key(self, request):
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        ],
        ids=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64",
            "datetime64tz",
            "datetime64tz",
            "timedelta64",
        ],
    )
    def to_key(self, request):
        return request.param

    @pytest.fixture
    def replacer(self, how, from_key, to_key):
        """
        Object we will pass to `Series.replace`
        """
        if how == "dict":
            replacer = dict(zip(self.rep[from_key], self.rep[to_key]))
        elif how == "series":
            replacer = pd.Series(self.rep[to_key], index=self.rep[from_key])
        else:
            raise ValueError
        return replacer

    def test_replace_series(self, how, to_key, from_key, replacer, using_infer_string):
        index = pd.Index([3, 4], name="xxx")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        obj = obj.astype(from_key)
        assert obj.dtype == from_key

        if from_key.startswith("datetime") and to_key.startswith("datetime"):
            # tested below
            return
        elif from_key in ["datetime64[ns, US/Eastern]", "datetime64[ns, UTC]"]:
            # tested below
            return

        if (from_key == "float64" and to_key in ("int64")) or (
            from_key == "complex128" and to_key in ("int64", "float64")
        ):
            if not IS64 or is_platform_windows():
                pytest.skip(f"32-bit platform buggy: {from_key} -> {to_key}")

            # Expected: do not downcast by replacement
            exp = pd.Series(self.rep[to_key], index=index, name="yyy", dtype=from_key)

        else:
            exp = pd.Series(self.rep[to_key], index=index, name="yyy")

        if using_infer_string and exp.dtype == "string":
            # with infer_string, we disable the deprecated downcasting behavior
            exp = exp.astype(object)

        msg = "Downcasting behavior in `replace`"
        warn = FutureWarning
        if (
            exp.dtype == obj.dtype
            or exp.dtype == object
            or (exp.dtype.kind in "iufc" and obj.dtype.kind in "iufc")
        ):
            warn = None
        with tm.assert_produces_warning(warn, match=msg):
            result = obj.replace(replacer)

        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "to_key",
        ["timedelta64[ns]", "bool", "object", "complex128", "float64", "int64"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key", ["datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"], indirect=True
    )
    def test_replace_series_datetime_tz(
        self, how, to_key, from_key, replacer, using_infer_string
    ):
        index = pd.Index([3, 4], name="xyz")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        assert obj.dtype == from_key

        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        if using_infer_string and exp.dtype == "string":
            # with infer_string, we disable the deprecated downcasting behavior
            exp = exp.astype(object)
        else:
            assert exp.dtype == to_key

        msg = "Downcasting behavior in `replace`"
        warn = FutureWarning if exp.dtype != object else None
        with tm.assert_produces_warning(warn, match=msg):
            result = obj.replace(replacer)

        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "to_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    def test_replace_series_datetime_datetime(self, how, to_key, from_key, replacer):
        index = pd.Index([3, 4], name="xyz")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        assert obj.dtype == from_key

        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        warn = FutureWarning
        if isinstance(obj.dtype, pd.DatetimeTZDtype) and isinstance(
            exp.dtype, pd.DatetimeTZDtype
        ):
            # with mismatched tzs, we retain the original dtype as of 2.0
            exp = exp.astype(obj.dtype)
            warn = None
        else:
            assert exp.dtype == to_key
            if to_key == from_key:
                warn = None

        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(warn, match=msg):
            result = obj.replace(replacer)

        tm.assert_series_equal(result, exp)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_replace_series_period(self):
        raise NotImplementedError