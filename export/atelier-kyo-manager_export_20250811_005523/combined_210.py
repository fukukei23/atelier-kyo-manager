
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_repr.py ===
from datetime import (
    datetime,
    timedelta,
)
from io import StringIO

import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    DataFrame,
    IntervalIndex,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    Timestamp,
    date_range,
    option_context,
    period_range,
)
import pandas._testing as tm


class TestDataFrameRepr:
    def test_repr_should_return_str(self):
        # https://docs.python.org/3/reference/datamodel.html#object.__repr__
        # "...The return value must be a string object."

        # (str on py2.x, str (unicode) on py3)

        data = [8, 5, 3, 5]
        index1 = ["\u03c3", "\u03c4", "\u03c5", "\u03c6"]
        cols = ["\u03c8"]
        df = DataFrame(data, columns=cols, index=index1)
        assert type(df.__repr__()) is str  # noqa: E721

        ser = df[cols[0]]
        assert type(ser.__repr__()) is str  # noqa: E721

    def test_repr_bytes_61_lines(self):
        # GH#12857
        lets = list("ACDEFGHIJKLMNOP")
        words = np.random.default_rng(2).choice(lets, (1000, 50))
        df = DataFrame(words).astype("U1")
        assert (df.dtypes == object).all()

        # smoke tests; at one point this raised with 61 but not 60
        repr(df)
        repr(df.iloc[:60, :])
        repr(df.iloc[:61, :])

    def test_repr_unicode_level_names(self, frame_or_series):
        index = MultiIndex.from_tuples([(0, 0), (1, 1)], names=["\u0394", "i1"])

        obj = DataFrame(np.random.default_rng(2).standard_normal((2, 4)), index=index)
        obj = tm.get_obj(obj, frame_or_series)
        repr(obj)

    def test_assign_index_sequences(self):
        # GH#2200
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}).set_index(
            ["a", "b"]
        )
        index = list(df.index)
        index[0] = ("faz", "boo")
        df.index = index
        repr(df)

        # this travels an improper code path
        index[0] = ["faz", "boo"]
        df.index = index
        repr(df)

    def test_repr_with_mi_nat(self):
        df = DataFrame({"X": [1, 2]}, index=[[NaT, Timestamp("20130101")], ["a", "b"]])
        result = repr(df)
        expected = "              X\nNaT        a  1\n2013-01-01 b  2"
        assert result == expected

    def test_repr_with_different_nulls(self):
        # GH45263
        df = DataFrame([1, 2, 3, 4], [True, None, np.nan, NaT])
        result = repr(df)
        expected = """      0
True  1
None  2
NaN   3
NaT   4"""
        assert result == expected

    def test_repr_with_different_nulls_cols(self):
        # GH45263
        d = {np.nan: [1, 2], None: [3, 4], NaT: [6, 7], True: [8, 9]}
        df = DataFrame(data=d)
        result = repr(df)
        expected = """   NaN  None  NaT  True
0    1     3    6     8
1    2     4    7     9"""
        assert result == expected

    def test_multiindex_na_repr(self):
        # only an issue with long columns
        df3 = DataFrame(
            {
                "A" * 30: {("A", "A0006000", "nuit"): "A0006000"},
                "B" * 30: {("A", "A0006000", "nuit"): np.nan},
                "C" * 30: {("A", "A0006000", "nuit"): np.nan},
                "D" * 30: {("A", "A0006000", "nuit"): np.nan},
                "E" * 30: {("A", "A0006000", "nuit"): "A"},
                "F" * 30: {("A", "A0006000", "nuit"): np.nan},
            }
        )

        idf = df3.set_index(["A" * 30, "C" * 30])
        repr(idf)

    def test_repr_name_coincide(self):
        index = MultiIndex.from_tuples(
            [("a", 0, "foo"), ("b", 1, "bar")], names=["a", "b", "c"]
        )

        df = DataFrame({"value": [0, 1]}, index=index)

        lines = repr(df).split("\n")
        assert lines[2].startswith("a 0 foo")

    def test_repr_to_string(
        self,
        multiindex_year_month_day_dataframe_random_data,
        multiindex_dataframe_random_data,
    ):
        ymd = multiindex_year_month_day_dataframe_random_data
        frame = multiindex_dataframe_random_data

        repr(frame)
        repr(ymd)
        repr(frame.T)
        repr(ymd.T)

        buf = StringIO()
        frame.to_string(buf=buf)
        ymd.to_string(buf=buf)
        frame.T.to_string(buf=buf)
        ymd.T.to_string(buf=buf)

    def test_repr_empty(self):
        # empty
        repr(DataFrame())

        # empty with index
        frame = DataFrame(index=np.arange(1000))
        repr(frame)

    def test_repr_mixed(self, float_string_frame):
        # mixed
        repr(float_string_frame)

    @pytest.mark.slow
    def test_repr_mixed_big(self):
        # big mixed
        biggie = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),
                "B": [str(i) for i in range(200)],
            },
            index=range(200),
        )
        biggie.loc[:20, "A"] = np.nan
        biggie.loc[:20, "B"] = np.nan

        repr(biggie)

    def test_repr(self):
        # columns but no index
        no_index = DataFrame(columns=[0, 1, 3])
        repr(no_index)

        df = DataFrame(["a\n\r\tb"], columns=["a\n\r\td"], index=["a\n\r\tf"])
        assert "\t" not in repr(df)
        assert "\r" not in repr(df)
        assert "a\n" not in repr(df)

    def test_repr_dimensions(self):
        df = DataFrame([[1, 2], [3, 4]])
        with option_context("display.show_dimensions", True):
            assert "2 rows x 2 columns" in repr(df)

        with option_context("display.show_dimensions", False):
            assert "2 rows x 2 columns" not in repr(df)

        with option_context("display.show_dimensions", "truncate"):
            assert "2 rows x 2 columns" not in repr(df)

    @pytest.mark.slow
    def test_repr_big(self):
        # big one
        biggie = DataFrame(np.zeros((200, 4)), columns=range(4), index=range(200))
        repr(biggie)

    def test_repr_unsortable(self):
        # columns are not sortable

        unsortable = DataFrame(
            {
                "foo": [1] * 50,
                datetime.today(): [1] * 50,
                "bar": ["bar"] * 50,
                datetime.today() + timedelta(1): ["bar"] * 50,
            },
            index=np.arange(50),
        )
        repr(unsortable)

    def test_repr_float_frame_options(self, float_frame):
        repr(float_frame)

        with option_context("display.precision", 3):
            repr(float_frame)

        with option_context("display.max_rows", 10, "display.max_columns", 2):
            repr(float_frame)

        with option_context("display.max_rows", 1000, "display.max_columns", 1000):
            repr(float_frame)

    def test_repr_unicode(self):
        uval = "\u03c3\u03c3\u03c3\u03c3"

        df = DataFrame({"A": [uval, uval]})

        result = repr(df)
        ex_top = "      A"
        assert result.split("\n")[0].rstrip() == ex_top

        df = DataFrame({"A": [uval, uval]})
        result = repr(df)
        assert result.split("\n")[0].rstrip() == ex_top

    def test_unicode_string_with_unicode(self):
        df = DataFrame({"A": ["\u05d0"]})
        str(df)

    def test_repr_unicode_columns(self):
        df = DataFrame({"\u05d0": [1, 2, 3], "\u05d1": [4, 5, 6], "c": [7, 8, 9]})
        repr(df.columns)  # should not raise UnicodeDecodeError

    def test_str_to_bytes_raises(self):
        # GH 26447
        df = DataFrame({"A": ["abc"]})
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_very_wide_repr(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 20)),
            columns=np.array(["a" * 10] * 20, dtype=object),
        )
        repr(df)

    def test_repr_column_name_unicode_truncation_bug(self):
        # #1906
        df = DataFrame(
            {
                "Id": [7117434],
                "StringCol": (
                    "Is it possible to modify drop plot code"
                    "so that the output graph is displayed "
                    "in iphone simulator, Is it possible to "
                    "modify drop plot code so that the "
                    "output graph is \xe2\x80\xa8displayed "
                    "in iphone simulator.Now we are adding "
                    "the CSV file externally. I want to Call "
                    "the File through the code.."
                ),
            }
        )

        with option_context("display.max_columns", 20):
            assert "StringCol" in repr(df)

    def test_latex_repr(self):
        pytest.importorskip("jinja2")
        expected = r"""\begin{tabular}{llll}
\toprule
 & 0 & 1 & 2 \\
\midrule
0 & $\alpha$ & b & c \\
1 & 1 & 2 & 3 \\
\bottomrule
\end{tabular}
"""
        with option_context(
            "styler.format.escape", None, "styler.render.repr", "latex"
        ):
            df = DataFrame([[r"$\alpha$", "b", "c"], [1, 2, 3]])
            result = df._repr_latex_()
            assert result == expected

        # GH 12182
        assert df._repr_latex_() is None

    def test_repr_with_datetimeindex(self):
        df = DataFrame({"A": [1, 2, 3]}, index=date_range("2000", periods=3))
        result = repr(df)
        expected = "            A\n2000-01-01  1\n2000-01-02  2\n2000-01-03  3"
        assert result == expected

    def test_repr_with_intervalindex(self):
        # https://github.com/pandas-dev/pandas/pull/24134/files
        df = DataFrame(
            {"A": [1, 2, 3, 4]}, index=IntervalIndex.from_breaks([0, 1, 2, 3, 4])
        )
        result = repr(df)
        expected = "        A\n(0, 1]  1\n(1, 2]  2\n(2, 3]  3\n(3, 4]  4"
        assert result == expected

    def test_repr_with_categorical_index(self):
        df = DataFrame({"A": [1, 2, 3]}, index=CategoricalIndex(["a", "b", "c"]))
        result = repr(df)
        expected = "   A\na  1\nb  2\nc  3"
        assert result == expected

    def test_repr_categorical_dates_periods(self):
        # normal DataFrame
        dt = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        p = period_range("2011-01", freq="M", periods=5)
        df = DataFrame({"dt": dt, "p": p})
        exp = """                         dt        p
0 2011-01-01 09:00:00-05:00  2011-01
1 2011-01-01 10:00:00-05:00  2011-02
2 2011-01-01 11:00:00-05:00  2011-03
3 2011-01-01 12:00:00-05:00  2011-04
4 2011-01-01 13:00:00-05:00  2011-05"""

        assert repr(df) == exp

        df2 = DataFrame({"dt": Categorical(dt), "p": Categorical(p)})
        assert repr(df2) == exp

    @pytest.mark.parametrize("arg", [np.datetime64, np.timedelta64])
    @pytest.mark.parametrize(
        "box, expected",
        [[Series, "0    NaT\ndtype: object"], [DataFrame, "     0\n0  NaT"]],
    )
    def test_repr_np_nat_with_object(self, arg, box, expected):
        # GH 25445
        result = repr(box([arg("NaT")], dtype=object))
        assert result == expected

    def test_frame_datetime64_pre1900_repr(self):
        df = DataFrame({"year": date_range("1/1/1700", periods=50, freq="YE-DEC")})
        # it works!
        repr(df)

    def test_frame_to_string_with_periodindex(self):
        index = PeriodIndex(["2011-1", "2011-2", "2011-3"], freq="M")
        frame = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), index=index)

        # it works!
        frame.to_string()

    def test_to_string_ea_na_in_multiindex(self):
        # GH#47986
        df = DataFrame(
            {"a": [1, 2]},
            index=MultiIndex.from_arrays([Series([NA, 1], dtype="Int64")]),
        )

        result = df.to_string()
        expected = """      a
<NA>  1
1     2"""
        assert result == expected

    def test_datetime64tz_slice_non_truncate(self):
        # GH 30263
        df = DataFrame({"x": date_range("2019", periods=10, tz="UTC")})
        expected = repr(df)
        df = df.iloc[:, :5]
        result = repr(df)
        assert result == expected

    def test_to_records_no_typeerror_in_repr(self):
        # GH 48526
        df = DataFrame([["a", "b"], ["c", "d"], ["e", "f"]], columns=["left", "right"])
        df["record"] = df[["left", "right"]].to_records()
        expected = """  left right     record
0    a     b  [0, a, b]
1    c     d  [1, c, d]
2    e     f  [2, e, f]"""
        result = repr(df)
        assert result == expected

    def test_to_records_with_na_record_value(self):
        # GH 48526
        df = DataFrame(
            [["a", np.nan], ["c", "d"], ["e", "f"]], columns=["left", "right"]
        )
        df["record"] = df[["left", "right"]].to_records()
        expected = """  left right       record
0    a   NaN  [0, a, nan]
1    c     d    [1, c, d]
2    e     f    [2, e, f]"""
        result = repr(df)
        assert result == expected

    def test_to_records_with_na_record(self):
        # GH 48526
        df = DataFrame(
            [["a", "b"], [np.nan, np.nan], ["e", "f"]], columns=[np.nan, "right"]
        )
        df["record"] = df[[np.nan, "right"]].to_records()
        expected = """   NaN right         record
0    a     b      [0, a, b]
1  NaN   NaN  [1, nan, nan]
2    e     f      [2, e, f]"""
        result = repr(df)
        assert result == expected

    def test_to_records_with_inf_as_na_record(self):
        # GH 48526
        expected = """   NaN  inf         record
0  inf    b    [0, inf, b]
1  NaN  NaN  [1, nan, nan]
2    e    f      [2, e, f]"""
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with option_context("use_inf_as_na", True):
                df = DataFrame(
                    [[np.inf, "b"], [np.nan, np.nan], ["e", "f"]],
                    columns=[np.nan, np.inf],
                )
                df["record"] = df[[np.nan, np.inf]].to_records()
                result = repr(df)
        assert result == expected

    def test_to_records_with_inf_record(self):
        # GH 48526
        expected = """   NaN  inf         record
0  inf    b    [0, inf, b]
1  NaN  NaN  [1, nan, nan]
2    e    f      [2, e, f]"""
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with option_context("use_inf_as_na", False):
                df = DataFrame(
                    [[np.inf, "b"], [np.nan, np.nan], ["e", "f"]],
                    columns=[np.nan, np.inf],
                )
                df["record"] = df[[np.nan, np.inf]].to_records()
                result = repr(df)
        assert result == expected

    def test_masked_ea_with_formatter(self):
        # GH#39336
        df = DataFrame(
            {
                "a": Series([0.123456789, 1.123456789], dtype="Float64"),
                "b": Series([1, 2], dtype="Int64"),
            }
        )
        result = df.to_string(formatters=["{:.2f}".format, "{:.2f}".format])
        expected = """      a     b
0  0.12  1.00
1  1.12  2.00"""
        assert result == expected

    def test_repr_ea_columns(self, any_string_dtype):
        # GH#54797
        pytest.importorskip("pyarrow")
        df = DataFrame({"long_column_name": [1, 2, 3], "col2": [4, 5, 6]})
        df.columns = df.columns.astype(any_string_dtype)
        expected = """   long_column_name  col2
0                 1     4
1                 2     5
2                 3     6"""
        assert repr(df) == expected


@pytest.mark.parametrize(
    "data,output",
    [
        ([2, complex("nan"), 1], [" 2.0+0.0j", " NaN+0.0j", " 1.0+0.0j"]),
        ([2, complex("nan"), -1], [" 2.0+0.0j", " NaN+0.0j", "-1.0+0.0j"]),
        ([-2, complex("nan"), -1], ["-2.0+0.0j", " NaN+0.0j", "-1.0+0.0j"]),
        ([-1.23j, complex("nan"), -1], ["-0.00-1.23j", "  NaN+0.00j", "-1.00+0.00j"]),
        ([1.23j, complex("nan"), 1.23], [" 0.00+1.23j", "  NaN+0.00j", " 1.23+0.00j"]),
        (
            [-1.23j, complex(np.nan, np.nan), 1],
            ["-0.00-1.23j", "  NaN+ NaNj", " 1.00+0.00j"],
        ),
        (
            [-1.23j, complex(1.2, np.nan), 1],
            ["-0.00-1.23j", " 1.20+ NaNj", " 1.00+0.00j"],
        ),
        (
            [-1.23j, complex(np.nan, -1.2), 1],
            ["-0.00-1.23j", "  NaN-1.20j", " 1.00+0.00j"],
        ),
    ],
)
@pytest.mark.parametrize("as_frame", [True, False])
def test_repr_with_complex_nans(data, output, as_frame):
    # GH#53762, GH#53841
    obj = Series(np.array(data))
    if as_frame:
        obj = obj.to_frame(name="val")
        reprs = [f"{i} {val}" for i, val in enumerate(output)]
        expected = f"{'val': >{len(reprs[0])}}\n" + "\n".join(reprs)
    else:
        reprs = [f"{i}   {val}" for i, val in enumerate(output)]
        expected = "\n".join(reprs) + "\ndtype: complex128"
    assert str(obj) == expected, f"\n{str(obj)}\n\n{expected}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_file_handling.py ===
import os

import numpy as np
import pytest

from pandas.compat import (
    PY311,
    is_ci_environment,
    is_platform_linux,
    is_platform_little_endian,
)
from pandas.errors import (
    ClosedFileError,
    PossibleDataLossError,
)

from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Series,
    _testing as tm,
    date_range,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
    tables,
)

from pandas.io import pytables
from pandas.io.pytables import Term

pytestmark = [pytest.mark.single_cpu]


@pytest.mark.parametrize("mode", ["r", "r+", "a", "w"])
def test_mode(setup_path, tmp_path, mode, using_infer_string):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    msg = r"[\S]* does not exist"
    path = tmp_path / setup_path

    # constructor
    if mode in ["r", "r+"]:
        with pytest.raises(OSError, match=msg):
            HDFStore(path, mode=mode)

    else:
        with HDFStore(path, mode=mode) as store:
            assert store._handle.mode == mode

    path = tmp_path / setup_path

    # context
    if mode in ["r", "r+"]:
        with pytest.raises(OSError, match=msg):
            with HDFStore(path, mode=mode) as store:
                pass
    else:
        with HDFStore(path, mode=mode) as store:
            assert store._handle.mode == mode

    path = tmp_path / setup_path

    # conv write
    if mode in ["r", "r+"]:
        with pytest.raises(OSError, match=msg):
            df.to_hdf(path, key="df", mode=mode)
        df.to_hdf(path, key="df", mode="w")
    else:
        df.to_hdf(path, key="df", mode=mode)

    # conv read
    if mode in ["w"]:
        msg = (
            "mode w is not allowed while performing a read. "
            r"Allowed modes are r, r\+ and a."
        )
        with pytest.raises(ValueError, match=msg):
            read_hdf(path, "df", mode=mode)
    else:
        result = read_hdf(path, "df", mode=mode)
        if using_infer_string:
            df.columns = df.columns.astype("str")
        tm.assert_frame_equal(result, df)


def test_default_mode(tmp_path, setup_path, using_infer_string):
    # read_hdf uses default mode
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="w")
    result = read_hdf(path, "df")
    expected = df.copy()
    if using_infer_string:
        expected.columns = expected.columns.astype("str")
    tm.assert_frame_equal(result, expected)


def test_reopen_handle(tmp_path, setup_path):
    path = tmp_path / setup_path

    store = HDFStore(path, mode="a")
    store["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )

    msg = (
        r"Re-opening the file \[[\S]*\] with mode \[a\] will delete the "
        "current file!"
    )
    # invalid mode change
    with pytest.raises(PossibleDataLossError, match=msg):
        store.open("w")

    store.close()
    assert not store.is_open

    # truncation ok here
    store.open("w")
    assert store.is_open
    assert len(store) == 0
    store.close()
    assert not store.is_open

    store = HDFStore(path, mode="a")
    store["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )

    # reopen as read
    store.open("r")
    assert store.is_open
    assert len(store) == 1
    assert store._mode == "r"
    store.close()
    assert not store.is_open

    # reopen as append
    store.open("a")
    assert store.is_open
    assert len(store) == 1
    assert store._mode == "a"
    store.close()
    assert not store.is_open

    # reopen as append (again)
    store.open("a")
    assert store.is_open
    assert len(store) == 1
    assert store._mode == "a"
    store.close()
    assert not store.is_open


def test_open_args(setup_path, using_infer_string):
    with tm.ensure_clean(setup_path) as path:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        # create an in memory store
        store = HDFStore(
            path, mode="a", driver="H5FD_CORE", driver_core_backing_store=0
        )
        store["df"] = df
        store.append("df2", df)

        expected = df.copy()
        if using_infer_string:
            expected.index = expected.index.astype("str")
            expected.columns = expected.columns.astype("str")

        tm.assert_frame_equal(store["df"], expected)
        tm.assert_frame_equal(store["df2"], expected)

        store.close()

    # the file should not have actually been written
    assert not os.path.exists(path)


def test_flush(setup_path):
    with ensure_clean_store(setup_path) as store:
        store["a"] = Series(range(5))
        store.flush()
        store.flush(fsync=True)


def test_complibs_default_settings(tmp_path, setup_path, using_infer_string):
    # GH15943
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # Set complevel and check if complib is automatically set to
    # default value
    tmpfile = tmp_path / setup_path
    df.to_hdf(tmpfile, key="df", complevel=9)
    result = read_hdf(tmpfile, "df")
    expected = df.copy()
    if using_infer_string:
        expected.index = expected.index.astype("str")
        expected.columns = expected.columns.astype("str")
    tm.assert_frame_equal(result, expected)

    with tables.open_file(tmpfile, mode="r") as h5file:
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            assert node.filters.complevel == 9
            assert node.filters.complib == "zlib"

    # Set complib and check to see if compression is disabled
    tmpfile = tmp_path / setup_path
    df.to_hdf(tmpfile, key="df", complib="zlib")
    result = read_hdf(tmpfile, "df")
    expected = df.copy()
    if using_infer_string:
        expected.index = expected.index.astype("str")
        expected.columns = expected.columns.astype("str")
    tm.assert_frame_equal(result, expected)

    with tables.open_file(tmpfile, mode="r") as h5file:
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            assert node.filters.complevel == 0
            assert node.filters.complib is None

    # Check if not setting complib or complevel results in no compression
    tmpfile = tmp_path / setup_path
    df.to_hdf(tmpfile, key="df")
    result = read_hdf(tmpfile, "df")
    expected = df.copy()
    if using_infer_string:
        expected.index = expected.index.astype("str")
        expected.columns = expected.columns.astype("str")
    tm.assert_frame_equal(result, expected)

    with tables.open_file(tmpfile, mode="r") as h5file:
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            assert node.filters.complevel == 0
            assert node.filters.complib is None


def test_complibs_default_settings_override(tmp_path, setup_path):
    # Check if file-defaults can be overridden on a per table basis
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    tmpfile = tmp_path / setup_path
    store = HDFStore(tmpfile)
    store.append("dfc", df, complevel=9, complib="blosc")
    store.append("df", df)
    store.close()

    with tables.open_file(tmpfile, mode="r") as h5file:
        for node in h5file.walk_nodes(where="/df", classname="Leaf"):
            assert node.filters.complevel == 0
            assert node.filters.complib is None
        for node in h5file.walk_nodes(where="/dfc", classname="Leaf"):
            assert node.filters.complevel == 9
            assert node.filters.complib == "blosc"


@pytest.mark.parametrize("lvl", range(10))
@pytest.mark.parametrize("lib", tables.filters.all_complibs)
@pytest.mark.filterwarnings("ignore:object name is not a valid")
@pytest.mark.skipif(
    not PY311 and is_ci_environment() and is_platform_linux(),
    reason="Segfaulting in a CI environment"
    # with xfail, would sometimes raise UnicodeDecodeError
    # invalid state byte
)
def test_complibs(tmp_path, lvl, lib, request):
    # GH14478
    if PY311 and is_platform_linux() and lib == "blosc2" and lvl != 0:
        request.applymarker(
            pytest.mark.xfail(reason=f"Fails for {lib} on Linux and PY > 3.11")
        )
    df = DataFrame(
        np.ones((30, 4)), columns=list("ABCD"), index=np.arange(30).astype(np.str_)
    )

    # Remove lzo if its not available on this platform
    if not tables.which_lib_version("lzo"):
        pytest.skip("lzo not available")
    # Remove bzip2 if its not available on this platform
    if not tables.which_lib_version("bzip2"):
        pytest.skip("bzip2 not available")

    tmpfile = tmp_path / f"{lvl}_{lib}.h5"
    gname = f"{lvl}_{lib}"

    # Write and read file to see if data is consistent
    df.to_hdf(tmpfile, key=gname, complib=lib, complevel=lvl)
    result = read_hdf(tmpfile, gname)
    tm.assert_frame_equal(result, df)

    # Open file and check metadata for correct amount of compression
    with tables.open_file(tmpfile, mode="r") as h5table:
        for node in h5table.walk_nodes(where="/" + gname, classname="Leaf"):
            assert node.filters.complevel == lvl
            if lvl == 0:
                assert node.filters.complib is None
            else:
                assert node.filters.complib == lib


@pytest.mark.skipif(
    not is_platform_little_endian(), reason="reason platform is not little endian"
)
def test_encoding(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({"A": "foo", "B": "bar"}, index=range(5))
        df.loc[2, "A"] = np.nan
        df.loc[3, "B"] = np.nan
        _maybe_remove(store, "df")
        store.append("df", df, encoding="ascii")
        tm.assert_frame_equal(store["df"], df)

        expected = df.reindex(columns=["A"])
        result = store.select("df", Term("columns=A", encoding="ascii"))
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "val",
    [
        [b"E\xc9, 17", b"", b"a", b"b", b"c"],
        [b"E\xc9, 17", b"a", b"b", b"c"],
        [b"EE, 17", b"", b"a", b"b", b"c"],
        [b"E\xc9, 17", b"\xf8\xfc", b"a", b"b", b"c"],
        [b"", b"a", b"b", b"c"],
        [b"\xf8\xfc", b"a", b"b", b"c"],
        [b"A\xf8\xfc", b"", b"a", b"b", b"c"],
        [np.nan, b"", b"b", b"c"],
        [b"A\xf8\xfc", np.nan, b"", b"b", b"c"],
    ],
)
@pytest.mark.parametrize("dtype", ["category", None])
def test_latin_encoding(tmp_path, setup_path, dtype, val):
    enc = "latin-1"
    nan_rep = ""
    key = "data"

    val = [x.decode(enc) if isinstance(x, bytes) else x for x in val]
    ser = Series(val, dtype=dtype)

    store = tmp_path / setup_path
    ser.to_hdf(store, key=key, format="table", encoding=enc, nan_rep=nan_rep)
    retr = read_hdf(store, key)

    # TODO:(3.0): once Categorical replace deprecation is enforced,
    #  we may be able to re-simplify the construction of s_nan
    if dtype == "category":
        if nan_rep in ser.cat.categories:
            s_nan = ser.cat.remove_categories([nan_rep])
        else:
            s_nan = ser
    else:
        s_nan = ser.replace(nan_rep, np.nan)

    tm.assert_series_equal(s_nan, retr)


def test_multiple_open_close(tmp_path, setup_path):
    # gh-4409: open & close multiple times

    path = tmp_path / setup_path

    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    df.to_hdf(path, key="df", mode="w", format="table")

    # single
    store = HDFStore(path)
    assert "CLOSED" not in store.info()
    assert store.is_open

    store.close()
    assert "CLOSED" in store.info()
    assert not store.is_open

    path = tmp_path / setup_path

    if pytables._table_file_open_policy_is_strict:
        # multiples
        store1 = HDFStore(path)
        msg = (
            r"The file [\S]* is already opened\.  Please close it before "
            r"reopening in write mode\."
        )
        with pytest.raises(ValueError, match=msg):
            HDFStore(path)

        store1.close()
    else:
        # multiples
        store1 = HDFStore(path)
        store2 = HDFStore(path)

        assert "CLOSED" not in store1.info()
        assert "CLOSED" not in store2.info()
        assert store1.is_open
        assert store2.is_open

        store1.close()
        assert "CLOSED" in store1.info()
        assert not store1.is_open
        assert "CLOSED" not in store2.info()
        assert store2.is_open

        store2.close()
        assert "CLOSED" in store1.info()
        assert "CLOSED" in store2.info()
        assert not store1.is_open
        assert not store2.is_open

        # nested close
        store = HDFStore(path, mode="w")
        store.append("df", df)

        store2 = HDFStore(path)
        store2.append("df2", df)
        store2.close()
        assert "CLOSED" in store2.info()
        assert not store2.is_open

        store.close()
        assert "CLOSED" in store.info()
        assert not store.is_open

        # double closing
        store = HDFStore(path, mode="w")
        store.append("df", df)

        store2 = HDFStore(path)
        store.close()
        assert "CLOSED" in store.info()
        assert not store.is_open

        store2.close()
        assert "CLOSED" in store2.info()
        assert not store2.is_open

    # ops on a closed store
    path = tmp_path / setup_path

    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    df.to_hdf(path, key="df", mode="w", format="table")

    store = HDFStore(path)
    store.close()

    msg = r"[\S]* file is not open!"
    with pytest.raises(ClosedFileError, match=msg):
        store.keys()

    with pytest.raises(ClosedFileError, match=msg):
        "df" in store

    with pytest.raises(ClosedFileError, match=msg):
        len(store)

    with pytest.raises(ClosedFileError, match=msg):
        store["df"]

    with pytest.raises(ClosedFileError, match=msg):
        store.select("df")

    with pytest.raises(ClosedFileError, match=msg):
        store.get("df")

    with pytest.raises(ClosedFileError, match=msg):
        store.append("df2", df)

    with pytest.raises(ClosedFileError, match=msg):
        store.put("df3", df)

    with pytest.raises(ClosedFileError, match=msg):
        store.get_storer("df2")

    with pytest.raises(ClosedFileError, match=msg):
        store.remove("df2")

    with pytest.raises(ClosedFileError, match=msg):
        store.select("df")

    msg = "'HDFStore' object has no attribute 'df'"
    with pytest.raises(AttributeError, match=msg):
        store.df


def test_fspath():
    with tm.ensure_clean("foo.h5") as path:
        with HDFStore(path) as store:
            assert os.fspath(store) == str(path)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_octave.py ===
from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
                        Tuple, Symbol, EulerGamma, GoldenRatio, Catalan,
                        Lambda, Mul, Pow, Mod, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.functions import (arg, atan2, bernoulli, beta, ceiling, chebyshevu,
                             chebyshevt, conjugate, DiracDelta, exp, expint,
                             factorial, floor, harmonic, Heaviside, im,
                             laguerre, LambertW, log, Max, Min, Piecewise,
                             polylog, re, RisingFactorial, sign, sinc, sqrt,
                             zeta, binomial, legendre, dirichlet_eta,
                             riemann_xi)
from sympy.functions import (sin, cos, tan, cot, sec, csc, asin, acos, acot,
                             atan, asec, acsc, sinh, cosh, tanh, coth, csch,
                             sech, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
                            HadamardProduct, SparseMatrix, HadamardPower)
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
                                            besselk, hankel1, hankel2, airyai,
                                            airybi, airyaiprime, airybiprime)
from sympy.functions.special.gamma_functions import (gamma, lowergamma,
                                                     uppergamma, loggamma,
                                                     polygamma)
from sympy.functions.special.error_functions import (Chi, Ci, erf, erfc, erfi,
                                                     erfcinv, erfinv, fresnelc,
                                                     fresnels, li, Shi, Si, Li,
                                                     erf2, Ei)
from sympy.printing.octave import octave_code, octave_code as mcode

x, y, z = symbols('x,y,z')


def test_Integer():
    assert mcode(Integer(67)) == "67"
    assert mcode(Integer(-1)) == "-1"


def test_Rational():
    assert mcode(Rational(3, 7)) == "3/7"
    assert mcode(Rational(18, 9)) == "2"
    assert mcode(Rational(3, -7)) == "-3/7"
    assert mcode(Rational(-3, -7)) == "3/7"
    assert mcode(x + Rational(3, 7)) == "x + 3/7"
    assert mcode(Rational(3, 7)*x) == "3*x/7"


def test_Relational():
    assert mcode(Eq(x, y)) == "x == y"
    assert mcode(Ne(x, y)) == "x != y"
    assert mcode(Le(x, y)) == "x <= y"
    assert mcode(Lt(x, y)) == "x < y"
    assert mcode(Gt(x, y)) == "x > y"
    assert mcode(Ge(x, y)) == "x >= y"


def test_Function():
    assert mcode(sin(x) ** cos(x)) == "sin(x).^cos(x)"
    assert mcode(sign(x)) == "sign(x)"
    assert mcode(exp(x)) == "exp(x)"
    assert mcode(log(x)) == "log(x)"
    assert mcode(factorial(x)) == "factorial(x)"
    assert mcode(floor(x)) == "floor(x)"
    assert mcode(atan2(y, x)) == "atan2(y, x)"
    assert mcode(beta(x, y)) == 'beta(x, y)'
    assert mcode(polylog(x, y)) == 'polylog(x, y)'
    assert mcode(harmonic(x)) == 'harmonic(x)'
    assert mcode(bernoulli(x)) == "bernoulli(x)"
    assert mcode(bernoulli(x, y)) == "bernoulli(x, y)"
    assert mcode(legendre(x, y)) == "legendre(x, y)"


def test_Function_change_name():
    assert mcode(abs(x)) == "abs(x)"
    assert mcode(ceiling(x)) == "ceil(x)"
    assert mcode(arg(x)) == "angle(x)"
    assert mcode(im(x)) == "imag(x)"
    assert mcode(re(x)) == "real(x)"
    assert mcode(conjugate(x)) == "conj(x)"
    assert mcode(chebyshevt(y, x)) == "chebyshevT(y, x)"
    assert mcode(chebyshevu(y, x)) == "chebyshevU(y, x)"
    assert mcode(laguerre(x, y)) == "laguerreL(x, y)"
    assert mcode(Chi(x)) == "coshint(x)"
    assert mcode(Shi(x)) ==  "sinhint(x)"
    assert mcode(Ci(x)) == "cosint(x)"
    assert mcode(Si(x)) ==  "sinint(x)"
    assert mcode(li(x)) ==  "logint(x)"
    assert mcode(loggamma(x)) ==  "gammaln(x)"
    assert mcode(polygamma(x, y)) == "psi(x, y)"
    assert mcode(RisingFactorial(x, y)) == "pochhammer(x, y)"
    assert mcode(DiracDelta(x)) == "dirac(x)"
    assert mcode(DiracDelta(x, 3)) == "dirac(3, x)"
    assert mcode(Heaviside(x)) == "heaviside(x, 1/2)"
    assert mcode(Heaviside(x, y)) == "heaviside(x, y)"
    assert mcode(binomial(x, y)) == "bincoeff(x, y)"
    assert mcode(Mod(x, y)) == "mod(x, y)"


def test_minmax():
    assert mcode(Max(x, y) + Min(x, y)) == "max(x, y) + min(x, y)"
    assert mcode(Max(x, y, z)) == "max(x, max(y, z))"
    assert mcode(Min(x, y, z)) == "min(x, min(y, z))"


def test_Pow():
    assert mcode(x**3) == "x.^3"
    assert mcode(x**(y**3)) == "x.^(y.^3)"
    assert mcode(x**Rational(2, 3)) == 'x.^(2/3)'
    g = implemented_function('g', Lambda(x, 2*x))
    assert mcode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2*x).^(-x + y.^x)./(x.^2 + y)"
    # For issue 14160
    assert mcode(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2*x./(y.*y)'


def test_basic_ops():
    assert mcode(x*y) == "x.*y"
    assert mcode(x + y) == "x + y"
    assert mcode(x - y) == "x - y"
    assert mcode(-x) == "-x"


def test_1_over_x_and_sqrt():
    # 1.0 and 0.5 would do something different in regular StrPrinter,
    # but these are exact in IEEE floating point so no different here.
    assert mcode(1/x) == '1./x'
    assert mcode(x**-1) == mcode(x**-1.0) == '1./x'
    assert mcode(1/sqrt(x)) == '1./sqrt(x)'
    assert mcode(x**-S.Half) == mcode(x**-0.5) == '1./sqrt(x)'
    assert mcode(sqrt(x)) == 'sqrt(x)'
    assert mcode(x**S.Half) == mcode(x**0.5) == 'sqrt(x)'
    assert mcode(1/pi) == '1/pi'
    assert mcode(pi**-1) == mcode(pi**-1.0) == '1/pi'
    assert mcode(pi**-0.5) == '1/sqrt(pi)'


def test_mix_number_mult_symbols():
    assert mcode(3*x) == "3*x"
    assert mcode(pi*x) == "pi*x"
    assert mcode(3/x) == "3./x"
    assert mcode(pi/x) == "pi./x"
    assert mcode(x/3) == "x/3"
    assert mcode(x/pi) == "x/pi"
    assert mcode(x*y) == "x.*y"
    assert mcode(3*x*y) == "3*x.*y"
    assert mcode(3*pi*x*y) == "3*pi*x.*y"
    assert mcode(x/y) == "x./y"
    assert mcode(3*x/y) == "3*x./y"
    assert mcode(x*y/z) == "x.*y./z"
    assert mcode(x/y*z) == "x.*z./y"
    assert mcode(1/x/y) == "1./(x.*y)"
    assert mcode(2*pi*x/y/z) == "2*pi*x./(y.*z)"
    assert mcode(3*pi/x) == "3*pi./x"
    assert mcode(S(3)/5) == "3/5"
    assert mcode(S(3)/5*x) == "3*x/5"
    assert mcode(x/y/z) == "x./(y.*z)"
    assert mcode((x+y)/z) == "(x + y)./z"
    assert mcode((x+y)/(z+x)) == "(x + y)./(x + z)"
    assert mcode((x+y)/EulerGamma) == "(x + y)/%s" % EulerGamma.evalf(17)
    assert mcode(x/3/pi) == "x/(3*pi)"
    assert mcode(S(3)/5*x*y/pi) == "3*x.*y/(5*pi)"


def test_mix_number_pow_symbols():
    assert mcode(pi**3) == 'pi^3'
    assert mcode(x**2) == 'x.^2'
    assert mcode(x**(pi**3)) == 'x.^(pi^3)'
    assert mcode(x**y) == 'x.^y'
    assert mcode(x**(y**z)) == 'x.^(y.^z)'
    assert mcode((x**y)**z) == '(x.^y).^z'


def test_imag():
    I = S('I')
    assert mcode(I) == "1i"
    assert mcode(5*I) == "5i"
    assert mcode((S(3)/2)*I) == "3*1i/2"
    assert mcode(3+4*I) == "3 + 4i"
    assert mcode(sqrt(3)*I) == "sqrt(3)*1i"


def test_constants():
    assert mcode(pi) == "pi"
    assert mcode(oo) == "inf"
    assert mcode(-oo) == "-inf"
    assert mcode(S.NegativeInfinity) == "-inf"
    assert mcode(S.NaN) == "NaN"
    assert mcode(S.Exp1) == "exp(1)"
    assert mcode(exp(1)) == "exp(1)"


def test_constants_other():
    assert mcode(2*GoldenRatio) == "2*(1+sqrt(5))/2"
    assert mcode(2*Catalan) == "2*%s" % Catalan.evalf(17)
    assert mcode(2*EulerGamma) == "2*%s" % EulerGamma.evalf(17)


def test_boolean():
    assert mcode(x & y) == "x & y"
    assert mcode(x | y) == "x | y"
    assert mcode(~x) == "~x"
    assert mcode(x & y & z) == "x & y & z"
    assert mcode(x | y | z) == "x | y | z"
    assert mcode((x & y) | z) == "z | x & y"
    assert mcode((x | y) & z) == "z & (x | y)"


def test_KroneckerDelta():
    from sympy.functions import KroneckerDelta
    assert mcode(KroneckerDelta(x, y)) == "double(x == y)"
    assert mcode(KroneckerDelta(x, y + 1)) == "double(x == (y + 1))"
    assert mcode(KroneckerDelta(2**x, y)) == "double((2.^x) == y)"


def test_Matrices():
    assert mcode(Matrix(1, 1, [10])) == "10"
    A = Matrix([[1, sin(x/2), abs(x)],
                [0, 1, pi],
                [0, exp(1), ceiling(x)]])
    expected = "[1 sin(x/2) abs(x); 0 1 pi; 0 exp(1) ceil(x)]"
    assert mcode(A) == expected
    # row and columns
    assert mcode(A[:,0]) == "[1; 0; 0]"
    assert mcode(A[0,:]) == "[1 sin(x/2) abs(x)]"
    # empty matrices
    assert mcode(Matrix(0, 0, [])) == '[]'
    assert mcode(Matrix(0, 3, [])) == 'zeros(0, 3)'
    # annoying to read but correct
    assert mcode(Matrix([[x, x - y, -y]])) == "[x x - y -y]"


def test_vector_entries_hadamard():
    # For a row or column, user might to use the other dimension
    A = Matrix([[1, sin(2/x), 3*pi/x/5]])
    assert mcode(A) == "[1 sin(2./x) 3*pi./(5*x)]"
    assert mcode(A.T) == "[1; sin(2./x); 3*pi./(5*x)]"


@XFAIL
def test_Matrices_entries_not_hadamard():
    # For Matrix with col >= 2, row >= 2, they need to be scalars
    # FIXME: is it worth worrying about this?  Its not wrong, just
    # leave it user's responsibility to put scalar data for x.
    A = Matrix([[1, sin(2/x), 3*pi/x/5], [1, 2, x*y]])
    expected = ("[1 sin(2/x) 3*pi/(5*x);\n"
                "1        2        x*y]") # <- we give x.*y
    assert mcode(A) == expected


def test_MatrixSymbol():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert mcode(A*B) == "A*B"
    assert mcode(B*A) == "B*A"
    assert mcode(2*A*B) == "2*A*B"
    assert mcode(B*2*A) == "2*B*A"
    assert mcode(A*(B + 3*Identity(n))) == "A*(3*eye(n) + B)"
    assert mcode(A**(x**2)) == "A^(x.^2)"
    assert mcode(A**3) == "A^3"
    assert mcode(A**S.Half) == "A^(1/2)"


def test_MatrixSolve():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    x = MatrixSymbol('x', n, 1)
    assert mcode(MatrixSolve(A, x)) == "A \\ x"

def test_special_matrices():
    assert mcode(6*Identity(3)) == "6*eye(3)"


def test_containers():
    assert mcode([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
        "{1, 2, 3, {4, 5, {6, 7}}, 8, {9, 10}, 11}"
    assert mcode((1, 2, (3, 4))) == "{1, 2, {3, 4}}"
    assert mcode([1]) == "{1}"
    assert mcode((1,)) == "{1}"
    assert mcode(Tuple(*[1, 2, 3])) == "{1, 2, 3}"
    assert mcode((1, x*y, (3, x**2))) == "{1, x.*y, {3, x.^2}}"
    # scalar, matrix, empty matrix and empty list
    assert mcode((1, eye(3), Matrix(0, 0, []), [])) == "{1, [1 0 0; 0 1 0; 0 0 1], [], {}}"


def test_octave_noninline():
    source = mcode((x+y)/Catalan, assign_to='me', inline=False)
    expected = (
        "Catalan = %s;\n"
        "me = (x + y)/Catalan;"
    ) % Catalan.evalf(17)
    assert source == expected


def test_octave_piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    assert mcode(expr) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))"
    assert mcode(expr, assign_to="r") == (
        "r = ((x < 1).*(x) + (~(x < 1)).*(x.^2));")
    assert mcode(expr, assign_to="r", inline=False) == (
        "if (x < 1)\n"
        "  r = x;\n"
        "else\n"
        "  r = x.^2;\n"
        "end")
    expr = Piecewise((x**2, x < 1), (x**3, x < 2), (x**4, x < 3), (x**5, True))
    expected = ("((x < 1).*(x.^2) + (~(x < 1)).*( ...\n"
                "(x < 2).*(x.^3) + (~(x < 2)).*( ...\n"
                "(x < 3).*(x.^4) + (~(x < 3)).*(x.^5))))")
    assert mcode(expr) == expected
    assert mcode(expr, assign_to="r") == "r = " + expected + ";"
    assert mcode(expr, assign_to="r", inline=False) == (
        "if (x < 1)\n"
        "  r = x.^2;\n"
        "elseif (x < 2)\n"
        "  r = x.^3;\n"
        "elseif (x < 3)\n"
        "  r = x.^4;\n"
        "else\n"
        "  r = x.^5;\n"
        "end")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: mcode(expr))


def test_octave_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x**2, True))
    assert mcode(2*pw) == "2*((x < 1).*(x) + (~(x < 1)).*(x.^2))"
    assert mcode(pw/x) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))./x"
    assert mcode(pw/(x*y)) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))./(x.*y)"
    assert mcode(pw/3) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))/3"


def test_octave_matrix_assign_to():
    A = Matrix([[1, 2, 3]])
    assert mcode(A, assign_to='a') == "a = [1 2 3];"
    A = Matrix([[1, 2], [3, 4]])
    assert mcode(A, assign_to='A') == "A = [1 2; 3 4];"


def test_octave_matrix_assign_to_more():
    # assigning to Symbol or MatrixSymbol requires lhs/rhs match
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert mcode(A, assign_to=B) == "B = [1 2 3];"
    raises(ValueError, lambda: mcode(A, assign_to=x))
    raises(ValueError, lambda: mcode(A, assign_to=C))


def test_octave_matrix_1x1():
    A = Matrix([[3]])
    B = MatrixSymbol('B', 1, 1)
    C = MatrixSymbol('C', 1, 2)
    assert mcode(A, assign_to=B) == "B = 3;"
    # FIXME?
    #assert mcode(A, assign_to=x) == "x = 3;"
    raises(ValueError, lambda: mcode(A, assign_to=C))


def test_octave_matrix_elements():
    A = Matrix([[x, 2, x*y]])
    assert mcode(A[0, 0]**2 + A[0, 1] + A[0, 2]) == "x.^2 + x.*y + 2"
    A = MatrixSymbol('AA', 1, 3)
    assert mcode(A) == "AA"
    assert mcode(A[0, 0]**2 + sin(A[0,1]) + A[0,2]) == \
           "sin(AA(1, 2)) + AA(1, 1).^2 + AA(1, 3)"
    assert mcode(sum(A)) == "AA(1, 1) + AA(1, 2) + AA(1, 3)"


def test_octave_boolean():
    assert mcode(True) == "true"
    assert mcode(S.true) == "true"
    assert mcode(False) == "false"
    assert mcode(S.false) == "false"


def test_octave_not_supported():
    with raises(NotImplementedError):
        mcode(S.ComplexInfinity)
    f = Function('f')
    assert mcode(f(x).diff(x), strict=False) == (
        "% Not supported in Octave:\n"
        "% Derivative\n"
        "Derivative(f(x), x)"
    )


def test_octave_not_supported_not_on_whitelist():
    from sympy.functions.special.polynomials import assoc_laguerre
    with raises(NotImplementedError):
        mcode(assoc_laguerre(x, y, z))


def test_octave_expint():
    assert mcode(expint(1, x)) == "expint(x)"
    with raises(NotImplementedError):
        mcode(expint(2, x))
    assert mcode(expint(y, x), strict=False) == (
        "% Not supported in Octave:\n"
        "% expint\n"
        "expint(y, x)"
    )


def test_trick_indent_with_end_else_words():
    # words starting with "end" or "else" do not confuse the indenter
    t1 = S('endless')
    t2 = S('elsewhere')
    pw = Piecewise((t1, x < 0), (t2, x <= 1), (1, True))
    assert mcode(pw, inline=False) == (
        "if (x < 0)\n"
        "  endless\n"
        "elseif (x <= 1)\n"
        "  elsewhere\n"
        "else\n"
        "  1\n"
        "end")


def test_hadamard():
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    C = HadamardProduct(A, B)
    n = Symbol('n')
    assert mcode(C) == "A.*B"
    assert mcode(C*v) == "(A.*B)*v"
    assert mcode(h*C*v) == "h*(A.*B)*v"
    assert mcode(C*A) == "(A.*B)*A"
    # mixing Hadamard and scalar strange b/c we vectorize scalars
    assert mcode(C*x*y) == "(x.*y)*(A.*B)"

    # Testing HadamardPower:
    assert mcode(HadamardPower(A, n)) == "A.**n"
    assert mcode(HadamardPower(A, 1+n)) == "A.**(n + 1)"
    assert mcode(HadamardPower(A*B.T, 1+n)) == "(A*B.T).**(n + 1)"


def test_sparse():
    M = SparseMatrix(5, 6, {})
    M[2, 2] = 10
    M[1, 2] = 20
    M[1, 3] = 22
    M[0, 3] = 30
    M[3, 0] = x*y
    assert mcode(M) == (
        "sparse([4 2 3 1 2], [1 3 3 4 4], [x.*y 20 10 30 22], 5, 6)"
    )


def test_sinc():
    assert mcode(sinc(x)) == 'sinc(x/pi)'
    assert mcode(sinc(x + 3)) == 'sinc((x + 3)/pi)'
    assert mcode(sinc(pi*(x + 3))) == 'sinc(x + 3)'


def test_trigfun():
    for f in (sin, cos, tan, cot, sec, csc, asin, acos, acot, atan, asec, acsc,
              sinh, cosh, tanh, coth, csch, sech, asinh, acosh, atanh, acoth,
              asech, acsch):
        assert octave_code(f(x) == f.__name__ + '(x)')


def test_specfun():
    n = Symbol('n')
    for f in [besselj, bessely, besseli, besselk]:
        assert octave_code(f(n, x)) == f.__name__ + '(n, x)'
    for f in (erfc, erfi, erf, erfinv, erfcinv, fresnelc, fresnels, gamma):
        assert octave_code(f(x)) == f.__name__ + '(x)'
    assert octave_code(hankel1(n, x)) == 'besselh(n, 1, x)'
    assert octave_code(hankel2(n, x)) == 'besselh(n, 2, x)'
    assert octave_code(airyai(x)) == 'airy(0, x)'
    assert octave_code(airyaiprime(x)) == 'airy(1, x)'
    assert octave_code(airybi(x)) == 'airy(2, x)'
    assert octave_code(airybiprime(x)) == 'airy(3, x)'
    assert octave_code(uppergamma(n, x)) == '(gammainc(x, n, \'upper\').*gamma(n))'
    assert octave_code(lowergamma(n, x)) == '(gammainc(x, n).*gamma(n))'
    assert octave_code(z**lowergamma(n, x)) == 'z.^(gammainc(x, n).*gamma(n))'
    assert octave_code(jn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*besselj(n + 1/2, x)/2'
    assert octave_code(yn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*bessely(n + 1/2, x)/2'
    assert octave_code(LambertW(x)) == 'lambertw(x)'
    assert octave_code(LambertW(x, n)) == 'lambertw(n, x)'

    # Automatic rewrite
    assert octave_code(Ei(x)) == '(logint(exp(x)))'
    assert octave_code(dirichlet_eta(x)) == '(((x == 1).*(log(2)) + (~(x == 1)).*((1 - 2.^(1 - x)).*zeta(x))))'
    assert octave_code(riemann_xi(x)) == '(pi.^(-x/2).*x.*(x - 1).*gamma(x/2).*zeta(x)/2)'


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert mcode(A[0, 0]) == "A(1, 1)"
    assert mcode(3 * A[0, 0]) == "3*A(1, 1)"

    F = C[0, 0].subs(C, A - B)
    assert mcode(F) == "(A - B)(1, 1)"


def test_zeta_printing_issue_14820():
    assert octave_code(zeta(x)) == 'zeta(x)'
    with raises(NotImplementedError):
        octave_code(zeta(x, y))


def test_automatic_rewrite():
    assert octave_code(Li(x)) == '(logint(x) - logint(2))'
    assert octave_code(erf2(x, y)) == '(-erf(x) + erf(y))'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_arithmetics.py ===
import operator

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


@pytest.fixture(params=["integer", "block"])
def kind(request):
    """kind kwarg to pass to SparseArray"""
    return request.param


@pytest.fixture(params=[True, False])
def mix(request):
    """
    Fixture returning True or False, determining whether to operate
    op(sparse, dense) instead of op(sparse, sparse)
    """
    return request.param


class TestSparseArrayArithmetics:
    def _assert(self, a, b):
        # We have to use tm.assert_sp_array_equal. See GH #45126
        tm.assert_numpy_array_equal(a, b)

    def _check_numeric_ops(self, a, b, a_dense, b_dense, mix: bool, op):
        # Check that arithmetic behavior matches non-Sparse Series arithmetic

        if isinstance(a_dense, np.ndarray):
            expected = op(pd.Series(a_dense), b_dense).values
        elif isinstance(b_dense, np.ndarray):
            expected = op(a_dense, pd.Series(b_dense)).values
        else:
            raise NotImplementedError

        with np.errstate(invalid="ignore", divide="ignore"):
            if mix:
                result = op(a, b_dense).to_dense()
            else:
                result = op(a, b).to_dense()

        self._assert(result, expected)

    def _check_bool_result(self, res):
        assert isinstance(res, SparseArray)
        assert isinstance(res.dtype, SparseDtype)
        assert res.dtype.subtype == np.bool_
        assert isinstance(res.fill_value, bool)

    def _check_comparison_ops(self, a, b, a_dense, b_dense):
        with np.errstate(invalid="ignore"):
            # Unfortunately, trying to wrap the computation of each expected
            # value is with np.errstate() is too tedious.
            #
            # sparse & sparse
            self._check_bool_result(a == b)
            self._assert((a == b).to_dense(), a_dense == b_dense)

            self._check_bool_result(a != b)
            self._assert((a != b).to_dense(), a_dense != b_dense)

            self._check_bool_result(a >= b)
            self._assert((a >= b).to_dense(), a_dense >= b_dense)

            self._check_bool_result(a <= b)
            self._assert((a <= b).to_dense(), a_dense <= b_dense)

            self._check_bool_result(a > b)
            self._assert((a > b).to_dense(), a_dense > b_dense)

            self._check_bool_result(a < b)
            self._assert((a < b).to_dense(), a_dense < b_dense)

            # sparse & dense
            self._check_bool_result(a == b_dense)
            self._assert((a == b_dense).to_dense(), a_dense == b_dense)

            self._check_bool_result(a != b_dense)
            self._assert((a != b_dense).to_dense(), a_dense != b_dense)

            self._check_bool_result(a >= b_dense)
            self._assert((a >= b_dense).to_dense(), a_dense >= b_dense)

            self._check_bool_result(a <= b_dense)
            self._assert((a <= b_dense).to_dense(), a_dense <= b_dense)

            self._check_bool_result(a > b_dense)
            self._assert((a > b_dense).to_dense(), a_dense > b_dense)

            self._check_bool_result(a < b_dense)
            self._assert((a < b_dense).to_dense(), a_dense < b_dense)

    def _check_logical_ops(self, a, b, a_dense, b_dense):
        # sparse & sparse
        self._check_bool_result(a & b)
        self._assert((a & b).to_dense(), a_dense & b_dense)

        self._check_bool_result(a | b)
        self._assert((a | b).to_dense(), a_dense | b_dense)
        # sparse & dense
        self._check_bool_result(a & b_dense)
        self._assert((a & b_dense).to_dense(), a_dense & b_dense)

        self._check_bool_result(a | b_dense)
        self._assert((a | b_dense).to_dense(), a_dense | b_dense)

    @pytest.mark.parametrize("scalar", [0, 1, 3])
    @pytest.mark.parametrize("fill_value", [None, 0, 2])
    def test_float_scalar(
        self, kind, mix, all_arithmetic_functions, fill_value, scalar, request
    ):
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        a = SparseArray(values, kind=kind, fill_value=fill_value)
        self._check_numeric_ops(a, scalar, values, scalar, mix, op)

    def test_float_scalar_comparison(self, kind):
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])

        a = SparseArray(values, kind=kind)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

        a = SparseArray(values, kind=kind, fill_value=0)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

        a = SparseArray(values, kind=kind, fill_value=2)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

    def test_float_same_index_without_nans(self, kind, mix, all_arithmetic_functions):
        # when sp_index are the same
        op = all_arithmetic_functions

        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_with_nans(
        self, kind, mix, all_arithmetic_functions, request
    ):
        # when sp_index are the same
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_comparison(self, kind):
        # when sp_index are the same
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_float_array(self, kind, mix, all_arithmetic_functions):
        op = all_arithmetic_functions

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_different_kind(self, mix, all_arithmetic_functions):
        op = all_arithmetic_functions

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind="integer")
        b = SparseArray(rvalues, kind="block")
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block")
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block", fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind="integer", fill_value=1)
        b = SparseArray(rvalues, kind="block", fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_comparison(self, kind):
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_int_array(self, kind, mix, all_arithmetic_functions):
        op = all_arithmetic_functions

        # have to specify dtype explicitly until fixing GH 667
        dtype = np.int64

        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)

        a = SparseArray(values, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, fill_value=0, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, fill_value=1, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype, fill_value=1)
        b = SparseArray(rvalues, fill_value=2, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_int_array_comparison(self, kind):
        dtype = "int64"
        # int32 NI ATM

        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)

        a = SparseArray(values, dtype=dtype, kind=kind)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=1)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    @pytest.mark.parametrize("fill_value", [True, False, np.nan])
    def test_bool_same_index(self, kind, fill_value):
        # GH 14000
        # when sp_index are the same
        values = np.array([True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, True, True], dtype=np.bool_)

        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    @pytest.mark.parametrize("fill_value", [True, False, np.nan])
    def test_bool_array_logical(self, kind, fill_value):
        # GH 14000
        # when sp_index are the same
        values = np.array([True, False, True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, False, True, False, True], dtype=np.bool_)

        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    def test_mixed_array_float_int(self, kind, mix, all_arithmetic_functions, request):
        op = all_arithmetic_functions
        rdtype = "int64"
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_mixed_array_comparison(self, kind):
        rdtype = "int64"
        # int32 NI ATM

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)

        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_xor(self):
        s = SparseArray([True, True, False, False])
        t = SparseArray([True, False, True, False])
        result = s ^ t
        sp_index = pd.core.arrays.sparse.IntIndex(4, np.array([0, 1, 2], dtype="int32"))
        expected = SparseArray([False, True, True], sparse_index=sp_index)
        tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("op", [operator.eq, operator.add])
def test_with_list(op):
    arr = SparseArray([0, 1], fill_value=0)
    result = op(arr, [0, 1])
    expected = op(arr, SparseArray([0, 1]))
    tm.assert_sp_array_equal(result, expected)


def test_with_dataframe():
    # GH#27910
    arr = SparseArray([0, 1], fill_value=0)
    df = pd.DataFrame([[1, 2], [3, 4]])
    result = arr.__add__(df)
    assert result is NotImplemented


def test_with_zerodim_ndarray():
    # GH#27910
    arr = SparseArray([0, 1], fill_value=0)

    result = arr * np.array(2)
    expected = arr * 2
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.abs, np.exp])
@pytest.mark.parametrize(
    "arr", [SparseArray([0, 0, -1, 1]), SparseArray([None, None, -1, 1])]
)
def test_ufuncs(ufunc, arr):
    result = ufunc(arr)
    fill_value = ufunc(arr.fill_value)
    expected = SparseArray(ufunc(np.asarray(arr)), fill_value=fill_value)
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        (SparseArray([0, 0, 0]), np.array([0, 1, 2])),
        (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])),
        (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])),
        (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])),
        (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])),
    ],
)
@pytest.mark.parametrize("ufunc", [np.add, np.greater])
def test_binary_ufuncs(ufunc, a, b):
    # can't say anything about fill value here.
    result = ufunc(a, b)
    expected = ufunc(np.asarray(a), np.asarray(b))
    assert isinstance(result, SparseArray)
    tm.assert_numpy_array_equal(np.asarray(result), expected)


def test_ndarray_inplace():
    sparray = SparseArray([0, 2, 0, 0])
    ndarray = np.array([0, 1, 2, 3])
    ndarray += sparray
    expected = np.array([0, 3, 2, 3])
    tm.assert_numpy_array_equal(ndarray, expected)


def test_sparray_inplace():
    sparray = SparseArray([0, 2, 0, 0])
    ndarray = np.array([0, 1, 2, 3])
    sparray += ndarray
    expected = SparseArray([0, 3, 2, 3], fill_value=0)
    tm.assert_sp_array_equal(sparray, expected)


@pytest.mark.parametrize("cons", [list, np.array, SparseArray])
def test_mismatched_length_cmp_op(cons):
    left = SparseArray([True, True])
    right = cons([True, True, True])
    with pytest.raises(ValueError, match="operands have mismatched length"):
        left & right


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "floordiv", "pow"])
@pytest.mark.parametrize("fill_value", [np.nan, 3])
def test_binary_operators(op, fill_value):
    op = getattr(operator, op)
    data1 = np.random.default_rng(2).standard_normal(20)
    data2 = np.random.default_rng(2).standard_normal(20)

    data1[::2] = fill_value
    data2[::3] = fill_value

    first = SparseArray(data1, fill_value=fill_value)
    second = SparseArray(data2, fill_value=fill_value)

    with np.errstate(all="ignore"):
        res = op(first, second)
        exp = SparseArray(
            op(first.to_dense(), second.to_dense()), fill_value=first.fill_value
        )
        assert isinstance(res, SparseArray)
        tm.assert_almost_equal(res.to_dense(), exp.to_dense())

        res2 = op(first, second.to_dense())
        assert isinstance(res2, SparseArray)
        tm.assert_sp_array_equal(res, res2)

        res3 = op(first.to_dense(), second)
        assert isinstance(res3, SparseArray)
        tm.assert_sp_array_equal(res, res3)

        res4 = op(first, 4)
        assert isinstance(res4, SparseArray)

        # Ignore this if the actual op raises (e.g. pow).
        try:
            exp = op(first.to_dense(), 4)
            exp_fv = op(first.fill_value, 4)
        except ValueError:
            pass
        else:
            tm.assert_almost_equal(res4.fill_value, exp_fv)
            tm.assert_almost_equal(res4.to_dense(), exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_common.py ===
"""
Collection of tests asserting things that should be true for
any index subclass except for MultiIndex. Makes use of the `index_flat`
fixture defined in pandas/conftest.py.
"""
from copy import (
    copy,
    deepcopy,
)
import re

import numpy as np
import pytest

from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_numeric_dtype,
)

import pandas as pd
from pandas import (
    CategoricalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
)
import pandas._testing as tm


class TestCommon:
    @pytest.mark.parametrize("name", [None, "new_name"])
    def test_to_frame(self, name, index_flat, using_copy_on_write):
        # see GH#15230, GH#22580
        idx = index_flat

        if name:
            idx_name = name
        else:
            idx_name = idx.name or 0

        df = idx.to_frame(name=idx_name)

        assert df.index is idx
        assert len(df.columns) == 1
        assert df.columns[0] == idx_name
        if not using_copy_on_write:
            assert df[idx_name].values is not idx.values

        df = idx.to_frame(index=False, name=idx_name)
        assert df.index is not idx

    def test_droplevel(self, index_flat):
        # GH 21115
        # MultiIndex is tested separately in test_multi.py
        index = index_flat

        assert index.droplevel([]).equals(index)

        for level in [index.name, [index.name]]:
            if isinstance(index.name, tuple) and level is index.name:
                # GH 21121 : droplevel with tuple name
                continue
            msg = (
                "Cannot remove 1 levels from an index with 1 levels: at least one "
                "level must be left."
            )
            with pytest.raises(ValueError, match=msg):
                index.droplevel(level)

        for level in "wrong", ["wrong"]:
            with pytest.raises(
                KeyError,
                match=r"'Requested level \(wrong\) does not match index name \(None\)'",
            ):
                index.droplevel(level)

    def test_constructor_non_hashable_name(self, index_flat):
        # GH 20527
        index = index_flat

        message = "Index.name must be a hashable type"
        renamed = [["1"]]

        # With .rename()
        with pytest.raises(TypeError, match=message):
            index.rename(name=renamed)

        # With .set_names()
        with pytest.raises(TypeError, match=message):
            index.set_names(names=renamed)

    def test_constructor_unwraps_index(self, index_flat):
        a = index_flat
        # Passing dtype is necessary for Index([True, False], dtype=object)
        #  case.
        b = type(a)(a, dtype=a.dtype)
        tm.assert_equal(a._data, b._data)

    def test_to_flat_index(self, index_flat):
        # 22866
        index = index_flat

        result = index.to_flat_index()
        tm.assert_index_equal(result, index)

    def test_set_name_methods(self, index_flat):
        # MultiIndex tested separately
        index = index_flat
        new_name = "This is the new name for this index"

        original_name = index.name
        new_ind = index.set_names([new_name])
        assert new_ind.name == new_name
        assert index.name == original_name
        res = index.rename(new_name, inplace=True)

        # should return None
        assert res is None
        assert index.name == new_name
        assert index.names == [new_name]
        with pytest.raises(ValueError, match="Level must be None"):
            index.set_names("a", level=0)

        # rename in place just leaves tuples and other containers alone
        name = ("A", "B")
        index.rename(name, inplace=True)
        assert index.name == name
        assert index.names == [name]

    @pytest.mark.xfail
    def test_set_names_single_label_no_level(self, index_flat):
        with pytest.raises(TypeError, match="list-like"):
            # should still fail even if it would be the right length
            index_flat.set_names("a")

    def test_copy_and_deepcopy(self, index_flat):
        index = index_flat

        for func in (copy, deepcopy):
            idx_copy = func(index)
            assert idx_copy is not index
            assert idx_copy.equals(index)

        new_copy = index.copy(deep=True, name="banana")
        assert new_copy.name == "banana"

    @pytest.mark.filterwarnings(r"ignore:Dtype inference:FutureWarning")
    def test_copy_name(self, index_flat):
        # GH#12309: Check that the "name" argument
        # passed at initialization is honored.
        index = index_flat

        first = type(index)(index, copy=True, name="mario")
        second = type(first)(first, copy=False)

        # Even though "copy=False", we want a new object.
        assert first is not second
        tm.assert_index_equal(first, second)

        # Not using tm.assert_index_equal() since names differ.
        assert index.equals(first)

        assert first.name == "mario"
        assert second.name == "mario"

        # TODO: belongs in series arithmetic tests?
        s1 = pd.Series(2, index=first)
        s2 = pd.Series(3, index=second[:-1])
        # See GH#13365
        s3 = s1 * s2
        assert s3.index.name == "mario"

    def test_copy_name2(self, index_flat):
        # GH#35592
        index = index_flat

        assert index.copy(name="mario").name == "mario"

        with pytest.raises(ValueError, match="Length of new names must be 1, got 2"):
            index.copy(name=["mario", "luigi"])

        msg = f"{type(index).__name__}.name must be a hashable type"
        with pytest.raises(TypeError, match=msg):
            index.copy(name=[["mario"]])

    def test_unique_level(self, index_flat):
        # don't test a MultiIndex here (as its tested separated)
        index = index_flat

        # GH 17896
        expected = index.drop_duplicates()
        for level in [0, index.name, None]:
            result = index.unique(level=level)
            tm.assert_index_equal(result, expected)

        msg = "Too many levels: Index has only 1 level, not 4"
        with pytest.raises(IndexError, match=msg):
            index.unique(level=3)

        msg = (
            rf"Requested level \(wrong\) does not match index name "
            rf"\({re.escape(index.name.__repr__())}\)"
        )
        with pytest.raises(KeyError, match=msg):
            index.unique(level="wrong")

    def test_unique(self, index_flat):
        # MultiIndex tested separately
        index = index_flat
        if not len(index):
            pytest.skip("Skip check for empty Index and MultiIndex")

        idx = index[[0] * 5]
        idx_unique = index[[0]]

        # We test against `idx_unique`, so first we make sure it's unique
        # and doesn't contain nans.
        assert idx_unique.is_unique is True
        try:
            assert idx_unique.hasnans is False
        except NotImplementedError:
            pass

        result = idx.unique()
        tm.assert_index_equal(result, idx_unique)

        # nans:
        if not index._can_hold_na:
            pytest.skip("Skip na-check if index cannot hold na")

        vals = index._values[[0] * 5]
        vals[0] = np.nan

        vals_unique = vals[:2]
        idx_nan = index._shallow_copy(vals)
        idx_unique_nan = index._shallow_copy(vals_unique)
        assert idx_unique_nan.is_unique is True

        assert idx_nan.dtype == index.dtype
        assert idx_unique_nan.dtype == index.dtype

        expected = idx_unique_nan
        for pos, i in enumerate([idx_nan, idx_unique_nan]):
            result = i.unique()
            tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Period with BDay freq:FutureWarning")
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_searchsorted_monotonic(self, index_flat, request):
        # GH17271
        index = index_flat
        # not implemented for tuple searches in MultiIndex
        # or Intervals searches in IntervalIndex
        if isinstance(index, pd.IntervalIndex):
            mark = pytest.mark.xfail(
                reason="IntervalIndex.searchsorted does not support Interval arg",
                raises=NotImplementedError,
            )
            request.applymarker(mark)

        # nothing to test if the index is empty
        if index.empty:
            pytest.skip("Skip check for empty Index")
        value = index[0]

        # determine the expected results (handle dupes for 'right')
        expected_left, expected_right = 0, (index == value).argmin()
        if expected_right == 0:
            # all values are the same, expected_right should be length
            expected_right = len(index)

        # test _searchsorted_monotonic in all cases
        # test searchsorted only for increasing
        if index.is_monotonic_increasing:
            ssm_left = index._searchsorted_monotonic(value, side="left")
            assert expected_left == ssm_left

            ssm_right = index._searchsorted_monotonic(value, side="right")
            assert expected_right == ssm_right

            ss_left = index.searchsorted(value, side="left")
            assert expected_left == ss_left

            ss_right = index.searchsorted(value, side="right")
            assert expected_right == ss_right

        elif index.is_monotonic_decreasing:
            ssm_left = index._searchsorted_monotonic(value, side="left")
            assert expected_left == ssm_left

            ssm_right = index._searchsorted_monotonic(value, side="right")
            assert expected_right == ssm_right
        else:
            # non-monotonic should raise.
            msg = "index must be monotonic increasing or decreasing"
            with pytest.raises(ValueError, match=msg):
                index._searchsorted_monotonic(value, side="left")

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_drop_duplicates(self, index_flat, keep):
        # MultiIndex is tested separately
        index = index_flat
        if isinstance(index, RangeIndex):
            pytest.skip(
                "RangeIndex is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )
        if len(index) == 0:
            pytest.skip(
                "empty index is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )

        # make unique index
        holder = type(index)
        unique_values = list(set(index))
        dtype = index.dtype if is_numeric_dtype(index) else None
        unique_idx = holder(unique_values, dtype=dtype)

        # make duplicated index
        n = len(unique_idx)
        duplicated_selection = np.random.default_rng(2).choice(n, int(n * 1.5))
        idx = holder(unique_idx.values[duplicated_selection])

        # Series.duplicated is tested separately
        expected_duplicated = (
            pd.Series(duplicated_selection).duplicated(keep=keep).values
        )
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected_duplicated)

        # Series.drop_duplicates is tested separately
        expected_dropped = holder(pd.Series(idx).drop_duplicates(keep=keep))
        tm.assert_index_equal(idx.drop_duplicates(keep=keep), expected_dropped)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_drop_duplicates_no_duplicates(self, index_flat):
        # MultiIndex is tested separately
        index = index_flat

        # make unique index
        if isinstance(index, RangeIndex):
            # RangeIndex cannot have duplicates
            unique_idx = index
        else:
            holder = type(index)
            unique_values = list(set(index))
            dtype = index.dtype if is_numeric_dtype(index) else None
            unique_idx = holder(unique_values, dtype=dtype)

        # check on unique index
        expected_duplicated = np.array([False] * len(unique_idx), dtype="bool")
        tm.assert_numpy_array_equal(unique_idx.duplicated(), expected_duplicated)
        result_dropped = unique_idx.drop_duplicates()
        tm.assert_index_equal(result_dropped, unique_idx)
        # validate shallow copy
        assert result_dropped is not unique_idx

    def test_drop_duplicates_inplace(self, index):
        msg = r"drop_duplicates\(\) got an unexpected keyword argument"
        with pytest.raises(TypeError, match=msg):
            index.drop_duplicates(inplace=True)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_has_duplicates(self, index_flat):
        # MultiIndex tested separately in:
        #   tests/indexes/multi/test_unique_and_duplicates.
        index = index_flat
        holder = type(index)
        if not len(index) or isinstance(index, RangeIndex):
            # MultiIndex tested separately in:
            #   tests/indexes/multi/test_unique_and_duplicates.
            # RangeIndex is unique by definition.
            pytest.skip("Skip check for empty Index, MultiIndex, and RangeIndex")

        idx = holder([index[0]] * 5)
        assert idx.is_unique is False
        assert idx.has_duplicates is True

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "uint64", "float64", "category", "datetime64[ns]", "timedelta64[ns]"],
    )
    def test_astype_preserves_name(self, index, dtype):
        # https://github.com/pandas-dev/pandas/issues/32013
        if isinstance(index, MultiIndex):
            index.names = ["idx" + str(i) for i in range(index.nlevels)]
        else:
            index.name = "idx"

        warn = None
        if index.dtype.kind == "c" and dtype in ["float64", "int64", "uint64"]:
            # imaginary components discarded
            if np_version_gte1p25:
                warn = np.exceptions.ComplexWarning
            else:
                warn = np.ComplexWarning

        is_pyarrow_str = str(index.dtype) == "string[pyarrow]" and dtype == "category"
        try:
            # Some of these conversions cannot succeed so we use a try / except
            with tm.assert_produces_warning(
                warn,
                raise_on_extra_warnings=is_pyarrow_str,
                check_stacklevel=False,
            ):
                result = index.astype(dtype)
        except (ValueError, TypeError, NotImplementedError, SystemError):
            return

        if isinstance(index, MultiIndex):
            assert result.names == index.names
        else:
            assert result.name == index.name

    def test_hasnans_isnans(self, index_flat):
        # GH#11343, added tests for hasnans / isnans
        index = index_flat

        # cases in indices doesn't include NaN
        idx = index.copy(deep=True)
        expected = np.array([False] * len(idx), dtype=bool)
        tm.assert_numpy_array_equal(idx._isnan, expected)
        assert idx.hasnans is False

        idx = index.copy(deep=True)
        values = idx._values

        if len(index) == 0:
            return
        elif is_integer_dtype(index.dtype):
            return
        elif index.dtype == bool:
            # values[1] = np.nan below casts to True!
            return

        values[1] = np.nan

        idx = type(index)(values)

        expected = np.array([False] * len(idx), dtype=bool)
        expected[1] = True
        tm.assert_numpy_array_equal(idx._isnan, expected)
        assert idx.hasnans is True


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("na_position", [None, "middle"])
def test_sort_values_invalid_na_position(index_with_missing, na_position):
    with pytest.raises(ValueError, match=f"invalid na_position: {na_position}"):
        index_with_missing.sort_values(na_position=na_position)


@pytest.mark.fails_arm_wheels
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_sort_values_with_missing(index_with_missing, na_position, request):
    # GH 35584. Test that sort_values works with missing values,
    # sort non-missing and place missing according to na_position

    if isinstance(index_with_missing, CategoricalIndex):
        request.applymarker(
            pytest.mark.xfail(
                reason="missing value sorting order not well-defined", strict=False
            )
        )

    missing_count = np.sum(index_with_missing.isna())
    not_na_vals = index_with_missing[index_with_missing.notna()].values
    sorted_values = np.sort(not_na_vals)
    if na_position == "first":
        sorted_values = np.concatenate([[None] * missing_count, sorted_values])
    else:
        sorted_values = np.concatenate([sorted_values, [None] * missing_count])

    # Explicitly pass dtype needed for Index backed by EA e.g. IntegerArray
    expected = type(index_with_missing)(sorted_values, dtype=index_with_missing.dtype)

    result = index_with_missing.sort_values(na_position=na_position)
    tm.assert_index_equal(result, expected)


def test_ndarray_compat_properties(index):
    if isinstance(index, PeriodIndex) and not IS64:
        pytest.skip("Overflow")
    idx = index
    assert idx.T.equals(idx)
    assert idx.transpose().equals(idx)

    values = idx.values

    assert idx.shape == values.shape
    assert idx.ndim == values.ndim
    assert idx.size == values.size

    if not isinstance(index, (RangeIndex, MultiIndex)):
        # These two are not backed by an ndarray
        assert idx.nbytes == values.nbytes

    # test for validity
    idx.nbytes
    idx.values.nbytes


def test_compare_read_only_array():
    # GH#57130
    arr = np.array([], dtype=object)
    arr.flags.writeable = False
    idx = pd.Index(arr)
    result = idx > 69
    assert result.dtype == bool

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\idna\compat.py ===
from typing import Any, Union

from .core import decode, encode


def ToASCII(label: str) -> bytes:
    return encode(label)


def ToUnicode(label: Union[bytes, bytearray]) -> str:
    return decode(label)


def nameprep(s: Any) -> None:
    raise NotImplementedError("IDNA 2008 does not utilise nameprep protocol")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_distributor_init.py ===
""" Distributor init file

Distributors: you can add custom code here to support particular distributions
of numpy.

For example, this is a good place to put any BLAS/LAPACK initialization code.

The numpy standard source distribution will not put code in this file, so you
can safely replace this file with your own version.
"""

try:
    from . import _distributor_init_local  # noqa: F401
except ImportError:
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_extended_precision.py ===
"""A module with platform-specific extended precision
`numpy.number` subclasses.

The subclasses are defined here (instead of ``__init__.pyi``) such
that they can be imported conditionally via the numpy's mypy plugin.
"""

import numpy as np

from . import _96Bit, _128Bit

float96 = np.floating[_96Bit]
float128 = np.floating[_128Bit]
complex192 = np.complexfloating[_96Bit, _96Bit]
complex256 = np.complexfloating[_128Bit, _128Bit]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\__init__.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

__title__ = "packaging"
__summary__ = "Core utilities for Python packages"
__uri__ = "https://github.com/pypa/packaging"

__version__ = "25.0"

__author__ = "Donald Stufft and individual contributors"
__email__ = "donald@stufft.io"

__license__ = "BSD-2-Clause or Apache-2.0"
__copyright__ = f"2014 {__author__}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\groupby\__init__.py ===
from pandas.core.groupby.generic import (
    DataFrameGroupBy,
    NamedAgg,
    SeriesGroupBy,
)
from pandas.core.groupby.groupby import GroupBy
from pandas.core.groupby.grouper import Grouper

__all__ = [
    "DataFrameGroupBy",
    "NamedAgg",
    "SeriesGroupBy",
    "GroupBy",
    "Grouper",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\json\__init__.py ===
from pandas.io.json._json import (
    read_json,
    to_json,
    ujson_dumps,
    ujson_loads,
)
from pandas.io.json._table_schema import build_table_schema

__all__ = [
    "ujson_dumps",
    "ujson_loads",
    "read_json",
    "to_json",
    "build_table_schema",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_array.py ===
import re

import numpy as np
import pytest

from pandas._libs.sparse import IntIndex
from pandas.compat.numpy import np_version_gt2

import pandas as pd
from pandas import (
    SparseDtype,
    isna,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


@pytest.fixture
def arr_data():
    """Fixture returning numpy array with valid and missing entries"""
    return np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 6])


@pytest.fixture
def arr(arr_data):
    """Fixture returning SparseArray from 'arr_data'"""
    return SparseArray(arr_data)


@pytest.fixture
def zarr():
    """Fixture returning SparseArray with integer entries and 'fill_value=0'"""
    return SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)


class TestSparseArray:
    @pytest.mark.parametrize("fill_value", [0, None, np.nan])
    def test_shift_fill_value(self, fill_value):
        # GH #24128
        sparse = SparseArray(np.array([1, 0, 0, 3, 0]), fill_value=8.0)
        res = sparse.shift(1, fill_value=fill_value)
        if isna(fill_value):
            fill_value = res.dtype.na_value
        exp = SparseArray(np.array([fill_value, 1, 0, 0, 3]), fill_value=8.0)
        tm.assert_sp_array_equal(res, exp)

    def test_set_fill_value(self):
        arr = SparseArray([1.0, np.nan, 2.0], fill_value=np.nan)
        arr.fill_value = 2
        assert arr.fill_value == 2

        arr = SparseArray([1, 0, 2], fill_value=0, dtype=np.int64)
        arr.fill_value = 2
        assert arr.fill_value == 2

        msg = "Allowing arbitrary scalar fill_value in SparseDtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            arr.fill_value = 3.1
        assert arr.fill_value == 3.1

        arr.fill_value = np.nan
        assert np.isnan(arr.fill_value)

        arr = SparseArray([True, False, True], fill_value=False, dtype=np.bool_)
        arr.fill_value = True
        assert arr.fill_value is True

        with tm.assert_produces_warning(FutureWarning, match=msg):
            arr.fill_value = 0

        arr.fill_value = np.nan
        assert np.isnan(arr.fill_value)

    @pytest.mark.parametrize("val", [[1, 2, 3], np.array([1, 2]), (1, 2, 3)])
    def test_set_fill_invalid_non_scalar(self, val):
        arr = SparseArray([True, False, True], fill_value=False, dtype=np.bool_)
        msg = "fill_value must be a scalar"

        with pytest.raises(ValueError, match=msg):
            arr.fill_value = val

    def test_copy(self, arr):
        arr2 = arr.copy()
        assert arr2.sp_values is not arr.sp_values
        assert arr2.sp_index is arr.sp_index

    def test_values_asarray(self, arr_data, arr):
        tm.assert_almost_equal(arr.to_dense(), arr_data)

    @pytest.mark.parametrize(
        "data,shape,dtype",
        [
            ([0, 0, 0, 0, 0], (5,), None),
            ([], (0,), None),
            ([0], (1,), None),
            (["A", "A", np.nan, "B"], (4,), object),
        ],
    )
    def test_shape(self, data, shape, dtype):
        # GH 21126
        out = SparseArray(data, dtype=dtype)
        assert out.shape == shape

    @pytest.mark.parametrize(
        "vals",
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [1, np.nan, np.nan, 3, np.nan],
            [1, np.nan, 0, 3, 0],
        ],
    )
    @pytest.mark.parametrize("fill_value", [None, 0])
    def test_dense_repr(self, vals, fill_value):
        vals = np.array(vals)
        arr = SparseArray(vals, fill_value=fill_value)

        res = arr.to_dense()
        tm.assert_numpy_array_equal(res, vals)

    @pytest.mark.parametrize("fix", ["arr", "zarr"])
    def test_pickle(self, fix, request):
        obj = request.getfixturevalue(fix)
        unpickled = tm.round_trip_pickle(obj)
        tm.assert_sp_array_equal(unpickled, obj)

    def test_generator_warnings(self):
        sp_arr = SparseArray([1, 2, 3])
        with tm.assert_produces_warning(None):
            for _ in sp_arr:
                pass

    def test_where_retain_fill_value(self):
        # GH#45691 don't lose fill_value on _where
        arr = SparseArray([np.nan, 1.0], fill_value=0)

        mask = np.array([True, False])

        res = arr._where(~mask, 1)
        exp = SparseArray([1, 1.0], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        ser = pd.Series(arr)
        res = ser.where(~mask, 1)
        tm.assert_series_equal(res, pd.Series(exp))

    def test_fillna(self):
        s = SparseArray([1, np.nan, np.nan, 3, np.nan])
        res = s.fillna(-1)
        exp = SparseArray([1, -1, -1, 3, -1], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([1, np.nan, np.nan, 3, np.nan], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([1, -1, -1, 3, -1], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([1, np.nan, 0, 3, 0])
        res = s.fillna(-1)
        exp = SparseArray([1, -1, 0, 3, 0], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([1, np.nan, 0, 3, 0], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([1, -1, 0, 3, 0], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([np.nan, np.nan, np.nan, np.nan])
        res = s.fillna(-1)
        exp = SparseArray([-1, -1, -1, -1], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([np.nan, np.nan, np.nan, np.nan], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([-1, -1, -1, -1], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        # float dtype's fill_value is np.nan, replaced by -1
        s = SparseArray([0.0, 0.0, 0.0, 0.0])
        res = s.fillna(-1)
        exp = SparseArray([0.0, 0.0, 0.0, 0.0], fill_value=-1)
        tm.assert_sp_array_equal(res, exp)

        # int dtype shouldn't have missing. No changes.
        s = SparseArray([0, 0, 0, 0])
        assert s.dtype == SparseDtype(np.int64)
        assert s.fill_value == 0
        res = s.fillna(-1)
        tm.assert_sp_array_equal(res, s)

        s = SparseArray([0, 0, 0, 0], fill_value=0)
        assert s.dtype == SparseDtype(np.int64)
        assert s.fill_value == 0
        res = s.fillna(-1)
        exp = SparseArray([0, 0, 0, 0], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        # fill_value can be nan if there is no missing hole.
        # only fill_value will be changed
        s = SparseArray([0, 0, 0, 0], fill_value=np.nan)
        assert s.dtype == SparseDtype(np.int64, fill_value=np.nan)
        assert np.isnan(s.fill_value)
        res = s.fillna(-1)
        exp = SparseArray([0, 0, 0, 0], fill_value=-1)
        tm.assert_sp_array_equal(res, exp)

    def test_fillna_overlap(self):
        s = SparseArray([1, np.nan, np.nan, 3, np.nan])
        # filling with existing value doesn't replace existing value with
        # fill_value, i.e. existing 3 remains in sp_values
        res = s.fillna(3)
        exp = np.array([1, 3, 3, 3, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(res.to_dense(), exp)

        s = SparseArray([1, np.nan, np.nan, 3, np.nan], fill_value=0)
        res = s.fillna(3)
        exp = SparseArray([1, 3, 3, 3, 3], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

    def test_nonzero(self):
        # Tests regression #21172.
        sa = SparseArray([float("nan"), float("nan"), 1, 0, 0, 2, 0, 0, 0, 3, 0, 0])
        expected = np.array([2, 5, 9], dtype=np.int32)
        (result,) = sa.nonzero()
        tm.assert_numpy_array_equal(expected, result)

        sa = SparseArray([0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0])
        (result,) = sa.nonzero()
        tm.assert_numpy_array_equal(expected, result)


class TestSparseArrayAnalytics:
    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                np.array([1, 2, 3, 4, 5], dtype=float),  # non-null data
                SparseArray(np.array([1.0, 3.0, 6.0, 10.0, 15.0])),
            ),
            (
                np.array([1, 2, np.nan, 4, 5], dtype=float),  # null data
                SparseArray(np.array([1.0, 3.0, np.nan, 7.0, 12.0])),
            ),
        ],
    )
    @pytest.mark.parametrize("numpy", [True, False])
    def test_cumsum(self, data, expected, numpy):
        cumsum = np.cumsum if numpy else lambda s: s.cumsum()

        out = cumsum(SparseArray(data))
        tm.assert_sp_array_equal(out, expected)

        out = cumsum(SparseArray(data, fill_value=np.nan))
        tm.assert_sp_array_equal(out, expected)

        out = cumsum(SparseArray(data, fill_value=2))
        tm.assert_sp_array_equal(out, expected)

        if numpy:  # numpy compatibility checks.
            msg = "the 'dtype' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), dtype=np.int64)

            msg = "the 'out' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), out=out)
        else:
            axis = 1  # SparseArray currently 1-D, so only axis = 0 is valid.
            msg = re.escape(f"axis(={axis}) out of bounds")
            with pytest.raises(ValueError, match=msg):
                SparseArray(data).cumsum(axis=axis)

    def test_ufunc(self):
        # GH 13853 make sure ufunc is applied to fill_value
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray([1, np.nan, 2, np.nan, 2])
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray([1, 2, 2], sparse_index=sparse.sp_index, fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=-1)
        exp = SparseArray([1, 1, 2, 2], fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), exp)
        tm.assert_sp_array_equal(np.abs(sparse), exp)

        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray(np.sin([1, np.nan, 2, np.nan, -2]))
        tm.assert_sp_array_equal(np.sin(sparse), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray(np.sin([1, -1, 2, -2]), fill_value=np.sin(1))
        tm.assert_sp_array_equal(np.sin(sparse), result)

        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        result = SparseArray(np.sin([1, -1, 0, -2]), fill_value=np.sin(0))
        tm.assert_sp_array_equal(np.sin(sparse), result)

    def test_ufunc_args(self):
        # GH 13853 make sure ufunc is applied to fill_value, including its arg
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray([2, np.nan, 3, np.nan, -1])
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray([2, 0, 3, -1], fill_value=2)
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        result = SparseArray([2, 0, 1, -1], fill_value=1)
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

    @pytest.mark.parametrize("fill_value", [0.0, np.nan])
    def test_modf(self, fill_value):
        # https://github.com/pandas-dev/pandas/issues/26946
        sparse = SparseArray([fill_value] * 10 + [1.1, 2.2], fill_value=fill_value)
        r1, r2 = np.modf(sparse)
        e1, e2 = np.modf(np.asarray(sparse))
        tm.assert_sp_array_equal(r1, SparseArray(e1, fill_value=fill_value))
        tm.assert_sp_array_equal(r2, SparseArray(e2, fill_value=fill_value))

    def test_nbytes_integer(self):
        arr = SparseArray([1, 0, 0, 0, 2], kind="integer")
        result = arr.nbytes
        # (2 * 8) + 2 * 4
        assert result == 24

    def test_nbytes_block(self):
        arr = SparseArray([1, 2, 0, 0, 0], kind="block")
        result = arr.nbytes
        # (2 * 8) + 4 + 4
        # sp_values, blocs, blengths
        assert result == 24

    def test_asarray_datetime64(self):
        s = SparseArray(pd.to_datetime(["2012", None, None, "2013"]))
        np.asarray(s)

    def test_density(self):
        arr = SparseArray([0, 1])
        assert arr.density == 0.5

    def test_npoints(self):
        arr = SparseArray([0, 1])
        assert arr.npoints == 1


def test_setting_fill_value_fillna_still_works():
    # This is why letting users update fill_value / dtype is bad
    # astype has the same problem.
    arr = SparseArray([1.0, np.nan, 1.0], fill_value=0.0)
    arr.fill_value = np.nan
    result = arr.isna()
    # Can't do direct comparison, since the sp_index will be different
    # So let's convert to ndarray and check there.
    result = np.asarray(result)

    expected = np.array([False, True, False])
    tm.assert_numpy_array_equal(result, expected)


def test_setting_fill_value_updates():
    arr = SparseArray([0.0, np.nan], fill_value=0)
    arr.fill_value = np.nan
    # use private constructor to get the index right
    # otherwise both nans would be un-stored.
    expected = SparseArray._simple_new(
        sparse_array=np.array([np.nan]),
        sparse_index=IntIndex(2, [1]),
        dtype=SparseDtype(float, np.nan),
    )
    tm.assert_sp_array_equal(arr, expected)


@pytest.mark.parametrize(
    "arr,fill_value,loc",
    [
        ([None, 1, 2], None, 0),
        ([0, None, 2], None, 1),
        ([0, 1, None], None, 2),
        ([0, 1, 1, None, None], None, 3),
        ([1, 1, 1, 2], None, -1),
        ([], None, -1),
        ([None, 1, 0, 0, None, 2], None, 0),
        ([None, 1, 0, 0, None, 2], 1, 1),
        ([None, 1, 0, 0, None, 2], 2, 5),
        ([None, 1, 0, 0, None, 2], 3, -1),
        ([None, 0, 0, 1, 2, 1], 0, 1),
        ([None, 0, 0, 1, 2, 1], 1, 3),
    ],
)
def test_first_fill_value_loc(arr, fill_value, loc):
    result = SparseArray(arr, fill_value=fill_value)._first_fill_value_loc()
    assert result == loc


@pytest.mark.parametrize(
    "arr",
    [
        [1, 2, np.nan, np.nan],
        [1, np.nan, 2, np.nan],
        [1, 2, np.nan],
        [np.nan, 1, 0, 0, np.nan, 2],
        [np.nan, 0, 0, 1, 2, 1],
    ],
)
@pytest.mark.parametrize("fill_value", [np.nan, 0, 1])
def test_unique_na_fill(arr, fill_value):
    a = SparseArray(arr, fill_value=fill_value).unique()
    b = pd.Series(arr).unique()
    assert isinstance(a, SparseArray)
    a = np.asarray(a)
    tm.assert_numpy_array_equal(a, b)


def test_unique_all_sparse():
    # https://github.com/pandas-dev/pandas/issues/23168
    arr = SparseArray([0, 0])
    result = arr.unique()
    expected = SparseArray([0])
    tm.assert_sp_array_equal(result, expected)


def test_map():
    arr = SparseArray([0, 1, 2])
    expected = SparseArray([10, 11, 12], fill_value=10)

    # dict
    result = arr.map({0: 10, 1: 11, 2: 12})
    tm.assert_sp_array_equal(result, expected)

    # series
    result = arr.map(pd.Series({0: 10, 1: 11, 2: 12}))
    tm.assert_sp_array_equal(result, expected)

    # function
    result = arr.map(pd.Series({0: 10, 1: 11, 2: 12}))
    expected = SparseArray([10, 11, 12], fill_value=10)
    tm.assert_sp_array_equal(result, expected)


def test_map_missing():
    arr = SparseArray([0, 1, 2])
    expected = SparseArray([10, 11, None], fill_value=10)

    result = arr.map({0: 10, 1: 11})
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("fill_value", [np.nan, 1])
def test_dropna(fill_value):
    # GH-28287
    arr = SparseArray([np.nan, 1], fill_value=fill_value)
    exp = SparseArray([1.0], fill_value=fill_value)
    tm.assert_sp_array_equal(arr.dropna(), exp)

    df = pd.DataFrame({"a": [0, 1], "b": arr})
    expected_df = pd.DataFrame({"a": [1], "b": exp}, index=pd.Index([1]))
    tm.assert_equal(df.dropna(), expected_df)


def test_drop_duplicates_fill_value():
    # GH 11726
    df = pd.DataFrame(np.zeros((5, 5))).apply(lambda x: SparseArray(x, fill_value=0))
    result = df.drop_duplicates()
    expected = pd.DataFrame({i: SparseArray([0.0], fill_value=0) for i in range(5)})
    tm.assert_frame_equal(result, expected)


def test_zero_sparse_column():
    # GH 27781
    df1 = pd.DataFrame({"A": SparseArray([0, 0, 0]), "B": [1, 2, 3]})
    df2 = pd.DataFrame({"A": SparseArray([0, 1, 0]), "B": [1, 2, 3]})
    result = df1.loc[df1["B"] != 2]
    expected = df2.loc[df2["B"] != 2]
    tm.assert_frame_equal(result, expected)

    expected = pd.DataFrame({"A": SparseArray([0, 0]), "B": [1, 3]}, index=[0, 2])
    tm.assert_frame_equal(result, expected)


def test_array_interface(arr_data, arr):
    # https://github.com/pandas-dev/pandas/pull/60046
    result = np.asarray(arr)
    tm.assert_numpy_array_equal(result, arr_data)

    # it always gives a copy by default
    result_copy1 = np.asarray(arr)
    result_copy2 = np.asarray(arr)
    assert not np.may_share_memory(result_copy1, result_copy2)

    # or with explicit copy=True
    result_copy1 = np.array(arr, copy=True)
    result_copy2 = np.array(arr, copy=True)
    assert not np.may_share_memory(result_copy1, result_copy2)

    if not np_version_gt2:
        # copy=False semantics are only supported in NumPy>=2.
        return

    msg = "Starting with NumPy 2.0, the behavior of the 'copy' keyword has changed"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        np.array(arr, copy=False)

    # except when there are actually no sparse filled values
    arr2 = SparseArray(np.array([1, 2, 3]))
    result_nocopy1 = np.array(arr2, copy=False)
    result_nocopy2 = np.array(arr2, copy=False)
    assert np.may_share_memory(result_nocopy1, result_nocopy2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\vcs\__init__.py ===
# Expose a limited set of classes and functions so callers outside of
# the vcs package don't need to import deeper than `pip._internal.vcs`.
# (The test directory may still need to import from a vcs sub-package.)
# Import all vcs modules to register each VCS in the VcsSupport object.
import pip._internal.vcs.bazaar
import pip._internal.vcs.git
import pip._internal.vcs.mercurial
import pip._internal.vcs.subversion  # noqa: F401
from pip._internal.vcs.versioncontrol import (  # noqa: F401
    RemoteNotFoundError,
    RemoteNotValidError,
    is_url,
    make_vcs_requirement_url,
    vcs,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\idna\compat.py ===
from typing import Any, Union

from .core import decode, encode


def ToASCII(label: str) -> bytes:
    return encode(label)


def ToUnicode(label: Union[bytes, bytearray]) -> str:
    return decode(label)


def nameprep(s: Any) -> None:
    raise NotImplementedError("IDNA 2008 does not utilise nameprep protocol")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\__init__.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

__title__ = "packaging"
__summary__ = "Core utilities for Python packages"
__uri__ = "https://github.com/pypa/packaging"

__version__ = "25.0"

__author__ = "Donald Stufft and individual contributors"
__email__ = "donald@stufft.io"

__license__ = "BSD-2-Clause or Apache-2.0"
__copyright__ = f"2014 {__author__}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\console\console_attributes.py ===
FOREGROUND_BLUE = 0x0001
FOREGROUND_GREEN = 0x0002
FOREGROUND_RED = 0x0004
FOREGROUND_INTENSITY = 0x0008
BACKGROUND_BLUE = 0x0010
BACKGROUND_GREEN = 0x0020
BACKGROUND_RED = 0x0040
BACKGROUND_INTENSITY = 0x0080
COMMON_LVB_LEADING_BYTE = 0x0100
COMMON_LVB_TRAILING_BYTE = 0x0200
COMMON_LVB_GRID_HORIZONTAL = 0x0400
COMMON_LVB_GRID_LVERTICAL = 0x0800
COMMON_LVB_GRID_RVERTICAL = 0x1000
COMMON_LVB_REVERSE_VIDEO = 0x2000
COMMON_LVB_UNDERSCORE = 0x4000

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\future\engine.py ===
# future/engine.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
"""2.0 API features.

this module is legacy as 2.0 APIs are now standard.

"""

from ..engine import Connection as Connection  # noqa: F401
from ..engine import create_engine as create_engine  # noqa: F401
from ..engine import Engine as Engine  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\benchmarks\bench_basic.py ===
from sympy.core import symbols, S

x, y = symbols('x,y')


def timeit_Symbol_meth_lookup():
    x.diff  # no call, just method lookup


def timeit_S_lookup():
    S.Exp1


def timeit_Symbol_eq_xy():
    x == y

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest1.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

f = _sm.S(3)
g = _sm.S(9.81)
a, b = _sm.symbols('a b', real=True)
s, s1 = _sm.symbols('s s1', real=True)
s2, s3 = _sm.symbols('s2 s3', real=True, nonnegative=True)
s4 = _sm.symbols('s4', real=True, nonpositive=True)
k1, k2, k3, k4, l1, l2, l3, p11, p12, p13, p21, p22, p23 = _sm.symbols('k1 k2 k3 k4 l1 l2 l3 p11 p12 p13 p21 p22 p23', real=True)
c11, c12, c13, c21, c22, c23 = _sm.symbols('c11 c12 c13 c21 c22 c23', real=True)
e1 = a*f+s2-g
e2 = f**2+k3*k2*g

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\characteristiczero.py ===
"""Implementation of :class:`CharacteristicZero` class. """


from sympy.polys.domains.domain import Domain
from sympy.utilities import public

@public
class CharacteristicZero(Domain):
    """Domain that has infinite number of elements. """

    has_CharacteristicZero = True

    def characteristic(self):
        """Return the characteristic of this domain. """
        return 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\simpledomain.py ===
"""Implementation of :class:`SimpleDomain` class. """


from sympy.polys.domains.domain import Domain
from sympy.utilities import public

@public
class SimpleDomain(Domain):
    """Base class for simple domains, e.g. ZZ, QQ. """

    is_Simple = True

    def inject(self, *gens):
        """Inject generators into this domain. """
        return self.poly_ring(*gens)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\__init__.py ===
"""

sympy.polys.matrices package.

The main export from this package is the DomainMatrix class which is a
lower-level implementation of matrices based on the polys Domains. This
implementation is typically a lot faster than SymPy's standard Matrix class
but is a work in progress and is still experimental.

"""
from .domainmatrix import DomainMatrix, DM

__all__ = [
    'DomainMatrix', 'DM',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\traversaltools.py ===
from sympy.core.traversal import use as _use
from sympy.utilities.decorator import deprecated

use = deprecated(
    """
    Using use from the sympy.simplify.traversaltools submodule is
    deprecated.

    Instead, use use from the top-level sympy namespace, like

        sympy.use
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved"
)(_use)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\test_indexed.py ===
from sympy.core import symbols, Symbol, Tuple, oo, Dummy
from sympy.tensor.indexed import IndexException
from sympy.testing.pytest import raises
from sympy.utilities.iterables import iterable

# import test:
from sympy.concrete.summations import Sum
from sympy.core.function import Function, Subs, Derivative
from sympy.core.relational import (StrictLessThan, GreaterThan,
    StrictGreaterThan, LessThan)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.series.order import Order
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import IndexedBase, Idx, Indexed


def test_Idx_construction():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i) != Idx(i, 1)
    assert Idx(i, a) == Idx(i, (0, a - 1))
    assert Idx(i, oo) == Idx(i, (0, oo))

    x = symbols('x', integer=False)
    raises(TypeError, lambda: Idx(x))
    raises(TypeError, lambda: Idx(0.5))
    raises(TypeError, lambda: Idx(i, x))
    raises(TypeError, lambda: Idx(i, 0.5))
    raises(TypeError, lambda: Idx(i, (x, 5)))
    raises(TypeError, lambda: Idx(i, (2, x)))
    raises(TypeError, lambda: Idx(i, (2, 3.5)))


def test_Idx_properties():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i).is_integer
    assert Idx(i).name == 'i'
    assert Idx(i + 2).name == 'i + 2'
    assert Idx('foo').name == 'foo'


def test_Idx_bounds():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i).lower is None
    assert Idx(i).upper is None
    assert Idx(i, a).lower == 0
    assert Idx(i, a).upper == a - 1
    assert Idx(i, 5).lower == 0
    assert Idx(i, 5).upper == 4
    assert Idx(i, oo).lower == 0
    assert Idx(i, oo).upper is oo
    assert Idx(i, (a, b)).lower == a
    assert Idx(i, (a, b)).upper == b
    assert Idx(i, (1, 5)).lower == 1
    assert Idx(i, (1, 5)).upper == 5
    assert Idx(i, (-oo, oo)).lower is -oo
    assert Idx(i, (-oo, oo)).upper is oo


def test_Idx_fixed_bounds():
    i, a, b, x = symbols('i a b x', integer=True)
    assert Idx(x).lower is None
    assert Idx(x).upper is None
    assert Idx(x, a).lower == 0
    assert Idx(x, a).upper == a - 1
    assert Idx(x, 5).lower == 0
    assert Idx(x, 5).upper == 4
    assert Idx(x, oo).lower == 0
    assert Idx(x, oo).upper is oo
    assert Idx(x, (a, b)).lower == a
    assert Idx(x, (a, b)).upper == b
    assert Idx(x, (1, 5)).lower == 1
    assert Idx(x, (1, 5)).upper == 5
    assert Idx(x, (-oo, oo)).lower is -oo
    assert Idx(x, (-oo, oo)).upper is oo


def test_Idx_inequalities():
    i14 = Idx("i14", (1, 4))
    i79 = Idx("i79", (7, 9))
    i46 = Idx("i46", (4, 6))
    i35 = Idx("i35", (3, 5))

    assert i14 <= 5
    assert i14 < 5
    assert not (i14 >= 5)
    assert not (i14 > 5)

    assert 5 >= i14
    assert 5 > i14
    assert not (5 <= i14)
    assert not (5 < i14)

    assert LessThan(i14, 5)
    assert StrictLessThan(i14, 5)
    assert not GreaterThan(i14, 5)
    assert not StrictGreaterThan(i14, 5)

    assert i14 <= 4
    assert isinstance(i14 < 4, StrictLessThan)
    assert isinstance(i14 >= 4, GreaterThan)
    assert not (i14 > 4)

    assert isinstance(i14 <= 1, LessThan)
    assert not (i14 < 1)
    assert i14 >= 1
    assert isinstance(i14 > 1, StrictGreaterThan)

    assert not (i14 <= 0)
    assert not (i14 < 0)
    assert i14 >= 0
    assert i14 > 0

    from sympy.abc import x

    assert isinstance(i14 < x, StrictLessThan)
    assert isinstance(i14 > x, StrictGreaterThan)
    assert isinstance(i14 <= x, LessThan)
    assert isinstance(i14 >= x, GreaterThan)

    assert i14 < i79
    assert i14 <= i79
    assert not (i14 > i79)
    assert not (i14 >= i79)

    assert i14 <= i46
    assert isinstance(i14 < i46, StrictLessThan)
    assert isinstance(i14 >= i46, GreaterThan)
    assert not (i14 > i46)

    assert isinstance(i14 < i35, StrictLessThan)
    assert isinstance(i14 > i35, StrictGreaterThan)
    assert isinstance(i14 <= i35, LessThan)
    assert isinstance(i14 >= i35, GreaterThan)

    iNone1 = Idx("iNone1")
    iNone2 = Idx("iNone2")

    assert isinstance(iNone1 < iNone2, StrictLessThan)
    assert isinstance(iNone1 > iNone2, StrictGreaterThan)
    assert isinstance(iNone1 <= iNone2, LessThan)
    assert isinstance(iNone1 >= iNone2, GreaterThan)


def test_Idx_inequalities_current_fails():
    i14 = Idx("i14", (1, 4))

    assert S(5) >= i14
    assert S(5) > i14
    assert not (S(5) <= i14)
    assert not (S(5) < i14)


def test_Idx_func_args():
    i, a, b = symbols('i a b', integer=True)
    ii = Idx(i)
    assert ii.func(*ii.args) == ii
    ii = Idx(i, a)
    assert ii.func(*ii.args) == ii
    ii = Idx(i, (a, b))
    assert ii.func(*ii.args) == ii


def test_Idx_subs():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i, a).subs(a, b) == Idx(i, b)
    assert Idx(i, a).subs(i, b) == Idx(b, a)

    assert Idx(i).subs(i, 2) == Idx(2)
    assert Idx(i, a).subs(a, 2) == Idx(i, 2)
    assert Idx(i, (a, b)).subs(i, 2) == Idx(2, (a, b))


def test_IndexedBase_sugar():
    i, j = symbols('i j', integer=True)
    a = symbols('a')
    A1 = Indexed(a, i, j)
    A2 = IndexedBase(a)
    assert A1 == A2[i, j]
    assert A1 == A2[(i, j)]
    assert A1 == A2[[i, j]]
    assert A1 == A2[Tuple(i, j)]
    assert all(a.is_Integer for a in A2[1, 0].args[1:])


def test_IndexedBase_subs():
    i = symbols('i', integer=True)
    a, b = symbols('a b')
    A = IndexedBase(a)
    B = IndexedBase(b)
    assert A[i] == B[i].subs(b, a)
    C = {1: 2}
    assert C[1] == A[1].subs(A, C)


def test_IndexedBase_shape():
    i, j, m, n = symbols('i j m n', integer=True)
    a = IndexedBase('a', shape=(m, m))
    b = IndexedBase('a', shape=(m, n))
    assert b.shape == Tuple(m, n)
    assert a[i, j] != b[i, j]
    assert a[i, j] == b[i, j].subs(n, m)
    assert b.func(*b.args) == b
    assert b[i, j].func(*b[i, j].args) == b[i, j]
    raises(IndexException, lambda: b[i])
    raises(IndexException, lambda: b[i, i, j])
    F = IndexedBase("F", shape=m)
    assert F.shape == Tuple(m)
    assert F[i].subs(i, j) == F[j]
    raises(IndexException, lambda: F[i, j])


def test_IndexedBase_assumptions():
    i = Symbol('i', integer=True)
    a = Symbol('a')
    A = IndexedBase(a, positive=True)
    for c in (A, A[i]):
        assert c.is_real
        assert c.is_complex
        assert not c.is_imaginary
        assert c.is_nonnegative
        assert c.is_nonzero
        assert c.is_commutative
        assert log(exp(c)) == c

    assert A != IndexedBase(a)
    assert A == IndexedBase(a, positive=True, real=True)
    assert A[i] != Indexed(a, i)


def test_IndexedBase_assumptions_inheritance():
    I = Symbol('I', integer=True)
    I_inherit = IndexedBase(I)
    I_explicit = IndexedBase('I', integer=True)

    assert I_inherit.is_integer
    assert I_explicit.is_integer
    assert I_inherit.label.is_integer
    assert I_explicit.label.is_integer
    assert I_inherit == I_explicit


def test_issue_17652():
    """Regression test issue #17652.

    IndexedBase.label should not upcast subclasses of Symbol
    """
    class SubClass(Symbol):
        pass

    x = SubClass('X')
    assert type(x) == SubClass
    base = IndexedBase(x)
    assert type(x) == SubClass
    assert type(base.label) == SubClass


def test_Indexed_constructor():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, j)
    assert A == Indexed(Symbol('A'), i, j)
    assert A == Indexed(IndexedBase('A'), i, j)
    raises(TypeError, lambda: Indexed(A, i, j))
    raises(IndexException, lambda: Indexed("A"))
    assert A.free_symbols == {A, A.base.label, i, j}


def test_Indexed_func_args():
    i, j = symbols('i j', integer=True)
    a = symbols('a')
    A = Indexed(a, i, j)
    assert A == A.func(*A.args)


def test_Indexed_subs():
    i, j, k = symbols('i j k', integer=True)
    a, b = symbols('a b')
    A = IndexedBase(a)
    B = IndexedBase(b)
    assert A[i, j] == B[i, j].subs(b, a)
    assert A[i, j] == A[i, k].subs(k, j)


def test_Indexed_properties():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, j)
    assert A.name == 'A[i, j]'
    assert A.rank == 2
    assert A.indices == (i, j)
    assert A.base == IndexedBase('A')
    assert A.ranges == [None, None]
    raises(IndexException, lambda: A.shape)

    n, m = symbols('n m', integer=True)
    assert Indexed('A', Idx(
        i, m), Idx(j, n)).ranges == [Tuple(0, m - 1), Tuple(0, n - 1)]
    assert Indexed('A', Idx(i, m), Idx(j, n)).shape == Tuple(m, n)
    raises(IndexException, lambda: Indexed("A", Idx(i, m), Idx(j)).shape)


def test_Indexed_shape_precedence():
    i, j = symbols('i j', integer=True)
    o, p = symbols('o p', integer=True)
    n, m = symbols('n m', integer=True)
    a = IndexedBase('a', shape=(o, p))
    assert a.shape == Tuple(o, p)
    assert Indexed(
        a, Idx(i, m), Idx(j, n)).ranges == [Tuple(0, m - 1), Tuple(0, n - 1)]
    assert Indexed(a, Idx(i, m), Idx(j, n)).shape == Tuple(o, p)
    assert Indexed(
        a, Idx(i, m), Idx(j)).ranges == [Tuple(0, m - 1), (None, None)]
    assert Indexed(a, Idx(i, m), Idx(j)).shape == Tuple(o, p)


def test_complex_indices():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, i + j)
    assert A.rank == 2
    assert A.indices == (i, i + j)


def test_not_interable():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, i + j)
    assert not iterable(A)


def test_Indexed_coeff():
    N = Symbol('N', integer=True)
    len_y = N
    i = Idx('i', len_y-1)
    y = IndexedBase('y', shape=(len_y,))
    a = (1/y[i+1]*y[i]).coeff(y[i])
    b = (y[i]/y[i+1]).coeff(y[i])
    assert a == b


def test_differentiation():
    from sympy.functions.special.tensor_functions import KroneckerDelta
    i, j, k, l = symbols('i j k l', cls=Idx)
    a = symbols('a')
    m, n = symbols("m, n", integer=True, finite=True)
    assert m.is_real
    h, L = symbols('h L', cls=IndexedBase)
    hi, hj = h[i], h[j]

    expr = hi
    assert expr.diff(hj) == KroneckerDelta(i, j)
    assert expr.diff(hi) == KroneckerDelta(i, i)

    expr = S(2) * hi
    assert expr.diff(hj) == S(2) * KroneckerDelta(i, j)
    assert expr.diff(hi) == S(2) * KroneckerDelta(i, i)
    assert expr.diff(a) is S.Zero

    assert Sum(expr, (i, -oo, oo)).diff(hj) == Sum(2*KroneckerDelta(i, j), (i, -oo, oo))
    assert Sum(expr.diff(hj), (i, -oo, oo)) == Sum(2*KroneckerDelta(i, j), (i, -oo, oo))
    assert Sum(expr, (i, -oo, oo)).diff(hj).doit() == 2

    assert Sum(expr.diff(hi), (i, -oo, oo)).doit() == Sum(2, (i, -oo, oo)).doit()
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() is oo

    expr = a * hj * hj / S(2)
    assert expr.diff(hi) == a * h[j] * KroneckerDelta(i, j)
    assert expr.diff(a) == hj * hj / S(2)
    assert expr.diff(a, 2) is S.Zero

    assert Sum(expr, (i, -oo, oo)).diff(hi) == Sum(a*KroneckerDelta(i, j)*h[j], (i, -oo, oo))
    assert Sum(expr.diff(hi), (i, -oo, oo)) == Sum(a*KroneckerDelta(i, j)*h[j], (i, -oo, oo))
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() == a*h[j]

    assert Sum(expr, (j, -oo, oo)).diff(hi) == Sum(a*KroneckerDelta(i, j)*h[j], (j, -oo, oo))
    assert Sum(expr.diff(hi), (j, -oo, oo)) == Sum(a*KroneckerDelta(i, j)*h[j], (j, -oo, oo))
    assert Sum(expr, (j, -oo, oo)).diff(hi).doit() == a*h[i]

    expr = a * sin(hj * hj)
    assert expr.diff(hi) == 2*a*cos(hj * hj) * hj * KroneckerDelta(i, j)
    assert expr.diff(hj) == 2*a*cos(hj * hj) * hj

    expr = a * L[i, j] * h[j]
    assert expr.diff(hi) == a*L[i, j]*KroneckerDelta(i, j)
    assert expr.diff(hj) == a*L[i, j]
    assert expr.diff(L[i, j]) == a*h[j]
    assert expr.diff(L[k, l]) == a*KroneckerDelta(i, k)*KroneckerDelta(j, l)*h[j]
    assert expr.diff(L[i, l]) == a*KroneckerDelta(j, l)*h[j]

    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]) == Sum(a * KroneckerDelta(i, k) * KroneckerDelta(j, l) * h[j], (j, -oo, oo))
    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]).doit() == a * KroneckerDelta(i, k) * h[l]

    assert h[m].diff(h[m]) == 1
    assert h[m].diff(h[n]) == KroneckerDelta(m, n)
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[n]) == Sum(a*KroneckerDelta(m, n), (m, -oo, oo))
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[n]).doit() == a
    assert Sum(a*h[m], (n, -oo, oo)).diff(h[n]) == Sum(a*KroneckerDelta(m, n), (n, -oo, oo))
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[m]).doit() == oo*a


def test_indexed_series():
    A = IndexedBase("A")
    i = symbols("i", integer=True)
    assert sin(A[i]).series(A[i]) == A[i] - A[i]**3/6 + A[i]**5/120 + Order(A[i]**6, A[i])


def test_indexed_is_constant():
    A = IndexedBase("A")
    i, j, k = symbols("i,j,k")
    assert not A[i].is_constant()
    assert A[i].is_constant(j)
    assert not A[1+2*i, k].is_constant()
    assert not A[1+2*i, k].is_constant(i)
    assert A[1+2*i, k].is_constant(j)
    assert not A[1+2*i, k].is_constant(k)


def test_issue_12533():
    d = IndexedBase('d')
    assert IndexedBase(range(5)) == Range(0, 5, 1)
    assert d[0].subs(Symbol("d"), range(5)) == 0
    assert d[0].subs(d, range(5)) == 0
    assert d[1].subs(d, range(5)) == 1
    assert Indexed(Range(5), 2) == 2


def test_issue_12780():
    n = symbols("n")
    i = Idx("i", (0, n))
    raises(TypeError, lambda: i.subs(n, 1.5))


def test_issue_18604():
    m = symbols("m")
    assert Idx("i", m).name == 'i'
    assert Idx("i", m).lower == 0
    assert Idx("i", m).upper == m - 1
    m = symbols("m", real=False)
    raises(TypeError, lambda: Idx("i", m))

def test_Subs_with_Indexed():
    A = IndexedBase("A")
    i, j, k = symbols("i,j,k")
    x, y, z = symbols("x,y,z")
    f = Function("f")

    assert Subs(A[i], A[i], A[j]).diff(A[j]) == 1
    assert Subs(A[i], A[i], x).diff(A[i]) == 0
    assert Subs(A[i], A[i], x).diff(A[j]) == 0
    assert Subs(A[i], A[i], x).diff(x) == 1
    assert Subs(A[i], A[i], x).diff(y) == 0
    assert Subs(A[i], A[i], A[j]).diff(A[k]) == KroneckerDelta(j, k)
    assert Subs(x, x, A[i]).diff(A[j]) == KroneckerDelta(i, j)
    assert Subs(f(A[i]), A[i], x).diff(A[j]) == 0
    assert Subs(f(A[i]), A[i], A[k]).diff(A[j]) == Derivative(f(A[k]), A[k])*KroneckerDelta(j, k)
    assert Subs(x, x, A[i]**2).diff(A[j]) == 2*KroneckerDelta(i, j)*A[i]
    assert Subs(A[i], A[i], A[j]**2).diff(A[k]) == 2*KroneckerDelta(j, k)*A[j]

    assert Subs(A[i]*x, x, A[i]).diff(A[i]) == 2*A[i]
    assert Subs(A[i]*x, x, A[i]).diff(A[j]) == 2*A[i]*KroneckerDelta(i, j)
    assert Subs(A[i]*x, x, A[j]).diff(A[i]) == A[j] + A[i]*KroneckerDelta(i, j)
    assert Subs(A[i]*x, x, A[j]).diff(A[j]) == A[i] + A[j]*KroneckerDelta(i, j)
    assert Subs(A[i]*x, x, A[i]).diff(A[k]) == 2*A[i]*KroneckerDelta(i, k)
    assert Subs(A[i]*x, x, A[j]).diff(A[k]) == KroneckerDelta(i, k)*A[j] + KroneckerDelta(j, k)*A[i]

    assert Subs(A[i]*x, A[i], x).diff(A[i]) == 0
    assert Subs(A[i]*x, A[i], x).diff(A[j]) == 0
    assert Subs(A[i]*x, A[j], x).diff(A[i]) == x
    assert Subs(A[i]*x, A[j], x).diff(A[j]) == x*KroneckerDelta(i, j)
    assert Subs(A[i]*x, A[i], x).diff(A[k]) == 0
    assert Subs(A[i]*x, A[j], x).diff(A[k]) == x*KroneckerDelta(i, k)


def test_complicated_derivative_with_Indexed():
    x, y = symbols("x,y", cls=IndexedBase)
    sigma = symbols("sigma")
    i, j, k = symbols("i,j,k")
    m0,m1,m2,m3,m4,m5 = symbols("m0:6")
    f = Function("f")

    expr = f((x[i] - y[i])**2/sigma)
    _xi_1 = symbols("xi_1", cls=Dummy)
    assert expr.diff(x[m0]).dummy_eq(
        (x[i] - y[i])*KroneckerDelta(i, m0)*\
        2*Subs(
            Derivative(f(_xi_1), _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2/sigma,)
        )/sigma
    )
    assert expr.diff(x[m0]).diff(x[m1]).dummy_eq(
        2*KroneckerDelta(i, m0)*\
        KroneckerDelta(i, m1)*Subs(
            Derivative(f(_xi_1), _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2/sigma,)
         )/sigma + \
        4*(x[i] - y[i])**2*KroneckerDelta(i, m0)*KroneckerDelta(i, m1)*\
        Subs(
            Derivative(f(_xi_1), _xi_1, _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2/sigma,)
        )/sigma**2
    )


def test_IndexedBase_commutative():
    t = IndexedBase('t', commutative=False)
    u = IndexedBase('u', commutative=False)
    v = IndexedBase('v')
    assert t[0]*v[0] == v[0]*t[0]
    assert t[0]*u[0] != u[0]*t[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\unify\__init__.py ===
""" Unification in SymPy

See sympy.unify.core docstring for algorithmic details

See http://matthewrocklin.com/blog/work/2012/11/01/Unification/ for discussion
"""

from .usympy import unify, rebuild
from .rewrite import rewriterule

__all__ = [
    'unify', 'rebuild',

    'rewriterule',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_run_context.py ===
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from ._run import Runner, Task


class RunContext(threading.local):
    runner: Runner
    task: Task


GLOBAL_RUN_CONTEXT: Final = RunContext()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\tests\test_code_quality.py ===
# coding=utf-8
from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast

from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding

# System path separator (usually slash or backslash) to be
# used with excluded files, e.g.
#     exclude = set([
#                    "%(sep)smpmath%(sep)s" % sepd,
#                   ])
sepd = {"sep": sep}

# path and sympy_path
SYMPY_PATH = abspath(join(split(__file__)[0], pardir, pardir))  # go to sympy/
assert exists(SYMPY_PATH)

TOP_PATH = abspath(join(SYMPY_PATH, pardir))
BIN_PATH = join(TOP_PATH, "bin")
EXAMPLES_PATH = join(TOP_PATH, "examples")

# Error messages
message_space = "File contains trailing whitespace: %s, line %s."
message_implicit = "File contains an implicit import: %s, line %s."
message_tabs = "File contains tabs instead of spaces: %s, line %s."
message_carriage = "File contains carriage returns at end of line: %s, line %s"
message_str_raise = "File contains string exception: %s, line %s"
message_gen_raise = "File contains generic exception: %s, line %s"
message_old_raise = "File contains old-style raise statement: %s, line %s, \"%s\""
message_eof = "File does not end with a newline: %s, line %s"
message_multi_eof = "File ends with more than 1 newline: %s, line %s"
message_test_suite_def = "Function should start with 'test_' or '_': %s, line %s"
message_duplicate_test = "This is a duplicate test function: %s, line %s"
message_self_assignments = "File contains assignments to self/cls: %s, line %s."
message_func_is = "File contains '.func is': %s, line %s."
message_bare_expr = "File contains bare expression: %s, line %s."

implicit_test_re = re.compile(r'^\s*(>>> )?(\.\.\. )?from .* import .*\*')
str_raise_re = re.compile(
    r'^\s*(>>> )?(\.\.\. )?raise(\s+(\'|\")|\s*(\(\s*)+(\'|\"))')
gen_raise_re = re.compile(
    r'^\s*(>>> )?(\.\.\. )?raise(\s+Exception|\s*(\(\s*)+Exception)')
old_raise_re = re.compile(r'^\s*(>>> )?(\.\.\. )?raise((\s*\(\s*)|\s+)\w+\s*,')
test_suite_def_re = re.compile(r'^def\s+(?!(_|test))[^(]*\(\s*\)\s*:$')
test_ok_def_re = re.compile(r'^def\s+test_.*:$')
test_file_re = re.compile(r'.*[/\\]test_.*\.py$')
func_is_re = re.compile(r'\.\s*func\s+is')


def tab_in_leading(s):
    """Returns True if there are tabs in the leading whitespace of a line,
    including the whitespace of docstring code samples."""
    n = len(s) - len(s.lstrip())
    if not s[n:n + 3] in ['...', '>>>']:
        check = s[:n]
    else:
        smore = s[n + 3:]
        check = s[:n] + smore[:len(smore) - len(smore.lstrip())]
    return not (check.expandtabs() == check)


def find_self_assignments(s):
    """Returns a list of "bad" assignments: if there are instances
    of assigning to the first argument of the class method (except
    for staticmethod's).
    """
    t = [n for n in ast.parse(s).body if isinstance(n, ast.ClassDef)]

    bad = []
    for c in t:
        for n in c.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            if any(d.id == 'staticmethod'
                   for d in n.decorator_list if isinstance(d, ast.Name)):
                continue
            if n.name == '__new__':
                continue
            if not n.args.args:
                continue
            first_arg = n.args.args[0].arg

            for m in ast.walk(n):
                if isinstance(m, ast.Assign):
                    for a in m.targets:
                        if isinstance(a, ast.Name) and a.id == first_arg:
                            bad.append(m)
                        elif (isinstance(a, ast.Tuple) and
                              any(q.id == first_arg for q in a.elts
                                  if isinstance(q, ast.Name))):
                            bad.append(m)

    return bad


def check_directory_tree(base_path, file_check, exclusions=set(), pattern="*.py"):
    """
    Checks all files in the directory tree (with base_path as starting point)
    with the file_check function provided, skipping files that contain
    any of the strings in the set provided by exclusions.
    """
    if not base_path:
        return
    for root, dirs, files in walk(base_path):
        check_files(glob(join(root, pattern)), file_check, exclusions)


def check_files(files, file_check, exclusions=set(), pattern=None):
    """
    Checks all files with the file_check function provided, skipping files
    that contain any of the strings in the set provided by exclusions.
    """
    if not files:
        return
    for fname in files:
        if not exists(fname) or not isfile(fname):
            continue
        if any(ex in fname for ex in exclusions):
            continue
        if pattern is None or re.match(pattern, fname):
            file_check(fname)


class _Visit(ast.NodeVisitor):
    """return the line number corresponding to the
    line on which a bare expression appears if it is a binary op
    or a comparison that is not in a with block.

    EXAMPLES
    ========

    >>> import ast
    >>> class _Visit(ast.NodeVisitor):
    ...     def visit_Expr(self, node):
    ...         if isinstance(node.value, (ast.BinOp, ast.Compare)):
    ...             print(node.lineno)
    ...     def visit_With(self, node):
    ...         pass  # no checking there
    ...
    >>> code='''x = 1    # line 1
    ... for i in range(3):
    ...     x == 2       # <-- 3
    ... if x == 2:
    ...     x == 3       # <-- 5
    ...     x + 1        # <-- 6
    ...     x = 1
    ...     if x == 1:
    ...         print(1)
    ... while x != 1:
    ...     x == 1       # <-- 11
    ... with raises(TypeError):
    ...     c == 1
    ...     raise TypeError
    ... assert x == 1
    ... '''
    >>> _Visit().visit(ast.parse(code))
    3
    5
    6
    11
    """
    def visit_Expr(self, node):
        if isinstance(node.value, (ast.BinOp, ast.Compare)):
            assert None, message_bare_expr % ('', node.lineno)
    def visit_With(self, node):
        pass


BareExpr = _Visit()


def line_with_bare_expr(code):
    """return None or else 0-based line number of code on which
    a bare expression appeared.
    """
    tree = ast.parse(code)
    try:
        BareExpr.visit(tree)
    except AssertionError as msg:
        assert msg.args
        msg = msg.args[0]
        assert msg.startswith(message_bare_expr.split(':', 1)[0])
        return int(msg.rsplit(' ', 1)[1].rstrip('.'))  # the line number


def test_files():
    """
    This test tests all files in SymPy and checks that:
      o no lines contains a trailing whitespace
      o no lines end with \r\n
      o no line uses tabs instead of spaces
      o that the file ends with a single newline
      o there are no general or string exceptions
      o there are no old style raise statements
      o name of arg-less test suite functions start with _ or test_
      o no duplicate function names that start with test_
      o no assignments to self variable in class methods
      o no lines contain ".func is" except in the test suite
      o there is no do-nothing expression like `a == b` or `x + 1`
    """

    def test(fname):
        with open(fname, encoding="utf8") as test_file:
            test_this_file(fname, test_file)
        with open(fname, encoding='utf8') as test_file:
            _test_this_file_encoding(fname, test_file)

    def test_this_file(fname, test_file):
        idx = None
        code = test_file.read()
        test_file.seek(0)  # restore reader to head
        py = fname if sep not in fname else fname.rsplit(sep, 1)[-1]
        if py.startswith('test_'):
            idx = line_with_bare_expr(code)
        if idx is not None:
            assert False, message_bare_expr % (fname, idx + 1)

        line = None  # to flag the case where there were no lines in file
        tests = 0
        test_set = set()
        for idx, line in enumerate(test_file):
            if test_file_re.match(fname):
                if test_suite_def_re.match(line):
                    assert False, message_test_suite_def % (fname, idx + 1)
                if test_ok_def_re.match(line):
                    tests += 1
                    test_set.add(line[3:].split('(')[0].strip())
                    if len(test_set) != tests:
                        assert False, message_duplicate_test % (fname, idx + 1)
            if line.endswith((" \n", "\t\n")):
                assert False, message_space % (fname, idx + 1)
            if line.endswith("\r\n"):
                assert False, message_carriage % (fname, idx + 1)
            if tab_in_leading(line):
                assert False, message_tabs % (fname, idx + 1)
            if str_raise_re.search(line):
                assert False, message_str_raise % (fname, idx + 1)
            if gen_raise_re.search(line):
                assert False, message_gen_raise % (fname, idx + 1)
            if (implicit_test_re.search(line) and
                    not list(filter(lambda ex: ex in fname, import_exclude))):
                assert False, message_implicit % (fname, idx + 1)
            if func_is_re.search(line) and not test_file_re.search(fname):
                assert False, message_func_is % (fname, idx + 1)

            result = old_raise_re.search(line)

            if result is not None:
                assert False, message_old_raise % (
                    fname, idx + 1, result.group(2))

        if line is not None:
            if line == '\n' and idx > 0:
                assert False, message_multi_eof % (fname, idx + 1)
            elif not line.endswith('\n'):
                # eof newline check
                assert False, message_eof % (fname, idx + 1)


    # Files to test at top level
    top_level_files = [join(TOP_PATH, file) for file in [
        "isympy.py",
        "build.py",
        "setup.py",
    ]]
    # Files to exclude from all tests
    exclude = {
        "%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevparser.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevlexer.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevlistener.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)slatex%(sep)s_antlr%(sep)slatexparser.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)slatex%(sep)s_antlr%(sep)slatexlexer.py" % sepd,
    }
    # Files to exclude from the implicit import test
    import_exclude = {
        # glob imports are allowed in top-level __init__.py:
        "%(sep)ssympy%(sep)s__init__.py" % sepd,
        # these __init__.py should be fixed:
        # XXX: not really, they use useful import pattern (DRY)
        "%(sep)svector%(sep)s__init__.py" % sepd,
        "%(sep)smechanics%(sep)s__init__.py" % sepd,
        "%(sep)squantum%(sep)s__init__.py" % sepd,
        "%(sep)spolys%(sep)s__init__.py" % sepd,
        "%(sep)spolys%(sep)sdomains%(sep)s__init__.py" % sepd,
        # interactive SymPy executes ``from sympy import *``:
        "%(sep)sinteractive%(sep)ssession.py" % sepd,
        # isympy.py executes ``from sympy import *``:
        "%(sep)sisympy.py" % sepd,
        # these two are import timing tests:
        "%(sep)sbin%(sep)ssympy_time.py" % sepd,
        "%(sep)sbin%(sep)ssympy_time_cache.py" % sepd,
        # Taken from Python stdlib:
        "%(sep)sparsing%(sep)ssympy_tokenize.py" % sepd,
        # this one should be fixed:
        "%(sep)splotting%(sep)spygletplot%(sep)s" % sepd,
        # False positive in the docstring
        "%(sep)sbin%(sep)stest_external_imports.py" % sepd,
        "%(sep)sbin%(sep)stest_submodule_imports.py" % sepd,
        # These are deprecated stubs that can be removed at some point:
        "%(sep)sutilities%(sep)sruntests.py" % sepd,
        "%(sep)sutilities%(sep)spytest.py" % sepd,
        "%(sep)sutilities%(sep)srandtest.py" % sepd,
        "%(sep)sutilities%(sep)stmpfiles.py" % sepd,
        "%(sep)sutilities%(sep)squality_unicode.py" % sepd,
    }
    check_files(top_level_files, test)
    check_directory_tree(BIN_PATH, test, {"~", ".pyc", ".sh"}, "*")
    check_directory_tree(SYMPY_PATH, test, exclude)
    check_directory_tree(EXAMPLES_PATH, test, exclude)


def _with_space(c):
    # return c with a random amount of leading space
    return random.randint(0, 10)*' ' + c


def test_raise_statement_regular_expression():
    candidates_ok = [
        "some text # raise Exception, 'text'",
        "raise ValueError('text') # raise Exception, 'text'",
        "raise ValueError('text')",
        "raise ValueError",
        "raise ValueError('text')",
        "raise ValueError('text') #,",
        # Talking about an exception in a docstring
        ''''"""This function will raise ValueError, except when it doesn't"""''',
        "raise (ValueError('text')",
    ]
    str_candidates_fail = [
        "raise 'exception'",
        "raise 'Exception'",
        'raise "exception"',
        'raise "Exception"',
        "raise 'ValueError'",
    ]
    gen_candidates_fail = [
        "raise Exception('text') # raise Exception, 'text'",
        "raise Exception('text')",
        "raise Exception",
        "raise Exception('text')",
        "raise Exception('text') #,",
        "raise Exception, 'text'",
        "raise Exception, 'text' # raise Exception('text')",
        "raise Exception, 'text' # raise Exception, 'text'",
        ">>> raise Exception, 'text'",
        ">>> raise Exception, 'text' # raise Exception('text')",
        ">>> raise Exception, 'text' # raise Exception, 'text'",
    ]
    old_candidates_fail = [
        "raise Exception, 'text'",
        "raise Exception, 'text' # raise Exception('text')",
        "raise Exception, 'text' # raise Exception, 'text'",
        ">>> raise Exception, 'text'",
        ">>> raise Exception, 'text' # raise Exception('text')",
        ">>> raise Exception, 'text' # raise Exception, 'text'",
        "raise ValueError, 'text'",
        "raise ValueError, 'text' # raise Exception('text')",
        "raise ValueError, 'text' # raise Exception, 'text'",
        ">>> raise ValueError, 'text'",
        ">>> raise ValueError, 'text' # raise Exception('text')",
        ">>> raise ValueError, 'text' # raise Exception, 'text'",
        "raise(ValueError,",
        "raise (ValueError,",
        "raise( ValueError,",
        "raise ( ValueError,",
        "raise(ValueError ,",
        "raise (ValueError ,",
        "raise( ValueError ,",
        "raise ( ValueError ,",
    ]

    for c in candidates_ok:
        assert str_raise_re.search(_with_space(c)) is None, c
        assert gen_raise_re.search(_with_space(c)) is None, c
        assert old_raise_re.search(_with_space(c)) is None, c
    for c in str_candidates_fail:
        assert str_raise_re.search(_with_space(c)) is not None, c
    for c in gen_candidates_fail:
        assert gen_raise_re.search(_with_space(c)) is not None, c
    for c in old_candidates_fail:
        assert old_raise_re.search(_with_space(c)) is not None, c


def test_implicit_imports_regular_expression():
    candidates_ok = [
        "from sympy import something",
        ">>> from sympy import something",
        "from sympy.somewhere import something",
        ">>> from sympy.somewhere import something",
        "import sympy",
        ">>> import sympy",
        "import sympy.something.something",
        "... import sympy",
        "... import sympy.something.something",
        "... from sympy import something",
        "... from sympy.somewhere import something",
        ">> from sympy import *",  # To allow 'fake' docstrings
        "# from sympy import *",
        "some text # from sympy import *",
    ]
    candidates_fail = [
        "from sympy import *",
        ">>> from sympy import *",
        "from sympy.somewhere import *",
        ">>> from sympy.somewhere import *",
        "... from sympy import *",
        "... from sympy.somewhere import *",
    ]
    for c in candidates_ok:
        assert implicit_test_re.search(_with_space(c)) is None, c
    for c in candidates_fail:
        assert implicit_test_re.search(_with_space(c)) is not None, c


def test_test_suite_defs():
    candidates_ok = [
        "    def foo():\n",
        "def foo(arg):\n",
        "def _foo():\n",
        "def test_foo():\n",
    ]
    candidates_fail = [
        "def foo():\n",
        "def foo() :\n",
        "def foo( ):\n",
        "def  foo():\n",
    ]
    for c in candidates_ok:
        assert test_suite_def_re.search(c) is None, c
    for c in candidates_fail:
        assert test_suite_def_re.search(c) is not None, c


def test_test_duplicate_defs():
    candidates_ok = [
        "def foo():\ndef foo():\n",
        "def test():\ndef test_():\n",
        "def test_():\ndef test__():\n",
    ]
    candidates_fail = [
        "def test_():\ndef test_ ():\n",
        "def test_1():\ndef  test_1():\n",
    ]
    ok = (None, 'check')
    def check(file):
        tests = 0
        test_set = set()
        for idx, line in enumerate(file.splitlines()):
            if test_ok_def_re.match(line):
                tests += 1
                test_set.add(line[3:].split('(')[0].strip())
                if len(test_set) != tests:
                    return False, message_duplicate_test % ('check', idx + 1)
        return None, 'check'
    for c in candidates_ok:
        assert check(c) == ok
    for c in candidates_fail:
        assert check(c) != ok


def test_find_self_assignments():
    candidates_ok = [
        "class A(object):\n    def foo(self, arg): arg = self\n",
        "class A(object):\n    def foo(self, arg): self.prop = arg\n",
        "class A(object):\n    def foo(self, arg): obj, obj2 = arg, self\n",
        "class A(object):\n    @classmethod\n    def bar(cls, arg): arg = cls\n",
        "class A(object):\n    def foo(var, arg): arg = var\n",
    ]
    candidates_fail = [
        "class A(object):\n    def foo(self, arg): self = arg\n",
        "class A(object):\n    def foo(self, arg): obj, self = arg, arg\n",
        "class A(object):\n    def foo(self, arg):\n        if arg: self = arg",
        "class A(object):\n    @classmethod\n    def foo(cls, arg): cls = arg\n",
        "class A(object):\n    def foo(var, arg): var = arg\n",
    ]

    for c in candidates_ok:
        assert find_self_assignments(c) == []
    for c in candidates_fail:
        assert find_self_assignments(c) != []


def test_test_unicode_encoding():
    unicode_whitelist = ['foo']
    unicode_strict_whitelist = ['bar']

    fname = 'abc'
    test_file = ['α']
    raises(AssertionError, lambda: _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist))

    fname = 'abc'
    test_file = ['abc']
    _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist)

    fname = 'foo'
    test_file = ['abc']
    raises(AssertionError, lambda: _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist))

    fname = 'bar'
    test_file = ['abc']
    _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_functions.py ===
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.physics.vector import Dyadic, Point, ReferenceFrame, Vector
from sympy.physics.vector.functions import (cross, dot, express,
                                            time_derivative,
                                            kinematic_equations, outer,
                                            partial_velocity,
                                            get_motion_params, dynamicsymbols)
from sympy.simplify import trigsimp
from sympy.testing.pytest import raises

q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
N = ReferenceFrame('N')
A = N.orientnew('A', 'Axis', [q1, N.z])
B = A.orientnew('B', 'Axis', [q2, A.x])
C = B.orientnew('C', 'Axis', [q3, B.y])


def test_dot():
    assert dot(A.x, A.x) == 1
    assert dot(A.x, A.y) == 0
    assert dot(A.x, A.z) == 0

    assert dot(A.y, A.x) == 0
    assert dot(A.y, A.y) == 1
    assert dot(A.y, A.z) == 0

    assert dot(A.z, A.x) == 0
    assert dot(A.z, A.y) == 0
    assert dot(A.z, A.z) == 1


def test_dot_different_frames():
    assert dot(N.x, A.x) == cos(q1)
    assert dot(N.x, A.y) == -sin(q1)
    assert dot(N.x, A.z) == 0
    assert dot(N.y, A.x) == sin(q1)
    assert dot(N.y, A.y) == cos(q1)
    assert dot(N.y, A.z) == 0
    assert dot(N.z, A.x) == 0
    assert dot(N.z, A.y) == 0
    assert dot(N.z, A.z) == 1

    assert trigsimp(dot(N.x, A.x + A.y)) == sqrt(2)*cos(q1 + pi/4)
    assert trigsimp(dot(N.x, A.x + A.y)) == trigsimp(dot(A.x + A.y, N.x))

    assert dot(A.x, C.x) == cos(q3)
    assert dot(A.x, C.y) == 0
    assert dot(A.x, C.z) == sin(q3)
    assert dot(A.y, C.x) == sin(q2)*sin(q3)
    assert dot(A.y, C.y) == cos(q2)
    assert dot(A.y, C.z) == -sin(q2)*cos(q3)
    assert dot(A.z, C.x) == -cos(q2)*sin(q3)
    assert dot(A.z, C.y) == sin(q2)
    assert dot(A.z, C.z) == cos(q2)*cos(q3)


def test_cross():
    assert cross(A.x, A.x) == 0
    assert cross(A.x, A.y) == A.z
    assert cross(A.x, A.z) == -A.y

    assert cross(A.y, A.x) == -A.z
    assert cross(A.y, A.y) == 0
    assert cross(A.y, A.z) == A.x

    assert cross(A.z, A.x) == A.y
    assert cross(A.z, A.y) == -A.x
    assert cross(A.z, A.z) == 0


def test_cross_different_frames():
    assert cross(N.x, A.x) == sin(q1)*A.z
    assert cross(N.x, A.y) == cos(q1)*A.z
    assert cross(N.x, A.z) == -sin(q1)*A.x - cos(q1)*A.y
    assert cross(N.y, A.x) == -cos(q1)*A.z
    assert cross(N.y, A.y) == sin(q1)*A.z
    assert cross(N.y, A.z) == cos(q1)*A.x - sin(q1)*A.y
    assert cross(N.z, A.x) == A.y
    assert cross(N.z, A.y) == -A.x
    assert cross(N.z, A.z) == 0

    assert cross(N.x, A.x) == sin(q1)*A.z
    assert cross(N.x, A.y) == cos(q1)*A.z
    assert cross(N.x, A.x + A.y) == sin(q1)*A.z + cos(q1)*A.z
    assert cross(A.x + A.y, N.x) == -sin(q1)*A.z - cos(q1)*A.z

    assert cross(A.x, C.x) == sin(q3)*C.y
    assert cross(A.x, C.y) == -sin(q3)*C.x + cos(q3)*C.z
    assert cross(A.x, C.z) == -cos(q3)*C.y
    assert cross(C.x, A.x) == -sin(q3)*C.y
    assert cross(C.y, A.x).express(C).simplify() == sin(q3)*C.x - cos(q3)*C.z
    assert cross(C.z, A.x) == cos(q3)*C.y

def test_operator_match():
    """Test that the output of dot, cross, outer functions match
    operator behavior.
    """
    A = ReferenceFrame('A')
    v = A.x + A.y
    d = v | v
    zerov = Vector(0)
    zerod = Dyadic(0)

    # dot products
    assert d & d == dot(d, d)
    assert d & zerod == dot(d, zerod)
    assert zerod & d == dot(zerod, d)
    assert d & v == dot(d, v)
    assert v & d == dot(v, d)
    assert d & zerov == dot(d, zerov)
    assert zerov & d == dot(zerov, d)
    raises(TypeError, lambda: dot(d, S.Zero))
    raises(TypeError, lambda: dot(S.Zero, d))
    raises(TypeError, lambda: dot(d, 0))
    raises(TypeError, lambda: dot(0, d))
    assert v & v == dot(v, v)
    assert v & zerov == dot(v, zerov)
    assert zerov & v == dot(zerov, v)
    raises(TypeError, lambda: dot(v, S.Zero))
    raises(TypeError, lambda: dot(S.Zero, v))
    raises(TypeError, lambda: dot(v, 0))
    raises(TypeError, lambda: dot(0, v))

    # cross products
    raises(TypeError, lambda: cross(d, d))
    raises(TypeError, lambda: cross(d, zerod))
    raises(TypeError, lambda: cross(zerod, d))
    assert d ^ v == cross(d, v)
    assert v ^ d == cross(v, d)
    assert d ^ zerov == cross(d, zerov)
    assert zerov ^ d == cross(zerov, d)
    assert zerov ^ d == cross(zerov, d)
    raises(TypeError, lambda: cross(d, S.Zero))
    raises(TypeError, lambda: cross(S.Zero, d))
    raises(TypeError, lambda: cross(d, 0))
    raises(TypeError, lambda: cross(0, d))
    assert v ^ v == cross(v, v)
    assert v ^ zerov == cross(v, zerov)
    assert zerov ^ v == cross(zerov, v)
    raises(TypeError, lambda: cross(v, S.Zero))
    raises(TypeError, lambda: cross(S.Zero, v))
    raises(TypeError, lambda: cross(v, 0))
    raises(TypeError, lambda: cross(0, v))

    # outer products
    raises(TypeError, lambda: outer(d, d))
    raises(TypeError, lambda: outer(d, zerod))
    raises(TypeError, lambda: outer(zerod, d))
    raises(TypeError, lambda: outer(d, v))
    raises(TypeError, lambda: outer(v, d))
    raises(TypeError, lambda: outer(d, zerov))
    raises(TypeError, lambda: outer(zerov, d))
    raises(TypeError, lambda: outer(zerov, d))
    raises(TypeError, lambda: outer(d, S.Zero))
    raises(TypeError, lambda: outer(S.Zero, d))
    raises(TypeError, lambda: outer(d, 0))
    raises(TypeError, lambda: outer(0, d))
    assert v | v == outer(v, v)
    assert v | zerov == outer(v, zerov)
    assert zerov | v == outer(zerov, v)
    raises(TypeError, lambda: outer(v, S.Zero))
    raises(TypeError, lambda: outer(S.Zero, v))
    raises(TypeError, lambda: outer(v, 0))
    raises(TypeError, lambda: outer(0, v))


def test_express():
    assert express(Vector(0), N) == Vector(0)
    assert express(S.Zero, N) is S.Zero
    assert express(A.x, C) == cos(q3)*C.x + sin(q3)*C.z
    assert express(A.y, C) == sin(q2)*sin(q3)*C.x + cos(q2)*C.y - \
        sin(q2)*cos(q3)*C.z
    assert express(A.z, C) == -sin(q3)*cos(q2)*C.x + sin(q2)*C.y + \
        cos(q2)*cos(q3)*C.z
    assert express(A.x, N) == cos(q1)*N.x + sin(q1)*N.y
    assert express(A.y, N) == -sin(q1)*N.x + cos(q1)*N.y
    assert express(A.z, N) == N.z
    assert express(A.x, A) == A.x
    assert express(A.y, A) == A.y
    assert express(A.z, A) == A.z
    assert express(A.x, B) == B.x
    assert express(A.y, B) == cos(q2)*B.y - sin(q2)*B.z
    assert express(A.z, B) == sin(q2)*B.y + cos(q2)*B.z
    assert express(A.x, C) == cos(q3)*C.x + sin(q3)*C.z
    assert express(A.y, C) == sin(q2)*sin(q3)*C.x + cos(q2)*C.y - \
        sin(q2)*cos(q3)*C.z
    assert express(A.z, C) == -sin(q3)*cos(q2)*C.x + sin(q2)*C.y + \
        cos(q2)*cos(q3)*C.z
    # Check to make sure UnitVectors get converted properly
    assert express(N.x, N) == N.x
    assert express(N.y, N) == N.y
    assert express(N.z, N) == N.z
    assert express(N.x, A) == (cos(q1)*A.x - sin(q1)*A.y)
    assert express(N.y, A) == (sin(q1)*A.x + cos(q1)*A.y)
    assert express(N.z, A) == A.z
    assert express(N.x, B) == (cos(q1)*B.x - sin(q1)*cos(q2)*B.y +
            sin(q1)*sin(q2)*B.z)
    assert express(N.y, B) == (sin(q1)*B.x + cos(q1)*cos(q2)*B.y -
            sin(q2)*cos(q1)*B.z)
    assert express(N.z, B) == (sin(q2)*B.y + cos(q2)*B.z)
    assert express(N.x, C) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*C.x -
        sin(q1)*cos(q2)*C.y +
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*C.z)
    assert express(N.y, C) == (
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.x +
        cos(q1)*cos(q2)*C.y +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.z)
    assert express(N.z, C) == (-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z)

    assert express(A.x, N) == (cos(q1)*N.x + sin(q1)*N.y)
    assert express(A.y, N) == (-sin(q1)*N.x + cos(q1)*N.y)
    assert express(A.z, N) == N.z
    assert express(A.x, A) == A.x
    assert express(A.y, A) == A.y
    assert express(A.z, A) == A.z
    assert express(A.x, B) == B.x
    assert express(A.y, B) == (cos(q2)*B.y - sin(q2)*B.z)
    assert express(A.z, B) == (sin(q2)*B.y + cos(q2)*B.z)
    assert express(A.x, C) == (cos(q3)*C.x + sin(q3)*C.z)
    assert express(A.y, C) == (sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
            sin(q2)*cos(q3)*C.z)
    assert express(A.z, C) == (-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z)

    assert express(B.x, N) == (cos(q1)*N.x + sin(q1)*N.y)
    assert express(B.y, N) == (-sin(q1)*cos(q2)*N.x +
            cos(q1)*cos(q2)*N.y + sin(q2)*N.z)
    assert express(B.z, N) == (sin(q1)*sin(q2)*N.x -
            sin(q2)*cos(q1)*N.y + cos(q2)*N.z)
    assert express(B.x, A) == A.x
    assert express(B.y, A) == (cos(q2)*A.y + sin(q2)*A.z)
    assert express(B.z, A) == (-sin(q2)*A.y + cos(q2)*A.z)
    assert express(B.x, B) == B.x
    assert express(B.y, B) == B.y
    assert express(B.z, B) == B.z
    assert express(B.x, C) == (cos(q3)*C.x + sin(q3)*C.z)
    assert express(B.y, C) == C.y
    assert express(B.z, C) == (-sin(q3)*C.x + cos(q3)*C.z)

    assert express(C.x, N) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*N.x +
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*N.y -
        sin(q3)*cos(q2)*N.z)
    assert express(C.y, N) == (
        -sin(q1)*cos(q2)*N.x + cos(q1)*cos(q2)*N.y + sin(q2)*N.z)
    assert express(C.z, N) == (
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*N.x +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*N.y +
        cos(q2)*cos(q3)*N.z)
    assert express(C.x, A) == (cos(q3)*A.x + sin(q2)*sin(q3)*A.y -
            sin(q3)*cos(q2)*A.z)
    assert express(C.y, A) == (cos(q2)*A.y + sin(q2)*A.z)
    assert express(C.z, A) == (sin(q3)*A.x - sin(q2)*cos(q3)*A.y +
            cos(q2)*cos(q3)*A.z)
    assert express(C.x, B) == (cos(q3)*B.x - sin(q3)*B.z)
    assert express(C.y, B) == B.y
    assert express(C.z, B) == (sin(q3)*B.x + cos(q3)*B.z)
    assert express(C.x, C) == C.x
    assert express(C.y, C) == C.y
    assert express(C.z, C) == C.z == (C.z)

    #  Check to make sure Vectors get converted back to UnitVectors
    assert N.x == express((cos(q1)*A.x - sin(q1)*A.y), N).simplify()
    assert N.y == express((sin(q1)*A.x + cos(q1)*A.y), N).simplify()
    assert N.x == express((cos(q1)*B.x - sin(q1)*cos(q2)*B.y +
            sin(q1)*sin(q2)*B.z), N).simplify()
    assert N.y == express((sin(q1)*B.x + cos(q1)*cos(q2)*B.y -
        sin(q2)*cos(q1)*B.z), N).simplify()
    assert N.z == express((sin(q2)*B.y + cos(q2)*B.z), N).simplify()

    """
    These don't really test our code, they instead test the auto simplification
    (or lack thereof) of SymPy.
    assert N.x == express((
            (cos(q1)*cos(q3)-sin(q1)*sin(q2)*sin(q3))*C.x -
            sin(q1)*cos(q2)*C.y +
            (sin(q3)*cos(q1)+sin(q1)*sin(q2)*cos(q3))*C.z), N)
    assert N.y == express((
            (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.x +
            cos(q1)*cos(q2)*C.y +
            (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.z), N)
    assert N.z == express((-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z), N)
    """

    assert A.x == express((cos(q1)*N.x + sin(q1)*N.y), A).simplify()
    assert A.y == express((-sin(q1)*N.x + cos(q1)*N.y), A).simplify()

    assert A.y == express((cos(q2)*B.y - sin(q2)*B.z), A).simplify()
    assert A.z == express((sin(q2)*B.y + cos(q2)*B.z), A).simplify()

    assert A.x == express((cos(q3)*C.x + sin(q3)*C.z), A).simplify()

    # Tripsimp messes up here too.
    #print express((sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
    #        sin(q2)*cos(q3)*C.z), A)
    assert A.y == express((sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
            sin(q2)*cos(q3)*C.z), A).simplify()

    assert A.z == express((-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z), A).simplify()
    assert B.x == express((cos(q1)*N.x + sin(q1)*N.y), B).simplify()
    assert B.y == express((-sin(q1)*cos(q2)*N.x +
            cos(q1)*cos(q2)*N.y + sin(q2)*N.z), B).simplify()

    assert B.z == express((sin(q1)*sin(q2)*N.x -
            sin(q2)*cos(q1)*N.y + cos(q2)*N.z), B).simplify()

    assert B.y == express((cos(q2)*A.y + sin(q2)*A.z), B).simplify()
    assert B.z == express((-sin(q2)*A.y + cos(q2)*A.z), B).simplify()
    assert B.x == express((cos(q3)*C.x + sin(q3)*C.z), B).simplify()
    assert B.z == express((-sin(q3)*C.x + cos(q3)*C.z), B).simplify()

    """
    assert C.x == express((
            (cos(q1)*cos(q3)-sin(q1)*sin(q2)*sin(q3))*N.x +
            (sin(q1)*cos(q3)+sin(q2)*sin(q3)*cos(q1))*N.y -
                sin(q3)*cos(q2)*N.z), C)
    assert C.y == express((
            -sin(q1)*cos(q2)*N.x + cos(q1)*cos(q2)*N.y + sin(q2)*N.z), C)
    assert C.z == express((
            (sin(q3)*cos(q1)+sin(q1)*sin(q2)*cos(q3))*N.x +
            (sin(q1)*sin(q3)-sin(q2)*cos(q1)*cos(q3))*N.y +
            cos(q2)*cos(q3)*N.z), C)
    """
    assert C.x == express((cos(q3)*A.x + sin(q2)*sin(q3)*A.y -
            sin(q3)*cos(q2)*A.z), C).simplify()
    assert C.y == express((cos(q2)*A.y + sin(q2)*A.z), C).simplify()
    assert C.z == express((sin(q3)*A.x - sin(q2)*cos(q3)*A.y +
            cos(q2)*cos(q3)*A.z), C).simplify()
    assert C.x == express((cos(q3)*B.x - sin(q3)*B.z), C).simplify()
    assert C.z == express((sin(q3)*B.x + cos(q3)*B.z), C).simplify()


def test_time_derivative():
    #The use of time_derivative for calculations pertaining to scalar
    #fields has been tested in test_coordinate_vars in test_essential.py
    A = ReferenceFrame('A')
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    B = A.orientnew('B', 'Axis', [q, A.z])
    d = A.x | A.x
    assert time_derivative(d, B) == (-qd) * (A.y | A.x) + \
           (-qd) * (A.x | A.y)
    d1 = A.x | B.y
    assert time_derivative(d1, A) == - qd*(A.x|B.x)
    assert time_derivative(d1, B) == - qd*(A.y|B.y)
    d2 = A.x | B.x
    assert time_derivative(d2, A) == qd*(A.x|B.y)
    assert time_derivative(d2, B) == - qd*(A.y|B.x)
    d3 = A.x | B.z
    assert time_derivative(d3, A) == 0
    assert time_derivative(d3, B) == - qd*(A.y|B.z)
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    q1dd, q2dd, q3dd, q4dd = dynamicsymbols('q1 q2 q3 q4', 2)
    C = B.orientnew('C', 'Axis', [q4, B.x])
    v1 = q1 * A.z
    v2 = q2*A.x + q3*B.y
    v3 = q1*A.x + q2*A.y + q3*A.z
    assert time_derivative(B.x, C) == 0
    assert time_derivative(B.y, C) == - q4d*B.z
    assert time_derivative(B.z, C) == q4d*B.y
    assert time_derivative(v1, B) == q1d*A.z
    assert time_derivative(v1, C) == - q1*sin(q)*q4d*A.x + \
           q1*cos(q)*q4d*A.y + q1d*A.z
    assert time_derivative(v2, A) == q2d*A.x - q3*qd*B.x + q3d*B.y
    assert time_derivative(v2, C) == q2d*A.x - q2*qd*A.y + \
           q2*sin(q)*q4d*A.z + q3d*B.y - q3*q4d*B.z
    assert time_derivative(v3, B) == (q2*qd + q1d)*A.x + \
           (-q1*qd + q2d)*A.y + q3d*A.z
    assert time_derivative(d, C) == - qd*(A.y|A.x) + \
           sin(q)*q4d*(A.z|A.x) - qd*(A.x|A.y) + sin(q)*q4d*(A.x|A.z)
    raises(ValueError, lambda: time_derivative(B.x, C, order=0.5))
    raises(ValueError, lambda: time_derivative(B.x, C, order=-1))


def test_get_motion_methods():
    #Initialization
    t = dynamicsymbols._t
    s1, s2, s3 = symbols('s1 s2 s3')
    S1, S2, S3 = symbols('S1 S2 S3')
    S4, S5, S6 = symbols('S4 S5 S6')
    t1, t2 = symbols('t1 t2')
    a, b, c = dynamicsymbols('a b c')
    ad, bd, cd = dynamicsymbols('a b c', 1)
    a2d, b2d, c2d = dynamicsymbols('a b c', 2)
    v0 = S1*N.x + S2*N.y + S3*N.z
    v01 = S4*N.x + S5*N.y + S6*N.z
    v1 = s1*N.x + s2*N.y + s3*N.z
    v2 = a*N.x + b*N.y + c*N.z
    v2d = ad*N.x + bd*N.y + cd*N.z
    v2dd = a2d*N.x + b2d*N.y + c2d*N.z
    #Test position parameter
    assert get_motion_params(frame = N) == (0, 0, 0)
    assert get_motion_params(N, position=v1) == (0, 0, v1)
    assert get_motion_params(N, position=v2) == (v2dd, v2d, v2)
    #Test velocity parameter
    assert get_motion_params(N, velocity=v1) == (0, v1, v1 * t)
    assert get_motion_params(N, velocity=v1, position=v0, timevalue1=t1) == \
           (0, v1, v0 + v1*(t - t1))
    answer = get_motion_params(N, velocity=v1, position=v2, timevalue1=t1)
    answer_expected = (0, v1, v1*t - v1*t1 + v2.subs(t, t1))
    assert answer == answer_expected

    answer = get_motion_params(N, velocity=v2, position=v0, timevalue1=t1)
    integral_vector = Integral(a, (t, t1, t))*N.x + Integral(b, (t, t1, t))*N.y \
            + Integral(c, (t, t1, t))*N.z
    answer_expected = (v2d, v2, v0 + integral_vector)
    assert answer == answer_expected

    #Test acceleration parameter
    assert get_motion_params(N, acceleration=v1) == \
           (v1, v1 * t, v1 * t**2/2)
    assert get_motion_params(N, acceleration=v1, velocity=v0,
                          position=v2, timevalue1=t1, timevalue2=t2) == \
           (v1, (v0 + v1*t - v1*t2),
            -v0*t1 + v1*t**2/2 + v1*t2*t1 - \
            v1*t1**2/2 + t*(v0 - v1*t2) + \
            v2.subs(t, t1))
    assert get_motion_params(N, acceleration=v1, velocity=v0,
                             position=v01, timevalue1=t1, timevalue2=t2) == \
           (v1, v0 + v1*t - v1*t2,
            -v0*t1 + v01 + v1*t**2/2 + \
            v1*t2*t1 - v1*t1**2/2 + \
            t*(v0 - v1*t2))
    answer = get_motion_params(N, acceleration=a*N.x, velocity=S1*N.x,
                          position=S2*N.x, timevalue1=t1, timevalue2=t2)
    i1 = Integral(a, (t, t2, t))
    answer_expected = (a*N.x, (S1 + i1)*N.x, \
        (S2 + Integral(S1 + i1, (t, t1, t)))*N.x)
    assert answer == answer_expected


def test_kin_eqs():
    q0, q1, q2, q3 = dynamicsymbols('q0 q1 q2 q3')
    q0d, q1d, q2d, q3d = dynamicsymbols('q0 q1 q2 q3', 1)
    u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    ke = kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', 313)
    assert ke == kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', '313')
    kds = kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'quaternion')
    assert kds == [-0.5 * q0 * u1 - 0.5 * q2 * u3 + 0.5 * q3 * u2 + q1d,
            -0.5 * q0 * u2 + 0.5 * q1 * u3 - 0.5 * q3 * u1 + q2d,
            -0.5 * q0 * u3 - 0.5 * q1 * u2 + 0.5 * q2 * u1 + q3d,
            0.5 * q1 * u1 + 0.5 * q2 * u2 + 0.5 * q3 * u3 + q0d]
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2], 'quaternion'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'quaternion', '123'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'foo'))
    raises(TypeError, lambda: kinematic_equations(u1, [q0, q1, q2, q3], 'quaternion'))
    raises(TypeError, lambda: kinematic_equations([u1], [q0, q1, q2, q3], 'quaternion'))
    raises(TypeError, lambda: kinematic_equations([u1, u2, u3], q0, 'quaternion'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'body'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'space'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2], 'body', '222'))
    assert kinematic_equations([0, 0, 0], [q0, q1, q2], 'space') == [S.Zero, S.Zero, S.Zero]


def test_partial_velocity():
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    u4, u5 = dynamicsymbols('u4, u5')
    r = symbols('r')

    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)

    C = Point('C')
    C.set_vel(N, u4 * L.x + u5 * (Y.z ^ L.x))
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)

    vel_list = [Dmc.vel(N), C.vel(N), R.ang_vel_in(N)]
    u_list = [u1, u2, u3, u4, u5]
    assert (partial_velocity(vel_list, u_list, N) ==
            [[- r*L.y, r*L.x, 0, L.x, cos(q2)*L.y - sin(q2)*L.z],
            [0, 0, 0, L.x, cos(q2)*L.y - sin(q2)*L.z],
            [L.x, L.y, L.z, 0, 0]])

    # Make sure that partial velocities can be computed regardless if the
    # orientation between frames is defined or not.
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    v = u4 * A.x + u5 * B.y
    assert partial_velocity((v, ), (u4, u5), A) == [[A.x, B.y]]

    raises(TypeError, lambda: partial_velocity(Dmc.vel(N), u_list, N))
    raises(TypeError, lambda: partial_velocity(vel_list, u1, N))

def test_dynamicsymbols():
    #Tests to check the assumptions applied to dynamicsymbols
    f1 = dynamicsymbols('f1')
    f2 = dynamicsymbols('f2', real=True)
    f3 = dynamicsymbols('f3', positive=True)
    f4, f5 = dynamicsymbols('f4,f5', commutative=False)
    f6 = dynamicsymbols('f6', integer=True)
    assert f1.is_real is None
    assert f2.is_real
    assert f3.is_positive
    assert f4*f5 != f5*f4
    assert f6.is_integer

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_finite_rv.py ===
from sympy.concrete.summations import Sum
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.beta_functions import beta
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix
from sympy.stats import (DiscreteUniform, Die, Bernoulli, Coin, Binomial, BetaBinomial,
                         Hypergeometric, Rademacher, IdealSoliton, RobustSoliton, P, E, variance,
                         covariance, skewness, density, where, FiniteRV, pspace, cdf,
                         correlation, moment, cmoment, smoment, characteristic_function,
                         moment_generating_function, quantile,  kurtosis, median, coskewness)
from sympy.stats.frv_types import DieDistribution, BinomialDistribution, \
    HypergeometricDistribution
from sympy.stats.rv import Density
from sympy.testing.pytest import raises


def BayesTest(A, B):
    assert P(A, B) == P(And(A, B)) / P(B)
    assert P(A, B) == P(B, A) * P(A) / P(B)


def test_discreteuniform():
    # Symbolic
    a, b, c, t = symbols('a b c t')
    X = DiscreteUniform('X', [a, b, c])

    assert E(X) == (a + b + c)/3
    assert simplify(variance(X)
                    - ((a**2 + b**2 + c**2)/3 - (a/3 + b/3 + c/3)**2)) == 0
    assert P(Eq(X, a)) == P(Eq(X, b)) == P(Eq(X, c)) == S('1/3')

    Y = DiscreteUniform('Y', range(-5, 5))

    # Numeric
    assert E(Y) == S('-1/2')
    assert variance(Y) == S('33/4')
    assert median(Y) == FiniteSet(-1, 0)

    for x in range(-5, 5):
        assert P(Eq(Y, x)) == S('1/10')
        assert P(Y <= x) == S(x + 6)/10
        assert P(Y >= x) == S(5 - x)/10

    assert dict(density(Die('D', 6)).items()) == \
           dict(density(DiscreteUniform('U', range(1, 7))).items())

    assert characteristic_function(X)(t) == exp(I*a*t)/3 + exp(I*b*t)/3 + exp(I*c*t)/3
    assert moment_generating_function(X)(t) == exp(a*t)/3 + exp(b*t)/3 + exp(c*t)/3
    # issue 18611
    raises(ValueError, lambda: DiscreteUniform('Z', [a, a, a, b, b, c]))

def test_dice():
    # TODO: Make iid method!
    X, Y, Z = Die('X', 6), Die('Y', 6), Die('Z', 6)
    a, b, t, p = symbols('a b t p')

    assert E(X) == 3 + S.Half
    assert variance(X) == Rational(35, 12)
    assert E(X + Y) == 7
    assert E(X + X) == 7
    assert E(a*X + b) == a*E(X) + b
    assert variance(X + Y) == variance(X) + variance(Y) == cmoment(X + Y, 2)
    assert variance(X + X) == 4 * variance(X) == cmoment(X + X, 2)
    assert cmoment(X, 0) == 1
    assert cmoment(4*X, 3) == 64*cmoment(X, 3)
    assert covariance(X, Y) is S.Zero
    assert covariance(X, X + Y) == variance(X)
    assert density(Eq(cos(X*S.Pi), 1))[True] == S.Half
    assert correlation(X, Y) == 0
    assert correlation(X, Y) == correlation(Y, X)
    assert smoment(X + Y, 3) == skewness(X + Y)
    assert smoment(X + Y, 4) == kurtosis(X + Y)
    assert smoment(X, 0) == 1
    assert P(X > 3) == S.Half
    assert P(2*X > 6) == S.Half
    assert P(X > Y) == Rational(5, 12)
    assert P(Eq(X, Y)) == P(Eq(X, 1))

    assert E(X, X > 3) == 5 == moment(X, 1, 0, X > 3)
    assert E(X, Y > 3) == E(X) == moment(X, 1, 0, Y > 3)
    assert E(X + Y, Eq(X, Y)) == E(2*X)
    assert moment(X, 0) == 1
    assert moment(5*X, 2) == 25*moment(X, 2)
    assert quantile(X)(p) == Piecewise((nan, (p > 1) | (p < 0)),\
        (S.One, p <= Rational(1, 6)), (S(2), p <= Rational(1, 3)), (S(3), p <= S.Half),\
        (S(4), p <= Rational(2, 3)), (S(5), p <= Rational(5, 6)), (S(6), p <= 1))

    assert P(X > 3, X > 3) is S.One
    assert P(X > Y, Eq(Y, 6)) is S.Zero
    assert P(Eq(X + Y, 12)) == Rational(1, 36)
    assert P(Eq(X + Y, 12), Eq(X, 6)) == Rational(1, 6)

    assert density(X + Y) == density(Y + Z) != density(X + X)
    d = density(2*X + Y**Z)
    assert d[S(22)] == Rational(1, 108) and d[S(4100)] == Rational(1, 216) and S(3130) not in d

    assert pspace(X).domain.as_boolean() == Or(
        *[Eq(X.symbol, i) for i in [1, 2, 3, 4, 5, 6]])

    assert where(X > 3).set == FiniteSet(4, 5, 6)

    assert characteristic_function(X)(t) == exp(6*I*t)/6 + exp(5*I*t)/6 + exp(4*I*t)/6 + exp(3*I*t)/6 + exp(2*I*t)/6 + exp(I*t)/6
    assert moment_generating_function(X)(t) == exp(6*t)/6 + exp(5*t)/6 + exp(4*t)/6 + exp(3*t)/6 + exp(2*t)/6 + exp(t)/6
    assert median(X) == FiniteSet(3, 4)
    D = Die('D', 7)
    assert median(D) == FiniteSet(4)
    # Bayes test for die
    BayesTest(X > 3, X + Y < 5)
    BayesTest(Eq(X - Y, Z), Z > Y)
    BayesTest(X > 3, X > 2)

    # arg test for die
    raises(ValueError, lambda: Die('X', -1))  # issue 8105: negative sides.
    raises(ValueError, lambda: Die('X', 0))
    raises(ValueError, lambda: Die('X', 1.5))  # issue 8103: non integer sides.

    # symbolic test for die
    n, k = symbols('n, k', positive=True)
    D = Die('D', n)
    dens = density(D).dict
    assert dens == Density(DieDistribution(n))
    assert set(dens.subs(n, 4).doit().keys()) == {1, 2, 3, 4}
    assert set(dens.subs(n, 4).doit().values()) == {Rational(1, 4)}
    k = Dummy('k', integer=True)
    assert E(D).dummy_eq(
        Sum(Piecewise((k/n, k <= n), (0, True)), (k, 1, n)))
    assert variance(D).subs(n, 6).doit() == Rational(35, 12)

    ki = Dummy('ki')
    cumuf = cdf(D)(k)
    assert cumuf.dummy_eq(
    Sum(Piecewise((1/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, k)))
    assert cumuf.subs({n: 6, k: 2}).doit() == Rational(1, 3)

    t = Dummy('t')
    cf = characteristic_function(D)(t)
    assert cf.dummy_eq(
    Sum(Piecewise((exp(ki*I*t)/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    assert cf.subs(n, 3).doit() == exp(3*I*t)/3 + exp(2*I*t)/3 + exp(I*t)/3
    mgf = moment_generating_function(D)(t)
    assert mgf.dummy_eq(
    Sum(Piecewise((exp(ki*t)/n, (ki >= 1) & (ki <= n)), (0, True)), (ki, 1, n)))
    assert mgf.subs(n, 3).doit() == exp(3*t)/3 + exp(2*t)/3 + exp(t)/3

def test_given():
    X = Die('X', 6)
    assert density(X, X > 5) == {S(6): S.One}
    assert where(X > 2, X > 5).as_boolean() == Eq(X.symbol, 6)


def test_domains():
    X, Y = Die('x', 6), Die('y', 6)
    x, y = X.symbol, Y.symbol
    # Domains
    d = where(X > Y)
    assert d.condition == (x > y)
    d = where(And(X > Y, Y > 3))
    assert d.as_boolean() == Or(And(Eq(x, 5), Eq(y, 4)), And(Eq(x, 6),
        Eq(y, 5)), And(Eq(x, 6), Eq(y, 4)))
    assert len(d.elements) == 3

    assert len(pspace(X + Y).domain.elements) == 36

    Z = Die('x', 4)

    raises(ValueError, lambda: P(X > Z))  # Two domains with same internal symbol

    assert pspace(X + Y).domain.set == FiniteSet(1, 2, 3, 4, 5, 6)**2

    assert where(X > 3).set == FiniteSet(4, 5, 6)
    assert X.pspace.domain.dict == FiniteSet(
        *[Dict({X.symbol: i}) for i in range(1, 7)])

    assert where(X > Y).dict == FiniteSet(*[Dict({X.symbol: i, Y.symbol: j})
            for i in range(1, 7) for j in range(1, 7) if i > j])

def test_bernoulli():
    p, a, b, t = symbols('p a b t')
    X = Bernoulli('B', p, a, b)

    assert E(X) == a*p + b*(-p + 1)
    assert density(X)[a] == p
    assert density(X)[b] == 1 - p
    assert characteristic_function(X)(t) == p * exp(I * a * t) + (-p + 1) * exp(I * b * t)
    assert moment_generating_function(X)(t) == p * exp(a * t) + (-p + 1) * exp(b * t)

    X = Bernoulli('B', p, 1, 0)
    z = Symbol("z")

    assert E(X) == p
    assert simplify(variance(X)) == p*(1 - p)
    assert E(a*X + b) == a*E(X) + b
    assert simplify(variance(a*X + b)) == simplify(a**2 * variance(X))
    assert quantile(X)(z) == Piecewise((nan, (z > 1) | (z < 0)), (0, z <= 1 - p), (1, z <= 1))
    Y = Bernoulli('Y', Rational(1, 2))
    assert median(Y) == FiniteSet(0, 1)
    Z = Bernoulli('Z', Rational(2, 3))
    assert median(Z) == FiniteSet(1)
    raises(ValueError, lambda: Bernoulli('B', 1.5))
    raises(ValueError, lambda: Bernoulli('B', -0.5))

    #issue 8248
    assert X.pspace.compute_expectation(1) == 1

    p = Rational(1, 5)
    X = Binomial('X', 5, p)
    Y = Binomial('Y', 7, 2*p)
    Z = Binomial('Z', 9, 3*p)
    assert coskewness(Y + Z, X + Y, X + Z).simplify() == 0
    assert coskewness(Y + 2*X + Z, X + 2*Y + Z, X + 2*Z + Y).simplify() == \
                        sqrt(1529)*Rational(12, 16819)
    assert coskewness(Y + 2*X + Z, X + 2*Y + Z, X + 2*Z + Y, X < 2).simplify() \
                        == -sqrt(357451121)*Rational(2812, 4646864573)

def test_cdf():
    D = Die('D', 6)
    o = S.One

    assert cdf(
        D) == sympify({1: o/6, 2: o/3, 3: o/2, 4: 2*o/3, 5: 5*o/6, 6: o})


def test_coins():
    C, D = Coin('C'), Coin('D')
    H, T = symbols('H, T')
    assert P(Eq(C, D)) == S.Half
    assert density(Tuple(C, D)) == {(H, H): Rational(1, 4), (H, T): Rational(1, 4),
            (T, H): Rational(1, 4), (T, T): Rational(1, 4)}
    assert dict(density(C).items()) == {H: S.Half, T: S.Half}

    F = Coin('F', Rational(1, 10))
    assert P(Eq(F, H)) == Rational(1, 10)

    d = pspace(C).domain

    assert d.as_boolean() == Or(Eq(C.symbol, H), Eq(C.symbol, T))

    raises(ValueError, lambda: P(C > D))  # Can't intelligently compare H to T

def test_binomial_verify_parameters():
    raises(ValueError, lambda: Binomial('b', .2, .5))
    raises(ValueError, lambda: Binomial('b', 3, 1.5))

def test_binomial_numeric():
    nvals = range(5)
    pvals = [0, Rational(1, 4), S.Half, Rational(3, 4), 1]

    for n in nvals:
        for p in pvals:
            X = Binomial('X', n, p)
            assert E(X) == n*p
            assert variance(X) == n*p*(1 - p)
            if n > 0 and 0 < p < 1:
                assert skewness(X) == (1 - 2*p)/sqrt(n*p*(1 - p))
                assert kurtosis(X) == 3 + (1 - 6*p*(1 - p))/(n*p*(1 - p))
            for k in range(n + 1):
                assert P(Eq(X, k)) == binomial(n, k)*p**k*(1 - p)**(n - k)

def test_binomial_quantile():
    X = Binomial('X', 50, S.Half)
    assert quantile(X)(0.95) == S(31)
    assert median(X) == FiniteSet(25)

    X = Binomial('X', 5, S.Half)
    p = Symbol("p", positive=True)
    assert quantile(X)(p) == Piecewise((nan, p > S.One), (S.Zero, p <= Rational(1, 32)),\
        (S.One, p <= Rational(3, 16)), (S(2), p <= S.Half), (S(3), p <= Rational(13, 16)),\
        (S(4), p <= Rational(31, 32)), (S(5), p <= S.One))
    assert median(X) == FiniteSet(2, 3)


def test_binomial_symbolic():
    n = 2
    p = symbols('p', positive=True)
    X = Binomial('X', n, p)
    t = Symbol('t')

    assert simplify(E(X)) == n*p == simplify(moment(X, 1))
    assert simplify(variance(X)) == n*p*(1 - p) == simplify(cmoment(X, 2))
    assert cancel(skewness(X) - (1 - 2*p)/sqrt(n*p*(1 - p))) == 0
    assert cancel((kurtosis(X)) - (3 + (1 - 6*p*(1 - p))/(n*p*(1 - p)))) == 0
    assert characteristic_function(X)(t) == p ** 2 * exp(2 * I * t) + 2 * p * (-p + 1) * exp(I * t) + (-p + 1) ** 2
    assert moment_generating_function(X)(t) == p ** 2 * exp(2 * t) + 2 * p * (-p + 1) * exp(t) + (-p + 1) ** 2

    # Test ability to change success/failure winnings
    H, T = symbols('H T')
    Y = Binomial('Y', n, p, succ=H, fail=T)
    assert simplify(E(Y) - (n*(H*p + T*(1 - p)))) == 0

    # test symbolic dimensions
    n = symbols('n')
    B = Binomial('B', n, p)
    raises(NotImplementedError, lambda: P(B > 2))
    assert density(B).dict == Density(BinomialDistribution(n, p, 1, 0))
    assert set(density(B).dict.subs(n, 4).doit().keys()) == \
    {S.Zero, S.One, S(2), S(3), S(4)}
    assert set(density(B).dict.subs(n, 4).doit().values()) == \
    {(1 - p)**4, 4*p*(1 - p)**3, 6*p**2*(1 - p)**2, 4*p**3*(1 - p), p**4}
    k = Dummy('k', integer=True)
    assert E(B > 2).dummy_eq(
        Sum(Piecewise((k*p**k*(1 - p)**(-k + n)*binomial(n, k), (k >= 0)
        & (k <= n) & (k > 2)), (0, True)), (k, 0, n)))

def test_beta_binomial():
    # verify parameters
    raises(ValueError, lambda: BetaBinomial('b', .2, 1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, -1, 2))
    raises(ValueError, lambda: BetaBinomial('b', 2, 1, -2))
    assert BetaBinomial('b', 2, 1, 1)

    # test numeric values
    nvals = range(1,5)
    alphavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]
    betavals = [Rational(1, 4), S.Half, Rational(3, 4), 1, 10]

    for n in nvals:
        for a in alphavals:
            for b in betavals:
                X = BetaBinomial('X', n, a, b)
                assert E(X) == moment(X, 1)
                assert variance(X) == cmoment(X, 2)

    # test symbolic
    n, a, b = symbols('a b n')
    assert BetaBinomial('x', n, a, b)
    n = 2 # Because we're using for loops, can't do symbolic n
    a, b = symbols('a b', positive=True)
    X = BetaBinomial('X', n, a, b)
    t = Symbol('t')

    assert E(X).expand() == moment(X, 1).expand()
    assert variance(X).expand() == cmoment(X, 2).expand()
    assert skewness(X) == smoment(X, 3)
    assert characteristic_function(X)(t) == exp(2*I*t)*beta(a + 2, b)/beta(a, b) +\
         2*exp(I*t)*beta(a + 1, b + 1)/beta(a, b) + beta(a, b + 2)/beta(a, b)
    assert moment_generating_function(X)(t) == exp(2*t)*beta(a + 2, b)/beta(a, b) +\
         2*exp(t)*beta(a + 1, b + 1)/beta(a, b) + beta(a, b + 2)/beta(a, b)

def test_hypergeometric_numeric():
    for N in range(1, 5):
        for m in range(0, N + 1):
            for n in range(1, N + 1):
                X = Hypergeometric('X', N, m, n)
                N, m, n = map(sympify, (N, m, n))
                assert sum(density(X).values()) == 1
                assert E(X) == n * m / N
                if N > 1:
                    assert variance(X) == n*(m/N)*(N - m)/N*(N - n)/(N - 1)
                # Only test for skewness when defined
                if N > 2 and 0 < m < N and n < N:
                    assert skewness(X) == simplify((N - 2*m)*sqrt(N - 1)*(N - 2*n)
                        / (sqrt(n*m*(N - m)*(N - n))*(N - 2)))

def test_hypergeometric_symbolic():
    N, m, n = symbols('N, m, n')
    H = Hypergeometric('H', N, m, n)
    dens = density(H).dict
    expec = E(H > 2)
    assert dens == Density(HypergeometricDistribution(N, m, n))
    assert dens.subs(N, 5).doit() == Density(HypergeometricDistribution(5, m, n))
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().keys()) == {S.Zero, S.One}
    assert set(dens.subs({N: 3, m: 2, n: 1}).doit().values()) == {Rational(1, 3), Rational(2, 3)}
    k = Dummy('k', integer=True)
    assert expec.dummy_eq(
        Sum(Piecewise((k*binomial(m, k)*binomial(N - m, -k + n)
        /binomial(N, n), k > 2), (0, True)), (k, 0, n)))

def test_rademacher():
    X = Rademacher('X')
    t = Symbol('t')

    assert E(X) == 0
    assert variance(X) == 1
    assert density(X)[-1] == S.Half
    assert density(X)[1] == S.Half
    assert characteristic_function(X)(t) == exp(I*t)/2 + exp(-I*t)/2
    assert moment_generating_function(X)(t) == exp(t) / 2 + exp(-t) / 2

def test_ideal_soliton():
    raises(ValueError, lambda : IdealSoliton('sol', -12))
    raises(ValueError, lambda : IdealSoliton('sol', 13.2))
    raises(ValueError, lambda : IdealSoliton('sol', 0))
    f = Function('f')
    raises(ValueError, lambda : density(IdealSoliton('sol', 10)).pmf(f))

    k = Symbol('k', integer=True, positive=True)
    x = Symbol('x', integer=True, positive=True)
    t = Symbol('t')
    sol = IdealSoliton('sol', k)
    assert density(sol).low == S.One
    assert density(sol).high == k
    assert density(sol).dict == Density(density(sol))
    assert density(sol).pmf(x) == Piecewise((1/k, Eq(x, 1)), (1/(x*(x - 1)), k >= x), (0, True))

    k_vals = [5, 20, 50, 100, 1000]
    for i in k_vals:
        assert E(sol.subs(k, i)) == harmonic(i) == moment(sol.subs(k, i), 1)
        assert variance(sol.subs(k, i)) == (i - 1) + harmonic(i) - harmonic(i)**2 == cmoment(sol.subs(k, i),2)
        assert skewness(sol.subs(k, i)) == smoment(sol.subs(k, i), 3)
        assert kurtosis(sol.subs(k, i)) == smoment(sol.subs(k, i), 4)

    assert exp(I*t)/10 + Sum(exp(I*t*x)/(x*x - x), (x, 2, k)).subs(k, 10).doit() == characteristic_function(sol.subs(k, 10))(t)
    assert exp(t)/10 + Sum(exp(t*x)/(x*x - x), (x, 2, k)).subs(k, 10).doit() == moment_generating_function(sol.subs(k, 10))(t)

def test_robust_soliton():
    raises(ValueError, lambda : RobustSoliton('robSol', -12, 0.1, 0.02))
    raises(ValueError, lambda : RobustSoliton('robSol', 13, 1.89, 0.1))
    raises(ValueError, lambda : RobustSoliton('robSol', 15, 0.6, -2.31))
    f = Function('f')
    raises(ValueError, lambda : density(RobustSoliton('robSol', 15, 0.6, 0.1)).pmf(f))

    k = Symbol('k', integer=True, positive=True)
    delta = Symbol('delta', positive=True)
    c = Symbol('c', positive=True)
    robSol = RobustSoliton('robSol', k, delta, c)
    assert density(robSol).low == 1
    assert density(robSol).high == k

    k_vals = [10, 20, 50]
    delta_vals = [0.2, 0.4, 0.6]
    c_vals = [0.01, 0.03, 0.05]
    for x in k_vals:
        for y in delta_vals:
            for z in c_vals:
                assert E(robSol.subs({k: x, delta: y, c: z})) == moment(robSol.subs({k: x, delta: y, c: z}), 1)
                assert variance(robSol.subs({k: x, delta: y, c: z})) == cmoment(robSol.subs({k: x, delta: y, c: z}), 2)
                assert skewness(robSol.subs({k: x, delta: y, c: z})) == smoment(robSol.subs({k: x, delta: y, c: z}), 3)
                assert kurtosis(robSol.subs({k: x, delta: y, c: z})) == smoment(robSol.subs({k: x, delta: y, c: z}), 4)

def test_FiniteRV():
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)}, check=True)
    p = Symbol("p", positive=True)

    assert dict(density(F).items()) == {S.One: S.Half, S(2): Rational(1, 4), S(3): Rational(1, 4)}
    assert P(F >= 2) == S.Half
    assert quantile(F)(p) == Piecewise((nan, p > S.One), (S.One, p <= S.Half),\
        (S(2), p <= Rational(3, 4)),(S(3), True))

    assert pspace(F).domain.as_boolean() == Or(
        *[Eq(F.symbol, i) for i in [1, 2, 3]])

    assert F.pspace.domain.set == FiniteSet(1, 2, 3)
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: S.Half, 3: S.Half}, check=True))
    raises(ValueError, lambda: FiniteRV('F', {1: S.Half, 2: Rational(-1, 2), 3: S.One}, check=True))
    raises(ValueError, lambda: FiniteRV('F', {1: S.One, 2: Rational(3, 2), 3: S.Zero,\
        4: Rational(-1, 2), 5: Rational(-3, 4), 6: Rational(-1, 4)}, check=True))

    # purposeful invalid pmf but it should not raise since check=False
    # see test_drv_types.test_ContinuousRV for explanation
    X = FiniteRV('X', {1: 1, 2: 2})
    assert E(X) == 5
    assert P(X <= 2) + P(X > 2) != 1

def test_density_call():
    from sympy.abc import p
    x = Bernoulli('x', p)
    d = density(x)
    assert d(0) == 1 - p
    assert d(S.Zero) == 1 - p
    assert d(5) == 0

    assert 0 in d
    assert 5 not in d
    assert d(S.Zero) == d[S.Zero]


def test_DieDistribution():
    from sympy.abc import x
    X = DieDistribution(6)
    assert X.pmf(S.Half) is S.Zero
    assert X.pmf(x).subs({x: 1}).doit() == Rational(1, 6)
    assert X.pmf(x).subs({x: 7}).doit() == 0
    assert X.pmf(x).subs({x: -1}).doit() == 0
    assert X.pmf(x).subs({x: Rational(1, 3)}).doit() == 0
    raises(ValueError, lambda: X.pmf(Matrix([0, 0])))
    raises(ValueError, lambda: X.pmf(x**2 - 1))

def test_FinitePSpace():
    X = Die('X', 6)
    space = pspace(X)
    assert space.density == DieDistribution(6)

def test_symbolic_conditions():
    B = Bernoulli('B', Rational(1, 4))
    D = Die('D', 4)
    b, n = symbols('b, n')
    Y = P(Eq(B, b))
    Z = E(D > n)
    assert Y == \
    Piecewise((Rational(1, 4), Eq(b, 1)), (0, True)) + \
    Piecewise((Rational(3, 4), Eq(b, 0)), (0, True))
    assert Z == \
    Piecewise((Rational(1, 4), n < 1), (0, True)) + Piecewise((S.Half, n < 2), (0, True)) + \
    Piecewise((Rational(3, 4), n < 3), (0, True)) + Piecewise((S.One, n < 4), (0, True))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_rank.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.algos import (
    Infinity,
    NegInfinity,
)

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestRank:
    s = Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])
    df = DataFrame({"A": s, "B": s})

    results = {
        "average": np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5]),
        "min": np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5]),
        "max": np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6]),
        "first": np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6]),
        "dense": np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3]),
    }

    @pytest.fixture(params=["average", "min", "max", "first", "dense"])
    def method(self, request):
        """
        Fixture for trying all rank methods
        """
        return request.param

    def test_rank(self, float_frame):
        sp_stats = pytest.importorskip("scipy.stats")

        float_frame.loc[::2, "A"] = np.nan
        float_frame.loc[::3, "B"] = np.nan
        float_frame.loc[::4, "C"] = np.nan
        float_frame.loc[::5, "D"] = np.nan

        ranks0 = float_frame.rank()
        ranks1 = float_frame.rank(1)
        mask = np.isnan(float_frame.values)

        fvals = float_frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        exp0[mask] = np.nan

        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)
        exp1[mask] = np.nan

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # integers
        df = DataFrame(
            np.random.default_rng(2).integers(0, 5, size=40).reshape((10, 4))
        )

        result = df.rank()
        exp = df.astype(float).rank()
        tm.assert_frame_equal(result, exp)

        result = df.rank(1)
        exp = df.astype(float).rank(1)
        tm.assert_frame_equal(result, exp)

    def test_rank2(self):
        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected = DataFrame([[1.0, 3.0, 2.0], [1, 2, 3]]) / 3.0
        result = df.rank(1, pct=True)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected = df.rank(0) / 2.0
        result = df.rank(0, pct=True)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([["b", "c", "a"], ["a", "c", "b"]])
        expected = DataFrame([[2.0, 3.0, 1.0], [1, 3, 2]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2.0, 1.5, 1.0], [1, 1.5, 2]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([["b", np.nan, "a"], ["a", "c", "b"]])
        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 3.0, 2.0]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 1.0, 2.0]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        # f7u12, this does not work without extensive workaround
        data = [
            [datetime(2001, 1, 5), np.nan, datetime(2001, 1, 2)],
            [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 1)],
        ]
        df = DataFrame(data)

        # check the rank
        expected = DataFrame([[2.0, np.nan, 1.0], [2.0, 3.0, 1.0]])
        result = df.rank(1, numeric_only=False, ascending=True)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[1.0, np.nan, 2.0], [2.0, 1.0, 3.0]])
        result = df.rank(1, numeric_only=False, ascending=False)
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": [1e-20, -5, 1e-20 + 1e-40, 10, 1e60, 1e80, 1e-30]})
        exp = DataFrame({"a": [3.5, 1.0, 3.5, 5.0, 6.0, 7.0, 2.0]})
        tm.assert_frame_equal(df.rank(), exp)

    def test_rank_does_not_mutate(self):
        # GH#18521
        # Check rank does not mutate DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), dtype="float64"
        )
        expected = df.copy()
        df.rank()
        result = df
        tm.assert_frame_equal(result, expected)

    def test_rank_mixed_frame(self, float_string_frame):
        float_string_frame["datetime"] = datetime.now()
        float_string_frame["timedelta"] = timedelta(days=1, seconds=1)

        float_string_frame.rank(numeric_only=False)
        with pytest.raises(TypeError, match="not supported between instances of"):
            float_string_frame.rank(axis=1)

    def test_rank_na_option(self, float_frame):
        sp_stats = pytest.importorskip("scipy.stats")

        float_frame.loc[::2, "A"] = np.nan
        float_frame.loc[::3, "B"] = np.nan
        float_frame.loc[::4, "C"] = np.nan
        float_frame.loc[::5, "D"] = np.nan

        # bottom
        ranks0 = float_frame.rank(na_option="bottom")
        ranks1 = float_frame.rank(1, na_option="bottom")

        fvals = float_frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # top
        ranks0 = float_frame.rank(na_option="top")
        ranks1 = float_frame.rank(1, na_option="top")

        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fval0)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fval1)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # descending

        # bottom
        ranks0 = float_frame.rank(na_option="top", ascending=False)
        ranks1 = float_frame.rank(1, na_option="top", ascending=False)

        fvals = float_frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fvals)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fvals)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # descending

        # top
        ranks0 = float_frame.rank(na_option="bottom", ascending=False)
        ranks1 = float_frame.rank(1, na_option="bottom", ascending=False)

        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fval0)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fval1)

        tm.assert_numpy_array_equal(ranks0.values, exp0)
        tm.assert_numpy_array_equal(ranks1.values, exp1)

        # bad values throw error
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"

        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option="bad", ascending=False)

        # invalid type
        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option=True, ascending=False)

    def test_rank_axis(self):
        # check if using axes' names gives the same result
        df = DataFrame([[2, 1], [4, 3]])
        tm.assert_frame_equal(df.rank(axis=0), df.rank(axis="index"))
        tm.assert_frame_equal(df.rank(axis=1), df.rank(axis="columns"))

    @pytest.mark.parametrize("ax", [0, 1])
    @pytest.mark.parametrize("m", ["average", "min", "max", "first", "dense"])
    def test_rank_methods_frame(self, ax, m):
        sp_stats = pytest.importorskip("scipy.stats")

        xs = np.random.default_rng(2).integers(0, 21, (100, 26))
        xs = (xs - 10.0) / 10.0
        cols = [chr(ord("z") - i) for i in range(xs.shape[1])]

        for vals in [xs, xs + 1e6, xs * 1e-6]:
            df = DataFrame(vals, columns=cols)

            result = df.rank(axis=ax, method=m)
            sprank = np.apply_along_axis(
                sp_stats.rankdata, ax, vals, m if m != "first" else "ordinal"
            )
            sprank = sprank.astype(np.float64)
            expected = DataFrame(sprank, columns=cols).astype("float64")
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["O", "f8", "i8"])
    def test_rank_descending(self, method, dtype):
        if "i" in dtype:
            df = self.df.dropna().astype(dtype)
        else:
            df = self.df.astype(dtype)

        res = df.rank(ascending=False)
        expected = (df.max() - df).rank()
        tm.assert_frame_equal(res, expected)

        expected = (df.max() - df).rank(method=method)

        if dtype != "O":
            res2 = df.rank(method=method, ascending=False, numeric_only=True)
            tm.assert_frame_equal(res2, expected)

        res3 = df.rank(method=method, ascending=False, numeric_only=False)
        tm.assert_frame_equal(res3, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dtype", [None, object])
    def test_rank_2d_tie_methods(self, method, axis, dtype):
        df = self.df

        def _check2d(df, expected, method="average", axis=0):
            exp_df = DataFrame({"A": expected, "B": expected})

            if axis == 1:
                df = df.T
                exp_df = exp_df.T

            result = df.rank(method=method, axis=axis)
            tm.assert_frame_equal(result, exp_df)

        frame = df if dtype is None else df.astype(dtype)
        _check2d(frame, self.results[method], method=method, axis=axis)

    @pytest.mark.parametrize(
        "method,exp",
        [
            ("dense", [[1.0, 1.0, 1.0], [1.0, 0.5, 2.0 / 3], [1.0, 0.5, 1.0 / 3]]),
            (
                "min",
                [
                    [1.0 / 3, 1.0, 1.0],
                    [1.0 / 3, 1.0 / 3, 2.0 / 3],
                    [1.0 / 3, 1.0 / 3, 1.0 / 3],
                ],
            ),
            (
                "max",
                [[1.0, 1.0, 1.0], [1.0, 2.0 / 3, 2.0 / 3], [1.0, 2.0 / 3, 1.0 / 3]],
            ),
            (
                "average",
                [[2.0 / 3, 1.0, 1.0], [2.0 / 3, 0.5, 2.0 / 3], [2.0 / 3, 0.5, 1.0 / 3]],
            ),
            (
                "first",
                [
                    [1.0 / 3, 1.0, 1.0],
                    [2.0 / 3, 1.0 / 3, 2.0 / 3],
                    [3.0 / 3, 2.0 / 3, 1.0 / 3],
                ],
            ),
        ],
    )
    def test_rank_pct_true(self, method, exp):
        # see gh-15630.

        df = DataFrame([[2012, 66, 3], [2012, 65, 2], [2012, 65, 1]])
        result = df.rank(method=method, pct=True)

        expected = DataFrame(exp)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self):
        # GH 18271
        df = DataFrame(
            {"A": np.arange(2**24 + 1), "B": np.arange(2**24 + 1, 0, -1)}
        )
        result = df.rank(pct=True).max()
        assert (result == 1).all()

    @pytest.mark.parametrize(
        "contents,dtype",
        [
            (
                [
                    -np.inf,
                    -50,
                    -1,
                    -1e-20,
                    -1e-25,
                    -1e-50,
                    0,
                    1e-40,
                    1e-20,
                    1e-10,
                    2,
                    40,
                    np.inf,
                ],
                "float64",
            ),
            (
                [
                    -np.inf,
                    -50,
                    -1,
                    -1e-20,
                    -1e-25,
                    -1e-45,
                    0,
                    1e-40,
                    1e-20,
                    1e-10,
                    2,
                    40,
                    np.inf,
                ],
                "float32",
            ),
            ([np.iinfo(np.uint8).min, 1, 2, 100, np.iinfo(np.uint8).max], "uint8"),
            (
                [
                    np.iinfo(np.int64).min,
                    -100,
                    0,
                    1,
                    9999,
                    100000,
                    1e10,
                    np.iinfo(np.int64).max,
                ],
                "int64",
            ),
            ([NegInfinity(), "1", "A", "BA", "Ba", "C", Infinity()], "object"),
            (
                [datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 5)],
                "datetime64",
            ),
        ],
    )
    def test_rank_inf_and_nan(self, contents, dtype, frame_or_series):
        dtype_na_map = {
            "float64": np.nan,
            "float32": np.nan,
            "object": None,
            "datetime64": np.datetime64("nat"),
        }
        # Insert nans at random positions if underlying dtype has missing
        # value. Then adjust the expected order by adding nans accordingly
        # This is for testing whether rank calculation is affected
        # when values are interwined with nan values.
        values = np.array(contents, dtype=dtype)
        exp_order = np.array(range(len(values)), dtype="float64") + 1.0
        if dtype in dtype_na_map:
            na_value = dtype_na_map[dtype]
            nan_indices = np.random.default_rng(2).choice(range(len(values)), 5)
            values = np.insert(values, nan_indices, na_value)
            exp_order = np.insert(exp_order, nan_indices, np.nan)

        # Shuffle the testing array and expected results in the same way
        random_order = np.random.default_rng(2).permutation(len(values))
        obj = frame_or_series(values[random_order])
        expected = frame_or_series(exp_order[random_order], dtype="float64")
        result = obj.rank()
        tm.assert_equal(result, expected)

    def test_df_series_inf_nan_consistency(self):
        # GH#32593
        index = [5, 4, 3, 2, 1, 6, 7, 8, 9, 10]
        col1 = [5, 4, 3, 5, 8, 5, 2, 1, 6, 6]
        col2 = [5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf]
        df = DataFrame(
            data={
                "col1": col1,
                "col2": col2,
            },
            index=index,
            dtype="f8",
        )
        df_result = df.rank()

        series_result = df.copy()
        series_result["col1"] = df["col1"].rank()
        series_result["col2"] = df["col2"].rank()

        tm.assert_frame_equal(df_result, series_result)

    def test_rank_both_inf(self):
        # GH#32593
        df = DataFrame({"a": [-np.inf, 0, np.inf]})
        expected = DataFrame({"a": [1.0, 2.0, 3.0]})
        result = df.rank()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "na_option,ascending,expected",
        [
            ("top", True, [3.0, 1.0, 2.0]),
            ("top", False, [2.0, 1.0, 3.0]),
            ("bottom", True, [2.0, 3.0, 1.0]),
            ("bottom", False, [1.0, 3.0, 2.0]),
        ],
    )
    def test_rank_inf_nans_na_option(
        self, frame_or_series, method, na_option, ascending, expected
    ):
        obj = frame_or_series([np.inf, np.nan, -np.inf])
        result = obj.rank(method=method, na_option=na_option, ascending=ascending)
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "na_option,ascending,expected",
        [
            ("bottom", True, [1.0, 2.0, 4.0, 3.0]),
            ("bottom", False, [1.0, 2.0, 4.0, 3.0]),
            ("top", True, [2.0, 3.0, 1.0, 4.0]),
            ("top", False, [2.0, 3.0, 1.0, 4.0]),
        ],
    )
    def test_rank_object_first(self, frame_or_series, na_option, ascending, expected):
        obj = frame_or_series(["foo", "foo", None, "foo"])
        result = obj.rank(method="first", na_option=na_option, ascending=ascending)
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                {"a": [1, 2, "a"], "b": [4, 5, 6]},
                DataFrame({"b": [1.0, 2.0, 3.0]}, columns=Index(["b"], dtype=object)),
            ),
            ({"a": [1, 2, "a"]}, DataFrame(index=range(3), columns=[])),
        ],
    )
    def test_rank_mixed_axis_zero(self, data, expected):
        df = DataFrame(data, columns=Index(list(data.keys()), dtype=object))
        with pytest.raises(TypeError, match="'<' not supported between instances of"):
            df.rank()
        result = df.rank(numeric_only=True)
        tm.assert_frame_equal(result, expected)

    def test_rank_string_dtype(self, string_dtype_no_object):
        # GH#55362
        obj = Series(["foo", "foo", None, "foo"], dtype=string_dtype_no_object)
        result = obj.rank(method="first")
        exp_dtype = (
            "Float64" if string_dtype_no_object == "string[pyarrow]" else "float64"
        )
        if string_dtype_no_object.storage == "python":
            # TODO nullable string[python] should also return nullable Int64
            exp_dtype = "float64"
        expected = Series([1, 2, None, 3], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\test_generic.py ===
from copy import (
    copy,
    deepcopy,
)

import numpy as np
import pytest

from pandas.core.dtypes.common import is_scalar

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm

# ----------------------------------------------------------------------
# Generic types test cases


def construct(box, shape, value=None, dtype=None, **kwargs):
    """
    construct an object for the given shape
    if value is specified use that if its a scalar
    if value is an array, repeat it as needed
    """
    if isinstance(shape, int):
        shape = tuple([shape] * box._AXIS_LEN)
    if value is not None:
        if is_scalar(value):
            if value == "empty":
                arr = None
                dtype = np.float64

                # remove the info axis
                kwargs.pop(box._info_axis_name, None)
            else:
                arr = np.empty(shape, dtype=dtype)
                arr.fill(value)
        else:
            fshape = np.prod(shape)
            arr = value.ravel()
            new_shape = fshape / arr.shape[0]
            if fshape % arr.shape[0] != 0:
                raise Exception("invalid value passed in construct")

            arr = np.repeat(arr, new_shape).reshape(shape)
    else:
        arr = np.random.default_rng(2).standard_normal(shape)
    return box(arr, dtype=dtype, **kwargs)


class TestGeneric:
    @pytest.mark.parametrize(
        "func",
        [
            str.lower,
            {x: x.lower() for x in list("ABCD")},
            Series({x: x.lower() for x in list("ABCD")}),
        ],
    )
    def test_rename(self, frame_or_series, func):
        # single axis
        idx = list("ABCD")

        for axis in frame_or_series._AXIS_ORDERS:
            kwargs = {axis: idx}
            obj = construct(frame_or_series, 4, **kwargs)

            # rename a single axis
            result = obj.rename(**{axis: func})
            expected = obj.copy()
            setattr(expected, axis, list("abcd"))
            tm.assert_equal(result, expected)

    def test_get_numeric_data(self, frame_or_series):
        n = 4
        kwargs = {
            frame_or_series._get_axis_name(i): list(range(n))
            for i in range(frame_or_series._AXIS_LEN)
        }

        # get the numeric data
        o = construct(frame_or_series, n, **kwargs)
        result = o._get_numeric_data()
        tm.assert_equal(result, o)

        # non-inclusion
        result = o._get_bool_data()
        expected = construct(frame_or_series, n, value="empty", **kwargs)
        if isinstance(o, DataFrame):
            # preserve columns dtype
            expected.columns = o.columns[:0]
        # https://github.com/pandas-dev/pandas/issues/50862
        tm.assert_equal(result.reset_index(drop=True), expected)

        # get the bool data
        arr = np.array([True, True, False, True])
        o = construct(frame_or_series, n, value=arr, **kwargs)
        result = o._get_numeric_data()
        tm.assert_equal(result, o)

    def test_nonzero(self, frame_or_series):
        # GH 4633
        # look at the boolean/nonzero behavior for objects
        obj = construct(frame_or_series, shape=4)
        msg = f"The truth value of a {frame_or_series.__name__} is ambiguous"
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        obj = construct(frame_or_series, shape=4, value=1)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        obj = construct(frame_or_series, shape=4, value=np.nan)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        # empty
        obj = construct(frame_or_series, shape=0)
        with pytest.raises(ValueError, match=msg):
            bool(obj)

        # invalid behaviors

        obj1 = construct(frame_or_series, shape=4, value=1)
        obj2 = construct(frame_or_series, shape=4, value=1)

        with pytest.raises(ValueError, match=msg):
            if obj1:
                pass

        with pytest.raises(ValueError, match=msg):
            obj1 and obj2
        with pytest.raises(ValueError, match=msg):
            obj1 or obj2
        with pytest.raises(ValueError, match=msg):
            not obj1

    def test_frame_or_series_compound_dtypes(self, frame_or_series):
        # see gh-5191
        # Compound dtypes should raise NotImplementedError.

        def f(dtype):
            return construct(frame_or_series, shape=3, value=1, dtype=dtype)

        msg = (
            "compound dtypes are not implemented "
            f"in the {frame_or_series.__name__} constructor"
        )

        with pytest.raises(NotImplementedError, match=msg):
            f([("A", "datetime64[h]"), ("B", "str"), ("C", "int32")])

        # these work (though results may be unexpected)
        f("int64")
        f("float64")
        f("M8[ns]")

    def test_metadata_propagation(self, frame_or_series):
        # check that the metadata matches up on the resulting ops

        o = construct(frame_or_series, shape=3)
        o.name = "foo"
        o2 = construct(frame_or_series, shape=3)
        o2.name = "bar"

        # ----------
        # preserving
        # ----------

        # simple ops with scalars
        for op in ["__add__", "__sub__", "__truediv__", "__mul__"]:
            result = getattr(o, op)(1)
            tm.assert_metadata_equivalent(o, result)

        # ops with like
        for op in ["__add__", "__sub__", "__truediv__", "__mul__"]:
            result = getattr(o, op)(o)
            tm.assert_metadata_equivalent(o, result)

        # simple boolean
        for op in ["__eq__", "__le__", "__ge__"]:
            v1 = getattr(o, op)(o)
            tm.assert_metadata_equivalent(o, v1)
            tm.assert_metadata_equivalent(o, v1 & v1)
            tm.assert_metadata_equivalent(o, v1 | v1)

        # combine_first
        result = o.combine_first(o2)
        tm.assert_metadata_equivalent(o, result)

        # ---------------------------
        # non-preserving (by default)
        # ---------------------------

        # add non-like
        result = o + o2
        tm.assert_metadata_equivalent(result)

        # simple boolean
        for op in ["__eq__", "__le__", "__ge__"]:
            # this is a name matching op
            v1 = getattr(o, op)(o)
            v2 = getattr(o, op)(o2)
            tm.assert_metadata_equivalent(v2)
            tm.assert_metadata_equivalent(v1 & v2)
            tm.assert_metadata_equivalent(v1 | v2)

    def test_size_compat(self, frame_or_series):
        # GH8846
        # size property should be defined

        o = construct(frame_or_series, shape=10)
        assert o.size == np.prod(o.shape)
        assert o.size == 10 ** len(o.axes)

    def test_split_compat(self, frame_or_series):
        # xref GH8846
        o = construct(frame_or_series, shape=10)
        with tm.assert_produces_warning(
            FutureWarning, match=".swapaxes' is deprecated", check_stacklevel=False
        ):
            assert len(np.array_split(o, 5)) == 5
            assert len(np.array_split(o, 2)) == 2

    # See gh-12301
    def test_stat_unexpected_keyword(self, frame_or_series):
        obj = construct(frame_or_series, 5)
        starwars = "Star Wars"
        errmsg = "unexpected keyword"

        with pytest.raises(TypeError, match=errmsg):
            obj.max(epic=starwars)  # stat_function
        with pytest.raises(TypeError, match=errmsg):
            obj.var(epic=starwars)  # stat_function_ddof
        with pytest.raises(TypeError, match=errmsg):
            obj.sum(epic=starwars)  # cum_function
        with pytest.raises(TypeError, match=errmsg):
            obj.any(epic=starwars)  # logical_function

    @pytest.mark.parametrize("func", ["sum", "cumsum", "any", "var"])
    def test_api_compat(self, func, frame_or_series):
        # GH 12021
        # compat for __name__, __qualname__

        obj = construct(frame_or_series, 5)
        f = getattr(obj, func)
        assert f.__name__ == func
        assert f.__qualname__.endswith(func)

    def test_stat_non_defaults_args(self, frame_or_series):
        obj = construct(frame_or_series, 5)
        out = np.array([0])
        errmsg = "the 'out' parameter is not supported"

        with pytest.raises(ValueError, match=errmsg):
            obj.max(out=out)  # stat_function
        with pytest.raises(ValueError, match=errmsg):
            obj.var(out=out)  # stat_function_ddof
        with pytest.raises(ValueError, match=errmsg):
            obj.sum(out=out)  # cum_function
        with pytest.raises(ValueError, match=errmsg):
            obj.any(out=out)  # logical_function

    def test_truncate_out_of_bounds(self, frame_or_series):
        # GH11382

        # small
        shape = [2000] + ([1] * (frame_or_series._AXIS_LEN - 1))
        small = construct(frame_or_series, shape, dtype="int8", value=1)
        tm.assert_equal(small.truncate(), small)
        tm.assert_equal(small.truncate(before=0, after=3e3), small)
        tm.assert_equal(small.truncate(before=-1, after=2e3), small)

        # big
        shape = [2_000_000] + ([1] * (frame_or_series._AXIS_LEN - 1))
        big = construct(frame_or_series, shape, dtype="int8", value=1)
        tm.assert_equal(big.truncate(), big)
        tm.assert_equal(big.truncate(before=0, after=3e6), big)
        tm.assert_equal(big.truncate(before=-1, after=2e6), big)

    @pytest.mark.parametrize(
        "func",
        [copy, deepcopy, lambda x: x.copy(deep=False), lambda x: x.copy(deep=True)],
    )
    @pytest.mark.parametrize("shape", [0, 1, 2])
    def test_copy_and_deepcopy(self, frame_or_series, shape, func):
        # GH 15444
        obj = construct(frame_or_series, shape)
        obj_copy = func(obj)
        assert obj_copy is not obj
        tm.assert_equal(obj_copy, obj)

    def test_data_deprecated(self, frame_or_series):
        obj = frame_or_series()
        msg = "(Series|DataFrame)._data is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            mgr = obj._data
        assert mgr is obj._mgr


class TestNDFrame:
    # tests that don't fit elsewhere

    @pytest.mark.parametrize(
        "ser",
        [
            Series(range(10), dtype=np.float64),
            Series([str(i) for i in range(10)], dtype=object),
        ],
    )
    def test_squeeze_series_noop(self, ser):
        # noop
        tm.assert_series_equal(ser.squeeze(), ser)

    def test_squeeze_frame_noop(self):
        # noop
        df = DataFrame(np.eye(2))
        tm.assert_frame_equal(df.squeeze(), df)

    def test_squeeze_frame_reindex(self):
        # squeezing
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        ).reindex(columns=["A"])
        tm.assert_series_equal(df.squeeze(), df["A"])

    def test_squeeze_0_len_dim(self):
        # don't fail with 0 length dimensions GH11229 & GH8999
        empty_series = Series([], name="five", dtype=np.float64)
        empty_frame = DataFrame([empty_series])
        tm.assert_series_equal(empty_series, empty_series.squeeze())
        tm.assert_series_equal(empty_series, empty_frame.squeeze())

    def test_squeeze_axis(self):
        # axis argument
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=1, freq="B"),
        ).iloc[:, :1]
        assert df.shape == (1, 1)
        tm.assert_series_equal(df.squeeze(axis=0), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis="index"), df.iloc[0])
        tm.assert_series_equal(df.squeeze(axis=1), df.iloc[:, 0])
        tm.assert_series_equal(df.squeeze(axis="columns"), df.iloc[:, 0])
        assert df.squeeze() == df.iloc[0, 0]
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis=2)
        msg = "No axis named x for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis="x")

    def test_squeeze_axis_len_3(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=3, freq="B"),
        )
        tm.assert_frame_equal(df.squeeze(axis=0), df)

    def test_numpy_squeeze(self):
        s = Series(range(2), dtype=np.float64)
        tm.assert_series_equal(np.squeeze(s), s)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        ).reindex(columns=["A"])
        tm.assert_series_equal(np.squeeze(df), df["A"])

    @pytest.mark.parametrize(
        "ser",
        [
            Series(range(10), dtype=np.float64),
            Series([str(i) for i in range(10)], dtype=object),
        ],
    )
    def test_transpose_series(self, ser):
        # calls implementation in pandas/core/base.py
        tm.assert_series_equal(ser.transpose(), ser)

    def test_transpose_frame(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        tm.assert_frame_equal(df.transpose().transpose(), df)

    def test_numpy_transpose(self, frame_or_series):
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        obj = tm.get_obj(obj, frame_or_series)

        if frame_or_series is Series:
            # 1D -> np.transpose is no-op
            tm.assert_series_equal(np.transpose(obj), obj)

        # round-trip preserved
        tm.assert_equal(np.transpose(np.transpose(obj)), obj)

        msg = "the 'axes' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.transpose(obj, axes=1)

    @pytest.mark.parametrize(
        "ser",
        [
            Series(range(10), dtype=np.float64),
            Series([str(i) for i in range(10)], dtype=object),
        ],
    )
    def test_take_series(self, ser):
        indices = [1, 5, -2, 6, 3, -1]
        out = ser.take(indices)
        expected = Series(
            data=ser.values.take(indices),
            index=ser.index.take(indices),
            dtype=ser.dtype,
        )
        tm.assert_series_equal(out, expected)

    def test_take_frame(self):
        indices = [1, 5, -2, 6, 3, -1]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        out = df.take(indices)
        expected = DataFrame(
            data=df.values.take(indices, axis=0),
            index=df.index.take(indices),
            columns=df.columns,
        )
        tm.assert_frame_equal(out, expected)

    def test_take_invalid_kwargs(self, frame_or_series):
        indices = [-3, 2, 0, 1]

        obj = DataFrame(range(5))
        obj = tm.get_obj(obj, frame_or_series)

        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            obj.take(indices, foo=2)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, out=indices)

        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, mode="clip")

    def test_axis_classmethods(self, frame_or_series):
        box = frame_or_series
        obj = box(dtype=object)
        values = box._AXIS_TO_AXIS_NUMBER.keys()
        for v in values:
            assert obj._get_axis_number(v) == box._get_axis_number(v)
            assert obj._get_axis_name(v) == box._get_axis_name(v)
            assert obj._get_block_manager_axis(v) == box._get_block_manager_axis(v)

    def test_flags_identity(self, frame_or_series):
        obj = Series([1, 2])
        if frame_or_series is DataFrame:
            obj = obj.to_frame()

        assert obj.flags is obj.flags
        obj2 = obj.copy()
        assert obj2.flags is not obj.flags

    def test_bool_dep(self) -> None:
        # GH-51749
        msg_warn = (
            "DataFrame.bool is now deprecated and will be removed "
            "in future version of pandas"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            DataFrame({"col": [False]}).bool()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_miscellaneous.py ===
import itertools as it

from sympy.core.expr import unchanged
from sympy.core.function import Function
from sympy.core.numbers import I, oo, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,
                                                      Max, real_root, Rem)
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.delta_functions import Heaviside

from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, skip, ignore_warnings

def test_Min():
    from sympy.abc import x, y, z
    n = Symbol('n', negative=True)
    n_ = Symbol('n_', negative=True)
    nn = Symbol('nn', nonnegative=True)
    nn_ = Symbol('nn_', nonnegative=True)
    p = Symbol('p', positive=True)
    p_ = Symbol('p_', positive=True)
    np = Symbol('np', nonpositive=True)
    np_ = Symbol('np_', nonpositive=True)
    r = Symbol('r', real=True)

    assert Min(5, 4) == 4
    assert Min(-oo, -oo) is -oo
    assert Min(-oo, n) is -oo
    assert Min(n, -oo) is -oo
    assert Min(-oo, np) is -oo
    assert Min(np, -oo) is -oo
    assert Min(-oo, 0) is -oo
    assert Min(0, -oo) is -oo
    assert Min(-oo, nn) is -oo
    assert Min(nn, -oo) is -oo
    assert Min(-oo, p) is -oo
    assert Min(p, -oo) is -oo
    assert Min(-oo, oo) is -oo
    assert Min(oo, -oo) is -oo
    assert Min(n, n) == n
    assert unchanged(Min, n, np)
    assert Min(np, n) == Min(n, np)
    assert Min(n, 0) == n
    assert Min(0, n) == n
    assert Min(n, nn) == n
    assert Min(nn, n) == n
    assert Min(n, p) == n
    assert Min(p, n) == n
    assert Min(n, oo) == n
    assert Min(oo, n) == n
    assert Min(np, np) == np
    assert Min(np, 0) == np
    assert Min(0, np) == np
    assert Min(np, nn) == np
    assert Min(nn, np) == np
    assert Min(np, p) == np
    assert Min(p, np) == np
    assert Min(np, oo) == np
    assert Min(oo, np) == np
    assert Min(0, 0) == 0
    assert Min(0, nn) == 0
    assert Min(nn, 0) == 0
    assert Min(0, p) == 0
    assert Min(p, 0) == 0
    assert Min(0, oo) == 0
    assert Min(oo, 0) == 0
    assert Min(nn, nn) == nn
    assert unchanged(Min, nn, p)
    assert Min(p, nn) == Min(nn, p)
    assert Min(nn, oo) == nn
    assert Min(oo, nn) == nn
    assert Min(p, p) == p
    assert Min(p, oo) == p
    assert Min(oo, p) == p
    assert Min(oo, oo) is oo

    assert Min(n, n_).func is Min
    assert Min(nn, nn_).func is Min
    assert Min(np, np_).func is Min
    assert Min(p, p_).func is Min

    # lists
    assert Min() is S.Infinity
    assert Min(x) == x
    assert Min(x, y) == Min(y, x)
    assert Min(x, y, z) == Min(z, y, x)
    assert Min(x, Min(y, z)) == Min(z, y, x)
    assert Min(x, Max(y, -oo)) == Min(x, y)
    assert Min(p, oo, n, p, p, p_) == n
    assert Min(p_, n_, p) == n_
    assert Min(n, oo, -7, p, p, 2) == Min(n, -7)
    assert Min(2, x, p, n, oo, n_, p, 2, -2, -2) == Min(-2, x, n, n_)
    assert Min(0, x, 1, y) == Min(0, x, y)
    assert Min(1000, 100, -100, x, p, n) == Min(n, x, -100)
    assert unchanged(Min, sin(x), cos(x))
    assert Min(sin(x), cos(x)) == Min(cos(x), sin(x))
    assert Min(cos(x), sin(x)).subs(x, 1) == cos(1)
    assert Min(cos(x), sin(x)).subs(x, S.Half) == sin(S.Half)
    raises(ValueError, lambda: Min(cos(x), sin(x)).subs(x, I))
    raises(ValueError, lambda: Min(I))
    raises(ValueError, lambda: Min(I, x))
    raises(ValueError, lambda: Min(S.ComplexInfinity, x))

    assert Min(1, x).diff(x) == Heaviside(1 - x)
    assert Min(x, 1).diff(x) == Heaviside(1 - x)
    assert Min(0, -x, 1 - 2*x).diff(x) == -Heaviside(x + Min(0, -2*x + 1)) \
        - 2*Heaviside(2*x + Min(0, -x) - 1)

    # issue 7619
    f = Function('f')
    assert Min(1, 2*Min(f(1), 2))  # doesn't fail

    # issue 7233
    e = Min(0, x)
    assert e.n().args == (0, x)

    # issue 8643
    m = Min(n, p_, n_, r)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True

    m = Min(p, p_)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Min(p, nn_, p_)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Min(nn, p, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None


def test_Max():
    from sympy.abc import x, y, z
    n = Symbol('n', negative=True)
    n_ = Symbol('n_', negative=True)
    nn = Symbol('nn', nonnegative=True)
    p = Symbol('p', positive=True)
    p_ = Symbol('p_', positive=True)
    r = Symbol('r', real=True)

    assert Max(5, 4) == 5

    # lists

    assert Max() is S.NegativeInfinity
    assert Max(x) == x
    assert Max(x, y) == Max(y, x)
    assert Max(x, y, z) == Max(z, y, x)
    assert Max(x, Max(y, z)) == Max(z, y, x)
    assert Max(x, Min(y, oo)) == Max(x, y)
    assert Max(n, -oo, n_, p, 2) == Max(p, 2)
    assert Max(n, -oo, n_, p) == p
    assert Max(2, x, p, n, -oo, S.NegativeInfinity, n_, p, 2) == Max(2, x, p)
    assert Max(0, x, 1, y) == Max(1, x, y)
    assert Max(r, r + 1, r - 1) == 1 + r
    assert Max(1000, 100, -100, x, p, n) == Max(p, x, 1000)
    assert Max(cos(x), sin(x)) == Max(sin(x), cos(x))
    assert Max(cos(x), sin(x)).subs(x, 1) == sin(1)
    assert Max(cos(x), sin(x)).subs(x, S.Half) == cos(S.Half)
    raises(ValueError, lambda: Max(cos(x), sin(x)).subs(x, I))
    raises(ValueError, lambda: Max(I))
    raises(ValueError, lambda: Max(I, x))
    raises(ValueError, lambda: Max(S.ComplexInfinity, 1))
    assert Max(n, -oo, n_,  p, 2) == Max(p, 2)
    assert Max(n, -oo, n_,  p, 1000) == Max(p, 1000)

    assert Max(1, x).diff(x) == Heaviside(x - 1)
    assert Max(x, 1).diff(x) == Heaviside(x - 1)
    assert Max(x**2, 1 + x, 1).diff(x) == \
        2*x*Heaviside(x**2 - Max(1, x + 1)) \
        + Heaviside(x - Max(1, x**2) + 1)

    e = Max(0, x)
    assert e.n().args == (0, x)

    # issue 8643
    m = Max(p, p_, n, r)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Max(n, n_)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True

    m = Max(n, n_, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None

    m = Max(n, nn, r)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False


def test_minmax_assumptions():
    r = Symbol('r', real=True)
    a = Symbol('a', real=True, algebraic=True)
    t = Symbol('t', real=True, transcendental=True)
    q = Symbol('q', rational=True)
    p = Symbol('p', irrational=True)
    n = Symbol('n', rational=True, integer=False)
    i = Symbol('i', integer=True)
    o = Symbol('o', odd=True)
    e = Symbol('e', even=True)
    k = Symbol('k', prime=True)
    reals = [r, a, t, q, p, n, i, o, e, k]

    for ext in (Max, Min):
        for x, y in it.product(reals, repeat=2):

            # Must be real
            assert ext(x, y).is_real

            # Algebraic?
            if x.is_algebraic and y.is_algebraic:
                assert ext(x, y).is_algebraic
            elif x.is_transcendental and y.is_transcendental:
                assert ext(x, y).is_transcendental
            else:
                assert ext(x, y).is_algebraic is None

            # Rational?
            if x.is_rational and y.is_rational:
                assert ext(x, y).is_rational
            elif x.is_irrational and y.is_irrational:
                assert ext(x, y).is_irrational
            else:
                assert ext(x, y).is_rational is None

            # Integer?
            if x.is_integer and y.is_integer:
                assert ext(x, y).is_integer
            elif x.is_noninteger and y.is_noninteger:
                assert ext(x, y).is_noninteger
            else:
                assert ext(x, y).is_integer is None

            # Odd?
            if x.is_odd and y.is_odd:
                assert ext(x, y).is_odd
            elif x.is_odd is False and y.is_odd is False:
                assert ext(x, y).is_odd is False
            else:
                assert ext(x, y).is_odd is None

            # Even?
            if x.is_even and y.is_even:
                assert ext(x, y).is_even
            elif x.is_even is False and y.is_even is False:
                assert ext(x, y).is_even is False
            else:
                assert ext(x, y).is_even is None

            # Prime?
            if x.is_prime and y.is_prime:
                assert ext(x, y).is_prime
            elif x.is_prime is False and y.is_prime is False:
                assert ext(x, y).is_prime is False
            else:
                assert ext(x, y).is_prime is None


def test_issue_8413():
    x = Symbol('x', real=True)
    # we can't evaluate in general because non-reals are not
    # comparable: Min(floor(3.2 + I), 3.2 + I) -> ValueError
    assert Min(floor(x), x) == floor(x)
    assert Min(ceiling(x), x) == x
    assert Max(floor(x), x) == x
    assert Max(ceiling(x), x) == ceiling(x)


def test_root():
    from sympy.abc import x
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)

    assert root(2, 2) == sqrt(2)
    assert root(2, 1) == 2
    assert root(2, 3) == 2**Rational(1, 3)
    assert root(2, 3) == cbrt(2)
    assert root(2, -5) == 2**Rational(4, 5)/2

    assert root(-2, 1) == -2

    assert root(-2, 2) == sqrt(2)*I
    assert root(-2, 1) == -2

    assert root(x, 2) == sqrt(x)
    assert root(x, 1) == x
    assert root(x, 3) == x**Rational(1, 3)
    assert root(x, 3) == cbrt(x)
    assert root(x, -5) == x**Rational(-1, 5)

    assert root(x, n) == x**(1/n)
    assert root(x, -n) == x**(-1/n)

    assert root(x, n, k) == (-1)**(2*k/n)*x**(1/n)


def test_real_root():
    assert real_root(-8, 3) == -2
    assert real_root(-16, 4) == root(-16, 4)
    r = root(-7, 4)
    assert real_root(r) == r
    r1 = root(-1, 3)
    r2 = r1**2
    r3 = root(-1, 4)
    assert real_root(r1 + r2 + r3) == -1 + r2 + r3
    assert real_root(root(-2, 3)) == -root(2, 3)
    assert real_root(-8., 3) == -2.0
    x = Symbol('x')
    n = Symbol('n')
    g = real_root(x, n)
    assert g.subs({"x": -8, "n": 3}) == -2
    assert g.subs({"x": 8, "n": 3}) == 2
    # give principle root if there is no real root -- if this is not desired
    # then maybe a Root class is needed to raise an error instead
    assert g.subs({"x": I, "n": 3}) == cbrt(I)
    assert g.subs({"x": -8, "n": 2}) == sqrt(-8)
    assert g.subs({"x": I, "n": 2}) == sqrt(I)


def test_issue_11463():
    numpy = import_module('numpy')
    if not numpy:
        skip("numpy not installed.")
    x = Symbol('x')
    f = lambdify(x, real_root((log(x/(x-2))), 3), 'numpy')
    # numpy.select evaluates all options before considering conditions,
    # so it raises a warning about root of negative number which does
    # not affect the outcome. This warning is suppressed here
    with ignore_warnings(RuntimeWarning):
        assert f(numpy.array(-1)) < -1


def test_rewrite_MaxMin_as_Heaviside():
    from sympy.abc import x
    assert Max(0, x).rewrite(Heaviside) == x*Heaviside(x)
    assert Max(3, x).rewrite(Heaviside) == x*Heaviside(x - 3) + \
        3*Heaviside(-x + 3)
    assert Max(0, x+2, 2*x).rewrite(Heaviside) == \
        2*x*Heaviside(2*x)*Heaviside(x - 2) + \
        (x + 2)*Heaviside(-x + 2)*Heaviside(x + 2)

    assert Min(0, x).rewrite(Heaviside) == x*Heaviside(-x)
    assert Min(3, x).rewrite(Heaviside) == x*Heaviside(-x + 3) + \
        3*Heaviside(x - 3)
    assert Min(x, -x, -2).rewrite(Heaviside) == \
        x*Heaviside(-2*x)*Heaviside(-x - 2) - \
        x*Heaviside(2*x)*Heaviside(x - 2) \
        - 2*Heaviside(-x + 2)*Heaviside(x + 2)


def test_rewrite_MaxMin_as_Piecewise():
    from sympy.core.symbol import symbols
    from sympy.functions.elementary.piecewise import Piecewise
    x, y, z, a, b = symbols('x y z a b', real=True)
    vx, vy, va = symbols('vx vy va')
    assert Max(a, b).rewrite(Piecewise) == Piecewise((a, a >= b), (b, True))
    assert Max(x, y, z).rewrite(Piecewise) == Piecewise((x, (x >= y) & (x >= z)), (y, y >= z), (z, True))
    assert Max(x, y, a, b).rewrite(Piecewise) == Piecewise((a, (a >= b) & (a >= x) & (a >= y)),
        (b, (b >= x) & (b >= y)), (x, x >= y), (y, True))
    assert Min(a, b).rewrite(Piecewise) == Piecewise((a, a <= b), (b, True))
    assert Min(x, y, z).rewrite(Piecewise) == Piecewise((x, (x <= y) & (x <= z)), (y, y <= z), (z, True))
    assert Min(x,  y, a, b).rewrite(Piecewise) ==  Piecewise((a, (a <= b) & (a <= x) & (a <= y)),
        (b, (b <= x) & (b <= y)), (x, x <= y), (y, True))

    # Piecewise rewriting of Min/Max does also takes place for not explicitly real arguments
    assert Max(vx, vy).rewrite(Piecewise) == Piecewise((vx, vx >= vy), (vy, True))
    assert Min(va, vx, vy).rewrite(Piecewise) == Piecewise((va, (va <= vx) & (va <= vy)), (vx, vx <= vy), (vy, True))


def test_issue_11099():
    from sympy.abc import x, y
    # some fixed value tests
    fixed_test_data = {x: -2, y: 3}
    assert Min(x, y).evalf(subs=fixed_test_data) == \
        Min(x, y).subs(fixed_test_data).evalf()
    assert Max(x, y).evalf(subs=fixed_test_data) == \
        Max(x, y).subs(fixed_test_data).evalf()
    # randomly generate some test data
    from sympy.core.random import randint
    for i in range(20):
        random_test_data = {x: randint(-100, 100), y: randint(-100, 100)}
        assert Min(x, y).evalf(subs=random_test_data) == \
            Min(x, y).subs(random_test_data).evalf()
        assert Max(x, y).evalf(subs=random_test_data) == \
            Max(x, y).subs(random_test_data).evalf()


def test_issue_12638():
    from sympy.abc import a, b, c
    assert Min(a, b, c, Max(a, b)) == Min(a, b, c)
    assert Min(a, b, Max(a, b, c)) == Min(a, b)
    assert Min(a, b, Max(a, c)) == Min(a, b)

def test_issue_21399():
    from sympy.abc import a, b, c
    assert Max(Min(a, b), Min(a, b, c)) == Min(a, b)


def test_instantiation_evaluation():
    from sympy.abc import v, w, x, y, z
    assert Min(1, Max(2, x)) == 1
    assert Max(3, Min(2, x)) == 3
    assert Min(Max(x, y), Max(x, z)) == Max(x, Min(y, z))
    assert set(Min(Max(w, x), Max(y, z)).args) == {
        Max(w, x), Max(y, z)}
    assert Min(Max(x, y), Max(x, z), w) == Min(
        w, Max(x, Min(y, z)))
    A, B = Min, Max
    for i in range(2):
        assert A(x, B(x, y)) == x
        assert A(x, B(y, A(x, w, z))) == A(x, B(y, A(w, z)))
        A, B = B, A
    assert Min(w, Max(x, y), Max(v, x, z)) == Min(
        w, Max(x, Min(y, Max(v, z))))

def test_rewrite_as_Abs():
    from itertools import permutations
    from sympy.functions.elementary.complexes import Abs
    from sympy.abc import x, y, z, w
    def test(e):
        free = e.free_symbols
        a = e.rewrite(Abs)
        assert not a.has(Min, Max)
        for i in permutations(range(len(free))):
            reps = dict(zip(free, i))
            assert a.xreplace(reps) == e.xreplace(reps)
    test(Min(x, y))
    test(Max(x, y))
    test(Min(x, y, z))
    test(Min(Max(w, x), Max(y, z)))

def test_issue_14000():
    assert isinstance(sqrt(4, evaluate=False), Pow) == True
    assert isinstance(cbrt(3.5, evaluate=False), Pow) == True
    assert isinstance(root(16, 4, evaluate=False), Pow) == True

    assert sqrt(4, evaluate=False) == Pow(4, S.Half, evaluate=False)
    assert cbrt(3.5, evaluate=False) == Pow(3.5, Rational(1, 3), evaluate=False)
    assert root(4, 2, evaluate=False) == Pow(4, S.Half, evaluate=False)

    assert root(16, 4, 2, evaluate=False).has(Pow) == True
    assert real_root(-8, 3, evaluate=False).has(Pow) == True

def test_issue_6899():
    from sympy.core.function import Lambda
    x = Symbol('x')
    eqn = Lambda(x, x)
    assert eqn.func(*eqn.args) == eqn

def test_Rem():
    from sympy.abc import x, y
    assert Rem(5, 3) == 2
    assert Rem(-5, 3) == -2
    assert Rem(5, -3) == 2
    assert Rem(-5, -3) == -2
    assert Rem(x**3, y) == Rem(x**3, y)
    assert Rem(Rem(-5, 3) + 3, 3) == 1


def test_minmax_no_evaluate():
    from sympy import evaluate
    p = Symbol('p', positive=True)

    assert Max(1, 3) == 3
    assert Max(1, 3).args == ()
    assert Max(0, p) == p
    assert Max(0, p).args == ()
    assert Min(0, p) == 0
    assert Min(0, p).args == ()

    assert Max(1, 3, evaluate=False) != 3
    assert Max(1, 3, evaluate=False).args == (1, 3)
    assert Max(0, p, evaluate=False) != p
    assert Max(0, p, evaluate=False).args == (0, p)
    assert Min(0, p, evaluate=False) != 0
    assert Min(0, p, evaluate=False).args == (0, p)

    with evaluate(False):
        assert Max(1, 3) != 3
        assert Max(1, 3).args == (1, 3)
        assert Max(0, p) != p
        assert Max(0, p).args == (0, p)
        assert Min(0, p) != 0
        assert Min(0, p).args == (0, p)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_function_base.py ===
import platform
import sys

import pytest

import numpy as np
from numpy import (
    arange,
    array,
    dtype,
    errstate,
    geomspace,
    isnan,
    linspace,
    logspace,
    ndarray,
    nextafter,
    sqrt,
    stack,
)
from numpy._core import sctypes
from numpy._core.function_base import add_newdoc
from numpy.testing import (
    IS_PYPY,
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
)


def _is_armhf():
    # Check if the current platform is ARMHF (32-bit ARM architecture)
    return platform.machine().startswith('arm') and platform.architecture()[0] == '32bit'

class PhysicalQuantity(float):
    def __new__(cls, value):
        return float.__new__(cls, value)

    def __add__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) + float(self))
    __radd__ = __add__

    def __sub__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(self) - float(x))

    def __rsub__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) - float(self))

    def __mul__(self, x):
        return PhysicalQuantity(float(x) * float(self))
    __rmul__ = __mul__

    def __truediv__(self, x):
        return PhysicalQuantity(float(self) / float(x))

    def __rtruediv__(self, x):
        return PhysicalQuantity(float(x) / float(self))


class PhysicalQuantity2(ndarray):
    __array_priority__ = 10


class TestLogspace:

    def test_basic(self):
        y = logspace(0, 6)
        assert_(len(y) == 50)
        y = logspace(0, 6, num=100)
        assert_(y[-1] == 10 ** 6)
        y = logspace(0, 6, endpoint=False)
        assert_(y[-1] < 10 ** 6)
        y = logspace(0, 6, num=7)
        assert_array_equal(y, [1, 10, 100, 1e3, 1e4, 1e5, 1e6])

    def test_start_stop_array(self):
        start = array([0., 1.])
        stop = array([6., 7.])
        t1 = logspace(start, stop, 6)
        t2 = stack([logspace(_start, _stop, 6)
                    for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = logspace(start, stop[0], 6)
        t4 = stack([logspace(_start, stop[0], 6)
                    for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = logspace(start, stop, 6, axis=-1)
        assert_equal(t5, t2.T)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_base_array(self, axis: int):
        start = 1
        stop = 2
        num = 6
        base = array([1, 2])
        t1 = logspace(start, stop, num=num, base=base, axis=axis)
        t2 = stack(
            [logspace(start, stop, num=num, base=_base) for _base in base],
            axis=(axis + 1) % t1.ndim,
        )
        assert_equal(t1, t2)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_stop_base_array(self, axis: int):
        start = 1
        stop = array([2, 3])
        num = 6
        base = array([1, 2])
        t1 = logspace(start, stop, num=num, base=base, axis=axis)
        t2 = stack(
            [logspace(start, _stop, num=num, base=_base)
             for _stop, _base in zip(stop, base)],
            axis=(axis + 1) % t1.ndim,
        )
        assert_equal(t1, t2)

    def test_dtype(self):
        y = logspace(0, 6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = logspace(0, 6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = logspace(0, 6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

    def test_physical_quantities(self):
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        assert_equal(logspace(a, b), logspace(1.0, 5.0))

    def test_subclass(self):
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        ls = logspace(a, b)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, logspace(1.0, 7.0))
        ls = logspace(a, b, 1)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, logspace(1.0, 7.0, 1))


class TestGeomspace:

    def test_basic(self):
        y = geomspace(1, 1e6)
        assert_(len(y) == 50)
        y = geomspace(1, 1e6, num=100)
        assert_(y[-1] == 10 ** 6)
        y = geomspace(1, 1e6, endpoint=False)
        assert_(y[-1] < 10 ** 6)
        y = geomspace(1, 1e6, num=7)
        assert_array_equal(y, [1, 10, 100, 1e3, 1e4, 1e5, 1e6])

        y = geomspace(8, 2, num=3)
        assert_allclose(y, [8, 4, 2])
        assert_array_equal(y.imag, 0)

        y = geomspace(-1, -100, num=3)
        assert_array_equal(y, [-1, -10, -100])
        assert_array_equal(y.imag, 0)

        y = geomspace(-100, -1, num=3)
        assert_array_equal(y, [-100, -10, -1])
        assert_array_equal(y.imag, 0)

    def test_boundaries_match_start_and_stop_exactly(self):
        # make sure that the boundaries of the returned array exactly
        # equal 'start' and 'stop' - this isn't obvious because
        # np.exp(np.log(x)) isn't necessarily exactly equal to x
        start = 0.3
        stop = 20.3

        y = geomspace(start, stop, num=1)
        assert_equal(y[0], start)

        y = geomspace(start, stop, num=1, endpoint=False)
        assert_equal(y[0], start)

        y = geomspace(start, stop, num=3)
        assert_equal(y[0], start)
        assert_equal(y[-1], stop)

        y = geomspace(start, stop, num=3, endpoint=False)
        assert_equal(y[0], start)

    def test_nan_interior(self):
        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4)

        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:-1]).all())
        assert_equal(y[3], 3.0)

        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4, endpoint=False)

        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:]).all())

    def test_complex(self):
        # Purely imaginary
        y = geomspace(1j, 16j, num=5)
        assert_allclose(y, [1j, 2j, 4j, 8j, 16j])
        assert_array_equal(y.real, 0)

        y = geomspace(-4j, -324j, num=5)
        assert_allclose(y, [-4j, -12j, -36j, -108j, -324j])
        assert_array_equal(y.real, 0)

        y = geomspace(1 + 1j, 1000 + 1000j, num=4)
        assert_allclose(y, [1 + 1j, 10 + 10j, 100 + 100j, 1000 + 1000j])

        y = geomspace(-1 + 1j, -1000 + 1000j, num=4)
        assert_allclose(y, [-1 + 1j, -10 + 10j, -100 + 100j, -1000 + 1000j])

        # Logarithmic spirals
        y = geomspace(-1, 1, num=3, dtype=complex)
        assert_allclose(y, [-1, 1j, +1])

        y = geomspace(0 + 3j, -3 + 0j, 3)
        assert_allclose(y, [0 + 3j, -3 / sqrt(2) + 3j / sqrt(2), -3 + 0j])
        y = geomspace(0 + 3j, 3 + 0j, 3)
        assert_allclose(y, [0 + 3j, 3 / sqrt(2) + 3j / sqrt(2), 3 + 0j])
        y = geomspace(-3 + 0j, 0 - 3j, 3)
        assert_allclose(y, [-3 + 0j, -3 / sqrt(2) - 3j / sqrt(2), 0 - 3j])
        y = geomspace(0 + 3j, -3 + 0j, 3)
        assert_allclose(y, [0 + 3j, -3 / sqrt(2) + 3j / sqrt(2), -3 + 0j])
        y = geomspace(-2 - 3j, 5 + 7j, 7)
        assert_allclose(y, [-2 - 3j, -0.29058977 - 4.15771027j,
                            2.08885354 - 4.34146838j, 4.58345529 - 3.16355218j,
                            6.41401745 - 0.55233457j, 6.75707386 + 3.11795092j,
                            5 + 7j])

        # Type promotion should prevent the -5 from becoming a NaN
        y = geomspace(3j, -5, 2)
        assert_allclose(y, [3j, -5])
        y = geomspace(-5, 3j, 2)
        assert_allclose(y, [-5, 3j])

    def test_complex_shortest_path(self):
        # test the shortest logarithmic spiral is used, see gh-25644
        x = 1.2 + 3.4j
        y = np.exp(1j * (np.pi - .1)) * x
        z = np.geomspace(x, y, 5)
        expected = np.array([1.2 + 3.4j, -1.47384 + 3.2905616j,
                        -3.33577588 + 1.36842949j, -3.36011056 - 1.30753855j,
                        -1.53343861 - 3.26321406j])
        np.testing.assert_array_almost_equal(z, expected)

    def test_dtype(self):
        y = geomspace(1, 1e6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = geomspace(1, 1e6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = geomspace(1, 1e6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

        # Native types
        y = geomspace(1, 1e6, dtype=float)
        assert_equal(y.dtype, dtype('float64'))
        y = geomspace(1, 1e6, dtype=complex)
        assert_equal(y.dtype, dtype('complex128'))

    def test_start_stop_array_scalar(self):
        lim1 = array([120, 100], dtype="int8")
        lim2 = array([-120, -100], dtype="int8")
        lim3 = array([1200, 1000], dtype="uint16")
        t1 = geomspace(lim1[0], lim1[1], 5)
        t2 = geomspace(lim2[0], lim2[1], 5)
        t3 = geomspace(lim3[0], lim3[1], 5)
        t4 = geomspace(120.0, 100.0, 5)
        t5 = geomspace(-120.0, -100.0, 5)
        t6 = geomspace(1200.0, 1000.0, 5)

        # t3 uses float32, t6 uses float64
        assert_allclose(t1, t4, rtol=1e-2)
        assert_allclose(t2, t5, rtol=1e-2)
        assert_allclose(t3, t6, rtol=1e-5)

    def test_start_stop_array(self):
        # Try to use all special cases.
        start = array([1.e0, 32., 1j, -4j, 1 + 1j, -1])
        stop = array([1.e4, 2., 16j, -324j, 10000 + 10000j, 1])
        t1 = geomspace(start, stop, 5)
        t2 = stack([geomspace(_start, _stop, 5)
                    for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = geomspace(start, stop[0], 5)
        t4 = stack([geomspace(_start, stop[0], 5)
                    for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = geomspace(start, stop, 5, axis=-1)
        assert_equal(t5, t2.T)

    def test_physical_quantities(self):
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        assert_equal(geomspace(a, b), geomspace(1.0, 5.0))

    def test_subclass(self):
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        gs = geomspace(a, b)
        assert type(gs) is PhysicalQuantity2
        assert_equal(gs, geomspace(1.0, 7.0))
        gs = geomspace(a, b, 1)
        assert type(gs) is PhysicalQuantity2
        assert_equal(gs, geomspace(1.0, 7.0, 1))

    def test_bounds(self):
        assert_raises(ValueError, geomspace, 0, 10)
        assert_raises(ValueError, geomspace, 10, 0)
        assert_raises(ValueError, geomspace, 0, 0)


class TestLinspace:

    def test_basic(self):
        y = linspace(0, 10)
        assert_(len(y) == 50)
        y = linspace(2, 10, num=100)
        assert_(y[-1] == 10)
        y = linspace(2, 10, endpoint=False)
        assert_(y[-1] < 10)
        assert_raises(ValueError, linspace, 0, 10, num=-1)

    def test_corner(self):
        y = list(linspace(0, 1, 1))
        assert_(y == [0.0], y)
        assert_raises(TypeError, linspace, 0, 1, num=2.5)

    def test_type(self):
        t1 = linspace(0, 1, 0).dtype
        t2 = linspace(0, 1, 1).dtype
        t3 = linspace(0, 1, 2).dtype
        assert_equal(t1, t2)
        assert_equal(t2, t3)

    def test_dtype(self):
        y = linspace(0, 6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = linspace(0, 6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = linspace(0, 6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

    def test_start_stop_array_scalar(self):
        lim1 = array([-120, 100], dtype="int8")
        lim2 = array([120, -100], dtype="int8")
        lim3 = array([1200, 1000], dtype="uint16")
        t1 = linspace(lim1[0], lim1[1], 5)
        t2 = linspace(lim2[0], lim2[1], 5)
        t3 = linspace(lim3[0], lim3[1], 5)
        t4 = linspace(-120.0, 100.0, 5)
        t5 = linspace(120.0, -100.0, 5)
        t6 = linspace(1200.0, 1000.0, 5)
        assert_equal(t1, t4)
        assert_equal(t2, t5)
        assert_equal(t3, t6)

    def test_start_stop_array(self):
        start = array([-120, 120], dtype="int8")
        stop = array([100, -100], dtype="int8")
        t1 = linspace(start, stop, 5)
        t2 = stack([linspace(_start, _stop, 5)
                    for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = linspace(start, stop[0], 5)
        t4 = stack([linspace(_start, stop[0], 5)
                    for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = linspace(start, stop, 5, axis=-1)
        assert_equal(t5, t2.T)

    def test_complex(self):
        lim1 = linspace(1 + 2j, 3 + 4j, 5)
        t1 = array([1.0 + 2.j, 1.5 + 2.5j,  2.0 + 3j, 2.5 + 3.5j, 3.0 + 4j])
        lim2 = linspace(1j, 10, 5)
        t2 = array([0.0 + 1.j, 2.5 + 0.75j, 5.0 + 0.5j, 7.5 + 0.25j, 10.0 + 0j])
        assert_equal(lim1, t1)
        assert_equal(lim2, t2)

    def test_physical_quantities(self):
        a = PhysicalQuantity(0.0)
        b = PhysicalQuantity(1.0)
        assert_equal(linspace(a, b), linspace(0.0, 1.0))

    def test_subclass(self):
        a = array(0).view(PhysicalQuantity2)
        b = array(1).view(PhysicalQuantity2)
        ls = linspace(a, b)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, linspace(0.0, 1.0))
        ls = linspace(a, b, 1)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, linspace(0.0, 1.0, 1))

    def test_array_interface(self):
        # Regression test for https://github.com/numpy/numpy/pull/6659
        # Ensure that start/stop can be objects that implement
        # __array_interface__ and are convertible to numeric scalars

        class Arrayish:
            """
            A generic object that supports the __array_interface__ and hence
            can in principle be converted to a numeric scalar, but is not
            otherwise recognized as numeric, but also happens to support
            multiplication by floats.

            Data should be an object that implements the buffer interface,
            and contains at least 4 bytes.
            """

            def __init__(self, data):
                self._data = data

            @property
            def __array_interface__(self):
                return {'shape': (), 'typestr': '<i4', 'data': self._data,
                        'version': 3}

            def __mul__(self, other):
                # For the purposes of this test any multiplication is an
                # identity operation :)
                return self

        one = Arrayish(array(1, dtype='<i4'))
        five = Arrayish(array(5, dtype='<i4'))

        assert_equal(linspace(one, five), linspace(1, 5))

    # even when not explicitly enabled via FPSCR register
    @pytest.mark.xfail(_is_armhf(),
                       reason="ARMHF/AArch32 platforms seem to FTZ subnormals")
    def test_denormal_numbers(self):
        # Regression test for gh-5437. Will probably fail when compiled
        # with ICC, which flushes denormals to zero
        for ftype in sctypes['float']:
            stop = nextafter(ftype(0), ftype(1)) * 5  # A denormal number
            assert_(any(linspace(0, stop, 10, endpoint=False, dtype=ftype)))

    def test_equivalent_to_arange(self):
        for j in range(1000):
            assert_equal(linspace(0, j, j + 1, dtype=int),
                         arange(j + 1, dtype=int))

    def test_retstep(self):
        for num in [0, 1, 2]:
            for ept in [False, True]:
                y = linspace(0, 1, num, endpoint=ept, retstep=True)
                assert isinstance(y, tuple) and len(y) == 2
                if num == 2:
                    y0_expect = [0.0, 1.0] if ept else [0.0, 0.5]
                    assert_array_equal(y[0], y0_expect)
                    assert_equal(y[1], y0_expect[1])
                elif num == 1 and not ept:
                    assert_array_equal(y[0], [0.0])
                    assert_equal(y[1], 1.0)
                else:
                    assert_array_equal(y[0], [0.0][:num])
                    assert isnan(y[1])

    def test_object(self):
        start = array(1, dtype='O')
        stop = array(2, dtype='O')
        y = linspace(start, stop, 3)
        assert_array_equal(y, array([1., 1.5, 2.]))

    def test_round_negative(self):
        y = linspace(-1, 3, num=8, dtype=int)
        t = array([-1, -1, 0, 0, 1, 1, 2, 3], dtype=int)
        assert_array_equal(y, t)

    def test_any_step_zero_and_not_mult_inplace(self):
        # any_step_zero is True, _mult_inplace is False
        start = array([0.0, 1.0])
        stop = array([2.0, 1.0])
        y = linspace(start, stop, 3)
        assert_array_equal(y, array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))


class TestAdd_newdoc:

    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    @pytest.mark.xfail(IS_PYPY, reason="PyPy does not modify tp_doc")
    def test_add_doc(self):
        # test that np.add_newdoc did attach a docstring successfully:
        tgt = "Current flat index into the array."
        assert_equal(np._core.flatiter.index.__doc__[:len(tgt)], tgt)
        assert_(len(np._core.ufunc.identity.__doc__) > 300)
        assert_(len(np.lib._index_tricks_impl.mgrid.__doc__) > 300)

    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    def test_errors_are_ignored(self):
        prev_doc = np._core.flatiter.index.__doc__
        # nothing changed, but error ignored, this should probably
        # give a warning (or even error) in the future.
        add_newdoc("numpy._core", "flatiter", ("index", "bad docstring"))
        assert prev_doc == np._core.flatiter.index.__doc__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_sparse.py ===
"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""

import numpy as np
import pytest

from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base


def make_data(fill_value):
    rng = np.random.default_rng(2)
    if np.isnan(fill_value):
        data = rng.uniform(size=100)
    else:
        data = rng.integers(1, 100, size=100, dtype=int)
        if data[0] == data[1]:
            data[0] += 1

    data[2::3] = fill_value
    return data


@pytest.fixture
def dtype():
    return SparseDtype()


@pytest.fixture(params=[0, np.nan])
def data(request):
    """Length-100 PeriodArray for semantics test."""
    res = SparseArray(make_data(request.param), fill_value=request.param)
    return res


@pytest.fixture
def data_for_twos():
    return SparseArray(np.ones(100) * 2)


@pytest.fixture(params=[0, np.nan])
def data_missing(request):
    """Length 2 array with [NA, Valid]"""
    return SparseArray([np.nan, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_repeated(request):
    """Return different versions of data for count times"""

    def gen(count):
        for _ in range(count):
            yield SparseArray(make_data(request.param), fill_value=request.param)

    yield gen


@pytest.fixture(params=[0, np.nan])
def data_for_sorting(request):
    return SparseArray([2, 3, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_missing_for_sorting(request):
    return SparseArray([2, np.nan, 1], fill_value=request.param)


@pytest.fixture
def na_cmp():
    return lambda left, right: pd.isna(left) and pd.isna(right)


@pytest.fixture(params=[0, np.nan])
def data_for_grouping(request):
    return SparseArray([1, 1, np.nan, np.nan, 2, 2, 1, 3], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_for_compare(request):
    return SparseArray([0, 0, np.nan, -2, -1, 4, 2, 3, 0, 0], fill_value=request.param)


class TestSparseArray(base.ExtensionTests):
    def _supports_reduction(self, obj, op_name: str) -> bool:
        return True

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in [
            "prod",
            "median",
            "var",
            "std",
            "sem",
            "skew",
            "kurt",
        ]:
            mark = pytest.mark.xfail(
                reason="This should be viable but is not implemented"
            )
            request.node.add_marker(mark)
        elif (
            all_numeric_reductions in ["sum", "max", "min", "mean"]
            and data.dtype.kind == "f"
            and not skipna
        ):
            mark = pytest.mark.xfail(reason="getting a non-nan float")
            request.node.add_marker(mark)

        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in [
            "prod",
            "median",
            "var",
            "std",
            "sem",
            "skew",
            "kurt",
        ]:
            mark = pytest.mark.xfail(
                reason="This should be viable but is not implemented"
            )
            request.node.add_marker(mark)
        elif (
            all_numeric_reductions in ["sum", "max", "min", "mean"]
            and data.dtype.kind == "f"
            and not skipna
        ):
            mark = pytest.mark.xfail(reason="ExtensionArray NA mask are different")
            request.node.add_marker(mark)

        super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def _check_unsupported(self, data):
        if data.dtype == SparseDtype(int, 0):
            pytest.skip("Can't store nan in int array.")

    def test_concat_mixed_dtypes(self, data):
        # https://github.com/pandas-dev/pandas/issues/20762
        # This should be the same, aside from concat([sparse, float])
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # dataframes
        result = pd.concat(dfs)
        expected = pd.concat(
            [x.apply(lambda s: np.asarray(s).astype(object)) for x in dfs]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        "columns",
        [
            ["A", "B"],
            pd.MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["outer", "inner"]
            ),
        ],
    )
    @pytest.mark.parametrize("future_stack", [True, False])
    def test_stack(self, data, columns, future_stack):
        super().test_stack(data, columns, future_stack)

    def test_concat_columns(self, data, na_value):
        self._check_unsupported(data)
        super().test_concat_columns(data, na_value)

    def test_concat_extension_arrays_copy_false(self, data, na_value):
        self._check_unsupported(data)
        super().test_concat_extension_arrays_copy_false(data, na_value)

    def test_align(self, data, na_value):
        self._check_unsupported(data)
        super().test_align(data, na_value)

    def test_align_frame(self, data, na_value):
        self._check_unsupported(data)
        super().test_align_frame(data, na_value)

    def test_align_series_frame(self, data, na_value):
        self._check_unsupported(data)
        super().test_align_series_frame(data, na_value)

    def test_merge(self, data, na_value):
        self._check_unsupported(data)
        super().test_merge(data, na_value)

    def test_get(self, data):
        ser = pd.Series(data, index=[2 * i for i in range(len(data))])
        if np.isnan(ser.values.fill_value):
            assert np.isnan(ser.get(4)) and np.isnan(ser.iloc[2])
        else:
            assert ser.get(4) == ser.iloc[2]
        assert ser.get(2) == ser.iloc[1]

    def test_reindex(self, data, na_value):
        self._check_unsupported(data)
        super().test_reindex(data, na_value)

    def test_isna(self, data_missing):
        sarr = SparseArray(data_missing)
        expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.fill_value))
        expected = SparseArray([True, False], dtype=expected_dtype)
        result = sarr.isna()
        tm.assert_sp_array_equal(result, expected)

        # test isna for arr without na
        sarr = sarr.fillna(0)
        expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.fill_value))
        expected = SparseArray([False, False], fill_value=False, dtype=expected_dtype)
        tm.assert_equal(sarr.isna(), expected)

    def test_fillna_limit_backfill(self, data_missing):
        warns = (PerformanceWarning, FutureWarning)
        with tm.assert_produces_warning(warns, check_stacklevel=False):
            super().test_fillna_limit_backfill(data_missing)

    def test_fillna_no_op_returns_copy(self, data, request):
        if np.isnan(data.fill_value):
            request.applymarker(
                pytest.mark.xfail(reason="returns array with different fill value")
            )
        super().test_fillna_no_op_returns_copy(data)

    @pytest.mark.xfail(reason="Unsupported")
    def test_fillna_series(self, data_missing):
        # this one looks doable.
        # TODO: this fails bc we do not pass through data_missing. If we did,
        #  the 0-fill case would xpass
        super().test_fillna_series()

    def test_fillna_frame(self, data_missing):
        # Have to override to specify that fill_value will change.
        fill_value = data_missing[1]

        result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

        if pd.isna(data_missing.fill_value):
            dtype = SparseDtype(data_missing.dtype, fill_value)
        else:
            dtype = data_missing.dtype

        expected = pd.DataFrame(
            {
                "A": data_missing._from_sequence([fill_value, fill_value], dtype=dtype),
                "B": [1, 2],
            }
        )

        tm.assert_frame_equal(result, expected)

    _combine_le_expected_dtype = "Sparse[bool]"

    def test_fillna_copy_frame(self, data_missing, using_copy_on_write):
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr}, copy=False)

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        if hasattr(df._mgr, "blocks"):
            if using_copy_on_write:
                assert df.values.base is result.values.base
            else:
                assert df.values.base is not result.values.base
        assert df.A._values.to_dense() is arr.to_dense()

    def test_fillna_copy_series(self, data_missing, using_copy_on_write):
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr, copy=False)

        filled_val = ser[0]
        result = ser.fillna(filled_val)

        if using_copy_on_write:
            assert ser._values is result._values

        else:
            assert ser._values is not result._values
        assert ser._values.to_dense() is arr.to_dense()

    @pytest.mark.xfail(reason="Not Applicable")
    def test_fillna_length_mismatch(self, data_missing):
        super().test_fillna_length_mismatch(data_missing)

    def test_where_series(self, data, na_value):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))

        cond = np.array([True, True, False, False])
        result = ser.where(cond)

        new_dtype = SparseDtype("float", 0.0)
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=new_dtype)
        )
        tm.assert_series_equal(result, expected)

        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        cond = np.array([True, False, True, True])
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        tm.assert_series_equal(result, expected)

    def test_searchsorted(self, data_for_sorting, as_series):
        with tm.assert_produces_warning(PerformanceWarning, check_stacklevel=False):
            super().test_searchsorted(data_for_sorting, as_series)

    def test_shift_0_periods(self, data):
        # GH#33856 shifting with periods=0 should return a copy, not same obj
        result = data.shift(0)

        data._sparse_values[0] = data._sparse_values[1]
        assert result._sparse_values[0] != result._sparse_values[1]

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(self, method, data, na_value):
        # overriding because Sparse[int64, 0] cannot handle na_value
        self._check_unsupported(data)
        super().test_argmin_argmax_all_na(method, data, na_value)

    @pytest.mark.fails_arm_wheels
    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        self._check_unsupported(data)
        super().test_equals(data, na_value, as_series, box)

    @pytest.mark.fails_arm_wheels
    def test_equals_same_data_different_object(self, data):
        super().test_equals_same_data_different_object(data)

    @pytest.mark.parametrize(
        "func, na_action, expected",
        [
            (lambda x: x, None, SparseArray([1.0, np.nan])),
            (lambda x: x, "ignore", SparseArray([1.0, np.nan])),
            (str, None, SparseArray(["1.0", "nan"], fill_value="nan")),
            (str, "ignore", SparseArray(["1.0", np.nan])),
        ],
    )
    def test_map(self, func, na_action, expected):
        # GH52096
        data = SparseArray([1, np.nan])
        result = data.map(func, na_action=na_action)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map_raises(self, data, na_action):
        # GH52096
        msg = "fill value in the sparse values not supported"
        with pytest.raises(ValueError, match=msg):
            data.map(lambda x: np.nan, na_action=na_action)

    @pytest.mark.xfail(raises=TypeError, reason="no sparse StringDtype")
    def test_astype_string(self, data, nullable_string_dtype):
        # TODO: this fails bc we do not pass through nullable_string_dtype;
        #  If we did, the 0-cases would xpass
        super().test_astype_string(data)

    series_scalar_exc = None
    frame_scalar_exc = None
    divmod_exc = None
    series_array_exc = None

    def _skip_if_different_combine(self, data):
        if data.fill_value == 0:
            # arith ops call on dtype.fill_value so that the sparsity
            # is maintained. Combine can't be called on a dtype in
            # general, so we can't make the expected. This is tested elsewhere
            pytest.skip("Incorrected expected from Series.combine and tested elsewhere")

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        if data.dtype.fill_value != 0:
            pass
        elif all_arithmetic_operators.strip("_") not in [
            "mul",
            "rmul",
            "floordiv",
            "rfloordiv",
            "pow",
            "mod",
            "rmod",
        ]:
            mark = pytest.mark.xfail(reason="result dtype.fill_value mismatch")
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def _compare_other(
        self, ser: pd.Series, data_for_compare: SparseArray, comparison_op, other
    ):
        op = comparison_op

        result = op(data_for_compare, other)
        if isinstance(other, pd.Series):
            assert isinstance(result, pd.Series)
            assert isinstance(result.dtype, SparseDtype)
        else:
            assert isinstance(result, SparseArray)
        assert result.dtype.subtype == np.bool_

        if isinstance(other, pd.Series):
            fill_value = op(data_for_compare.fill_value, other._values.fill_value)
            expected = SparseArray(
                op(data_for_compare.to_dense(), np.asarray(other)),
                fill_value=fill_value,
                dtype=np.bool_,
            )

        else:
            fill_value = np.all(
                op(np.asarray(data_for_compare.fill_value), np.asarray(other))
            )

            expected = SparseArray(
                op(data_for_compare.to_dense(), np.asarray(other)),
                fill_value=fill_value,
                dtype=np.bool_,
            )
        if isinstance(other, pd.Series):
            # error: Incompatible types in assignment
            expected = pd.Series(expected)  # type: ignore[assignment]
        tm.assert_equal(result, expected)

    def test_scalar(self, data_for_compare: SparseArray, comparison_op):
        ser = pd.Series(data_for_compare)
        self._compare_other(ser, data_for_compare, comparison_op, 0)
        self._compare_other(ser, data_for_compare, comparison_op, 1)
        self._compare_other(ser, data_for_compare, comparison_op, -1)
        self._compare_other(ser, data_for_compare, comparison_op, np.nan)

    def test_array(self, data_for_compare: SparseArray, comparison_op, request):
        if data_for_compare.dtype.fill_value == 0 and comparison_op.__name__ in [
            "eq",
            "ge",
            "le",
        ]:
            mark = pytest.mark.xfail(reason="Wrong fill_value")
            request.applymarker(mark)

        arr = np.linspace(-4, 5, 10)
        ser = pd.Series(data_for_compare)
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    def test_sparse_array(self, data_for_compare: SparseArray, comparison_op, request):
        if data_for_compare.dtype.fill_value == 0 and comparison_op.__name__ != "gt":
            mark = pytest.mark.xfail(reason="Wrong fill_value")
            request.applymarker(mark)

        ser = pd.Series(data_for_compare)
        arr = data_for_compare + 1
        self._compare_other(ser, data_for_compare, comparison_op, arr)
        arr = data_for_compare * 2
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    @pytest.mark.xfail(reason="Different repr")
    def test_array_repr(self, data, size):
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason="result does not match expected")
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        super().test_groupby_extension_agg(as_index, data_for_grouping)


def test_array_type_with_arg(dtype):
    assert dtype.construct_array_type() is SparseArray

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\constructors\test_from_records.py ===
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest
import pytz

from pandas._config import using_string_dtype

from pandas.compat import is_platform_little_endian

from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    Interval,
    RangeIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestFromRecords:
    def test_from_records_dt64tz_frame(self):
        # GH#51162 don't lose tz when calling from_records with DataFrame input
        dti = date_range("2016-01-01", periods=10, tz="US/Pacific")
        df = DataFrame({i: dti for i in range(4)})
        with tm.assert_produces_warning(FutureWarning):
            res = DataFrame.from_records(df)
        tm.assert_frame_equal(res, df)

    def test_from_records_with_datetimes(self):
        # this may fail on certain platforms because of a numpy issue
        # related GH#6140
        if not is_platform_little_endian():
            pytest.skip("known failure of test on non-little endian")

        # construction with a null in a recarray
        # GH#6140
        expected = DataFrame({"EXPIRY": [datetime(2005, 3, 1, 0, 0), None]})

        arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
        dtypes = [("EXPIRY", "<M8[ns]")]

        recarray = np.rec.fromarrays(arrdata, dtype=dtypes)

        result = DataFrame.from_records(recarray)
        tm.assert_frame_equal(result, expected)

        # coercion should work too
        arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
        dtypes = [("EXPIRY", "<M8[m]")]
        recarray = np.rec.fromarrays(arrdata, dtype=dtypes)
        result = DataFrame.from_records(recarray)
        # we get the closest supported unit, "s"
        expected["EXPIRY"] = expected["EXPIRY"].astype("M8[s]")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="dtype checking logic doesn't work")
    def test_from_records_sequencelike(self):
        df = DataFrame(
            {
                "A": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "A1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "B": np.array(np.arange(6), dtype=np.int64),
                "C": ["foo"] * 6,
                "D": np.array([True, False] * 3, dtype=bool),
                "E": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "E1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "F": np.array(np.arange(6), dtype=np.int32),
            }
        )

        # this is actually tricky to create the recordlike arrays and
        # have the dtypes be intact
        blocks = df._to_dict_of_blocks()
        tuples = []
        columns = []
        dtypes = []
        for dtype, b in blocks.items():
            columns.extend(b.columns)
            dtypes.extend([(c, np.dtype(dtype).descr[0][1]) for c in b.columns])
        for i in range(len(df.index)):
            tup = []
            for _, b in blocks.items():
                tup.extend(b.iloc[i].values)
            tuples.append(tuple(tup))

        recarray = np.array(tuples, dtype=dtypes).view(np.rec.recarray)
        recarray2 = df.to_records()
        lists = [list(x) for x in tuples]

        # tuples (lose the dtype info)
        result = DataFrame.from_records(tuples, columns=columns).reindex(
            columns=df.columns
        )

        # created recarray and with to_records recarray (have dtype info)
        result2 = DataFrame.from_records(recarray, columns=columns).reindex(
            columns=df.columns
        )
        result3 = DataFrame.from_records(recarray2, columns=columns).reindex(
            columns=df.columns
        )

        # list of tuples (no dtype info)
        result4 = DataFrame.from_records(lists, columns=columns).reindex(
            columns=df.columns
        )

        tm.assert_frame_equal(result, df, check_dtype=False)
        tm.assert_frame_equal(result2, df)
        tm.assert_frame_equal(result3, df)
        tm.assert_frame_equal(result4, df, check_dtype=False)

        # tuples is in the order of the columns
        result = DataFrame.from_records(tuples)
        tm.assert_index_equal(result.columns, RangeIndex(8))

        # test exclude parameter & we are casting the results here (as we don't
        # have dtype info to recover)
        columns_to_test = [columns.index("C"), columns.index("E1")]

        exclude = list(set(range(8)) - set(columns_to_test))
        result = DataFrame.from_records(tuples, exclude=exclude)
        result.columns = [columns[i] for i in sorted(columns_to_test)]
        tm.assert_series_equal(result["C"], df["C"])
        tm.assert_series_equal(result["E1"], df["E1"])

    def test_from_records_sequencelike_empty(self):
        # empty case
        result = DataFrame.from_records([], columns=["foo", "bar", "baz"])
        assert len(result) == 0
        tm.assert_index_equal(result.columns, Index(["foo", "bar", "baz"]))

        result = DataFrame.from_records([])
        assert len(result) == 0
        assert len(result.columns) == 0

    def test_from_records_dictlike(self):
        # test the dict methods
        df = DataFrame(
            {
                "A": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "A1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "B": np.array(np.arange(6), dtype=np.int64),
                "C": ["foo"] * 6,
                "D": np.array([True, False] * 3, dtype=bool),
                "E": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "E1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "F": np.array(np.arange(6), dtype=np.int32),
            }
        )

        # columns is in a different order here than the actual items iterated
        # from the dict
        blocks = df._to_dict_of_blocks()
        columns = []
        for b in blocks.values():
            columns.extend(b.columns)

        asdict = dict(df.items())
        asdict2 = {x: y.values for x, y in df.items()}

        # dict of series & dict of ndarrays (have dtype info)
        results = []
        results.append(DataFrame.from_records(asdict).reindex(columns=df.columns))
        results.append(
            DataFrame.from_records(asdict, columns=columns).reindex(columns=df.columns)
        )
        results.append(
            DataFrame.from_records(asdict2, columns=columns).reindex(columns=df.columns)
        )

        for r in results:
            tm.assert_frame_equal(r, df)

    def test_from_records_with_index_data(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["A", "B", "C"]
        )

        data = np.random.default_rng(2).standard_normal(10)
        with tm.assert_produces_warning(FutureWarning):
            df1 = DataFrame.from_records(df, index=data)
        tm.assert_index_equal(df1.index, Index(data))

    def test_from_records_bad_index_column(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["A", "B", "C"]
        )

        # should pass
        with tm.assert_produces_warning(FutureWarning):
            df1 = DataFrame.from_records(df, index=["C"])
        tm.assert_index_equal(df1.index, Index(df.C))

        with tm.assert_produces_warning(FutureWarning):
            df1 = DataFrame.from_records(df, index="C")
        tm.assert_index_equal(df1.index, Index(df.C))

        # should fail
        msg = "|".join(
            [
                r"'None of \[2\] are in the columns'",
            ]
        )
        with pytest.raises(KeyError, match=msg):
            with tm.assert_produces_warning(FutureWarning):
                DataFrame.from_records(df, index=[2])
        with pytest.raises(KeyError, match=msg):
            with tm.assert_produces_warning(FutureWarning):
                DataFrame.from_records(df, index=2)

    def test_from_records_non_tuple(self):
        class Record:
            def __init__(self, *args) -> None:
                self.args = args

            def __getitem__(self, i):
                return self.args[i]

            def __iter__(self) -> Iterator:
                return iter(self.args)

        recs = [Record(1, 2, 3), Record(4, 5, 6), Record(7, 8, 9)]
        tups = [tuple(rec) for rec in recs]

        result = DataFrame.from_records(recs)
        expected = DataFrame.from_records(tups)
        tm.assert_frame_equal(result, expected)

    def test_from_records_len0_with_columns(self):
        # GH#2633
        result = DataFrame.from_records([], index="foo", columns=["foo", "bar"])
        expected = Index(["bar"])

        assert len(result) == 0
        assert result.index.name == "foo"
        tm.assert_index_equal(result.columns, expected)

    def test_from_records_series_list_dict(self):
        # GH#27358
        expected = DataFrame([[{"a": 1, "b": 2}, {"a": 3, "b": 4}]]).T
        data = Series([[{"a": 1, "b": 2}], [{"a": 3, "b": 4}]])
        result = DataFrame.from_records(data)
        tm.assert_frame_equal(result, expected)

    def test_from_records_series_categorical_index(self):
        # GH#32805
        index = CategoricalIndex(
            [Interval(-20, -10), Interval(-10, 0), Interval(0, 10)]
        )
        series_of_dicts = Series([{"a": 1}, {"a": 2}, {"b": 3}], index=index)
        frame = DataFrame.from_records(series_of_dicts, index=index)
        expected = DataFrame(
            {"a": [1, 2, np.nan], "b": [np.nan, np.nan, 3]}, index=index
        )
        tm.assert_frame_equal(frame, expected)

    def test_frame_from_records_utc(self):
        rec = {"datum": 1.5, "begin_time": datetime(2006, 4, 27, tzinfo=pytz.utc)}

        # it works
        DataFrame.from_records([rec], index="begin_time")

    def test_from_records_to_records(self):
        # from numpy documentation
        arr = np.zeros((2,), dtype=("i4,f4,S10"))
        arr[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]

        DataFrame.from_records(arr)

        index = Index(np.arange(len(arr))[::-1])
        indexed_frame = DataFrame.from_records(arr, index=index)
        tm.assert_index_equal(indexed_frame.index, index)

        # without names, it should go to last ditch
        arr2 = np.zeros((2, 3))
        tm.assert_frame_equal(DataFrame.from_records(arr2), DataFrame(arr2))

        # wrong length
        msg = "|".join(
            [
                r"Length of values \(2\) does not match length of index \(1\)",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_records(arr, index=index[:-1])

        indexed_frame = DataFrame.from_records(arr, index="f1")

        # what to do?
        records = indexed_frame.to_records()
        assert len(records.dtype.names) == 3

        records = indexed_frame.to_records(index=False)
        assert len(records.dtype.names) == 2
        assert "index" not in records.dtype.names

    def test_from_records_nones(self):
        tuples = [(1, 2, None, 3), (1, 2, None, 3), (None, 2, 5, 3)]

        df = DataFrame.from_records(tuples, columns=["a", "b", "c", "d"])
        assert np.isnan(df["c"][0])

    def test_from_records_iterator(self):
        arr = np.array(
            [(1.0, 1.0, 2, 2), (3.0, 3.0, 4, 4), (5.0, 5.0, 6, 6), (7.0, 7.0, 8, 8)],
            dtype=[
                ("x", np.float64),
                ("u", np.float32),
                ("y", np.int64),
                ("z", np.int32),
            ],
        )
        df = DataFrame.from_records(iter(arr), nrows=2)
        xp = DataFrame(
            {
                "x": np.array([1.0, 3.0], dtype=np.float64),
                "u": np.array([1.0, 3.0], dtype=np.float32),
                "y": np.array([2, 4], dtype=np.int64),
                "z": np.array([2, 4], dtype=np.int32),
            }
        )
        tm.assert_frame_equal(df.reindex_like(xp), xp)

        # no dtypes specified here, so just compare with the default
        arr = [(1.0, 2), (3.0, 4), (5.0, 6), (7.0, 8)]
        df = DataFrame.from_records(iter(arr), columns=["x", "y"], nrows=2)
        tm.assert_frame_equal(df, xp.reindex(columns=["x", "y"]), check_dtype=False)

    def test_from_records_tuples_generator(self):
        def tuple_generator(length):
            for i in range(length):
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                yield (i, letters[i % len(letters)], i / length)

        columns_names = ["Integer", "String", "Float"]
        columns = [
            [i[j] for i in tuple_generator(10)] for j in range(len(columns_names))
        ]
        data = {"Integer": columns[0], "String": columns[1], "Float": columns[2]}
        expected = DataFrame(data, columns=columns_names)

        generator = tuple_generator(10)
        result = DataFrame.from_records(generator, columns=columns_names)
        tm.assert_frame_equal(result, expected)

    def test_from_records_lists_generator(self):
        def list_generator(length):
            for i in range(length):
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                yield [i, letters[i % len(letters)], i / length]

        columns_names = ["Integer", "String", "Float"]
        columns = [
            [i[j] for i in list_generator(10)] for j in range(len(columns_names))
        ]
        data = {"Integer": columns[0], "String": columns[1], "Float": columns[2]}
        expected = DataFrame(data, columns=columns_names)

        generator = list_generator(10)
        result = DataFrame.from_records(generator, columns=columns_names)
        tm.assert_frame_equal(result, expected)

    def test_from_records_columns_not_modified(self):
        tuples = [(1, 2, 3), (1, 2, 3), (2, 5, 3)]

        columns = ["a", "b", "c"]
        original_columns = list(columns)

        DataFrame.from_records(tuples, columns=columns, index="a")

        assert columns == original_columns

    def test_from_records_decimal(self):
        tuples = [(Decimal("1.5"),), (Decimal("2.5"),), (None,)]

        df = DataFrame.from_records(tuples, columns=["a"])
        assert df["a"].dtype == object

        df = DataFrame.from_records(tuples, columns=["a"], coerce_float=True)
        assert df["a"].dtype == np.float64
        assert np.isnan(df["a"].values[-1])

    def test_from_records_duplicates(self):
        result = DataFrame.from_records([(1, 2, 3), (4, 5, 6)], columns=["a", "b", "a"])

        expected = DataFrame([(1, 2, 3), (4, 5, 6)], columns=["a", "b", "a"])

        tm.assert_frame_equal(result, expected)

    def test_from_records_set_index_name(self):
        def create_dict(order_id):
            return {
                "order_id": order_id,
                "quantity": np.random.default_rng(2).integers(1, 10),
                "price": np.random.default_rng(2).integers(1, 10),
            }

        documents = [create_dict(i) for i in range(10)]
        # demo missing data
        documents.append({"order_id": 10, "quantity": 5})

        result = DataFrame.from_records(documents, index="order_id")
        assert result.index.name == "order_id"

        # MultiIndex
        result = DataFrame.from_records(documents, index=["order_id", "quantity"])
        assert result.index.names == ("order_id", "quantity")

    def test_from_records_misc_brokenness(self):
        # GH#2179

        data = {1: ["foo"], 2: ["bar"]}

        result = DataFrame.from_records(data, columns=["a", "b"])
        exp = DataFrame(data, columns=["a", "b"])
        tm.assert_frame_equal(result, exp)

        # overlap in index/index_names

        data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        result = DataFrame.from_records(data, index=["a", "b", "c"])
        exp = DataFrame(data, index=["a", "b", "c"])
        tm.assert_frame_equal(result, exp)

    def test_from_records_misc_brokenness2(self):
        # GH#2623
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), "hi"])  # test col upconverts to obj
        result = DataFrame.from_records(rows, columns=["date", "test"])
        expected = DataFrame(
            {"date": [row[0] for row in rows], "test": [row[1] for row in rows]}
        )
        tm.assert_frame_equal(result, expected)
        assert result.dtypes["test"] == np.dtype(object)

    def test_from_records_misc_brokenness3(self):
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), 1])
        result = DataFrame.from_records(rows, columns=["date", "test"])
        expected = DataFrame(
            {"date": [row[0] for row in rows], "test": [row[1] for row in rows]}
        )
        tm.assert_frame_equal(result, expected)

    def test_from_records_empty(self):
        # GH#3562
        result = DataFrame.from_records([], columns=["a", "b", "c"])
        expected = DataFrame(columns=["a", "b", "c"])
        tm.assert_frame_equal(result, expected)

        result = DataFrame.from_records([], columns=["a", "b", "b"])
        expected = DataFrame(columns=["a", "b", "b"])
        tm.assert_frame_equal(result, expected)

    def test_from_records_empty_with_nonempty_fields_gh3682(self):
        a = np.array([(1, 2)], dtype=[("id", np.int64), ("value", np.int64)])
        df = DataFrame.from_records(a, index="id")

        ex_index = Index([1], name="id")
        expected = DataFrame({"value": [2]}, index=ex_index, columns=["value"])
        tm.assert_frame_equal(df, expected)

        b = a[:0]
        df2 = DataFrame.from_records(b, index="id")
        tm.assert_frame_equal(df2, df.iloc[:0])

    def test_from_records_empty2(self):
        # GH#42456
        dtype = [("prop", int)]
        shape = (0, len(dtype))
        arr = np.empty(shape, dtype=dtype)

        result = DataFrame.from_records(arr)
        expected = DataFrame({"prop": np.array([], dtype=int)})
        tm.assert_frame_equal(result, expected)

        alt = DataFrame(arr)
        tm.assert_frame_equal(alt, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_order.py ===
from sympy.core.add import Add
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O, Order
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises
from sympy.abc import w, x, y, z
from sympy.testing.pytest import XFAIL


def test_caching_bug():
    #needs to be a first test, so that all caches are clean
    #cache it
    O(w)
    #and test that this won't raise an exception
    O(w**(-1/x/log(3)*log(5)), w)


def test_free_symbols():
    assert Order(1).free_symbols == set()
    assert Order(x).free_symbols == {x}
    assert Order(1, x).free_symbols == {x}
    assert Order(x*y).free_symbols == {x, y}
    assert Order(x, x, y).free_symbols == {x, y}


def test_simple_1():
    o = Rational(0)
    assert Order(2*x) == Order(x)
    assert Order(x)*3 == Order(x)
    assert -28*Order(x) == Order(x)
    assert Order(Order(x)) == Order(x)
    assert Order(Order(x), y) == Order(Order(x), x, y)
    assert Order(-23) == Order(1)
    assert Order(exp(x)) == Order(1, x)
    assert Order(exp(1/x)).expr == exp(1/x)
    assert Order(x*exp(1/x)).expr == x*exp(1/x)
    assert Order(x**(o/3)).expr == x**(o/3)
    assert Order(x**(o*Rational(5, 3))).expr == x**(o*Rational(5, 3))
    assert Order(x**2 + x + y, x) == O(1, x)
    assert Order(x**2 + x + y, y) == O(1, y)
    raises(ValueError, lambda: Order(exp(x), x, x))
    raises(TypeError, lambda: Order(x, 2 - x))


def test_simple_2():
    assert Order(2*x)*x == Order(x**2)
    assert Order(2*x)/x == Order(1, x)
    assert Order(2*x)*x*exp(1/x) == Order(x**2*exp(1/x))
    assert (Order(2*x)*x*exp(1/x)/log(x)**3).expr == x**2*exp(1/x)*log(x)**-3


def test_simple_3():
    assert Order(x) + x == Order(x)
    assert Order(x) + 2 == 2 + Order(x)
    assert Order(x) + x**2 == Order(x)
    assert Order(x) + 1/x == 1/x + Order(x)
    assert Order(1/x) + 1/x**2 == 1/x**2 + Order(1/x)
    assert Order(x) + exp(1/x) == Order(x) + exp(1/x)


def test_simple_4():
    assert Order(x)**2 == Order(x**2)


def test_simple_5():
    assert Order(x) + Order(x**2) == Order(x)
    assert Order(x) + Order(x**-2) == Order(x**-2)
    assert Order(x) + Order(1/x) == Order(1/x)


def test_simple_6():
    assert Order(x) - Order(x) == Order(x)
    assert Order(x) + Order(1) == Order(1)
    assert Order(x) + Order(x**2) == Order(x)
    assert Order(1/x) + Order(1) == Order(1/x)
    assert Order(x) + Order(exp(1/x)) == Order(exp(1/x))
    assert Order(x**3) + Order(exp(2/x)) == Order(exp(2/x))
    assert Order(x**-3) + Order(exp(2/x)) == Order(exp(2/x))


def test_simple_7():
    assert 1 + O(1) == O(1)
    assert 2 + O(1) == O(1)
    assert x + O(1) == O(1)
    assert 1/x + O(1) == 1/x + O(1)


def test_simple_8():
    assert O(sqrt(-x)) == O(sqrt(x))
    assert O(x**2*sqrt(x)) == O(x**Rational(5, 2))
    assert O(x**3*sqrt(-(-x)**3)) == O(x**Rational(9, 2))
    assert O(x**Rational(3, 2)*sqrt((-x)**3)) == O(x**3)
    assert O(x*(-2*x)**(I/2)) == O(x*(-x)**(I/2))


def test_as_expr_variables():
    assert Order(x).as_expr_variables(None) == (x, ((x, 0),))
    assert Order(x).as_expr_variables(((x, 0),)) == (x, ((x, 0),))
    assert Order(y).as_expr_variables(((x, 0),)) == (y, ((x, 0), (y, 0)))
    assert Order(y).as_expr_variables(((x, 0), (y, 0))) == (y, ((x, 0), (y, 0)))


def test_contains_0():
    assert Order(1, x).contains(Order(1, x))
    assert Order(1, x).contains(Order(1))
    assert Order(1).contains(Order(1, x)) is False


def test_contains_1():
    assert Order(x).contains(Order(x))
    assert Order(x).contains(Order(x**2))
    assert not Order(x**2).contains(Order(x))
    assert not Order(x).contains(Order(1/x))
    assert not Order(1/x).contains(Order(exp(1/x)))
    assert not Order(x).contains(Order(exp(1/x)))
    assert Order(1/x).contains(Order(x))
    assert Order(exp(1/x)).contains(Order(x))
    assert Order(exp(1/x)).contains(Order(1/x))
    assert Order(exp(1/x)).contains(Order(exp(1/x)))
    assert Order(exp(2/x)).contains(Order(exp(1/x)))
    assert not Order(exp(1/x)).contains(Order(exp(2/x)))


def test_contains_2():
    assert Order(x).contains(Order(y)) is None
    assert Order(x).contains(Order(y*x))
    assert Order(y*x).contains(Order(x))
    assert Order(y).contains(Order(x*y))
    assert Order(x).contains(Order(y**2*x))


def test_contains_3():
    assert Order(x*y**2).contains(Order(x**2*y)) is None
    assert Order(x**2*y).contains(Order(x*y**2)) is None


def test_contains_4():
    assert Order(sin(1/x**2)).contains(Order(cos(1/x**2))) is True
    assert Order(cos(1/x**2)).contains(Order(sin(1/x**2))) is True


def test_contains():
    assert Order(1, x) not in Order(1)
    assert Order(1) in Order(1, x)
    raises(TypeError, lambda: Order(x*y**2) in Order(x**2*y))


def test_add_1():
    assert Order(x + x) == Order(x)
    assert Order(3*x - 2*x**2) == Order(x)
    assert Order(1 + x) == Order(1, x)
    assert Order(1 + 1/x) == Order(1/x)
    # TODO : A better output for Order(log(x) + 1/log(x))
    # could be Order(log(x)). Currently Order for expressions
    # where all arguments would involve a log term would fall
    # in this category and outputs for these should be improved.
    assert Order(log(x) + 1/log(x)) == Order((log(x)**2 + 1)/log(x))
    assert Order(exp(1/x) + x) == Order(exp(1/x))
    assert Order(exp(1/x) + 1/x**20) == Order(exp(1/x))


def test_ln_args():
    assert O(log(x)) + O(log(2*x)) == O(log(x))
    assert O(log(x)) + O(log(x**3)) == O(log(x))
    assert O(log(x*y)) + O(log(x) + log(y)) == O(log(x) + log(y), x, y)


def test_multivar_0():
    assert Order(x*y).expr == x*y
    assert Order(x*y**2).expr == x*y**2
    assert Order(x*y, x).expr == x
    assert Order(x*y**2, y).expr == y**2
    assert Order(x*y*z).expr == x*y*z
    assert Order(x/y).expr == x/y
    assert Order(x*exp(1/y)).expr == x*exp(1/y)
    assert Order(exp(x)*exp(1/y)).expr == exp(x)*exp(1/y)


def test_multivar_0a():
    assert Order(exp(1/x)*exp(1/y)).expr == exp(1/x)*exp(1/y)


def test_multivar_1():
    assert Order(x + y).expr == x + y
    assert Order(x + 2*y).expr == x + y
    assert (Order(x + y) + x).expr == (x + y)
    assert (Order(x + y) + x**2) == Order(x + y)
    assert (Order(x + y) + 1/x) == 1/x + Order(x + y)
    assert Order(x**2 + y*x).expr == x**2 + y*x


def test_multivar_2():
    assert Order(x**2*y + y**2*x, x, y).expr == x**2*y + y**2*x


def test_multivar_mul_1():
    assert Order(x + y)*x == Order(x**2 + y*x, x, y)


def test_multivar_3():
    assert (Order(x) + Order(y)).args in [
        (Order(x), Order(y)),
        (Order(y), Order(x))]
    assert Order(x) + Order(y) + Order(x + y) == Order(x + y)
    assert (Order(x**2*y) + Order(y**2*x)).args in [
        (Order(x*y**2), Order(y*x**2)),
        (Order(y*x**2), Order(x*y**2))]
    assert (Order(x**2*y) + Order(y*x)) == Order(x*y)


def test_issue_3468():
    y = Symbol('y', negative=True)
    z = Symbol('z', complex=True)

    # check that Order does not modify assumptions about symbols
    Order(x)
    Order(y)
    Order(z)

    assert x.is_positive is None
    assert y.is_positive is False
    assert z.is_positive is None


def test_leading_order():
    assert (x + 1 + 1/x**5).extract_leading_order(x) == ((1/x**5, O(1/x**5)),)
    assert (1 + 1/x).extract_leading_order(x) == ((1/x, O(1/x)),)
    assert (1 + x).extract_leading_order(x) == ((1, O(1, x)),)
    assert (1 + x**2).extract_leading_order(x) == ((1, O(1, x)),)
    assert (2 + x**2).extract_leading_order(x) == ((2, O(1, x)),)
    assert (x + x**2).extract_leading_order(x) == ((x, O(x)),)


def test_leading_order2():
    assert set((2 + pi + x**2).extract_leading_order(x)) == {(pi, O(1, x)),
            (S(2), O(1, x))}
    assert set((2*x + pi*x + x**2).extract_leading_order(x)) == {(2*x, O(x)),
            (x*pi, O(x))}


def test_order_leadterm():
    assert O(x**2)._eval_as_leading_term(x, None, 1) == O(x**2)


def test_order_symbols():
    e = x*y*sin(x)*Integral(x, (x, 1, 2))
    assert O(e) == O(x**2*y, x, y)
    assert O(e, x) == O(x**2)


def test_nan():
    assert O(nan) is nan
    assert not O(x).contains(nan)


def test_O1():
    assert O(1, x) * x == O(x)
    assert O(1, y) * x == O(1, y)


def test_getn():
    # other lines are tested incidentally by the suite
    assert O(x).getn() == 1
    assert O(x/log(x)).getn() == 1
    assert O(x**2/log(x)**2).getn() == 2
    assert O(x*log(x)).getn() == 1
    raises(NotImplementedError, lambda: (O(x) + O(y)).getn())


def test_diff():
    assert O(x**2).diff(x) == O(x)


def test_getO():
    assert (x).getO() is None
    assert (x).removeO() == x
    assert (O(x)).getO() == O(x)
    assert (O(x)).removeO() == 0
    assert (z + O(x) + O(y)).getO() == O(x) + O(y)
    assert (z + O(x) + O(y)).removeO() == z
    raises(NotImplementedError, lambda: (O(x) + O(y)).getn())


def test_leading_term():
    from sympy.functions.special.gamma_functions import digamma
    assert O(1/digamma(1/x)) == O(1/log(x))


def test_eval():
    assert Order(x).subs(Order(x), 1) == 1
    assert Order(x).subs(x, y) == Order(y)
    assert Order(x).subs(y, x) == Order(x)
    assert Order(x).subs(x, x + y) == Order(x + y, (x, -y))
    assert (O(1)**x).is_Pow


def test_issue_4279():
    a, b = symbols('a b')
    assert O(a, a, b) + O(1, a, b) == O(1, a, b)
    assert O(b, a, b) + O(1, a, b) == O(1, a, b)
    assert O(a + b, a, b) + O(1, a, b) == O(1, a, b)
    assert O(1, a, b) + O(a, a, b) == O(1, a, b)
    assert O(1, a, b) + O(b, a, b) == O(1, a, b)
    assert O(1, a, b) + O(a + b, a, b) == O(1, a, b)


def test_issue_4855():
    assert 1/O(1) != O(1)
    assert 1/O(x) != O(1/x)
    assert 1/O(x, (x, oo)) != O(1/x, (x, oo))

    f = Function('f')
    assert 1/O(f(x)) != O(1/x)


def test_order_conjugate_transpose():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    assert conjugate(Order(x)) == Order(conjugate(x))
    assert conjugate(Order(y)) == Order(conjugate(y))
    assert conjugate(Order(x**2)) == Order(conjugate(x)**2)
    assert conjugate(Order(y**2)) == Order(conjugate(y)**2)
    assert transpose(Order(x)) == Order(transpose(x))
    assert transpose(Order(y)) == Order(transpose(y))
    assert transpose(Order(x**2)) == Order(transpose(x)**2)
    assert transpose(Order(y**2)) == Order(transpose(y)**2)


def test_order_noncommutative():
    A = Symbol('A', commutative=False)
    assert Order(A + A*x, x) == Order(1, x)
    assert (A + A*x)*Order(x) == Order(x)
    assert (A*x)*Order(x) == Order(x**2, x)
    assert expand((1 + Order(x))*A*A*x) == A*A*x + Order(x**2, x)
    assert expand((A*A + Order(x))*x) == A*A*x + Order(x**2, x)
    assert expand((A + Order(x))*A*x) == A*A*x + Order(x**2, x)


def test_issue_6753():
    assert (1 + x**2)**10000*O(x) == O(x)


def test_order_at_infinity():
    assert Order(1 + x, (x, oo)) == Order(x, (x, oo))
    assert Order(3*x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo))*3 == Order(x, (x, oo))
    assert -28*Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(Order(x, (x, oo)), (x, oo)) == Order(x, (x, oo))
    assert Order(Order(x, (x, oo)), (y, oo)) == Order(x, (x, oo), (y, oo))
    assert Order(3, (x, oo)) == Order(1, (x, oo))
    assert Order(x**2 + x + y, (x, oo)) == O(x**2, (x, oo))
    assert Order(x**2 + x + y, (y, oo)) == O(y, (y, oo))

    assert Order(2*x, (x, oo))*x == Order(x**2, (x, oo))
    assert Order(2*x, (x, oo))/x == Order(1, (x, oo))
    assert Order(2*x, (x, oo))*x*exp(1/x) == Order(x**2*exp(1/x), (x, oo))
    assert Order(2*x, (x, oo))*x*exp(1/x)/log(x)**3 == Order(x**2*exp(1/x)*log(x)**-3, (x, oo))

    assert Order(x, (x, oo)) + 1/x == 1/x + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + 1 == 1 + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + x == x + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + x**2 == x**2 + Order(x, (x, oo))
    assert Order(1/x, (x, oo)) + 1/x**2 == 1/x**2 + Order(1/x, (x, oo)) == Order(1/x, (x, oo))
    assert Order(x, (x, oo)) + exp(1/x) == exp(1/x) + Order(x, (x, oo))

    assert Order(x, (x, oo))**2 == Order(x**2, (x, oo))

    assert Order(x, (x, oo)) + Order(x**2, (x, oo)) == Order(x**2, (x, oo))
    assert Order(x, (x, oo)) + Order(x**-2, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(1/x, (x, oo)) == Order(x, (x, oo))

    assert Order(x, (x, oo)) - Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(1, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(x**2, (x, oo)) == Order(x**2, (x, oo))
    assert Order(1/x, (x, oo)) + Order(1, (x, oo)) == Order(1, (x, oo))
    assert Order(x, (x, oo)) + Order(exp(1/x), (x, oo)) == Order(x, (x, oo))
    assert Order(x**3, (x, oo)) + Order(exp(2/x), (x, oo)) == Order(x**3, (x, oo))
    assert Order(x**-3, (x, oo)) + Order(exp(2/x), (x, oo)) == Order(exp(2/x), (x, oo))

    # issue 7207
    assert Order(exp(x), (x, oo)).expr == Order(2*exp(x), (x, oo)).expr == exp(x)
    assert Order(y**x, (x, oo)).expr == Order(2*y**x, (x, oo)).expr == exp(x*log(y))

    # issue 19545
    assert Order(1/x - 3/(3*x + 2), (x, oo)).expr == x**(-2)

def test_mixing_order_at_zero_and_infinity():
    assert (Order(x, (x, 0)) + Order(x, (x, oo))).is_Add
    assert Order(x, (x, 0)) + Order(x, (x, oo)) == Order(x, (x, oo)) + Order(x, (x, 0))
    assert Order(Order(x, (x, oo))) == Order(x, (x, oo))

    # not supported (yet)
    raises(NotImplementedError, lambda: Order(x, (x, 0))*Order(x, (x, oo)))
    raises(NotImplementedError, lambda: Order(x, (x, oo))*Order(x, (x, 0)))
    raises(NotImplementedError, lambda: Order(Order(x, (x, oo)), y))
    raises(NotImplementedError, lambda: Order(Order(x), (x, oo)))


def test_order_at_some_point():
    assert Order(x, (x, 1)) == Order(1, (x, 1))
    assert Order(2*x - 2, (x, 1)) == Order(x - 1, (x, 1))
    assert Order(-x + 1, (x, 1)) == Order(x - 1, (x, 1))
    assert Order(x - 1, (x, 1))**2 == Order((x - 1)**2, (x, 1))
    assert Order(x - 2, (x, 2)) - O(x - 2, (x, 2)) == Order(x - 2, (x, 2))


def test_order_subs_limits():
    # issue 3333
    assert (1 + Order(x)).subs(x, 1/x) == 1 + Order(1/x, (x, oo))
    assert (1 + Order(x)).limit(x, 0) == 1
    # issue 5769
    assert ((x + Order(x**2))/x).limit(x, 0) == 1

    assert Order(x**2).subs(x, y - 1) == Order((y - 1)**2, (y, 1))
    assert Order(10*x**2, (x, 2)).subs(x, y - 1) == Order(1, (y, 3))

    #issue 19120
    assert O(x).subs(x, O(x)) == O(x)
    assert O(x**2).subs(x, x + O(x)) == O(x**2)
    assert O(x, (x, oo)).subs(x, O(x, (x, oo))) == O(x, (x, oo))
    assert O(x**2, (x, oo)).subs(x, x + O(x, (x, oo))) == O(x**2, (x, oo))
    assert (x + O(x**2)).subs(x, x + O(x**2)) == x + O(x**2)
    assert (x**2 + O(x**2) + 1/x**2).subs(x, x + O(x**2)) == (x + O(x**2))**(-2) + O(x**2)
    assert (x**2 + O(x**2) + 1).subs(x, x + O(x**2)) == 1 + O(x**2)
    assert O(x, (x, oo)).subs(x, x + O(x**2, (x, oo))) == O(x**2, (x, oo))
    assert sin(x).series(n=8).subs(x,sin(x).series(n=8)).expand() == x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8)
    assert cos(x).series(n=8).subs(x,sin(x).series(n=8)).expand() == 1 - x**2/2 + 5*x**4/24 - 37*x**6/720 + O(x**8)
    assert O(x).subs(x, O(1/x, (x, oo))) == O(1/x, (x, oo))

@XFAIL
def test_order_failing_due_to_solveset():
    assert O(x**3).subs(x, exp(-x**2)) == O(exp(-3*x**2), (x, -oo))
    raises(NotImplementedError, lambda: O(x).subs(x, O(1/x))) # mixing of order at different points


def test_issue_9351():
    assert exp(x).series(x, 10, 1) == exp(10) + Order(x - 10, (x, 10))


def test_issue_9192():
    assert O(1)*O(1) == O(1)
    assert O(1)**O(1) == O(1)


def test_issue_9910():
    assert O(x*log(x) + sin(x), (x, oo)) == O(x*log(x), (x, oo))


def test_performance_of_adding_order():
    l = [x**i for i in range(1000)]
    l.append(O(x**1001))
    assert Add(*l).subs(x,1) == O(1)

def test_issue_14622():
    assert (x**(-4) + x**(-3) + x**(-1) + O(x**(-6), (x, oo))).as_numer_denom() == (
        x**4 + x**5 + x**7 + O(x**2, (x, oo)), x**8)
    assert (x**3 + O(x**2, (x, oo))).is_Add
    assert O(x**2, (x, oo)).contains(x**3) is False
    assert O(x, (x, oo)).contains(O(x, (x, 0))) is None
    assert O(x, (x, 0)).contains(O(x, (x, oo))) is None
    raises(NotImplementedError, lambda: O(x**3).contains(x**w))


def test_issue_15539():
    assert O(1/x**2 + 1/x**4, (x, -oo)) == O(1/x**2, (x, -oo))
    assert O(1/x**4 + exp(x), (x, -oo)) == O(1/x**4, (x, -oo))
    assert O(1/x**4 + exp(-x), (x, -oo)) == O(exp(-x), (x, -oo))
    assert O(1/x, (x, oo)).subs(x, -x) == O(-1/x, (x, -oo))

def test_issue_18606():
    assert unchanged(Order, 0)


def test_issue_22165():
    assert O(log(x)).contains(2)


def test_issue_23231():
    # This test checks Order for expressions having
    # arguments containing variables in exponents/powers.
    assert O(x**x + 2**x, (x, oo)) == O(exp(x*log(x)), (x, oo))
    assert O(x**x + x**2, (x, oo)) == O(exp(x*log(x)), (x, oo))
    assert O(x**x + 1/x**2, (x, oo)) == O(exp(x*log(x)), (x, oo))
    assert O(2**x + 3**x , (x, oo)) == O(exp(x*log(3)), (x, oo))


def test_issue_9917():
    assert O(x*sin(x) + 1, (x, oo)) == O(x, (x, oo))


def test_issue_22836():
    assert O(2**x + factorial(x), (x, oo)) == O(factorial(x), (x, oo))
    assert O(2**x + factorial(x) + x**x, (x, oo)) == O(exp(x*log(x)), (x, oo))
    assert O(x + factorial(x), (x, oo)) == O(factorial(x), (x, oo))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py ===
import re

import numpy as np
import pytest

from pandas.compat import PY311

from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    Series,
    StringDtype,
)
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories


class TestCategoricalAPI:
    def test_to_list_deprecated(self):
        # GH#51254
        cat1 = Categorical(list("acb"), ordered=False)
        msg = "Categorical.to_list is deprecated and will be removed"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            cat1.to_list()

    def test_ordered_api(self):
        # GH 9347
        cat1 = Categorical(list("acb"), ordered=False)
        tm.assert_index_equal(cat1.categories, Index(["a", "b", "c"]))
        assert not cat1.ordered

        cat2 = Categorical(list("acb"), categories=list("bca"), ordered=False)
        tm.assert_index_equal(cat2.categories, Index(["b", "c", "a"]))
        assert not cat2.ordered

        cat3 = Categorical(list("acb"), ordered=True)
        tm.assert_index_equal(cat3.categories, Index(["a", "b", "c"]))
        assert cat3.ordered

        cat4 = Categorical(list("acb"), categories=list("bca"), ordered=True)
        tm.assert_index_equal(cat4.categories, Index(["b", "c", "a"]))
        assert cat4.ordered

    def test_set_ordered(self):
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        cat2 = cat.as_unordered()
        assert not cat2.ordered
        cat2 = cat.as_ordered()
        assert cat2.ordered

        assert cat2.set_ordered(True).ordered
        assert not cat2.set_ordered(False).ordered

        # removed in 0.19.0
        msg = (
            "property 'ordered' of 'Categorical' object has no setter"
            if PY311
            else "can't set attribute"
        )
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = True
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = False

    def test_rename_categories(self):
        cat = Categorical(["a", "b", "c", "a"])

        # inplace=False: the old one must not be changed
        res = cat.rename_categories([1, 2, 3])
        tm.assert_numpy_array_equal(
            res.__array__(), np.array([1, 2, 3, 1], dtype=np.int64)
        )
        tm.assert_index_equal(res.categories, Index([1, 2, 3]))

        exp_cat = np.array(["a", "b", "c", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        exp_cat = Index(["a", "b", "c"])
        tm.assert_index_equal(cat.categories, exp_cat)

        # GH18862 (let rename_categories take callables)
        result = cat.rename_categories(lambda x: x.upper())
        expected = Categorical(["A", "B", "C", "A"])
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("new_categories", [[1, 2, 3, 4], [1, 2]])
    def test_rename_categories_wrong_length_raises(self, new_categories):
        cat = Categorical(["a", "b", "c", "a"])
        msg = (
            "new categories need to have the same number of items as the "
            "old categories!"
        )
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)

    def test_rename_categories_series(self):
        # https://github.com/pandas-dev/pandas/issues/17981
        c = Categorical(["a", "b"])
        result = c.rename_categories(Series([0, 1], index=["a", "b"]))
        expected = Categorical([0, 1])
        tm.assert_categorical_equal(result, expected)

    def test_rename_categories_dict(self):
        # GH 17336
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 4, "b": 3, "c": 2, "d": 1})
        expected = Index([4, 3, 2, 1])
        tm.assert_index_equal(res.categories, expected)

        # Test for dicts of smaller length
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 1, "c": 3})

        expected = Index([1, "b", 3, "d"])
        tm.assert_index_equal(res.categories, expected)

        # Test for dicts with bigger length
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6})
        expected = Index([1, 2, 3, 4])
        tm.assert_index_equal(res.categories, expected)

        # Test for dicts with no items from old categories
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"f": 1, "g": 3})

        expected = Index(["a", "b", "c", "d"])
        tm.assert_index_equal(res.categories, expected)

    def test_reorder_categories(self):
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        old = cat.copy()
        new = Categorical(
            ["a", "b", "c", "a"], categories=["c", "b", "a"], ordered=True
        )

        res = cat.reorder_categories(["c", "b", "a"])
        # cat must be the same as before
        tm.assert_categorical_equal(cat, old)
        # only res is changed
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize(
        "new_categories",
        [
            ["a"],  # not all "old" included in "new"
            ["a", "b", "d"],  # still not all "old" in "new"
            ["a", "b", "c", "d"],  # all "old" included in "new", but too long
        ],
    )
    def test_reorder_categories_raises(self, new_categories):
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        msg = "items in new_categories are not the same as in old categories"
        with pytest.raises(ValueError, match=msg):
            cat.reorder_categories(new_categories)

    def test_add_categories(self):
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        old = cat.copy()
        new = Categorical(
            ["a", "b", "c", "a"], categories=["a", "b", "c", "d"], ordered=True
        )

        res = cat.add_categories("d")
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        res = cat.add_categories(["d"])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        # GH 9927
        cat = Categorical(list("abc"), ordered=True)
        expected = Categorical(list("abc"), categories=list("abcde"), ordered=True)
        # test with Series, np.array, index, list
        res = cat.add_categories(Series(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(np.array(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(Index(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(["d", "e"])
        tm.assert_categorical_equal(res, expected)

    def test_add_categories_existing_raises(self):
        # new is in old categories
        cat = Categorical(["a", "b", "c", "d"], ordered=True)
        msg = re.escape("new categories must not include old categories: {'d'}")
        with pytest.raises(ValueError, match=msg):
            cat.add_categories(["d"])

    def test_add_categories_losing_dtype_information(self):
        # GH#48812
        cat = Categorical(Series([1, 2], dtype="Int64"))
        ser = Series([4], dtype="Int64")
        result = cat.add_categories(ser)
        expected = Categorical(
            Series([1, 2], dtype="Int64"), categories=Series([1, 2, 4], dtype="Int64")
        )
        tm.assert_categorical_equal(result, expected)

        cat = Categorical(Series(["a", "b", "a"], dtype=StringDtype()))
        ser = Series(["d"], dtype=StringDtype())
        result = cat.add_categories(ser)
        expected = Categorical(
            Series(["a", "b", "a"], dtype=StringDtype()),
            categories=Series(["a", "b", "d"], dtype=StringDtype()),
        )
        tm.assert_categorical_equal(result, expected)

    def test_set_categories(self):
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        exp_categories = Index(["c", "b", "a"])
        exp_values = np.array(["a", "b", "c", "a"], dtype=np.object_)

        cat = cat.set_categories(["c", "b", "a"])
        res = cat.set_categories(["a", "b", "c"])
        # cat must be the same as before
        tm.assert_index_equal(cat.categories, exp_categories)
        tm.assert_numpy_array_equal(cat.__array__(), exp_values)
        # only res is changed
        exp_categories_back = Index(["a", "b", "c"])
        tm.assert_index_equal(res.categories, exp_categories_back)
        tm.assert_numpy_array_equal(res.__array__(), exp_values)

        # not all "old" included in "new" -> all not included ones are now
        # np.nan
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        res = cat.set_categories(["a"])
        tm.assert_numpy_array_equal(res.codes, np.array([0, -1, -1, 0], dtype=np.int8))

        # still not all "old" in "new"
        res = cat.set_categories(["a", "b", "d"])
        tm.assert_numpy_array_equal(res.codes, np.array([0, 1, -1, 0], dtype=np.int8))
        tm.assert_index_equal(res.categories, Index(["a", "b", "d"]))

        # all "old" included in "new"
        cat = cat.set_categories(["a", "b", "c", "d"])
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_index_equal(cat.categories, exp_categories)

        # internals...
        c = Categorical([1, 2, 3, 4, 1], categories=[1, 2, 3, 4], ordered=True)
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, 2, 3, 0], dtype=np.int8))
        tm.assert_index_equal(c.categories, Index([1, 2, 3, 4]))

        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)

        # all "pointers" to '4' must be changed from 3 to 0,...
        c = c.set_categories([4, 3, 2, 1])

        # positions are changed
        tm.assert_numpy_array_equal(c._codes, np.array([3, 2, 1, 0, 3], dtype=np.int8))

        # categories are now in new order
        tm.assert_index_equal(c.categories, Index([4, 3, 2, 1]))

        # output is the same
        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)
        assert c.min() == 4
        assert c.max() == 1

        # set_categories should set the ordering if specified
        c2 = c.set_categories([4, 3, 2, 1], ordered=False)
        assert not c2.ordered

        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))

        # set_categories should pass thru the ordering
        c2 = c.set_ordered(False).set_categories([4, 3, 2, 1])
        assert not c2.ordered

        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))

    @pytest.mark.parametrize(
        "values, categories, new_categories",
        [
            # No NaNs, same cats, same order
            (["a", "b", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["a", "b", "a"], ["a", "b"], ["b", "a"]),
            # Same, unsorted
            (["b", "a", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["b", "a", "a"], ["a", "b"], ["b", "a"]),
            # NaNs
            (["a", "b", "c"], ["a", "b"], ["a", "b"]),
            (["a", "b", "c"], ["a", "b"], ["b", "a"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            # Introduce NaNs
            (["a", "b", "c"], ["a", "b"], ["a"]),
            (["a", "b", "c"], ["a", "b"], ["b"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            # No overlap
            (["a", "b", "c"], ["a", "b"], ["d", "e"]),
        ],
    )
    @pytest.mark.parametrize("ordered", [True, False])
    def test_set_categories_many(self, values, categories, new_categories, ordered):
        c = Categorical(values, categories)
        expected = Categorical(values, new_categories, ordered)
        result = c.set_categories(new_categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_rename_less(self):
        # GH 24675
        cat = Categorical(["A", "B"])
        result = cat.set_categories(["A"], rename=True)
        expected = Categorical(["A", np.nan])
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_private(self):
        cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
        cat._set_categories(["a", "c", "d", "e"])
        expected = Categorical(["a", "c", "d"], categories=list("acde"))
        tm.assert_categorical_equal(cat, expected)

        # fastpath
        cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
        cat._set_categories(["a", "c", "d", "e"], fastpath=True)
        expected = Categorical(["a", "c", "d"], categories=list("acde"))
        tm.assert_categorical_equal(cat, expected)

    def test_remove_categories(self):
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        old = cat.copy()
        new = Categorical(["a", "b", np.nan, "a"], categories=["a", "b"], ordered=True)

        res = cat.remove_categories("c")
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        res = cat.remove_categories(["c"])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize("removals", [["c"], ["c", np.nan], "c", ["c", "c"]])
    def test_remove_categories_raises(self, removals):
        cat = Categorical(["a", "b", "a"])
        message = re.escape("removals must all be in old categories: {'c'}")

        with pytest.raises(ValueError, match=message):
            cat.remove_categories(removals)

    def test_remove_unused_categories(self):
        c = Categorical(["a", "b", "c", "d", "a"], categories=["a", "b", "c", "d", "e"])
        exp_categories_all = Index(["a", "b", "c", "d", "e"])
        exp_categories_dropped = Index(["a", "b", "c", "d"])

        tm.assert_index_equal(c.categories, exp_categories_all)

        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, exp_categories_dropped)
        tm.assert_index_equal(c.categories, exp_categories_all)

        # with NaN values (GH11599)
        c = Categorical(["a", "b", "c", np.nan], categories=["a", "b", "c", "d", "e"])
        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, Index(np.array(["a", "b", "c"])))
        exp_codes = np.array([0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(res.codes, exp_codes)
        tm.assert_index_equal(c.categories, exp_categories_all)

        val = ["F", np.nan, "D", "B", "D", "F", np.nan]
        cat = Categorical(values=val, categories=list("ABCDEFG"))
        out = cat.remove_unused_categories()
        tm.assert_index_equal(out.categories, Index(["B", "D", "F"]))
        exp_codes = np.array([2, -1, 1, 0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(out.codes, exp_codes)
        assert out.tolist() == val

        alpha = list("abcdefghijklmnopqrstuvwxyz")
        val = np.random.default_rng(2).choice(alpha[::2], 10000).astype("object")
        val[np.random.default_rng(2).choice(len(val), 100)] = np.nan

        cat = Categorical(values=val, categories=alpha)
        out = cat.remove_unused_categories()
        assert out.tolist() == val.tolist()


class TestCategoricalAPIWithFactor:
    def test_describe(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        # string type
        desc = factor.describe()
        assert factor.ordered
        exp_index = CategoricalIndex(
            ["a", "b", "c"], name="categories", ordered=factor.ordered
        )
        expected = DataFrame(
            {"counts": [3, 2, 3], "freqs": [3 / 8.0, 2 / 8.0, 3 / 8.0]}, index=exp_index
        )
        tm.assert_frame_equal(desc, expected)

        # check unused categories
        cat = factor.copy()
        cat = cat.set_categories(["a", "b", "c", "d"])
        desc = cat.describe()

        exp_index = CategoricalIndex(
            list("abcd"), ordered=factor.ordered, name="categories"
        )
        expected = DataFrame(
            {"counts": [3, 2, 3, 0], "freqs": [3 / 8.0, 2 / 8.0, 3 / 8.0, 0]},
            index=exp_index,
        )
        tm.assert_frame_equal(desc, expected)

        # check an integer one
        cat = Categorical([1, 2, 3, 1, 2, 3, 3, 2, 1, 1, 1])
        desc = cat.describe()
        exp_index = CategoricalIndex([1, 2, 3], ordered=cat.ordered, name="categories")
        expected = DataFrame(
            {"counts": [5, 3, 3], "freqs": [5 / 11.0, 3 / 11.0, 3 / 11.0]},
            index=exp_index,
        )
        tm.assert_frame_equal(desc, expected)

        # https://github.com/pandas-dev/pandas/issues/3678
        # describe should work with NaN
        cat = Categorical([np.nan, 1, 2, 2])
        desc = cat.describe()
        expected = DataFrame(
            {"counts": [1, 2, 1], "freqs": [1 / 4.0, 2 / 4.0, 1 / 4.0]},
            index=CategoricalIndex(
                [1, 2, np.nan], categories=[1, 2], name="categories"
            ),
        )
        tm.assert_frame_equal(desc, expected)


class TestPrivateCategoricalAPI:
    def test_codes_immutable(self):
        # Codes should be read only
        c = Categorical(["a", "b", "c", "a", np.nan])
        exp = np.array([0, 1, 2, 0, -1], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)

        # Assignments to codes should raise
        msg = (
            "property 'codes' of 'Categorical' object has no setter"
            if PY311
            else "can't set attribute"
        )
        with pytest.raises(AttributeError, match=msg):
            c.codes = np.array([0, 1, 2, 0, 1], dtype="int8")

        # changes in the codes array should raise
        codes = c.codes

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            codes[4] = 1

        # But even after getting the codes, the original array should still be
        # writeable!
        c[4] = "a"
        exp = np.array([0, 1, 2, 0, 0], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)
        c._codes[4] = 2
        exp = np.array([0, 1, 2, 0, 2], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)

    @pytest.mark.parametrize(
        "codes, old, new, expected",
        [
            ([0, 1], ["a", "b"], ["a", "b"], [0, 1]),
            ([0, 1], ["b", "a"], ["b", "a"], [0, 1]),
            ([0, 1], ["a", "b"], ["b", "a"], [1, 0]),
            ([0, 1], ["b", "a"], ["a", "b"], [1, 0]),
            ([0, 1, 0, 1], ["a", "b"], ["a", "b", "c"], [0, 1, 0, 1]),
            ([0, 1, 2, 2], ["a", "b", "c"], ["a", "b"], [0, 1, -1, -1]),
            ([0, 1, -1], ["a", "b", "c"], ["a", "b", "c"], [0, 1, -1]),
            ([0, 1, -1], ["a", "b", "c"], ["b"], [-1, 0, -1]),
            ([0, 1, -1], ["a", "b", "c"], ["d"], [-1, -1, -1]),
            ([0, 1, -1], ["a", "b", "c"], [], [-1, -1, -1]),
            ([-1, -1], [], ["a", "b"], [-1, -1]),
            ([1, 0], ["b", "a"], ["a", "b"], [0, 1]),
        ],
    )
    def test_recode_to_categories(self, codes, old, new, expected):
        codes = np.asanyarray(codes, dtype=np.int8)
        expected = np.asanyarray(expected, dtype=np.int8)
        old = Index(old)
        new = Index(new)
        result = recode_for_categories(codes, old, new)
        tm.assert_numpy_array_equal(result, expected)

    def test_recode_to_categories_large(self):
        N = 1000
        codes = np.arange(N)
        old = Index(codes)
        expected = np.arange(N - 1, -1, -1, dtype=np.int16)
        new = Index(expected)
        result = recode_for_categories(codes, old, new)
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_inequalities.py ===
"""Tests for tools for solving inequalities and systems of inequalities. """

from sympy.concrete.summations import Sum
from sympy.core.function import Function
from sympy.core.numbers import I, Rational, oo, pi
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Or
from sympy.polys.polytools import Poly, PurePoly
from sympy.sets.sets import FiniteSet, Interval, Union
from sympy.solvers.inequalities import (reduce_inequalities,
                                        solve_poly_inequality as psolve,
                                        reduce_rational_inequalities,
                                        solve_univariate_inequality as isolve,
                                        reduce_abs_inequality,
                                        _solve_inequality)
from sympy.polys.rootoftools import rootof
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import solveset
from sympy.core.mod import Mod
from sympy.abc import x, y

from sympy.testing.pytest import raises, XFAIL


inf = oo.evalf()


def test_solve_poly_inequality():
    assert psolve(Poly(0, x), '==') == [S.Reals]
    assert psolve(Poly(1, x), '==') == [S.EmptySet]
    assert psolve(PurePoly(x + 1, x), ">") == [Interval(-1, oo, True, False)]


def test_reduce_poly_inequalities_real_interval():
    assert reduce_rational_inequalities(
        [[Eq(x**2, 0)]], x, relational=False) == FiniteSet(0)
    assert reduce_rational_inequalities(
        [[Le(x**2, 0)]], x, relational=False) == FiniteSet(0)
    assert reduce_rational_inequalities(
        [[Lt(x**2, 0)]], x, relational=False) == S.EmptySet
    assert reduce_rational_inequalities(
        [[Ge(x**2, 0)]], x, relational=False) == \
        S.Reals if x.is_real else Interval(-oo, oo)
    assert reduce_rational_inequalities(
        [[Gt(x**2, 0)]], x, relational=False) == \
        FiniteSet(0).complement(S.Reals)
    assert reduce_rational_inequalities(
        [[Ne(x**2, 0)]], x, relational=False) == \
        FiniteSet(0).complement(S.Reals)

    assert reduce_rational_inequalities(
        [[Eq(x**2, 1)]], x, relational=False) == FiniteSet(-1, 1)
    assert reduce_rational_inequalities(
        [[Le(x**2, 1)]], x, relational=False) == Interval(-1, 1)
    assert reduce_rational_inequalities(
        [[Lt(x**2, 1)]], x, relational=False) == Interval(-1, 1, True, True)
    assert reduce_rational_inequalities(
        [[Ge(x**2, 1)]], x, relational=False) == \
        Union(Interval(-oo, -1), Interval(1, oo))
    assert reduce_rational_inequalities(
        [[Gt(x**2, 1)]], x, relational=False) == \
        Interval(-1, 1).complement(S.Reals)
    assert reduce_rational_inequalities(
        [[Ne(x**2, 1)]], x, relational=False) == \
        FiniteSet(-1, 1).complement(S.Reals)
    assert reduce_rational_inequalities([[Eq(
        x**2, 1.0)]], x, relational=False) == FiniteSet(-1.0, 1.0).evalf()
    assert reduce_rational_inequalities(
        [[Le(x**2, 1.0)]], x, relational=False) == Interval(-1.0, 1.0)
    assert reduce_rational_inequalities([[Lt(
        x**2, 1.0)]], x, relational=False) == Interval(-1.0, 1.0, True, True)
    assert reduce_rational_inequalities(
        [[Ge(x**2, 1.0)]], x, relational=False) == \
        Union(Interval(-inf, -1.0), Interval(1.0, inf))
    assert reduce_rational_inequalities(
        [[Gt(x**2, 1.0)]], x, relational=False) == \
        Union(Interval(-inf, -1.0, right_open=True),
        Interval(1.0, inf, left_open=True))
    assert reduce_rational_inequalities([[Ne(
        x**2, 1.0)]], x, relational=False) == \
        FiniteSet(-1.0, 1.0).complement(S.Reals)

    s = sqrt(2)

    assert reduce_rational_inequalities([[Lt(
        x**2 - 1, 0), Gt(x**2 - 1, 0)]], x, relational=False) == S.EmptySet
    assert reduce_rational_inequalities([[Le(x**2 - 1, 0), Ge(
        x**2 - 1, 0)]], x, relational=False) == FiniteSet(-1, 1)
    assert reduce_rational_inequalities(
        [[Le(x**2 - 2, 0), Ge(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, False, False), Interval(1, s, False, False))
    assert reduce_rational_inequalities(
        [[Le(x**2 - 2, 0), Gt(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, False, True), Interval(1, s, True, False))
    assert reduce_rational_inequalities(
        [[Lt(x**2 - 2, 0), Ge(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, True, False), Interval(1, s, False, True))
    assert reduce_rational_inequalities(
        [[Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, True, True), Interval(1, s, True, True))
    assert reduce_rational_inequalities(
        [[Lt(x**2 - 2, 0), Ne(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, True, True), Interval(-1, 1, True, True),
        Interval(1, s, True, True))

    assert reduce_rational_inequalities([[Lt(x**2, -1.)]], x) is S.false


def test_reduce_poly_inequalities_complex_relational():
    assert reduce_rational_inequalities(
        [[Eq(x**2, 0)]], x, relational=True) == Eq(x, 0)
    assert reduce_rational_inequalities(
        [[Le(x**2, 0)]], x, relational=True) == Eq(x, 0)
    assert reduce_rational_inequalities(
        [[Lt(x**2, 0)]], x, relational=True) == False
    assert reduce_rational_inequalities(
        [[Ge(x**2, 0)]], x, relational=True) == And(Lt(-oo, x), Lt(x, oo))
    assert reduce_rational_inequalities(
        [[Gt(x**2, 0)]], x, relational=True) == \
        And(Gt(x, -oo), Lt(x, oo), Ne(x, 0))
    assert reduce_rational_inequalities(
        [[Ne(x**2, 0)]], x, relational=True) == \
        And(Gt(x, -oo), Lt(x, oo), Ne(x, 0))

    for one in (S.One, S(1.0)):
        inf = one*oo
        assert reduce_rational_inequalities(
            [[Eq(x**2, one)]], x, relational=True) == \
            Or(Eq(x, -one), Eq(x, one))
        assert reduce_rational_inequalities(
            [[Le(x**2, one)]], x, relational=True) == \
            And(And(Le(-one, x), Le(x, one)))
        assert reduce_rational_inequalities(
            [[Lt(x**2, one)]], x, relational=True) == \
            And(And(Lt(-one, x), Lt(x, one)))
        assert reduce_rational_inequalities(
            [[Ge(x**2, one)]], x, relational=True) == \
            And(Or(And(Le(one, x), Lt(x, inf)), And(Le(x, -one), Lt(-inf, x))))
        assert reduce_rational_inequalities(
            [[Gt(x**2, one)]], x, relational=True) == \
            And(Or(And(Lt(-inf, x), Lt(x, -one)), And(Lt(one, x), Lt(x, inf))))
        assert reduce_rational_inequalities(
            [[Ne(x**2, one)]], x, relational=True) == \
            Or(And(Lt(-inf, x), Lt(x, -one)),
               And(Lt(-one, x), Lt(x, one)),
               And(Lt(one, x), Lt(x, inf)))


def test_reduce_rational_inequalities_real_relational():
    assert reduce_rational_inequalities([], x) == False
    assert reduce_rational_inequalities(
        [[(x**2 + 3*x + 2)/(x**2 - 16) >= 0]], x, relational=False) == \
        Union(Interval.open(-oo, -4), Interval(-2, -1), Interval.open(4, oo))

    assert reduce_rational_inequalities(
        [[((-2*x - 10)*(3 - x))/((x**2 + 5)*(x - 2)**2) < 0]], x,
        relational=False) == \
        Union(Interval.open(-5, 2), Interval.open(2, 3))

    assert reduce_rational_inequalities([[(x + 1)/(x - 5) <= 0]], x,
        relational=False) == \
        Interval.Ropen(-1, 5)

    assert reduce_rational_inequalities([[(x**2 + 4*x + 3)/(x - 1) > 0]], x,
        relational=False) == \
        Union(Interval.open(-3, -1), Interval.open(1, oo))

    assert reduce_rational_inequalities([[(x**2 - 16)/(x - 1)**2 < 0]], x,
        relational=False) == \
        Union(Interval.open(-4, 1), Interval.open(1, 4))

    assert reduce_rational_inequalities([[(3*x + 1)/(x + 4) >= 1]], x,
        relational=False) == \
        Union(Interval.open(-oo, -4), Interval.Ropen(Rational(3, 2), oo))

    assert reduce_rational_inequalities([[(x - 8)/x <= 3 - x]], x,
        relational=False) == \
        Union(Interval.Lopen(-oo, -2), Interval.Lopen(0, 4))

    # issue sympy/sympy#10237
    assert reduce_rational_inequalities(
        [[x < oo, x >= 0, -oo < x]], x, relational=False) == Interval(0, oo)


def test_reduce_abs_inequalities():
    e = abs(x - 5) < 3
    ans = And(Lt(2, x), Lt(x, 8))
    assert reduce_inequalities(e) == ans
    assert reduce_inequalities(e, x) == ans
    assert reduce_inequalities(abs(x - 5)) == Eq(x, 5)
    assert reduce_inequalities(
        abs(2*x + 3) >= 8) == Or(And(Le(Rational(5, 2), x), Lt(x, oo)),
        And(Le(x, Rational(-11, 2)), Lt(-oo, x)))
    assert reduce_inequalities(abs(x - 4) + abs(
        3*x - 5) < 7) == And(Lt(S.Half, x), Lt(x, 4))
    assert reduce_inequalities(abs(x - 4) + abs(3*abs(x) - 5) < 7) == \
        Or(And(S(-2) < x, x < -1), And(S.Half < x, x < 4))

    nr = Symbol('nr', extended_real=False)
    raises(TypeError, lambda: reduce_inequalities(abs(nr - 5) < 3))
    assert reduce_inequalities(x < 3, symbols=[x, nr]) == And(-oo < x, x < 3)


def test_reduce_inequalities_general():
    assert reduce_inequalities(Ge(sqrt(2)*x, 1)) == And(sqrt(2)/2 <= x, x < oo)
    assert reduce_inequalities(x + 1 > 0) == And(S.NegativeOne < x, x < oo)


def test_reduce_inequalities_boolean():
    assert reduce_inequalities(
        [Eq(x**2, 0), True]) == Eq(x, 0)
    assert reduce_inequalities([Eq(x**2, 0), False]) == False
    assert reduce_inequalities(x**2 >= 0) is S.true  # issue 10196


def test_reduce_inequalities_multivariate():
    assert reduce_inequalities([Ge(x**2, 1), Ge(y**2, 1)]) == And(
        Or(And(Le(S.One, x), Lt(x, oo)), And(Le(x, -1), Lt(-oo, x))),
        Or(And(Le(S.One, y), Lt(y, oo)), And(Le(y, -1), Lt(-oo, y))))


def test_reduce_inequalities_errors():
    raises(NotImplementedError, lambda: reduce_inequalities(Ge(sin(x) + x, 1)))
    raises(NotImplementedError, lambda: reduce_inequalities(Ge(x**2*y + y, 1)))


def test__solve_inequalities():
    assert reduce_inequalities(x + y < 1, symbols=[x]) == (x < 1 - y)
    assert reduce_inequalities(x + y >= 1, symbols=[x]) == (x < oo) & (x >= -y + 1)
    assert reduce_inequalities(Eq(0, x - y), symbols=[x]) == Eq(x, y)
    assert reduce_inequalities(Ne(0, x - y), symbols=[x]) == Ne(x, y)


def test_issue_6343():
    eq = -3*x**2/2 - x*Rational(45, 4) + Rational(33, 2) > 0
    assert reduce_inequalities(eq) == \
        And(x < Rational(-15, 4) + sqrt(401)/4, -sqrt(401)/4 - Rational(15, 4) < x)


def test_issue_8235():
    assert reduce_inequalities(x**2 - 1 < 0) == \
        And(S.NegativeOne < x, x < 1)
    assert reduce_inequalities(x**2 - 1 <= 0) == \
        And(S.NegativeOne <= x, x <= 1)
    assert reduce_inequalities(x**2 - 1 > 0) == \
        Or(And(-oo < x, x < -1), And(x < oo, S.One < x))
    assert reduce_inequalities(x**2 - 1 >= 0) == \
        Or(And(-oo < x, x <= -1), And(S.One <= x, x < oo))

    eq = x**8 + x - 9  # we want CRootOf solns here
    sol = solve(eq >= 0)
    tru = Or(And(rootof(eq, 1) <= x, x < oo), And(-oo < x, x <= rootof(eq, 0)))
    assert sol == tru

    # recast vanilla as real
    assert solve(sqrt((-x + 1)**2) < 1) == And(S.Zero < x, x < 2)


def test_issue_5526():
    assert reduce_inequalities(0 <=
        x + Integral(y**2, (y, 1, 3)) - 1, [x]) == \
        (x >= -Integral(y**2, (y, 1, 3)) + 1)
    f = Function('f')
    e = Sum(f(x), (x, 1, 3))
    assert reduce_inequalities(0 <= x + e + y**2, [x]) == \
        (x >= -y**2 - Sum(f(x), (x, 1, 3)))


def test_solve_univariate_inequality():
    assert isolve(x**2 >= 4, x, relational=False) == Union(Interval(-oo, -2),
        Interval(2, oo))
    assert isolve(x**2 >= 4, x) == Or(And(Le(2, x), Lt(x, oo)), And(Le(x, -2),
        Lt(-oo, x)))
    assert isolve((x - 1)*(x - 2)*(x - 3) >= 0, x, relational=False) == \
        Union(Interval(1, 2), Interval(3, oo))
    assert isolve((x - 1)*(x - 2)*(x - 3) >= 0, x) == \
        Or(And(Le(1, x), Le(x, 2)), And(Le(3, x), Lt(x, oo)))
    assert isolve((x - 1)*(x - 2)*(x - 4) < 0, x, domain = FiniteSet(0, 3)) == \
        Or(Eq(x, 0), Eq(x, 3))
    # issue 2785:
    assert isolve(x**3 - 2*x - 1 > 0, x, relational=False) == \
        Union(Interval(-1, -sqrt(5)/2 + S.Half, True, True),
              Interval(S.Half + sqrt(5)/2, oo, True, True))
    # issue 2794:
    assert isolve(x**3 - x**2 + x - 1 > 0, x, relational=False) == \
        Interval(1, oo, True)
    #issue 13105
    assert isolve((x + I)*(x + 2*I) < 0, x) == Eq(x, 0)
    assert isolve(((x - 1)*(x - 2) + I)*((x - 1)*(x - 2) + 2*I) < 0, x) == Or(Eq(x, 1), Eq(x, 2))
    assert isolve((((x - 1)*(x - 2) + I)*((x - 1)*(x - 2) + 2*I))/(x - 2) > 0, x) == Eq(x, 1)
    raises (ValueError, lambda: isolve((x**2 - 3*x*I + 2)/x < 0, x))

    # numerical testing in valid() is needed
    assert isolve(x**7 - x - 2 > 0, x) == \
        And(rootof(x**7 - x - 2, 0) < x, x < oo)

    # handle numerator and denominator; although these would be handled as
    # rational inequalities, these test confirm that the right thing is done
    # when the domain is EX (e.g. when 2 is replaced with sqrt(2))
    assert isolve(1/(x - 2) > 0, x) == And(S(2) < x, x < oo)
    den = ((x - 1)*(x - 2)).expand()
    assert isolve((x - 1)/den <= 0, x) == \
        (x > -oo) & (x < 2) & Ne(x, 1)

    n = Dummy('n')
    raises(NotImplementedError, lambda: isolve(Abs(x) <= n, x, relational=False))
    c1 = Dummy("c1", positive=True)
    raises(NotImplementedError, lambda: isolve(n/c1 < 0, c1))
    n = Dummy('n', negative=True)
    assert isolve(n/c1 > -2, c1) == (-n/2 < c1)
    assert isolve(n/c1 < 0, c1) == True
    assert isolve(n/c1 > 0, c1) == False

    zero = cos(1)**2 + sin(1)**2 - 1
    raises(NotImplementedError, lambda: isolve(x**2 < zero, x))
    raises(NotImplementedError, lambda: isolve(
        x**2 < zero*I, x))
    raises(NotImplementedError, lambda: isolve(1/(x - y) < 2, x))
    raises(NotImplementedError, lambda: isolve(1/(x - y) < 0, x))
    raises(TypeError, lambda: isolve(x - I < 0, x))

    zero = x**2 + x - x*(x + 1)
    assert isolve(zero < 0, x, relational=False) is S.EmptySet
    assert isolve(zero <= 0, x, relational=False) is S.Reals

    # make sure iter_solutions gets a default value
    raises(NotImplementedError, lambda: isolve(
        Eq(cos(x)**2 + sin(x)**2, 1), x))


def test_trig_inequalities():
    # all the inequalities are solved in a periodic interval.
    assert isolve(sin(x) < S.Half, x, relational=False) == \
        Union(Interval(0, pi/6, False, True), Interval.open(pi*Rational(5, 6), 2*pi))
    assert isolve(sin(x) > S.Half, x, relational=False) == \
        Interval(pi/6, pi*Rational(5, 6), True, True)
    assert isolve(cos(x) < S.Zero, x, relational=False) == \
        Interval(pi/2, pi*Rational(3, 2), True, True)
    assert isolve(cos(x) >= S.Zero, x, relational=False) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))

    assert isolve(tan(x) < S.One, x, relational=False) == \
        Union(Interval.Ropen(0, pi/4), Interval.open(pi/2, pi))

    assert isolve(sin(x) <= S.Zero, x, relational=False) == \
        Union(FiniteSet(S.Zero), Interval.Ropen(pi, 2*pi))

    assert isolve(sin(x) <= S.One, x, relational=False) == S.Reals
    assert isolve(cos(x) < S(-2), x, relational=False) == S.EmptySet
    assert isolve(sin(x) >= S.NegativeOne, x, relational=False) == S.Reals
    assert isolve(cos(x) > S.One, x, relational=False) == S.EmptySet


def test_issue_9954():
    assert isolve(x**2 >= 0, x, relational=False) == S.Reals
    assert isolve(x**2 >= 0, x, relational=True) == S.Reals.as_relational(x)
    assert isolve(x**2 < 0, x, relational=False) == S.EmptySet
    assert isolve(x**2 < 0, x, relational=True) == S.EmptySet.as_relational(x)


@XFAIL
def test_slow_general_univariate():
    r = rootof(x**5 - x**2 + 1, 0)
    assert solve(sqrt(x) + 1/root(x, 3) > 1) == \
        Or(And(0 < x, x < r**6), And(r**6 < x, x < oo))


def test_issue_8545():
    eq = 1 - x - abs(1 - x)
    ans = And(Lt(1, x), Lt(x, oo))
    assert reduce_abs_inequality(eq, '<', x) == ans
    eq = 1 - x - sqrt((1 - x)**2)
    assert reduce_inequalities(eq < 0) == ans


def test_issue_8974():
    assert isolve(-oo < x, x) == And(-oo < x, x < oo)
    assert isolve(oo > x, x) == And(-oo < x, x < oo)


def test_issue_10198():
    assert reduce_inequalities(
        -1 + 1/abs(1/x - 1) < 0) == (x > -oo) & (x < S(1)/2) & Ne(x, 0)

    assert reduce_inequalities(abs(1/sqrt(x)) - 1, x) == Eq(x, 1)
    assert reduce_abs_inequality(-3 + 1/abs(1 - 1/x), '<', x) == \
        Or(And(-oo < x, x < 0),
        And(S.Zero < x, x < Rational(3, 4)), And(Rational(3, 2) < x, x < oo))
    raises(ValueError,lambda: reduce_abs_inequality(-3 + 1/abs(
        1 - 1/sqrt(x)), '<', x))


def test_issue_10047():
    # issue 10047: this must remain an inequality, not True, since if x
    # is not real the inequality is invalid
    # assert solve(sin(x) < 2) == (x <= oo)

    # with PR 16956, (x <= oo) autoevaluates when x is extended_real
    # which is assumed in the current implementation of inequality solvers
    assert solve(sin(x) < 2) == True
    assert solveset(sin(x) < 2, domain=S.Reals) == S.Reals


def test_issue_10268():
    assert solve(log(x) < 1000) == And(S.Zero < x, x < exp(1000))


@XFAIL
def test_isolve_Sets():
    n = Dummy('n')
    assert isolve(Abs(x) <= n, x, relational=False) == \
        Piecewise((S.EmptySet, n < 0), (Interval(-n, n), True))


def test_integer_domain_relational_isolve():

    dom = FiniteSet(0, 3)
    x = Symbol('x',zero=False)
    assert isolve((x - 1)*(x - 2)*(x - 4) < 0, x, domain=dom) == Eq(x, 3)

    x = Symbol('x')
    assert isolve(x + 2 < 0, x, domain=S.Integers) == \
           (x <= -3) & (x > -oo) & Eq(Mod(x, 1), 0)
    assert isolve(2 * x + 3 > 0, x, domain=S.Integers) == \
           (x >= -1) & (x < oo)  & Eq(Mod(x, 1), 0)
    assert isolve((x ** 2 + 3 * x - 2) < 0, x, domain=S.Integers) == \
           (x >= -3) & (x <= 0)  & Eq(Mod(x, 1), 0)
    assert isolve((x ** 2 + 3 * x - 2) > 0, x, domain=S.Integers) == \
           ((x >= 1) & (x < oo)  & Eq(Mod(x, 1), 0)) | (
               (x <= -4) & (x > -oo)  & Eq(Mod(x, 1), 0))


def test_issue_10671_12466():
    assert solveset(sin(y), y, Interval(0, pi)) == FiniteSet(0, pi)
    i = Interval(1, 10)
    assert solveset((1/x).diff(x) < 0, x, i) == i
    assert solveset((log(x - 6)/x) <= 0, x, S.Reals) == \
        Interval.Lopen(6, 7)


def test__solve_inequality():
    for op in (Gt, Lt, Le, Ge, Eq, Ne):
        assert _solve_inequality(op(x, 1), x).lhs == x
        assert _solve_inequality(op(S.One, x), x).lhs == x
    # don't get tricked by symbol on right: solve it
    assert _solve_inequality(Eq(2*x - 1, x), x) == Eq(x, 1)
    ie = Eq(S.One, y)
    assert _solve_inequality(ie, x) == ie
    for fx in (x**2, exp(x), sin(x) + cos(x), x*(1 + x)):
        for c in (0, 1):
            e = 2*fx - c > 0
            assert _solve_inequality(e, x, linear=True) == (
                fx > c/S(2))
    assert _solve_inequality(2*x**2 + 2*x - 1 < 0, x, linear=True) == (
        x*(x + 1) < S.Half)
    assert _solve_inequality(Eq(x*y, 1), x) == Eq(x*y, 1)
    nz = Symbol('nz', nonzero=True)
    assert _solve_inequality(Eq(x*nz, 1), x) == Eq(x, 1/nz)
    assert _solve_inequality(x*nz < 1, x) == (x*nz < 1)
    a = Symbol('a', positive=True)
    assert _solve_inequality(a/x > 1, x) == (S.Zero < x) & (x < a)
    assert _solve_inequality(a/x > 1, x, linear=True) == (1/x > 1/a)
    # make sure to include conditions under which solution is valid
    e = Eq(1 - x, x*(1/x - 1))
    assert _solve_inequality(e, x) == Ne(x, 0)
    assert _solve_inequality(x < x*(1/x - 1), x) == (x < S.Half) & Ne(x, 0)


def test__pt():
    from sympy.solvers.inequalities import _pt
    assert _pt(-oo, oo) == 0
    assert _pt(S.One, S(3)) == 2
    assert _pt(S.One, oo) == _pt(oo, S.One) == 2
    assert _pt(S.One, -oo) == _pt(-oo, S.One) == S.Half
    assert _pt(S.NegativeOne, oo) == _pt(oo, S.NegativeOne) == Rational(-1, 2)
    assert _pt(S.NegativeOne, -oo) == _pt(-oo, S.NegativeOne) == -2
    assert _pt(x, oo) == _pt(oo, x) == x + 1
    assert _pt(x, -oo) == _pt(-oo, x) == x - 1
    raises(ValueError, lambda: _pt(Dummy('i', infinite=True), S.One))


def test_issue_25697():
    assert _solve_inequality(log(x, 3) <= 2, x) == (x <= 9) & (S.Zero < x)


def test_issue_25738():
    assert reduce_inequalities(3 < abs(x)
        ) == reduce_inequalities(pi < abs(x)).subs(pi, 3)


def test_issue_25983():
    assert(reduce_inequalities(pi/Abs(x) <= 1) == ((pi <= x) & (x < oo)) | ((-oo < x) & (x <= -pi)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_datetime.py ===
"""
Also test support for datetime64[ns] in Series / DataFrame
"""
from datetime import (
    datetime,
    timedelta,
)
import re

from dateutil.tz import (
    gettz,
    tzutc,
)
import numpy as np
import pytest
import pytz

from pandas._libs import index as libindex

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


def test_fancy_getitem():
    dti = date_range(
        freq="WOM-1FRI", start=datetime(2005, 1, 1), end=datetime(2010, 1, 1)
    )

    s = Series(np.arange(len(dti)), index=dti)

    msg = "Series.__getitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert s[48] == 48
    assert s["1/2/2009"] == 48
    assert s["2009-1-2"] == 48
    assert s[datetime(2009, 1, 2)] == 48
    assert s[Timestamp(datetime(2009, 1, 2))] == 48
    with pytest.raises(KeyError, match=r"^'2009-1-3'$"):
        s["2009-1-3"]
    tm.assert_series_equal(
        s["3/6/2009":"2009-06-05"], s[datetime(2009, 3, 6) : datetime(2009, 6, 5)]
    )


def test_fancy_setitem():
    dti = date_range(
        freq="WOM-1FRI", start=datetime(2005, 1, 1), end=datetime(2010, 1, 1)
    )

    s = Series(np.arange(len(dti)), index=dti)

    msg = "Series.__setitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        s[48] = -1
    assert s.iloc[48] == -1
    s["1/2/2009"] = -2
    assert s.iloc[48] == -2
    s["1/2/2009":"2009-06-05"] = -3
    assert (s[48:54] == -3).all()


@pytest.mark.parametrize("tz_source", ["pytz", "dateutil"])
def test_getitem_setitem_datetime_tz(tz_source):
    if tz_source == "pytz":
        tzget = pytz.timezone
    else:
        # handle special case for utc in dateutil
        tzget = lambda x: tzutc() if x == "UTC" else gettz(x)

    N = 50
    # testing with timezone, GH #2785
    rng = date_range("1/1/1990", periods=N, freq="h", tz=tzget("US/Eastern"))
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)

    # also test Timestamp tz handling, GH #2789
    result = ts.copy()
    result["1990-01-01 09:00:00+00:00"] = 0
    result["1990-01-01 09:00:00+00:00"] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    result = ts.copy()
    result["1990-01-01 03:00:00-06:00"] = 0
    result["1990-01-01 03:00:00-06:00"] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    # repeat with datetimes
    result = ts.copy()
    result[datetime(1990, 1, 1, 9, tzinfo=tzget("UTC"))] = 0
    result[datetime(1990, 1, 1, 9, tzinfo=tzget("UTC"))] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    result = ts.copy()
    dt = Timestamp(1990, 1, 1, 3).tz_localize(tzget("US/Central"))
    dt = dt.to_pydatetime()
    result[dt] = 0
    result[dt] = ts.iloc[4]
    tm.assert_series_equal(result, ts)


def test_getitem_setitem_datetimeindex():
    N = 50
    # testing with timezone, GH #2785
    rng = date_range("1/1/1990", periods=N, freq="h", tz="US/Eastern")
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)

    result = ts["1990-01-01 04:00:00"]
    expected = ts.iloc[4]
    assert result == expected

    result = ts.copy()
    result["1990-01-01 04:00:00"] = 0
    result["1990-01-01 04:00:00"] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    result = ts["1990-01-01 04:00:00":"1990-01-01 07:00:00"]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    result = ts.copy()
    result["1990-01-01 04:00:00":"1990-01-01 07:00:00"] = 0
    result["1990-01-01 04:00:00":"1990-01-01 07:00:00"] = ts[4:8]
    tm.assert_series_equal(result, ts)

    lb = "1990-01-01 04:00:00"
    rb = "1990-01-01 07:00:00"
    # GH#18435 strings get a pass from tzawareness compat
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    lb = "1990-01-01 04:00:00-0500"
    rb = "1990-01-01 07:00:00-0500"
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    # But we do not give datetimes a pass on tzawareness compat
    msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
    naive = datetime(1990, 1, 1, 4)
    for key in [naive, Timestamp(naive), np.datetime64(naive, "ns")]:
        with pytest.raises(KeyError, match=re.escape(repr(key))):
            # GH#36148 as of 2.0 we require tzawareness-compat
            ts[key]

    result = ts.copy()
    # GH#36148 as of 2.0 we do not ignore tzawareness mismatch in indexing,
    #  so setting it as a new key casts to object rather than matching
    #  rng[4]
    result[naive] = ts.iloc[4]
    assert result.index.dtype == object
    tm.assert_index_equal(result.index[:-1], rng.astype(object))
    assert result.index[-1] == naive

    msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
    with pytest.raises(TypeError, match=msg):
        # GH#36148 require tzawareness compat as of 2.0
        ts[naive : datetime(1990, 1, 1, 7)]

    result = ts.copy()
    with pytest.raises(TypeError, match=msg):
        # GH#36148 require tzawareness compat as of 2.0
        result[naive : datetime(1990, 1, 1, 7)] = 0
    with pytest.raises(TypeError, match=msg):
        # GH#36148 require tzawareness compat as of 2.0
        result[naive : datetime(1990, 1, 1, 7)] = 99
    # the __setitems__ here failed, so result should still match ts
    tm.assert_series_equal(result, ts)

    lb = naive
    rb = datetime(1990, 1, 1, 7)
    msg = r"Invalid comparison between dtype=datetime64\[ns, US/Eastern\] and datetime"
    with pytest.raises(TypeError, match=msg):
        # tznaive vs tzaware comparison is invalid
        # see GH#18376, GH#18162
        ts[(ts.index >= lb) & (ts.index <= rb)]

    lb = Timestamp(naive).tz_localize(rng.tzinfo)
    rb = Timestamp(datetime(1990, 1, 1, 7)).tz_localize(rng.tzinfo)
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    result = ts[ts.index[4]]
    expected = ts.iloc[4]
    assert result == expected

    result = ts[ts.index[4:8]]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    result = ts.copy()
    result[ts.index[4:8]] = 0
    result.iloc[4:8] = ts.iloc[4:8]
    tm.assert_series_equal(result, ts)

    # also test partial date slicing
    result = ts["1990-01-02"]
    expected = ts[24:48]
    tm.assert_series_equal(result, expected)

    result = ts.copy()
    result["1990-01-02"] = 0
    result["1990-01-02"] = ts[24:48]
    tm.assert_series_equal(result, ts)


def test_getitem_setitem_periodindex():
    N = 50
    rng = period_range("1/1/1990", periods=N, freq="h")
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)

    result = ts["1990-01-01 04"]
    expected = ts.iloc[4]
    assert result == expected

    result = ts.copy()
    result["1990-01-01 04"] = 0
    result["1990-01-01 04"] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    result = ts["1990-01-01 04":"1990-01-01 07"]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    result = ts.copy()
    result["1990-01-01 04":"1990-01-01 07"] = 0
    result["1990-01-01 04":"1990-01-01 07"] = ts[4:8]
    tm.assert_series_equal(result, ts)

    lb = "1990-01-01 04"
    rb = "1990-01-01 07"
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    # GH 2782
    result = ts[ts.index[4]]
    expected = ts.iloc[4]
    assert result == expected

    result = ts[ts.index[4:8]]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    result = ts.copy()
    result[ts.index[4:8]] = 0
    result.iloc[4:8] = ts.iloc[4:8]
    tm.assert_series_equal(result, ts)


def test_datetime_indexing():
    index = date_range("1/1/2000", "1/7/2000")
    index = index.repeat(3)

    s = Series(len(index), index=index)
    stamp = Timestamp("1/8/2000")

    with pytest.raises(KeyError, match=re.escape(repr(stamp))):
        s[stamp]
    s[stamp] = 0
    assert s[stamp] == 0

    # not monotonic
    s = Series(len(index), index=index)
    s = s[::-1]

    with pytest.raises(KeyError, match=re.escape(repr(stamp))):
        s[stamp]
    s[stamp] = 0
    assert s[stamp] == 0


# test duplicates in time series


def test_indexing_with_duplicate_datetimeindex(
    rand_series_with_duplicate_datetimeindex,
):
    ts = rand_series_with_duplicate_datetimeindex

    uniques = ts.index.unique()
    for date in uniques:
        result = ts[date]

        mask = ts.index == date
        total = (ts.index == date).sum()
        expected = ts[mask]
        if total > 1:
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_almost_equal(result, expected.iloc[0])

        cp = ts.copy()
        cp[date] = 0
        expected = Series(np.where(mask, 0, ts), index=ts.index)
        tm.assert_series_equal(cp, expected)

    key = datetime(2000, 1, 6)
    with pytest.raises(KeyError, match=re.escape(repr(key))):
        ts[key]

    # new index
    ts[datetime(2000, 1, 6)] = 0
    assert ts[datetime(2000, 1, 6)] == 0


def test_loc_getitem_over_size_cutoff(monkeypatch):
    # #1821

    monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 1000)

    # create large list of non periodic datetime
    dates = []
    sec = timedelta(seconds=1)
    half_sec = timedelta(microseconds=500000)
    d = datetime(2011, 12, 5, 20, 30)
    n = 1100
    for i in range(n):
        dates.append(d)
        dates.append(d + sec)
        dates.append(d + sec + half_sec)
        dates.append(d + sec + sec + half_sec)
        d += 3 * sec

    # duplicate some values in the list
    duplicate_positions = np.random.default_rng(2).integers(0, len(dates) - 1, 20)
    for p in duplicate_positions:
        dates[p + 1] = dates[p]

    df = DataFrame(
        np.random.default_rng(2).standard_normal((len(dates), 4)),
        index=dates,
        columns=list("ABCD"),
    )

    pos = n * 3
    timestamp = df.index[pos]
    assert timestamp in df.index

    # it works!
    df.loc[timestamp]
    assert len(df.loc[[timestamp]]) > 0


def test_indexing_over_size_cutoff_period_index(monkeypatch):
    # GH 27136

    monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 1000)

    n = 1100
    idx = period_range("1/1/2000", freq="min", periods=n)
    assert idx._engine.over_size_threshold

    s = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)

    pos = n - 1
    timestamp = idx[pos]
    assert timestamp in s.index

    # it works!
    s[timestamp]
    assert len(s.loc[[timestamp]]) > 0


def test_indexing_unordered():
    # GH 2437
    rng = date_range(start="2011-01-01", end="2011-01-15")
    ts = Series(np.random.default_rng(2).random(len(rng)), index=rng)
    ts2 = pd.concat([ts[0:4], ts[-4:], ts[4:-4]])

    for t in ts.index:
        expected = ts[t]
        result = ts2[t]
        assert expected == result

    # GH 3448 (ranges)
    def compare(slobj):
        result = ts2[slobj].copy()
        result = result.sort_index()
        expected = ts[slobj]
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    for key in [
        slice("2011-01-01", "2011-01-15"),
        slice("2010-12-30", "2011-01-15"),
        slice("2011-01-01", "2011-01-16"),
        # partial ranges
        slice("2011-01-01", "2011-01-6"),
        slice("2011-01-06", "2011-01-8"),
        slice("2011-01-06", "2011-01-12"),
    ]:
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            compare(key)

    # single values
    result = ts2["2011"].sort_index()
    expected = ts["2011"]
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)


def test_indexing_unordered2():
    # diff freq
    rng = date_range(datetime(2005, 1, 1), periods=20, freq="ME")
    ts = Series(np.arange(len(rng)), index=rng)
    ts = ts.take(np.random.default_rng(2).permutation(20))

    result = ts["2005"]
    for t in result.index:
        assert t.year == 2005


def test_indexing():
    idx = date_range("2001-1-1", periods=20, freq="ME")
    ts = Series(np.random.default_rng(2).random(len(idx)), index=idx)

    # getting

    # GH 3070, make sure semantics work on Series/Frame
    result = ts["2001"]
    tm.assert_series_equal(result, ts.iloc[:12])

    df = DataFrame({"A": ts.copy()})

    # GH#36179 pre-2.0 df["2001"] operated as slicing on rows. in 2.0 it behaves
    #  like any other key, so raises
    with pytest.raises(KeyError, match="2001"):
        df["2001"]

    # setting
    ts = Series(np.random.default_rng(2).random(len(idx)), index=idx)
    expected = ts.copy()
    expected.iloc[:12] = 1
    ts["2001"] = 1
    tm.assert_series_equal(ts, expected)

    expected = df.copy()
    expected.iloc[:12, 0] = 1
    df.loc["2001", "A"] = 1
    tm.assert_frame_equal(df, expected)


def test_getitem_str_month_with_datetimeindex():
    # GH3546 (not including times on the last day)
    idx = date_range(start="2013-05-31 00:00", end="2013-05-31 23:00", freq="h")
    ts = Series(range(len(idx)), index=idx)
    expected = ts["2013-05"]
    tm.assert_series_equal(expected, ts)

    idx = date_range(start="2013-05-31 00:00", end="2013-05-31 23:59", freq="s")
    ts = Series(range(len(idx)), index=idx)
    expected = ts["2013-05"]
    tm.assert_series_equal(expected, ts)


def test_getitem_str_year_with_datetimeindex():
    idx = [
        Timestamp("2013-05-31 00:00"),
        Timestamp(datetime(2013, 5, 31, 23, 59, 59, 999999)),
    ]
    ts = Series(range(len(idx)), index=idx)
    expected = ts["2013"]
    tm.assert_series_equal(expected, ts)


def test_getitem_str_second_with_datetimeindex():
    # GH14826, indexing with a seconds resolution string / datetime object
    df = DataFrame(
        np.random.default_rng(2).random((5, 5)),
        columns=["open", "high", "low", "close", "volume"],
        index=date_range("2012-01-02 18:01:00", periods=5, tz="US/Central", freq="s"),
    )

    # this is a single date, so will raise
    with pytest.raises(KeyError, match=r"^'2012-01-02 18:01:02'$"):
        df["2012-01-02 18:01:02"]

    msg = r"Timestamp\('2012-01-02 18:01:02-0600', tz='US/Central'\)"
    with pytest.raises(KeyError, match=msg):
        df[df.index[2]]


def test_compare_datetime_with_all_none():
    # GH#54870
    ser = Series(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    ser2 = Series([None, None])
    result = ser > ser2
    expected = Series([False, False])
    tm.assert_series_equal(result, expected)