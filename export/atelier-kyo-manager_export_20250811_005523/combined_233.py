
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_cumulative.py ===
"""
Tests for DataFrame cumulative operations

See also
--------
tests.series.test_cumulative
"""

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameCumulativeOps:
    # ---------------------------------------------------------------------
    # Cumulative Operations - cumsum, cummax, ...

    def test_cumulative_ops_smoke(self):
        # it works
        df = DataFrame({"A": np.arange(20)}, index=np.arange(20))
        df.cummax()
        df.cummin()
        df.cumsum()

        dm = DataFrame(np.arange(20).reshape(4, 5), index=range(4), columns=range(5))
        # TODO(wesm): do something with this?
        dm.cumsum()

    def test_cumprod_smoke(self, datetime_frame):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan

        # ints
        df = datetime_frame.fillna(0).astype(int)
        df.cumprod(0)
        df.cumprod(1)

        # ints32
        df = datetime_frame.fillna(0).astype(np.int32)
        df.cumprod(0)
        df.cumprod(1)

    @pytest.mark.parametrize("method", ["cumsum", "cumprod", "cummin", "cummax"])
    def test_cumulative_ops_match_series_apply(self, datetime_frame, method):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan

        # axis = 0
        result = getattr(datetime_frame, method)()
        expected = datetime_frame.apply(getattr(Series, method))
        tm.assert_frame_equal(result, expected)

        # axis = 1
        result = getattr(datetime_frame, method)(axis=1)
        expected = datetime_frame.apply(getattr(Series, method), axis=1)
        tm.assert_frame_equal(result, expected)

        # fix issue TODO: GH ref?
        assert np.shape(result) == np.shape(datetime_frame)

    def test_cumsum_preserve_dtypes(self):
        # GH#19296 dont incorrectly upcast to object
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3.0], "C": [True, False, False]})

        result = df.cumsum()

        expected = DataFrame(
            {
                "A": Series([1, 3, 6], dtype=np.int64),
                "B": Series([1, 3, 6], dtype=np.float64),
                "C": df["C"].cumsum(),
            }
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_verbose.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import pytest

import pandas._testing as tm

depr_msg = "The 'verbose' keyword in pd.read_csv is deprecated"


def test_verbose_read(all_parsers, capsys):
    parser = all_parsers
    data = """a,b,c,d
one,1,2,3
one,1,2,3
,1,2,3
one,1,2,3
,1,2,3
,1,2,3
one,1,2,3
two,1,2,3"""

    if parser.engine == "pyarrow":
        msg = "The 'verbose' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                FutureWarning, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(StringIO(data), verbose=True)
        return

    # Engines are verbose in different ways.
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        parser.read_csv(StringIO(data), verbose=True)
    captured = capsys.readouterr()

    if parser.engine == "c":
        assert "Tokenization took:" in captured.out
        assert "Parser memory cleanup took:" in captured.out
    else:  # Python engine
        assert captured.out == "Filled 3 NA values in column a\n"


def test_verbose_read2(all_parsers, capsys):
    parser = all_parsers
    data = """a,b,c,d
one,1,2,3
two,1,2,3
three,1,2,3
four,1,2,3
five,1,2,3
,1,2,3
seven,1,2,3
eight,1,2,3"""

    if parser.engine == "pyarrow":
        msg = "The 'verbose' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                FutureWarning, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(StringIO(data), verbose=True, index_col=0)
        return

    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        parser.read_csv(StringIO(data), verbose=True, index_col=0)
    captured = capsys.readouterr()

    # Engines are verbose in different ways.
    if parser.engine == "c":
        assert "Tokenization took:" in captured.out
        assert "Parser memory cleanup took:" in captured.out
    else:  # Python engine
        assert captured.out == "Filled 1 NA values in column a\n"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_to_pydatetime.py ===
from datetime import (
    datetime,
    timedelta,
)

import pytz

from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
import pandas.util._test_decorators as td

from pandas import Timestamp
import pandas._testing as tm


class TestTimestampToPyDatetime:
    def test_to_pydatetime_fold(self):
        # GH#45087
        tzstr = "dateutil/usr/share/zoneinfo/America/Chicago"
        ts = Timestamp(year=2013, month=11, day=3, hour=1, minute=0, fold=1, tz=tzstr)
        dt = ts.to_pydatetime()
        assert dt.fold == 1

    def test_to_pydatetime_nonzero_nano(self):
        ts = Timestamp("2011-01-01 9:00:00.123456789")

        # Warn the user of data loss (nanoseconds).
        with tm.assert_produces_warning(UserWarning):
            expected = datetime(2011, 1, 1, 9, 0, 0, 123456)
            result = ts.to_pydatetime()
            assert result == expected

    def test_timestamp_to_datetime(self):
        stamp = Timestamp("20090415", tz="US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_pydatetime_dateutil(self):
        stamp = Timestamp("20090415", tz="dateutil/US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_pydatetime_explicit_pytz(self):
        stamp = Timestamp("20090415", tz=pytz.timezone("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    @td.skip_if_windows
    def test_timestamp_to_pydatetime_explicit_dateutil(self):
        stamp = Timestamp("20090415", tz=gettz("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_to_pydatetime_bijective(self):
        # Ensure that converting to datetime and back only loses precision
        # by going from nanoseconds to microseconds.
        exp_warning = None if Timestamp.max.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_max = Timestamp.max.to_pydatetime()

        assert (
            Timestamp(pydt_max).as_unit("ns")._value / 1000
            == Timestamp.max._value / 1000
        )

        exp_warning = None if Timestamp.min.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_min = Timestamp.min.to_pydatetime()

        # The next assertion can be enabled once GH#39221 is merged
        #  assert pydt_min < Timestamp.min  # this is bc nanos are dropped
        tdus = timedelta(microseconds=1)
        assert pydt_min + tdus > Timestamp.min

        assert (
            Timestamp(pydt_min + tdus).as_unit("ns")._value / 1000
            == Timestamp.min._value / 1000
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_round.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm


class TestSeriesRound:
    def test_round(self, datetime_series):
        datetime_series.index.name = "index_name"
        result = datetime_series.round(2)
        expected = Series(
            np.round(datetime_series.values, 2), index=datetime_series.index, name="ts"
        )
        tm.assert_series_equal(result, expected)
        assert result.name == datetime_series.name

    def test_round_numpy(self, any_float_dtype):
        # See GH#12600
        ser = Series([1.53, 1.36, 0.06], dtype=any_float_dtype)
        out = np.round(ser, decimals=0)
        expected = Series([2.0, 1.0, 0.0], dtype=any_float_dtype)
        tm.assert_series_equal(out, expected)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.round(ser, decimals=0, out=ser)

    def test_round_numpy_with_nan(self, any_float_dtype):
        # See GH#14197
        ser = Series([1.53, np.nan, 0.06], dtype=any_float_dtype)
        with tm.assert_produces_warning(None):
            result = ser.round()
        expected = Series([2.0, np.nan, 0.0], dtype=any_float_dtype)
        tm.assert_series_equal(result, expected)

    def test_round_builtin(self, any_float_dtype):
        ser = Series(
            [1.123, 2.123, 3.123],
            index=range(3),
            dtype=any_float_dtype,
        )
        result = round(ser)
        expected_rounded0 = Series(
            [1.0, 2.0, 3.0], index=range(3), dtype=any_float_dtype
        )
        tm.assert_series_equal(result, expected_rounded0)

        decimals = 2
        expected_rounded = Series(
            [1.12, 2.12, 3.12], index=range(3), dtype=any_float_dtype
        )
        result = round(ser, decimals)
        tm.assert_series_equal(result, expected_rounded)

    @pytest.mark.parametrize("method", ["round", "floor", "ceil"])
    @pytest.mark.parametrize("freq", ["s", "5s", "min", "5min", "h", "5h"])
    def test_round_nat(self, method, freq, unit):
        # GH14940, GH#56158
        ser = Series([pd.NaT], dtype=f"M8[{unit}]")
        expected = Series(pd.NaT, dtype=f"M8[{unit}]")
        round_method = getattr(ser.dt, method)
        result = round_method(freq)
        tm.assert_series_equal(result, expected)

    def test_round_ea_boolean(self):
        # GH#55936
        ser = Series([True, False], dtype="boolean")
        expected = ser.copy()
        result = ser.round(2)
        tm.assert_series_equal(result, expected)
        result.iloc[0] = False
        tm.assert_series_equal(ser, expected)

    def test_round_dtype_object(self):
        # GH#61206
        ser = Series([0.2], dtype="object")
        msg = "Expected numeric dtype, got object instead."
        with pytest.raises(TypeError, match=msg):
            ser.round()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_interval_array_equal.py ===
import pytest

from pandas import interval_range
import pandas._testing as tm


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 0, "periods": 4},
        {"start": 1, "periods": 5},
        {"start": 5, "end": 10, "closed": "left"},
    ],
)
def test_interval_array_equal(kwargs):
    arr = interval_range(**kwargs).values
    tm.assert_interval_array_equal(arr, arr)


def test_interval_array_equal_closed_mismatch():
    kwargs = {"start": 0, "periods": 5}
    arr1 = interval_range(closed="left", **kwargs).values
    arr2 = interval_range(closed="right", **kwargs).values

    msg = """\
IntervalArray are different

Attribute "closed" are different
\\[left\\]:  left
\\[right\\]: right"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


def test_interval_array_equal_periods_mismatch():
    kwargs = {"start": 0}
    arr1 = interval_range(periods=5, **kwargs).values
    arr2 = interval_range(periods=6, **kwargs).values

    msg = """\
IntervalArray.left are different

IntervalArray.left shapes are different
\\[left\\]:  \\(5,\\)
\\[right\\]: \\(6,\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


def test_interval_array_equal_end_mismatch():
    kwargs = {"start": 0, "periods": 5}
    arr1 = interval_range(end=10, **kwargs).values
    arr2 = interval_range(end=20, **kwargs).values

    msg = """\
IntervalArray.left are different

IntervalArray.left values are different \\(80.0 %\\)
\\[left\\]:  \\[0, 2, 4, 6, 8\\]
\\[right\\]: \\[0, 4, 8, 12, 16\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


def test_interval_array_equal_start_mismatch():
    kwargs = {"periods": 4}
    arr1 = interval_range(start=0, **kwargs).values
    arr2 = interval_range(start=1, **kwargs).values

    msg = """\
IntervalArray.left are different

IntervalArray.left values are different \\(100.0 %\\)
\\[left\\]:  \\[0, 1, 2, 3\\]
\\[right\\]: \\[1, 2, 3, 4\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_show_versions.py ===
import json
import os
import re

from pandas.util._print_versions import (
    _get_dependency_info,
    _get_sys_info,
)

import pandas as pd


def test_show_versions(tmpdir):
    # GH39701
    as_json = os.path.join(tmpdir, "test_output.json")

    pd.show_versions(as_json=as_json)

    with open(as_json, encoding="utf-8") as fd:
        # check if file output is valid JSON, will raise an exception if not
        result = json.load(fd)

    # Basic check that each version element is found in output
    expected = {
        "system": _get_sys_info(),
        "dependencies": _get_dependency_info(),
    }

    assert result == expected


def test_show_versions_console_json(capsys):
    # GH39701
    pd.show_versions(as_json=True)
    stdout = capsys.readouterr().out

    # check valid json is printed to the console if as_json is True
    result = json.loads(stdout)

    # Basic check that each version element is found in output
    expected = {
        "system": _get_sys_info(),
        "dependencies": _get_dependency_info(),
    }

    assert result == expected


def test_show_versions_console(capsys):
    # gh-32041
    # gh-32041
    pd.show_versions(as_json=False)
    result = capsys.readouterr().out

    # check header
    assert "INSTALLED VERSIONS" in result

    # check full commit hash
    assert re.search(r"commit\s*:\s[0-9a-f]{40}\n", result)

    # check required dependency
    # 2020-12-09 npdev has "dirty" in the tag
    # 2022-05-25 npdev released with RC wo/ "dirty".
    # Just ensure we match [0-9]+\..* since npdev version is variable
    assert re.search(r"numpy\s*:\s[0-9]+\..*\n", result)

    # check optional dependency
    assert re.search(r"pyarrow\s*:\s([0-9]+.*|None)\n", result)


def test_json_output_match(capsys, tmpdir):
    # GH39701
    pd.show_versions(as_json=True)
    result_console = capsys.readouterr().out

    out_path = os.path.join(tmpdir, "test_json.json")
    pd.show_versions(as_json=out_path)
    with open(out_path, encoding="utf-8") as out_fd:
        result_file = out_fd.read()

    assert result_console == result_file

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_commutator.py ===
from sympy.core.numbers import Integer
from sympy.core.symbol import symbols

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.operator import Operator


a, b, c = symbols('a,b,c')
n = symbols('n', integer=True)
A, B, C, D = symbols('A,B,C,D', commutative=False)


def test_commutator():
    c = Comm(A, B)
    assert c.is_commutative is False
    assert isinstance(c, Comm)
    assert c.subs(A, C) == Comm(C, B)


def test_commutator_identities():
    assert Comm(a*A, b*B) == a*b*Comm(A, B)
    assert Comm(A, A) == 0
    assert Comm(a, b) == 0
    assert Comm(A, B) == -Comm(B, A)
    assert Comm(A, B).doit() == A*B - B*A
    assert Comm(A, B*C).expand(commutator=True) == Comm(A, B)*C + B*Comm(A, C)
    assert Comm(A*B, C*D).expand(commutator=True) == \
        A*C*Comm(B, D) + A*Comm(B, C)*D + C*Comm(A, D)*B + Comm(A, C)*D*B
    assert Comm(A, B**2).expand(commutator=True) == Comm(A, B)*B + B*Comm(A, B)
    assert Comm(A**2, C**2).expand(commutator=True) == \
        Comm(A*B, C*D).expand(commutator=True).replace(B, A).replace(D, C) == \
        A*C*Comm(A, C) + A*Comm(A, C)*C + C*Comm(A, C)*A + Comm(A, C)*C*A
    assert Comm(A, C**-2).expand(commutator=True) == \
        Comm(A, (1/C)*(1/D)).expand(commutator=True).replace(D, C)
    assert Comm(A + B, C + D).expand(commutator=True) == \
        Comm(A, C) + Comm(A, D) + Comm(B, C) + Comm(B, D)
    assert Comm(A, B + C).expand(commutator=True) == Comm(A, B) + Comm(A, C)
    assert Comm(A**n, B).expand(commutator=True) == Comm(A**n, B)

    e = Comm(A, Comm(B, C)) + Comm(B, Comm(C, A)) + Comm(C, Comm(A, B))
    assert e.doit().expand() == 0


def test_commutator_dagger():
    comm = Comm(A*B, C)
    assert Dagger(comm).expand(commutator=True) == \
        - Comm(Dagger(B), Dagger(C))*Dagger(A) - \
        Dagger(B)*Comm(Dagger(A), Dagger(C))


class Foo(Operator):

    def _eval_commutator_Bar(self, bar):
        return Integer(0)


class Bar(Operator):
    pass


class Tam(Operator):

    def _eval_commutator_Foo(self, foo):
        return Integer(1)


def test_eval_commutator():
    F = Foo('F')
    B = Bar('B')
    T = Tam('T')
    assert Comm(F, B).doit() == 0
    assert Comm(B, F).doit() == 0
    assert Comm(F, T).doit() == -1
    assert Comm(T, F).doit() == 1
    assert Comm(B, T).doit() == B*T - T*B
    assert Comm(F**2, B).expand(commutator=True).doit() == 0
    assert Comm(F**2, T).expand(commutator=True).doit() == -2*F
    assert Comm(F, T**2).expand(commutator=True).doit() == -2*T
    assert Comm(T**2, F).expand(commutator=True).doit() == 2*T
    assert Comm(T**2, F**3).expand(commutator=True).doit() == 2*F*T*F + 2*F**2*T + 2*T*F**2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_pipe.py ===
import numpy as np

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


def test_pipe():
    # Test the pipe method of DataFrameGroupBy.
    # Issue #17871

    random_state = np.random.default_rng(2)

    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": random_state.standard_normal(8),
            "C": random_state.standard_normal(8),
        }
    )

    def f(dfgb):
        return dfgb.B.max() - dfgb.C.min().min()

    def square(srs):
        return srs**2

    # Note that the transformations are
    # GroupBy -> Series
    # Series -> Series
    # This then chains the GroupBy.pipe and the
    # NDFrame.pipe methods
    result = df.groupby("A").pipe(f).pipe(square)

    index = Index(["bar", "foo"], name="A")
    expected = pd.Series([3.749306591013693, 6.717707873081384], name="B", index=index)

    tm.assert_series_equal(expected, result)


def test_pipe_args():
    # Test passing args to the pipe method of DataFrameGroupBy.
    # Issue #17871

    df = DataFrame(
        {
            "group": ["A", "A", "B", "B", "C"],
            "x": [1.0, 2.0, 3.0, 2.0, 5.0],
            "y": [10.0, 100.0, 1000.0, -100.0, -1000.0],
        }
    )

    def f(dfgb, arg1):
        filtered = dfgb.filter(lambda grp: grp.y.mean() > arg1, dropna=False)
        return filtered.groupby("group")

    def g(dfgb, arg2):
        return dfgb.sum() / dfgb.sum().sum() + arg2

    def h(df, arg3):
        return df.x + df.y - arg3

    result = df.groupby("group").pipe(f, 0).pipe(g, 10).pipe(h, 100)

    # Assert the results here
    index = Index(["A", "B"], name="group")
    expected = pd.Series([-79.5160891089, -78.4839108911], index=index)

    tm.assert_series_equal(result, expected)

    # test SeriesGroupby.pipe
    ser = pd.Series([1, 1, 2, 2, 3, 3])
    result = ser.groupby(ser).pipe(lambda grp: grp.sum() * grp.count())

    expected = pd.Series([4, 8, 12], index=Index([1, 2, 3], dtype=np.int64))

    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_searchsorted.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    NaT,
    Period,
    PeriodIndex,
)
import pandas._testing as tm


class TestSearchsorted:
    @pytest.mark.parametrize("freq", ["D", "2D"])
    def test_searchsorted(self, freq):
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq=freq,
        )

        p1 = Period("2014-01-01", freq=freq)
        assert pidx.searchsorted(p1) == 0

        p2 = Period("2014-01-04", freq=freq)
        assert pidx.searchsorted(p2) == 3

        assert pidx.searchsorted(NaT) == 5

        msg = "Input has different freq=h from PeriodArray"
        with pytest.raises(IncompatibleFrequency, match=msg):
            pidx.searchsorted(Period("2014-01-01", freq="h"))

        msg = "Input has different freq=5D from PeriodArray"
        with pytest.raises(IncompatibleFrequency, match=msg):
            pidx.searchsorted(Period("2014-01-01", freq="5D"))

    def test_searchsorted_different_argument_classes(self, listlike_box):
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq="D",
        )
        result = pidx.searchsorted(listlike_box(pidx))
        expected = np.arange(len(pidx), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = pidx._data.searchsorted(listlike_box(pidx))
        tm.assert_numpy_array_equal(result, expected)

    def test_searchsorted_invalid(self):
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq="D",
        )

        other = np.array([0, 1], dtype=np.int64)

        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Period', 'NaT', or array of those. Got",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(other)

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(other.astype("timedelta64[ns]"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.timedelta64(4))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.timedelta64("NaT", "ms"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.datetime64(4, "ns"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.datetime64("NaT", "ns"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\methods\test_as_unit.py ===
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta

from pandas import Timedelta


class TestAsUnit:
    def test_as_unit(self):
        td = Timedelta(days=1)

        assert td.as_unit("ns") is td

        res = td.as_unit("us")
        assert res._value == td._value // 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_us.value

        rt = res.as_unit("ns")
        assert rt._value == td._value
        assert rt._creso == td._creso

        res = td.as_unit("ms")
        assert res._value == td._value // 1_000_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

        rt = res.as_unit("ns")
        assert rt._value == td._value
        assert rt._creso == td._creso

        res = td.as_unit("s")
        assert res._value == td._value // 1_000_000_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_s.value

        rt = res.as_unit("ns")
        assert rt._value == td._value
        assert rt._creso == td._creso

    def test_as_unit_overflows(self):
        # microsecond that would be just out of bounds for nano
        us = 9223372800000000
        td = Timedelta._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value)

        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td.as_unit("ns")

        res = td.as_unit("ms")
        assert res._value == us // 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

    def test_as_unit_rounding(self):
        td = Timedelta(microseconds=1500)
        res = td.as_unit("ms")

        expected = Timedelta(milliseconds=1)
        assert res == expected

        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res._value == 1

        with pytest.raises(ValueError, match="Cannot losslessly convert units"):
            td.as_unit("ms", round_ok=False)

    def test_as_unit_non_nano(self):
        # case where we are going neither to nor from nano
        td = Timedelta(days=1).as_unit("ms")
        assert td.days == 1
        assert td._value == 86_400_000
        assert td.components.days == 1
        assert td._d == 1
        assert td.total_seconds() == 86400

        res = td.as_unit("us")
        assert res._value == 86_400_000_000
        assert res.components.days == 1
        assert res.components.hours == 0
        assert res._d == 1
        assert res._h == 0
        assert res.total_seconds() == 86400

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_seed_sequence.py ===
import numpy as np
from numpy.random import SeedSequence
from numpy.testing import assert_array_compare, assert_array_equal


def test_reference_data():
    """ Check that SeedSequence generates data the same as the C++ reference.

    https://gist.github.com/imneme/540829265469e673d045
    """
    inputs = [
        [3735928559, 195939070, 229505742, 305419896],
        [3668361503, 4165561550, 1661411377, 3634257570],
        [164546577, 4166754639, 1765190214, 1303880213],
        [446610472, 3941463886, 522937693, 1882353782],
        [1864922766, 1719732118, 3882010307, 1776744564],
        [4141682960, 3310988675, 553637289, 902896340],
        [1134851934, 2352871630, 3699409824, 2648159817],
        [1240956131, 3107113773, 1283198141, 1924506131],
        [2669565031, 579818610, 3042504477, 2774880435],
        [2766103236, 2883057919, 4029656435, 862374500],
    ]
    outputs = [
        [3914649087, 576849849, 3593928901, 2229911004],
        [2240804226, 3691353228, 1365957195, 2654016646],
        [3562296087, 3191708229, 1147942216, 3726991905],
        [1403443605, 3591372999, 1291086759, 441919183],
        [1086200464, 2191331643, 560336446, 3658716651],
        [3249937430, 2346751812, 847844327, 2996632307],
        [2584285912, 4034195531, 3523502488, 169742686],
        [959045797, 3875435559, 1886309314, 359682705],
        [3978441347, 432478529, 3223635119, 138903045],
        [296367413, 4262059219, 13109864, 3283683422],
    ]
    outputs64 = [
        [2477551240072187391, 9577394838764454085],
        [15854241394484835714, 11398914698975566411],
        [13708282465491374871, 16007308345579681096],
        [15424829579845884309, 1898028439751125927],
        [9411697742461147792, 15714068361935982142],
        [10079222287618677782, 12870437757549876199],
        [17326737873898640088, 729039288628699544],
        [16644868984619524261, 1544825456798124994],
        [1857481142255628931, 596584038813451439],
        [18305404959516669237, 14103312907920476776],
    ]
    for seed, expected, expected64 in zip(inputs, outputs, outputs64):
        expected = np.array(expected, dtype=np.uint32)
        ss = SeedSequence(seed)
        state = ss.generate_state(len(expected))
        assert_array_equal(state, expected)
        state64 = ss.generate_state(len(expected64), dtype=np.uint64)
        assert_array_equal(state64, expected64)


def test_zero_padding():
    """ Ensure that the implicit zero-padding does not cause problems.
    """
    # Ensure that large integers are inserted in little-endian fashion to avoid
    # trailing 0s.
    ss0 = SeedSequence(42)
    ss1 = SeedSequence(42 << 32)
    assert_array_compare(
        np.not_equal,
        ss0.generate_state(4),
        ss1.generate_state(4))

    # Ensure backwards compatibility with the original 0.17 release for small
    # integers and no spawn key.
    expected42 = np.array([3444837047, 2669555309, 2046530742, 3581440988],
                          dtype=np.uint32)
    assert_array_equal(SeedSequence(42).generate_state(4), expected42)

    # Regression test for gh-16539 to ensure that the implicit 0s don't
    # conflict with spawn keys.
    assert_array_compare(
        np.not_equal,
        SeedSequence(42, spawn_key=(0,)).generate_state(4),
        expected42)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_unary.py ===
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import SparseArray


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.parametrize("fill_value", [0, np.nan])
@pytest.mark.parametrize("op", [operator.pos, operator.neg])
def test_unary_op(op, fill_value):
    arr = np.array([0, 1, np.nan, 2])
    sparray = SparseArray(arr, fill_value=fill_value)
    result = op(sparray)
    expected = SparseArray(op(arr), fill_value=op(fill_value))
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("fill_value", [True, False])
def test_invert(fill_value):
    arr = np.array([True, False, False, True])
    sparray = SparseArray(arr, fill_value=fill_value)
    result = ~sparray
    expected = SparseArray(~arr, fill_value=not fill_value)
    tm.assert_sp_array_equal(result, expected)

    result = ~pd.Series(sparray)
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)

    result = ~pd.DataFrame({"A": sparray})
    expected = pd.DataFrame({"A": expected})
    tm.assert_frame_equal(result, expected)


class TestUnaryMethods:
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_neg_operator(self):
        arr = SparseArray([-1, -2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
        res = -arr
        exp = SparseArray([1, 2, np.nan, -3], fill_value=np.nan, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

        arr = SparseArray([-1, -2, 1, 3], fill_value=-1, dtype=np.int8)
        res = -arr
        exp = SparseArray([1, 2, -1, -3], fill_value=1, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_abs_operator(self):
        arr = SparseArray([-1, -2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
        res = abs(arr)
        exp = SparseArray([1, 2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

        arr = SparseArray([-1, -2, 1, 3], fill_value=-1, dtype=np.int8)
        res = abs(arr)
        exp = SparseArray([1, 2, 1, 3], fill_value=1, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

    def test_invert_operator(self):
        arr = SparseArray([False, True, False, True], fill_value=False, dtype=np.bool_)
        exp = SparseArray(
            np.invert([False, True, False, True]), fill_value=True, dtype=np.bool_
        )
        res = ~arr
        tm.assert_sp_array_equal(exp, res)

        arr = SparseArray([0, 1, 0, 2, 3, 0], fill_value=0, dtype=np.int32)
        res = ~arr
        exp = SparseArray([-1, -2, -1, -3, -4, -1], fill_value=-1, dtype=np.int32)
        tm.assert_sp_array_equal(exp, res)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_can_hold_element.py ===
import numpy as np

from pandas.core.dtypes.cast import can_hold_element


def test_can_hold_element_range(any_int_numpy_dtype):
    # GH#44261
    dtype = np.dtype(any_int_numpy_dtype)
    arr = np.array([], dtype=dtype)

    rng = range(2, 127)
    assert can_hold_element(arr, rng)

    # negatives -> can't be held by uint dtypes
    rng = range(-2, 127)
    if dtype.kind == "i":
        assert can_hold_element(arr, rng)
    else:
        assert not can_hold_element(arr, rng)

    rng = range(2, 255)
    if dtype == "int8":
        assert not can_hold_element(arr, rng)
    else:
        assert can_hold_element(arr, rng)

    rng = range(-255, 65537)
    if dtype.kind == "u":
        assert not can_hold_element(arr, rng)
    elif dtype.itemsize < 4:
        assert not can_hold_element(arr, rng)
    else:
        assert can_hold_element(arr, rng)

    # empty
    rng = range(-(10**10), -(10**10))
    assert len(rng) == 0
    # assert can_hold_element(arr, rng)

    rng = range(10**10, 10**10)
    assert len(rng) == 0
    assert can_hold_element(arr, rng)


def test_can_hold_element_int_values_float_ndarray():
    arr = np.array([], dtype=np.int64)

    element = np.array([1.0, 2.0])
    assert can_hold_element(arr, element)

    assert not can_hold_element(arr, element + 0.5)

    # integer but not losslessly castable to int64
    element = np.array([3, 2**65], dtype=np.float64)
    assert not can_hold_element(arr, element)


def test_can_hold_element_int8_int():
    arr = np.array([], dtype=np.int8)

    element = 2
    assert can_hold_element(arr, element)
    assert can_hold_element(arr, np.int8(element))
    assert can_hold_element(arr, np.uint8(element))
    assert can_hold_element(arr, np.int16(element))
    assert can_hold_element(arr, np.uint16(element))
    assert can_hold_element(arr, np.int32(element))
    assert can_hold_element(arr, np.uint32(element))
    assert can_hold_element(arr, np.int64(element))
    assert can_hold_element(arr, np.uint64(element))

    element = 2**9
    assert not can_hold_element(arr, element)
    assert not can_hold_element(arr, np.int16(element))
    assert not can_hold_element(arr, np.uint16(element))
    assert not can_hold_element(arr, np.int32(element))
    assert not can_hold_element(arr, np.uint32(element))
    assert not can_hold_element(arr, np.int64(element))
    assert not can_hold_element(arr, np.uint64(element))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_to_dict_of_blocks.py ===
import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray

pytestmark = td.skip_array_manager_invalid_test


class TestToDictOfBlocks:
    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_no_copy_blocks(self, float_frame, using_copy_on_write):
        # GH#9607
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]

        _last_df = None
        # use the copy=False, change a column
        blocks = df._to_dict_of_blocks()
        for _df in blocks.values():
            _last_df = _df
            if column in _df:
                _df.loc[:, column] = _df[column] + 1

        if not using_copy_on_write:
            # make sure we did change the original DataFrame
            assert _last_df is not None and _last_df[column].equals(df[column])
        else:
            assert _last_df is not None and not _last_df[column].equals(df[column])


@pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
def test_to_dict_of_blocks_item_cache(using_copy_on_write, warn_copy_on_write):
    # Calling to_dict_of_blocks should not poison item_cache
    df = DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    df["c"] = NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
    mgr = df._mgr
    assert len(mgr.blocks) == 3  # i.e. not consolidated

    ser = df["b"]  # populations item_cache["b"]

    df._to_dict_of_blocks()

    if using_copy_on_write:
        with pytest.raises(ValueError, match="read-only"):
            ser.values[0] = "foo"
    elif warn_copy_on_write:
        ser.values[0] = "foo"
        assert df.loc[0, "b"] == "foo"
        # with warning mode, the item cache is disabled
        assert df["b"] is not ser
    else:
        # Check that the to_dict_of_blocks didn't break link between ser and df
        ser.values[0] = "foo"
        assert df.loc[0, "b"] == "foo"

        assert df["b"] is ser


def test_set_change_dtype_slice():
    # GH#8850
    cols = MultiIndex.from_tuples([("1st", "a"), ("2nd", "b"), ("3rd", "c")])
    df = DataFrame([[1.0, 2, 3], [4.0, 5, 6]], columns=cols)
    df["2nd"] = df["2nd"] * 2.0

    blocks = df._to_dict_of_blocks()
    assert sorted(blocks.keys()) == ["float64", "int64"]
    tm.assert_frame_equal(
        blocks["float64"], DataFrame([[1.0, 4.0], [4.0, 10.0]], columns=cols[:2])
    )
    tm.assert_frame_equal(blocks["int64"], DataFrame([[3], [6]], columns=cols[2:]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_float.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import numpy as np
import pytest

from pandas.compat import is_platform_linux

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@skip_pyarrow  # ParserError: CSV parse error: Empty CSV file or block
def test_float_parser(all_parsers):
    # see gh-9565
    parser = all_parsers
    data = "45e-1,4.5,45.,inf,-inf"
    result = parser.read_csv(StringIO(data), header=None)

    expected = DataFrame([[float(s) for s in data.split(",")]])
    tm.assert_frame_equal(result, expected)


def test_scientific_no_exponent(all_parsers_all_precisions):
    # see gh-12215
    df = DataFrame.from_dict({"w": ["2e"], "x": ["3E"], "y": ["42e"], "z": ["632E"]})
    data = df.to_csv(index=False)
    parser, precision = all_parsers_all_precisions

    df_roundtrip = parser.read_csv(StringIO(data), float_precision=precision)
    tm.assert_frame_equal(df_roundtrip, df)


@pytest.mark.parametrize(
    "neg_exp",
    [
        -617,
        -100000,
        pytest.param(-99999999999999999, marks=pytest.mark.skip_ubsan),
    ],
)
def test_very_negative_exponent(all_parsers_all_precisions, neg_exp):
    # GH#38753
    parser, precision = all_parsers_all_precisions

    data = f"data\n10E{neg_exp}"
    result = parser.read_csv(StringIO(data), float_precision=precision)
    expected = DataFrame({"data": [0.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.skip_ubsan
@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
@pytest.mark.parametrize("exp", [999999999999999999, -999999999999999999])
def test_too_many_exponent_digits(all_parsers_all_precisions, exp, request):
    # GH#38753
    parser, precision = all_parsers_all_precisions
    data = f"data\n10E{exp}"
    result = parser.read_csv(StringIO(data), float_precision=precision)
    if precision == "round_trip":
        if exp == 999999999999999999 and is_platform_linux():
            mark = pytest.mark.xfail(reason="GH38794, on Linux gives object result")
            request.applymarker(mark)

        value = np.inf if exp > 0 else 0.0
        expected = DataFrame({"data": [value]})
    else:
        expected = DataFrame({"data": [f"10E{exp}"]})

    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_util.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    date_range,
)
import pandas._testing as tm
from pandas.core.reshape.util import cartesian_product


class TestCartesianProduct:
    def test_simple(self):
        x, y = list("ABC"), [1, 22]
        result1, result2 = cartesian_product([x, y])
        expected1 = np.array(["A", "A", "B", "B", "C", "C"])
        expected2 = np.array([1, 22, 1, 22, 1, 22])
        tm.assert_numpy_array_equal(result1, expected1)
        tm.assert_numpy_array_equal(result2, expected2)

    def test_datetimeindex(self):
        # regression test for GitHub issue #6439
        # make sure that the ordering on datetimeindex is consistent
        x = date_range("2000-01-01", periods=2)
        result1, result2 = (Index(y).day for y in cartesian_product([x, x]))
        expected1 = Index([1, 1, 2, 2], dtype=np.int32)
        expected2 = Index([1, 2, 1, 2], dtype=np.int32)
        tm.assert_index_equal(result1, expected1)
        tm.assert_index_equal(result2, expected2)

    def test_tzaware_retained(self):
        x = date_range("2000-01-01", periods=2, tz="US/Pacific")
        y = np.array([3, 4])
        result1, result2 = cartesian_product([x, y])

        expected = x.repeat(2)
        tm.assert_index_equal(result1, expected)

    def test_tzaware_retained_categorical(self):
        x = date_range("2000-01-01", periods=2, tz="US/Pacific").astype("category")
        y = np.array([3, 4])
        result1, result2 = cartesian_product([x, y])

        expected = x.repeat(2)
        tm.assert_index_equal(result1, expected)

    @pytest.mark.parametrize("x, y", [[[], []], [[0, 1], []], [[], ["a", "b", "c"]]])
    def test_empty(self, x, y):
        # product of empty factors
        expected1 = np.array([], dtype=np.asarray(x).dtype)
        expected2 = np.array([], dtype=np.asarray(y).dtype)
        result1, result2 = cartesian_product([x, y])
        tm.assert_numpy_array_equal(result1, expected1)
        tm.assert_numpy_array_equal(result2, expected2)

    def test_empty_input(self):
        # empty product (empty input):
        result = cartesian_product([])
        expected = []
        assert result == expected

    @pytest.mark.parametrize(
        "X", [1, [1], [1, 2], [[1], 2], "a", ["a"], ["a", "b"], [["a"], "b"]]
    )
    def test_invalid_input(self, X):
        msg = "Input must be a list-like of list-likes"

        with pytest.raises(TypeError, match=msg):
            cartesian_product(X=X)

    def test_exceed_product_space(self):
        # GH31355: raise useful error when produce space is too large
        msg = "Product space too large to allocate arrays!"

        with pytest.raises(ValueError, match=msg):
            dims = [np.arange(0, 22, dtype=np.int16) for i in range(12)] + [
                (np.arange(15128, dtype=np.int16)),
            ]
            cartesian_product(X=dims)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_deltafunctions.py ===
from sympy.core.function import Function
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.integrals.deltafunctions import change_mul, deltaintegrate

f = Function("f")
x_1, x_2, x, y, z = symbols("x_1 x_2 x y z")


def test_change_mul():
    assert change_mul(x, x) == (None, None)
    assert change_mul(x*y, x) == (None, None)
    assert change_mul(x*y*DiracDelta(x), x) == (DiracDelta(x), x*y)
    assert change_mul(x*y*DiracDelta(x)*DiracDelta(y), x) == \
        (DiracDelta(x), x*y*DiracDelta(y))
    assert change_mul(DiracDelta(x)**2, x) == \
        (DiracDelta(x), DiracDelta(x))
    assert change_mul(y*DiracDelta(x)**2, x) == \
        (DiracDelta(x), y*DiracDelta(x))


def test_deltaintegrate():
    assert deltaintegrate(x, x) is None
    assert deltaintegrate(x + DiracDelta(x), x) is None
    assert deltaintegrate(DiracDelta(x, 0), x) == Heaviside(x)
    for n in range(10):
        assert deltaintegrate(DiracDelta(x, n + 1), x) == DiracDelta(x, n)
    assert deltaintegrate(DiracDelta(x), x) == Heaviside(x)
    assert deltaintegrate(DiracDelta(-x), x) == Heaviside(x)
    assert deltaintegrate(DiracDelta(x - y), x) == Heaviside(x - y)
    assert deltaintegrate(DiracDelta(y - x), x) == Heaviside(x - y)

    assert deltaintegrate(x*DiracDelta(x), x) == 0
    assert deltaintegrate((x - y)*DiracDelta(x - y), x) == 0

    assert deltaintegrate(DiracDelta(x)**2, x) == DiracDelta(0)*Heaviside(x)
    assert deltaintegrate(y*DiracDelta(x)**2, x) == \
        y*DiracDelta(0)*Heaviside(x)
    assert deltaintegrate(DiracDelta(x, 1), x) == DiracDelta(x, 0)
    assert deltaintegrate(y*DiracDelta(x, 1), x) == y*DiracDelta(x, 0)
    assert deltaintegrate(DiracDelta(x, 1)**2, x) == -DiracDelta(0, 2)*Heaviside(x)
    assert deltaintegrate(y*DiracDelta(x, 1)**2, x) == -y*DiracDelta(0, 2)*Heaviside(x)


    assert deltaintegrate(DiracDelta(x) * f(x), x) == f(0) * Heaviside(x)
    assert deltaintegrate(DiracDelta(-x) * f(x), x) == f(0) * Heaviside(x)
    assert deltaintegrate(DiracDelta(x - 1) * f(x), x) == f(1) * Heaviside(x - 1)
    assert deltaintegrate(DiracDelta(1 - x) * f(x), x) == f(1) * Heaviside(x - 1)
    assert deltaintegrate(DiracDelta(x**2 + x - 2), x) == \
        Heaviside(x - 1)/3 + Heaviside(x + 2)/3

    p = cos(x)*(DiracDelta(x) + DiracDelta(x**2 - 1))*sin(x)*(x - pi)
    assert deltaintegrate(p, x) - (-pi*(cos(1)*Heaviside(-1 + x)*sin(1)/2 - \
        cos(1)*Heaviside(1 + x)*sin(1)/2) + \
        cos(1)*Heaviside(1 + x)*sin(1)/2 + \
        cos(1)*Heaviside(-1 + x)*sin(1)/2) == 0

    p = x_2*DiracDelta(x - x_2)*DiracDelta(x_2 - x_1)
    assert deltaintegrate(p, x_2) == x*DiracDelta(x - x_1)*Heaviside(x_2 - x)

    p = x*y**2*z*DiracDelta(y - x)*DiracDelta(y - z)*DiracDelta(x - z)
    assert deltaintegrate(p, y) == x**3*z*DiracDelta(x - z)**2*Heaviside(y - x)
    assert deltaintegrate((x + 1)*DiracDelta(2*x), x) == S.Half * Heaviside(x)
    assert deltaintegrate((x + 1)*DiracDelta(x*Rational(2, 3) + Rational(4, 9)), x) == \
        S.Half * Heaviside(x + Rational(2, 3))

    a, b, c = symbols('a b c', commutative=False)
    assert deltaintegrate(DiracDelta(x - y)*f(x - b)*f(x - a), x) == \
        f(y - b)*f(y - a)*Heaviside(x - y)

    p = f(x - a)*DiracDelta(x - y)*f(x - c)*f(x - b)
    assert deltaintegrate(p, x) == f(y - a)*f(y - c)*f(y - b)*Heaviside(x - y)

    p = DiracDelta(x - z)*f(x - b)*f(x - a)*DiracDelta(x - y)
    assert deltaintegrate(p, x) == DiracDelta(y - z)*f(y - b)*f(y - a) * \
        Heaviside(x - y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\fail_initialstub_already_started.py ===
"""
Testing initialstub throwing an already started exception.
"""

import greenlet

a = None
b = None
c = None
main = greenlet.getcurrent()

# If we switch into a dead greenlet,
# we go looking for its parents.
# if a parent is not yet started, we start it.

results = []

def a_run(*args):
    #results.append('A')
    results.append(('Begin A', args))


def c_run():
    results.append('Begin C')
    b.switch('From C')
    results.append('C done')

class A(greenlet.greenlet): pass

class B(greenlet.greenlet):
    doing_it = False
    def __getattribute__(self, name):
        if name == 'run' and not self.doing_it:
            assert greenlet.getcurrent() is c
            self.doing_it = True
            results.append('Switch to b from B.__getattribute__ in '
                           + type(greenlet.getcurrent()).__name__)
            b.switch()
            results.append('B.__getattribute__ back from main in '
                           + type(greenlet.getcurrent()).__name__)
        if name == 'run':
            name = '_B_run'
        return object.__getattribute__(self, name)

    def _B_run(self, *arg):
        results.append(('Begin B', arg))
        results.append('_B_run switching to main')
        main.switch('From B')

class C(greenlet.greenlet):
    pass
a = A(a_run)
b = B(parent=a)
c = C(c_run, b)

# Start a child; while running, it will start B,
# but starting B will ALSO start B.
result = c.switch()
results.append(('main from c', result))

# Switch back to C, which was in the middle of switching
# already. This will throw the ``GreenletStartedWhileInPython``
# exception, which results in parent A getting started (B is finished)
c.switch()

results.append(('A dead?', a.dead, 'B dead?', b.dead, 'C dead?', c.dead))

# A and B should both be dead now.
assert a.dead
assert b.dead
assert not c.dead

result = c.switch()
results.append(('main from c.2', result))
# Now C is dead
assert c.dead

print("RESULTS:", results)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_warnings.py ===
"""
Tests which scan for certain occurrences in the code, they may not find
all of these occurrences but should catch almost all.
"""
import ast
import tokenize
from pathlib import Path

import pytest

import numpy


class ParseCall(ast.NodeVisitor):
    def __init__(self):
        self.ls = []

    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.ls.append(node.attr)

    def visit_Name(self, node):
        self.ls.append(node.id)


class FindFuncs(ast.NodeVisitor):
    def __init__(self, filename):
        super().__init__()
        self.__filename = filename

    def visit_Call(self, node):
        p = ParseCall()
        p.visit(node.func)
        ast.NodeVisitor.generic_visit(self, node)

        if p.ls[-1] == 'simplefilter' or p.ls[-1] == 'filterwarnings':
            if node.args[0].value == "ignore":
                raise AssertionError(
                    "warnings should have an appropriate stacklevel; "
                    f"found in {self.__filename} on line {node.lineno}")

        if p.ls[-1] == 'warn' and (
                len(p.ls) == 1 or p.ls[-2] == 'warnings'):

            if "testing/tests/test_warnings.py" == self.__filename:
                # This file
                return

            # See if stacklevel exists:
            if len(node.args) == 3:
                return
            args = {kw.arg for kw in node.keywords}
            if "stacklevel" in args:
                return
            raise AssertionError(
                "warnings should have an appropriate stacklevel; "
                f"found in {self.__filename} on line {node.lineno}")


@pytest.mark.slow
def test_warning_calls():
    # combined "ignore" and stacklevel error
    base = Path(numpy.__file__).parent

    for path in base.rglob("*.py"):
        if base / "testing" in path.parents:
            continue
        if path == base / "__init__.py":
            continue
        if path == base / "random" / "__init__.py":
            continue
        if path == base / "conftest.py":
            continue
        # use tokenize to auto-detect encoding on systems where no
        # default encoding is defined (e.g. LANG='C')
        with tokenize.open(str(path)) as file:
            tree = ast.parse(file.read())
            FindFuncs(path).visit(tree)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_first_valid_index.py ===
"""
Includes test for last_valid_index.
"""
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)


class TestFirstValidIndex:
    def test_first_valid_index_single_nan(self, frame_or_series):
        # GH#9752 Series/DataFrame should both return None, not raise
        obj = frame_or_series([np.nan])

        assert obj.first_valid_index() is None
        assert obj.iloc[:0].first_valid_index() is None

    @pytest.mark.parametrize(
        "empty", [DataFrame(), Series(dtype=object), Series([], index=[], dtype=object)]
    )
    def test_first_valid_index_empty(self, empty):
        # GH#12800
        assert empty.last_valid_index() is None
        assert empty.first_valid_index() is None

    @pytest.mark.parametrize(
        "data,idx,expected_first,expected_last",
        [
            ({"A": [1, 2, 3]}, [1, 1, 2], 1, 2),
            ({"A": [1, 2, 3]}, [1, 2, 2], 1, 2),
            ({"A": [1, 2, 3, 4]}, ["d", "d", "d", "d"], "d", "d"),
            ({"A": [1, np.nan, 3]}, [1, 1, 2], 1, 2),
            ({"A": [np.nan, np.nan, 3]}, [1, 1, 2], 2, 2),
            ({"A": [1, np.nan, 3]}, [1, 2, 2], 1, 2),
        ],
    )
    def test_first_last_valid_frame(self, data, idx, expected_first, expected_last):
        # GH#21441
        df = DataFrame(data, index=idx)
        assert expected_first == df.first_valid_index()
        assert expected_last == df.last_valid_index()

    @pytest.mark.parametrize(
        "index",
        [Index([str(i) for i in range(20)]), date_range("2020-01-01", periods=20)],
    )
    def test_first_last_valid(self, index):
        mat = np.random.default_rng(2).standard_normal(len(index))
        mat[:5] = np.nan
        mat[-5:] = np.nan

        frame = DataFrame({"foo": mat}, index=index)
        assert frame.first_valid_index() == frame.index[5]
        assert frame.last_valid_index() == frame.index[-6]

        ser = frame["foo"]
        assert ser.first_valid_index() == frame.index[5]
        assert ser.last_valid_index() == frame.index[-6]

    @pytest.mark.parametrize(
        "index",
        [Index([str(i) for i in range(10)]), date_range("2020-01-01", periods=10)],
    )
    def test_first_last_valid_all_nan(self, index):
        # GH#17400: no valid entries
        frame = DataFrame(np.nan, columns=["foo"], index=index)

        assert frame.last_valid_index() is None
        assert frame.first_valid_index() is None

        ser = frame["foo"]
        assert ser.first_valid_index() is None
        assert ser.last_valid_index() is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_is_monotonic.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "in_vals, out_vals",
    [
        # Basics: strictly increasing (T), strictly decreasing (F),
        # abs val increasing (F), non-strictly increasing (T)
        ([1, 2, 5, 3, 2, 0, 4, 5, -6, 1, 1], [True, False, False, True]),
        # Test with inf vals
        (
            [1, 2.1, np.inf, 3, 2, np.inf, -np.inf, 5, 11, 1, -np.inf],
            [True, False, True, False],
        ),
        # Test with nan vals; should always be False
        (
            [1, 2, np.nan, 3, 2, np.nan, np.nan, 5, -np.inf, 1, np.nan],
            [False, False, False, False],
        ),
    ],
)
def test_is_monotonic_increasing(in_vals, out_vals):
    # GH 17015
    source_dict = {
        "A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "B": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
        "C": in_vals,
    }
    df = DataFrame(source_dict)
    result = df.groupby("B").C.is_monotonic_increasing
    index = Index(list("abcd"), name="B")
    expected = Series(index=index, data=out_vals, name="C")
    tm.assert_series_equal(result, expected)

    # Also check result equal to manually taking x.is_monotonic_increasing.
    expected = df.groupby(["B"]).C.apply(lambda x: x.is_monotonic_increasing)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "in_vals, out_vals",
    [
        # Basics: strictly decreasing (T), strictly increasing (F),
        # abs val decreasing (F), non-strictly increasing (T)
        ([10, 9, 7, 3, 4, 5, -3, 2, 0, 1, 1], [True, False, False, True]),
        # Test with inf vals
        (
            [np.inf, 1, -np.inf, np.inf, 2, -3, -np.inf, 5, -3, -np.inf, -np.inf],
            [True, True, False, True],
        ),
        # Test with nan vals; should always be False
        (
            [1, 2, np.nan, 3, 2, np.nan, np.nan, 5, -np.inf, 1, np.nan],
            [False, False, False, False],
        ),
    ],
)
def test_is_monotonic_decreasing(in_vals, out_vals):
    # GH 17015
    source_dict = {
        "A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "B": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
        "C": in_vals,
    }

    df = DataFrame(source_dict)
    result = df.groupby("B").C.is_monotonic_decreasing
    index = Index(list("abcd"), name="B")
    expected = Series(index=index, data=out_vals, name="C")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\test_constructors.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestIndexConstructor:
    # Tests for the Index constructor, specifically for cases that do
    #  not return a subclass

    @pytest.mark.parametrize("value", [1, np.int64(1)])
    def test_constructor_corner(self, value):
        # corner case
        msg = (
            r"Index\(\.\.\.\) must be called with a collection of some "
            f"kind, {value} was passed"
        )
        with pytest.raises(TypeError, match=msg):
            Index(value)

    @pytest.mark.parametrize("index_vals", [[("A", 1), "B"], ["B", ("A", 1)]])
    def test_construction_list_mixed_tuples(self, index_vals):
        # see gh-10697: if we are constructing from a mixed list of tuples,
        # make sure that we are independent of the sorting order.
        index = Index(index_vals)
        assert isinstance(index, Index)
        assert not isinstance(index, MultiIndex)

    def test_constructor_cast(self):
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            Index(["a", "b", "c"], dtype=float)

    @pytest.mark.parametrize("tuple_list", [[()], [(), ()]])
    def test_construct_empty_tuples(self, tuple_list):
        # GH #45608
        result = Index(tuple_list)
        expected = MultiIndex.from_tuples(tuple_list)

        tm.assert_index_equal(result, expected)

    def test_index_string_inference(self):
        # GH#54430
        expected = Index(["a", "b"], dtype=pd.StringDtype(na_value=np.nan))
        with pd.option_context("future.infer_string", True):
            ser = Index(["a", "b"])
        tm.assert_index_equal(ser, expected)

        expected = Index(["a", 1], dtype="object")
        with pd.option_context("future.infer_string", True):
            ser = Index(["a", 1])
        tm.assert_index_equal(ser, expected)

    def test_inference_on_pandas_objects(self):
        # GH#56012
        idx = Index([pd.Timestamp("2019-12-31")], dtype=object)
        with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
            result = Index(idx)
        assert result.dtype != np.object_

        ser = Series([pd.Timestamp("2019-12-31")], dtype=object)

        with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
            result = Index(ser)
        assert result.dtype != np.object_

    def test_constructor_not_read_only(self):
        # GH#57130
        ser = Series([1, 2], dtype=object)
        with pd.option_context("mode.copy_on_write", True):
            idx = Index(ser)
            assert idx._values.flags.writeable

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_reindex.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    Interval,
)
import pandas._testing as tm


class TestReindex:
    def test_reindex_list_non_unique(self):
        # GH#11586
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(["a", "c"])

    def test_reindex_categorical_non_unique(self):
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(Categorical(["a", "c"]))

    def test_reindex_list_non_unique_unused_category(self):
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(["a", "c"])

    def test_reindex_categorical_non_unique_unused_category(self):
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(Categorical(["a", "c"]))

    def test_reindex_duplicate_target(self):
        # See GH25459
        cat = CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c", "d"])
        res, indexer = cat.reindex(["a", "c", "c"])
        exp = Index(["a", "c", "c"])
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))

        res, indexer = cat.reindex(
            CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])
        )
        exp = CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))

    def test_reindex_empty_index(self):
        # See GH16770
        c = CategoricalIndex([])
        res, indexer = c.reindex(["a", "b"])
        tm.assert_index_equal(res, Index(["a", "b"]), exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([-1, -1], dtype=np.intp))

    def test_reindex_categorical_added_category(self):
        # GH 42424
        ci = CategoricalIndex(
            [Interval(0, 1, closed="right"), Interval(1, 2, closed="right")],
            ordered=True,
        )
        ci_add = CategoricalIndex(
            [
                Interval(0, 1, closed="right"),
                Interval(1, 2, closed="right"),
                Interval(2, 3, closed="right"),
                Interval(3, 4, closed="right"),
            ],
            ordered=True,
        )
        result, _ = ci.reindex(ci_add)
        expected = ci_add
        tm.assert_index_equal(expected, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_take.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_take(idx):
    indexer = [4, 3, 0, 2]
    result = idx.take(indexer)
    expected = idx[indexer]
    assert result.equals(expected)

    # GH 10791
    msg = "'MultiIndex' object has no attribute 'freq'"
    with pytest.raises(AttributeError, match=msg):
        idx.freq


def test_take_invalid_kwargs(idx):
    indices = [1, 2]

    msg = r"take\(\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        idx.take(indices, foo=2)

    msg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        idx.take(indices, out=indices)

    msg = "the 'mode' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        idx.take(indices, mode="clip")


def test_take_fill_value():
    # GH 12631
    vals = [["A", "B"], [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]]
    idx = pd.MultiIndex.from_product(vals, names=["str", "dt"])

    result = idx.take(np.array([1, 0, -1]))
    exp_vals = [
        ("A", pd.Timestamp("2011-01-02")),
        ("A", pd.Timestamp("2011-01-01")),
        ("B", pd.Timestamp("2011-01-02")),
    ]
    expected = pd.MultiIndex.from_tuples(exp_vals, names=["str", "dt"])
    tm.assert_index_equal(result, expected)

    # fill_value
    result = idx.take(np.array([1, 0, -1]), fill_value=True)
    exp_vals = [
        ("A", pd.Timestamp("2011-01-02")),
        ("A", pd.Timestamp("2011-01-01")),
        (np.nan, pd.NaT),
    ]
    expected = pd.MultiIndex.from_tuples(exp_vals, names=["str", "dt"])
    tm.assert_index_equal(result, expected)

    # allow_fill=False
    result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
    exp_vals = [
        ("A", pd.Timestamp("2011-01-02")),
        ("A", pd.Timestamp("2011-01-01")),
        ("B", pd.Timestamp("2011-01-02")),
    ]
    expected = pd.MultiIndex.from_tuples(exp_vals, names=["str", "dt"])
    tm.assert_index_equal(result, expected)

    msg = "When allow_fill=True and fill_value is not None, all indices must be >= -1"
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -2]), fill_value=True)
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -5]), fill_value=True)

    msg = "index -5 is out of bounds for( axis 0 with)? size 4"
    with pytest.raises(IndexError, match=msg):
        idx.take(np.array([1, -5]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_inf.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import numpy as np
import pytest

from pandas import (
    DataFrame,
    option_context,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@xfail_pyarrow  # AssertionError: DataFrame.index are different
@pytest.mark.parametrize("na_filter", [True, False])
def test_inf_parsing(all_parsers, na_filter):
    parser = all_parsers
    data = """\
,A
a,inf
b,-inf
c,+Inf
d,-Inf
e,INF
f,-INF
g,+INf
h,-INf
i,inF
j,-inF"""
    expected = DataFrame(
        {"A": [float("inf"), float("-inf")] * 5},
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # AssertionError: DataFrame.index are different
@pytest.mark.parametrize("na_filter", [True, False])
def test_infinity_parsing(all_parsers, na_filter):
    parser = all_parsers
    data = """\
,A
a,Infinity
b,-Infinity
c,+Infinity
"""
    expected = DataFrame(
        {"A": [float("infinity"), float("-infinity"), float("+infinity")]},
        index=["a", "b", "c"],
    )
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


def test_read_csv_with_use_inf_as_na(all_parsers):
    # https://github.com/pandas-dev/pandas/issues/35493
    parser = all_parsers
    data = "1.0\nNaN\n3.0"
    msg = "use_inf_as_na option is deprecated"
    warn = FutureWarning
    if parser.engine == "pyarrow":
        warn = (FutureWarning, DeprecationWarning)

    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        with option_context("use_inf_as_na", True):
            result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([1.0, np.nan, 3.0])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_particle.py ===
from sympy import symbols
from sympy.physics.mechanics import Point, Particle, ReferenceFrame, inertia
from sympy.physics.mechanics.body_base import BodyBase
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_particle_default():
    # Test default
    p = Particle('P')
    assert p.name == 'P'
    assert p.mass == symbols('P_mass')
    assert p.masscenter.name == 'P_masscenter'
    assert p.potential_energy == 0
    assert p.__str__() == 'P'
    assert p.__repr__() == ("Particle('P', masscenter=P_masscenter, "
                            "mass=P_mass)")
    raises(AttributeError, lambda: p.frame)


def test_particle():
    # Test initializing with parameters
    m, m2, v1, v2, v3, r, g, h = symbols('m m2 v1 v2 v3 r g h')
    P = Point('P')
    P2 = Point('P2')
    p = Particle('pa', P, m)
    assert isinstance(p, BodyBase)
    assert p.mass == m
    assert p.point == P
    # Test the mass setter
    p.mass = m2
    assert p.mass == m2
    # Test the point setter
    p.point = P2
    assert p.point == P2
    # Test the linear momentum function
    N = ReferenceFrame('N')
    O = Point('O')
    P2.set_pos(O, r * N.y)
    P2.set_vel(N, v1 * N.x)
    raises(TypeError, lambda: Particle(P, P, m))
    raises(TypeError, lambda: Particle('pa', m, m))
    assert p.linear_momentum(N) == m2 * v1 * N.x
    assert p.angular_momentum(O, N) == -m2 * r * v1 * N.z
    P2.set_vel(N, v2 * N.y)
    assert p.linear_momentum(N) == m2 * v2 * N.y
    assert p.angular_momentum(O, N) == 0
    P2.set_vel(N, v3 * N.z)
    assert p.linear_momentum(N) == m2 * v3 * N.z
    assert p.angular_momentum(O, N) == m2 * r * v3 * N.x
    P2.set_vel(N, v1 * N.x + v2 * N.y + v3 * N.z)
    assert p.linear_momentum(N) == m2 * (v1 * N.x + v2 * N.y + v3 * N.z)
    assert p.angular_momentum(O, N) == m2 * r * (v3 * N.x - v1 * N.z)
    p.potential_energy = m * g * h
    assert p.potential_energy == m * g * h
    # TODO make the result not be system-dependent
    assert p.kinetic_energy(
        N) in [m2 * (v1 ** 2 + v2 ** 2 + v3 ** 2) / 2,
               m2 * v1 ** 2 / 2 + m2 * v2 ** 2 / 2 + m2 * v3 ** 2 / 2]


def test_parallel_axis():
    N = ReferenceFrame('N')
    m, a, b = symbols('m, a, b')
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    P = Particle('P', o, m)
    Ip = P.parallel_axis(p, N)
    Ip_expected = inertia(N, m * b ** 2, m * a ** 2, m * (a ** 2 + b ** 2),
                          ixy=-m * a * b)
    assert Ip == Ip_expected


def test_deprecated_set_potential_energy():
    m, g, h = symbols('m g h')
    P = Point('P')
    p = Particle('pa', P, m)
    with warns_deprecated_sympy():
        p.set_potential_energy(m * g * h)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_ratsimp.py ===
from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.error_functions import erf
from sympy.polys.domains import GF
from sympy.simplify.ratsimp import (ratsimp, ratsimpmodprime)

from sympy.abc import x, y, z, t, a, b, c, d, e


def test_ratsimp():
    f, g = 1/x + 1/y, (x + y)/(x*y)

    assert f != g and ratsimp(f) == g

    f, g = 1/(1 + 1/x), 1 - 1/(x + 1)

    assert f != g and ratsimp(f) == g

    f, g = x/(x + y) + y/(x + y), 1

    assert f != g and ratsimp(f) == g

    f, g = -x - y - y**2/(x + y) + x**2/(x + y), -2*y

    assert f != g and ratsimp(f) == g

    f = (a*c*x*y + a*c*z - b*d*x*y - b*d*z - b*t*x*y - b*t*x - b*t*z +
         e*x)/(x*y + z)
    G = [a*c - b*d - b*t + (-b*t*x + e*x)/(x*y + z),
         a*c - b*d - b*t - ( b*t*x - e*x)/(x*y + z)]

    assert f != g and ratsimp(f) in G

    A = sqrt(pi)

    B = log(erf(x) - 1)
    C = log(erf(x) + 1)

    D = 8 - 8*erf(x)

    f = A*B/D - A*C/D + A*C*erf(x)/D - A*B*erf(x)/D + 2*A/D

    assert ratsimp(f) == A*B/8 - A*C/8 - A/(4*erf(x) - 4)


def test_ratsimpmodprime():
    a = y**5 + x + y
    b = x - y
    F = [x*y**5 - x - y]
    assert ratsimpmodprime(a/b, F, x, y, order='lex') == \
        (-x**2 - x*y - x - y) / (-x**2 + x*y)

    a = x + y**2 - 2
    b = x + y**2 - y - 1
    F = [x*y - 1]
    assert ratsimpmodprime(a/b, F, x, y, order='lex') == \
        (1 + y - x)/(y - x)

    a = 5*x**3 + 21*x**2 + 4*x*y + 23*x + 12*y + 15
    b = 7*x**3 - y*x**2 + 31*x**2 + 2*x*y + 15*y + 37*x + 21
    F = [x**2 + y**2 - 1]
    assert ratsimpmodprime(a/b, F, x, y, order='lex') == \
        (1 + 5*y - 5*x)/(8*y - 6*x)

    a = x*y - x - 2*y + 4
    b = x + y**2 - 2*y
    F = [x - 2, y - 3]
    assert ratsimpmodprime(a/b, F, x, y, order='lex') == \
        Rational(2, 5)

    # Test a bug where denominators would be dropped
    assert ratsimpmodprime(x, [y - 2*x], order='lex') == \
        y/2

    a = (x**5 + 2*x**4 + 2*x**3 + 2*x**2 + x + 2/x + x**(-2))
    assert ratsimpmodprime(a, [x + 1], domain=GF(2)) == 1
    assert ratsimpmodprime(a, [x + 1], domain=GF(3)) == -1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\tests\test_rl.py ===
from sympy.core.singleton import S
from sympy.strategies.rl import (
    rm_id, glom, flatten, unpack, sort, distribute, subs, rebuild)
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x


def test_rm_id():
    rmzeros = rm_id(lambda x: x == 0)
    assert rmzeros(Basic(S(0), S(1))) == Basic(S(1))
    assert rmzeros(Basic(S(0), S(0))) == Basic(S(0))
    assert rmzeros(Basic(S(2), S(1))) == Basic(S(2), S(1))


def test_glom():
    def key(x):
        return x.as_coeff_Mul()[1]

    def count(x):
        return x.as_coeff_Mul()[0]

    def newargs(cnt, arg):
        return cnt * arg

    rl = glom(key, count, newargs)

    result = rl(Add(x, -x, 3 * x, 2, 3, evaluate=False))
    expected = Add(3 * x, 5)
    assert set(result.args) == set(expected.args)


def test_flatten():
    assert flatten(Basic(S(1), S(2), Basic(S(3), S(4)))) == \
        Basic(S(1), S(2), S(3), S(4))


def test_unpack():
    assert unpack(Basic(S(2))) == 2
    assert unpack(Basic(S(2), S(3))) == Basic(S(2), S(3))


def test_sort():
    assert sort(str)(Basic(S(3), S(1), S(2))) == Basic(S(1), S(2), S(3))


def test_distribute():
    class T1(Basic):
        pass

    class T2(Basic):
        pass

    distribute_t12 = distribute(T1, T2)
    assert distribute_t12(T1(S(1), S(2), T2(S(3), S(4)), S(5))) == \
        T2(T1(S(1), S(2), S(3), S(5)), T1(S(1), S(2), S(4), S(5)))
    assert distribute_t12(T1(S(1), S(2), S(3))) == T1(S(1), S(2), S(3))


def test_distribute_add_mul():
    x, y = symbols('x, y')
    expr = Mul(2, Add(x, y), evaluate=False)
    expected = Add(Mul(2, x), Mul(2, y))
    distribute_mul = distribute(Mul, Add)
    assert distribute_mul(expr) == expected


def test_subs():
    rl = subs(1, 2)
    assert rl(1) == 2
    assert rl(3) == 3


def test_rebuild():
    expr = Basic.__new__(Add, S(1), S(2))
    assert rebuild(expr) == 3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_arrayexpr_derivatives.py ===
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayTensorProduct, \
    PermuteDims, ArrayDiagonal, ArrayElementwiseApplyFunc, ArrayContraction, _permute_dims, Reshape
from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive

k = symbols("k")

I = Identity(k)
X = MatrixSymbol("X", k, k)
x = MatrixSymbol("x", k, 1)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

A1 = ArraySymbol("A", (3, 2, k))


def test_arrayexpr_derivatives1():

    res = array_derive(X, X)
    assert res == PermuteDims(ArrayTensorProduct(I, I), [0, 2, 1, 3])

    cg = ArrayTensorProduct(A, X, B)
    res = array_derive(cg, X)
    assert res == _permute_dims(
        ArrayTensorProduct(I, A, I, B),
        [0, 4, 2, 3, 1, 5, 6, 7])

    cg = ArrayContraction(X, (0, 1))
    res = array_derive(cg, X)
    assert res == ArrayContraction(ArrayTensorProduct(I, I), (1, 3))

    cg = ArrayDiagonal(X, (0, 1))
    res = array_derive(cg, X)
    assert res == ArrayDiagonal(ArrayTensorProduct(I, I), (1, 3))

    cg = ElementwiseApplyFunction(sin, X)
    res = array_derive(cg, X)
    assert res.dummy_eq(ArrayDiagonal(
        ArrayTensorProduct(
            ElementwiseApplyFunction(cos, X),
            I,
            I
        ), (0, 3), (1, 5)))

    cg = ArrayElementwiseApplyFunc(sin, X)
    res = array_derive(cg, X)
    assert res.dummy_eq(ArrayDiagonal(
        ArrayTensorProduct(
            I,
            I,
            ArrayElementwiseApplyFunc(cos, X)
        ), (1, 4), (3, 5)))

    res = array_derive(A1, A1)
    assert res == PermuteDims(
        ArrayTensorProduct(Identity(3), Identity(2), Identity(k)),
        [0, 2, 4, 1, 3, 5]
    )

    cg = ArrayElementwiseApplyFunc(sin, A1)
    res = array_derive(cg, A1)
    assert res.dummy_eq(ArrayDiagonal(
        ArrayTensorProduct(
            Identity(3), Identity(2), Identity(k),
            ArrayElementwiseApplyFunc(cos, A1)
        ), (1, 6), (3, 7), (5, 8)
    ))

    cg = Reshape(A, (k**2,))
    res = array_derive(cg, A)
    assert res == Reshape(PermuteDims(ArrayTensorProduct(I, I), [0, 2, 1, 3]), (k, k, k**2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\test_array_comprehension.py ===
from sympy.tensor.array.array_comprehension import ArrayComprehension, ArrayComprehensionMap
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.abc import i, j, k, l
from sympy.testing.pytest import raises
from sympy.matrices import Matrix


def test_array_comprehension():
    a = ArrayComprehension(i*j, (i, 1, 3), (j, 2, 4))
    b = ArrayComprehension(i, (i, 1, j+1))
    c = ArrayComprehension(i+j+k+l, (i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5))
    d = ArrayComprehension(k, (i, 1, 5))
    e = ArrayComprehension(i, (j, k+1, k+5))
    assert a.doit().tolist() == [[2, 3, 4], [4, 6, 8], [6, 9, 12]]
    assert a.shape == (3, 3)
    assert a.is_shape_numeric == True
    assert a.tolist() == [[2, 3, 4], [4, 6, 8], [6, 9, 12]]
    assert a.tomatrix() == Matrix([
                           [2, 3, 4],
                           [4, 6, 8],
                           [6, 9, 12]])
    assert len(a) == 9
    assert isinstance(b.doit(), ArrayComprehension)
    assert isinstance(a.doit(), ImmutableDenseNDimArray)
    assert b.subs(j, 3) == ArrayComprehension(i, (i, 1, 4))
    assert b.free_symbols == {j}
    assert b.shape == (j + 1,)
    assert b.rank() == 1
    assert b.is_shape_numeric == False
    assert c.free_symbols == set()
    assert c.function == i + j + k + l
    assert c.limits == ((i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5))
    assert c.doit().tolist() == [[[[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11]],
                                  [[5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]],
                                  [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13]]],
                                 [[[5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]],
                                  [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13]],
                                  [[7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13], [10, 11, 12, 13, 14]]]]
    assert c.free_symbols == set()
    assert c.variables == [i, j, k, l]
    assert c.bound_symbols == [i, j, k, l]
    assert d.doit().tolist() == [k, k, k, k, k]
    assert len(e) == 5
    raises(TypeError, lambda: ArrayComprehension(i*j, (i, 1, 3), (j, 2, [1, 3, 2])))
    raises(ValueError, lambda: ArrayComprehension(i*j, (i, 1, 3), (j, 2, 1)))
    raises(ValueError, lambda: ArrayComprehension(i*j, (i, 1, 3), (j, 2, j+1)))
    raises(ValueError, lambda: len(ArrayComprehension(i*j, (i, 1, 3), (j, 2, j+4))))
    raises(TypeError, lambda: ArrayComprehension(i*j, (i, 0, i + 1.5), (j, 0, 2)))
    raises(ValueError, lambda: b.tolist())
    raises(ValueError, lambda: b.tomatrix())
    raises(ValueError, lambda: c.tomatrix())

def test_arraycomprehensionmap():
    a = ArrayComprehensionMap(lambda i: i+1, (i, 1, 5))
    assert a.doit().tolist() == [2, 3, 4, 5, 6]
    assert a.shape == (5,)
    assert a.is_shape_numeric
    assert a.tolist() == [2, 3, 4, 5, 6]
    assert len(a) == 5
    assert isinstance(a.doit(), ImmutableDenseNDimArray)
    expr = ArrayComprehensionMap(lambda i: i+1, (i, 1, k))
    assert expr.doit() == expr
    assert expr.subs(k, 4) == ArrayComprehensionMap(lambda i: i+1, (i, 1, 4))
    assert expr.subs(k, 4).doit() == ImmutableDenseNDimArray([2, 3, 4, 5])
    b = ArrayComprehensionMap(lambda i: i+1, (i, 1, 2), (i, 1, 3), (i, 1, 4), (i, 1, 5))
    assert b.doit().tolist() == [[[[2, 3, 4, 5, 6], [3, 5, 7, 9, 11], [4, 7, 10, 13, 16], [5, 9, 13, 17, 21]],
                                  [[3, 5, 7, 9, 11], [5, 9, 13, 17, 21], [7, 13, 19, 25, 31], [9, 17, 25, 33, 41]],
                                  [[4, 7, 10, 13, 16], [7, 13, 19, 25, 31], [10, 19, 28, 37, 46], [13, 25, 37, 49, 61]]],
                                 [[[3, 5, 7, 9, 11], [5, 9, 13, 17, 21], [7, 13, 19, 25, 31], [9, 17, 25, 33, 41]],
                                  [[5, 9, 13, 17, 21], [9, 17, 25, 33, 41], [13, 25, 37, 49, 61], [17, 33, 49, 65, 81]],
                                  [[7, 13, 19, 25, 31], [13, 25, 37, 49, 61], [19, 37, 55, 73, 91], [25, 49, 73, 97, 121]]]]

    # tests about lambda expression
    assert ArrayComprehensionMap(lambda: 3, (i, 1, 5)).doit().tolist() == [3, 3, 3, 3, 3]
    assert ArrayComprehensionMap(lambda i: i+1, (i, 1, 5)).doit().tolist() == [2, 3, 4, 5, 6]
    raises(ValueError, lambda: ArrayComprehensionMap(i*j, (i, 1, 3), (j, 2, 4)))
    a = ArrayComprehensionMap(lambda i, j: i+j, (i, 1, 5))
    raises(ValueError, lambda: a.doit())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_compatibility.py ===
from mpmath import *
from random import seed, randint, random
import math

# Test compatibility with Python floats, which are
# IEEE doubles (53-bit)

N = 5000
seed(1)

# Choosing exponents between roughly -140, 140 ensures that
# the Python floats don't overflow or underflow
xs = [(random()-1) * 10**randint(-140, 140) for x in range(N)]
ys = [(random()-1) * 10**randint(-140, 140) for x in range(N)]

# include some equal values
ys[int(N*0.8):] = xs[int(N*0.8):]

# Detect whether Python is compiled to use 80-bit floating-point
# instructions, in which case the double compatibility test breaks
uses_x87 = -4.1974624032366689e+117 / -8.4657370748010221e-47 \
    == 4.9581771393902231e+163

def test_double_compatibility():
    mp.prec = 53
    for x, y in zip(xs, ys):
        mpx = mpf(x)
        mpy = mpf(y)
        assert mpf(x) == x
        assert (mpx < mpy) == (x < y)
        assert (mpx > mpy) == (x > y)
        assert (mpx == mpy) == (x == y)
        assert (mpx != mpy) == (x != y)
        assert (mpx <= mpy) == (x <= y)
        assert (mpx >= mpy) == (x >= y)
        assert mpx == mpx
        if uses_x87:
            mp.prec = 64
            a = mpx + mpy
            b = mpx * mpy
            c = mpx / mpy
            d = mpx % mpy
            mp.prec = 53
            assert +a == x + y
            assert +b == x * y
            assert +c == x / y
            assert +d == x % y
        else:
            assert mpx + mpy == x + y
            assert mpx * mpy == x * y
            assert mpx / mpy == x / y
            assert mpx % mpy == x % y
        assert abs(mpx) == abs(x)
        assert mpf(repr(x)) == x
        assert ceil(mpx) == math.ceil(x)
        assert floor(mpx) == math.floor(x)

def test_sqrt():
    # this fails quite often. it appers to be float
    # that rounds the wrong way, not mpf
    fail = 0
    mp.prec = 53
    for x in xs:
        x = abs(x)
        mp.prec = 100
        mp_high = mpf(x)**0.5
        mp.prec = 53
        mp_low = mpf(x)**0.5
        fp = x**0.5
        assert abs(mp_low-mp_high) <= abs(fp-mp_high)
        fail += mp_low != fp
    assert fail < N/10

def test_bugs():
    # particular bugs
    assert mpf(4.4408920985006262E-16) < mpf(1.7763568394002505E-15)
    assert mpf(-4.4408920985006262E-16) > mpf(-1.7763568394002505E-15)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_set_value.py ===
import numpy as np

from pandas.core.dtypes.common import is_float_dtype

from pandas import (
    DataFrame,
    isna,
)
import pandas._testing as tm


class TestSetValue:
    def test_set_value(self, float_frame):
        for idx in float_frame.index:
            for col in float_frame.columns:
                float_frame._set_value(idx, col, 1)
                assert float_frame[col][idx] == 1

    def test_set_value_resize(self, float_frame, using_infer_string):
        res = float_frame._set_value("foobar", "B", 0)
        assert res is None
        assert float_frame.index[-1] == "foobar"
        assert float_frame._get_value("foobar", "B") == 0

        float_frame.loc["foobar", "qux"] = 0
        assert float_frame._get_value("foobar", "qux") == 0

        res = float_frame.copy()
        res._set_value("foobar", "baz", "sam")
        if using_infer_string:
            assert res["baz"].dtype == "str"
        else:
            assert res["baz"].dtype == np.object_
        res = float_frame.copy()
        res._set_value("foobar", "baz", True)
        assert res["baz"].dtype == np.object_

        res = float_frame.copy()
        res._set_value("foobar", "baz", 5)
        assert is_float_dtype(res["baz"])
        assert isna(res["baz"].drop(["foobar"])).all()

        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            res._set_value("foobar", "baz", "sam")
        assert res.loc["foobar", "baz"] == "sam"

    def test_set_value_with_index_dtype_change(self):
        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=range(3),
            columns=list("ABC"),
        )

        # this is actually ambiguous as the 2 is interpreted as a positional
        # so column is not created
        df = df_orig.copy()
        df._set_value("C", 2, 1.0)
        assert list(df.index) == list(df_orig.index) + ["C"]
        # assert list(df.columns) == list(df_orig.columns) + [2]

        df = df_orig.copy()
        df.loc["C", 2] = 1.0
        assert list(df.index) == list(df_orig.index) + ["C"]
        # assert list(df.columns) == list(df_orig.columns) + [2]

        # create both new
        df = df_orig.copy()
        df._set_value("C", "D", 1.0)
        assert list(df.index) == list(df_orig.index) + ["C"]
        assert list(df.columns) == list(df_orig.columns) + ["D"]

        df = df_orig.copy()
        df.loc["C", "D"] = 1.0
        assert list(df.index) == list(df_orig.index) + ["C"]
        assert list(df.columns) == list(df_orig.columns) + ["D"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_unique.py ===
from datetime import (
    datetime,
    timedelta,
)

from pandas import (
    DatetimeIndex,
    NaT,
    Timestamp,
)
import pandas._testing as tm


def test_unique(tz_naive_fixture):
    idx = DatetimeIndex(["2017"] * 2, tz=tz_naive_fixture)
    expected = idx[:1]

    result = idx.unique()
    tm.assert_index_equal(result, expected)
    # GH#21737
    # Ensure the underlying data is consistent
    assert result[0] == expected[0]


def test_index_unique(rand_series_with_duplicate_datetimeindex):
    dups = rand_series_with_duplicate_datetimeindex
    index = dups.index

    uniques = index.unique()
    expected = DatetimeIndex(
        [
            datetime(2000, 1, 2),
            datetime(2000, 1, 3),
            datetime(2000, 1, 4),
            datetime(2000, 1, 5),
        ],
        dtype=index.dtype,
    )
    assert uniques.dtype == index.dtype  # sanity
    tm.assert_index_equal(uniques, expected)
    assert index.nunique() == 4

    # GH#2563
    assert isinstance(uniques, DatetimeIndex)

    dups_local = index.tz_localize("US/Eastern")
    dups_local.name = "foo"
    result = dups_local.unique()
    expected = DatetimeIndex(expected, name="foo")
    expected = expected.tz_localize("US/Eastern")
    assert result.tz is not None
    assert result.name == "foo"
    tm.assert_index_equal(result, expected)


def test_index_unique2():
    # NaT, note this is excluded
    arr = [1370745748 + t for t in range(20)] + [NaT._value]
    idx = DatetimeIndex(arr * 3)
    tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
    assert idx.nunique() == 20
    assert idx.nunique(dropna=False) == 21


def test_index_unique3():
    arr = [
        Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)
    ] + [NaT]
    idx = DatetimeIndex(arr * 3)
    tm.assert_index_equal(idx.unique(), DatetimeIndex(arr))
    assert idx.nunique() == 20
    assert idx.nunique(dropna=False) == 21


def test_is_unique_monotonic(rand_series_with_duplicate_datetimeindex):
    index = rand_series_with_duplicate_datetimeindex.index
    assert not index.is_unique

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\excel\test_odf.py ===
import functools

import numpy as np
import pytest

from pandas.compat import is_platform_windows

import pandas as pd
import pandas._testing as tm

pytest.importorskip("odf")

if is_platform_windows():
    pytestmark = pytest.mark.single_cpu


@pytest.fixture(autouse=True)
def cd_and_set_engine(monkeypatch, datapath):
    func = functools.partial(pd.read_excel, engine="odf")
    monkeypatch.setattr(pd, "read_excel", func)
    monkeypatch.chdir(datapath("io", "data", "excel"))


def test_read_invalid_types_raises():
    # the invalid_value_type.ods required manually editing
    # of the included content.xml file
    with pytest.raises(ValueError, match="Unrecognized type awesome_new_type"):
        pd.read_excel("invalid_value_type.ods")


def test_read_writer_table():
    # Also test reading tables from an text OpenDocument file
    # (.odt)
    index = pd.Index(["Row 1", "Row 2", "Row 3"], name="Header")
    expected = pd.DataFrame(
        [[1, np.nan, 7], [2, np.nan, 8], [3, np.nan, 9]],
        index=index,
        columns=["Column 1", "Unnamed: 2", "Column 3"],
    )

    result = pd.read_excel("writertable.odt", sheet_name="Table1", index_col=0)

    tm.assert_frame_equal(result, expected)


def test_read_newlines_between_xml_elements_table():
    # GH#45598
    expected = pd.DataFrame(
        [[1.0, 4.0, 7], [np.nan, np.nan, 8], [3.0, 6.0, 9]],
        columns=["Column 1", "Column 2", "Column 3"],
    )

    result = pd.read_excel("test_newlines.ods")

    tm.assert_frame_equal(result, expected)


def test_read_unempty_cells():
    expected = pd.DataFrame(
        [1, np.nan, 3, np.nan, 5],
        columns=["Column 1"],
    )

    result = pd.read_excel("test_unempty_cells.ods")

    tm.assert_frame_equal(result, expected)


def test_read_cell_annotation():
    expected = pd.DataFrame(
        ["test", np.nan, "test 3"],
        columns=["Column 1"],
    )

    result = pd.read_excel("test_cell_annotation.ods")

    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_duplicated.py ===
import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True], name="name")),
        ("last", Series([True, True, False, False, False], name="name")),
        (False, Series([True, True, True, False, True], name="name")),
    ],
)
def test_duplicated_keep(keep, expected):
    ser = Series(["a", "b", "b", "c", "a"], name="name")

    result = ser.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True])),
        ("last", Series([True, True, False, False, False])),
        (False, Series([True, True, True, False, True])),
    ],
)
def test_duplicated_nan_none(keep, expected):
    ser = Series([np.nan, 3, 3, None, np.nan], dtype=object)

    result = ser.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)


def test_duplicated_categorical_bool_na(nulls_fixture):
    # GH#44351
    ser = Series(
        Categorical(
            [True, False, True, False, nulls_fixture],
            categories=[True, False],
            ordered=True,
        )
    )
    result = ser.duplicated()
    expected = Series([False, False, True, True, False])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "keep, vals",
    [
        ("last", [True, True, False]),
        ("first", [False, True, True]),
        (False, [True, True, True]),
    ],
)
def test_duplicated_mask(keep, vals):
    # GH#48150
    ser = Series([1, 2, NA, NA, NA], dtype="Int64")
    result = ser.duplicated(keep=keep)
    expected = Series([False, False] + vals)
    tm.assert_series_equal(result, expected)


def test_duplicated_mask_no_duplicated_na(keep):
    # GH#48150
    ser = Series([1, 2, NA], dtype="Int64")
    result = ser.duplicated(keep=keep)
    expected = Series([False, False, False])
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_searchsorted.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import is_scalar


class TestSeriesSearchSorted:
    def test_searchsorted(self):
        ser = Series([1, 2, 3])

        result = ser.searchsorted(1, side="left")
        assert is_scalar(result)
        assert result == 0

        result = ser.searchsorted(1, side="right")
        assert is_scalar(result)
        assert result == 1

    def test_searchsorted_numeric_dtypes_scalar(self):
        ser = Series([1, 2, 90, 1000, 3e9])
        res = ser.searchsorted(30)
        assert is_scalar(res)
        assert res == 2

        res = ser.searchsorted([30])
        exp = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_numeric_dtypes_vector(self):
        ser = Series([1, 2, 90, 1000, 3e9])
        res = ser.searchsorted([91, 2e6])
        exp = np.array([3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_datetime64_scalar(self):
        ser = Series(date_range("20120101", periods=10, freq="2D"))
        val = Timestamp("20120102")
        res = ser.searchsorted(val)
        assert is_scalar(res)
        assert res == 1

    def test_searchsorted_datetime64_scalar_mixed_timezones(self):
        # GH 30086
        ser = Series(date_range("20120101", periods=10, freq="2D", tz="UTC"))
        val = Timestamp("20120102", tz="America/New_York")
        res = ser.searchsorted(val)
        assert is_scalar(res)
        assert res == 1

    def test_searchsorted_datetime64_list(self):
        ser = Series(date_range("20120101", periods=10, freq="2D"))
        vals = [Timestamp("20120102"), Timestamp("20120104")]
        res = ser.searchsorted(vals)
        exp = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_sorter(self):
        # GH8490
        ser = Series([3, 1, 2])
        res = ser.searchsorted([0, 3], sorter=np.argsort(ser))
        exp = np.array([0, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_dataframe_fail(self):
        # GH#49620
        ser = Series([1, 2, 3, 4, 5])
        vals = pd.DataFrame([[1, 2], [3, 4]])
        msg = "Value must be 1-D array-like or scalar, DataFrame is not supported"
        with pytest.raises(ValueError, match=msg):
            ser.searchsorted(vals)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_interactions.py ===
"""
We have a few different kind of Matrices
Matrix, ImmutableMatrix, MatrixExpr

Here we test the extent to which they cooperate
"""

from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
        ImmutableMatrix)
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.matrixbase import classof
from sympy.testing.pytest import raises

SM = MatrixSymbol('X', 3, 3)
SV = MatrixSymbol('v', 3, 1)
MM = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
IM = ImmutableMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
meye = eye(3)
imeye = ImmutableMatrix(eye(3))
ideye = Identity(3)
a, b, c = symbols('a,b,c')


def test_IM_MM():
    assert isinstance(MM + IM, ImmutableMatrix)
    assert isinstance(IM + MM, ImmutableMatrix)
    assert isinstance(2*IM + MM, ImmutableMatrix)
    assert MM.equals(IM)


def test_ME_MM():
    assert isinstance(Identity(3) + MM, MatrixExpr)
    assert isinstance(SM + MM, MatAdd)
    assert isinstance(MM + SM, MatAdd)
    assert (Identity(3) + MM)[1, 1] == 6


def test_equality():
    a, b, c = Identity(3), eye(3), ImmutableMatrix(eye(3))
    for x in [a, b, c]:
        for y in [a, b, c]:
            assert x.equals(y)


def test_matrix_symbol_MM():
    X = MatrixSymbol('X', 3, 3)
    Y = eye(3) + X
    assert Y[1, 1] == 1 + X[1, 1]


def test_matrix_symbol_vector_matrix_multiplication():
    A = MM * SV
    B = IM * SV
    assert A == B
    C = (SV.T * MM.T).T
    assert B == C
    D = (SV.T * IM.T).T
    assert C == D


def test_indexing_interactions():
    assert (a * IM)[1, 1] == 5*a
    assert (SM + IM)[1, 1] == SM[1, 1] + IM[1, 1]
    assert (SM * IM)[1, 1] == SM[1, 0]*IM[0, 1] + SM[1, 1]*IM[1, 1] + \
        SM[1, 2]*IM[2, 1]


def test_classof():
    A = Matrix(3, 3, range(9))
    B = ImmutableMatrix(3, 3, range(9))
    C = MatrixSymbol('C', 3, 3)
    assert classof(A, A) == Matrix
    assert classof(B, B) == ImmutableMatrix
    assert classof(A, B) == ImmutableMatrix
    assert classof(B, A) == ImmutableMatrix
    raises(TypeError, lambda: classof(A, C))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_continued_fraction.py ===
import itertools
from sympy.core import GoldenRatio as phi
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.continued_fraction import \
    (continued_fraction_periodic as cf_p,
     continued_fraction_iterator as cf_i,
     continued_fraction_convergents as cf_c,
     continued_fraction_reduce as cf_r,
     continued_fraction as cf)
from sympy.testing.pytest import raises


def test_continued_fraction():
    assert cf_p(1, 1, 10, 0) == cf_p(1, 1, 0, 1)
    assert cf_p(1, -1, 10, 1) == cf_p(-1, 1, 10, -1)
    t = sqrt(2)
    assert cf((1 + t)*(1 - t)) == cf(-1)
    for n in [0, 2, Rational(2, 3), sqrt(2), 3*sqrt(2), 1 + 2*sqrt(3)/5,
            (2 - 3*sqrt(5))/7, 1 + sqrt(2), (-5 + sqrt(17))/4]:
        assert (cf_r(cf(n)) - n).expand() == 0
        assert (cf_r(cf(-n)) + n).expand() == 0
    raises(ValueError, lambda: cf(sqrt(2 + sqrt(3))))
    raises(ValueError, lambda: cf(sqrt(2) + sqrt(3)))
    raises(ValueError, lambda: cf(pi))
    raises(ValueError, lambda: cf(.1))

    raises(ValueError, lambda: cf_p(1, 0, 0))
    raises(ValueError, lambda: cf_p(1, 1, -1))
    assert cf_p(4, 3, 0) == [1, 3]
    assert cf_p(0, 3, 5) == [0, 1, [2, 1, 12, 1, 2, 2]]
    assert cf_p(1, 1, 0) == [1]
    assert cf_p(3, 4, 0) == [0, 1, 3]
    assert cf_p(4, 5, 0) == [0, 1, 4]
    assert cf_p(5, 6, 0) == [0, 1, 5]
    assert cf_p(11, 13, 0) == [0, 1, 5, 2]
    assert cf_p(16, 19, 0) == [0, 1, 5, 3]
    assert cf_p(27, 32, 0) == [0, 1, 5, 2, 2]
    assert cf_p(1, 2, 5) == [[1]]
    assert cf_p(0, 1, 2) == [1, [2]]
    assert cf_p(6, 7, 49) == [1, 1, 6]
    assert cf_p(3796, 1387, 0) == [2, 1, 2, 1, 4]
    assert cf_p(3245, 10000) == [0, 3, 12, 4, 13]
    assert cf_p(1932, 2568) == [0, 1, 3, 26, 2]
    assert cf_p(6589, 2569) == [2, 1, 1, 3, 2, 1, 3, 1, 23]

    def take(iterator, n=7):
        return list(itertools.islice(iterator, n))

    assert take(cf_i(phi)) == [1, 1, 1, 1, 1, 1, 1]
    assert take(cf_i(pi)) == [3, 7, 15, 1, 292, 1, 1]

    assert list(cf_i(Rational(17, 12))) == [1, 2, 2, 2]
    assert list(cf_i(Rational(-17, 12))) == [-2, 1, 1, 2, 2]

    assert list(cf_c([1, 6, 1, 8])) == [S.One, Rational(7, 6), Rational(8, 7), Rational(71, 62)]
    assert list(cf_c([2])) == [S(2)]
    assert list(cf_c([1, 1, 1, 1, 1, 1, 1])) == [S.One, S(2), Rational(3, 2), Rational(5, 3),
                                                 Rational(8, 5), Rational(13, 8), Rational(21, 13)]
    assert list(cf_c([1, 6, Rational(-1, 2), 4])) == [S.One, Rational(7, 6), Rational(5, 4), Rational(3, 2)]
    assert take(cf_c([[1]])) == [S.One, S(2), Rational(3, 2), Rational(5, 3), Rational(8, 5),
                                 Rational(13, 8), Rational(21, 13)]
    assert take(cf_c([1, [1, 2]])) == [S.One, S(2), Rational(5, 3), Rational(7, 4), Rational(19, 11),
                                    Rational(26, 15), Rational(71, 41)]

    cf_iter_e = (2 if i == 1 else i // 3 * 2 if i % 3 == 0 else 1 for i in itertools.count(1))
    assert take(cf_c(cf_iter_e)) == [S(2), S(3), Rational(8, 3), Rational(11, 4), Rational(19, 7),
                                     Rational(87, 32), Rational(106, 39)]

    assert cf_r([1, 6, 1, 8]) == Rational(71, 62)
    assert cf_r([3]) == S(3)
    assert cf_r([-1, 5, 1, 4]) == Rational(-24, 29)
    assert (cf_r([0, 1, 1, 7, [24, 8]]) - (sqrt(3) + 2)/7).expand() == 0
    assert cf_r([1, 5, 9]) == Rational(55, 46)
    assert (cf_r([[1]]) - (sqrt(5) + 1)/2).expand() == 0
    assert cf_r([-3, 1, 1, [2]]) == -1 - sqrt(2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\tests\test_experimental_lambdify.py ===
from sympy.core.symbol import symbols, Symbol
from sympy.functions import Max
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.plotting.intervalmath.interval_arithmetic import \
    interval, intervalMembership


# Tests for exception handling in experimental_lambdify
def test_experimental_lambify():
    x = Symbol('x')
    f = experimental_lambdify([x], Max(x, 5))
    # XXX should f be tested? If f(2) is attempted, an
    # error is raised because a complex produced during wrapping of the arg
    # is being compared with an int.
    assert Max(2, 5) == 5
    assert Max(5, 7) == 7

    x = Symbol('x-3')
    f = experimental_lambdify([x], x + 1)
    assert f(1) == 2


def test_composite_boolean_region():
    x, y = symbols('x y')

    r1 = (x - 1)**2 + y**2 < 2
    r2 = (x + 1)**2 + y**2 < 2

    f = experimental_lambdify((x, y), r1 & r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), r1 | r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), r1 & ~r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), ~r1 & r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), ~r1 & ~r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(True, True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_codeprinter.py ===
from sympy.printing.codeprinter import CodePrinter, PrintMethodNotImplementedError
from sympy.core import symbols
from sympy.core.symbol import Dummy
from sympy.testing.pytest import raises
from sympy import cos
from sympy.utilities.lambdify import lambdify
from math import cos as math_cos
from sympy.printing.lambdarepr import LambdaPrinter


def setup_test_printer(**kwargs):
    p = CodePrinter(settings=kwargs)
    p._not_supported = set()
    p._number_symbols = set()
    return p


def test_print_Dummy():
    d = Dummy('d')
    p = setup_test_printer()
    assert p._print_Dummy(d) == "d_%i" % d.dummy_index

def test_print_Symbol():

    x, y = symbols('x, if')

    p = setup_test_printer()
    assert p._print(x) == 'x'
    assert p._print(y) == 'if'

    p.reserved_words.update(['if'])
    assert p._print(y) == 'if_'

    p = setup_test_printer(error_on_reserved=True)
    p.reserved_words.update(['if'])
    with raises(ValueError):
        p._print(y)

    p = setup_test_printer(reserved_word_suffix='_He_Man')
    p.reserved_words.update(['if'])
    assert p._print(y) == 'if_He_Man'


def test_lambdify_LaTeX_symbols_issue_23374():
    # Create symbols with Latex style names
    x1, x2 = symbols("x_{1} x_2")

    # Lambdify the function
    f1 = lambdify([x1, x2], cos(x1 ** 2 + x2 ** 2))

    # Test that the function works correctly (numerically)
    assert f1(1, 2) == math_cos(1 ** 2 + 2 ** 2)

    # Explicitly generate a custom printer to verify the naming convention
    p = LambdaPrinter()
    expr_str = p.doprint(cos(x1 ** 2 + x2 ** 2))
    assert 'x_1' in expr_str
    assert 'x_2' in expr_str


def test_issue_15791():
    class CrashingCodePrinter(CodePrinter):
        def emptyPrinter(self, obj):
            raise NotImplementedError

    from sympy.matrices import (
        MutableSparseMatrix,
        ImmutableSparseMatrix,
    )

    c = CrashingCodePrinter()

    # these should not silently succeed
    with raises(PrintMethodNotImplementedError):
        c.doprint(ImmutableSparseMatrix(2, 2, {}))
    with raises(PrintMethodNotImplementedError):
        c.doprint(MutableSparseMatrix(2, 2, {}))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\tests\ansi_test.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
import sys
from unittest import TestCase, main

from ..ansi import Back, Fore, Style
from ..ansitowin32 import AnsiToWin32

stdout_orig = sys.stdout
stderr_orig = sys.stderr


class AnsiTest(TestCase):

    def setUp(self):
        # sanity check: stdout should be a file or StringIO object.
        # It will only be AnsiToWin32 if init() has previously wrapped it
        self.assertNotEqual(type(sys.stdout), AnsiToWin32)
        self.assertNotEqual(type(sys.stderr), AnsiToWin32)

    def tearDown(self):
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig


    def testForeAttributes(self):
        self.assertEqual(Fore.BLACK, '\033[30m')
        self.assertEqual(Fore.RED, '\033[31m')
        self.assertEqual(Fore.GREEN, '\033[32m')
        self.assertEqual(Fore.YELLOW, '\033[33m')
        self.assertEqual(Fore.BLUE, '\033[34m')
        self.assertEqual(Fore.MAGENTA, '\033[35m')
        self.assertEqual(Fore.CYAN, '\033[36m')
        self.assertEqual(Fore.WHITE, '\033[37m')
        self.assertEqual(Fore.RESET, '\033[39m')

        # Check the light, extended versions.
        self.assertEqual(Fore.LIGHTBLACK_EX, '\033[90m')
        self.assertEqual(Fore.LIGHTRED_EX, '\033[91m')
        self.assertEqual(Fore.LIGHTGREEN_EX, '\033[92m')
        self.assertEqual(Fore.LIGHTYELLOW_EX, '\033[93m')
        self.assertEqual(Fore.LIGHTBLUE_EX, '\033[94m')
        self.assertEqual(Fore.LIGHTMAGENTA_EX, '\033[95m')
        self.assertEqual(Fore.LIGHTCYAN_EX, '\033[96m')
        self.assertEqual(Fore.LIGHTWHITE_EX, '\033[97m')


    def testBackAttributes(self):
        self.assertEqual(Back.BLACK, '\033[40m')
        self.assertEqual(Back.RED, '\033[41m')
        self.assertEqual(Back.GREEN, '\033[42m')
        self.assertEqual(Back.YELLOW, '\033[43m')
        self.assertEqual(Back.BLUE, '\033[44m')
        self.assertEqual(Back.MAGENTA, '\033[45m')
        self.assertEqual(Back.CYAN, '\033[46m')
        self.assertEqual(Back.WHITE, '\033[47m')
        self.assertEqual(Back.RESET, '\033[49m')

        # Check the light, extended versions.
        self.assertEqual(Back.LIGHTBLACK_EX, '\033[100m')
        self.assertEqual(Back.LIGHTRED_EX, '\033[101m')
        self.assertEqual(Back.LIGHTGREEN_EX, '\033[102m')
        self.assertEqual(Back.LIGHTYELLOW_EX, '\033[103m')
        self.assertEqual(Back.LIGHTBLUE_EX, '\033[104m')
        self.assertEqual(Back.LIGHTMAGENTA_EX, '\033[105m')
        self.assertEqual(Back.LIGHTCYAN_EX, '\033[106m')
        self.assertEqual(Back.LIGHTWHITE_EX, '\033[107m')


    def testStyleAttributes(self):
        self.assertEqual(Style.DIM, '\033[2m')
        self.assertEqual(Style.NORMAL, '\033[22m')
        self.assertEqual(Style.BRIGHT, '\033[1m')


if __name__ == '__main__':
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\multiarray.py ===
import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64] = np.array([1.0])
AR_i4 = np.array([1], dtype=np.int32)
AR_u1 = np.array([1], dtype=np.uint8)

AR_LIKE_f = [1.5]
AR_LIKE_i = [1]

b_f8 = np.broadcast(AR_f8)
b_i4_f8_f8 = np.broadcast(AR_i4, AR_f8, AR_f8)

next(b_f8)
b_f8.reset()
b_f8.index
b_f8.iters
b_f8.nd
b_f8.ndim
b_f8.numiter
b_f8.shape
b_f8.size

next(b_i4_f8_f8)
b_i4_f8_f8.reset()
b_i4_f8_f8.ndim
b_i4_f8_f8.index
b_i4_f8_f8.iters
b_i4_f8_f8.nd
b_i4_f8_f8.numiter
b_i4_f8_f8.shape
b_i4_f8_f8.size

np.inner(AR_f8, AR_i4)

np.where([True, True, False])
np.where([True, True, False], 1, 0)

np.lexsort([0, 1, 2])

np.can_cast(np.dtype("i8"), int)
np.can_cast(AR_f8, "f8")
np.can_cast(AR_f8, np.complex128, casting="unsafe")

np.min_scalar_type([1])
np.min_scalar_type(AR_f8)

np.result_type(int, AR_i4)
np.result_type(AR_f8, AR_u1)
np.result_type(AR_f8, np.complex128)

np.dot(AR_LIKE_f, AR_i4)
np.dot(AR_u1, 1)
np.dot(1.5j, 1)
np.dot(AR_u1, 1, out=AR_f8)

np.vdot(AR_LIKE_f, AR_i4)
np.vdot(AR_u1, 1)
np.vdot(1.5j, 1)

np.bincount(AR_i4)

np.copyto(AR_f8, [1.6])

np.putmask(AR_f8, [True], 1.5)

np.packbits(AR_i4)
np.packbits(AR_u1)

np.unpackbits(AR_u1)

np.shares_memory(1, 2)
np.shares_memory(AR_f8, AR_f8, max_work=1)

np.may_share_memory(1, 2)
np.may_share_memory(AR_f8, AR_f8, max_work=1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_iter.py ===
import dateutil.tz
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    date_range,
    to_datetime,
)
from pandas.core.arrays import datetimes


class TestDatetimeIndexIteration:
    @pytest.mark.parametrize(
        "tz", [None, "UTC", "US/Central", dateutil.tz.tzoffset(None, -28800)]
    )
    def test_iteration_preserves_nanoseconds(self, tz):
        # GH#19603
        index = DatetimeIndex(
            ["2018-02-08 15:00:00.168456358", "2018-02-08 15:00:00.168456359"], tz=tz
        )
        for i, ts in enumerate(index):
            assert ts == index[i]  # pylint: disable=unnecessary-list-index-lookup

    def test_iter_readonly(self):
        # GH#28055 ints_to_pydatetime with readonly array
        arr = np.array([np.datetime64("2012-02-15T12:00:00.000000000")])
        arr.setflags(write=False)
        dti = to_datetime(arr)
        list(dti)

    def test_iteration_preserves_tz(self):
        # see GH#8890
        index = date_range("2012-01-01", periods=3, freq="h", tz="US/Eastern")

        for i, ts in enumerate(index):
            result = ts
            expected = index[i]  # pylint: disable=unnecessary-list-index-lookup
            assert result == expected

    def test_iteration_preserves_tz2(self):
        index = date_range(
            "2012-01-01", periods=3, freq="h", tz=dateutil.tz.tzoffset(None, -28800)
        )

        for i, ts in enumerate(index):
            result = ts
            expected = index[i]  # pylint: disable=unnecessary-list-index-lookup
            assert result._repr_base == expected._repr_base
            assert result == expected

    def test_iteration_preserves_tz3(self):
        # GH#9100
        index = DatetimeIndex(
            ["2014-12-01 03:32:39.987000-08:00", "2014-12-01 04:12:34.987000-08:00"]
        )
        for i, ts in enumerate(index):
            result = ts
            expected = index[i]  # pylint: disable=unnecessary-list-index-lookup
            assert result._repr_base == expected._repr_base
            assert result == expected

    @pytest.mark.parametrize("offset", [-5, -1, 0, 1])
    def test_iteration_over_chunksize(self, offset, monkeypatch):
        # GH#21012
        chunksize = 5
        index = date_range(
            "2000-01-01 00:00:00", periods=chunksize - offset, freq="min"
        )
        num = 0
        with monkeypatch.context() as m:
            m.setattr(datetimes, "_ITER_CHUNKSIZE", chunksize)
            for stamp in index:
                assert index[num] == stamp
                num += 1
        assert num == len(index)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\methods\test_shift.py ===
import pytest

from pandas.errors import NullFrequencyError

import pandas as pd
from pandas import TimedeltaIndex
import pandas._testing as tm


class TestTimedeltaIndexShift:
    # -------------------------------------------------------------
    # TimedeltaIndex.shift is used by __add__/__sub__

    def test_tdi_shift_empty(self):
        # GH#9903
        idx = TimedeltaIndex([], name="xxx")
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        tm.assert_index_equal(idx.shift(3, freq="h"), idx)

    def test_tdi_shift_hours(self):
        # GH#9903
        idx = TimedeltaIndex(["5 hours", "6 hours", "9 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        exp = TimedeltaIndex(["8 hours", "9 hours", "12 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(3, freq="h"), exp)
        exp = TimedeltaIndex(["2 hours", "3 hours", "6 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(-3, freq="h"), exp)

    def test_tdi_shift_minutes(self):
        # GH#9903
        idx = TimedeltaIndex(["5 hours", "6 hours", "9 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(0, freq="min"), idx)
        exp = TimedeltaIndex(["05:03:00", "06:03:00", "9:03:00"], name="xxx")
        tm.assert_index_equal(idx.shift(3, freq="min"), exp)
        exp = TimedeltaIndex(["04:57:00", "05:57:00", "8:57:00"], name="xxx")
        tm.assert_index_equal(idx.shift(-3, freq="min"), exp)

    def test_tdi_shift_int(self):
        # GH#8083
        tdi = pd.to_timedelta(range(5), unit="d")
        trange = tdi._with_freq("infer") + pd.offsets.Hour(1)
        result = trange.shift(1)
        expected = TimedeltaIndex(
            [
                "1 days 01:00:00",
                "2 days 01:00:00",
                "3 days 01:00:00",
                "4 days 01:00:00",
                "5 days 01:00:00",
            ],
            freq="D",
        )
        tm.assert_index_equal(result, expected)

    def test_tdi_shift_nonstandard_freq(self):
        # GH#8083
        tdi = pd.to_timedelta(range(5), unit="d")
        trange = tdi._with_freq("infer") + pd.offsets.Hour(1)
        result = trange.shift(3, freq="2D 1s")
        expected = TimedeltaIndex(
            [
                "6 days 01:00:03",
                "7 days 01:00:03",
                "8 days 01:00:03",
                "9 days 01:00:03",
                "10 days 01:00:03",
            ],
            freq="D",
        )
        tm.assert_index_equal(result, expected)

    def test_shift_no_freq(self):
        # GH#19147
        tdi = TimedeltaIndex(["1 days 01:00:00", "2 days 01:00:00"], freq=None)
        with pytest.raises(NullFrequencyError, match="Cannot shift with no freq"):
            tdi.shift(2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\excel\test_xlrd.py ===
import io

import numpy as np
import pytest

from pandas.compat import is_platform_windows

import pandas as pd
import pandas._testing as tm

from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format

xlrd = pytest.importorskip("xlrd")

if is_platform_windows():
    pytestmark = pytest.mark.single_cpu


@pytest.fixture(params=[".xls"])
def read_ext_xlrd(request):
    """
    Valid extensions for reading Excel files with xlrd.

    Similar to read_ext, but excludes .ods, .xlsb, and for xlrd>2 .xlsx, .xlsm
    """
    return request.param


def test_read_xlrd_book(read_ext_xlrd, datapath):
    engine = "xlrd"
    sheet_name = "Sheet1"
    pth = datapath("io", "data", "excel", "test1.xls")
    with xlrd.open_workbook(pth) as book:
        with ExcelFile(book, engine=engine) as xl:
            result = pd.read_excel(xl, sheet_name=sheet_name, index_col=0)

        expected = pd.read_excel(
            book, sheet_name=sheet_name, engine=engine, index_col=0
        )
    tm.assert_frame_equal(result, expected)


def test_read_xlsx_fails(datapath):
    # GH 29375
    from xlrd.biffh import XLRDError

    path = datapath("io", "data", "excel", "test1.xlsx")
    with pytest.raises(XLRDError, match="Excel xlsx file; not supported"):
        pd.read_excel(path, engine="xlrd")


def test_nan_in_xls(datapath):
    # GH 54564
    path = datapath("io", "data", "excel", "test6.xls")

    expected = pd.DataFrame({0: np.r_[0, 2].astype("int64"), 1: np.r_[1, np.nan]})

    result = pd.read_excel(path, header=None)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "file_header",
    [
        b"\x09\x00\x04\x00\x07\x00\x10\x00",
        b"\x09\x02\x06\x00\x00\x00\x10\x00",
        b"\x09\x04\x06\x00\x00\x00\x10\x00",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    ],
)
def test_read_old_xls_files(file_header):
    # GH 41226
    f = io.BytesIO(file_header)
    assert inspect_excel_format(f) == "xls"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_unique.py ===
import numpy as np

from pandas import (
    Categorical,
    IntervalIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestUnique:
    def test_unique_uint64(self):
        ser = Series([1, 2, 2**63, 2**63], dtype=np.uint64)
        res = ser.unique()
        exp = np.array([1, 2, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(res, exp)

    def test_unique_data_ownership(self):
        # it works! GH#1807
        Series(Series(["a", "c", "b"]).unique()).sort_values()

    def test_unique(self):
        # GH#714 also, dtype=float
        ser = Series([1.2345] * 100)
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

        # explicit f4 dtype
        ser = Series([1.2345] * 100, dtype="f4")
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

    def test_unique_nan_object_dtype(self):
        # NAs in object arrays GH#714
        ser = Series(["foo"] * 100, dtype="O")
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

    def test_unique_none(self):
        # decision about None
        ser = Series([1, 2, 3, None, None, None], dtype=object)
        result = ser.unique()
        expected = np.array([1, 2, 3, None], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_unique_categorical(self):
        # GH#18051
        cat = Categorical([])
        ser = Series(cat)
        result = ser.unique()
        tm.assert_categorical_equal(result, cat)

        cat = Categorical([np.nan])
        ser = Series(cat)
        result = ser.unique()
        tm.assert_categorical_equal(result, cat)

    def test_tz_unique(self):
        # GH 46128
        dti1 = date_range("2016-01-01", periods=3)
        ii1 = IntervalIndex.from_breaks(dti1)
        ser1 = Series(ii1)
        uni1 = ser1.unique()
        tm.assert_interval_array_equal(ser1.array, uni1)

        dti2 = date_range("2016-01-01", periods=3, tz="US/Eastern")
        ii2 = IntervalIndex.from_breaks(dti2)
        ser2 = Series(ii2)
        uni2 = ser2.unique()
        tm.assert_interval_array_equal(ser2.array, uni2)

        assert uni1.dtype != uni2.dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_singleton.py ===
from sympy.core.basic import Basic
from sympy.core.numbers import Rational
from sympy.core.singleton import S, Singleton

def test_Singleton():

    class MySingleton(Basic, metaclass=Singleton):
        pass

    MySingleton() # force instantiation
    assert MySingleton() is not Basic()
    assert MySingleton() is MySingleton()
    assert S.MySingleton is MySingleton()

    class MySingleton_sub(MySingleton):
        pass

    MySingleton_sub()
    assert MySingleton_sub() is not MySingleton()
    assert MySingleton_sub() is MySingleton_sub()

def test_singleton_redefinition():
    class TestSingleton(Basic, metaclass=Singleton):
        pass

    assert TestSingleton() is S.TestSingleton

    class TestSingleton(Basic, metaclass=Singleton):
        pass

    assert TestSingleton() is S.TestSingleton

def test_names_in_namespace():
    # Every singleton name should be accessible from the 'from sympy import *'
    # namespace in addition to the S object. However, it does not need to be
    # by the same name (e.g., oo instead of S.Infinity).

    # As a general rule, things should only be added to the singleton registry
    # if they are used often enough that code can benefit either from the
    # performance benefit of being able to use 'is' (this only matters in very
    # tight loops), or from the memory savings of having exactly one instance
    # (this matters for the numbers singletons, but very little else). The
    # singleton registry is already a bit overpopulated, and things cannot be
    # removed from it without breaking backwards compatibility. So if you got
    # here by adding something new to the singletons, ask yourself if it
    # really needs to be singletonized. Note that SymPy classes compare to one
    # another just fine, so Class() == Class() will give True even if each
    # Class() returns a new instance. Having unique instances is only
    # necessary for the above noted performance gains. It should not be needed
    # for any behavioral purposes.

    # If you determine that something really should be a singleton, it must be
    # accessible to sympify() without using 'S' (hence this test). Also, its
    # str printer should print a form that does not use S. This is because
    # sympify() disables attribute lookups by default for safety purposes.
    d = {}
    exec('from sympy import *', d)

    for name in dir(S) + list(S._classes_to_install):
        if name.startswith('_'):
            continue
        if name == 'register':
            continue
        if isinstance(getattr(S, name), Rational):
            continue
        if getattr(S, name).__module__.startswith('sympy.physics'):
            continue
        if name in ['MySingleton', 'MySingleton_sub', 'TestSingleton']:
            # From the tests above
            continue
        if name == 'NegativeInfinity':
            # Accessible by -oo
            continue

        # Use is here to ensure it is the exact same object
        assert any(getattr(S, name) is i for i in d.values()), name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_semicolon_split.py ===
import platform

import pytest

from numpy.testing import IS_64BIT

from . import util


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Prone to error when run with numpy/f2py/tests on mac os, "
    "but not when run in isolation",
)
@pytest.mark.skipif(
    not IS_64BIT, reason="32-bit builds are buggy"
)
class TestMultiline(util.F2PyTest):
    suffix = ".pyf"
    module_name = "multiline"
    code = f"""
python module {module_name}
    usercode '''
void foo(int* x) {{
    char dummy = ';';
    *x = 42;
}}
'''
    interface
        subroutine foo(x)
            intent(c) foo
            integer intent(out) :: x
        end subroutine foo
    end interface
end python module {module_name}
    """

    def test_multiline(self):
        assert self.module.foo() == 42


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Prone to error when run with numpy/f2py/tests on mac os, "
    "but not when run in isolation",
)
@pytest.mark.skipif(
    not IS_64BIT, reason="32-bit builds are buggy"
)
@pytest.mark.slow
class TestCallstatement(util.F2PyTest):
    suffix = ".pyf"
    module_name = "callstatement"
    code = f"""
python module {module_name}
    usercode '''
void foo(int* x) {{
}}
'''
    interface
        subroutine foo(x)
            intent(c) foo
            integer intent(out) :: x
            callprotoargument int*
            callstatement {{ &
                ; &
                x = 42; &
            }}
        end subroutine foo
    end interface
end python module {module_name}
    """

    def test_callstatement(self):
        assert self.module.foo() == 42

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_arrayobject.py ===
import pytest

import numpy as np
from numpy.testing import assert_array_equal


def test_matrix_transpose_raises_error_for_1d():
    msg = "matrix transpose with ndim < 2 is undefined"
    arr = np.arange(48)
    with pytest.raises(ValueError, match=msg):
        arr.mT


def test_matrix_transpose_equals_transpose_2d():
    arr = np.arange(48).reshape((6, 8))
    assert_array_equal(arr.T, arr.mT)


ARRAY_SHAPES_TO_TEST = (
    (5, 2),
    (5, 2, 3),
    (5, 2, 3, 4),
)


@pytest.mark.parametrize("shape", ARRAY_SHAPES_TO_TEST)
def test_matrix_transpose_equals_swapaxes(shape):
    num_of_axes = len(shape)
    vec = np.arange(shape[-1])
    arr = np.broadcast_to(vec, shape)
    tgt = np.swapaxes(arr, num_of_axes - 2, num_of_axes - 1)
    mT = arr.mT
    assert_array_equal(tgt, mT)


class MyArr(np.ndarray):
    def __array_wrap__(self, arr, context=None, return_scalar=None):
        return super().__array_wrap__(arr, context, return_scalar)


class MyArrNoWrap(np.ndarray):
    pass


@pytest.mark.parametrize("subclass_self", [np.ndarray, MyArr, MyArrNoWrap])
@pytest.mark.parametrize("subclass_arr", [np.ndarray, MyArr, MyArrNoWrap])
def test_array_wrap(subclass_self, subclass_arr):
    # NumPy should allow `__array_wrap__` to be called on arrays, it's logic
    # is designed in a way that:
    #
    # * Subclasses never return scalars by default (to preserve their
    #   information).  They can choose to if they wish.
    # * NumPy returns scalars, if `return_scalar` is passed as True to allow
    #   manual calls to `arr.__array_wrap__` to do the right thing.
    # * The type of the input should be ignored (it should be a base-class
    #   array, but I am not sure this is guaranteed).

    arr = np.arange(3).view(subclass_self)

    arr0d = np.array(3, dtype=np.int8).view(subclass_arr)
    # With third argument True, ndarray allows "decay" to scalar.
    # (I don't think NumPy would pass `None`, but it seems clear to support)
    if subclass_self is np.ndarray:
        assert type(arr.__array_wrap__(arr0d, None, True)) is np.int8
    else:
        assert type(arr.__array_wrap__(arr0d, None, True)) is type(arr)

    # Otherwise, result should be viewed as the subclass
    assert type(arr.__array_wrap__(arr0d)) is type(arr)
    assert type(arr.__array_wrap__(arr0d, None, None)) is type(arr)
    assert type(arr.__array_wrap__(arr0d, None, False)) is type(arr)

    # Non 0-D array can't be converted to scalar, so we ignore that
    arr1d = np.array([3], dtype=np.int8).view(subclass_arr)
    assert type(arr.__array_wrap__(arr1d, None, True)) is type(arr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\test_ndarray_backed.py ===
"""
Tests for subclasses of NDArrayBackedExtensionArray
"""
import numpy as np

from pandas import (
    CategoricalIndex,
    date_range,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    NumpyExtensionArray,
    TimedeltaArray,
)


class TestEmpty:
    def test_empty_categorical(self):
        ci = CategoricalIndex(["a", "b", "c"], ordered=True)
        dtype = ci.dtype

        # case with int8 codes
        shape = (4,)
        result = Categorical._empty(shape, dtype=dtype)
        assert isinstance(result, Categorical)
        assert result.shape == shape
        assert result._ndarray.dtype == np.int8

        # case where repr would segfault if we didn't override base implementation
        result = Categorical._empty((4096,), dtype=dtype)
        assert isinstance(result, Categorical)
        assert result.shape == (4096,)
        assert result._ndarray.dtype == np.int8
        repr(result)

        # case with int16 codes
        ci = CategoricalIndex(list(range(512)) * 4, ordered=False)
        dtype = ci.dtype
        result = Categorical._empty(shape, dtype=dtype)
        assert isinstance(result, Categorical)
        assert result.shape == shape
        assert result._ndarray.dtype == np.int16

    def test_empty_dt64tz(self):
        dti = date_range("2016-01-01", periods=2, tz="Asia/Tokyo")
        dtype = dti.dtype

        shape = (0,)
        result = DatetimeArray._empty(shape, dtype=dtype)
        assert result.dtype == dtype
        assert isinstance(result, DatetimeArray)
        assert result.shape == shape

    def test_empty_dt64(self):
        shape = (3, 9)
        result = DatetimeArray._empty(shape, dtype="datetime64[ns]")
        assert isinstance(result, DatetimeArray)
        assert result.shape == shape

    def test_empty_td64(self):
        shape = (3, 9)
        result = TimedeltaArray._empty(shape, dtype="m8[ns]")
        assert isinstance(result, TimedeltaArray)
        assert result.shape == shape

    def test_empty_pandas_array(self):
        arr = NumpyExtensionArray(np.array([1, 2]))
        dtype = arr.dtype

        shape = (3, 9)
        result = NumpyExtensionArray._empty(shape, dtype=dtype)
        assert isinstance(result, NumpyExtensionArray)
        assert result.dtype == dtype
        assert result.shape == shape

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_na_indexing.py ===
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "values, dtype",
    [
        ([], "object"),
        ([1, 2, 3], "int64"),
        ([1.0, 2.0, 3.0], "float64"),
        (["a", "b", "c"], "object"),
        (["a", "b", "c"], "string"),
        ([1, 2, 3], "datetime64[ns]"),
        ([1, 2, 3], "datetime64[ns, CET]"),
        ([1, 2, 3], "timedelta64[ns]"),
        (["2000", "2001", "2002"], "Period[D]"),
        ([1, 0, 3], "Sparse"),
        ([pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(3, 4)], "interval"),
    ],
)
@pytest.mark.parametrize(
    "mask", [[True, False, False], [True, True, True], [False, False, False]]
)
@pytest.mark.parametrize("indexer_class", [list, pd.array, pd.Index, pd.Series])
@pytest.mark.parametrize("frame", [True, False])
def test_series_mask_boolean(values, dtype, mask, indexer_class, frame):
    # In case len(values) < 3
    index = ["a", "b", "c"][: len(values)]
    mask = mask[: len(values)]

    obj = pd.Series(values, dtype=dtype, index=index)
    if frame:
        if len(values) == 0:
            # Otherwise obj is an empty DataFrame with shape (0, 1)
            obj = pd.DataFrame(dtype=dtype, index=index)
        else:
            obj = obj.to_frame()

    if indexer_class is pd.array:
        mask = pd.array(mask, dtype="boolean")
    elif indexer_class is pd.Series:
        mask = pd.Series(mask, index=obj.index, dtype="boolean")
    else:
        mask = indexer_class(mask)

    expected = obj[mask]

    result = obj[mask]
    tm.assert_equal(result, expected)

    if indexer_class is pd.Series:
        msg = "iLocation based boolean indexing cannot use an indexable as a mask"
        with pytest.raises(ValueError, match=msg):
            result = obj.iloc[mask]
            tm.assert_equal(result, expected)
    else:
        result = obj.iloc[mask]
        tm.assert_equal(result, expected)

    result = obj.loc[mask]
    tm.assert_equal(result, expected)


def test_na_treated_as_false(frame_or_series, indexer_sli):
    # https://github.com/pandas-dev/pandas/issues/31503
    obj = frame_or_series([1, 2, 3])

    mask = pd.array([True, False, None], dtype="boolean")

    result = indexer_sli(obj)[mask]
    expected = indexer_sli(obj)[mask.fillna(False)]

    tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_compat.py ===
import pytest

import pandas as pd
import pandas._testing as tm

tables = pytest.importorskip("tables")


@pytest.fixture
def pytables_hdf5_file(tmp_path):
    """
    Use PyTables to create a simple HDF5 file.
    """
    table_schema = {
        "c0": tables.Time64Col(pos=0),
        "c1": tables.StringCol(5, pos=1),
        "c2": tables.Int64Col(pos=2),
    }

    t0 = 1_561_105_000.0

    testsamples = [
        {"c0": t0, "c1": "aaaaa", "c2": 1},
        {"c0": t0 + 1, "c1": "bbbbb", "c2": 2},
        {"c0": t0 + 2, "c1": "ccccc", "c2": 10**5},
        {"c0": t0 + 3, "c1": "ddddd", "c2": 4_294_967_295},
    ]

    objname = "pandas_test_timeseries"

    path = tmp_path / "written_with_pytables.h5"
    with tables.open_file(path, mode="w") as f:
        t = f.create_table("/", name=objname, description=table_schema)
        for sample in testsamples:
            for key, value in sample.items():
                t.row[key] = value
            t.row.append()

    yield path, objname, pd.DataFrame(testsamples)


class TestReadPyTablesHDF5:
    """
    A group of tests which covers reading HDF5 files written by plain PyTables
    (not written by pandas).

    Was introduced for regression-testing issue 11188.
    """

    def test_read_complete(self, pytables_hdf5_file):
        path, objname, df = pytables_hdf5_file
        result = pd.read_hdf(path, key=objname)
        expected = df
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_read_with_start(self, pytables_hdf5_file):
        path, objname, df = pytables_hdf5_file
        # This is a regression test for pandas-dev/pandas/issues/11188
        result = pd.read_hdf(path, key=objname, start=1)
        expected = df[1:].reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_read_with_stop(self, pytables_hdf5_file):
        path, objname, df = pytables_hdf5_file
        # This is a regression test for pandas-dev/pandas/issues/11188
        result = pd.read_hdf(path, key=objname, stop=1)
        expected = df[:1].reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_read_with_startstop(self, pytables_hdf5_file):
        path, objname, df = pytables_hdf5_file
        # This is a regression test for pandas-dev/pandas/issues/11188
        result = pd.read_hdf(path, key=objname, start=1, stop=2)
        expected = df[1:2].reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_index_type=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_between.py ===
import numpy as np
import pytest

from pandas import (
    Series,
    bdate_range,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestBetween:
    def test_between(self):
        series = Series(date_range("1/1/2000", periods=10))
        left, right = series[[2, 7]]

        result = series.between(left, right)
        expected = (series >= left) & (series <= right)
        tm.assert_series_equal(result, expected)

    def test_between_datetime_object_dtype(self):
        ser = Series(bdate_range("1/1/2000", periods=20), dtype=object)
        ser[::2] = np.nan

        result = ser[ser.between(ser[3], ser[17])]
        expected = ser[3:18].dropna()
        tm.assert_series_equal(result, expected)

        result = ser[ser.between(ser[3], ser[17], inclusive="neither")]
        expected = ser[5:16].dropna()
        tm.assert_series_equal(result, expected)

    def test_between_period_values(self):
        ser = Series(period_range("2000-01-01", periods=10, freq="D"))
        left, right = ser[[2, 7]]
        result = ser.between(left, right)
        expected = (ser >= left) & (ser <= right)
        tm.assert_series_equal(result, expected)

    def test_between_inclusive_string(self):
        # GH 40628
        series = Series(date_range("1/1/2000", periods=10))
        left, right = series[[2, 7]]

        result = series.between(left, right, inclusive="both")
        expected = (series >= left) & (series <= right)
        tm.assert_series_equal(result, expected)

        result = series.between(left, right, inclusive="left")
        expected = (series >= left) & (series < right)
        tm.assert_series_equal(result, expected)

        result = series.between(left, right, inclusive="right")
        expected = (series > left) & (series <= right)
        tm.assert_series_equal(result, expected)

        result = series.between(left, right, inclusive="neither")
        expected = (series > left) & (series < right)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("inclusive", ["yes", True, False])
    def test_between_error_args(self, inclusive):
        # GH 40628
        series = Series(date_range("1/1/2000", periods=10))
        left, right = series[[2, 7]]

        value_error_msg = (
            "Inclusive has to be either string of 'both',"
            "'left', 'right', or 'neither'."
        )

        with pytest.raises(ValueError, match=value_error_msg):
            series = Series(date_range("1/1/2000", periods=10))
            series.between(left, right, inclusive=inclusive)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_kind.py ===
"""Tests for sympy.physics.quantum.kind."""

from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.symbol import symbols

from sympy.physics.quantum.kind import (
    OperatorKind, KetKind, BraKind
)
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.tensorproduct import TensorProduct

k = Ket('k')
b = Bra('k')
A = Operator('A')
B = Operator('B')
x, y, z = symbols('x y z', integer=True)

def test_bra_ket():
    assert k.kind == KetKind
    assert b.kind == BraKind
    assert (b*k).kind == NumberKind # inner product
    assert (x*k).kind == KetKind
    assert (x*b).kind == BraKind


def test_operator_kind():
    assert A.kind == OperatorKind
    assert (A*B).kind == OperatorKind
    assert (x*A).kind == OperatorKind
    assert (x*A*B).kind == OperatorKind
    assert (x*k*b).kind == OperatorKind # outer product


def test_undefind_kind():
    # Because of limitations in the kind dispatcher API, we are currently
    # unable to have OperatorKind*KetKind -> KetKind (and similar for bras).
    assert (A*k).kind == UndefinedKind
    assert (b*A).kind == UndefinedKind
    assert (x*b*A*k).kind == UndefinedKind


def test_dagger_kind():
    assert Dagger(k).kind == BraKind
    assert Dagger(b).kind == KetKind
    assert Dagger(A).kind == OperatorKind


def test_commutator_kind():
    assert Commutator(A, B).kind == OperatorKind
    assert Commutator(A, x*B).kind == OperatorKind
    assert Commutator(x*A, B).kind == OperatorKind
    assert Commutator(x*A, x*B).kind == OperatorKind


def test_anticommutator_kind():
    assert AntiCommutator(A, B).kind == OperatorKind
    assert AntiCommutator(A, x*B).kind == OperatorKind
    assert AntiCommutator(x*A, B).kind == OperatorKind
    assert AntiCommutator(x*A, x*B).kind == OperatorKind


def test_tensorproduct_kind():
    assert TensorProduct(k,k).kind == KetKind
    assert TensorProduct(b,b).kind == BraKind
    assert TensorProduct(x*k,y*k).kind == KetKind
    assert TensorProduct(x*b,y*b).kind == BraKind
    assert TensorProduct(x*b*k, y*b*k).kind == NumberKind
    assert TensorProduct(x*k*b, y*k*b).kind == OperatorKind
    assert TensorProduct(A, B).kind == OperatorKind
    assert TensorProduct(A, x*B).kind == OperatorKind
    assert TensorProduct(x*A, B).kind == OperatorKind

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_transforms.py ===
"""Tests of transforms of quantum expressions for Mul and Pow."""

from sympy.core.symbol import symbols
from sympy.testing.pytest import raises

from sympy.physics.quantum.operator import (
    Operator, OuterProduct
)
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.tensorproduct import TensorProduct


k1 = Ket('k1')
k2 = Ket('k2')
k3 = Ket('k3')
b1 = Bra('b1')
b2 = Bra('b2')
b3 = Bra('b3')
A = Operator('A')
B = Operator('B')
C = Operator('C')
x, y, z = symbols('x y z')


def test_bra_ket():
    assert b1*k1 == InnerProduct(b1, k1)
    assert k1*b1 == OuterProduct(k1, b1)
    # Test priority of inner product
    assert OuterProduct(k1, b1)*k2 == InnerProduct(b1, k2)*k1
    assert b1*OuterProduct(k1, b2) == InnerProduct(b1, k1)*b2


def test_tensor_product():
    # We are attempting to be rigourous and raise TypeError when a user tries
    # to combine bras, kets, and operators in a manner that doesn't make sense.
    # In particular, we are not trying to interpret regular ``*`` multiplication
    # as a tensor product.
    with raises(TypeError):
        k1*k1
    with raises(TypeError):
        b1*b1
    with raises(TypeError):
        k1*TensorProduct(k2, k3)
    with raises(TypeError):
        b1*TensorProduct(b2, b3)
    with raises(TypeError):
        TensorProduct(k2, k3)*k1
    with raises(TypeError):
        TensorProduct(b2, b3)*b1

    assert TensorProduct(A, B, C)*TensorProduct(k1, k2, k3) == \
        TensorProduct(A*k1, B*k2, C*k3)
    assert TensorProduct(b1, b2, b3)*TensorProduct(A, B, C) == \
        TensorProduct(b1*A, b2*B, b3*C)
    assert TensorProduct(b1, b2, b3)*TensorProduct(k1, k2, k3) == \
        InnerProduct(b1, k1)*InnerProduct(b2, k2)*InnerProduct(b3, k3)
    assert TensorProduct(b1, b2, b3)*TensorProduct(A, B, C)*TensorProduct(k1, k2, k3) == \
        TensorProduct(b1*A*k1, b2*B*k2, b3*C*k3)


def test_outer_product():
    assert OuterProduct(k1, b1)*OuterProduct(k2, b2) == \
        InnerProduct(b1, k2)*OuterProduct(k1, b2)


def test_compound():
    e1 = b1*A*B*k1*b2*k2*b3
    assert e1 == InnerProduct(b2, k2)*b1*A*B*OuterProduct(k1, b3)

    e2 = TensorProduct(k1, k2)*TensorProduct(b1, b2)
    assert e2 == TensorProduct(
        OuterProduct(k1, b1),
        OuterProduct(k2, b2)
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_output.py ===
from sympy.core.singleton import S
from sympy.physics.vector import Vector, ReferenceFrame, Dyadic
from sympy.testing.pytest import raises

A = ReferenceFrame('A')


def test_output_type():
    A = ReferenceFrame('A')
    v = A.x + A.y
    d = v | v
    zerov = Vector(0)
    zerod = Dyadic(0)

    # dot products
    assert isinstance(d & d, Dyadic)
    assert isinstance(d & zerod, Dyadic)
    assert isinstance(zerod & d, Dyadic)
    assert isinstance(d & v, Vector)
    assert isinstance(v & d, Vector)
    assert isinstance(d & zerov, Vector)
    assert isinstance(zerov & d, Vector)
    raises(TypeError, lambda: d & S.Zero)
    raises(TypeError, lambda: S.Zero & d)
    raises(TypeError, lambda: d & 0)
    raises(TypeError, lambda: 0 & d)
    assert not isinstance(v & v, (Vector, Dyadic))
    assert not isinstance(v & zerov, (Vector, Dyadic))
    assert not isinstance(zerov & v, (Vector, Dyadic))
    raises(TypeError, lambda: v & S.Zero)
    raises(TypeError, lambda: S.Zero & v)
    raises(TypeError, lambda: v & 0)
    raises(TypeError, lambda: 0 & v)

    # cross products
    raises(TypeError, lambda: d ^ d)
    raises(TypeError, lambda: d ^ zerod)
    raises(TypeError, lambda: zerod ^ d)
    assert isinstance(d ^ v, Dyadic)
    assert isinstance(v ^ d, Dyadic)
    assert isinstance(d ^ zerov, Dyadic)
    assert isinstance(zerov ^ d, Dyadic)
    assert isinstance(zerov ^ d, Dyadic)
    raises(TypeError, lambda: d ^ S.Zero)
    raises(TypeError, lambda: S.Zero ^ d)
    raises(TypeError, lambda: d ^ 0)
    raises(TypeError, lambda: 0 ^ d)
    assert isinstance(v ^ v, Vector)
    assert isinstance(v ^ zerov, Vector)
    assert isinstance(zerov ^ v, Vector)
    raises(TypeError, lambda: v ^ S.Zero)
    raises(TypeError, lambda: S.Zero ^ v)
    raises(TypeError, lambda: v ^ 0)
    raises(TypeError, lambda: 0 ^ v)

    # outer products
    raises(TypeError, lambda: d | d)
    raises(TypeError, lambda: d | zerod)
    raises(TypeError, lambda: zerod | d)
    raises(TypeError, lambda: d | v)
    raises(TypeError, lambda: v | d)
    raises(TypeError, lambda: d | zerov)
    raises(TypeError, lambda: zerov | d)
    raises(TypeError, lambda: zerov | d)
    raises(TypeError, lambda: d | S.Zero)
    raises(TypeError, lambda: S.Zero | d)
    raises(TypeError, lambda: d | 0)
    raises(TypeError, lambda: 0 | d)
    assert isinstance(v | v, Dyadic)
    assert isinstance(v | zerov, Dyadic)
    assert isinstance(zerov | v, Dyadic)
    raises(TypeError, lambda: v | S.Zero)
    raises(TypeError, lambda: S.Zero | v)
    raises(TypeError, lambda: v | 0)
    raises(TypeError, lambda: 0 | v)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_combsimp.py ===
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial)
from sympy.functions.special.gamma_functions import gamma
from sympy.simplify.combsimp import combsimp
from sympy.abc import x


def test_combsimp():
    k, m, n = symbols('k m n', integer = True)

    assert combsimp(factorial(n)) == factorial(n)
    assert combsimp(binomial(n, k)) == binomial(n, k)

    assert combsimp(factorial(n)/factorial(n - 3)) == n*(-1 + n)*(-2 + n)
    assert combsimp(binomial(n + 1, k + 1)/binomial(n, k)) == (1 + n)/(1 + k)

    assert combsimp(binomial(3*n + 4, n + 1)/binomial(3*n + 1, n)) == \
        Rational(3, 2)*((3*n + 2)*(3*n + 4)/((n + 1)*(2*n + 3)))

    assert combsimp(factorial(n)**2/factorial(n - 3)) == \
        factorial(n)*n*(-1 + n)*(-2 + n)
    assert combsimp(factorial(n)*binomial(n + 1, k + 1)/binomial(n, k)) == \
        factorial(n + 1)/(1 + k)

    assert combsimp(gamma(n + 3)) == factorial(n + 2)

    assert combsimp(factorial(x)) == gamma(x + 1)

    # issue 9699
    assert combsimp((n + 1)*factorial(n)) == factorial(n + 1)
    assert combsimp(factorial(n)/n) == factorial(n-1)

    # issue 6658
    assert combsimp(binomial(n, n - k)) == binomial(n, k)

    # issue 6341, 7135
    assert combsimp(factorial(n)/(factorial(k)*factorial(n - k))) == \
        binomial(n, k)
    assert combsimp(factorial(k)*factorial(n - k)/factorial(n)) == \
        1/binomial(n, k)
    assert combsimp(factorial(2*n)/factorial(n)**2) == binomial(2*n, n)
    assert combsimp(factorial(2*n)*factorial(k)*factorial(n - k)/
        factorial(n)**3) == binomial(2*n, n)/binomial(n, k)

    assert combsimp(factorial(n*(1 + n) - n**2 - n)) == 1

    assert combsimp(6*FallingFactorial(-4, n)/factorial(n)) == \
        (-1)**n*(n + 1)*(n + 2)*(n + 3)
    assert combsimp(6*FallingFactorial(-4, n - 1)/factorial(n - 1)) == \
        (-1)**(n - 1)*n*(n + 1)*(n + 2)
    assert combsimp(6*FallingFactorial(-4, n - 3)/factorial(n - 3)) == \
        (-1)**(n - 3)*n*(n - 1)*(n - 2)
    assert combsimp(6*FallingFactorial(-4, -n - 1)/factorial(-n - 1)) == \
        -(-1)**(-n - 1)*n*(n - 1)*(n - 2)

    assert combsimp(6*RisingFactorial(4, n)/factorial(n)) == \
        (n + 1)*(n + 2)*(n + 3)
    assert combsimp(6*RisingFactorial(4, n - 1)/factorial(n - 1)) == \
        n*(n + 1)*(n + 2)
    assert combsimp(6*RisingFactorial(4, n - 3)/factorial(n - 3)) == \
        n*(n - 1)*(n - 2)
    assert combsimp(6*RisingFactorial(4, -n - 1)/factorial(-n - 1)) == \
        -n*(n - 1)*(n - 2)


def test_issue_6878():
    n = symbols('n', integer=True)
    assert combsimp(RisingFactorial(-10, n)) == 3628800*(-1)**n/factorial(10 - n)


def test_issue_14528():
    p = symbols("p", integer=True, positive=True)
    assert combsimp(binomial(1,p)) == 1/(factorial(p)*factorial(1-p))
    assert combsimp(factorial(2-p)) == factorial(2-p)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_reloading.py ===
import pickle
import subprocess
import sys
import textwrap
from importlib import reload

import pytest

import numpy.exceptions as ex
from numpy.testing import (
    IS_WASM,
    assert_,
    assert_equal,
    assert_raises,
    assert_warns,
)


def test_numpy_reloading():
    # gh-7844. Also check that relevant globals retain their identity.
    import numpy as np
    import numpy._globals

    _NoValue = np._NoValue
    VisibleDeprecationWarning = ex.VisibleDeprecationWarning
    ModuleDeprecationWarning = ex.ModuleDeprecationWarning

    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is ex.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is ex.VisibleDeprecationWarning)

    assert_raises(RuntimeError, reload, numpy._globals)
    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is ex.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is ex.VisibleDeprecationWarning)

def test_novalue():
    import numpy as np
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        assert_equal(repr(np._NoValue), '<no value>')
        assert_(pickle.loads(pickle.dumps(np._NoValue,
                                          protocol=proto)) is np._NoValue)


@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
def test_full_reimport():
    """At the time of writing this, it is *not* truly supported, but
    apparently enough users rely on it, for it to be an annoying change
    when it started failing previously.
    """
    # Test within a new process, to ensure that we do not mess with the
    # global state during the test run (could lead to cryptic test failures).
    # This is generally unsafe, especially, since we also reload the C-modules.
    code = textwrap.dedent(r"""
        import sys
        from pytest import warns
        import numpy as np

        for k in list(sys.modules.keys()):
            if "numpy" in k:
                del sys.modules[k]

        with warns(UserWarning):
            import numpy as np
        """)
    p = subprocess.run([sys.executable, '-c', code], capture_output=True)
    if p.returncode:
        raise AssertionError(
            f"Non-zero return code: {p.returncode!r}\n\n{p.stderr.decode()}"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\masked\test_function.py ===
import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray

arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]
arrays += [
    pd.array([0.141, -0.268, 5.895, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES
]


@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])
def data(request):
    """
    Fixture returning parametrized 'data' array with different integer and
    floating point types
    """
    return request.param


@pytest.fixture()
def numpy_dtype(data):
    """
    Fixture returning numpy dtype from 'data' input array.
    """
    # For integer dtype, the numpy conversion must be done to float
    if is_integer_dtype(data):
        numpy_dtype = float
    else:
        numpy_dtype = data.dtype.type
    return numpy_dtype


def test_round(data, numpy_dtype):
    # No arguments
    result = data.round()
    expected = pd.array(
        np.round(data.to_numpy(dtype=numpy_dtype, na_value=None)), dtype=data.dtype
    )
    tm.assert_extension_array_equal(result, expected)

    # Decimals argument
    result = data.round(decimals=2)
    expected = pd.array(
        np.round(data.to_numpy(dtype=numpy_dtype, na_value=None), decimals=2),
        dtype=data.dtype,
    )
    tm.assert_extension_array_equal(result, expected)


def test_tolist(data):
    result = data.tolist()
    expected = list(data)
    tm.assert_equal(result, expected)


def test_to_numpy():
    # GH#56991

    class MyStringArray(BaseMaskedArray):
        dtype = pd.StringDtype()
        _dtype_cls = pd.StringDtype
        _internal_fill_value = pd.NA

    arr = MyStringArray(
        values=np.array(["a", "b", "c"]), mask=np.array([False, True, False])
    )
    result = arr.to_numpy()
    expected = np.array(["a", pd.NA, "c"])
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_reorder_levels.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm


class TestReorderLevels:
    def test_reorder_levels(self, frame_or_series):
        index = MultiIndex(
            levels=[["bar"], ["one", "two", "three"], [0, 1]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
            names=["L0", "L1", "L2"],
        )
        df = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=index)
        obj = tm.get_obj(df, frame_or_series)

        # no change, position
        result = obj.reorder_levels([0, 1, 2])
        tm.assert_equal(obj, result)

        # no change, labels
        result = obj.reorder_levels(["L0", "L1", "L2"])
        tm.assert_equal(obj, result)

        # rotate, position
        result = obj.reorder_levels([1, 2, 0])
        e_idx = MultiIndex(
            levels=[["one", "two", "three"], [0, 1], ["bar"]],
            codes=[[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]],
            names=["L1", "L2", "L0"],
        )
        expected = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=e_idx)
        expected = tm.get_obj(expected, frame_or_series)
        tm.assert_equal(result, expected)

        result = obj.reorder_levels([0, 0, 0])
        e_idx = MultiIndex(
            levels=[["bar"], ["bar"], ["bar"]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            names=["L0", "L0", "L0"],
        )
        expected = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=e_idx)
        expected = tm.get_obj(expected, frame_or_series)
        tm.assert_equal(result, expected)

        result = obj.reorder_levels(["L0", "L0", "L0"])
        tm.assert_equal(result, expected)

    def test_reorder_levels_swaplevel_equivalence(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        ymd = multiindex_year_month_day_dataframe_random_data

        result = ymd.reorder_levels(["month", "day", "year"])
        expected = ymd.swaplevel(0, 1).swaplevel(1, 2)
        tm.assert_frame_equal(result, expected)

        result = ymd["A"].reorder_levels(["month", "day", "year"])
        expected = ymd["A"].swaplevel(0, 1).swaplevel(1, 2)
        tm.assert_series_equal(result, expected)

        result = ymd.T.reorder_levels(["month", "day", "year"], axis=1)
        expected = ymd.T.swaplevel(0, 1, axis=1).swaplevel(1, 2, axis=1)
        tm.assert_frame_equal(result, expected)

        with pytest.raises(TypeError, match="hierarchical axis"):
            ymd.reorder_levels([1, 2], axis=1)

        with pytest.raises(IndexError, match="Too many levels"):
            ymd.index.reorder_levels([1, 2, 3])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\tests\test_euler.py ===
from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.calculus.euler import euler_equations as euler


def test_euler_interface():
    x = Function('x')
    y = Symbol('y')
    t = Symbol('t')
    raises(TypeError, lambda: euler())
    raises(TypeError, lambda: euler(D(x(t), t)*y(t), [x(t), y]))
    raises(ValueError, lambda: euler(D(x(t), t)*x(y), [x(t), x(y)]))
    raises(TypeError, lambda: euler(D(x(t), t)**2, x(0)))
    raises(TypeError, lambda: euler(D(x(t), t)*y(t), [t]))
    assert euler(D(x(t), t)**2/2, {x(t)}) == [Eq(-D(x(t), t, t), 0)]
    assert euler(D(x(t), t)**2/2, x(t), {t}) == [Eq(-D(x(t), t, t), 0)]


def test_euler_pendulum():
    x = Function('x')
    t = Symbol('t')
    L = D(x(t), t)**2/2 + cos(x(t))
    assert euler(L, x(t), t) == [Eq(-sin(x(t)) - D(x(t), t, t), 0)]


def test_euler_henonheiles():
    x = Function('x')
    y = Function('y')
    t = Symbol('t')
    L = sum(D(z(t), t)**2/2 - z(t)**2/2 for z in [x, y])
    L += -x(t)**2*y(t) + y(t)**3/3
    assert euler(L, [x(t), y(t)], t) == [Eq(-2*x(t)*y(t) - x(t) -
                                            D(x(t), t, t), 0),
                                         Eq(-x(t)**2 + y(t)**2 -
                                            y(t) - D(y(t), t, t), 0)]


def test_euler_sineg():
    psi = Function('psi')
    t = Symbol('t')
    x = Symbol('x')
    L = D(psi(t, x), t)**2/2 - D(psi(t, x), x)**2/2 + cos(psi(t, x))
    assert euler(L, psi(t, x), [t, x]) == [Eq(-sin(psi(t, x)) -
                                              D(psi(t, x), t, t) +
                                              D(psi(t, x), x, x), 0)]


def test_euler_high_order():
    # an example from hep-th/0309038
    m = Symbol('m')
    k = Symbol('k')
    x = Function('x')
    y = Function('y')
    t = Symbol('t')
    L = (m*D(x(t), t)**2/2 + m*D(y(t), t)**2/2 -
         k*D(x(t), t)*D(y(t), t, t) + k*D(y(t), t)*D(x(t), t, t))
    assert euler(L, [x(t), y(t)]) == [Eq(2*k*D(y(t), t, t, t) -
                                         m*D(x(t), t, t), 0),
                                      Eq(-2*k*D(x(t), t, t, t) -
                                         m*D(y(t), t, t), 0)]

    w = Symbol('w')
    L = D(x(t, w), t, w)**2/2
    assert euler(L) == [Eq(D(x(t, w), t, t, w, w), 0)]

def test_issue_18653():
    x, y, z = symbols("x y z")
    f, g, h = symbols("f g h", cls=Function, args=(x, y))
    f, g, h = f(), g(), h()
    expr2 = f.diff(x)*h.diff(z)
    assert euler(expr2, (f,), (x, y)) == []

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_prufer.py ===
from sympy.combinatorics.prufer import Prufer
from sympy.testing.pytest import raises


def test_prufer():
    # number of nodes is optional
    assert Prufer([[0, 1], [0, 2], [0, 3], [0, 4]], 5).nodes == 5
    assert Prufer([[0, 1], [0, 2], [0, 3], [0, 4]]).nodes == 5

    a = Prufer([[0, 1], [0, 2], [0, 3], [0, 4]])
    assert a.rank == 0
    assert a.nodes == 5
    assert a.prufer_repr == [0, 0, 0]

    a = Prufer([[2, 4], [1, 4], [1, 3], [0, 5], [0, 4]])
    assert a.rank == 924
    assert a.nodes == 6
    assert a.tree_repr == [[2, 4], [1, 4], [1, 3], [0, 5], [0, 4]]
    assert a.prufer_repr == [4, 1, 4, 0]

    assert Prufer.edges([0, 1, 2, 3], [1, 4, 5], [1, 4, 6]) == \
        ([[0, 1], [1, 2], [1, 4], [2, 3], [4, 5], [4, 6]], 7)
    assert Prufer([0]*4).size == Prufer([6]*4).size == 1296

    # accept iterables but convert to list of lists
    tree = [(0, 1), (1, 5), (0, 3), (0, 2), (2, 6), (4, 7), (2, 4)]
    tree_lists = [list(t) for t in tree]
    assert Prufer(tree).tree_repr == tree_lists
    assert sorted(Prufer(set(tree)).tree_repr) == sorted(tree_lists)

    raises(ValueError, lambda: Prufer([[1, 2], [3, 4]]))  # 0 is missing
    raises(ValueError, lambda: Prufer([[2, 3], [3, 4]]))  # 0, 1 are missing
    assert Prufer(*Prufer.edges([1, 2], [3, 4])).prufer_repr == [1, 3]
    raises(ValueError, lambda: Prufer.edges(
        [1, 3], [3, 4]))  # a broken tree but edges doesn't care
    raises(ValueError, lambda: Prufer.edges([1, 2], [5, 6]))
    raises(ValueError, lambda: Prufer([[]]))

    a = Prufer([[0, 1], [0, 2], [0, 3]])
    b = a.next()
    assert b.tree_repr == [[0, 2], [0, 1], [1, 3]]
    assert b.rank == 1


def test_round_trip():
    def doit(t, b):
        e, n = Prufer.edges(*t)
        t = Prufer(e, n)
        a = sorted(t.tree_repr)
        b = [i - 1 for i in b]
        assert t.prufer_repr == b
        assert sorted(Prufer(b).tree_repr) == a
        assert Prufer.unrank(t.rank, n).prufer_repr == b

    doit([[1, 2]], [])
    doit([[2, 1, 3]], [1])
    doit([[1, 3, 2]], [3])
    doit([[1, 2, 3]], [2])
    doit([[2, 1, 4], [1, 3]], [1, 1])
    doit([[3, 2, 1, 4]], [2, 1])
    doit([[3, 2, 1], [2, 4]], [2, 2])
    doit([[1, 3, 2, 4]], [3, 2])
    doit([[1, 4, 2, 3]], [4, 2])
    doit([[3, 1, 4, 2]], [4, 1])
    doit([[4, 2, 1, 3]], [1, 2])
    doit([[1, 2, 4, 3]], [2, 4])
    doit([[1, 3, 4, 2]], [3, 4])
    doit([[2, 4, 1], [4, 3]], [4, 4])
    doit([[1, 2, 3, 4]], [2, 3])
    doit([[2, 3, 1], [3, 4]], [3, 3])
    doit([[1, 4, 3, 2]], [4, 3])
    doit([[2, 1, 4, 3]], [1, 4])
    doit([[2, 1, 3, 4]], [1, 3])
    doit([[6, 2, 1, 4], [1, 3, 5, 8], [3, 7]], [1, 2, 1, 3, 3, 5])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\unify\tests\test_rewrite.py ===
from sympy.unify.rewrite import rewriterule
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y
from sympy.strategies.rl import rebuild
from sympy.assumptions import Q

p, q = Symbol('p'), Symbol('q')

def test_simple():
    rl = rewriterule(Basic(p, S(1)), Basic(p, S(2)), variables=(p,))
    assert list(rl(Basic(S(3), S(1)))) == [Basic(S(3), S(2))]

    p1 = p**2
    p2 = p**3
    rl = rewriterule(p1, p2, variables=(p,))

    expr = x**2
    assert list(rl(expr)) == [x**3]

def test_simple_variables():
    rl = rewriterule(Basic(x, S(1)), Basic(x, S(2)), variables=(x,))
    assert list(rl(Basic(S(3), S(1)))) == [Basic(S(3), S(2))]

    rl = rewriterule(x**2, x**3, variables=(x,))
    assert list(rl(y**2)) == [y**3]

def test_moderate():
    p1 = p**2 + q**3
    p2 = (p*q)**4
    rl = rewriterule(p1, p2, (p, q))

    expr = x**2 + y**3
    assert list(rl(expr)) == [(x*y)**4]

def test_sincos():
    p1 = sin(p)**2 + sin(p)**2
    p2 = 1
    rl = rewriterule(p1, p2, (p, q))

    assert list(rl(sin(x)**2 + sin(x)**2)) == [1]
    assert list(rl(sin(y)**2 + sin(y)**2)) == [1]

def test_Exprs_ok():
    rl = rewriterule(p+q, q+p, (p, q))
    next(rl(x+y)).is_commutative
    str(next(rl(x+y)))

def test_condition_simple():
    rl = rewriterule(x, x+1, [x], lambda x: x < 10)
    assert not list(rl(S(15)))
    assert rebuild(next(rl(S(5)))) == 6


def test_condition_multiple():
    rl = rewriterule(x + y, x**y, [x,y], lambda x, y: x.is_integer)

    a = Symbol('a')
    b = Symbol('b', integer=True)
    expr = a + b
    assert list(rl(expr)) == [b**a]

    c = Symbol('c', integer=True)
    d = Symbol('d', integer=True)
    assert set(rl(c + d)) == {c**d, d**c}

def test_assumptions():
    rl = rewriterule(x + y, x**y, [x, y], assume=Q.integer(x))

    a, b = map(Symbol, 'ab')
    expr = a + b
    assert list(rl(expr, Q.integer(b))) == [b**a]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_cpp.py ===
from __future__ import print_function
from __future__ import absolute_import

import subprocess
import unittest

import greenlet
from . import _test_extension_cpp
from . import TestCase
from . import WIN

class CPPTests(TestCase):
    def test_exception_switch(self):
        greenlets = []
        for i in range(4):
            g = greenlet.greenlet(_test_extension_cpp.test_exception_switch)
            g.switch(i)
            greenlets.append(g)
        for i, g in enumerate(greenlets):
            self.assertEqual(g.switch(), i)

    def _do_test_unhandled_exception(self, target):
        import os
        import sys
        script = os.path.join(
            os.path.dirname(__file__),
            'fail_cpp_exception.py',
        )
        args = [sys.executable, script, target.__name__ if not isinstance(target, str) else target]
        __traceback_info__ = args
        with self.assertRaises(subprocess.CalledProcessError) as exc:
            subprocess.check_output(
                args,
                encoding='utf-8',
                stderr=subprocess.STDOUT
            )

        ex = exc.exception
        expected_exit = self.get_expected_returncodes_for_aborted_process()
        self.assertIn(ex.returncode, expected_exit)
        self.assertIn('fail_cpp_exception is running', ex.output)
        return ex.output


    def test_unhandled_nonstd_exception_aborts(self):
        # verify that plain unhandled throw aborts
        self._do_test_unhandled_exception(_test_extension_cpp.test_exception_throw_nonstd)

    def test_unhandled_std_exception_aborts(self):
        # verify that plain unhandled throw aborts
        self._do_test_unhandled_exception(_test_extension_cpp.test_exception_throw_std)

    @unittest.skipIf(WIN, "XXX: This does not crash on Windows")
    # Meaning the exception is getting lost somewhere...
    def test_unhandled_std_exception_as_greenlet_function_aborts(self):
        # verify that plain unhandled throw aborts
        output = self._do_test_unhandled_exception('run_as_greenlet_target')
        self.assertIn(
            # We really expect this to be prefixed with "greenlet: Unhandled C++ exception:"
            # as added by our handler for std::exception (see TUserGreenlet.cpp), but
            # that's not correct everywhere --- our handler never runs before std::terminate
            # gets called (for example, on arm32).
            'Thrown from an extension.',
            output
        )

    def test_unhandled_exception_in_greenlet_aborts(self):
        # verify that unhandled throw called in greenlet aborts too
        self._do_test_unhandled_exception('run_unhandled_exception_in_greenlet_aborts')


if __name__ == '__main__':
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_ode.py ===
#from mpmath.calculus import ODE_step_euler, ODE_step_rk4, odeint, arange
from mpmath import odefun, cos, sin, mpf, sinc, mp

'''
solvers = [ODE_step_euler, ODE_step_rk4]

def test_ode1():
    """
    Let's solve:

    x'' + w**2 * x = 0

    i.e. x1 = x, x2 = x1':

    x1' =  x2
    x2' = -x1
    """
    def derivs((x1, x2), t):
        return x2, -x1

    for solver in solvers:
        t = arange(0, 3.1415926, 0.005)
        sol = odeint(derivs, (0., 1.), t, solver)
        x1 = [a[0] for a in sol]
        x2 = [a[1] for a in sol]
        # the result is x1 = sin(t), x2 = cos(t)
        # let's just check the end points for t = pi
        assert abs(x1[-1]) < 1e-2
        assert abs(x2[-1] - (-1)) < 1e-2

def test_ode2():
    """
    Let's solve:

    x' - x = 0

    i.e. x = exp(x)

    """
    def derivs((x), t):
        return x

    for solver in solvers:
        t = arange(0, 1, 1e-3)
        sol = odeint(derivs, (1.,), t, solver)
        x = [a[0] for a in sol]
        # the result is x = exp(t)
        # let's just check the end point for t = 1, i.e. x = e
        assert abs(x[-1] - 2.718281828) < 1e-2
'''

def test_odefun_rational():
    mp.dps = 15
    # A rational function
    f = lambda t: 1/(1+mpf(t)**2)
    g = odefun(lambda x, y: [-2*x*y[0]**2], 0, [f(0)])
    assert f(2).ae(g(2)[0])

def test_odefun_sinc_large():
    mp.dps = 15
    # Sinc function; test for large x
    f = sinc
    g = odefun(lambda x, y: [(cos(x)-y[0])/x], 1, [f(1)], tol=0.01, degree=5)
    assert abs(f(100) - g(100)[0])/f(100) < 0.01

def test_odefun_harmonic():
    mp.dps = 15
    # Harmonic oscillator
    f = odefun(lambda x, y: [-y[1], y[0]], 0, [1, 0])
    for x in [0, 1, 2.5, 8, 3.7]:    #  we go back to 3.7 to check caching
        c, s = f(x)
        assert c.ae(cos(x))
        assert s.ae(sin(x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\string_\test_concat.py ===
import numpy as np
import pytest

from pandas.compat import HAS_PYARROW

from pandas.core.dtypes.cast import find_common_type

import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version


@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        # same types
        ([("pyarrow", pd.NA), ("pyarrow", pd.NA)], ("pyarrow", pd.NA)),
        ([("pyarrow", np.nan), ("pyarrow", np.nan)], ("pyarrow", np.nan)),
        ([("python", pd.NA), ("python", pd.NA)], ("python", pd.NA)),
        ([("python", np.nan), ("python", np.nan)], ("python", np.nan)),
        # pyarrow preference
        ([("pyarrow", pd.NA), ("python", pd.NA)], ("pyarrow", pd.NA)),
        # NA preference
        ([("python", pd.NA), ("python", np.nan)], ("python", pd.NA)),
    ],
)
def test_concat_series(request, to_concat_dtypes, result_dtype):
    if any(storage == "pyarrow" for storage, _ in to_concat_dtypes) and not HAS_PYARROW:
        pytest.skip("Could not import 'pyarrow'")

    ser_list = [
        pd.Series(["a", "b", None], dtype=pd.StringDtype(storage, na_value))
        for storage, na_value in to_concat_dtypes
    ]

    result = pd.concat(ser_list, ignore_index=True)
    expected = pd.Series(
        ["a", "b", None, "a", "b", None], dtype=pd.StringDtype(*result_dtype)
    )
    tm.assert_series_equal(result, expected)

    # order doesn't matter for result
    result = pd.concat(ser_list[::1], ignore_index=True)
    tm.assert_series_equal(result, expected)


def test_concat_with_object(string_dtype_arguments):
    # _get_common_dtype cannot inspect values, so object dtype with strings still
    # results in object dtype
    result = pd.concat(
        [
            pd.Series(["a", "b", None], dtype=pd.StringDtype(*string_dtype_arguments)),
            pd.Series(["a", "b", None], dtype=object),
        ]
    )
    assert result.dtype == np.dtype("object")


def test_concat_with_numpy(string_dtype_arguments):
    # common type with a numpy string dtype always preserves the pandas string dtype
    dtype = pd.StringDtype(*string_dtype_arguments)
    assert find_common_type([dtype, np.dtype("U")]) == dtype
    assert find_common_type([np.dtype("U"), dtype]) == dtype
    assert find_common_type([dtype, np.dtype("U10")]) == dtype
    assert find_common_type([np.dtype("U10"), dtype]) == dtype

    # with any other numpy dtype -> object
    assert find_common_type([dtype, np.dtype("S")]) == np.dtype("object")
    assert find_common_type([dtype, np.dtype("int64")]) == np.dtype("object")

    if Version(np.__version__) >= Version("2"):
        assert find_common_type([dtype, np.dtypes.StringDType()]) == dtype
        assert find_common_type([np.dtypes.StringDType(), dtype]) == dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\interval\test_contains.py ===
import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)


class TestContains:
    def test_contains(self):
        interval = Interval(0, 1)
        assert 0.5 in interval
        assert 1 in interval
        assert 0 not in interval

        interval_both = Interval(0, 1, "both")
        assert 0 in interval_both
        assert 1 in interval_both

        interval_neither = Interval(0, 1, closed="neither")
        assert 0 not in interval_neither
        assert 0.5 in interval_neither
        assert 1 not in interval_neither

    def test_contains_interval(self, inclusive_endpoints_fixture):
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(0, 1, inclusive_endpoints_fixture)
        assert interval1 in interval1
        assert interval2 in interval2
        assert interval2 in interval1
        assert interval1 not in interval2 or inclusive_endpoints_fixture == "both"

    def test_contains_infinite_length(self):
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(float("-inf"), float("inf"), "neither")
        assert interval1 in interval2
        assert interval2 not in interval1

    def test_contains_zero_length(self):
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(-1, -1, "both")
        interval3 = Interval(0.5, 0.5, "both")
        assert interval2 not in interval1
        assert interval3 in interval1
        assert interval2 not in interval3 and interval3 not in interval2
        assert interval1 not in interval2 and interval1 not in interval3

    @pytest.mark.parametrize(
        "type1",
        [
            (0, 1),
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),
            (Timedelta("0h"), Timedelta("1h")),
        ],
    )
    @pytest.mark.parametrize(
        "type2",
        [
            (0, 1),
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),
            (Timedelta("0h"), Timedelta("1h")),
        ],
    )
    def test_contains_mixed_types(self, type1, type2):
        interval1 = Interval(*type1)
        interval2 = Interval(*type2)
        if type1 == type2:
            assert interval1 in interval2
        else:
            msg = "^'<=' not supported between instances of"
            with pytest.raises(TypeError, match=msg):
                interval1 in interval2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\test_ndim_array.py ===
from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.tensor.array import Array
from sympy.tensor.array.dense_ndim_array import (
    ImmutableDenseNDimArray, MutableDenseNDimArray)
from sympy.tensor.array.sparse_ndim_array import (
    ImmutableSparseNDimArray, MutableSparseNDimArray)

from sympy.abc import x, y

mutable_array_types = [
    MutableDenseNDimArray,
    MutableSparseNDimArray
]

array_types = [
    ImmutableDenseNDimArray,
    ImmutableSparseNDimArray,
    MutableDenseNDimArray,
    MutableSparseNDimArray
]


def test_array_negative_indices():
    for ArrayType in array_types:
        test_array = ArrayType([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        assert test_array[:, -1] == Array([5, 10])
        assert test_array[:, -2] == Array([4, 9])
        assert test_array[:, -3] == Array([3, 8])
        assert test_array[:, -4] == Array([2, 7])
        assert test_array[:, -5] == Array([1, 6])
        assert test_array[:, 0] == Array([1, 6])
        assert test_array[:, 1] == Array([2, 7])
        assert test_array[:, 2] == Array([3, 8])
        assert test_array[:, 3] == Array([4, 9])
        assert test_array[:, 4] == Array([5, 10])

        raises(ValueError, lambda: test_array[:, -6])
        raises(ValueError, lambda: test_array[-3, :])

        assert test_array[-1, -1] == 10


def test_issue_18361():
    A = Array([sin(2 * x) - 2 * sin(x) * cos(x)])
    B = Array([sin(x)**2 + cos(x)**2, 0])
    C = Array([(x + x**2)/(x*sin(y)**2 + x*cos(y)**2), 2*sin(x)*cos(x)])
    assert simplify(A) == Array([0])
    assert simplify(B) == Array([1, 0])
    assert simplify(C) == Array([x + 1, sin(2*x)])


def test_issue_20222():
    A = Array([[1, 2], [3, 4]])
    B = Matrix([[1,2],[3,4]])
    raises(TypeError, lambda: A - B)


def test_issue_17851():
    for array_type in array_types:
        A = array_type([])
        assert isinstance(A, array_type)
        assert A.shape == (0,)
        assert list(A) == []


def test_issue_and_18715():
    for array_type in mutable_array_types:
        A = array_type([0, 1, 2])
        A[0] += 5
        assert A[0] == 5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\_locales.py ===
"""Provide class for testing in French locale

"""
import locale
import sys

import pytest

__ALL__ = ['CommaDecimalPointLocale']


def find_comma_decimal_point_locale():
    """See if platform has a decimal point as comma locale.

    Find a locale that uses a comma instead of a period as the
    decimal point.

    Returns
    -------
    old_locale: str
        Locale when the function was called.
    new_locale: {str, None)
        First French locale found, None if none found.

    """
    if sys.platform == 'win32':
        locales = ['FRENCH']
    else:
        locales = ['fr_FR', 'fr_FR.UTF-8', 'fi_FI', 'fi_FI.UTF-8']

    old_locale = locale.getlocale(locale.LC_NUMERIC)
    new_locale = None
    try:
        for loc in locales:
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
                new_locale = loc
                break
            except locale.Error:
                pass
    finally:
        locale.setlocale(locale.LC_NUMERIC, locale=old_locale)
    return old_locale, new_locale


class CommaDecimalPointLocale:
    """Sets LC_NUMERIC to a locale with comma as decimal point.

    Classes derived from this class have setup and teardown methods that run
    tests with locale.LC_NUMERIC set to a locale where commas (',') are used as
    the decimal point instead of periods ('.'). On exit the locale is restored
    to the initial locale. It also serves as context manager with the same
    effect. If no such locale is available, the test is skipped.

    """
    (cur_locale, tst_locale) = find_comma_decimal_point_locale()

    def setup_method(self):
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    def teardown_method(self):
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)

    def __enter__(self):
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    def __exit__(self, type, value, traceback):
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_pop.py ===
import numpy as np

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestDataFramePop:
    def test_pop(self, float_frame, warn_copy_on_write):
        float_frame.columns.name = "baz"

        float_frame.pop("A")
        assert "A" not in float_frame

        float_frame["foo"] = "bar"
        float_frame.pop("foo")
        assert "foo" not in float_frame
        assert float_frame.columns.name == "baz"

        # gh-10912: inplace ops cause caching issue
        a = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"], index=["X", "Y"])
        b = a.pop("B")
        with tm.assert_cow_warning(warn_copy_on_write):
            b += 1

        # original frame
        expected = DataFrame([[1, 3], [4, 6]], columns=["A", "C"], index=["X", "Y"])
        tm.assert_frame_equal(a, expected)

        # result
        expected = Series([2, 5], index=["X", "Y"], name="B") + 1
        tm.assert_series_equal(b, expected)

    def test_pop_non_unique_cols(self):
        df = DataFrame({0: [0, 1], 1: [0, 1], 2: [4, 5]})
        df.columns = ["a", "b", "a"]

        res = df.pop("a")
        assert type(res) == DataFrame
        assert len(res) == 2
        assert len(df.columns) == 1
        assert "b" in df.columns
        assert "a" not in df.columns
        assert len(df.index) == 2

    def test_mixed_depth_pop(self):
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        df1 = df.copy()
        df2 = df.copy()
        result = df1.pop("a")
        expected = df2.pop(("a", "", ""))
        tm.assert_series_equal(expected, result, check_names=False)
        tm.assert_frame_equal(df1, df2)
        assert result.name == "a"

        expected = df1["top"]
        df1 = df1.drop(["top"], axis=1)
        result = df2.pop("top")
        tm.assert_frame_equal(expected, result)
        tm.assert_frame_equal(df1, df2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_freq_attr.py ===
import pytest

from pandas import TimedeltaIndex

from pandas.tseries.offsets import (
    DateOffset,
    Day,
    Hour,
    MonthEnd,
)


class TestFreq:
    @pytest.mark.parametrize("values", [["0 days", "2 days", "4 days"], []])
    @pytest.mark.parametrize("freq", ["2D", Day(2), "48h", Hour(48)])
    def test_freq_setter(self, values, freq):
        # GH#20678
        idx = TimedeltaIndex(values)

        # can set to an offset, converting from string if necessary
        idx._data.freq = freq
        assert idx.freq == freq
        assert isinstance(idx.freq, DateOffset)

        # can reset to None
        idx._data.freq = None
        assert idx.freq is None

    def test_with_freq_empty_requires_tick(self):
        idx = TimedeltaIndex([])

        off = MonthEnd(1)
        msg = "TimedeltaArray/Index freq must be a Tick"
        with pytest.raises(TypeError, match=msg):
            idx._with_freq(off)
        with pytest.raises(TypeError, match=msg):
            idx._data._with_freq(off)

    def test_freq_setter_errors(self):
        # GH#20678
        idx = TimedeltaIndex(["0 days", "2 days", "4 days"])

        # setting with an incompatible freq
        msg = (
            "Inferred frequency 2D from passed values does not conform to "
            "passed frequency 5D"
        )
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "5D"

        # setting with a non-fixed frequency
        msg = r"<2 \* BusinessDays> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "2B"

        # setting with non-freq string
        with pytest.raises(ValueError, match="Invalid frequency"):
            idx._data.freq = "foo"

    def test_freq_view_safe(self):
        # Setting the freq for one TimedeltaIndex shouldn't alter the freq
        #  for another that views the same data

        tdi = TimedeltaIndex(["0 days", "2 days", "4 days"], freq="2D")
        tda = tdi._data

        tdi2 = TimedeltaIndex(tda)._with_freq(None)
        assert tdi2.freq is None

        # Original was not altered
        assert tdi.freq == "2D"
        assert tda.freq == "2D"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_console.py ===
import locale

import pytest

from pandas._config import detect_console_encoding


class MockEncoding:
    """
    Used to add a side effect when accessing the 'encoding' property. If the
    side effect is a str in nature, the value will be returned. Otherwise, the
    side effect should be an exception that will be raised.
    """

    def __init__(self, encoding) -> None:
        super().__init__()
        self.val = encoding

    @property
    def encoding(self):
        return self.raise_or_return(self.val)

    @staticmethod
    def raise_or_return(val):
        if isinstance(val, str):
            return val
        else:
            raise val


@pytest.mark.parametrize("empty,filled", [["stdin", "stdout"], ["stdout", "stdin"]])
def test_detect_console_encoding_from_stdout_stdin(monkeypatch, empty, filled):
    # Ensures that when sys.stdout.encoding or sys.stdin.encoding is used when
    # they have values filled.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr(f"sys.{empty}", MockEncoding(""))
        context.setattr(f"sys.{filled}", MockEncoding(filled))
        assert detect_console_encoding() == filled


@pytest.mark.parametrize("encoding", [AttributeError, OSError, "ascii"])
def test_detect_console_encoding_fallback_to_locale(monkeypatch, encoding):
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr("locale.getpreferredencoding", lambda: "foo")
        context.setattr("sys.stdout", MockEncoding(encoding))
        assert detect_console_encoding() == "foo"


@pytest.mark.parametrize(
    "std,locale",
    [
        ["ascii", "ascii"],
        ["ascii", locale.Error],
        [AttributeError, "ascii"],
        [AttributeError, locale.Error],
        [OSError, "ascii"],
        [OSError, locale.Error],
    ],
)
def test_detect_console_encoding_fallback_to_default(monkeypatch, std, locale):
    # When both the stdout/stdin encoding and locale preferred encoding checks
    # fail (or return 'ascii', we should default to the sys default encoding.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr(
            "locale.getpreferredencoding", lambda: MockEncoding.raise_or_return(locale)
        )
        context.setattr("sys.stdout", MockEncoding(std))
        context.setattr("sys.getdefaultencoding", lambda: "sysDefaultEncoding")
        assert detect_console_encoding() == "sysDefaultEncoding"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_decimal.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import pytest

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.mark.parametrize(
    "data,thousands,decimal",
    [
        (
            """A|B|C
1|2,334.01|5
10|13|10.
""",
            ",",
            ".",
        ),
        (
            """A|B|C
1|2.334,01|5
10|13|10,
""",
            ".",
            ",",
        ),
    ],
)
def test_1000_sep_with_decimal(all_parsers, data, thousands, decimal):
    parser = all_parsers
    expected = DataFrame({"A": [1, 10], "B": [2334.01, 13], "C": [5, 10.0]})

    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), sep="|", thousands=thousands, decimal=decimal
            )
        return

    result = parser.read_csv(
        StringIO(data), sep="|", thousands=thousands, decimal=decimal
    )
    tm.assert_frame_equal(result, expected)


def test_euro_decimal_format(all_parsers):
    parser = all_parsers
    data = """Id;Number1;Number2;Text1;Text2;Number3
1;1521,1541;187101,9543;ABC;poi;4,738797819
2;121,12;14897,76;DEF;uyt;0,377320872
3;878,158;108013,434;GHI;rez;2,735694704"""

    result = parser.read_csv(StringIO(data), sep=";", decimal=",")
    expected = DataFrame(
        [
            [1, 1521.1541, 187101.9543, "ABC", "poi", 4.738797819],
            [2, 121.12, 14897.76, "DEF", "uyt", 0.377320872],
            [3, 878.158, 108013.434, "GHI", "rez", 2.735694704],
        ],
        columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_time_series.py ===
import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    _testing as tm,
    date_range,
    period_range,
)
from pandas.tests.io.pytables.common import ensure_clean_store

pytestmark = pytest.mark.single_cpu


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_store_datetime_fractional_secs(setup_path, unit):
    dt = datetime.datetime(2012, 1, 2, 3, 4, 5, 123456)
    dti = DatetimeIndex([dt], dtype=f"M8[{unit}]")
    series = Series([0], index=dti)
    with ensure_clean_store(setup_path) as store:
        store["a"] = series
        assert store["a"].index[0] == dt


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_tseries_indices_series(setup_path):
    with ensure_clean_store(setup_path) as store:
        idx = date_range("2020-01-01", periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        store["a"] = ser
        result = store["a"]

        tm.assert_series_equal(result, ser)
        assert result.index.freq == ser.index.freq
        tm.assert_class_equal(result.index, ser.index, obj="series index")

        idx = period_range("2020-01-01", periods=10, freq="D")
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        store["a"] = ser
        result = store["a"]

        tm.assert_series_equal(result, ser)
        assert result.index.freq == ser.index.freq
        tm.assert_class_equal(result.index, ser.index, obj="series index")


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_tseries_indices_frame(setup_path):
    with ensure_clean_store(setup_path) as store:
        idx = date_range("2020-01-01", periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        store["a"] = df
        result = store["a"]

        tm.assert_frame_equal(result, df)
        assert result.index.freq == df.index.freq
        tm.assert_class_equal(result.index, df.index, obj="dataframe index")

        idx = period_range("2020-01-01", periods=10, freq="D")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), idx)
        store["a"] = df
        result = store["a"]

        tm.assert_frame_equal(result, df)
        assert result.index.freq == df.index.freq
        tm.assert_class_equal(result.index, df.index, obj="dataframe index")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\frame\test_frame_groupby.py ===
""" Test cases for DataFrame.plot """

import pytest

from pandas import DataFrame
from pandas.tests.plotting.common import _check_visible

pytest.importorskip("matplotlib")


class TestDataFramePlotsGroupby:
    def _assert_ytickslabels_visibility(self, axes, expected):
        for ax, exp in zip(axes, expected):
            _check_visible(ax.get_yticklabels(), visible=exp)

    def _assert_xtickslabels_visibility(self, axes, expected):
        for ax, exp in zip(axes, expected):
            _check_visible(ax.get_xticklabels(), visible=exp)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            # behavior without keyword
            ({}, [True, False, True, False]),
            # set sharey=True should be identical
            ({"sharey": True}, [True, False, True, False]),
            # sharey=False, all yticklabels should be visible
            ({"sharey": False}, [True, True, True, True]),
        ],
    )
    def test_groupby_boxplot_sharey(self, kwargs, expected):
        # https://github.com/pandas-dev/pandas/issues/20968
        # sharey can now be switched check whether the right
        # pair of axes is turned on or off
        df = DataFrame(
            {
                "a": [-1.43, -0.15, -3.70, -1.43, -0.14],
                "b": [0.56, 0.84, 0.29, 0.56, 0.85],
                "c": [0, 1, 2, 3, 1],
            },
            index=[0, 1, 2, 3, 4],
        )
        axes = df.groupby("c").boxplot(**kwargs)
        self._assert_ytickslabels_visibility(axes, expected)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            # behavior without keyword
            ({}, [True, True, True, True]),
            # set sharex=False should be identical
            ({"sharex": False}, [True, True, True, True]),
            # sharex=True, xticklabels should be visible
            # only for bottom plots
            ({"sharex": True}, [False, False, True, True]),
        ],
    )
    def test_groupby_boxplot_sharex(self, kwargs, expected):
        # https://github.com/pandas-dev/pandas/issues/20968
        # sharex can now be switched check whether the right
        # pair of axes is turned on or off

        df = DataFrame(
            {
                "a": [-1.43, -0.15, -3.70, -1.43, -0.14],
                "b": [0.56, 0.84, 0.29, 0.56, 0.85],
                "c": [0, 1, 2, 3, 1],
            },
            index=[0, 1, 2, 3, 4],
        )
        axes = df.groupby("c").boxplot(**kwargs)
        self._assert_xtickslabels_visibility(axes, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tools\test_to_time.py ===
from datetime import time
import locale

import numpy as np
import pytest

from pandas.compat import PY311

from pandas import Series
import pandas._testing as tm
from pandas.core.tools.times import to_time

# The tests marked with this are locale-dependent.
# They pass, except when the machine locale is zh_CN or it_IT.
fails_on_non_english = pytest.mark.xfail(
    locale.getlocale()[0] in ("zh_CN", "it_IT"),
    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
    strict=False,
)


class TestToTime:
    @pytest.mark.parametrize(
        "time_string",
        [
            "14:15",
            "1415",
            pytest.param("2:15pm", marks=fails_on_non_english),
            pytest.param("0215pm", marks=fails_on_non_english),
            "14:15:00",
            "141500",
            pytest.param("2:15:00pm", marks=fails_on_non_english),
            pytest.param("021500pm", marks=fails_on_non_english),
            time(14, 15),
        ],
    )
    def test_parsers_time(self, time_string):
        # GH#11818
        assert to_time(time_string) == time(14, 15)

    def test_odd_format(self):
        new_string = "14.15"
        msg = r"Cannot convert arg \['14\.15'\] to a time"
        if not PY311:
            with pytest.raises(ValueError, match=msg):
                to_time(new_string)
        assert to_time(new_string, format="%H.%M") == time(14, 15)

    def test_arraylike(self):
        arg = ["14:15", "20:20"]
        expected_arr = [time(14, 15), time(20, 20)]
        assert to_time(arg) == expected_arr
        assert to_time(arg, format="%H:%M") == expected_arr
        assert to_time(arg, infer_time_format=True) == expected_arr
        assert to_time(arg, format="%I:%M%p", errors="coerce") == [None, None]

        msg = "errors='ignore' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = to_time(arg, format="%I:%M%p", errors="ignore")
        tm.assert_numpy_array_equal(res, np.array(arg, dtype=np.object_))

        msg = "Cannot convert.+to a time with given format"
        with pytest.raises(ValueError, match=msg):
            to_time(arg, format="%I:%M%p", errors="raise")

        tm.assert_series_equal(
            to_time(Series(arg, name="test")), Series(expected_arr, name="test")
        )

        res = to_time(np.array(arg))
        assert isinstance(res, list)
        assert res == expected_arr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\moments\conftest.py ===
import itertools

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    notna,
)


def create_series():
    return [
        Series(dtype=np.float64, name="a"),
        Series([np.nan] * 5),
        Series([1.0] * 5),
        Series(range(5, 0, -1)),
        Series(range(5)),
        Series([np.nan, 1.0, np.nan, 1.0, 1.0]),
        Series([np.nan, 1.0, np.nan, 2.0, 3.0]),
        Series([np.nan, 1.0, np.nan, 3.0, 2.0]),
    ]


def create_dataframes():
    return [
        DataFrame(columns=["a", "a"]),
        DataFrame(np.arange(15).reshape((5, 3)), columns=["a", "a", 99]),
    ] + [DataFrame(s) for s in create_series()]


def is_constant(x):
    values = x.values.ravel("K")
    return len(set(values[notna(values)])) == 1


@pytest.fixture(
    params=(
        obj
        for obj in itertools.chain(create_series(), create_dataframes())
        if is_constant(obj)
    ),
)
def consistent_data(request):
    return request.param


@pytest.fixture(params=create_series())
def series_data(request):
    return request.param


@pytest.fixture(params=itertools.chain(create_series(), create_dataframes()))
def all_data(request):
    """
    Test:
        - Empty Series / DataFrame
        - All NaN
        - All consistent value
        - Monotonically decreasing
        - Monotonically increasing
        - Monotonically consistent with NaNs
        - Monotonically increasing with NaNs
        - Monotonically decreasing with NaNs
    """
    return request.param


@pytest.fixture(params=[0, 2])
def min_periods(request):
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_graycode.py ===
from sympy.combinatorics.graycode import (GrayCode, bin_to_gray,
    random_bitstring, get_subset_from_bitstring, graycode_subsets,
    gray_to_bin)
from sympy.testing.pytest import raises

def test_graycode():
    g = GrayCode(2)
    got = []
    for i in g.generate_gray():
        if i.startswith('0'):
            g.skip()
        got.append(i)
    assert got == '00 11 10'.split()
    a = GrayCode(6)
    assert a.current == '0'*6
    assert a.rank == 0
    assert len(list(a.generate_gray())) == 64
    codes = ['011001', '011011', '011010',
    '011110', '011111', '011101', '011100', '010100', '010101', '010111',
    '010110', '010010', '010011', '010001', '010000', '110000', '110001',
    '110011', '110010', '110110', '110111', '110101', '110100', '111100',
    '111101', '111111', '111110', '111010', '111011', '111001', '111000',
    '101000', '101001', '101011', '101010', '101110', '101111', '101101',
    '101100', '100100', '100101', '100111', '100110', '100010', '100011',
    '100001', '100000']
    assert list(a.generate_gray(start='011001')) == codes
    assert list(
        a.generate_gray(rank=GrayCode(6, start='011001').rank)) == codes
    assert a.next().current == '000001'
    assert a.next(2).current == '000011'
    assert a.next(-1).current == '100000'

    a = GrayCode(5, start='10010')
    assert a.rank == 28
    a = GrayCode(6, start='101000')
    assert a.rank == 48

    assert GrayCode(6, rank=4).current == '000110'
    assert GrayCode(6, rank=4).rank == 4
    assert [GrayCode(4, start=s).rank for s in
            GrayCode(4).generate_gray()] == [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                             9, 10, 11, 12, 13, 14, 15]
    a = GrayCode(15, rank=15)
    assert a.current == '000000000001000'

    assert bin_to_gray('111') == '100'

    a = random_bitstring(5)
    assert type(a) is str
    assert len(a) == 5
    assert all(i in ['0', '1'] for i in a)

    assert get_subset_from_bitstring(
        ['a', 'b', 'c', 'd'], '0011') == ['c', 'd']
    assert get_subset_from_bitstring('abcd', '1001') == ['a', 'd']
    assert list(graycode_subsets(['a', 'b', 'c'])) == \
        [[], ['c'], ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'],
         ['a', 'c'], ['a']]

    raises(ValueError, lambda: GrayCode(0))
    raises(ValueError, lambda: GrayCode(2.2))
    raises(ValueError, lambda: GrayCode(2, start=[1, 1, 0]))
    raises(ValueError, lambda: GrayCode(2, rank=2.5))
    raises(ValueError, lambda: get_subset_from_bitstring(['c', 'a', 'c'], '1100'))
    raises(ValueError, lambda: list(GrayCode(3).generate_gray(start="1111")))


def test_live_issue_117():
    assert bin_to_gray('0100') == '0110'
    assert bin_to_gray('0101') == '0111'
    for bits in ('0100', '0101'):
        assert gray_to_bin(bin_to_gray(bits)) == bits

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_data.py ===
import pytest

import numpy as np
from numpy.f2py.crackfortran import crackfortran

from . import util


class TestData(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_stmts.f90")]

    # For gh-23276
    @pytest.mark.slow
    def test_data_stmts(self):
        assert self.module.cmplxdat.i == 2
        assert self.module.cmplxdat.j == 3
        assert self.module.cmplxdat.x == 1.5
        assert self.module.cmplxdat.y == 2.0
        assert self.module.cmplxdat.pi == 3.1415926535897932384626433832795028841971693993751058209749445923078164062
        assert self.module.cmplxdat.medium_ref_index == np.array(1. + 0.j)
        assert np.all(self.module.cmplxdat.z == np.array([3.5, 7.0]))
        assert np.all(self.module.cmplxdat.my_array == np.array([ 1. + 2.j, -3. + 4.j]))
        assert np.all(self.module.cmplxdat.my_real_array == np.array([ 1., 2., 3.]))
        assert np.all(self.module.cmplxdat.ref_index_one == np.array([13.0 + 21.0j]))
        assert np.all(self.module.cmplxdat.ref_index_two == np.array([-30.0 + 43.0j]))

    def test_crackedlines(self):
        mod = crackfortran(self.sources)
        assert mod[0]['vars']['x']['='] == '1.5'
        assert mod[0]['vars']['y']['='] == '2.0'
        assert mod[0]['vars']['pi']['='] == '3.1415926535897932384626433832795028841971693993751058209749445923078164062d0'
        assert mod[0]['vars']['my_real_array']['='] == '(/1.0d0, 2.0d0, 3.0d0/)'
        assert mod[0]['vars']['ref_index_one']['='] == '(13.0d0, 21.0d0)'
        assert mod[0]['vars']['ref_index_two']['='] == '(-30.0d0, 43.0d0)'
        assert mod[0]['vars']['my_array']['='] == '(/(1.0d0, 2.0d0), (-3.0d0, 4.0d0)/)'
        assert mod[0]['vars']['z']['='] == '(/3.5,  7.0/)'

class TestDataF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_common.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.mycom.mydata == 0

    def test_crackedlines(self):
        mod = crackfortran(str(self.sources[0]))
        print(mod[0]['vars'])
        assert mod[0]['vars']['mydata']['='] == '0'


class TestDataMultiplierF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_multiplier.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.mycom.ivar1 == 3
        assert self.module.mycom.ivar2 == 3
        assert self.module.mycom.ivar3 == 2
        assert self.module.mycom.ivar4 == 2
        assert self.module.mycom.evar5 == 0


class TestDataWithCommentsF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_with_comments.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert len(self.module.mycom.mytab) == 3
        assert self.module.mycom.mytab[0] == 0
        assert self.module.mycom.mytab[1] == 4
        assert self.module.mycom.mytab[2] == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_delete.py ===
from pandas import (
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexDelete:
    def test_delete(self):
        idx = timedelta_range(start="1 Days", periods=5, freq="D", name="idx")

        # preserve freq
        expected_0 = timedelta_range(start="2 Days", periods=4, freq="D", name="idx")
        expected_4 = timedelta_range(start="1 Days", periods=4, freq="D", name="idx")

        # reset freq to None
        expected_1 = TimedeltaIndex(
            ["1 day", "3 day", "4 day", "5 day"], freq=None, name="idx"
        )

        cases = {
            0: expected_0,
            -5: expected_0,
            -1: expected_4,
            4: expected_4,
            1: expected_1,
        }
        for n, expected in cases.items():
            result = idx.delete(n)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

        with tm.external_error_raised((IndexError, ValueError)):
            # either depending on numpy version
            idx.delete(5)

    def test_delete_slice(self):
        idx = timedelta_range(start="1 days", periods=10, freq="D", name="idx")

        # preserve freq
        expected_0_2 = timedelta_range(start="4 days", periods=7, freq="D", name="idx")
        expected_7_9 = timedelta_range(start="1 days", periods=7, freq="D", name="idx")

        # reset freq to None
        expected_3_5 = TimedeltaIndex(
            ["1 d", "2 d", "3 d", "7 d", "8 d", "9 d", "10d"], freq=None, name="idx"
        )

        cases = {
            (0, 1, 2): expected_0_2,
            (7, 8, 9): expected_7_9,
            (3, 4, 5): expected_3_5,
        }
        for n, expected in cases.items():
            result = idx.delete(n)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

            result = idx.delete(slice(n[0], n[-1] + 1))
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

    def test_delete_doesnt_infer_freq(self):
        # GH#30655 behavior matches DatetimeIndex

        tdi = TimedeltaIndex(["1 Day", "2 Days", None, "3 Days", "4 Days"])
        result = tdi.delete(2)
        assert result.freq is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_inertia.py ===
from sympy import symbols
from sympy.testing.pytest import raises
from sympy.physics.mechanics import (inertia, inertia_of_point_mass,
                                     Inertia, ReferenceFrame, Point)


def test_inertia_dyadic():
    N = ReferenceFrame('N')
    ixx, iyy, izz = symbols('ixx iyy izz')
    ixy, iyz, izx = symbols('ixy iyz izx')
    assert inertia(N, ixx, iyy, izz) == (ixx * (N.x | N.x) + iyy *
            (N.y | N.y) + izz * (N.z | N.z))
    assert inertia(N, 0, 0, 0) == 0 * (N.x | N.x)
    raises(TypeError, lambda: inertia(0, 0, 0, 0))
    assert inertia(N, ixx, iyy, izz, ixy, iyz, izx) == (ixx * (N.x | N.x) +
            ixy * (N.x | N.y) + izx * (N.x | N.z) + ixy * (N.y | N.x) + iyy *
        (N.y | N.y) + iyz * (N.y | N.z) + izx * (N.z | N.x) + iyz * (N.z |
            N.y) + izz * (N.z | N.z))


def test_inertia_of_point_mass():
    r, s, t, m = symbols('r s t m')
    N = ReferenceFrame('N')

    px = r * N.x
    I = inertia_of_point_mass(m, px, N)
    assert I == m * r**2 * (N.y | N.y) + m * r**2 * (N.z | N.z)

    py = s * N.y
    I = inertia_of_point_mass(m, py, N)
    assert I == m * s**2 * (N.x | N.x) + m * s**2 * (N.z | N.z)

    pz = t * N.z
    I = inertia_of_point_mass(m, pz, N)
    assert I == m * t**2 * (N.x | N.x) + m * t**2 * (N.y | N.y)

    p = px + py + pz
    I = inertia_of_point_mass(m, p, N)
    assert I == (m * (s**2 + t**2) * (N.x | N.x) -
                 m * r * s * (N.x | N.y) -
                 m * r * t * (N.x | N.z) -
                 m * r * s * (N.y | N.x) +
                 m * (r**2 + t**2) * (N.y | N.y) -
                 m * s * t * (N.y | N.z) -
                 m * r * t * (N.z | N.x) -
                 m * s * t * (N.z | N.y) +
                 m * (r**2 + s**2) * (N.z | N.z))


def test_inertia_object():
    N = ReferenceFrame('N')
    O = Point('O')
    ixx, iyy, izz = symbols('ixx iyy izz')
    I_dyadic = ixx * (N.x | N.x) + iyy * (N.y | N.y) + izz * (N.z | N.z)
    I = Inertia(inertia(N, ixx, iyy, izz), O)
    assert isinstance(I, tuple)
    assert I.__repr__() == ('Inertia(dyadic=ixx*(N.x|N.x) + iyy*(N.y|N.y) + '
                            'izz*(N.z|N.z), point=O)')
    assert I.dyadic == I_dyadic
    assert I.point == O
    assert I[0] == I_dyadic
    assert I[1] == O
    assert I == (I_dyadic, O)  # Test tuple equal
    raises(TypeError, lambda: I != (O, I_dyadic))  # Incorrect tuple order
    assert I == Inertia(O, I_dyadic)  # Parse changed argument order
    assert I == Inertia.from_inertia_scalars(O, N, ixx, iyy, izz)
    # Test invalid tuple operations
    raises(TypeError, lambda: I + (1, 2))
    raises(TypeError, lambda: (1, 2) + I)
    raises(TypeError, lambda: I * 2)
    raises(TypeError, lambda: 2 * I)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_innerproduct.py ===
from sympy.core.numbers import (I, Integer)

from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import Bra, Ket, StateBase


def test_innerproduct():
    k = Ket('k')
    b = Bra('b')
    ip = InnerProduct(b, k)
    assert isinstance(ip, InnerProduct)
    assert ip.bra == b
    assert ip.ket == k
    assert b*k == InnerProduct(b, k)
    assert k*(b*k)*b == k*InnerProduct(b, k)*b
    assert InnerProduct(b, k).subs(b, Dagger(k)) == Dagger(k)*k


def test_innerproduct_dagger():
    k = Ket('k')
    b = Bra('b')
    ip = b*k
    assert Dagger(ip) == Dagger(k)*Dagger(b)


class FooState(StateBase):
    pass


class FooKet(Ket, FooState):

    @classmethod
    def dual_class(self):
        return FooBra

    def _eval_innerproduct_FooBra(self, bra):
        return Integer(1)

    def _eval_innerproduct_BarBra(self, bra):
        return I


class FooBra(Bra, FooState):
    @classmethod
    def dual_class(self):
        return FooKet


class BarState(StateBase):
    pass


class BarKet(Ket, BarState):
    @classmethod
    def dual_class(self):
        return BarBra


class BarBra(Bra, BarState):
    @classmethod
    def dual_class(self):
        return BarKet


def test_doit():
    f = FooKet('foo')
    b = BarBra('bar')
    assert InnerProduct(b, f).doit() == I
    assert InnerProduct(Dagger(f), Dagger(b)).doit() == -I
    assert InnerProduct(Dagger(f), f).doit() == Integer(1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_validate_args.py ===
import pytest

from pandas.util._validators import validate_args


@pytest.fixture
def _fname():
    return "func"


def test_bad_min_fname_arg_count(_fname):
    msg = "'max_fname_arg_count' must be non-negative"

    with pytest.raises(ValueError, match=msg):
        validate_args(_fname, (None,), -1, "foo")


def test_bad_arg_length_max_value_single(_fname):
    args = (None, None)
    compat_args = ("foo",)

    min_fname_arg_count = 0
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(args) + min_fname_arg_count
    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"argument \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args(_fname, args, min_fname_arg_count, compat_args)


def test_bad_arg_length_max_value_multiple(_fname):
    args = (None, None)
    compat_args = {"foo": None}

    min_fname_arg_count = 2
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(args) + min_fname_arg_count
    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"arguments \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args(_fname, args, min_fname_arg_count, compat_args)


@pytest.mark.parametrize("i", range(1, 3))
def test_not_all_defaults(i, _fname):
    bad_arg = "foo"
    msg = (
        f"the '{bad_arg}' parameter is not supported "
        rf"in the pandas implementation of {_fname}\(\)"
    )

    compat_args = {"foo": 2, "bar": -1, "baz": 3}
    arg_vals = (1, -1, 3)

    with pytest.raises(ValueError, match=msg):
        validate_args(_fname, arg_vals[:i], 2, compat_args)


def test_validation(_fname):
    # No exceptions should be raised.
    validate_args(_fname, (None,), 2, {"out": None})

    compat_args = {"axis": 1, "out": None}
    validate_args(_fname, (1, None), 2, compat_args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_named_groups.py ===
from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
                                              DihedralGroup, AlternatingGroup,
                                              AbelianGroup, RubikGroup)
from sympy.testing.pytest import raises


def test_SymmetricGroup():
    G = SymmetricGroup(5)
    elements = list(G.generate())
    assert (G.generators[0]).size == 5
    assert len(elements) == 120
    assert G.is_solvable is False
    assert G.is_abelian is False
    assert G.is_nilpotent is False
    assert G.is_transitive() is True
    H = SymmetricGroup(1)
    assert H.order() == 1
    L = SymmetricGroup(2)
    assert L.order() == 2


def test_CyclicGroup():
    G = CyclicGroup(10)
    elements = list(G.generate())
    assert len(elements) == 10
    assert (G.derived_subgroup()).order() == 1
    assert G.is_abelian is True
    assert G.is_solvable is True
    assert G.is_nilpotent is True
    H = CyclicGroup(1)
    assert H.order() == 1
    L = CyclicGroup(2)
    assert L.order() == 2


def test_DihedralGroup():
    G = DihedralGroup(6)
    elements = list(G.generate())
    assert len(elements) == 12
    assert G.is_transitive() is True
    assert G.is_abelian is False
    assert G.is_solvable is True
    assert G.is_nilpotent is False
    H = DihedralGroup(1)
    assert H.order() == 2
    L = DihedralGroup(2)
    assert L.order() == 4
    assert L.is_abelian is True
    assert L.is_nilpotent is True


def test_AlternatingGroup():
    G = AlternatingGroup(5)
    elements = list(G.generate())
    assert len(elements) == 60
    assert [perm.is_even for perm in elements] == [True]*60
    H = AlternatingGroup(1)
    assert H.order() == 1
    L = AlternatingGroup(2)
    assert L.order() == 1


def test_AbelianGroup():
    A = AbelianGroup(3, 3, 3)
    assert A.order() == 27
    assert A.is_abelian is True


def test_RubikGroup():
    raises(ValueError, lambda: RubikGroup(1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_concat.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        (["Int64", "Int64"], "Int64"),
        (["UInt64", "UInt64"], "UInt64"),
        (["Int8", "Int8"], "Int8"),
        (["Int8", "Int16"], "Int16"),
        (["UInt8", "Int8"], "Int16"),
        (["Int32", "UInt32"], "Int64"),
        (["Int64", "UInt64"], "Float64"),
        (["Int64", "boolean"], "object"),
        (["UInt8", "boolean"], "object"),
    ],
)
def test_concat_series(to_concat_dtypes, result_dtype):
    # we expect the same dtypes as we would get with non-masked inputs,
    #  just masked where available.

    result = pd.concat([pd.Series([0, 1, pd.NA], dtype=t) for t in to_concat_dtypes])
    expected = pd.concat([pd.Series([0, 1, pd.NA], dtype=object)] * 2).astype(
        result_dtype
    )
    tm.assert_series_equal(result, expected)

    # order doesn't matter for result
    result = pd.concat(
        [pd.Series([0, 1, pd.NA], dtype=t) for t in to_concat_dtypes[::-1]]
    )
    expected = pd.concat([pd.Series([0, 1, pd.NA], dtype=object)] * 2).astype(
        result_dtype
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        (["Int64", "int64"], "Int64"),
        (["UInt64", "uint64"], "UInt64"),
        (["Int8", "int8"], "Int8"),
        (["Int8", "int16"], "Int16"),
        (["UInt8", "int8"], "Int16"),
        (["Int32", "uint32"], "Int64"),
        (["Int64", "uint64"], "Float64"),
        (["Int64", "bool"], "object"),
        (["UInt8", "bool"], "object"),
    ],
)
def test_concat_series_with_numpy(to_concat_dtypes, result_dtype):
    # we expect the same dtypes as we would get with non-masked inputs,
    #  just masked where available.

    s1 = pd.Series([0, 1, pd.NA], dtype=to_concat_dtypes[0])
    s2 = pd.Series(np.array([0, 1], dtype=to_concat_dtypes[1]))
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series([0, 1, pd.NA, 0, 1], dtype=object).astype(result_dtype)
    tm.assert_series_equal(result, expected)

    # order doesn't matter for result
    result = pd.concat([s2, s1], ignore_index=True)
    expected = pd.Series([0, 1, 0, 1, pd.NA], dtype=object).astype(result_dtype)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\index\test_datetimeindex.py ===
import pytest

from pandas import (
    DatetimeIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting a value on a view:FutureWarning"
)


@pytest.mark.parametrize(
    "cons",
    [
        lambda x: DatetimeIndex(x),
        lambda x: DatetimeIndex(DatetimeIndex(x)),
    ],
)
def test_datetimeindex(using_copy_on_write, cons):
    dt = date_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    idx = cons(ser)
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31")
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected)


def test_datetimeindex_tz_convert(using_copy_on_write):
    dt = date_range("2019-12-31", periods=3, freq="D", tz="Europe/Berlin")
    ser = Series(dt)
    idx = DatetimeIndex(ser).tz_convert("US/Eastern")
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31", tz="Europe/Berlin")
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected)


def test_datetimeindex_tz_localize(using_copy_on_write):
    dt = date_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    idx = DatetimeIndex(ser).tz_localize("Europe/Berlin")
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31")
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected)


def test_datetimeindex_isocalendar(using_copy_on_write):
    dt = date_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    df = DatetimeIndex(ser).isocalendar()
    expected = df.index.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31")
    if using_copy_on_write:
        tm.assert_index_equal(df.index, expected)


def test_index_values(using_copy_on_write):
    idx = date_range("2019-12-31", periods=3, freq="D")
    result = idx.values
    if using_copy_on_write:
        assert result.flags.writeable is False
    else:
        assert result.flags.writeable is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_delitem.py ===
import pytest

from pandas import (
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


class TestSeriesDelItem:
    def test_delitem(self):
        # GH#5542
        # should delete the item inplace
        s = Series(range(5))
        del s[0]

        expected = Series(range(1, 5), index=range(1, 5))
        tm.assert_series_equal(s, expected)

        del s[1]
        expected = Series(range(2, 5), index=range(2, 5))
        tm.assert_series_equal(s, expected)

        # only 1 left, del, add, del
        s = Series(1)
        del s[0]
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype="int64")))
        s[0] = 1
        tm.assert_series_equal(s, Series(1))
        del s[0]
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype="int64")))

    def test_delitem_object_index(self):
        # Index(dtype=object)
        s = Series(1, index=Index(["a"], dtype="str"))
        del s["a"]
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype="str")))
        s["a"] = 1
        tm.assert_series_equal(s, Series(1, index=Index(["a"], dtype="str")))
        del s["a"]
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype="str")))

    def test_delitem_missing_key(self):
        # empty
        s = Series(dtype=object)

        with pytest.raises(KeyError, match=r"^0$"):
            del s[0]

    def test_delitem_extension_dtype(self):
        # GH#40386
        # DatetimeTZDtype
        dti = date_range("2016-01-01", periods=3, tz="US/Pacific")
        ser = Series(dti)

        expected = ser[[0, 2]]
        del ser[1]
        assert ser.dtype == dti.dtype
        tm.assert_series_equal(ser, expected)

        # PeriodDtype
        pi = dti.tz_localize(None).to_period("D")
        ser = Series(pi)

        expected = ser[:2]
        del ser[2]
        assert ser.dtype == pi.dtype
        tm.assert_series_equal(ser, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_mask.py ===
import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


def test_mask():
    # compare with tested results in test_where
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.where(~cond, np.nan)
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.where(~cond)
    rs2 = s.mask(cond)
    tm.assert_series_equal(rs, rs2)

    rs = s.where(~cond, -s)
    rs2 = s.mask(cond, -s)
    tm.assert_series_equal(rs, rs2)

    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())
    rs = s2.where(~cond[:3])
    rs2 = s2.mask(cond[:3])
    tm.assert_series_equal(rs, rs2)

    rs = s2.where(~cond[:3], -s2)
    rs2 = s2.mask(cond[:3], -s2)
    tm.assert_series_equal(rs, rs2)

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.mask(1)
    with pytest.raises(ValueError, match=msg):
        s.mask(cond[:3].values, -s)


def test_mask_casts():
    # dtype changes
    ser = Series([1, 2, 3, 4])
    result = ser.mask(ser > 2, np.nan)
    expected = Series([1, 2, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_mask_casts2():
    # see gh-21891
    ser = Series([1, 2])
    res = ser.mask([True, False])

    exp = Series([np.nan, 2])
    tm.assert_series_equal(res, exp)


def test_mask_inplace():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.copy()
    rs.mask(cond, inplace=True)
    tm.assert_series_equal(rs.dropna(), s[~cond])
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.copy()
    rs.mask(cond, -s, inplace=True)
    tm.assert_series_equal(rs, s.mask(cond, -s))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\frequencies\test_freq_code.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import (
    Period,
    to_offset,
)


@pytest.mark.parametrize(
    "freqstr,exp_freqstr",
    [("D", "D"), ("W", "D"), ("ME", "D"), ("s", "s"), ("min", "s"), ("h", "s")],
)
def test_get_to_timestamp_base(freqstr, exp_freqstr):
    off = to_offset(freqstr)
    per = Period._from_ordinal(1, off)
    exp_code = to_offset(exp_freqstr)._period_dtype_code

    result_code = per._dtype._get_to_timestamp_base()
    assert result_code == exp_code


@pytest.mark.parametrize(
    "args,expected",
    [
        ((1.5, "min"), (90, "s")),
        ((62.4, "min"), (3744, "s")),
        ((1.04, "h"), (3744, "s")),
        ((1, "D"), (1, "D")),
        ((0.342931, "h"), (1234551600, "us")),
        ((1.2345, "D"), (106660800, "ms")),
    ],
)
def test_resolution_bumping(args, expected):
    # see gh-14378
    off = to_offset(str(args[0]) + args[1])
    assert off.n == expected[0]
    assert off._prefix == expected[1]


@pytest.mark.parametrize(
    "args",
    [
        (0.5, "ns"),
        # Too much precision in the input can prevent.
        (0.3429324798798269273987982, "h"),
    ],
)
def test_cat(args):
    msg = "Invalid frequency"

    with pytest.raises(ValueError, match=msg):
        to_offset(str(args[0]) + args[1])


@pytest.mark.parametrize(
    "freqstr,expected",
    [
        ("1h", "2021-01-01T09:00:00"),
        ("1D", "2021-01-02T08:00:00"),
        ("1W", "2021-01-03T08:00:00"),
        ("1ME", "2021-01-31T08:00:00"),
        ("1YE", "2021-12-31T08:00:00"),
    ],
)
def test_compatibility(freqstr, expected):
    ts_np = np.datetime64("2021-01-01T08:00:00.00")
    do = to_offset(freqstr)
    assert ts_np + do == np.datetime64(expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_validate_kwargs.py ===
import pytest

from pandas.util._validators import (
    validate_bool_kwarg,
    validate_kwargs,
)


@pytest.fixture
def _fname():
    return "func"


def test_bad_kwarg(_fname):
    good_arg = "f"
    bad_arg = good_arg + "o"

    compat_args = {good_arg: "foo", bad_arg + "o": "bar"}
    kwargs = {good_arg: "foo", bad_arg: "bar"}

    msg = rf"{_fname}\(\) got an unexpected keyword argument '{bad_arg}'"

    with pytest.raises(TypeError, match=msg):
        validate_kwargs(_fname, kwargs, compat_args)


@pytest.mark.parametrize("i", range(1, 3))
def test_not_all_none(i, _fname):
    bad_arg = "foo"
    msg = (
        rf"the '{bad_arg}' parameter is not supported "
        rf"in the pandas implementation of {_fname}\(\)"
    )

    compat_args = {"foo": 1, "bar": "s", "baz": None}

    kwarg_keys = ("foo", "bar", "baz")
    kwarg_vals = (2, "s", None)

    kwargs = dict(zip(kwarg_keys[:i], kwarg_vals[:i]))

    with pytest.raises(ValueError, match=msg):
        validate_kwargs(_fname, kwargs, compat_args)


def test_validation(_fname):
    # No exceptions should be raised.
    compat_args = {"f": None, "b": 1, "ba": "s"}

    kwargs = {"f": None, "b": 1}
    validate_kwargs(_fname, kwargs, compat_args)


@pytest.mark.parametrize("name", ["inplace", "copy"])
@pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
def test_validate_bool_kwarg_fail(name, value):
    msg = (
        f'For argument "{name}" expected type bool, '
        f"received type {type(value).__name__}"
    )

    with pytest.raises(ValueError, match=msg):
        validate_bool_kwarg(value, name)


@pytest.mark.parametrize("name", ["inplace", "copy"])
@pytest.mark.parametrize("value", [True, False, None])
def test_validate_bool_kwarg(name, value):
    assert validate_bool_kwarg(value, name) == value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_numpy_nodes.py ===
from itertools import product
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.printing.repr import srepr
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2, minimum, maximum, amax, amin
from sympy.testing.pytest import raises

x, y, z = symbols('x y z')

def test_logaddexp():
    lae_xy = logaddexp(x, y)
    ref_xy = log(exp(x) + exp(y))
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert (
            lae_xy.diff(wrt, deriv_order) -
            ref_xy.diff(wrt, deriv_order)
        ).rewrite(log).simplify() == 0

    one_third_e = 1*exp(1)/3
    two_thirds_e = 2*exp(1)/3
    logThirdE = log(one_third_e)
    logTwoThirdsE = log(two_thirds_e)
    lae_sum_to_e = logaddexp(logThirdE, logTwoThirdsE)
    assert lae_sum_to_e.rewrite(log) == 1
    assert lae_sum_to_e.simplify() == 1
    was = logaddexp(2, 3)
    assert srepr(was) == srepr(was.simplify())  # cannot simplify with 2, 3


def test_logaddexp2():
    lae2_xy = logaddexp2(x, y)
    ref2_xy = log(2**x + 2**y)/log(2)
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert (
            lae2_xy.diff(wrt, deriv_order) -
            ref2_xy.diff(wrt, deriv_order)
        ).rewrite(log).cancel() == 0

    def lb(x):
        return log(x)/log(2)

    two_thirds = S.One*2/3
    four_thirds = 2*two_thirds
    lbTwoThirds = lb(two_thirds)
    lbFourThirds = lb(four_thirds)
    lae2_sum_to_2 = logaddexp2(lbTwoThirds, lbFourThirds)
    assert lae2_sum_to_2.rewrite(log) == 1
    assert lae2_sum_to_2.simplify() == 1
    was = logaddexp2(x, y)
    assert srepr(was) == srepr(was.simplify())  # cannot simplify with x, y


def test_minimum_maximum():
    for MM, mm in zip([Min, Max], [minimum, maximum]):
        ref = MM(x, y, z)
        m = mm(x, y, z)
        assert m != ref
        assert m.rewrite(MM) == ref


def test_amin_amax():
    for am in [amin, amax]:
        assert am(x).array == x
        assert am(x).axis == None
        assert am(x, axis=3).axis == 3
        with raises(ValueError):
            am(x, y, z)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_inverse.py ===
from sympy.core import symbols, S
from sympy.matrices.expressions import MatrixSymbol, Inverse, MatPow, ZeroMatrix, OneMatrix
from sympy.matrices.exceptions import NonInvertibleMatrixError, NonSquareMatrixError
from sympy.matrices import eye, Identity
from sympy.testing.pytest import raises
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine

n, m, l = symbols('n m l', integer=True)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)


def test_inverse():
    assert Inverse(C).args == (C, S.NegativeOne)
    assert Inverse(C).shape == (n, n)
    assert Inverse(A*E).shape == (n, n)
    assert Inverse(E*A).shape == (m, m)
    assert Inverse(C).inverse() == C
    assert Inverse(Inverse(C)).doit() == C
    assert isinstance(Inverse(Inverse(C)), Inverse)

    assert Inverse(*Inverse(E*A).args) == Inverse(E*A)

    assert C.inverse().inverse() == C

    assert C.inverse()*C == Identity(C.rows)

    assert Identity(n).inverse() == Identity(n)
    assert (3*Identity(n)).inverse() == Identity(n)/3

    # Simplifies Muls if possible (i.e. submatrices are square)
    assert (C*D).inverse() == D.I*C.I
    # But still works when not possible
    assert isinstance((A*E).inverse(), Inverse)
    assert Inverse(C*D).doit(inv_expand=False) == Inverse(C*D)

    assert Inverse(eye(3)).doit() == eye(3)
    assert Inverse(eye(3)).doit(deep=False) == eye(3)

    assert OneMatrix(1, 1).I == Identity(1)
    assert isinstance(OneMatrix(n, n).I, Inverse)

def test_inverse_non_invertible():
    raises(NonInvertibleMatrixError, lambda: ZeroMatrix(n, n).I)
    raises(NonInvertibleMatrixError, lambda: OneMatrix(2, 2).I)

def test_refine():
    assert refine(C.I, Q.orthogonal(C)) == C.T


def test_inverse_matpow_canonicalization():
    A = MatrixSymbol('A', 3, 3)
    assert Inverse(MatPow(A, 3)).doit() == MatPow(Inverse(A), 3).doit()


def test_nonsquare_error():
    A = MatrixSymbol('A', 3, 4)
    raises(NonSquareMatrixError, lambda: Inverse(A))


def test_adjoint_trnaspose_conjugate():
    A = MatrixSymbol('A', n, n)
    assert A.transpose().inverse() == A.inverse().transpose()
    assert A.conjugate().inverse() == A.inverse().conjugate()
    assert A.adjoint().inverse() == A.inverse().adjoint()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_transpose.py ===
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices.expressions import MatrixSymbol, Adjoint, trace, Transpose
from sympy.matrices import eye, Matrix
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.singleton import S
from sympy.core.symbol import symbols

n, m, l, k, p = symbols('n m l k p', integer=True)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)


def test_transpose():
    Sq = MatrixSymbol('Sq', n, n)

    assert transpose(A) == Transpose(A)
    assert Transpose(A).shape == (m, n)
    assert Transpose(A*B).shape == (l, n)
    assert transpose(Transpose(A)) == A
    assert isinstance(Transpose(Transpose(A)), Transpose)

    assert adjoint(Transpose(A)) == Adjoint(Transpose(A))
    assert conjugate(Transpose(A)) == Adjoint(A)

    assert Transpose(eye(3)).doit() == eye(3)

    assert Transpose(S(5)).doit() == S(5)

    assert Transpose(Matrix([[1, 2], [3, 4]])).doit() == Matrix([[1, 3], [2, 4]])

    assert transpose(trace(Sq)) == trace(Sq)
    assert trace(Transpose(Sq)) == trace(Sq)

    assert Transpose(Sq)[0, 1] == Sq[1, 0]

    assert Transpose(A*B).doit() == Transpose(B) * Transpose(A)


def test_transpose_MatAdd_MatMul():
    # Issue 16807
    from sympy.functions.elementary.trigonometric import cos

    x = symbols('x')
    M = MatrixSymbol('M', 3, 3)
    N = MatrixSymbol('N', 3, 3)

    assert (N + (cos(x) * M)).T == cos(x)*M.T + N.T


def test_refine():
    assert refine(C.T, Q.symmetric(C)) == C


def test_transpose1x1():
    m = MatrixSymbol('m', 1, 1)
    assert m == refine(m.T)
    assert m == refine(m.T.T)

def test_issue_9817():
    from sympy.matrices.expressions import Identity
    v = MatrixSymbol('v', 3, 1)
    A = MatrixSymbol('A', 3, 3)
    x = Matrix([i + 1 for i in range(3)])
    X = Identity(3)
    quadratic = v.T * A * v
    subbed = quadratic.xreplace({v:x, A:X})
    assert subbed.as_explicit() == Matrix([[14]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_custom_latex.py ===
import os
import tempfile
from pathlib import Path

import sympy
from sympy.testing.pytest import raises
from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark
from sympy.external import import_module

lark = import_module("lark")

# disable tests if lark is not present
disabled = lark is None

grammar_file = os.path.join(os.path.dirname(__file__), "../latex/lark/grammar/latex.lark")

modification1 = """
%override DIV_SYMBOL: DIV
%override MUL_SYMBOL: MUL | CMD_TIMES
"""

modification2 = r"""
%override number: /\d+(,\d*)?/
"""

def init_custom_parser(modification, transformer=None):
    latex_grammar = Path(grammar_file).read_text(encoding="utf-8")
    latex_grammar += modification

    with tempfile.NamedTemporaryFile() as f:
        f.write(bytes(latex_grammar, encoding="utf8"))
        f.flush()

        parser = LarkLaTeXParser(grammar_file=f.name, transformer=transformer)

    return parser

def test_custom1():
    # Removes the parser's ability to understand \cdot and \div.

    parser = init_custom_parser(modification1)

    with raises(lark.exceptions.UnexpectedCharacters):
        parser.doparse(r"a \cdot b")
        parser.doparse(r"x \div y")

class CustomTransformer(TransformToSymPyExpr):
    def number(self, tokens):
        if "," in tokens[0]:
            # The Float constructor expects a dot as the decimal separator
            return sympy.core.numbers.Float(tokens[0].replace(",", "."))
        else:
            return sympy.core.numbers.Integer(tokens[0])

def test_custom2():
    # Makes the parser parse commas as the decimal separator instead of dots

    parser = init_custom_parser(modification2, CustomTransformer)

    with raises(lark.exceptions.UnexpectedCharacters):
        # Asserting that the default parser cannot parse numbers which have commas as
        # the decimal separator
        parse_latex_lark("100,1")
        parse_latex_lark("0,009")

    parser.doparse("100,1")
    parser.doparse("0,009")
    parser.doparse("2,71828")
    parser.doparse("3,14159")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_circuitplot.py ===
from sympy.physics.quantum.circuitplot import labeller, render_label, Mz, CreateOneQubitGate,\
     CreateCGate
from sympy.physics.quantum.gate import CNOT, H, SWAP, CGate, S, T
from sympy.external import import_module
from sympy.testing.pytest import skip

mpl = import_module('matplotlib')

def test_render_label():
    assert render_label('q0') == r'$\left|q0\right\rangle$'
    assert render_label('q0', {'q0': '0'}) == r'$\left|q0\right\rangle=\left|0\right\rangle$'

def test_Mz():
    assert str(Mz(0)) == 'Mz(0)'

def test_create1():
    Qgate = CreateOneQubitGate('Q')
    assert str(Qgate(0)) == 'Q(0)'

def test_createc():
    Qgate = CreateCGate('Q')
    assert str(Qgate([1],0)) == 'C((1),Q(0))'

def test_labeller():
    """Test the labeller utility"""
    assert labeller(2) == ['q_1', 'q_0']
    assert labeller(3,'j') == ['j_2', 'j_1', 'j_0']

def test_cnot():
    """Test a simple cnot circuit. Right now this only makes sure the code doesn't
    raise an exception, and some simple properties
    """
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    c = CircuitPlot(CNOT(1,0),2,labels=labeller(2))
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == ['q_1', 'q_0']

    c = CircuitPlot(CNOT(1,0),2)
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == []

def test_ex1():
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    c = CircuitPlot(CNOT(1,0)*H(1),2,labels=labeller(2))
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == ['q_1', 'q_0']

def test_ex4():
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    c = CircuitPlot(SWAP(0,2)*H(0)* CGate((0,),S(1)) *H(1)*CGate((0,),T(2))\
                    *CGate((1,),S(2))*H(2),3,labels=labeller(3,'j'))
    assert c.ngates == 7
    assert c.nqubits == 3
    assert c.labels == ['j_2', 'j_1', 'j_0']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\conftest.py ===
import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.integer import (
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
)


@pytest.fixture(
    params=[
        Int8Dtype,
        Int16Dtype,
        Int32Dtype,
        Int64Dtype,
        UInt8Dtype,
        UInt16Dtype,
        UInt32Dtype,
        UInt64Dtype,
    ]
)
def dtype(request):
    """Parametrized fixture returning integer 'dtype'"""
    return request.param()


@pytest.fixture
def data(dtype):
    """
    Fixture returning 'data' array with valid and missing values according to
    parametrized integer 'dtype'.

    Used to test dtype conversion with and without missing values.
    """
    return pd.array(
        list(range(8)) + [np.nan] + list(range(10, 98)) + [np.nan] + [99, 100],
        dtype=dtype,
    )


@pytest.fixture
def data_missing(dtype):
    """
    Fixture returning array with exactly one NaN and one valid integer,
    according to parametrized integer 'dtype'.

    Used to test dtype conversion with and without missing values.
    """
    return pd.array([np.nan, 1], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_tz_localize.py ===
from datetime import timezone

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestTZLocalize:
    # See also:
    # test_tz_convert_and_localize in test_tz_convert

    def test_tz_localize(self, frame_or_series):
        rng = date_range("1/1/2011", periods=100, freq="h")

        obj = DataFrame({"a": 1}, index=rng)
        obj = tm.get_obj(obj, frame_or_series)

        result = obj.tz_localize("utc")
        expected = DataFrame({"a": 1}, rng.tz_localize("UTC"))
        expected = tm.get_obj(expected, frame_or_series)

        assert result.index.tz is timezone.utc
        tm.assert_equal(result, expected)

    def test_tz_localize_axis1(self):
        rng = date_range("1/1/2011", periods=100, freq="h")

        df = DataFrame({"a": 1}, index=rng)

        df = df.T
        result = df.tz_localize("utc", axis=1)
        assert result.columns.tz is timezone.utc

        expected = DataFrame({"a": 1}, rng.tz_localize("UTC"))

        tm.assert_frame_equal(result, expected.T)

    def test_tz_localize_naive(self, frame_or_series):
        # Can't localize if already tz-aware
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")
        ts = Series(1, index=rng)
        ts = frame_or_series(ts)

        with pytest.raises(TypeError, match="Already tz-aware"):
            ts.tz_localize("US/Eastern")

    @pytest.mark.parametrize("copy", [True, False])
    def test_tz_localize_copy_inplace_mutate(self, copy, frame_or_series):
        # GH#6326
        obj = frame_or_series(
            np.arange(0, 5), index=date_range("20131027", periods=5, freq="1h", tz=None)
        )
        orig = obj.copy()
        result = obj.tz_localize("UTC", copy=copy)
        expected = frame_or_series(
            np.arange(0, 5),
            index=date_range("20131027", periods=5, freq="1h", tz="UTC"),
        )
        tm.assert_equal(result, expected)
        tm.assert_equal(obj, orig)
        assert result.index is not obj.index
        assert result is not obj

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_operatorset.py ===
from sympy.core.singleton import S

from sympy.physics.quantum.operatorset import (
    operators_to_state, state_to_operators
)

from sympy.physics.quantum.cartesian import (
    XOp, XKet, PxOp, PxKet, XBra, PxBra
)

from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.spin import (
    JxKet, JyKet, JzKet, JxBra, JyBra, JzBra,
    JxOp, JyOp, JzOp, J2Op
)

from sympy.testing.pytest import raises


def test_spin():
    assert operators_to_state({J2Op, JxOp}) == JxKet
    assert operators_to_state({J2Op, JyOp}) == JyKet
    assert operators_to_state({J2Op, JzOp}) == JzKet
    assert operators_to_state({J2Op(), JxOp()}) == JxKet
    assert operators_to_state({J2Op(), JyOp()}) == JyKet
    assert operators_to_state({J2Op(), JzOp()}) == JzKet

    assert state_to_operators(JxKet) == {J2Op, JxOp}
    assert state_to_operators(JyKet) == {J2Op, JyOp}
    assert state_to_operators(JzKet) == {J2Op, JzOp}
    assert state_to_operators(JxBra) == {J2Op, JxOp}
    assert state_to_operators(JyBra) == {J2Op, JyOp}
    assert state_to_operators(JzBra) == {J2Op, JzOp}

    assert state_to_operators(JxKet(S.Half, S.Half)) == {J2Op(), JxOp()}
    assert state_to_operators(JyKet(S.Half, S.Half)) == {J2Op(), JyOp()}
    assert state_to_operators(JzKet(S.Half, S.Half)) == {J2Op(), JzOp()}
    assert state_to_operators(JxBra(S.Half, S.Half)) == {J2Op(), JxOp()}
    assert state_to_operators(JyBra(S.Half, S.Half)) == {J2Op(), JyOp()}
    assert state_to_operators(JzBra(S.Half, S.Half)) == {J2Op(), JzOp()}


def test_op_to_state():
    assert operators_to_state(XOp) == XKet()
    assert operators_to_state(PxOp) == PxKet()
    assert operators_to_state(Operator) == Ket()

    assert state_to_operators(operators_to_state(XOp("Q"))) == XOp("Q")
    assert state_to_operators(operators_to_state(XOp())) == XOp()

    raises(NotImplementedError, lambda: operators_to_state(XKet))


def test_state_to_op():
    assert state_to_operators(XKet) == XOp()
    assert state_to_operators(PxKet) == PxOp()
    assert state_to_operators(XBra) == XOp()
    assert state_to_operators(PxBra) == PxOp()
    assert state_to_operators(Ket) == Operator()
    assert state_to_operators(Bra) == Operator()

    assert operators_to_state(state_to_operators(XKet("test"))) == XKet("test")
    assert operators_to_state(state_to_operators(XBra("test"))) == XKet("test")
    assert operators_to_state(state_to_operators(XKet())) == XKet()
    assert operators_to_state(state_to_operators(XBra())) == XKet()

    raises(NotImplementedError, lambda: state_to_operators(XOp))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_return_complex.py ===
import pytest

from numpy import array

from . import util


@pytest.mark.slow
class TestReturnComplex(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t8", "s0", "s8"]:
            err = 1e-5
        else:
            err = 0.0
        assert abs(t(234j) - 234.0j) <= err
        assert abs(t(234.6) - 234.6) <= err
        assert abs(t(234) - 234.0) <= err
        assert abs(t(234.6 + 3j) - (234.6 + 3j)) <= err
        # assert abs(t('234')-234.)<=err
        # assert abs(t('234.6')-234.6)<=err
        assert abs(t(-234) + 234.0) <= err
        assert abs(t([234]) - 234.0) <= err
        assert abs(t((234, )) - 234.0) <= err
        assert abs(t(array(234)) - 234.0) <= err
        assert abs(t(array(23 + 4j, "F")) - (23 + 4j)) <= err
        assert abs(t(array([234])) - 234.0) <= err
        assert abs(t(array([[234]])) - 234.0) <= err
        assert abs(t(array([234]).astype("b")) + 22.0) <= err
        assert abs(t(array([234], "h")) - 234.0) <= err
        assert abs(t(array([234], "i")) - 234.0) <= err
        assert abs(t(array([234], "l")) - 234.0) <= err
        assert abs(t(array([234], "q")) - 234.0) <= err
        assert abs(t(array([234], "f")) - 234.0) <= err
        assert abs(t(array([234], "d")) - 234.0) <= err
        assert abs(t(array([234 + 3j], "F")) - (234 + 3j)) <= err
        assert abs(t(array([234], "D")) - 234.0) <= err

        # pytest.raises(TypeError, t, array([234], 'S1'))
        pytest.raises(TypeError, t, "abc")

        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        pytest.raises(TypeError, t, t)
        pytest.raises(TypeError, t, {})

        try:
            r = t(10**400)
            assert repr(r) in ["(inf+0j)", "(Infinity+0j)"]
        except OverflowError:
            pass


class TestFReturnComplex(TestReturnComplex):
    sources = [
        util.getpath("tests", "src", "return_complex", "foo77.f"),
        util.getpath("tests", "src", "return_complex", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name", ["t0", "t8", "t16", "td", "s0", "s8", "s16", "sd"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name", ["t0", "t8", "t16", "td", "s0", "s8", "s16", "sd"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_complex, name),
                            name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_repr.py ===
import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.integer import (
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
)


def test_dtypes(dtype):
    # smoke tests on auto dtype construction

    if dtype.is_signed_integer:
        assert np.dtype(dtype.type).kind == "i"
    else:
        assert np.dtype(dtype.type).kind == "u"
    assert dtype.name is not None


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (Int8Dtype(), "Int8Dtype()"),
        (Int16Dtype(), "Int16Dtype()"),
        (Int32Dtype(), "Int32Dtype()"),
        (Int64Dtype(), "Int64Dtype()"),
        (UInt8Dtype(), "UInt8Dtype()"),
        (UInt16Dtype(), "UInt16Dtype()"),
        (UInt32Dtype(), "UInt32Dtype()"),
        (UInt64Dtype(), "UInt64Dtype()"),
    ],
)
def test_repr_dtype(dtype, expected):
    assert repr(dtype) == expected


def test_repr_array():
    result = repr(pd.array([1, None, 3]))
    expected = "<IntegerArray>\n[1, <NA>, 3]\nLength: 3, dtype: Int64"
    assert result == expected


def test_repr_array_long():
    data = pd.array([1, 2, None] * 1000)
    expected = (
        "<IntegerArray>\n"
        "[   1,    2, <NA>,    1,    2, <NA>,    1,    2, <NA>,    1,\n"
        " ...\n"
        " <NA>,    1,    2, <NA>,    1,    2, <NA>,    1,    2, <NA>]\n"
        "Length: 3000, dtype: Int64"
    )
    result = repr(data)
    assert result == expected


def test_frame_repr(data_missing):
    df = pd.DataFrame({"A": data_missing})
    result = repr(df)
    expected = "      A\n0  <NA>\n1     1"
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\period\test_astype.py ===
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import period_array


@pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
def test_astype_int(dtype):
    # We choose to ignore the sign and size of integers for
    # Period/Datetime/Timedelta astype
    arr = period_array(["2000", "2001", None], freq="D")

    if np.dtype(dtype) != np.int64:
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype(dtype)
        return

    result = arr.astype(dtype)
    expected = arr._ndarray.view("i8")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_copies():
    arr = period_array(["2000", "2001", None], freq="D")
    result = arr.astype(np.int64, copy=False)

    # Add the `.base`, since we now use `.asi8` which returns a view.
    # We could maybe override it in PeriodArray to return ._ndarray directly.
    assert result.base is arr._ndarray

    result = arr.astype(np.int64, copy=True)
    assert result is not arr._ndarray
    tm.assert_numpy_array_equal(result, arr._ndarray.view("i8"))


def test_astype_categorical():
    arr = period_array(["2000", "2001", "2001", None], freq="D")
    result = arr.astype("category")
    categories = pd.PeriodIndex(["2000", "2001"], freq="D")
    expected = pd.Categorical.from_codes([0, 1, 1, -1], categories=categories)
    tm.assert_categorical_equal(result, expected)


def test_astype_period():
    arr = period_array(["2000", "2001", None], freq="D")
    result = arr.astype(PeriodDtype("M"))
    expected = period_array(["2000", "2001", None], freq="M")
    tm.assert_period_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_astype_datetime(dtype):
    arr = period_array(["2000", "2001", None], freq="D")
    # slice off the [ns] so that the regex matches.
    if dtype == "timedelta64[ns]":
        with pytest.raises(TypeError, match=dtype[:-4]):
            arr.astype(dtype)

    else:
        # GH#45038 allow period->dt64 because we allow dt64->period
        result = arr.astype(dtype)
        expected = pd.DatetimeIndex(["2000", "2001", pd.NaT], dtype=dtype)._data
        tm.assert_datetime_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\interval\test_overlaps.py ===
import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)


@pytest.fixture(
    params=[
        (Timedelta("0 days"), Timedelta("1 day")),
        (Timestamp("2018-01-01"), Timedelta("1 day")),
        (0, 1),
    ],
    ids=lambda x: type(x[0]).__name__,
)
def start_shift(request):
    """
    Fixture for generating intervals of types from a start value and a shift
    value that can be added to start to generate an endpoint
    """
    return request.param


class TestOverlaps:
    def test_overlaps_self(self, start_shift, closed):
        start, shift = start_shift
        interval = Interval(start, start + shift, closed)
        assert interval.overlaps(interval)

    def test_overlaps_nested(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + 3 * shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # nested intervals should always overlap
        assert interval1.overlaps(interval2)

    def test_overlaps_disjoint(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + 2 * shift, start + 3 * shift, closed)

        # disjoint intervals should never overlap
        assert not interval1.overlaps(interval2)

    def test_overlaps_endpoint(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # overlap if shared endpoint is closed for both (overlap at a point)
        result = interval1.overlaps(interval2)
        expected = interval1.closed_right and interval2.closed_left
        assert result == expected

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(self, other):
        interval = Interval(0, 1)
        msg = f"`other` must be an Interval, got {type(other).__name__}"
        with pytest.raises(TypeError, match=msg):
            interval.overlaps(other)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_truncate.py ===
from datetime import datetime

import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm


class TestTruncate:
    def test_truncate_datetimeindex_tz(self):
        # GH 9243
        idx = date_range("4/1/2005", "4/30/2005", freq="D", tz="US/Pacific")
        s = Series(range(len(idx)), index=idx)
        with pytest.raises(TypeError, match="Cannot compare tz-naive"):
            # GH#36148 as of 2.0 we require tzawareness compat
            s.truncate(datetime(2005, 4, 2), datetime(2005, 4, 4))

        lb = idx[1]
        ub = idx[3]
        result = s.truncate(lb.to_pydatetime(), ub.to_pydatetime())
        expected = Series([1, 2, 3], index=idx[1:4])
        tm.assert_series_equal(result, expected)

    def test_truncate_periodindex(self):
        # GH 17717
        idx1 = pd.PeriodIndex(
            [pd.Period("2017-09-02"), pd.Period("2017-09-02"), pd.Period("2017-09-03")]
        )
        series1 = Series([1, 2, 3], index=idx1)
        result1 = series1.truncate(after="2017-09-02")

        expected_idx1 = pd.PeriodIndex(
            [pd.Period("2017-09-02"), pd.Period("2017-09-02")]
        )
        tm.assert_series_equal(result1, Series([1, 2], index=expected_idx1))

        idx2 = pd.PeriodIndex(
            [pd.Period("2017-09-03"), pd.Period("2017-09-02"), pd.Period("2017-09-03")]
        )
        series2 = Series([1, 2, 3], index=idx2)
        result2 = series2.sort_index().truncate(after="2017-09-02")

        expected_idx2 = pd.PeriodIndex([pd.Period("2017-09-02")])
        tm.assert_series_equal(result2, Series([2], index=expected_idx2))

    def test_truncate_one_element_series(self):
        # GH 35544
        series = Series([0.1], index=pd.DatetimeIndex(["2020-08-04"]))
        before = pd.Timestamp("2020-08-02")
        after = pd.Timestamp("2020-08-04")

        result = series.truncate(before=before, after=after)

        # the input Series and the expected Series are the same
        tm.assert_series_equal(result, series)

    def test_truncate_index_only_one_unique_value(self):
        # GH 42365
        obj = Series(0, index=date_range("2021-06-30", "2021-06-30")).repeat(5)

        truncated = obj.truncate("2021-06-28", "2021-07-01")

        tm.assert_series_equal(truncated, obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\tests\test_ordinals.py ===
from sympy.sets.ordinals import Ordinal, OmegaPower, ord0, omega
from sympy.testing.pytest import raises

def test_string_ordinals():
    assert str(omega) == 'w'
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(3, 2))) == 'w**5*3 + w**3*2'
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(0, 5))) == 'w**5*3 + 5'
    assert str(Ordinal(OmegaPower(1, 3), OmegaPower(0, 5))) == 'w*3 + 5'
    assert str(Ordinal(OmegaPower(omega + 1, 1), OmegaPower(3, 2))) == 'w**(w + 1) + w**3*2'

def test_addition_with_integers():
    assert 3 + Ordinal(OmegaPower(5, 3)) == Ordinal(OmegaPower(5, 3))
    assert Ordinal(OmegaPower(5, 3))+3 == Ordinal(OmegaPower(5, 3), OmegaPower(0, 3))
    assert Ordinal(OmegaPower(5, 3), OmegaPower(0, 2))+3 == \
        Ordinal(OmegaPower(5, 3), OmegaPower(0, 5))


def test_addition_with_ordinals():
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) + Ordinal(OmegaPower(3, 3)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(3, 5))
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) + Ordinal(OmegaPower(4, 2)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(4, 2))
    assert Ordinal(OmegaPower(omega, 2), OmegaPower(3, 2)) + Ordinal(OmegaPower(4, 2)) == \
        Ordinal(OmegaPower(omega, 2), OmegaPower(4, 2))

def test_comparison():
    assert Ordinal(OmegaPower(5, 3)) > Ordinal(OmegaPower(4, 3), OmegaPower(2, 1))
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) < Ordinal(OmegaPower(5, 4))
    assert Ordinal(OmegaPower(5, 4)) < Ordinal(OmegaPower(5, 5), OmegaPower(4, 1))

    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(3, 2))
    assert not Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) == Ordinal(OmegaPower(5, 3))
    assert Ordinal(OmegaPower(omega, 3)) > Ordinal(OmegaPower(5, 3))

def test_multiplication_with_integers():
    w = omega
    assert 3*w == w
    assert w*9 == Ordinal(OmegaPower(1, 9))

def test_multiplication():
    w = omega
    assert w*(w + 1) == w*w + w
    assert (w + 1)*(w + 1) ==  w*w + w + 1
    assert w*1 == w
    assert 1*w == w
    assert w*ord0 == ord0
    assert ord0*w == ord0
    assert w**w == w * w**w
    assert (w**w)*w*w == w**(w + 2)

def test_exponentiation():
    w = omega
    assert w**2 == w*w
    assert w**3 == w*w*w
    assert w**(w + 1) == Ordinal(OmegaPower(omega + 1, 1))
    assert (w**w)*(w**w) == w**(w*2)

def test_comapre_not_instance():
    w = OmegaPower(omega + 1, 1)
    assert(not (w == None))
    assert(not (w < 5))
    raises(TypeError, lambda: w < 6.66)

def test_is_successort():
    w = Ordinal(OmegaPower(5, 1))
    assert not w.is_successor_ordinal

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_spherical_harmonics.py ===
from sympy.core.function import diff
from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin)
from sympy.functions.special.spherical_harmonics import Ynm, Znm, Ynm_c


def test_Ynm():
    # https://en.wikipedia.org/wiki/Spherical_harmonics
    th, ph = Symbol("theta", real=True), Symbol("phi", real=True)
    from sympy.abc import n,m

    assert Ynm(0, 0, th, ph).expand(func=True) == 1/(2*sqrt(pi))
    assert Ynm(1, -1, th, ph) == -exp(-2*I*ph)*Ynm(1, 1, th, ph)
    assert Ynm(1, -1, th, ph).expand(func=True) == sqrt(6)*sin(th)*exp(-I*ph)/(4*sqrt(pi))
    assert Ynm(1, 0, th, ph).expand(func=True) == sqrt(3)*cos(th)/(2*sqrt(pi))
    assert Ynm(1, 1, th, ph).expand(func=True) == -sqrt(6)*sin(th)*exp(I*ph)/(4*sqrt(pi))
    assert Ynm(2, 0, th, ph).expand(func=True) == 3*sqrt(5)*cos(th)**2/(4*sqrt(pi)) - sqrt(5)/(4*sqrt(pi))
    assert Ynm(2, 1, th, ph).expand(func=True) == -sqrt(30)*sin(th)*exp(I*ph)*cos(th)/(4*sqrt(pi))
    assert Ynm(2, -2, th, ph).expand(func=True) == (-sqrt(30)*exp(-2*I*ph)*cos(th)**2/(8*sqrt(pi))
                                                    + sqrt(30)*exp(-2*I*ph)/(8*sqrt(pi)))
    assert Ynm(2, 2, th, ph).expand(func=True) == (-sqrt(30)*exp(2*I*ph)*cos(th)**2/(8*sqrt(pi))
                                                   + sqrt(30)*exp(2*I*ph)/(8*sqrt(pi)))

    assert diff(Ynm(n, m, th, ph), th) == (m*cot(th)*Ynm(n, m, th, ph)
                                           + sqrt((-m + n)*(m + n + 1))*exp(-I*ph)*Ynm(n, m + 1, th, ph))
    assert diff(Ynm(n, m, th, ph), ph) == I*m*Ynm(n, m, th, ph)

    assert conjugate(Ynm(n, m, th, ph)) == (-1)**(2*m)*exp(-2*I*m*ph)*Ynm(n, m, th, ph)

    assert Ynm(n, m, -th, ph) == Ynm(n, m, th, ph)
    assert Ynm(n, m, th, -ph) == exp(-2*I*m*ph)*Ynm(n, m, th, ph)
    assert Ynm(n, -m, th, ph) == (-1)**m*exp(-2*I*m*ph)*Ynm(n, m, th, ph)


def test_Ynm_c():
    th, ph = Symbol("theta", real=True), Symbol("phi", real=True)
    from sympy.abc import n,m

    assert Ynm_c(n, m, th, ph) == (-1)**(2*m)*exp(-2*I*m*ph)*Ynm(n, m, th, ph)


def test_Znm():
    # https://en.wikipedia.org/wiki/Solid_harmonics#List_of_lowest_functions
    th, ph = Symbol("theta", real=True), Symbol("phi", real=True)

    assert Znm(0, 0, th, ph) == Ynm(0, 0, th, ph)
    assert Znm(1, -1, th, ph) == (-sqrt(2)*I*(Ynm(1, 1, th, ph)
                                  - exp(-2*I*ph)*Ynm(1, 1, th, ph))/2)
    assert Znm(1, 0, th, ph) == Ynm(1, 0, th, ph)
    assert Znm(1, 1, th, ph) == (sqrt(2)*(Ynm(1, 1, th, ph)
                                 + exp(-2*I*ph)*Ynm(1, 1, th, ph))/2)
    assert Znm(0, 0, th, ph).expand(func=True) == 1/(2*sqrt(pi))
    assert Znm(1, -1, th, ph).expand(func=True) == (sqrt(3)*I*sin(th)*exp(I*ph)/(4*sqrt(pi))
                                                    - sqrt(3)*I*sin(th)*exp(-I*ph)/(4*sqrt(pi)))
    assert Znm(1, 0, th, ph).expand(func=True) == sqrt(3)*cos(th)/(2*sqrt(pi))
    assert Znm(1, 1, th, ph).expand(func=True) == (-sqrt(3)*sin(th)*exp(I*ph)/(4*sqrt(pi))
                                                   - sqrt(3)*sin(th)*exp(-I*ph)/(4*sqrt(pi)))
    assert Znm(2, -1, th, ph).expand(func=True) == (sqrt(15)*I*sin(th)*exp(I*ph)*cos(th)/(4*sqrt(pi))
                                                    - sqrt(15)*I*sin(th)*exp(-I*ph)*cos(th)/(4*sqrt(pi)))
    assert Znm(2, 0, th, ph).expand(func=True) == 3*sqrt(5)*cos(th)**2/(4*sqrt(pi)) - sqrt(5)/(4*sqrt(pi))
    assert Znm(2, 1, th, ph).expand(func=True) == (-sqrt(15)*sin(th)*exp(I*ph)*cos(th)/(4*sqrt(pi))
                                                   - sqrt(15)*sin(th)*exp(-I*ph)*cos(th)/(4*sqrt(pi)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_return_logical.py ===
import pytest

from numpy import array

from . import util


class TestReturnLogical(util.F2PyTest):
    def check_function(self, t):
        assert t(True) == 1
        assert t(False) == 0
        assert t(0) == 0
        assert t(None) == 0
        assert t(0.0) == 0
        assert t(0j) == 0
        assert t(1j) == 1
        assert t(234) == 1
        assert t(234.6) == 1
        assert t(234.6 + 3j) == 1
        assert t("234") == 1
        assert t("aaa") == 1
        assert t("") == 0
        assert t([]) == 0
        assert t(()) == 0
        assert t({}) == 0
        assert t(t) == 1
        assert t(-234) == 1
        assert t(10**100) == 1
        assert t([234]) == 1
        assert t((234, )) == 1
        assert t(array(234)) == 1
        assert t(array([234])) == 1
        assert t(array([[234]])) == 1
        assert t(array([127], "b")) == 1
        assert t(array([234], "h")) == 1
        assert t(array([234], "i")) == 1
        assert t(array([234], "l")) == 1
        assert t(array([234], "f")) == 1
        assert t(array([234], "d")) == 1
        assert t(array([234 + 3j], "F")) == 1
        assert t(array([234], "D")) == 1
        assert t(array(0)) == 0
        assert t(array([0])) == 0
        assert t(array([[0]])) == 0
        assert t(array([0j])) == 0
        assert t(array([1])) == 1
        pytest.raises(ValueError, t, array([0, 0]))


class TestFReturnLogical(TestReturnLogical):
    sources = [
        util.getpath("tests", "src", "return_logical", "foo77.f"),
        util.getpath("tests", "src", "return_logical", "foo90.f90"),
    ]

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["t0", "t1", "t2", "t4", "s0", "s1", "s2", "s4"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name))

    @pytest.mark.slow
    @pytest.mark.parametrize("name",
                             ["t0", "t1", "t2", "t4", "t8", "s0", "s1", "s2", "s4", "s8"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_logical, name))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_comparison.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.tests.arrays.masked_shared import (
    ComparisonOps,
    NumericOps,
)


class TestComparisonOps(NumericOps, ComparisonOps):
    @pytest.mark.parametrize("other", [True, False, pd.NA, -1.0, 0.0, 1])
    def test_scalar(self, other, comparison_op, dtype):
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_compare_with_integerarray(self, comparison_op):
        op = comparison_op
        a = pd.array([0, 1, None] * 3, dtype="Int64")
        b = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype="Float64")
        other = b.astype("Int64")
        expected = op(a, other)
        result = op(a, b)
        tm.assert_extension_array_equal(result, expected)
        expected = op(other, a)
        result = op(b, a)
        tm.assert_extension_array_equal(result, expected)


def test_equals():
    # GH-30652
    # equals is generally tested in /tests/extension/base/methods, but this
    # specifically tests that two arrays of the same class but different dtype
    # do not evaluate equal
    a1 = pd.array([1, 2, None], dtype="Float64")
    a2 = pd.array([1, 2, None], dtype="Float32")
    assert a1.equals(a2) is False


def test_equals_nan_vs_na():
    # GH#44382

    mask = np.zeros(3, dtype=bool)
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)

    left = FloatingArray(data, mask)
    assert left.equals(left)
    tm.assert_extension_array_equal(left, left)

    assert left.equals(left.copy())
    assert left.equals(FloatingArray(data.copy(), mask.copy()))

    mask2 = np.array([False, True, False], dtype=bool)
    data2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    right = FloatingArray(data2, mask2)
    assert right.equals(right)
    tm.assert_extension_array_equal(right, right)

    assert not left.equals(right)

    # with mask[1] = True, the only difference is data[1], which should
    #  not matter for equals
    mask[1] = True
    assert left.equals(right)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_bin_groupby.py ===
import numpy as np
import pytest

from pandas._libs import lib
import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


def assert_block_lengths(x):
    assert len(x) == len(x._mgr.blocks[0].mgr_locs)
    return 0


def cumsum_max(x):
    x.cumsum().max()
    return 0


@pytest.mark.parametrize(
    "func",
    [
        cumsum_max,
        pytest.param(assert_block_lengths, marks=td.skip_array_manager_invalid_test),
    ],
)
def test_mgr_locs_updated(func):
    # https://github.com/pandas-dev/pandas/issues/31802
    # Some operations may require creating new blocks, which requires
    # valid mgr_locs
    df = pd.DataFrame({"A": ["a", "a", "a"], "B": ["a", "b", "b"], "C": [1, 1, 1]})
    result = df.groupby(["A", "B"]).agg(func)
    expected = pd.DataFrame(
        {"C": [0, 0]},
        index=pd.MultiIndex.from_product([["a"], ["a", "b"]], names=["A", "B"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "binner,closed,expected",
    [
        (
            np.array([0, 3, 6, 9], dtype=np.int64),
            "left",
            np.array([2, 5, 6], dtype=np.int64),
        ),
        (
            np.array([0, 3, 6, 9], dtype=np.int64),
            "right",
            np.array([3, 6, 6], dtype=np.int64),
        ),
        (np.array([0, 3, 6], dtype=np.int64), "left", np.array([2, 5], dtype=np.int64)),
        (
            np.array([0, 3, 6], dtype=np.int64),
            "right",
            np.array([3, 6], dtype=np.int64),
        ),
    ],
)
def test_generate_bins(binner, closed, expected):
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = lib.generate_bins_dt64(values, binner, closed=closed)
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_api.py ===
"""Tests that the tslibs API is locked down"""

from pandas._libs import tslibs


def test_namespace():
    submodules = [
        "base",
        "ccalendar",
        "conversion",
        "dtypes",
        "fields",
        "nattype",
        "np_datetime",
        "offsets",
        "parsing",
        "period",
        "strptime",
        "vectorized",
        "timedeltas",
        "timestamps",
        "timezones",
        "tzconversion",
    ]

    api = [
        "BaseOffset",
        "NaT",
        "NaTType",
        "iNaT",
        "nat_strings",
        "OutOfBoundsDatetime",
        "OutOfBoundsTimedelta",
        "Period",
        "IncompatibleFrequency",
        "Resolution",
        "Tick",
        "Timedelta",
        "dt64arr_to_periodarr",
        "Timestamp",
        "is_date_array_normalized",
        "ints_to_pydatetime",
        "normalize_i8_timestamps",
        "get_resolution",
        "delta_to_nanoseconds",
        "ints_to_pytimedelta",
        "localize_pydatetime",
        "tz_convert_from_utc",
        "tz_convert_from_utc_single",
        "to_offset",
        "tz_compare",
        "is_unitless",
        "astype_overflowsafe",
        "get_unit_from_dtype",
        "periods_per_day",
        "periods_per_second",
        "guess_datetime_format",
        "add_overflowsafe",
        "get_supported_dtype",
        "is_supported_dtype",
    ]

    expected = set(submodules + api)
    names = [x for x in dir(tslibs) if not x.startswith("__")]
    assert set(names) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_determinant.py ===
from sympy.core import S, symbols
from sympy.matrices import eye, ones, Matrix, ShapeError
from sympy.matrices.expressions import (
    Identity, MatrixExpr, MatrixSymbol, Determinant,
    det, per, ZeroMatrix, Transpose,
    Permanent, MatMul
)
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine

n = symbols('n', integer=True)
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)
C = MatrixSymbol('C', 3, 4)


def test_det():
    assert isinstance(Determinant(A), Determinant)
    assert not isinstance(Determinant(A), MatrixExpr)
    raises(ShapeError, lambda: Determinant(C))
    assert det(eye(3)) == 1
    assert det(Matrix(3, 3, [1, 3, 2, 4, 1, 3, 2, 5, 2])) == 17
    _ = A / det(A)  # Make sure this is possible

    raises(TypeError, lambda: Determinant(S.One))

    assert Determinant(A).arg is A


def test_eval_determinant():
    assert det(Identity(n)) == 1
    assert det(ZeroMatrix(n, n)) == 0
    assert det(OneMatrix(n, n)) == Determinant(OneMatrix(n, n))
    assert det(OneMatrix(1, 1)) == 1
    assert det(OneMatrix(2, 2)) == 0
    assert det(Transpose(A)) == det(A)
    assert Determinant(MatMul(eye(2), eye(2))).doit(deep=True) == 1


def test_refine():
    assert refine(det(A), Q.orthogonal(A)) == 1
    assert refine(det(A), Q.singular(A)) == 0
    assert refine(det(A), Q.unit_triangular(A)) == 1
    assert refine(det(A), Q.normal(A)) == det(A)


def test_commutative():
    det_a = Determinant(A)
    det_b = Determinant(B)
    assert det_a.is_commutative
    assert det_b.is_commutative
    assert det_a * det_b == det_b * det_a


def test_permanent():
    assert isinstance(Permanent(A), Permanent)
    assert not isinstance(Permanent(A), MatrixExpr)
    assert isinstance(Permanent(C), Permanent)
    assert Permanent(ones(3, 3)).doit() == 6
    _ = C / per(C)
    assert per(Matrix(3, 3, [1, 3, 2, 4, 1, 3, 2, 5, 2])) == 103
    raises(TypeError, lambda: Permanent(S.One))
    assert Permanent(A).arg is A

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_slice.py ===
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions import MatrixSymbol
from sympy.abc import a, b, c, d, k, l, m, n
from sympy.testing.pytest import raises, XFAIL
from sympy.functions.elementary.integers import floor
from sympy.assumptions import assuming, Q


X = MatrixSymbol('X', n, m)
Y = MatrixSymbol('Y', m, k)

def test_shape():
    B = MatrixSlice(X, (a, b), (c, d))
    assert B.shape == (b - a, d - c)

def test_entry():
    B = MatrixSlice(X, (a, b), (c, d))
    assert B[0,0] == X[a, c]
    assert B[k,l] == X[a+k, c+l]
    raises(IndexError, lambda : MatrixSlice(X, 1, (2, 5))[1, 0])

    assert X[1::2, :][1, 3] == X[1+2, 3]
    assert X[:, 1::2][3, 1] == X[3, 1+2]

def test_on_diag():
    assert not MatrixSlice(X, (a, b), (c, d)).on_diag
    assert MatrixSlice(X, (a, b), (a, b)).on_diag

def test_inputs():
    assert MatrixSlice(X, 1, (2, 5)) == MatrixSlice(X, (1, 2), (2, 5))
    assert MatrixSlice(X, 1, (2, 5)).shape == (1, 3)

def test_slicing():
    assert X[1:5, 2:4] == MatrixSlice(X, (1, 5), (2, 4))
    assert X[1, 2:4] == MatrixSlice(X, 1, (2, 4))
    assert X[1:5, :].shape == (4, X.shape[1])
    assert X[:, 1:5].shape == (X.shape[0], 4)

    assert X[::2, ::2].shape == (floor(n/2), floor(m/2))
    assert X[2, :] == MatrixSlice(X, 2, (0, m))
    assert X[k, :] == MatrixSlice(X, k, (0, m))

def test_exceptions():
    X = MatrixSymbol('x', 10, 20)
    raises(IndexError, lambda: X[0:12, 2])
    raises(IndexError, lambda: X[0:9, 22])
    raises(IndexError, lambda: X[-1:5, 2])

@XFAIL
def test_symmetry():
    X = MatrixSymbol('x', 10, 10)
    Y = X[:5, 5:]
    with assuming(Q.symmetric(X)):
        assert Y.T == X[5:, :5]

def test_slice_of_slice():
    X = MatrixSymbol('x', 10, 10)
    assert X[2, :][:, 3][0, 0] == X[2, 3]
    assert X[:5, :5][:4, :4] == X[:4, :4]
    assert X[1:5, 2:6][1:3, 2] == X[2:4, 4]
    assert X[1:9:2, 2:6][1:3, 2] == X[3:7:2, 4]

def test_negative_index():
    X = MatrixSymbol('x', 10, 10)
    assert X[-1, :] == X[9, :]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_lseries.py ===
from sympy.core.numbers import E
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import Order
from sympy.abc import x, y


def test_sin():
    e = sin(x).lseries(x)
    assert next(e) == x
    assert next(e) == -x**3/6
    assert next(e) == x**5/120


def test_cos():
    e = cos(x).lseries(x)
    assert next(e) == 1
    assert next(e) == -x**2/2
    assert next(e) == x**4/24


def test_exp():
    e = exp(x).lseries(x)
    assert next(e) == 1
    assert next(e) == x
    assert next(e) == x**2/2
    assert next(e) == x**3/6


def test_exp2():
    e = exp(cos(x)).lseries(x)
    assert next(e) == E
    assert next(e) == -E*x**2/2
    assert next(e) == E*x**4/6
    assert next(e) == -31*E*x**6/720


def test_simple():
    assert list(x.lseries()) == [x]
    assert list(S.One.lseries(x)) == [1]
    assert not next((x/(x + y)).lseries(y)).has(Order)


def test_issue_5183():
    s = (x + 1/x).lseries()
    assert list(s) == [1/x, x]
    assert next((x + x**2).lseries()) == x
    assert next(((1 + x)**7).lseries(x)) == 1
    assert next((sin(x + y)).series(x, n=3).lseries(y)) == x
    # it would be nice if all terms were grouped, but in the
    # following case that would mean that all the terms would have
    # to be known since, for example, every term has a constant in it.
    s = ((1 + x)**7).series(x, 1, n=None)
    assert [next(s) for i in range(2)] == [128, -448 + 448*x]


def test_issue_6999():
    s = tanh(x).lseries(x, 1)
    assert next(s) == tanh(1)
    assert next(s) == x - (x - 1)*tanh(1)**2 - 1
    assert next(s) == -(x - 1)**2*tanh(1) + (x - 1)**2*tanh(1)**3
    assert next(s) == -(x - 1)**3*tanh(1)**4 - (x - 1)**3/3 + \
        4*(x - 1)**3*tanh(1)**2/3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_docs.py ===
from pathlib import Path

import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from . import util


def get_docdir():
    parents = Path(__file__).resolve().parents
    try:
        # Assumes that spin is used to run tests
        nproot = parents[8]
    except IndexError:
        docdir = None
    else:
        docdir = nproot / "doc" / "source" / "f2py" / "code"
    if docdir and docdir.is_dir():
        return docdir
    # Assumes that an editable install is used to run tests
    return parents[3] / "doc" / "source" / "f2py" / "code"


pytestmark = pytest.mark.skipif(
    not get_docdir().is_dir(),
    reason=f"Could not find f2py documentation sources"
    f"({get_docdir()} does not exist)",
)

def _path(*args):
    return get_docdir().joinpath(*args)

@pytest.mark.slow
class TestDocAdvanced(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/build-f2py']
    sources = [_path('asterisk1.f90'), _path('asterisk2.f90'),
               _path('ftype.f')]

    def test_asterisk1(self):
        foo = self.module.foo1
        assert_equal(foo(), b'123456789A12')

    def test_asterisk2(self):
        foo = self.module.foo2
        assert_equal(foo(2), b'12')
        assert_equal(foo(12), b'123456789A12')
        assert_equal(foo(20), b'123456789A123456789B')

    def test_ftype(self):
        ftype = self.module
        ftype.foo()
        assert_equal(ftype.data.a, 0)
        ftype.data.a = 3
        ftype.data.x = [1, 2, 3]
        assert_equal(ftype.data.a, 3)
        assert_array_equal(ftype.data.x,
                           np.array([1, 2, 3], dtype=np.float32))
        ftype.data.x[1] = 45
        assert_array_equal(ftype.data.x,
                           np.array([1, 45, 3], dtype=np.float32))

    # TODO: implement test methods for other example Fortran codes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\ufunc_config.py ===
"""Typing tests for `numpy._core._ufunc_config`."""

import numpy as np


def func1(a: str, b: int) -> None:
    return None


def func2(a: str, b: int, c: float = 1.0) -> None:
    return None


def func3(a: str, b: int) -> int:
    return 0


class Write1:
    def write(self, a: str) -> None:
        return None


class Write2:
    def write(self, a: str, b: int = 1) -> None:
        return None


class Write3:
    def write(self, a: str) -> int:
        return 0


_err_default = np.geterr()
_bufsize_default = np.getbufsize()
_errcall_default = np.geterrcall()

try:
    np.seterr(all=None)
    np.seterr(divide="ignore")
    np.seterr(over="warn")
    np.seterr(under="call")
    np.seterr(invalid="raise")
    np.geterr()

    np.setbufsize(4096)
    np.getbufsize()

    np.seterrcall(func1)
    np.seterrcall(func2)
    np.seterrcall(func3)
    np.seterrcall(Write1())
    np.seterrcall(Write2())
    np.seterrcall(Write3())
    np.geterrcall()

    with np.errstate(call=func1, all="call"):
        pass
    with np.errstate(call=Write1(), divide="log", over="log"):
        pass

finally:
    np.seterr(**_err_default)
    np.setbufsize(_bufsize_default)
    np.seterrcall(_errcall_default)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_copy.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import DataFrame
import pandas._testing as tm


class TestCopy:
    @pytest.mark.parametrize("attr", ["index", "columns"])
    def test_copy_index_name_checking(self, float_frame, attr):
        # don't want to be able to modify the index stored elsewhere after
        # making a copy
        ind = getattr(float_frame, attr)
        ind.name = None
        cp = float_frame.copy()
        getattr(cp, attr).name = "foo"
        assert getattr(float_frame, attr).name is None

    @td.skip_copy_on_write_invalid_test
    def test_copy_cache(self):
        # GH#31784 _item_cache not cleared on copy causes incorrect reads after updates
        df = DataFrame({"a": [1]})

        df["x"] = [0]
        df["a"]

        df.copy()

        df["a"].values[0] = -1

        tm.assert_frame_equal(df, DataFrame({"a": [-1], "x": [0]}))

        df["y"] = [0]

        assert df["a"].values[0] == -1
        tm.assert_frame_equal(df, DataFrame({"a": [-1], "x": [0], "y": [0]}))

    def test_copy(self, float_frame, float_string_frame):
        cop = float_frame.copy()
        cop["E"] = cop["A"]
        assert "E" not in float_frame

        # copy objects
        copy = float_string_frame.copy()
        assert copy._mgr is not float_string_frame._mgr

    @td.skip_array_manager_invalid_test
    def test_copy_consolidates(self):
        # GH#42477
        df = DataFrame(
            {
                "a": np.random.default_rng(2).integers(0, 100, size=55),
                "b": np.random.default_rng(2).integers(0, 100, size=55),
            }
        )

        for i in range(10):
            df.loc[:, f"n_{i}"] = np.random.default_rng(2).integers(0, 100, size=55)

        assert len(df._mgr.blocks) == 11
        result = df.copy()
        assert len(result._mgr.blocks) == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_qexpr.py ===
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.concrete import Sum
from sympy.physics.quantum.qexpr import QExpr, _qsympify_sequence
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.core.containers import Tuple

x = Symbol('x')
y = Symbol('y')
n = Symbol('n', integer=True)
m = Symbol('m', integer=True)


def test_qexpr_new():
    q = QExpr(0)
    assert q.label == (0,)
    assert q.hilbert_space == HilbertSpace()
    assert q.is_commutative is False

    q = QExpr(0, 1)
    assert q.label == (Integer(0), Integer(1))

    q = QExpr._new_rawargs(HilbertSpace(), Integer(0), Integer(1))
    assert q.label == (Integer(0), Integer(1))
    assert q.hilbert_space == HilbertSpace()


def test_qexpr_commutative():
    q1 = QExpr(x)
    q2 = QExpr(y)
    assert q1.is_commutative is False
    assert q2.is_commutative is False
    assert q1*q2 != q2*q1

    q = QExpr._new_rawargs(Integer(0), Integer(1), HilbertSpace())
    assert q.is_commutative is False


def test_qexpr_free_symbols():
    q1 = QExpr(x, y)
    assert q1.free_symbols == {x, y}


def test_qexpr_sum():
    q1 = Sum(QExpr(n), (n,0,2))
    assert q1.doit() == QExpr(0) + QExpr(1) + QExpr(2)

    q2 = Sum(QExpr(n, m), (n, 0, 2), (m, 0, 2))
    assert q2.doit() == QExpr(0, 0) + QExpr(0, 1) + QExpr(0, 2) + \
        QExpr(1, 0) + QExpr(1, 1) + QExpr(1, 2) + \
        QExpr(2, 0) + QExpr(2, 1) + QExpr(2, 2)


def test_qexpr_subs():
    q1 = QExpr(x, y)
    assert q1.subs(x, y) == QExpr(y, y)
    assert q1.subs({x: 1, y: 2}) == QExpr(1, 2)


def test_qsympify():
    assert _qsympify_sequence([[1, 2], [1, 3]]) == (Tuple(1, 2), Tuple(1, 3))
    assert _qsympify_sequence(([1, 2, [3, 4, [2, ]], 1], 3)) == \
        (Tuple(1, 2, Tuple(3, 4, Tuple(2,)), 1), 3)
    assert _qsympify_sequence((1,)) == (1,)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dotenv\version.py ===
__version__ = "1.1.0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\idna\package_data.py ===
__version__ = "3.10"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\__version__.py ===
from numpy.version import version  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\pivot\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\reader\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\writer\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\common.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import (
    DataFrame,
    concat,
)

if TYPE_CHECKING:
    from pandas._typing import AxisInt


def _check_mixed_float(df, dtype=None):
    # float16 are most likely to be upcasted to float32
    dtypes = {"A": "float32", "B": "float32", "C": "float16", "D": "float64"}
    if isinstance(dtype, str):
        dtypes = {k: dtype for k, v in dtypes.items()}
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    if dtypes.get("A"):
        assert df.dtypes["A"] == dtypes["A"]
    if dtypes.get("B"):
        assert df.dtypes["B"] == dtypes["B"]
    if dtypes.get("C"):
        assert df.dtypes["C"] == dtypes["C"]
    if dtypes.get("D"):
        assert df.dtypes["D"] == dtypes["D"]


def _check_mixed_int(df, dtype=None):
    dtypes = {"A": "int32", "B": "uint64", "C": "uint8", "D": "int64"}
    if isinstance(dtype, str):
        dtypes = {k: dtype for k, v in dtypes.items()}
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    if dtypes.get("A"):
        assert df.dtypes["A"] == dtypes["A"]
    if dtypes.get("B"):
        assert df.dtypes["B"] == dtypes["B"]
    if dtypes.get("C"):
        assert df.dtypes["C"] == dtypes["C"]
    if dtypes.get("D"):
        assert df.dtypes["D"] == dtypes["D"]


def zip_frames(frames: list[DataFrame], axis: AxisInt = 1) -> DataFrame:
    """
    take a list of frames, zip them together under the
    assumption that these all have the first frames' index/columns.

    Returns
    -------
    new_frame : DataFrame
    """
    if axis == 1:
        columns = frames[0].columns
        zipped = [f.loc[:, c] for c in columns for f in frames]
        return concat(zipped, axis=1)
    else:
        index = frames[0].index
        zipped = [f.loc[i, :] for i in index for f in frames]
        return DataFrame(zipped)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_to_frame.py ===
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestToFrame:
    def test_to_frame_respects_name_none(self):
        # GH#44212 if we explicitly pass name=None, then that should be respected,
        #  not changed to 0
        # GH-45448 this is first deprecated & enforced in 2.0
        ser = Series(range(3))
        result = ser.to_frame(None)

        exp_index = Index([None], dtype=object)
        tm.assert_index_equal(result.columns, exp_index)

        result = ser.rename("foo").to_frame(None)
        exp_index = Index([None], dtype=object)
        tm.assert_index_equal(result.columns, exp_index)

    def test_to_frame(self, datetime_series):
        datetime_series.name = None
        rs = datetime_series.to_frame()
        xp = DataFrame(datetime_series.values, index=datetime_series.index)
        tm.assert_frame_equal(rs, xp)

        datetime_series.name = "testname"
        rs = datetime_series.to_frame()
        xp = DataFrame(
            {"testname": datetime_series.values}, index=datetime_series.index
        )
        tm.assert_frame_equal(rs, xp)

        rs = datetime_series.to_frame(name="testdifferent")
        xp = DataFrame(
            {"testdifferent": datetime_series.values}, index=datetime_series.index
        )
        tm.assert_frame_equal(rs, xp)

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_to_frame_expanddim(self):
        # GH#9762

        class SubclassedSeries(Series):
            @property
            def _constructor_expanddim(self):
                return SubclassedFrame

        class SubclassedFrame(DataFrame):
            pass

        ser = SubclassedSeries([1, 2, 3], name="X")
        result = ser.to_frame()
        assert isinstance(result, SubclassedFrame)
        expected = SubclassedFrame({"X": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_ccalendar.py ===
from datetime import (
    date,
    datetime,
)

from hypothesis import given
import numpy as np
import pytest

from pandas._libs.tslibs import ccalendar

from pandas._testing._hypothesis import DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ


@pytest.mark.parametrize(
    "date_tuple,expected",
    [
        ((2001, 3, 1), 60),
        ((2004, 3, 1), 61),
        ((1907, 12, 31), 365),  # End-of-year, non-leap year.
        ((2004, 12, 31), 366),  # End-of-year, leap year.
    ],
)
def test_get_day_of_year_numeric(date_tuple, expected):
    assert ccalendar.get_day_of_year(*date_tuple) == expected


def test_get_day_of_year_dt():
    dt = datetime.fromordinal(1 + np.random.default_rng(2).integers(365 * 4000))
    result = ccalendar.get_day_of_year(dt.year, dt.month, dt.day)

    expected = (dt - dt.replace(month=1, day=1)).days + 1
    assert result == expected


@pytest.mark.parametrize(
    "input_date_tuple, expected_iso_tuple",
    [
        [(2020, 1, 1), (2020, 1, 3)],
        [(2019, 12, 31), (2020, 1, 2)],
        [(2019, 12, 30), (2020, 1, 1)],
        [(2009, 12, 31), (2009, 53, 4)],
        [(2010, 1, 1), (2009, 53, 5)],
        [(2010, 1, 3), (2009, 53, 7)],
        [(2010, 1, 4), (2010, 1, 1)],
        [(2006, 1, 1), (2005, 52, 7)],
        [(2005, 12, 31), (2005, 52, 6)],
        [(2008, 12, 28), (2008, 52, 7)],
        [(2008, 12, 29), (2009, 1, 1)],
    ],
)
def test_dt_correct_iso_8601_year_week_and_day(input_date_tuple, expected_iso_tuple):
    result = ccalendar.get_iso_calendar(*input_date_tuple)
    expected_from_date_isocalendar = date(*input_date_tuple).isocalendar()
    assert result == expected_from_date_isocalendar
    assert result == expected_iso_tuple


@given(DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ)
def test_isocalendar(dt):
    expected = dt.isocalendar()
    result = ccalendar.get_iso_calendar(dt.year, dt.month, dt.day)
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_deprecate.py ===
from textwrap import dedent

import pytest

from pandas.util._decorators import deprecate

import pandas._testing as tm


def new_func():
    """
    This is the summary. The deprecate directive goes next.

    This is the extended summary. The deprecate directive goes before this.
    """
    return "new_func called"


def new_func_no_docstring():
    return "new_func_no_docstring called"


def new_func_wrong_docstring():
    """Summary should be in the next line."""
    return "new_func_wrong_docstring called"


def new_func_with_deprecation():
    """
    This is the summary. The deprecate directive goes next.

    .. deprecated:: 1.0
        Use new_func instead.

    This is the extended summary. The deprecate directive goes before this.
    """


def test_deprecate_ok():
    depr_func = deprecate("depr_func", new_func, "1.0", msg="Use new_func instead.")

    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()

    assert result == "new_func called"
    assert depr_func.__doc__ == dedent(new_func_with_deprecation.__doc__)


def test_deprecate_no_docstring():
    depr_func = deprecate(
        "depr_func", new_func_no_docstring, "1.0", msg="Use new_func instead."
    )
    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()
    assert result == "new_func_no_docstring called"


def test_deprecate_wrong_docstring():
    msg = "deprecate needs a correctly formatted docstring"
    with pytest.raises(AssertionError, match=msg):
        deprecate(
            "depr_func", new_func_wrong_docstring, "1.0", msg="Use new_func instead."
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\index\__init__.py ===
"""Index interaction code"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\models\__init__.py ===
"""A package that contains models that represent entities."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\network\__init__.py ===
"""Contains purely network-related utilities."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\install\__init__.py ===
"""For modules related to installing packages."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\idna\package_data.py ===
__version__ = "3.10"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\galgebra.py ===
raise ImportError("""As of SymPy 1.0 the galgebra module is maintained separately at https://github.com/pygae/galgebra""")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\release.py ===
__version__ = "1.14.0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_subsets.py ===
from sympy.combinatorics.subsets import Subset, ksubsets
from sympy.testing.pytest import raises


def test_subset():
    a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
    assert a.next_binary() == Subset(['b'], ['a', 'b', 'c', 'd'])
    assert a.prev_binary() == Subset(['c'], ['a', 'b', 'c', 'd'])
    assert a.next_lexicographic() == Subset(['d'], ['a', 'b', 'c', 'd'])
    assert a.prev_lexicographic() == Subset(['c'], ['a', 'b', 'c', 'd'])
    assert a.next_gray() == Subset(['c'], ['a', 'b', 'c', 'd'])
    assert a.prev_gray() == Subset(['d'], ['a', 'b', 'c', 'd'])
    assert a.rank_binary == 3
    assert a.rank_lexicographic == 14
    assert a.rank_gray == 2
    assert a.cardinality == 16
    assert a.size == 2
    assert Subset.bitlist_from_subset(a, ['a', 'b', 'c', 'd']) == '0011'

    a = Subset([2, 5, 7], [1, 2, 3, 4, 5, 6, 7])
    assert a.next_binary() == Subset([2, 5, 6], [1, 2, 3, 4, 5, 6, 7])
    assert a.prev_binary() == Subset([2, 5], [1, 2, 3, 4, 5, 6, 7])
    assert a.next_lexicographic() == Subset([2, 6], [1, 2, 3, 4, 5, 6, 7])
    assert a.prev_lexicographic() == Subset([2, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
    assert a.next_gray() == Subset([2, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
    assert a.prev_gray() == Subset([2, 5], [1, 2, 3, 4, 5, 6, 7])
    assert a.rank_binary == 37
    assert a.rank_lexicographic == 93
    assert a.rank_gray == 57
    assert a.cardinality == 128

    superset = ['a', 'b', 'c', 'd']
    assert Subset.unrank_binary(4, superset).rank_binary == 4
    assert Subset.unrank_gray(10, superset).rank_gray == 10

    superset = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert Subset.unrank_binary(33, superset).rank_binary == 33
    assert Subset.unrank_gray(25, superset).rank_gray == 25

    a = Subset([], ['a', 'b', 'c', 'd'])
    i = 1
    while a.subset != Subset(['d'], ['a', 'b', 'c', 'd']).subset:
        a = a.next_lexicographic()
        i = i + 1
    assert i == 16

    i = 1
    while a.subset != Subset([], ['a', 'b', 'c', 'd']).subset:
        a = a.prev_lexicographic()
        i = i + 1
    assert i == 16

    raises(ValueError, lambda: Subset(['a', 'b'], ['a']))
    raises(ValueError, lambda: Subset(['a'], ['b', 'c']))
    raises(ValueError, lambda: Subset.subset_from_bitlist(['a', 'b'], '010'))

    assert Subset(['a'], ['a', 'b']) != Subset(['b'], ['a', 'b'])
    assert Subset(['a'], ['a', 'b']) != Subset(['a'], ['a', 'c'])

def test_ksubsets():
    assert list(ksubsets([1, 2, 3], 2)) == [(1, 2), (1, 3), (2, 3)]
    assert list(ksubsets([1, 2, 3, 4, 5], 2)) == [(1, 2), (1, 3), (1, 4),
               (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\combinatorial\__init__.py ===
# Stub __init__.py for sympy.functions.combinatorial

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\__init__.py ===
# Stub __init__.py for sympy.functions.elementary

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\__init__.py ===
# Stub __init__.py for the sympy.functions.special package

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_ecm.py ===
from sympy.external.gmpy import invert
from sympy.ntheory.ecm import ecm, Point
from sympy.testing.pytest import slow

@slow
def test_ecm():
    assert ecm(3146531246531241245132451321) == {3, 100327907731, 10454157497791297}
    assert ecm(46167045131415113) == {43, 2634823, 407485517}
    assert ecm(631211032315670776841) == {9312934919, 67777885039}
    assert ecm(398883434337287) == {99476569, 4009823}
    assert ecm(64211816600515193) == {281719, 359641, 633767}
    assert ecm(4269021180054189416198169786894227) == {184039, 241603, 333331, 477973, 618619, 974123}
    assert ecm(4516511326451341281684513) == {3, 39869, 131743543, 95542348571}
    assert ecm(4132846513818654136451) == {47, 160343, 2802377, 195692803}
    assert ecm(168541512131094651323) == {79, 113, 11011069, 1714635721}
    #This takes ~10secs while factorint is not able to factorize this even in ~10mins
    assert ecm(7060005655815754299976961394452809, B1=100000, B2=1000000) == {6988699669998001, 1010203040506070809}


def test_Point():
    #The curve is of the form y**2 = x**3 + a*x**2 + x
    mod = 101
    a = 10
    a_24 = (a + 2)*invert(4, mod)
    p1 = Point(10, 17, a_24, mod)
    p2 = p1.double()
    assert p2 == Point(68, 56, a_24, mod)
    p4 = p2.double()
    assert p4 == Point(22, 64, a_24, mod)
    p8 = p4.double()
    assert p8 == Point(71, 95, a_24, mod)
    p16 = p8.double()
    assert p16 == Point(5, 16, a_24, mod)
    p32 = p16.double()
    assert p32 == Point(33, 96, a_24, mod)

    # p3 = p2 + p1
    p3 = p2.add(p1, p1)
    assert p3 == Point(1, 61, a_24, mod)
    # p5 = p3 + p2 or p4 + p1
    p5 = p3.add(p2, p1)
    assert p5 == Point(49, 90, a_24, mod)
    assert p5 == p4.add(p1, p3)
    # p6 = 2*p3
    p6 = p3.double()
    assert p6 == Point(87, 43, a_24, mod)
    assert p6 == p4.add(p2, p2)
    # p7 = p5 + p2
    p7 = p5.add(p2, p3)
    assert p7 == Point(69, 23, a_24, mod)
    assert p7 == p4.add(p3, p1)
    assert p7 == p6.add(p1, p5)
    # p9 = p5 + p4
    p9 = p5.add(p4, p1)
    assert p9 == Point(56, 99, a_24, mod)
    assert p9 == p6.add(p3, p3)
    assert p9 == p7.add(p2, p5)
    assert p9 == p8.add(p1, p7)

    assert p5 == p1.mont_ladder(5)
    assert p9 == p1.mont_ladder(9)
    assert p16 == p1.mont_ladder(16)
    assert p9 == p3.mont_ladder(3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\c\__init__.py ===
"""Used for translating C source code into a SymPy expression"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\fortran\__init__.py ===
"""Used for translating Fortran source code into a SymPy expression. """

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_rationaltools.py ===
"""Tests for tools for manipulation of rational expressions. """

from sympy.polys.rationaltools import together

from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.abc import x, y, z

A, B = symbols('A,B', commutative=False)


def test_together():
    assert together(0) == 0
    assert together(1) == 1

    assert together(x*y*z) == x*y*z
    assert together(x + y) == x + y

    assert together(1/x) == 1/x

    assert together(1/x + 1) == (x + 1)/x
    assert together(1/x + 3) == (3*x + 1)/x
    assert together(1/x + x) == (x**2 + 1)/x

    assert together(1/x + S.Half) == (x + 2)/(2*x)
    assert together(S.Half + x/2) == Mul(S.Half, x + 1, evaluate=False)

    assert together(1/x + 2/y) == (2*x + y)/(y*x)
    assert together(1/(1 + 1/x)) == x/(1 + x)
    assert together(x/(1 + 1/x)) == x**2/(1 + x)

    assert together(1/x + 1/y + 1/z) == (x*y + x*z + y*z)/(x*y*z)
    assert together(1/(1 + x + 1/y + 1/z)) == y*z/(y + z + y*z + x*y*z)

    assert together(1/(x*y) + 1/(x*y)**2) == y**(-2)*x**(-2)*(1 + x*y)
    assert together(1/(x*y) + 1/(x*y)**4) == y**(-4)*x**(-4)*(1 + x**3*y**3)
    assert together(1/(x**7*y) + 1/(x*y)**4) == y**(-4)*x**(-7)*(x**3 + y**3)

    assert together(5/(2 + 6/(3 + 7/(4 + 8/(5 + 9/x))))) == \
        Rational(5, 2)*((171 + 119*x)/(279 + 203*x))

    assert together(1 + 1/(x + 1)**2) == (1 + (x + 1)**2)/(x + 1)**2
    assert together(1 + 1/(x*(1 + x))) == (1 + x*(1 + x))/(x*(1 + x))
    assert together(
        1/(x*(x + 1)) + 1/(x*(x + 2))) == (3 + 2*x)/(x*(1 + x)*(2 + x))
    assert together(1 + 1/(2*x + 2)**2) == (4*(x + 1)**2 + 1)/(4*(x + 1)**2)

    assert together(sin(1/x + 1/y)) == sin(1/x + 1/y)
    assert together(sin(1/x + 1/y), deep=True) == sin((x + y)/(x*y))

    assert together(1/exp(x) + 1/(x*exp(x))) == (1 + x)/(x*exp(x))
    assert together(1/exp(2*x) + 1/(x*exp(3*x))) == (1 + exp(x)*x)/(x*exp(3*x))

    assert together(Integral(1/x + 1/y, x)) == Integral((x + y)/(x*y), x)
    assert together(Eq(1/x + 1/y, 1 + 1/z)) == Eq((x + y)/(x*y), (z + 1)/z)

    assert together((A*B)**-1 + (B*A)**-1) == (A*B)**-1 + (B*A)**-1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_as_explicit.py ===
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, \
    ArrayTensorProduct, PermuteDims, ArrayDiagonal, ArrayContraction, ArrayAdd
from sympy.testing.pytest import raises


def test_array_as_explicit_call():

    assert ZeroArray(3, 2, 4).as_explicit() == ImmutableDenseNDimArray.zeros(3, 2, 4)
    assert OneArray(3, 2, 4).as_explicit() == ImmutableDenseNDimArray([1 for i in range(3*2*4)]).reshape(3, 2, 4)

    k = Symbol("k")
    X = ArraySymbol("X", (k, 3, 2))
    raises(ValueError, lambda: X.as_explicit())
    raises(ValueError, lambda: ZeroArray(k, 2, 3).as_explicit())
    raises(ValueError, lambda: OneArray(2, k, 2).as_explicit())

    A = ArraySymbol("A", (3, 3))
    B = ArraySymbol("B", (3, 3))

    texpr = tensorproduct(A, B)
    assert isinstance(texpr, ArrayTensorProduct)
    assert texpr.as_explicit() == tensorproduct(A.as_explicit(), B.as_explicit())

    texpr = tensorcontraction(A, (0, 1))
    assert isinstance(texpr, ArrayContraction)
    assert texpr.as_explicit() == A[0, 0] + A[1, 1] + A[2, 2]

    texpr = tensordiagonal(A, (0, 1))
    assert isinstance(texpr, ArrayDiagonal)
    assert texpr.as_explicit() == ImmutableDenseNDimArray([A[0, 0], A[1, 1], A[2, 2]])

    texpr = permutedims(A, [1, 0])
    assert isinstance(texpr, PermuteDims)
    assert texpr.as_explicit() == permutedims(A.as_explicit(), [1, 0])


def test_array_as_explicit_matrix_symbol():

    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)

    texpr = tensorproduct(A, B)
    assert isinstance(texpr, ArrayTensorProduct)
    assert texpr.as_explicit() == tensorproduct(A.as_explicit(), B.as_explicit())

    texpr = tensorcontraction(A, (0, 1))
    assert isinstance(texpr, ArrayContraction)
    assert texpr.as_explicit() == A[0, 0] + A[1, 1] + A[2, 2]

    texpr = tensordiagonal(A, (0, 1))
    assert isinstance(texpr, ArrayDiagonal)
    assert texpr.as_explicit() == ImmutableDenseNDimArray([A[0, 0], A[1, 1], A[2, 2]])

    texpr = permutedims(A, [1, 0])
    assert isinstance(texpr, PermuteDims)
    assert texpr.as_explicit() == permutedims(A.as_explicit(), [1, 0])

    expr = ArrayAdd(ArrayTensorProduct(A, B), ArrayTensorProduct(B, A))
    assert expr.as_explicit() == expr.args[0].as_explicit() + expr.args[1].as_explicit()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio_websocket\_version.py ===
__version__ = '0.12.2'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\__init__.py ===
__version__ = "4.0.2"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\api\test_types.py ===
from __future__ import annotations

import pandas._testing as tm
from pandas.api import types
from pandas.tests.api.test_api import Base


class TestTypes(Base):
    allowed = [
        "is_any_real_numeric_dtype",
        "is_bool",
        "is_bool_dtype",
        "is_categorical_dtype",
        "is_complex",
        "is_complex_dtype",
        "is_datetime64_any_dtype",
        "is_datetime64_dtype",
        "is_datetime64_ns_dtype",
        "is_datetime64tz_dtype",
        "is_dtype_equal",
        "is_float",
        "is_float_dtype",
        "is_int64_dtype",
        "is_integer",
        "is_integer_dtype",
        "is_number",
        "is_numeric_dtype",
        "is_object_dtype",
        "is_scalar",
        "is_sparse",
        "is_string_dtype",
        "is_signed_integer_dtype",
        "is_timedelta64_dtype",
        "is_timedelta64_ns_dtype",
        "is_unsigned_integer_dtype",
        "is_period_dtype",
        "is_interval",
        "is_interval_dtype",
        "is_re",
        "is_re_compilable",
        "is_dict_like",
        "is_iterator",
        "is_file_like",
        "is_list_like",
        "is_hashable",
        "is_array_like",
        "is_named_tuple",
        "pandas_dtype",
        "union_categoricals",
        "infer_dtype",
        "is_extension_array_dtype",
    ]
    deprecated: list[str] = []
    dtypes = ["CategoricalDtype", "DatetimeTZDtype", "PeriodDtype", "IntervalDtype"]

    def test_types(self):
        self.check(types, self.allowed + self.dtypes + self.deprecated)

    def test_deprecated_from_api_types(self):
        for t in self.deprecated:
            with tm.assert_produces_warning(FutureWarning):
                getattr(types, t)(1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_reduction.py ===
import numpy as np
import pytest

import pandas as pd


@pytest.fixture
def data():
    """Fixture returning boolean array, with valid and missing values."""
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.mark.parametrize(
    "values, exp_any, exp_all, exp_any_noskip, exp_all_noskip",
    [
        ([True, pd.NA], True, True, True, pd.NA),
        ([False, pd.NA], False, False, pd.NA, False),
        ([pd.NA], False, True, pd.NA, pd.NA),
        ([], False, True, False, True),
        # GH-33253: all True / all False values buggy with skipna=False
        ([True, True], True, True, True, True),
        ([False, False], False, False, False, False),
    ],
)
def test_any_all(values, exp_any, exp_all, exp_any_noskip, exp_all_noskip):
    # the methods return numpy scalars
    exp_any = pd.NA if exp_any is pd.NA else np.bool_(exp_any)
    exp_all = pd.NA if exp_all is pd.NA else np.bool_(exp_all)
    exp_any_noskip = pd.NA if exp_any_noskip is pd.NA else np.bool_(exp_any_noskip)
    exp_all_noskip = pd.NA if exp_all_noskip is pd.NA else np.bool_(exp_all_noskip)

    for con in [pd.array, pd.Series]:
        a = con(values, dtype="boolean")
        assert a.any() is exp_any
        assert a.all() is exp_all
        assert a.any(skipna=False) is exp_any_noskip
        assert a.all(skipna=False) is exp_all_noskip

        assert np.any(a.any()) is exp_any
        assert np.all(a.all()) is exp_all


@pytest.mark.parametrize("dropna", [True, False])
def test_reductions_return_types(dropna, data, all_numeric_reductions):
    op = all_numeric_reductions
    s = pd.Series(data)
    if dropna:
        s = s.dropna()

    if op in ("sum", "prod"):
        assert isinstance(getattr(s, op)(), np.int_)
    elif op == "count":
        # Oddly on the 32 bit build (but not Windows), this is intc (!= intp)
        assert isinstance(getattr(s, op)(), np.integer)
    elif op in ("min", "max"):
        assert isinstance(getattr(s, op)(), np.bool_)
    else:
        # "mean", "std", "var", "median", "kurt", "skew"
        assert isinstance(getattr(s, op)(), np.float64)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_combine_concat.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


class TestSparseArrayConcat:
    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_basic(self, kind):
        a = SparseArray([1, 0, 0, 2], kind=kind)
        b = SparseArray([1, 0, 2, 2], kind=kind)

        result = SparseArray._concat_same_type([a, b])
        # Can't make any assertions about the sparse index itself
        # since we aren't don't merge sparse blocs across arrays
        # in to_concat
        expected = np.array([1, 2, 1, 2, 2], dtype="int64")
        tm.assert_numpy_array_equal(result.sp_values, expected)
        assert result.kind == kind

    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_uses_first_kind(self, kind):
        other = "integer" if kind == "block" else "block"
        a = SparseArray([1, 0, 0, 2], kind=kind)
        b = SparseArray([1, 0, 2, 2], kind=other)

        result = SparseArray._concat_same_type([a, b])
        expected = np.array([1, 2, 1, 2, 2], dtype="int64")
        tm.assert_numpy_array_equal(result.sp_values, expected)
        assert result.kind == kind


@pytest.mark.parametrize(
    "other, expected_dtype",
    [
        # compatible dtype -> preserve sparse
        (pd.Series([3, 4, 5], dtype="int64"), pd.SparseDtype("int64", 0)),
        # (pd.Series([3, 4, 5], dtype="Int64"), pd.SparseDtype("int64", 0)),
        # incompatible dtype -> Sparse[common dtype]
        (pd.Series([1.5, 2.5, 3.5], dtype="float64"), pd.SparseDtype("float64", 0)),
        # incompatible dtype -> Sparse[object] dtype
        (pd.Series(["a", "b", "c"], dtype=object), pd.SparseDtype(object, 0)),
        # categorical with compatible categories -> dtype of the categories
        (pd.Series([3, 4, 5], dtype="category"), np.dtype("int64")),
        (pd.Series([1.5, 2.5, 3.5], dtype="category"), np.dtype("float64")),
        # categorical with incompatible categories -> object dtype
        (pd.Series(["a", "b", "c"], dtype="category"), np.dtype(object)),
    ],
)
def test_concat_with_non_sparse(other, expected_dtype):
    # https://github.com/pandas-dev/pandas/issues/34336
    s_sparse = pd.Series([1, 0, 2], dtype=pd.SparseDtype("int64", 0))

    result = pd.concat([s_sparse, other], ignore_index=True)
    expected = pd.Series(list(s_sparse) + list(other)).astype(expected_dtype)
    tm.assert_series_equal(result, expected)

    result = pd.concat([other, s_sparse], ignore_index=True)
    expected = pd.Series(list(other) + list(s_sparse)).astype(expected_dtype)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_append.py ===
import pytest

from pandas import (
    CategoricalIndex,
    Index,
)
import pandas._testing as tm


class TestAppend:
    @pytest.fixture
    def ci(self):
        categories = list("cab")
        return CategoricalIndex(list("aabbca"), categories=categories, ordered=False)

    def test_append(self, ci):
        # append cats with the same categories
        result = ci[:3].append(ci[3:])
        tm.assert_index_equal(result, ci, exact=True)

        foos = [ci[:1], ci[1:3], ci[3:]]
        result = foos[0].append(foos[1:])
        tm.assert_index_equal(result, ci, exact=True)

    def test_append_empty(self, ci):
        # empty
        result = ci.append([])
        tm.assert_index_equal(result, ci, exact=True)

    def test_append_mismatched_categories(self, ci):
        # appending with different categories or reordered is not ok
        msg = "all inputs must be Index"
        with pytest.raises(TypeError, match=msg):
            ci.append(ci.values.set_categories(list("abcd")))
        with pytest.raises(TypeError, match=msg):
            ci.append(ci.values.reorder_categories(list("abc")))

    def test_append_category_objects(self, ci):
        # with objects
        result = ci.append(Index(["c", "a"]))
        expected = CategoricalIndex(list("aabbcaca"), categories=ci.categories)
        tm.assert_index_equal(result, expected, exact=True)

    def test_append_non_categories(self, ci):
        # invalid objects -> cast to object via concat_compat
        result = ci.append(Index(["a", "d"]))
        expected = Index(["a", "a", "b", "b", "c", "a", "a", "d"])
        tm.assert_index_equal(result, expected, exact=True)

    def test_append_object(self, ci):
        # GH#14298 - if base object is not categorical -> coerce to object
        result = Index(["c", "a"]).append(ci)
        expected = Index(list("caaabbca"))
        tm.assert_index_equal(result, expected, exact=True)

    def test_append_to_another(self):
        # hits Index._concat
        fst = Index(["a", "b"])
        snd = CategoricalIndex(["d", "e"])
        result = fst.append(snd)
        expected = Index(["a", "b", "d", "e"])
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_fillna.py ===
import pytest

import pandas as pd
import pandas._testing as tm


class TestDatetimeIndexFillNA:
    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])
    def test_fillna_datetime64(self, tz):
        # GH 11343
        idx = pd.DatetimeIndex(["2011-01-01 09:00", pd.NaT, "2011-01-01 11:00"])

        exp = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"]
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00")), exp)

        # tz mismatch
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00"),
                pd.Timestamp("2011-01-01 10:00", tz=tz),
                pd.Timestamp("2011-01-01 11:00"),
            ],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00", tz=tz)), exp)

        # object
        exp = pd.Index(
            [pd.Timestamp("2011-01-01 09:00"), "x", pd.Timestamp("2011-01-01 11:00")],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna("x"), exp)

        idx = pd.DatetimeIndex(["2011-01-01 09:00", pd.NaT, "2011-01-01 11:00"], tz=tz)

        exp = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"], tz=tz
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00", tz=tz)), exp)

        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00", tz=tz),
                pd.Timestamp("2011-01-01 10:00"),
                pd.Timestamp("2011-01-01 11:00", tz=tz),
            ],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00")), exp)

        # object
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00", tz=tz),
                "x",
                pd.Timestamp("2011-01-01 11:00", tz=tz),
            ],
            dtype=object,
        )
        tm.assert_index_equal(idx.fillna("x"), exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_var.py ===
from sympy.core.function import (Function, FunctionClass)
from sympy.core.symbol import (Symbol, var)
from sympy.testing.pytest import raises

def test_var():
    ns = {"var": var, "raises": raises}
    eval("var('a')", ns)
    assert ns["a"] == Symbol("a")

    eval("var('b bb cc zz _x')", ns)
    assert ns["b"] == Symbol("b")
    assert ns["bb"] == Symbol("bb")
    assert ns["cc"] == Symbol("cc")
    assert ns["zz"] == Symbol("zz")
    assert ns["_x"] == Symbol("_x")

    v = eval("var(['d', 'e', 'fg'])", ns)
    assert ns['d'] == Symbol('d')
    assert ns['e'] == Symbol('e')
    assert ns['fg'] == Symbol('fg')

# check return value
    assert v != ['d', 'e', 'fg']
    assert v == [Symbol('d'), Symbol('e'), Symbol('fg')]


def test_var_return():
    ns = {"var": var, "raises": raises}
    "raises(ValueError, lambda: var(''))"
    v2 = eval("var('q')", ns)
    v3 = eval("var('q p')", ns)

    assert v2 == Symbol('q')
    assert v3 == (Symbol('q'), Symbol('p'))


def test_var_accepts_comma():
    ns = {"var": var}
    v1 = eval("var('x y z')", ns)
    v2 = eval("var('x,y,z')", ns)
    v3 = eval("var('x,y z')", ns)

    assert v1 == v2
    assert v1 == v3


def test_var_keywords():
    ns = {"var": var}
    eval("var('x y', real=True)", ns)
    assert ns['x'].is_real and ns['y'].is_real


def test_var_cls():
    ns = {"var": var, "Function": Function}
    eval("var('f', cls=Function)", ns)

    assert isinstance(ns['f'], FunctionClass)

    eval("var('g,h', cls=Function)", ns)

    assert isinstance(ns['g'], FunctionClass)
    assert isinstance(ns['h'], FunctionClass)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_repmatrix.py ===
from sympy.testing.pytest import raises
from sympy.matrices.exceptions import NonSquareMatrixError, NonInvertibleMatrixError

from sympy import Matrix, Rational


def test_lll():
    A = Matrix([[1, 0, 0, 0, -20160],
                [0, 1, 0, 0, 33768],
                [0, 0, 1, 0, 39578],
                [0, 0, 0, 1, 47757]])
    L = Matrix([[ 10, -3,  -2,  8,  -4],
                [  3, -9,   8,  1, -11],
                [ -3, 13,  -9, -3,  -9],
                [-12, -7, -11,  9,  -1]])
    T = Matrix([[ 10, -3,  -2,  8],
                [  3, -9,   8,  1],
                [ -3, 13,  -9, -3],
                [-12, -7, -11,  9]])
    assert A.lll() == L
    assert A.lll_transform() == (L, T)
    assert T * A == L


def test_matrix_inv_mod():
    A = Matrix(2, 1, [1, 0])
    raises(NonSquareMatrixError, lambda: A.inv_mod(2))
    A = Matrix(2, 2, [1, 0, 0, 0])
    raises(NonInvertibleMatrixError, lambda: A.inv_mod(2))
    A = Matrix(2, 2, [1, 2, 3, 4])
    Ai = Matrix(2, 2, [1, 1, 0, 1])
    assert A.inv_mod(3) == Ai
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert A.inv_mod(2) == A
    A = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    raises(NonInvertibleMatrixError, lambda: A.inv_mod(5))
    A = Matrix(3, 3, [5, 1, 3, 2, 6, 0, 2, 1, 1])
    Ai = Matrix(3, 3, [6, 8, 0, 1, 5, 6, 5, 6, 4])
    assert A.inv_mod(9) == Ai
    A = Matrix(3, 3, [1, 6, -3, 4, 1, -5, 3, -5, 5])
    Ai = Matrix(3, 3, [4, 3, 3, 1, 2, 5, 1, 5, 1])
    assert A.inv_mod(6) == Ai
    A = Matrix(3, 3, [1, 6, 1, 4, 1, 5, 3, 2, 5])
    Ai = Matrix(3, 3, [6, 0, 3, 6, 6, 4, 1, 6, 1])
    assert A.inv_mod(7) == Ai
    A = Matrix([[1, 2], [3, Rational(3,4)]])
    raises(ValueError, lambda: A.inv_mod(2))
    A = Matrix([[1, 2], [3, 4]])
    raises(TypeError, lambda: A.inv_mod(Rational(1, 2)))
    # https://github.com/sympy/sympy/issues/27663
    M = Matrix([
        [2, 3, 1, 4],
        [1, 5, 3, 2],
        [3, 2, 4, 1],
        [4, 1, 2, 5],
    ])
    assert M.inv_mod(26) == Matrix([
        [7, 21, 10, 10],
        [1, 7, 19, 3],
        [14, 1, 15, 1],
        [25, 23, 3, 12],
    ])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\tests\test_conflict.py ===
from sympy.multipledispatch.conflict import (supercedes, ordering, ambiguities,
        ambiguous, super_signature, consistent)


class A: pass
class B(A): pass
class C: pass


def test_supercedes():
    assert supercedes([B], [A])
    assert supercedes([B, A], [A, A])
    assert not supercedes([B, A], [A, B])
    assert not supercedes([A], [B])


def test_consistent():
    assert consistent([A], [A])
    assert consistent([B], [B])
    assert not consistent([A], [C])
    assert consistent([A, B], [A, B])
    assert consistent([B, A], [A, B])
    assert not consistent([B, A], [B])
    assert not consistent([B, A], [B, C])


def test_super_signature():
    assert super_signature([[A]]) == [A]
    assert super_signature([[A], [B]]) == [B]
    assert super_signature([[A, B], [B, A]]) == [B, B]
    assert super_signature([[A, A, B], [A, B, A], [B, A, A]]) == [B, B, B]


def test_ambiguous():
    assert not ambiguous([A], [A])
    assert not ambiguous([A], [B])
    assert not ambiguous([B], [B])
    assert not ambiguous([A, B], [B, B])
    assert ambiguous([A, B], [B, A])


def test_ambiguities():
    signatures = [[A], [B], [A, B], [B, A], [A, C]]
    expected = {((A, B), (B, A))}
    result = ambiguities(signatures)
    assert set(map(frozenset, expected)) == set(map(frozenset, result))

    signatures = [[A], [B], [A, B], [B, A], [A, C], [B, B]]
    expected = set()
    result = ambiguities(signatures)
    assert set(map(frozenset, expected)) == set(map(frozenset, result))


def test_ordering():
    signatures = [[A, A], [A, B], [B, A], [B, B], [A, C]]
    ord = ordering(signatures)
    assert ord[0] == (B, B) or ord[0] == (A, C)
    assert ord[-1] == (A, A) or ord[-1] == (A, C)


def test_type_mro():
    assert super_signature([[object], [type]]) == [type]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_fermion.py ===
from pytest import raises

import sympy
from sympy.physics.quantum import Dagger, AntiCommutator, qapply
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.fermion import FermionFockKet, FermionFockBra
from sympy import Symbol


def test_fermionoperator():
    c = FermionOp('c')
    d = FermionOp('d')

    assert isinstance(c, FermionOp)
    assert isinstance(Dagger(c), FermionOp)

    assert c.is_annihilation
    assert not Dagger(c).is_annihilation

    assert FermionOp("c") == FermionOp("c", True)
    assert FermionOp("c") != FermionOp("d")
    assert FermionOp("c", True) != FermionOp("c", False)

    assert AntiCommutator(c, Dagger(c)).doit() == 1

    assert AntiCommutator(c, Dagger(d)).doit() == c * Dagger(d) + Dagger(d) * c


def test_fermion_states():
    c = FermionOp("c")

    # Fock states
    assert (FermionFockBra(0) * FermionFockKet(1)).doit() == 0
    assert (FermionFockBra(1) * FermionFockKet(1)).doit() == 1

    assert qapply(c * FermionFockKet(1)) == FermionFockKet(0)
    assert qapply(c * FermionFockKet(0)) == 0

    assert qapply(Dagger(c) * FermionFockKet(0)) == FermionFockKet(1)
    assert qapply(Dagger(c) * FermionFockKet(1)) == 0


def test_power():
    c = FermionOp("c")
    assert c**0 == 1
    assert c**1 == c
    assert c**2 == 0
    assert c**3 == 0
    assert Dagger(c)**1 == Dagger(c)
    assert Dagger(c)**2 == 0

    assert (c**Symbol('a')).func == sympy.core.power.Pow
    assert (c**Symbol('a')).args == (c, Symbol('a'))

    with raises(ValueError):
        c**-1

    with raises(ValueError):
        c**3.2

    with raises(TypeError):
        c**1j

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_diff.py ===
from mpmath import *

def test_diff():
    mp.dps = 15
    assert diff(log, 2.0, n=0).ae(log(2))
    assert diff(cos, 1.0).ae(-sin(1))
    assert diff(abs, 0.0) == 0
    assert diff(abs, 0.0, direction=1) == 1
    assert diff(abs, 0.0, direction=-1) == -1
    assert diff(exp, 1.0).ae(e)
    assert diff(exp, 1.0, n=5).ae(e)
    assert diff(exp, 2.0, n=5, direction=3*j).ae(e**2)
    assert diff(lambda x: x**2, 3.0, method='quad').ae(6)
    assert diff(lambda x: 3+x**5, 3.0, n=2, method='quad').ae(540)
    assert diff(lambda x: 3+x**5, 3.0, n=2, method='step').ae(540)
    assert diffun(sin)(2).ae(cos(2))
    assert diffun(sin, n=2)(2).ae(-sin(2))

def test_diffs():
    mp.dps = 15
    assert [chop(d) for d in diffs(sin, 0, 1)] == [0, 1]
    assert [chop(d) for d in diffs(sin, 0, 1, method='quad')] == [0, 1]
    assert [chop(d) for d in diffs(sin, 0, 2)] == [0, 1, 0]
    assert [chop(d) for d in diffs(sin, 0, 2, method='quad')] == [0, 1, 0]

def test_taylor():
    mp.dps = 15
    # Easy to test since the coefficients are exact in floating-point
    assert taylor(sqrt, 1, 4) == [1, 0.5, -0.125, 0.0625, -0.0390625]

def test_diff_partial():
    mp.dps = 15
    x,y,z = xyz = 2,3,7
    f = lambda x,y,z: 3*x**2 * (y+2)**3 * z**5
    assert diff(f, xyz, (0,0,0)).ae(25210500)
    assert diff(f, xyz, (0,0,1)).ae(18007500)
    assert diff(f, xyz, (0,0,2)).ae(10290000)
    assert diff(f, xyz, (0,1,0)).ae(15126300)
    assert diff(f, xyz, (0,1,1)).ae(10804500)
    assert diff(f, xyz, (0,1,2)).ae(6174000)
    assert diff(f, xyz, (0,2,0)).ae(6050520)
    assert diff(f, xyz, (0,2,1)).ae(4321800)
    assert diff(f, xyz, (0,2,2)).ae(2469600)
    assert diff(f, xyz, (1,0,0)).ae(25210500)
    assert diff(f, xyz, (1,0,1)).ae(18007500)
    assert diff(f, xyz, (1,0,2)).ae(10290000)
    assert diff(f, xyz, (1,1,0)).ae(15126300)
    assert diff(f, xyz, (1,1,1)).ae(10804500)
    assert diff(f, xyz, (1,1,2)).ae(6174000)
    assert diff(f, xyz, (1,2,0)).ae(6050520)
    assert diff(f, xyz, (1,2,1)).ae(4321800)
    assert diff(f, xyz, (1,2,2)).ae(2469600)
    assert diff(f, xyz, (2,0,0)).ae(12605250)
    assert diff(f, xyz, (2,0,1)).ae(9003750)
    assert diff(f, xyz, (2,0,2)).ae(5145000)
    assert diff(f, xyz, (2,1,0)).ae(7563150)
    assert diff(f, xyz, (2,1,1)).ae(5402250)
    assert diff(f, xyz, (2,1,2)).ae(3087000)
    assert diff(f, xyz, (2,2,0)).ae(3025260)
    assert diff(f, xyz, (2,2,1)).ae(2160900)
    assert diff(f, xyz, (2,2,2)).ae(1234800)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_freq_attr.py ===
import pytest

from pandas import (
    DatetimeIndex,
    date_range,
)

from pandas.tseries.offsets import (
    BDay,
    DateOffset,
    Day,
    Hour,
)


class TestFreq:
    def test_freq_setter_errors(self):
        # GH#20678
        idx = DatetimeIndex(["20180101", "20180103", "20180105"])

        # setting with an incompatible freq
        msg = (
            "Inferred frequency 2D from passed values does not conform to "
            "passed frequency 5D"
        )
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "5D"

        # setting with non-freq string
        with pytest.raises(ValueError, match="Invalid frequency"):
            idx._data.freq = "foo"

    @pytest.mark.parametrize("values", [["20180101", "20180103", "20180105"], []])
    @pytest.mark.parametrize("freq", ["2D", Day(2), "2B", BDay(2), "48h", Hour(48)])
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_freq_setter(self, values, freq, tz):
        # GH#20678
        idx = DatetimeIndex(values, tz=tz)

        # can set to an offset, converting from string if necessary
        idx._data.freq = freq
        assert idx.freq == freq
        assert isinstance(idx.freq, DateOffset)

        # can reset to None
        idx._data.freq = None
        assert idx.freq is None

    def test_freq_view_safe(self):
        # Setting the freq for one DatetimeIndex shouldn't alter the freq
        #  for another that views the same data

        dti = date_range("2016-01-01", periods=5)
        dta = dti._data

        dti2 = DatetimeIndex(dta)._with_freq(None)
        assert dti2.freq is None

        # Original was not altered
        assert dti.freq == "D"
        assert dta.freq == "D"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_timedelta.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    Series,
    Timedelta,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndex:
    def test_misc_coverage(self):
        rng = timedelta_range("1 day", periods=5)
        result = rng.groupby(rng.days)
        assert isinstance(next(iter(result.values()))[0], Timedelta)

    def test_map(self):
        # test_map_dictlike generally tests

        rng = timedelta_range("1 day", periods=10)

        f = lambda x: x.days
        result = rng.map(f)
        exp = Index([f(x) for x in rng], dtype=np.int64)
        tm.assert_index_equal(result, exp)

    def test_fields(self):
        rng = timedelta_range("1 days, 10:11:12.100123456", periods=2, freq="s")
        tm.assert_index_equal(rng.days, Index([1, 1], dtype=np.int64))
        tm.assert_index_equal(
            rng.seconds,
            Index([10 * 3600 + 11 * 60 + 12, 10 * 3600 + 11 * 60 + 13], dtype=np.int32),
        )
        tm.assert_index_equal(
            rng.microseconds,
            Index([100 * 1000 + 123, 100 * 1000 + 123], dtype=np.int32),
        )
        tm.assert_index_equal(rng.nanoseconds, Index([456, 456], dtype=np.int32))

        msg = "'TimedeltaIndex' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # with nat
        s = Series(rng)
        s[1] = np.nan

        tm.assert_series_equal(s.dt.days, Series([1, np.nan], index=[0, 1]))
        tm.assert_series_equal(
            s.dt.seconds, Series([10 * 3600 + 11 * 60 + 12, np.nan], index=[0, 1])
        )

        # preserve name (GH15589)
        rng.name = "name"
        assert rng.days.name == "name"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_indexers.py ===
# Tests aimed at pandas.core.indexers
import numpy as np
import pytest

from pandas.core.indexers import (
    is_scalar_indexer,
    length_of_indexer,
    validate_indices,
)


def test_length_of_indexer():
    arr = np.zeros(4, dtype=bool)
    arr[0] = 1
    result = length_of_indexer(arr)
    assert result == 1


def test_is_scalar_indexer():
    indexer = (0, 1)
    assert is_scalar_indexer(indexer, 2)
    assert not is_scalar_indexer(indexer[0], 2)

    indexer = (np.array([2]), 1)
    assert not is_scalar_indexer(indexer, 2)

    indexer = (np.array([2]), np.array([3]))
    assert not is_scalar_indexer(indexer, 2)

    indexer = (np.array([2]), np.array([3, 4]))
    assert not is_scalar_indexer(indexer, 2)

    assert not is_scalar_indexer(slice(None), 1)

    indexer = 0
    assert is_scalar_indexer(indexer, 1)

    indexer = (0,)
    assert is_scalar_indexer(indexer, 1)


class TestValidateIndices:
    def test_validate_indices_ok(self):
        indices = np.asarray([0, 1])
        validate_indices(indices, 2)
        validate_indices(indices[:0], 0)
        validate_indices(np.array([-1, -1]), 0)

    def test_validate_indices_low(self):
        indices = np.asarray([0, -2])
        with pytest.raises(ValueError, match="'indices' contains"):
            validate_indices(indices, 2)

    def test_validate_indices_high(self):
        indices = np.asarray([0, 1, 2])
        with pytest.raises(IndexError, match="indices are out"):
            validate_indices(indices, 2)

    def test_validate_indices_empty(self):
        with pytest.raises(IndexError, match="indices are out"):
            validate_indices(np.array([0, 1]), 0)