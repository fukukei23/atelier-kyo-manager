
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_engines.py ===
import re

import numpy as np
import pytest

from pandas._libs import index as libindex

import pandas as pd


@pytest.fixture(
    params=[
        (libindex.Int64Engine, np.int64),
        (libindex.Int32Engine, np.int32),
        (libindex.Int16Engine, np.int16),
        (libindex.Int8Engine, np.int8),
        (libindex.UInt64Engine, np.uint64),
        (libindex.UInt32Engine, np.uint32),
        (libindex.UInt16Engine, np.uint16),
        (libindex.UInt8Engine, np.uint8),
        (libindex.Float64Engine, np.float64),
        (libindex.Float32Engine, np.float32),
    ],
    ids=lambda x: x[0].__name__,
)
def numeric_indexing_engine_type_and_dtype(request):
    return request.param


class TestDatetimeEngine:
    @pytest.mark.parametrize(
        "scalar",
        [
            pd.Timedelta(pd.Timestamp("2016-01-01").asm8.view("m8[ns]")),
            pd.Timestamp("2016-01-01")._value,
            pd.Timestamp("2016-01-01").to_pydatetime(),
            pd.Timestamp("2016-01-01").to_datetime64(),
        ],
    )
    def test_not_contains_requires_timestamp(self, scalar):
        dti1 = pd.date_range("2016-01-01", periods=3)
        dti2 = dti1.insert(1, pd.NaT)  # non-monotonic
        dti3 = dti1.insert(3, dti1[0])  # non-unique
        dti4 = pd.date_range("2016-01-01", freq="ns", periods=2_000_000)
        dti5 = dti4.insert(0, dti4[0])  # over size threshold, not unique

        msg = "|".join([re.escape(str(scalar)), re.escape(repr(scalar))])
        for dti in [dti1, dti2, dti3, dti4, dti5]:
            with pytest.raises(TypeError, match=msg):
                scalar in dti._engine

            with pytest.raises(KeyError, match=msg):
                dti._engine.get_loc(scalar)


class TestTimedeltaEngine:
    @pytest.mark.parametrize(
        "scalar",
        [
            pd.Timestamp(pd.Timedelta(days=42).asm8.view("datetime64[ns]")),
            pd.Timedelta(days=42)._value,
            pd.Timedelta(days=42).to_pytimedelta(),
            pd.Timedelta(days=42).to_timedelta64(),
        ],
    )
    def test_not_contains_requires_timedelta(self, scalar):
        tdi1 = pd.timedelta_range("42 days", freq="9h", periods=1234)
        tdi2 = tdi1.insert(1, pd.NaT)  # non-monotonic
        tdi3 = tdi1.insert(3, tdi1[0])  # non-unique
        tdi4 = pd.timedelta_range("42 days", freq="ns", periods=2_000_000)
        tdi5 = tdi4.insert(0, tdi4[0])  # over size threshold, not unique

        msg = "|".join([re.escape(str(scalar)), re.escape(repr(scalar))])
        for tdi in [tdi1, tdi2, tdi3, tdi4, tdi5]:
            with pytest.raises(TypeError, match=msg):
                scalar in tdi._engine

            with pytest.raises(KeyError, match=msg):
                tdi._engine.get_loc(scalar)


class TestNumericEngine:
    def test_is_monotonic(self, numeric_indexing_engine_type_and_dtype):
        engine_type, dtype = numeric_indexing_engine_type_and_dtype
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)

        # monotonic increasing
        engine = engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False

        # monotonic decreasing
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True

        # neither monotonic increasing or decreasing
        arr = np.array([1] * num + [2] * num + [1] * num, dtype=dtype)
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self, numeric_indexing_engine_type_and_dtype):
        engine_type, dtype = numeric_indexing_engine_type_and_dtype

        # unique
        arr = np.array([1, 3, 2], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is True

        # not unique
        arr = np.array([1, 2, 1], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self, numeric_indexing_engine_type_and_dtype):
        engine_type, dtype = numeric_indexing_engine_type_and_dtype

        # unique
        arr = np.array([1, 2, 3], dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == 1

        # monotonic
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == slice(1000, 2000)

        # not monotonic
        arr = np.array([1, 2, 3] * num, dtype=dtype)
        engine = engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc(2)
        assert (result == expected).all()


class TestObjectEngine:
    engine_type = libindex.ObjectEngine
    dtype = np.object_
    values = list("abc")

    def test_is_monotonic(self):
        num = 1000
        arr = np.array(["a"] * num + ["a"] * num + ["c"] * num, dtype=self.dtype)

        # monotonic increasing
        engine = self.engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False

        # monotonic decreasing
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True

        # neither monotonic increasing or decreasing
        arr = np.array(["a"] * num + ["b"] * num + ["a"] * num, dtype=self.dtype)
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self):
        # unique
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is True

        # not unique
        arr = np.array(["a", "b", "a"], dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self):
        # unique
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc("b") == 1

        # monotonic
        num = 1000
        arr = np.array(["a"] * num + ["b"] * num + ["c"] * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc("b") == slice(1000, 2000)

        # not monotonic
        arr = np.array(self.values * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc("b")
        assert (result == expected).all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\interval\test_arithmetic.py ===
from datetime import timedelta

import numpy as np
import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestIntervalArithmetic:
    def test_interval_add(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(1, 2, closed=closed)

        result = interval + 1
        assert result == expected

        result = 1 + interval
        assert result == expected

        result = interval
        result += 1
        assert result == expected

        msg = r"unsupported operand type\(s\) for \+"
        with pytest.raises(TypeError, match=msg):
            interval + interval

        with pytest.raises(TypeError, match=msg):
            interval + "foo"

    def test_interval_sub(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(-1, 0, closed=closed)

        result = interval - 1
        assert result == expected

        result = interval
        result -= 1
        assert result == expected

        msg = r"unsupported operand type\(s\) for -"
        with pytest.raises(TypeError, match=msg):
            interval - interval

        with pytest.raises(TypeError, match=msg):
            interval - "foo"

    def test_interval_mult(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 2, closed=closed)

        result = interval * 2
        assert result == expected

        result = 2 * interval
        assert result == expected

        result = interval
        result *= 2
        assert result == expected

        msg = r"unsupported operand type\(s\) for \*"
        with pytest.raises(TypeError, match=msg):
            interval * interval

        msg = r"can\'t multiply sequence by non-int"
        with pytest.raises(TypeError, match=msg):
            interval * "foo"

    def test_interval_div(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 0.5, closed=closed)

        result = interval / 2.0
        assert result == expected

        result = interval
        result /= 2.0
        assert result == expected

        msg = r"unsupported operand type\(s\) for /"
        with pytest.raises(TypeError, match=msg):
            interval / interval

        with pytest.raises(TypeError, match=msg):
            interval / "foo"

    def test_interval_floordiv(self, closed):
        interval = Interval(1, 2, closed=closed)
        expected = Interval(0, 1, closed=closed)

        result = interval // 2
        assert result == expected

        result = interval
        result //= 2
        assert result == expected

        msg = r"unsupported operand type\(s\) for //"
        with pytest.raises(TypeError, match=msg):
            interval // interval

        with pytest.raises(TypeError, match=msg):
            interval // "foo"

    @pytest.mark.parametrize("method", ["__add__", "__sub__"])
    @pytest.mark.parametrize(
        "interval",
        [
            Interval(
                Timestamp("2017-01-01 00:00:00"), Timestamp("2018-01-01 00:00:00")
            ),
            Interval(Timedelta(days=7), Timedelta(days=14)),
        ],
    )
    @pytest.mark.parametrize(
        "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
    )
    def test_time_interval_add_subtract_timedelta(self, interval, delta, method):
        # https://github.com/pandas-dev/pandas/issues/32023
        result = getattr(interval, method)(delta)
        left = getattr(interval.left, method)(delta)
        right = getattr(interval.right, method)(delta)
        expected = Interval(left, right)

        assert result == expected

    @pytest.mark.parametrize("interval", [Interval(1, 2), Interval(1.0, 2.0)])
    @pytest.mark.parametrize(
        "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
    )
    def test_numeric_interval_add_timedelta_raises(self, interval, delta):
        # https://github.com/pandas-dev/pandas/issues/32023
        msg = "|".join(
            [
                "unsupported operand",
                "cannot use operands",
                "Only numeric, Timestamp and Timedelta endpoints are allowed",
            ]
        )
        with pytest.raises((TypeError, ValueError), match=msg):
            interval + delta

        with pytest.raises((TypeError, ValueError), match=msg):
            delta + interval

    @pytest.mark.parametrize("klass", [timedelta, np.timedelta64, Timedelta])
    def test_timedelta_add_timestamp_interval(self, klass):
        delta = klass(0)
        expected = Interval(Timestamp("2020-01-01"), Timestamp("2020-02-01"))

        result = delta + expected
        assert result == expected

        result = expected + delta
        assert result == expected


class TestIntervalComparisons:
    def test_interval_equal(self):
        assert Interval(0, 1) == Interval(0, 1, closed="right")
        assert Interval(0, 1) != Interval(0, 1, closed="left")
        assert Interval(0, 1) != 0

    def test_interval_comparison(self):
        msg = (
            "'<' not supported between instances of "
            "'pandas._libs.interval.Interval' and 'int'"
        )
        with pytest.raises(TypeError, match=msg):
            Interval(0, 1) < 2

        assert Interval(0, 1) < Interval(1, 2)
        assert Interval(0, 1) < Interval(0, 2)
        assert Interval(0, 1) < Interval(0.5, 1.5)
        assert Interval(0, 1) <= Interval(0, 1)
        assert Interval(0, 1) > Interval(-1, 2)
        assert Interval(0, 1) >= Interval(0, 1)

    def test_equality_comparison_broadcasts_over_array(self):
        # https://github.com/pandas-dev/pandas/issues/35931
        interval = Interval(0, 1)
        arr = np.array([interval, interval])
        result = interval == arr
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_info.py ===
from io import StringIO
from string import ascii_uppercase
import textwrap

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import (
    HAS_PYARROW,
    PYPY,
)

from pandas import (
    CategoricalIndex,
    Index,
    MultiIndex,
    Series,
    date_range,
)


def test_info_categorical_column_just_works():
    n = 2500
    data = np.array(list("abcdefghij")).take(
        np.random.default_rng(2).integers(0, 10, size=n, dtype=int)
    )
    s = Series(data).astype("category")
    s.isna()
    buf = StringIO()
    s.info(buf=buf)

    s2 = s[s == "d"]
    buf = StringIO()
    s2.info(buf=buf)


def test_info_categorical():
    # GH14298
    idx = CategoricalIndex(["a", "b"])
    s = Series(np.zeros(2), index=idx)
    buf = StringIO()
    s.info(buf=buf)


@pytest.mark.parametrize("verbose", [True, False])
def test_info_series(
    lexsorted_two_level_string_multiindex, verbose, using_infer_string
):
    index = lexsorted_two_level_string_multiindex
    ser = Series(range(len(index)), index=index, name="sth")
    buf = StringIO()
    ser.info(verbose=verbose, buf=buf)
    result = buf.getvalue()

    expected = textwrap.dedent(
        """\
        <class 'pandas.core.series.Series'>
        MultiIndex: 10 entries, ('foo', 'one') to ('qux', 'three')
        """
    )
    if verbose:
        expected += textwrap.dedent(
            """\
            Series name: sth
            Non-Null Count  Dtype
            --------------  -----
            10 non-null     int64
            """
        )
    qualifier = "" if using_infer_string and HAS_PYARROW else "+"
    expected += textwrap.dedent(
        f"""\
        dtypes: int64(1)
        memory usage: {ser.memory_usage()}.0{qualifier} bytes
        """
    )
    assert result == expected


def test_info_memory():
    s = Series([1, 2], dtype="i8")
    buf = StringIO()
    s.info(buf=buf)
    result = buf.getvalue()
    memory_bytes = float(s.memory_usage())
    expected = textwrap.dedent(
        f"""\
    <class 'pandas.core.series.Series'>
    RangeIndex: 2 entries, 0 to 1
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    2 non-null      int64
    dtypes: int64(1)
    memory usage: {memory_bytes} bytes
    """
    )
    assert result == expected


def test_info_wide():
    s = Series(np.random.default_rng(2).standard_normal(101))
    msg = "Argument `max_cols` can only be passed in DataFrame.info, not Series.info"
    with pytest.raises(ValueError, match=msg):
        s.info(max_cols=1)


def test_info_shows_dtypes():
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    n = 10
    for dtype in dtypes:
        s = Series(np.random.default_rng(2).integers(2, size=n).astype(dtype))
        buf = StringIO()
        s.info(buf=buf)
        res = buf.getvalue()
        name = f"{n:d} non-null     {dtype}"
        assert name in res


@pytest.mark.xfail(PYPY, reason="on PyPy deep=True doesn't change result")
def test_info_memory_usage_deep_not_pypy():
    s_with_object_index = Series({"a": [1]}, index=["foo"])
    assert s_with_object_index.memory_usage(
        index=True, deep=True
    ) > s_with_object_index.memory_usage(index=True)

    s_object = Series({"a": ["a"]})
    assert s_object.memory_usage(deep=True) > s_object.memory_usage()


@pytest.mark.xfail(not PYPY, reason="on PyPy deep=True does not change result")
def test_info_memory_usage_deep_pypy():
    s_with_object_index = Series({"a": [1]}, index=["foo"])
    assert s_with_object_index.memory_usage(
        index=True, deep=True
    ) == s_with_object_index.memory_usage(index=True)

    s_object = Series({"a": ["a"]})
    assert s_object.memory_usage(deep=True) == s_object.memory_usage()


@pytest.mark.parametrize(
    "index, plus",
    [
        ([1, 2, 3], False),
        (Index(list("ABC"), dtype="str"), not (using_string_dtype() and HAS_PYARROW)),
        (Index(list("ABC"), dtype=object), True),
        (MultiIndex.from_product([range(3), range(3)]), False),
        (
            MultiIndex.from_product([range(3), ["foo", "bar"]]),
            not (using_string_dtype() and HAS_PYARROW),
        ),
    ],
)
def test_info_memory_usage_qualified(index, plus):
    series = Series(1, index=index)
    buf = StringIO()
    series.info(buf=buf)
    if plus:
        assert "+" in buf.getvalue()
    else:
        assert "+" not in buf.getvalue()


def test_info_memory_usage_bug_on_multiindex():
    # GH 14308
    # memory usage introspection should not materialize .values
    N = 100
    M = len(ascii_uppercase)
    index = MultiIndex.from_product(
        [list(ascii_uppercase), date_range("20160101", periods=N)],
        names=["id", "date"],
    )
    s = Series(np.random.default_rng(2).standard_normal(N * M), index=index)

    unstacked = s.unstack("id")
    assert s.values.nbytes == unstacked.values.nbytes
    assert s.memory_usage(deep=True) > unstacked.memory_usage(deep=True).sum()

    # high upper bound
    diff = unstacked.memory_usage(deep=True).sum() - s.memory_usage(deep=True)
    assert diff < 2000

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_wtf\recaptcha\__init__.py ===
from .fields import RecaptchaField
from .validators import Recaptcha
from .widgets import RecaptchaWidget

__all__ = ["RecaptchaField", "RecaptchaWidget", "Recaptcha"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\internal\python_edition_defaults.py ===
"""
This file contains the serialized FeatureSetDefaults object corresponding to
the Pure Python runtime.  This is used for feature resolution under Editions.
"""
_PROTOBUF_INTERNAL_PYTHON_EDITION_DEFAULTS = b"\n\027\030\204\007\"\000*\020\010\001\020\002\030\002 \003(\0010\0028\002@\001\n\027\030\347\007\"\000*\020\010\002\020\001\030\001 \002(\0010\0018\002@\001\n\027\030\350\007\"\014\010\001\020\001\030\001 \002(\0010\001*\0048\002@\001 \346\007(\350\007"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\__main__.py ===
# See:
# https://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e
from numpy.f2py.f2py2e import main

main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\sparse\api.py ===
from pandas.core.dtypes.dtypes import SparseDtype

from pandas.core.arrays.sparse import SparseArray

__all__ = ["SparseArray", "SparseDtype"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_datetime.py ===
import re

import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndex:
    def test_get_loc_naive_dti_aware_str_deprecated(self):
        # GH#46903
        ts = Timestamp("20130101")._value
        dti = pd.DatetimeIndex([ts + 50 + i for i in range(100)])
        ser = Series(range(100), index=dti)

        key = "2013-01-01 00:00:00.000000050+0000"
        msg = re.escape(repr(key))
        with pytest.raises(KeyError, match=msg):
            ser[key]

        with pytest.raises(KeyError, match=msg):
            dti.get_loc(key)

    def test_indexing_with_datetime_tz(self):
        # GH#8260
        # support datetime64 with tz

        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        dr = date_range("20130110", periods=3)
        df = DataFrame({"A": idx, "B": dr})
        df["C"] = idx
        df.iloc[1, 1] = pd.NaT
        df.iloc[1, 2] = pd.NaT

        expected = Series(
            [Timestamp("2013-01-02 00:00:00-0500", tz="US/Eastern"), pd.NaT, pd.NaT],
            index=list("ABC"),
            dtype="object",
            name=1,
        )

        # indexing
        result = df.iloc[1]
        tm.assert_series_equal(result, expected)
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_indexing_fast_xs(self):
        # indexing - fast_xs
        df = DataFrame({"a": date_range("2014-01-01", periods=10, tz="UTC")})
        result = df.iloc[5]
        expected = Series(
            [Timestamp("2014-01-06 00:00:00+0000", tz="UTC")],
            index=["a"],
            name=5,
            dtype="M8[ns, UTC]",
        )
        tm.assert_series_equal(result, expected)

        result = df.loc[5]
        tm.assert_series_equal(result, expected)

        # indexing - boolean
        result = df[df.a > df.a[3]]
        expected = df.iloc[4:]
        tm.assert_frame_equal(result, expected)

    def test_consistency_with_tz_aware_scalar(self):
        # xef gh-12938
        # various ways of indexing the same tz-aware scalar
        df = Series([Timestamp("2016-03-30 14:35:25", tz="Europe/Brussels")]).to_frame()

        df = pd.concat([df, df]).reset_index(drop=True)
        expected = Timestamp("2016-03-30 14:35:25+0200", tz="Europe/Brussels")

        result = df[0][0]
        assert result == expected

        result = df.iloc[0, 0]
        assert result == expected

        result = df.loc[0, 0]
        assert result == expected

        result = df.iat[0, 0]
        assert result == expected

        result = df.at[0, 0]
        assert result == expected

        result = df[0].loc[0]
        assert result == expected

        result = df[0].at[0]
        assert result == expected

    def test_indexing_with_datetimeindex_tz(self, indexer_sl):
        # GH 12050
        # indexing on a series with a datetimeindex with tz
        index = date_range("2015-01-01", periods=2, tz="utc")

        ser = Series(range(2), index=index, dtype="int64")

        # list-like indexing

        for sel in (index, list(index)):
            # getitem
            result = indexer_sl(ser)[sel]
            expected = ser.copy()
            if sel is not index:
                expected.index = expected.index._with_freq(None)
            tm.assert_series_equal(result, expected)

            # setitem
            result = ser.copy()
            indexer_sl(result)[sel] = 1
            expected = Series(1, index=index)
            tm.assert_series_equal(result, expected)

        # single element indexing

        # getitem
        assert indexer_sl(ser)[index[1]] == 1

        # setitem
        result = ser.copy()
        indexer_sl(result)[index[1]] = 5
        expected = Series([0, 5], index=index)
        tm.assert_series_equal(result, expected)

    def test_nanosecond_getitem_setitem_with_tz(self):
        # GH 11679
        data = ["2016-06-28 08:30:00.123456789"]
        index = pd.DatetimeIndex(data, dtype="datetime64[ns, America/Chicago]")
        df = DataFrame({"a": [10]}, index=index)
        result = df.loc[df.index[0]]
        expected = Series(10, index=["a"], name=df.index[0])
        tm.assert_series_equal(result, expected)

        result = df.copy()
        result.loc[df.index[0], "a"] = -1
        expected = DataFrame(-1, index=index, columns=["a"])
        tm.assert_frame_equal(result, expected)

    def test_getitem_str_slice_millisecond_resolution(self, frame_or_series):
        # GH#33589

        keys = [
            "2017-10-25T16:25:04.151",
            "2017-10-25T16:25:04.252",
            "2017-10-25T16:50:05.237",
            "2017-10-25T16:50:05.238",
        ]
        obj = frame_or_series(
            [1, 2, 3, 4],
            index=[Timestamp(x) for x in keys],
        )
        result = obj[keys[1] : keys[2]]
        expected = frame_or_series(
            [2, 3],
            index=[
                Timestamp(keys[1]),
                Timestamp(keys[2]),
            ],
        )
        tm.assert_equal(result, expected)

    def test_getitem_pyarrow_index(self, frame_or_series):
        # GH 53644
        pytest.importorskip("pyarrow")
        obj = frame_or_series(
            range(5),
            index=date_range("2020", freq="D", periods=5).astype(
                "timestamp[us][pyarrow]"
            ),
        )
        result = obj.loc[obj.index[:-3]]
        expected = frame_or_series(
            range(2),
            index=date_range("2020", freq="D", periods=2).astype(
                "timestamp[us][pyarrow]"
            ),
        )
        tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\themes.py ===
from .default_styles import DEFAULT_STYLES
from .theme import Theme


DEFAULT = Theme(DEFAULT_STYLES)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\ply\__init__.py ===
# PLY package
# Author: David Beazley (dave@dabeaz.com)

__version__ = '3.9'
__all__ = ['lex','yacc']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\predicates\__init__.py ===
"""
Module to implement predicate classes.

Class of every predicate registered to ``Q`` is defined here.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\exceptions.py ===
"""Geometry Errors."""

class GeometryError(ValueError):
    """An exception raised by classes in the geometry module."""
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\_antlr\__init__.py ===
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\backends\matplotlibbackend\__init__.py ===
from sympy.plotting.backends.matplotlibbackend.matplotlib import (
    MatplotlibBackend, _matplotlib_list
)

__all__ = ["MatplotlibBackend", "_matplotlib_list"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\__init__.py ===
"""Module for algebraic geometry and commutative algebra."""

from .homomorphisms import homomorphism

__all__ = ['homomorphism']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\defaults.py ===
from sympy.core._print_helpers import Printable

# alias for compatibility
Printable.__module__ = __name__
DefaultPrinting = Printable

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\diophantine\__init__.py ===
from .diophantine import diophantine, classify_diop, diop_solve

__all__ = [
    'diophantine', 'classify_diop', 'diop_solve'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_dlpack.py ===
import sys

import pytest

import numpy as np
from numpy.testing import IS_PYPY, assert_array_equal


def new_and_old_dlpack():
    yield np.arange(5)

    class OldDLPack(np.ndarray):
        # Support only the "old" version
        def __dlpack__(self, stream=None):
            return super().__dlpack__(stream=None)

    yield np.arange(5).view(OldDLPack)


class TestDLPack:
    @pytest.mark.skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    @pytest.mark.parametrize("max_version", [(0, 0), None, (1, 0), (100, 3)])
    def test_dunder_dlpack_refcount(self, max_version):
        x = np.arange(5)
        y = x.__dlpack__(max_version=max_version)
        startcount = sys.getrefcount(x)
        del y
        assert startcount - sys.getrefcount(x) == 1

    def test_dunder_dlpack_stream(self):
        x = np.arange(5)
        x.__dlpack__(stream=None)

        with pytest.raises(RuntimeError):
            x.__dlpack__(stream=1)

    def test_dunder_dlpack_copy(self):
        # Checks the argument parsing of __dlpack__ explicitly.
        # Honoring the flag is tested in the from_dlpack round-tripping test.
        x = np.arange(5)
        x.__dlpack__(copy=True)
        x.__dlpack__(copy=None)
        x.__dlpack__(copy=False)

        with pytest.raises(ValueError):
            # NOTE: The copy converter should be stricter, but not just here.
            x.__dlpack__(copy=np.array([1, 2, 3]))

    def test_strides_not_multiple_of_itemsize(self):
        dt = np.dtype([('int', np.int32), ('char', np.int8)])
        y = np.zeros((5,), dtype=dt)
        z = y['int']

        with pytest.raises(BufferError):
            np.from_dlpack(z)

    @pytest.mark.skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    @pytest.mark.parametrize("arr", new_and_old_dlpack())
    def test_from_dlpack_refcount(self, arr):
        arr = arr.copy()
        y = np.from_dlpack(arr)
        startcount = sys.getrefcount(arr)
        del y
        assert startcount - sys.getrefcount(arr) == 1

    @pytest.mark.parametrize("dtype", [
        np.bool,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128
    ])
    @pytest.mark.parametrize("arr", new_and_old_dlpack())
    def test_dtype_passthrough(self, arr, dtype):
        x = arr.astype(dtype)
        y = np.from_dlpack(x)

        assert y.dtype == x.dtype
        assert_array_equal(x, y)

    def test_invalid_dtype(self):
        x = np.asarray(np.datetime64('2021-05-27'))

        with pytest.raises(BufferError):
            np.from_dlpack(x)

    def test_invalid_byte_swapping(self):
        dt = np.dtype('=i8').newbyteorder()
        x = np.arange(5, dtype=dt)

        with pytest.raises(BufferError):
            np.from_dlpack(x)

    def test_non_contiguous(self):
        x = np.arange(25).reshape((5, 5))

        y1 = x[0]
        assert_array_equal(y1, np.from_dlpack(y1))

        y2 = x[:, 0]
        assert_array_equal(y2, np.from_dlpack(y2))

        y3 = x[1, :]
        assert_array_equal(y3, np.from_dlpack(y3))

        y4 = x[1]
        assert_array_equal(y4, np.from_dlpack(y4))

        y5 = np.diagonal(x).copy()
        assert_array_equal(y5, np.from_dlpack(y5))

    @pytest.mark.parametrize("ndim", range(33))
    def test_higher_dims(self, ndim):
        shape = (1,) * ndim
        x = np.zeros(shape, dtype=np.float64)

        assert shape == np.from_dlpack(x).shape

    def test_dlpack_device(self):
        x = np.arange(5)
        assert x.__dlpack_device__() == (1, 0)
        y = np.from_dlpack(x)
        assert y.__dlpack_device__() == (1, 0)
        z = y[::2]
        assert z.__dlpack_device__() == (1, 0)

    def dlpack_deleter_exception(self, max_version):
        x = np.arange(5)
        _ = x.__dlpack__(max_version=max_version)
        raise RuntimeError

    @pytest.mark.parametrize("max_version", [None, (1, 0)])
    def test_dlpack_destructor_exception(self, max_version):
        with pytest.raises(RuntimeError):
            self.dlpack_deleter_exception(max_version=max_version)

    def test_readonly(self):
        x = np.arange(5)
        x.flags.writeable = False
        # Raises without max_version
        with pytest.raises(BufferError):
            x.__dlpack__()

        # But works fine if we try with version
        y = np.from_dlpack(x)
        assert not y.flags.writeable

    def test_writeable(self):
        x_new, x_old = new_and_old_dlpack()

        # new dlpacks respect writeability
        y = np.from_dlpack(x_new)
        assert y.flags.writeable

        # old dlpacks are not writeable for backwards compatibility
        y = np.from_dlpack(x_old)
        assert not y.flags.writeable

    def test_ndim0(self):
        x = np.array(1.0)
        y = np.from_dlpack(x)
        assert_array_equal(x, y)

    def test_size1dims_arrays(self):
        x = np.ndarray(dtype='f8', shape=(10, 5, 1), strides=(8, 80, 4),
                       buffer=np.ones(1000, dtype=np.uint8), order='F')
        y = np.from_dlpack(x)
        assert_array_equal(x, y)

    def test_copy(self):
        x = np.arange(5)

        y = np.from_dlpack(x)
        assert np.may_share_memory(x, y)
        y = np.from_dlpack(x, copy=False)
        assert np.may_share_memory(x, y)
        y = np.from_dlpack(x, copy=True)
        assert not np.may_share_memory(x, y)

    def test_device(self):
        x = np.arange(5)
        # requesting (1, 0), i.e. CPU device works in both calls:
        x.__dlpack__(dl_device=(1, 0))
        np.from_dlpack(x, device="cpu")
        np.from_dlpack(x, device=None)

        with pytest.raises(ValueError):
            x.__dlpack__(dl_device=(10, 0))
        with pytest.raises(ValueError):
            np.from_dlpack(x, device="gpu")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\test_constructors.py ===
from datetime import datetime
import sys

import numpy as np
import pytest

from pandas.compat import PYPY

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)


def series_via_frame_from_dict(x, **kwargs):
    return DataFrame({"a": x}, **kwargs)["a"]


def series_via_frame_from_scalar(x, **kwargs):
    return DataFrame(x, **kwargs)[0]


@pytest.fixture(
    params=[
        Series,
        series_via_frame_from_dict,
        series_via_frame_from_scalar,
        Index,
    ],
    ids=["Series", "DataFrame-dict", "DataFrame-array", "Index"],
)
def constructor(request):
    return request.param


class TestPandasDelegate:
    class Delegator:
        _properties = ["prop"]
        _methods = ["test_method"]

        def _set_prop(self, value):
            self.prop = value

        def _get_prop(self):
            return self.prop

        prop = property(_get_prop, _set_prop, doc="foo property")

        def test_method(self, *args, **kwargs):
            """a test method"""

    class Delegate(PandasDelegate, PandasObject):
        def __init__(self, obj) -> None:
            self.obj = obj

    def test_invalid_delegation(self):
        # these show that in order for the delegation to work
        # the _delegate_* methods need to be overridden to not raise
        # a TypeError

        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator,
            accessors=self.Delegator._properties,
            typ="property",
        )
        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator, accessors=self.Delegator._methods, typ="method"
        )

        delegate = self.Delegate(self.Delegator())

        msg = "You cannot access the property prop"
        with pytest.raises(TypeError, match=msg):
            delegate.prop

        msg = "The property prop cannot be set"
        with pytest.raises(TypeError, match=msg):
            delegate.prop = 5

        msg = "You cannot access the property prop"
        with pytest.raises(TypeError, match=msg):
            delegate.prop

    @pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
    def test_memory_usage(self):
        # Delegate does not implement memory_usage.
        # Check that we fall back to in-built `__sizeof__`
        # GH 12924
        delegate = self.Delegate(self.Delegator())
        sys.getsizeof(delegate)


class TestNoNewAttributesMixin:
    def test_mixin(self):
        class T(NoNewAttributesMixin):
            pass

        t = T()
        assert not hasattr(t, "__frozen")

        t.a = "test"
        assert t.a == "test"

        t._freeze()
        assert "__frozen" in dir(t)
        assert getattr(t, "__frozen")
        msg = "You cannot add any new attribute"
        with pytest.raises(AttributeError, match=msg):
            t.b = "test"

        assert not hasattr(t, "b")


class TestConstruction:
    # test certain constructor behaviours on dtype inference across Series,
    # Index and DataFrame

    @pytest.mark.parametrize(
        "a",
        [
            np.array(["2263-01-01"], dtype="datetime64[D]"),
            np.array([datetime(2263, 1, 1)], dtype=object),
            np.array([np.datetime64("2263-01-01", "D")], dtype=object),
            np.array(["2263-01-01"], dtype=object),
        ],
        ids=[
            "datetime64[D]",
            "object-datetime.datetime",
            "object-numpy-scalar",
            "object-string",
        ],
    )
    def test_constructor_datetime_outofbound(
        self, a, constructor, request, using_infer_string
    ):
        # GH-26853 (+ bug GH-26206 out of bound non-ns unit)

        # No dtype specified (dtype inference)
        # datetime64[non-ns] raise error, other cases result in object dtype
        # and preserve original data
        if a.dtype.kind == "M":
            # Can't fit in nanosecond bounds -> get the nearest supported unit
            result = constructor(a)
            assert result.dtype == "M8[s]"
        else:
            result = constructor(a)
            if using_infer_string and "object-string" in request.node.callspec.id:
                assert result.dtype == "string"
            else:
                assert result.dtype == "object"
            tm.assert_numpy_array_equal(result.to_numpy(), a)

        # Explicit dtype specified
        # Forced conversion fails for all -> all cases raise error
        msg = "Out of bounds|Out of bounds .* present at position 0"
        with pytest.raises(pd.errors.OutOfBoundsDatetime, match=msg):
            constructor(a, dtype="datetime64[ns]")

    def test_constructor_datetime_nonns(self, constructor):
        arr = np.array(["2020-01-01T00:00:00.000000"], dtype="datetime64[us]")
        dta = pd.core.arrays.DatetimeArray._simple_new(arr, dtype=arr.dtype)
        expected = constructor(dta)
        assert expected.dtype == arr.dtype

        result = constructor(arr)
        tm.assert_equal(result, expected)

        # https://github.com/pandas-dev/pandas/issues/34843
        arr.flags.writeable = False
        result = constructor(arr)
        tm.assert_equal(result, expected)

    def test_constructor_from_dict_keys(self, constructor, using_infer_string):
        # https://github.com/pandas-dev/pandas/issues/60343
        d = {"a": 1, "b": 2}
        result = constructor(d.keys(), dtype="str")
        if using_infer_string:
            assert result.dtype == "str"
        else:
            assert result.dtype == "object"
        expected = constructor(list(d.keys()), dtype="str")
        tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\test_misc.py ===
import sys

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import PYPY

from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_object_dtype,
)

import pandas as pd
from pandas import (
    Index,
    Series,
)
import pandas._testing as tm


def test_isnull_notnull_docstrings():
    # GH#41855 make sure its clear these are aliases
    doc = pd.DataFrame.notnull.__doc__
    assert doc.startswith("\nDataFrame.notnull is an alias for DataFrame.notna.\n")
    doc = pd.DataFrame.isnull.__doc__
    assert doc.startswith("\nDataFrame.isnull is an alias for DataFrame.isna.\n")

    doc = Series.notnull.__doc__
    assert doc.startswith("\nSeries.notnull is an alias for Series.notna.\n")
    doc = Series.isnull.__doc__
    assert doc.startswith("\nSeries.isnull is an alias for Series.isna.\n")


@pytest.mark.parametrize(
    "op_name, op",
    [
        ("add", "+"),
        ("sub", "-"),
        ("mul", "*"),
        ("mod", "%"),
        ("pow", "**"),
        ("truediv", "/"),
        ("floordiv", "//"),
    ],
)
def test_binary_ops_docstring(frame_or_series, op_name, op):
    # not using the all_arithmetic_functions fixture with _get_opstr
    # as _get_opstr is used internally in the dynamic implementation of the docstring
    klass = frame_or_series

    operand1 = klass.__name__.lower()
    operand2 = "other"
    expected_str = " ".join([operand1, op, operand2])
    assert expected_str in getattr(klass, op_name).__doc__

    # reverse version of the binary ops
    expected_str = " ".join([operand2, op, operand1])
    assert expected_str in getattr(klass, "r" + op_name).__doc__


def test_ndarray_compat_properties(index_or_series_obj):
    obj = index_or_series_obj

    # Check that we work.
    for p in ["shape", "dtype", "T", "nbytes"]:
        assert getattr(obj, p, None) is not None

    # deprecated properties
    for p in ["strides", "itemsize", "base", "data"]:
        assert not hasattr(obj, p)

    msg = "can only convert an array of size 1 to a Python scalar"
    with pytest.raises(ValueError, match=msg):
        obj.item()  # len > 1

    assert obj.ndim == 1
    assert obj.size == len(obj)

    assert Index([1]).item() == 1
    assert Series([1]).item() == 1


@pytest.mark.skipif(
    PYPY or using_string_dtype(),
    reason="not relevant for PyPy doesn't work properly for arrow strings",
)
def test_memory_usage(index_or_series_memory_obj):
    obj = index_or_series_memory_obj
    # Clear index caches so that len(obj) == 0 report 0 memory usage
    if isinstance(obj, Series):
        is_ser = True
        obj.index._engine.clear_mapping()
    else:
        is_ser = False
        obj._engine.clear_mapping()

    res = obj.memory_usage()
    res_deep = obj.memory_usage(deep=True)

    is_object = is_object_dtype(obj) or (is_ser and is_object_dtype(obj.index))
    is_categorical = isinstance(obj.dtype, pd.CategoricalDtype) or (
        is_ser and isinstance(obj.index.dtype, pd.CategoricalDtype)
    )
    is_object_string = is_dtype_equal(obj, "string[python]") or (
        is_ser and is_dtype_equal(obj.index.dtype, "string[python]")
    )

    if len(obj) == 0:
        expected = 0
        assert res_deep == res == expected
    elif is_object or is_categorical or is_object_string:
        # only deep will pick them up
        assert res_deep > res
    else:
        assert res == res_deep

    # sys.getsizeof will call the .memory_usage with
    # deep=True, and add on some GC overhead
    diff = res_deep - sys.getsizeof(obj)
    assert abs(diff) < 100


def test_memory_usage_components_series(series_with_simple_index):
    series = series_with_simple_index
    total_usage = series.memory_usage(index=True)
    non_index_usage = series.memory_usage(index=False)
    index_usage = series.index.memory_usage()
    assert total_usage == non_index_usage + index_usage


@pytest.mark.parametrize("dtype", tm.NARROW_NP_DTYPES)
def test_memory_usage_components_narrow_series(dtype):
    series = Series(range(5), dtype=dtype, index=[f"i-{i}" for i in range(5)], name="a")
    total_usage = series.memory_usage(index=True)
    non_index_usage = series.memory_usage(index=False)
    index_usage = series.index.memory_usage()
    assert total_usage == non_index_usage + index_usage


def test_searchsorted(request, index_or_series_obj):
    # numpy.searchsorted calls obj.searchsorted under the hood.
    # See gh-12238
    obj = index_or_series_obj

    if isinstance(obj, pd.MultiIndex):
        # See gh-14833
        request.applymarker(
            pytest.mark.xfail(
                reason="np.searchsorted doesn't work on pd.MultiIndex: GH 14833"
            )
        )
    elif obj.dtype.kind == "c" and isinstance(obj, Index):
        # TODO: Should Series cases also raise? Looks like they use numpy
        #  comparison semantics https://github.com/numpy/numpy/issues/15981
        mark = pytest.mark.xfail(reason="complex objects are not comparable")
        request.applymarker(mark)

    max_obj = max(obj, default=0)
    index = np.searchsorted(obj, max_obj)
    assert 0 <= index <= len(obj)

    index = np.searchsorted(obj, max_obj, sorter=range(len(obj)))
    assert 0 <= index <= len(obj)


@pytest.mark.filterwarnings(r"ignore:Dtype inference:FutureWarning")
def test_access_by_position(index_flat):
    index = index_flat

    if len(index) == 0:
        pytest.skip("Test doesn't make sense on empty data")

    series = Series(index)
    assert index[0] == series.iloc[0]
    assert index[5] == series.iloc[5]
    assert index[-1] == series.iloc[-1]

    size = len(index)
    assert index[-1] == index[size - 1]

    msg = f"index {size} is out of bounds for axis 0 with size {size}"
    if isinstance(index.dtype, pd.StringDtype) and index.dtype.storage == "pyarrow":
        msg = "index out of bounds"
    with pytest.raises(IndexError, match=msg):
        index[size]
    msg = "single positional indexer is out-of-bounds"
    with pytest.raises(IndexError, match=msg):
        series.iloc[size]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\missing.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class BaseMissingTests:
    def test_isna(self, data_missing):
        expected = np.array([True, False])

        result = pd.isna(data_missing)
        tm.assert_numpy_array_equal(result, expected)

        result = pd.Series(data_missing).isna()
        expected = pd.Series(expected)
        tm.assert_series_equal(result, expected)

        # GH 21189
        result = pd.Series(data_missing).drop([0, 1]).isna()
        expected = pd.Series([], dtype=bool)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_func", ["isna", "notna"])
    def test_isna_returns_copy(self, data_missing, na_func):
        result = pd.Series(data_missing)
        expected = result.copy()
        mask = getattr(result, na_func)()
        if isinstance(mask.dtype, pd.SparseDtype):
            # TODO: GH 57739
            mask = np.array(mask)
            mask.flags.writeable = True

        mask[:] = True
        tm.assert_series_equal(result, expected)

    def test_dropna_array(self, data_missing):
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_dropna_series(self, data_missing):
        ser = pd.Series(data_missing)
        result = ser.dropna()
        expected = ser.iloc[[1]]
        tm.assert_series_equal(result, expected)

    def test_dropna_frame(self, data_missing):
        df = pd.DataFrame({"A": data_missing}, columns=pd.Index(["A"], dtype=object))

        # defaults
        result = df.dropna()
        expected = df.iloc[[1]]
        tm.assert_frame_equal(result, expected)

        # axis = 1
        result = df.dropna(axis="columns")
        expected = pd.DataFrame(index=pd.RangeIndex(2), columns=pd.Index([]))
        tm.assert_frame_equal(result, expected)

        # multiple
        df = pd.DataFrame({"A": data_missing, "B": [1, np.nan]})
        result = df.dropna()
        expected = df.iloc[:0]
        tm.assert_frame_equal(result, expected)

    def test_fillna_scalar(self, data_missing):
        valid = data_missing[1]
        result = data_missing.fillna(valid)
        expected = data_missing.fillna(valid)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:Series.fillna with 'method' is deprecated:FutureWarning"
    )
    def test_fillna_limit_pad(self, data_missing):
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).ffill(limit=2)
        expected = pd.Series(data_missing.take([1, 1, 1, 0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),
        ],
    )
    def test_ffill_limit_area(
        self, data_missing, limit_area, input_ilocs, expected_ilocs
    ):
        # GH#56616
        arr = data_missing.take(input_ilocs)
        result = pd.Series(arr).ffill(limit_area=limit_area)
        expected = pd.Series(data_missing.take(expected_ilocs))
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:Series.fillna with 'method' is deprecated:FutureWarning"
    )
    def test_fillna_limit_backfill(self, data_missing):
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).fillna(method="backfill", limit=2)
        expected = pd.Series(data_missing.take([1, 0, 1, 1, 1]))
        tm.assert_series_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

        result = data._pad_or_backfill(method="backfill")
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        result = ser.fillna(fill_value)
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )
        tm.assert_series_equal(result, expected)

        # Fill with a series
        result = ser.fillna(expected)
        tm.assert_series_equal(result, expected)

        # Fill with a series not affecting the missing values
        result = ser.fillna(ser)
        tm.assert_series_equal(result, ser)

    def test_fillna_series_method(self, data_missing, fillna_method):
        fill_value = data_missing[1]

        if fillna_method == "ffill":
            data_missing = data_missing[::-1]

        result = getattr(pd.Series(data_missing), fillna_method)()
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )

        tm.assert_series_equal(result, expected)

    def test_fillna_frame(self, data_missing):
        fill_value = data_missing[1]

        result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

        expected = pd.DataFrame(
            {
                "A": data_missing._from_sequence(
                    [fill_value, fill_value], dtype=data_missing.dtype
                ),
                "B": [1, 2],
            }
        )

        tm.assert_frame_equal(result, expected)

    def test_fillna_fill_other(self, data):
        result = pd.DataFrame({"A": data, "B": [np.nan] * len(data)}).fillna({"B": 0.0})

        expected = pd.DataFrame({"A": data, "B": [0.0] * len(result)})

        tm.assert_frame_equal(result, expected)

    def test_use_inf_as_na_no_effect(self, data_missing):
        ser = pd.Series(data_missing)
        expected = ser.isna()
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context("mode.use_inf_as_na", True):
                result = ser.isna()
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_drop.py ===
import numpy as np
import pytest

from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


def test_drop(idx):
    dropped = idx.drop([("foo", "two"), ("qux", "one")])

    index = MultiIndex.from_tuples([("foo", "two"), ("qux", "one")])
    dropped2 = idx.drop(index)

    expected = idx[[0, 2, 3, 5]]
    tm.assert_index_equal(dropped, expected)
    tm.assert_index_equal(dropped2, expected)

    dropped = idx.drop(["bar"])
    expected = idx[[0, 1, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    dropped = idx.drop("foo")
    expected = idx[[2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    index = MultiIndex.from_tuples([("bar", "two")])
    with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
        idx.drop([("bar", "two")])
    with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
        idx.drop(index)
    with pytest.raises(KeyError, match=r"^'two'$"):
        idx.drop(["foo", "two"])

    # partially correct argument
    mixed_index = MultiIndex.from_tuples([("qux", "one"), ("bar", "two")])
    with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
        idx.drop(mixed_index)

    # error='ignore'
    dropped = idx.drop(index, errors="ignore")
    expected = idx[[0, 1, 2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    dropped = idx.drop(mixed_index, errors="ignore")
    expected = idx[[0, 1, 2, 3, 5]]
    tm.assert_index_equal(dropped, expected)

    dropped = idx.drop(["foo", "two"], errors="ignore")
    expected = idx[[2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    # mixed partial / full drop
    dropped = idx.drop(["foo", ("qux", "one")])
    expected = idx[[2, 3, 5]]
    tm.assert_index_equal(dropped, expected)

    # mixed partial / full drop / error='ignore'
    mixed_index = ["foo", ("qux", "one"), "two"]
    with pytest.raises(KeyError, match=r"^'two'$"):
        idx.drop(mixed_index)
    dropped = idx.drop(mixed_index, errors="ignore")
    expected = idx[[2, 3, 5]]
    tm.assert_index_equal(dropped, expected)


def test_droplevel_with_names(idx):
    index = idx[idx.get_loc("foo")]
    dropped = index.droplevel(0)
    assert dropped.name == "second"

    index = MultiIndex(
        levels=[Index(range(4)), Index(range(4)), Index(range(4))],
        codes=[
            np.array([0, 0, 1, 2, 2, 2, 3, 3]),
            np.array([0, 1, 0, 0, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        ],
        names=["one", "two", "three"],
    )
    dropped = index.droplevel(0)
    assert dropped.names == ("two", "three")

    dropped = index.droplevel("two")
    expected = index.droplevel(1)
    assert dropped.equals(expected)


def test_droplevel_list():
    index = MultiIndex(
        levels=[Index(range(4)), Index(range(4)), Index(range(4))],
        codes=[
            np.array([0, 0, 1, 2, 2, 2, 3, 3]),
            np.array([0, 1, 0, 0, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        ],
        names=["one", "two", "three"],
    )

    dropped = index[:2].droplevel(["three", "one"])
    expected = index[:2].droplevel(2).droplevel(0)
    assert dropped.equals(expected)

    dropped = index[:2].droplevel([])
    expected = index[:2]
    assert dropped.equals(expected)

    msg = (
        "Cannot remove 3 levels from an index with 3 levels: "
        "at least one level must be left"
    )
    with pytest.raises(ValueError, match=msg):
        index[:2].droplevel(["one", "two", "three"])

    with pytest.raises(KeyError, match="'Level four not found'"):
        index[:2].droplevel(["one", "four"])


def test_drop_not_lexsorted():
    # GH 12078

    # define the lexsorted version of the multi-index
    tuples = [("a", ""), ("b1", "c1"), ("b2", "c2")]
    lexsorted_mi = MultiIndex.from_tuples(tuples, names=["b", "c"])
    assert lexsorted_mi._is_lexsorted()

    # and the not-lexsorted version
    df = pd.DataFrame(
        columns=["a", "b", "c", "d"], data=[[1, "b1", "c1", 3], [1, "b2", "c2", 4]]
    )
    df = df.pivot_table(index="a", columns=["b", "c"], values="d")
    df = df.reset_index()
    not_lexsorted_mi = df.columns
    assert not not_lexsorted_mi._is_lexsorted()

    # compare the results
    tm.assert_index_equal(lexsorted_mi, not_lexsorted_mi)
    with tm.assert_produces_warning(PerformanceWarning):
        tm.assert_index_equal(lexsorted_mi.drop("a"), not_lexsorted_mi.drop("a"))


def test_drop_with_nan_in_index(nulls_fixture):
    # GH#18853
    mi = MultiIndex.from_tuples([("blah", nulls_fixture)], names=["name", "date"])
    msg = r"labels \[Timestamp\('2001-01-01 00:00:00'\)\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop(pd.Timestamp("2001"), level="date")


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_drop_with_non_monotonic_duplicates():
    # GH#33494
    mi = MultiIndex.from_tuples([(1, 2), (2, 3), (1, 2)])
    result = mi.drop((1, 2))
    expected = MultiIndex.from_tuples([(2, 3)])
    tm.assert_index_equal(result, expected)


def test_single_level_drop_partially_missing_elements():
    # GH 37820

    mi = MultiIndex.from_tuples([(1, 2), (2, 2), (3, 2)])
    msg = r"labels \[4\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop(4, level=0)
    with pytest.raises(KeyError, match=msg):
        mi.drop([1, 4], level=0)
    msg = r"labels \[nan\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan], level=0)
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan, 1, 2, 3], level=0)

    mi = MultiIndex.from_tuples([(np.nan, 1), (1, 2)])
    msg = r"labels \['a'\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan, 1, "a"], level=0)


def test_droplevel_multiindex_one_level():
    # GH#37208
    index = MultiIndex.from_tuples([(2,)], names=("b",))
    result = index.droplevel([])
    expected = Index([2], name="b")
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\tests\initialise_test.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
import sys
from unittest import TestCase, main, skipUnless

try:
    from unittest.mock import patch, Mock
except ImportError:
    from mock import patch, Mock

from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by

orig_stdout = sys.stdout
orig_stderr = sys.stderr


class InitTest(TestCase):

    @skipUnless(sys.stdout.isatty(), "sys.stdout is not a tty")
    def setUp(self):
        # sanity check
        self.assertNotWrapped()

    def tearDown(self):
        _wipe_internal_state_for_tests()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    def assertWrapped(self):
        self.assertIsNot(sys.stdout, orig_stdout, 'stdout should be wrapped')
        self.assertIsNot(sys.stderr, orig_stderr, 'stderr should be wrapped')
        self.assertTrue(isinstance(sys.stdout, StreamWrapper),
            'bad stdout wrapper')
        self.assertTrue(isinstance(sys.stderr, StreamWrapper),
            'bad stderr wrapper')

    def assertNotWrapped(self):
        self.assertIs(sys.stdout, orig_stdout, 'stdout should not be wrapped')
        self.assertIs(sys.stderr, orig_stderr, 'stderr should not be wrapped')

    @patch('colorama.initialise.reset_all')
    @patch('colorama.ansitowin32.winapi_test', lambda *_: True)
    @patch('colorama.ansitowin32.enable_vt_processing', lambda *_: False)
    def testInitWrapsOnWindows(self, _):
        with osname("nt"):
            init()
            self.assertWrapped()

    @patch('colorama.initialise.reset_all')
    @patch('colorama.ansitowin32.winapi_test', lambda *_: False)
    def testInitDoesntWrapOnEmulatedWindows(self, _):
        with osname("nt"):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapOnNonWindows(self):
        with osname("posix"):
            init()
            self.assertNotWrapped()

    def testInitDoesntWrapIfNone(self):
        with replace_by(None):
            init()
            # We can't use assertNotWrapped here because replace_by(None)
            # changes stdout/stderr already.
            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)

    def testInitAutoresetOnWrapsOnAllPlatforms(self):
        with osname("posix"):
            init(autoreset=True)
            self.assertWrapped()

    def testInitWrapOffDoesntWrapOnWindows(self):
        with osname("nt"):
            init(wrap=False)
            self.assertNotWrapped()

    def testInitWrapOffIncompatibleWithAutoresetOn(self):
        self.assertRaises(ValueError, lambda: init(autoreset=True, wrap=False))

    @patch('colorama.win32.SetConsoleTextAttribute')
    @patch('colorama.initialise.AnsiToWin32')
    def testAutoResetPassedOn(self, mockATW32, _):
        with osname("nt"):
            init(autoreset=True)
            self.assertEqual(len(mockATW32.call_args_list), 2)
            self.assertEqual(mockATW32.call_args_list[1][1]['autoreset'], True)
            self.assertEqual(mockATW32.call_args_list[0][1]['autoreset'], True)

    @patch('colorama.initialise.AnsiToWin32')
    def testAutoResetChangeable(self, mockATW32):
        with osname("nt"):
            init()

            init(autoreset=True)
            self.assertEqual(len(mockATW32.call_args_list), 4)
            self.assertEqual(mockATW32.call_args_list[2][1]['autoreset'], True)
            self.assertEqual(mockATW32.call_args_list[3][1]['autoreset'], True)

            init()
            self.assertEqual(len(mockATW32.call_args_list), 6)
            self.assertEqual(
                mockATW32.call_args_list[4][1]['autoreset'], False)
            self.assertEqual(
                mockATW32.call_args_list[5][1]['autoreset'], False)


    @patch('colorama.initialise.atexit.register')
    def testAtexitRegisteredOnlyOnce(self, mockRegister):
        init()
        self.assertTrue(mockRegister.called)
        mockRegister.reset_mock()
        init()
        self.assertFalse(mockRegister.called)


class JustFixWindowsConsoleTest(TestCase):
    def _reset(self):
        _wipe_internal_state_for_tests()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    def tearDown(self):
        self._reset()

    @patch("colorama.ansitowin32.winapi_test", lambda: True)
    def testJustFixWindowsConsole(self):
        if sys.platform != "win32":
            # just_fix_windows_console should be a no-op
            just_fix_windows_console()
            self.assertIs(sys.stdout, orig_stdout)
            self.assertIs(sys.stderr, orig_stderr)
        else:
            def fake_std():
                # Emulate stdout=not a tty, stderr=tty
                # to check that we handle both cases correctly
                stdout = Mock()
                stdout.closed = False
                stdout.isatty.return_value = False
                stdout.fileno.return_value = 1
                sys.stdout = stdout

                stderr = Mock()
                stderr.closed = False
                stderr.isatty.return_value = True
                stderr.fileno.return_value = 2
                sys.stderr = stderr

            for native_ansi in [False, True]:
                with patch(
                    'colorama.ansitowin32.enable_vt_processing',
                    lambda *_: native_ansi
                ):
                    self._reset()
                    fake_std()

                    # Regular single-call test
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(sys.stdout, prev_stdout)
                    if native_ansi:
                        self.assertIs(sys.stderr, prev_stderr)
                    else:
                        self.assertIsNot(sys.stderr, prev_stderr)

                    # second call without resetting is always a no-op
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(sys.stdout, prev_stdout)
                    self.assertIs(sys.stderr, prev_stderr)

                    self._reset()
                    fake_std()

                    # If init() runs first, just_fix_windows_console should be a no-op
                    init()
                    prev_stdout = sys.stdout
                    prev_stderr = sys.stderr
                    just_fix_windows_console()
                    self.assertIs(prev_stdout, sys.stdout)
                    self.assertIs(prev_stderr, sys.stderr)


if __name__ == '__main__':
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_numpy_compat.py ===
import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    PeriodIndex,
    TimedeltaIndex,
    isna,
)
import pandas._testing as tm
from pandas.api.types import (
    is_complex_dtype,
    is_numeric_dtype,
)
from pandas.core.arrays import BooleanArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin


def test_numpy_ufuncs_out(index):
    result = index == index

    out = np.empty(index.shape, dtype=bool)
    np.equal(index, index, out=out)
    tm.assert_numpy_array_equal(out, result)

    if not index._is_multi:
        # same thing on the ExtensionArray
        out = np.empty(index.shape, dtype=bool)
        np.equal(index.array, index.array, out=out)
        tm.assert_numpy_array_equal(out, result)


@pytest.mark.parametrize(
    "func",
    [
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        np.sqrt,
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.deg2rad,
        np.rad2deg,
    ],
    ids=lambda x: x.__name__,
)
def test_numpy_ufuncs_basic(index, func):
    # test ufuncs of numpy, see:
    # https://numpy.org/doc/stable/reference/ufuncs.html

    if isinstance(index, DatetimeIndexOpsMixin):
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all="ignore"):
                func(index)
    elif is_numeric_dtype(index) and not (
        is_complex_dtype(index) and func in [np.deg2rad, np.rad2deg]
    ):
        # coerces to float (e.g. np.sin)
        with np.errstate(all="ignore"):
            result = func(index)
            arr_result = func(index.values)
            if arr_result.dtype == np.float16:
                arr_result = arr_result.astype(np.float32)
            exp = Index(arr_result, name=index.name)

        tm.assert_index_equal(result, exp)
        if isinstance(index.dtype, np.dtype) and is_numeric_dtype(index):
            if is_complex_dtype(index):
                assert result.dtype == index.dtype
            elif index.dtype in ["bool", "int8", "uint8"]:
                assert result.dtype in ["float16", "float32"]
            elif index.dtype in ["int16", "uint16", "float32"]:
                assert result.dtype == "float32"
            else:
                assert result.dtype == "float64"
        else:
            # e.g. np.exp with Int64 -> Float64
            assert type(result) is Index
    # raise AttributeError or TypeError
    elif len(index) == 0:
        pass
    else:
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all="ignore"):
                func(index)


@pytest.mark.parametrize(
    "func", [np.isfinite, np.isinf, np.isnan, np.signbit], ids=lambda x: x.__name__
)
def test_numpy_ufuncs_other(index, func):
    # test ufuncs of numpy, see:
    # https://numpy.org/doc/stable/reference/ufuncs.html
    if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
        if func in (np.isfinite, np.isinf, np.isnan):
            # numpy 1.18 changed isinf and isnan to not raise on dt64/td64
            result = func(index)
            assert isinstance(result, np.ndarray)

            out = np.empty(index.shape, dtype=bool)
            func(index, out=out)
            tm.assert_numpy_array_equal(out, result)
        else:
            with tm.external_error_raised(TypeError):
                func(index)

    elif isinstance(index, PeriodIndex):
        with tm.external_error_raised(TypeError):
            func(index)

    elif is_numeric_dtype(index) and not (
        is_complex_dtype(index) and func is np.signbit
    ):
        # Results in bool array
        result = func(index)
        if not isinstance(index.dtype, np.dtype):
            # e.g. Int64 we expect to get BooleanArray back
            assert isinstance(result, BooleanArray)
        else:
            assert isinstance(result, np.ndarray)

        out = np.empty(index.shape, dtype=bool)
        func(index, out=out)

        if not isinstance(index.dtype, np.dtype):
            tm.assert_numpy_array_equal(out, result._data)
        else:
            tm.assert_numpy_array_equal(out, result)

    elif len(index) == 0:
        pass
    else:
        with tm.external_error_raised(TypeError):
            func(index)


@pytest.mark.parametrize("func", [np.maximum, np.minimum])
def test_numpy_ufuncs_reductions(index, func, request):
    # TODO: overlap with tests.series.test_ufunc.test_reductions
    if len(index) == 0:
        pytest.skip("Test doesn't make sense for empty index.")

    if isinstance(index, CategoricalIndex) and index.dtype.ordered is False:
        with pytest.raises(TypeError, match="is not ordered for"):
            func.reduce(index)
        return
    else:
        result = func.reduce(index)

    if func is np.maximum:
        expected = index.max(skipna=False)
    else:
        expected = index.min(skipna=False)
        # TODO: do we have cases both with and without NAs?

    assert type(result) is type(expected)
    if isna(result):
        assert isna(expected)
    else:
        assert result == expected


@pytest.mark.parametrize("func", [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
def test_numpy_ufuncs_bitwise(func):
    # https://github.com/pandas-dev/pandas/issues/46769
    idx1 = Index([1, 2, 3, 4], dtype="int64")
    idx2 = Index([3, 4, 5, 6], dtype="int64")

    with tm.assert_produces_warning(None):
        result = func(idx1, idx2)

    expected = Index(func(idx1.values, idx2.values))
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_asfreq.py ===
import re

import pytest

from pandas import (
    PeriodIndex,
    Series,
    period_range,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestPeriodIndex:
    def test_asfreq(self):
        pi1 = period_range(freq="Y", start="1/1/2001", end="1/1/2001")
        pi2 = period_range(freq="Q", start="1/1/2001", end="1/1/2001")
        pi3 = period_range(freq="M", start="1/1/2001", end="1/1/2001")
        pi4 = period_range(freq="D", start="1/1/2001", end="1/1/2001")
        pi5 = period_range(freq="h", start="1/1/2001", end="1/1/2001 00:00")
        pi6 = period_range(freq="Min", start="1/1/2001", end="1/1/2001 00:00")
        pi7 = period_range(freq="s", start="1/1/2001", end="1/1/2001 00:00:00")

        assert pi1.asfreq("Q", "s") == pi2
        assert pi1.asfreq("Q", "s") == pi2
        assert pi1.asfreq("M", "start") == pi3
        assert pi1.asfreq("D", "StarT") == pi4
        assert pi1.asfreq("h", "beGIN") == pi5
        assert pi1.asfreq("Min", "s") == pi6
        assert pi1.asfreq("s", "s") == pi7

        assert pi2.asfreq("Y", "s") == pi1
        assert pi2.asfreq("M", "s") == pi3
        assert pi2.asfreq("D", "s") == pi4
        assert pi2.asfreq("h", "s") == pi5
        assert pi2.asfreq("Min", "s") == pi6
        assert pi2.asfreq("s", "s") == pi7

        assert pi3.asfreq("Y", "s") == pi1
        assert pi3.asfreq("Q", "s") == pi2
        assert pi3.asfreq("D", "s") == pi4
        assert pi3.asfreq("h", "s") == pi5
        assert pi3.asfreq("Min", "s") == pi6
        assert pi3.asfreq("s", "s") == pi7

        assert pi4.asfreq("Y", "s") == pi1
        assert pi4.asfreq("Q", "s") == pi2
        assert pi4.asfreq("M", "s") == pi3
        assert pi4.asfreq("h", "s") == pi5
        assert pi4.asfreq("Min", "s") == pi6
        assert pi4.asfreq("s", "s") == pi7

        assert pi5.asfreq("Y", "s") == pi1
        assert pi5.asfreq("Q", "s") == pi2
        assert pi5.asfreq("M", "s") == pi3
        assert pi5.asfreq("D", "s") == pi4
        assert pi5.asfreq("Min", "s") == pi6
        assert pi5.asfreq("s", "s") == pi7

        assert pi6.asfreq("Y", "s") == pi1
        assert pi6.asfreq("Q", "s") == pi2
        assert pi6.asfreq("M", "s") == pi3
        assert pi6.asfreq("D", "s") == pi4
        assert pi6.asfreq("h", "s") == pi5
        assert pi6.asfreq("s", "s") == pi7

        assert pi7.asfreq("Y", "s") == pi1
        assert pi7.asfreq("Q", "s") == pi2
        assert pi7.asfreq("M", "s") == pi3
        assert pi7.asfreq("D", "s") == pi4
        assert pi7.asfreq("h", "s") == pi5
        assert pi7.asfreq("Min", "s") == pi6

        msg = "How must be one of S or E"
        with pytest.raises(ValueError, match=msg):
            pi7.asfreq("T", "foo")
        result1 = pi1.asfreq("3M")
        result2 = pi1.asfreq("M")
        expected = period_range(freq="M", start="2001-12", end="2001-12")
        tm.assert_numpy_array_equal(result1.asi8, expected.asi8)
        assert result1.freqstr == "3M"
        tm.assert_numpy_array_equal(result2.asi8, expected.asi8)
        assert result2.freqstr == "M"

    def test_asfreq_nat(self):
        idx = PeriodIndex(["2011-01", "2011-02", "NaT", "2011-04"], freq="M")
        result = idx.asfreq(freq="Q")
        expected = PeriodIndex(["2011Q1", "2011Q1", "NaT", "2011Q2"], freq="Q")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("freq", ["D", "3D"])
    def test_asfreq_mult_pi(self, freq):
        pi = PeriodIndex(["2001-01", "2001-02", "NaT", "2001-03"], freq="2M")

        result = pi.asfreq(freq)
        exp = PeriodIndex(["2001-02-28", "2001-03-31", "NaT", "2001-04-30"], freq=freq)
        tm.assert_index_equal(result, exp)
        assert result.freq == exp.freq

        result = pi.asfreq(freq, how="S")
        exp = PeriodIndex(["2001-01-01", "2001-02-01", "NaT", "2001-03-01"], freq=freq)
        tm.assert_index_equal(result, exp)
        assert result.freq == exp.freq

    def test_asfreq_combined_pi(self):
        pi = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq="h")
        exp = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq="25h")
        for freq, how in zip(["1D1h", "1h1D"], ["S", "E"]):
            result = pi.asfreq(freq, how=how)
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

        for freq in ["1D1h", "1h1D"]:
            pi = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq=freq)
            result = pi.asfreq("h")
            exp = PeriodIndex(["2001-01-02 00:00", "2001-01-03 02:00", "NaT"], freq="h")
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

            pi = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq=freq)
            result = pi.asfreq("h", how="S")
            exp = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq="h")
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

    def test_astype_asfreq(self):
        pi1 = PeriodIndex(["2011-01-01", "2011-02-01", "2011-03-01"], freq="D")
        exp = PeriodIndex(["2011-01", "2011-02", "2011-03"], freq="M")
        tm.assert_index_equal(pi1.asfreq("M"), exp)
        tm.assert_index_equal(pi1.astype("period[M]"), exp)

        exp = PeriodIndex(["2011-01", "2011-02", "2011-03"], freq="3M")
        tm.assert_index_equal(pi1.asfreq("3M"), exp)
        tm.assert_index_equal(pi1.astype("period[3M]"), exp)

    def test_asfreq_with_different_n(self):
        ser = Series([1, 2], index=PeriodIndex(["2020-01", "2020-03"], freq="2M"))
        result = ser.asfreq("M")

        excepted = Series([1, 2], index=PeriodIndex(["2020-02", "2020-04"], freq="M"))
        tm.assert_series_equal(result, excepted)

    @pytest.mark.parametrize(
        "freq",
        [
            "2BMS",
            "2YS-MAR",
            "2bh",
        ],
    )
    def test_pi_asfreq_not_supported_frequency(self, freq):
        # GH#55785
        msg = f"{freq[1:]} is not supported as period frequency"

        pi = PeriodIndex(["2020-01-01", "2021-01-01"], freq="M")
        with pytest.raises(ValueError, match=msg):
            pi.asfreq(freq=freq)

    @pytest.mark.parametrize(
        "freq",
        [
            "2BME",
            "2YE-MAR",
            "2QE",
        ],
    )
    def test_pi_asfreq_invalid_frequency(self, freq):
        # GH#55785
        msg = f"Invalid frequency: {freq}"

        pi = PeriodIndex(["2020-01-01", "2021-01-01"], freq="M")
        with pytest.raises(ValueError, match=msg):
            pi.asfreq(freq=freq)

    @pytest.mark.parametrize(
        "freq",
        [
            offsets.MonthBegin(2),
            offsets.BusinessMonthEnd(2),
        ],
    )
    def test_pi_asfreq_invalid_baseoffset(self, freq):
        # GH#56945
        msg = re.escape(f"{freq} is not supported as period frequency")

        pi = PeriodIndex(["2020-01-01", "2021-01-01"], freq="M")
        with pytest.raises(ValueError, match=msg):
            pi.asfreq(freq=freq)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_bitwise.py ===
"""
Test bit-level integer and mpf operations
"""

from mpmath import *
from mpmath.libmp import *

def test_bitcount():
    assert bitcount(0) == 0
    assert bitcount(1) == 1
    assert bitcount(7) == 3
    assert bitcount(8) == 4
    assert bitcount(2**100) == 101
    assert bitcount(2**100-1) == 100

def test_trailing():
    assert trailing(0) == 0
    assert trailing(1) == 0
    assert trailing(2) == 1
    assert trailing(7) == 0
    assert trailing(8) == 3
    assert trailing(2**100) == 100
    assert trailing(2**100-1) == 0

def test_round_down():
    assert from_man_exp(0, -4, 4, round_down)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(0xff, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(-0xf0, -4, 4, round_down)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_down)[:3] == (1, 15, 0)
    assert from_man_exp(-0xff, -4, 4, round_down)[:3] == (1, 15, 0)

def test_round_up():
    assert from_man_exp(0, -4, 4, round_up)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_up)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_up)[:3] == (0, 1, 4)
    assert from_man_exp(0xff, -4, 4, round_up)[:3] == (0, 1, 4)
    assert from_man_exp(-0xf0, -4, 4, round_up)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_up)[:3] == (1, 1, 4)
    assert from_man_exp(-0xff, -4, 4, round_up)[:3] == (1, 1, 4)

def test_round_floor():
    assert from_man_exp(0, -4, 4, round_floor)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(0xff, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(-0xf0, -4, 4, round_floor)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_floor)[:3] == (1, 1, 4)
    assert from_man_exp(-0xff, -4, 4, round_floor)[:3] == (1, 1, 4)

def test_round_ceiling():
    assert from_man_exp(0, -4, 4, round_ceiling)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_ceiling)[:3] == (0, 15, 0)
    assert from_man_exp(0xf1, -4, 4, round_ceiling)[:3] == (0, 1, 4)
    assert from_man_exp(0xff, -4, 4, round_ceiling)[:3] == (0, 1, 4)
    assert from_man_exp(-0xf0, -4, 4, round_ceiling)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf1, -4, 4, round_ceiling)[:3] == (1, 15, 0)
    assert from_man_exp(-0xff, -4, 4, round_ceiling)[:3] == (1, 15, 0)

def test_round_nearest():
    assert from_man_exp(0, -4, 4, round_nearest)[:3] == (0, 0, 0)
    assert from_man_exp(0xf0, -4, 4, round_nearest)[:3] == (0, 15, 0)
    assert from_man_exp(0xf7, -4, 4, round_nearest)[:3] == (0, 15, 0)
    assert from_man_exp(0xf8, -4, 4, round_nearest)[:3] == (0, 1, 4)    # 1111.1000 -> 10000.0
    assert from_man_exp(0xf9, -4, 4, round_nearest)[:3] == (0, 1, 4)    # 1111.1001 -> 10000.0
    assert from_man_exp(0xe8, -4, 4, round_nearest)[:3] == (0, 7, 1)    # 1110.1000 -> 1110.0
    assert from_man_exp(0xe9, -4, 4, round_nearest)[:3] == (0, 15, 0)     # 1110.1001 -> 1111.0
    assert from_man_exp(-0xf0, -4, 4, round_nearest)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf7, -4, 4, round_nearest)[:3] == (1, 15, 0)
    assert from_man_exp(-0xf8, -4, 4, round_nearest)[:3] == (1, 1, 4)
    assert from_man_exp(-0xf9, -4, 4, round_nearest)[:3] == (1, 1, 4)
    assert from_man_exp(-0xe8, -4, 4, round_nearest)[:3] == (1, 7, 1)
    assert from_man_exp(-0xe9, -4, 4, round_nearest)[:3] == (1, 15, 0)

def test_rounding_bugs():
    # 1 less than power-of-two cases
    assert from_man_exp(72057594037927935, -56, 53, round_up) == (0, 1, 0, 1)
    assert from_man_exp(73786976294838205979, -65, 53, round_nearest) == (0, 1, 1, 1)
    assert from_man_exp(31, 0, 4, round_up) == (0, 1, 5, 1)
    assert from_man_exp(-31, 0, 4, round_floor) == (1, 1, 5, 1)
    assert from_man_exp(255, 0, 7, round_up) == (0, 1, 8, 1)
    assert from_man_exp(-255, 0, 7, round_floor) == (1, 1, 8, 1)

def test_rounding_issue_200():
    a = from_man_exp(9867,-100)
    b = from_man_exp(9867,-200)
    c = from_man_exp(-1,0)
    z = (1, 1023, -10, 10)
    assert mpf_add(a, c, 10, 'd') == z
    assert mpf_add(b, c, 10, 'd') == z
    assert mpf_add(c, a, 10, 'd') == z
    assert mpf_add(c, b, 10, 'd') == z

def test_perturb():
    a = fone
    b = from_float(0.99999999999999989)
    c = from_float(1.0000000000000002)
    assert mpf_perturb(a, 0, 53, round_nearest) == a
    assert mpf_perturb(a, 1, 53, round_nearest) == a
    assert mpf_perturb(a, 0, 53, round_up) == c
    assert mpf_perturb(a, 0, 53, round_ceiling) == c
    assert mpf_perturb(a, 0, 53, round_down) == a
    assert mpf_perturb(a, 0, 53, round_floor) == a
    assert mpf_perturb(a, 1, 53, round_up) == a
    assert mpf_perturb(a, 1, 53, round_ceiling) == a
    assert mpf_perturb(a, 1, 53, round_down) == b
    assert mpf_perturb(a, 1, 53, round_floor) == b
    a = mpf_neg(a)
    b = mpf_neg(b)
    c = mpf_neg(c)
    assert mpf_perturb(a, 0, 53, round_nearest) == a
    assert mpf_perturb(a, 1, 53, round_nearest) == a
    assert mpf_perturb(a, 0, 53, round_up) == a
    assert mpf_perturb(a, 0, 53, round_floor) == a
    assert mpf_perturb(a, 0, 53, round_down) == b
    assert mpf_perturb(a, 0, 53, round_ceiling) == b
    assert mpf_perturb(a, 1, 53, round_up) == c
    assert mpf_perturb(a, 1, 53, round_floor) == c
    assert mpf_perturb(a, 1, 53, round_down) == a
    assert mpf_perturb(a, 1, 53, round_ceiling) == a

def test_add_exact():
    ff = from_float
    assert mpf_add(ff(3.0), ff(2.5)) == ff(5.5)
    assert mpf_add(ff(3.0), ff(-2.5)) == ff(0.5)
    assert mpf_add(ff(-3.0), ff(2.5)) == ff(-0.5)
    assert mpf_add(ff(-3.0), ff(-2.5)) == ff(-5.5)
    assert mpf_sub(mpf_add(fone, ff(1e-100)), fone) == ff(1e-100)
    assert mpf_sub(mpf_add(ff(1e-100), fone), fone) == ff(1e-100)
    assert mpf_sub(mpf_add(fone, ff(-1e-100)), fone) == ff(-1e-100)
    assert mpf_sub(mpf_add(ff(-1e-100), fone), fone) == ff(-1e-100)
    assert mpf_add(fone, fzero) == fone
    assert mpf_add(fzero, fone) == fone
    assert mpf_add(fzero, fzero) == fzero

def test_long_exponent_shifts():
    mp.dps = 15
    # Check for possible bugs due to exponent arithmetic overflow
    # in a C implementation
    x = mpf(1)
    for p in [32, 64]:
        a = ldexp(1,2**(p-1))
        b = ldexp(1,2**p)
        c = ldexp(1,2**(p+1))
        d = ldexp(1,-2**(p-1))
        e = ldexp(1,-2**p)
        f = ldexp(1,-2**(p+1))
        assert (x+a) == a
        assert (x+b) == b
        assert (x+c) == c
        assert (x+d) == x
        assert (x+e) == x
        assert (x+f) == x
        assert (a+x) == a
        assert (b+x) == b
        assert (c+x) == c
        assert (d+x) == x
        assert (e+x) == x
        assert (f+x) == x
        assert (x-a) == -a
        assert (x-b) == -b
        assert (x-c) == -c
        assert (x-d) == x
        assert (x-e) == x
        assert (x-f) == x
        assert (a-x) == a
        assert (b-x) == b
        assert (c-x) == c
        assert (d-x) == -x
        assert (e-x) == -x
        assert (f-x) == -x

def test_float_rounding():
    mp.prec = 64
    for x in [mpf(1), mpf(1)+eps, mpf(1)-eps, -mpf(1)+eps, -mpf(1)-eps]:
        fa = float(x)
        fb = float(fadd(x,0,prec=53,rounding='n'))
        assert fa == fb
        z = mpc(x,x)
        ca = complex(z)
        cb = complex(fadd(z,0,prec=53,rounding='n'))
        assert ca == cb
        for rnd in ['n', 'd', 'u', 'f', 'c']:
            fa = to_float(x._mpf_, rnd=rnd)
            fb = to_float(fadd(x,0,prec=53,rounding=rnd)._mpf_, rnd=rnd)
            assert fa == fb
    mp.prec = 53

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\date\array.py ===
from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas.core.dtypes.dtypes import register_extension_dtype

from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)
from pandas.api.types import pandas_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        Dtype,
        PositionalIndexer,
    )


@register_extension_dtype
class DateDtype(ExtensionDtype):
    @property
    def type(self):
        return dt.date

    @property
    def name(self):
        return "DateDtype"

    @classmethod
    def construct_from_string(cls, string: str):
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string == cls.__name__:
            return cls()
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return DateArray

    @property
    def na_value(self):
        return dt.date.min

    def __repr__(self) -> str:
        return self.name


class DateArray(ExtensionArray):
    def __init__(
        self,
        dates: (
            dt.date
            | Sequence[dt.date]
            | tuple[np.ndarray, np.ndarray, np.ndarray]
            | np.ndarray
        ),
    ) -> None:
        if isinstance(dates, dt.date):
            self._year = np.array([dates.year])
            self._month = np.array([dates.month])
            self._day = np.array([dates.year])
            return

        ldates = len(dates)
        if isinstance(dates, list):
            # pre-allocate the arrays since we know the size before hand
            self._year = np.zeros(ldates, dtype=np.uint16)  # 65535 (0, 9999)
            self._month = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 31)
            self._day = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 12)
            # populate them
            for i, (y, m, d) in enumerate(
                (date.year, date.month, date.day) for date in dates
            ):
                self._year[i] = y
                self._month[i] = m
                self._day[i] = d

        elif isinstance(dates, tuple):
            # only support triples
            if ldates != 3:
                raise ValueError("only triples are valid")
            # check if all elements have the same type
            if any(not isinstance(x, np.ndarray) for x in dates):
                raise TypeError("invalid type")
            ly, lm, ld = (len(cast(np.ndarray, d)) for d in dates)
            if not ly == lm == ld:
                raise ValueError(
                    f"tuple members must have the same length: {(ly, lm, ld)}"
                )
            self._year = dates[0].astype(np.uint16)
            self._month = dates[1].astype(np.uint8)
            self._day = dates[2].astype(np.uint8)

        elif isinstance(dates, np.ndarray) and dates.dtype == "U10":
            self._year = np.zeros(ldates, dtype=np.uint16)  # 65535 (0, 9999)
            self._month = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 31)
            self._day = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 12)

            # error: "object_" object is not iterable
            obj = np.char.split(dates, sep="-")
            for (i,), (y, m, d) in np.ndenumerate(obj):  # type: ignore[misc]
                self._year[i] = int(y)
                self._month[i] = int(m)
                self._day[i] = int(d)

        else:
            raise TypeError(f"{type(dates)} is not supported")

    @property
    def dtype(self) -> ExtensionDtype:
        return DateDtype()

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)

        if isinstance(dtype, DateDtype):
            data = self.copy() if copy else self
        else:
            data = self.to_numpy(dtype=dtype, copy=copy, na_value=dt.date.min)

        return data

    @property
    def nbytes(self) -> int:
        return self._year.nbytes + self._month.nbytes + self._day.nbytes

    def __len__(self) -> int:
        return len(self._year)  # all 3 arrays are enforced to have the same length

    def __getitem__(self, item: PositionalIndexer):
        if isinstance(item, int):
            return dt.date(self._year[item], self._month[item], self._day[item])
        else:
            raise NotImplementedError("only ints are supported as indexes")

    def __setitem__(self, key: int | slice | np.ndarray, value: Any) -> None:
        if not isinstance(key, int):
            raise NotImplementedError("only ints are supported as indexes")

        if not isinstance(value, dt.date):
            raise TypeError("you can only set datetime.date types")

        self._year[key] = value.year
        self._month[key] = value.month
        self._day[key] = value.day

    def __repr__(self) -> str:
        return f"DateArray{list(zip(self._year, self._month, self._day))}"

    def copy(self) -> DateArray:
        return DateArray((self._year.copy(), self._month.copy(), self._day.copy()))

    def isna(self) -> np.ndarray:
        return np.logical_and(
            np.logical_and(
                self._year == dt.date.min.year, self._month == dt.date.min.month
            ),
            self._day == dt.date.min.day,
        )

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy=False):
        if isinstance(scalars, dt.date):
            raise TypeError
        elif isinstance(scalars, DateArray):
            if dtype is not None:
                return scalars.astype(dtype, copy=copy)
            if copy:
                return scalars.copy()
            return scalars[:]
        elif isinstance(scalars, np.ndarray):
            scalars = scalars.astype("U10")  # 10 chars for yyyy-mm-dd
            return DateArray(scalars)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_monotonic.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    MultiIndex,
)


def test_is_monotonic_increasing_lexsorted(lexsorted_two_level_string_multiindex):
    # string ordering
    mi = lexsorted_two_level_string_multiindex
    assert mi.is_monotonic_increasing is False
    assert Index(mi.values).is_monotonic_increasing is False
    assert mi._is_strictly_monotonic_increasing is False
    assert Index(mi.values)._is_strictly_monotonic_increasing is False


def test_is_monotonic_increasing():
    i = MultiIndex.from_product([np.arange(10), np.arange(10)], names=["one", "two"])
    assert i.is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True
    assert Index(i.values).is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True

    i = MultiIndex.from_product(
        [np.arange(10, 0, -1), np.arange(10)], names=["one", "two"]
    )
    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False
    assert Index(i.values).is_monotonic_increasing is False
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    i = MultiIndex.from_product(
        [np.arange(10), np.arange(10, 0, -1)], names=["one", "two"]
    )
    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False
    assert Index(i.values).is_monotonic_increasing is False
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    i = MultiIndex.from_product([[1.0, np.nan, 2.0], ["a", "b", "c"]])
    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False
    assert Index(i.values).is_monotonic_increasing is False
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    i = MultiIndex(
        levels=[["bar", "baz", "foo", "qux"], ["mom", "next", "zenith"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    assert i.is_monotonic_increasing is True
    assert Index(i.values).is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True
    assert Index(i.values)._is_strictly_monotonic_increasing is True

    # mixed levels, hits the TypeError
    i = MultiIndex(
        levels=[
            [1, 2, 3, 4],
            [
                "gb00b03mlx29",
                "lu0197800237",
                "nl0000289783",
                "nl0000289965",
                "nl0000301109",
            ],
        ],
        codes=[[0, 1, 1, 2, 2, 2, 3], [4, 2, 0, 0, 1, 3, -1]],
        names=["household_id", "asset_id"],
    )

    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False

    # empty
    i = MultiIndex.from_arrays([[], []])
    assert i.is_monotonic_increasing is True
    assert Index(i.values).is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True
    assert Index(i.values)._is_strictly_monotonic_increasing is True


def test_is_monotonic_decreasing():
    i = MultiIndex.from_product(
        [np.arange(9, -1, -1), np.arange(9, -1, -1)], names=["one", "two"]
    )
    assert i.is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True
    assert Index(i.values).is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True

    i = MultiIndex.from_product(
        [np.arange(10), np.arange(10, 0, -1)], names=["one", "two"]
    )
    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    i = MultiIndex.from_product(
        [np.arange(10, 0, -1), np.arange(10)], names=["one", "two"]
    )
    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    i = MultiIndex.from_product([[2.0, np.nan, 1.0], ["c", "b", "a"]])
    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    # string ordering
    i = MultiIndex(
        levels=[["qux", "foo", "baz", "bar"], ["three", "two", "one"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    assert i.is_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    i = MultiIndex(
        levels=[["qux", "foo", "baz", "bar"], ["zenith", "next", "mom"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    assert i.is_monotonic_decreasing is True
    assert Index(i.values).is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True
    assert Index(i.values)._is_strictly_monotonic_decreasing is True

    # mixed levels, hits the TypeError
    i = MultiIndex(
        levels=[
            [4, 3, 2, 1],
            [
                "nl0000301109",
                "nl0000289965",
                "nl0000289783",
                "lu0197800237",
                "gb00b03mlx29",
            ],
        ],
        codes=[[0, 1, 1, 2, 2, 2, 3], [4, 2, 0, 0, 1, 3, -1]],
        names=["household_id", "asset_id"],
    )

    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False

    # empty
    i = MultiIndex.from_arrays([[], []])
    assert i.is_monotonic_decreasing is True
    assert Index(i.values).is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True
    assert Index(i.values)._is_strictly_monotonic_decreasing is True


def test_is_strictly_monotonic_increasing():
    idx = MultiIndex(
        levels=[["bar", "baz"], ["mom", "next"]], codes=[[0, 0, 1, 1], [0, 0, 0, 1]]
    )
    assert idx.is_monotonic_increasing is True
    assert idx._is_strictly_monotonic_increasing is False


def test_is_strictly_monotonic_decreasing():
    idx = MultiIndex(
        levels=[["baz", "bar"], ["next", "mom"]], codes=[[0, 0, 1, 1], [0, 0, 0, 1]]
    )
    assert idx.is_monotonic_decreasing is True
    assert idx._is_strictly_monotonic_decreasing is False


@pytest.mark.parametrize("attr", ["is_monotonic_increasing", "is_monotonic_decreasing"])
@pytest.mark.parametrize(
    "values",
    [[(np.nan,), (1,), (2,)], [(1,), (np.nan,), (2,)], [(1,), (2,), (np.nan,)]],
)
def test_is_monotonic_with_nans(values, attr):
    # GH: 37220
    idx = MultiIndex.from_tuples(values, names=["test"])
    assert getattr(idx, attr) is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_greenlet_trash.py ===
# -*- coding: utf-8 -*-
"""
Tests for greenlets interacting with the CPython trash can API.

The CPython trash can API is not designed to be re-entered from a
single thread. But this can happen using greenlets, if something
during the object deallocation process switches greenlets, and this second
greenlet then causes the trash can to get entered again. Here, we do this
very explicitly, but in other cases (like gevent) it could be arbitrarily more
complicated: for example, a weakref callback might try to acquire a lock that's
already held by another greenlet; that would allow a greenlet switch to occur.

See https://github.com/gevent/gevent/issues/1909

This test is fragile and relies on details of the CPython
implementation (like most of the rest of this package):

    - We enter the trashcan and deferred deallocation after
      ``_PyTrash_UNWIND_LEVEL`` calls. This constant, defined in
      CPython's object.c, is generally 50. That's basically how many objects are required to
      get us into the deferred deallocation situation.

    - The test fails by hitting an ``assert()`` in object.c; if the
      build didn't enable assert, then we don't catch this.

    - If the test fails in that way, the interpreter crashes.
"""
from __future__ import print_function, absolute_import, division

import unittest


class TestTrashCanReEnter(unittest.TestCase):

    def test_it(self):
        try:
            # pylint:disable-next=no-name-in-module
            from greenlet._greenlet import get_tstate_trash_delete_nesting # pylint:disable=unused-import
        except ImportError:
            import sys
            # Python 3.13 has not "trash delete nesting" anymore (but "delete later")
            assert sys.version_info[:2] >= (3, 13)
            self.skipTest("get_tstate_trash_delete_nesting is not available.")

        # Try several times to trigger it, because it isn't 100%
        # reliable.
        for _ in range(10):
            self.check_it()

    def check_it(self): # pylint:disable=too-many-statements
        import greenlet
        from greenlet._greenlet import get_tstate_trash_delete_nesting # pylint:disable=no-name-in-module
        main = greenlet.getcurrent()

        assert get_tstate_trash_delete_nesting() == 0

        # We expect to be in deferred deallocation after this many
        # deallocations have occurred. TODO: I wish we had a better way to do
        # this --- that was before get_tstate_trash_delete_nesting; perhaps
        # we can use that API to do better?
        TRASH_UNWIND_LEVEL = 50
        # How many objects to put in a container; it's the container that
        # queues objects for deferred deallocation.
        OBJECTS_PER_CONTAINER = 500

        class Dealloc: # define the class here because we alter class variables each time we run.
            """
            An object with a ``__del__`` method. When it starts getting deallocated
            from a deferred trash can run, it switches greenlets, allocates more objects
            which then also go in the trash can. If we don't save state appropriately,
            nesting gets out of order and we can crash the interpreter.
            """

            #: Has our deallocation actually run and switched greenlets?
            #: When it does, this will be set to the current greenlet. This should
            #: be happening in the main greenlet, so we check that down below.
            SPAWNED = False

            #: Has the background greenlet run?
            BG_RAN = False

            BG_GLET = None

            #: How many of these things have ever been allocated.
            CREATED = 0

            #: How many of these things have ever been deallocated.
            DESTROYED = 0

            #: How many were destroyed not in the main greenlet. There should always
            #: be some.
            #: If the test is broken or things change in the trashcan implementation,
            #: this may not be correct.
            DESTROYED_BG = 0

            def __init__(self, sequence_number):
                """
                :param sequence_number: The ordinal of this object during
                   one particular creation run. This is used to detect (guess, really)
                   when we have entered the trash can's deferred deallocation.
                """
                self.i = sequence_number
                Dealloc.CREATED += 1

            def __del__(self):
                if self.i == TRASH_UNWIND_LEVEL and not self.SPAWNED:
                    Dealloc.SPAWNED = greenlet.getcurrent()
                    other = Dealloc.BG_GLET = greenlet.greenlet(background_greenlet)
                    x = other.switch()
                    assert x == 42
                    # It's important that we don't switch back to the greenlet,
                    # we leave it hanging there in an incomplete state. But we don't let it
                    # get collected, either. If we complete it now, while we're still
                    # in the scope of the initial trash can, things work out and we
                    # don't see the problem. We need this greenlet to complete
                    # at some point in the future, after we've exited this trash can invocation.
                    del other
                elif self.i == 40 and greenlet.getcurrent() is not main:
                    Dealloc.BG_RAN = True
                    try:
                        main.switch(42)
                    except greenlet.GreenletExit as ex:
                        # We expect this; all references to us go away
                        # while we're still running, and we need to finish deleting
                        # ourself.
                        Dealloc.BG_RAN = type(ex)
                        del ex

                # Record the fact that we're dead last of all. This ensures that
                # we actually get returned too.
                Dealloc.DESTROYED += 1
                if greenlet.getcurrent() is not main:
                    Dealloc.DESTROYED_BG += 1


        def background_greenlet():
            # We direct through a second function, instead of
            # directly calling ``make_some()``, so that we have complete
            # control over when these objects are destroyed: we need them
            # to be destroyed in the context of the background greenlet
            t = make_some()
            del t # Triggere deletion.

        def make_some():
            t = ()
            i = OBJECTS_PER_CONTAINER
            while i:
                # Nest the tuples; it's the recursion that gets us
                # into trash.
                t = (Dealloc(i), t)
                i -= 1
            return t


        some = make_some()
        self.assertEqual(Dealloc.CREATED, OBJECTS_PER_CONTAINER)
        self.assertEqual(Dealloc.DESTROYED, 0)

        # If we're going to crash, it should be on the following line.
        # We only crash if ``assert()`` is enabled, of course.
        del some

        # For non-debug builds of CPython, we won't crash. The best we can do is check
        # the nesting level explicitly.
        self.assertEqual(0, get_tstate_trash_delete_nesting())

        # Discard this, raising GreenletExit into where it is waiting.
        Dealloc.BG_GLET = None
        # The same nesting level maintains.
        self.assertEqual(0, get_tstate_trash_delete_nesting())

        # We definitely cleaned some up in the background
        self.assertGreater(Dealloc.DESTROYED_BG, 0)

        # Make sure all the cleanups happened.
        self.assertIs(Dealloc.SPAWNED, main)
        self.assertTrue(Dealloc.BG_RAN)
        self.assertEqual(Dealloc.BG_RAN, greenlet.GreenletExit)
        self.assertEqual(Dealloc.CREATED, Dealloc.DESTROYED )
        self.assertEqual(Dealloc.CREATED, OBJECTS_PER_CONTAINER * 2)

        import gc
        gc.collect()


if __name__ == '__main__':
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_regression.py ===
import os
import platform

import pytest

import numpy as np
import numpy.testing as npt

from . import util


class TestIntentInOut(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    @pytest.mark.slow
    def test_inout(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert np.allclose(x, [3, 1, 2])


class TestDataOnlyMultiModule(util.F2PyTest):
    # Check that modules without subroutines work
    sources = [util.getpath("tests", "src", "regression", "datonly.f90")]

    @pytest.mark.slow
    def test_mdat(self):
        assert self.module.datonly.max_value == 100
        assert self.module.dat.max_ == 1009
        int_in = 5
        assert self.module.simple_subroutine(5) == 1014


class TestModuleWithDerivedType(util.F2PyTest):
    # Check that modules with derived types work
    sources = [util.getpath("tests", "src", "regression", "mod_derived_types.f90")]

    @pytest.mark.slow
    def test_mtypes(self):
        assert self.module.no_type_subroutine(10) == 110
        assert self.module.type_subroutine(10) == 210


class TestNegativeBounds(util.F2PyTest):
    # Check that negative bounds work correctly
    sources = [util.getpath("tests", "src", "negative_bounds", "issue_20853.f90")]

    @pytest.mark.slow
    def test_negbound(self):
        xvec = np.arange(12)
        xlow = -6
        xhigh = 4

        # Calculate the upper bound,
        # Keeping the 1 index in mind

        def ubound(xl, xh):
            return xh - xl + 1
        rval = self.module.foo(is_=xlow, ie_=xhigh,
                        arr=xvec[:ubound(xlow, xhigh)])
        expval = np.arange(11, dtype=np.float32)
        assert np.allclose(rval, expval)


class TestNumpyVersionAttribute(util.F2PyTest):
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    @pytest.mark.slow
    def test_numpy_version_attribute(self):

        # Check that self.module has an attribute named "__f2py_numpy_version__"
        assert hasattr(self.module, "__f2py_numpy_version__")

        # Check that the attribute __f2py_numpy_version__ is a string
        assert isinstance(self.module.__f2py_numpy_version__, str)

        # Check that __f2py_numpy_version__ has the value numpy.__version__
        assert np.__version__ == self.module.__f2py_numpy_version__


def test_include_path():
    incdir = np.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ("fortranobject.c", "fortranobject.h"):
        assert fname in fnames_in_dir


class TestIncludeFiles(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "incfile.f90")]
    options = [f"-I{util.getpath('tests', 'src', 'regression')}",
               f"--include-paths {util.getpath('tests', 'src', 'regression')}"]

    @pytest.mark.slow
    def test_gh25344(self):
        exp = 7.0
        res = self.module.add(3.0, 4.0)
        assert exp == res

class TestF77Comments(util.F2PyTest):
    # Check that comments are stripped from F77 continuation lines
    sources = [util.getpath("tests", "src", "regression", "f77comments.f")]

    @pytest.mark.slow
    def test_gh26148(self):
        x1 = np.array(3, dtype=np.int32)
        x2 = np.array(5, dtype=np.int32)
        res = self.module.testsub(x1, x2)
        assert res[0] == 8
        assert res[1] == 15

    @pytest.mark.slow
    def test_gh26466(self):
        # Check that comments after PARAMETER directions are stripped
        expected = np.arange(1, 11, dtype=np.float32) * 2
        res = self.module.testsub2()
        npt.assert_allclose(expected, res)

class TestF90Contiuation(util.F2PyTest):
    # Check that comments are stripped from F90 continuation lines
    sources = [util.getpath("tests", "src", "regression", "f90continuation.f90")]

    @pytest.mark.slow
    def test_gh26148b(self):
        x1 = np.array(3, dtype=np.int32)
        x2 = np.array(5, dtype=np.int32)
        res = self.module.testsub(x1, x2)
        assert res[0] == 8
        assert res[1] == 15

class TestLowerF2PYDirectives(util.F2PyTest):
    # Check variables are cased correctly
    sources = [util.getpath("tests", "src", "regression", "lower_f2py_fortran.f90")]

    @pytest.mark.slow
    def test_gh28014(self):
        self.module.inquire_next(3)
        assert True

@pytest.mark.slow
def test_gh26623():
    # Including libraries with . should not generate an incorrect meson.build
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f90continuation.f90")],
            ["-lfoo.bar"],
            module_name="Blah",
        )
    except RuntimeError as rerr:
        assert "lparen got assign" not in str(rerr)


@pytest.mark.slow
@pytest.mark.skipif(platform.system() not in ['Linux', 'Darwin'], reason='Unsupported on this platform for now')
def test_gh25784():
    # Compile dubious file using passed flags
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f77fixedform.f95")],
            options=[
                # Meson will collect and dedup these to pass to fortran_args:
                "--f77flags='-ffixed-form -O2'",
                "--f90flags=\"-ffixed-form -Og\"",
            ],
            module_name="Blah",
        )
    except ImportError as rerr:
        assert "unknown_subroutine_" in str(rerr)


@pytest.mark.slow
class TestAssignmentOnlyModules(util.F2PyTest):
    # Ensure that variables are exposed without functions or subroutines in a module
    sources = [util.getpath("tests", "src", "regression", "assignOnlyModule.f90")]

    @pytest.mark.slow
    def test_gh27167(self):
        assert (self.module.f_globals.n_max == 16)
        assert (self.module.f_globals.i_max == 18)
        assert (self.module.f_globals.j_max == 72)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\methods\test_round.py ===
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import iNaT
from pandas.errors import OutOfBoundsTimedelta

from pandas import Timedelta


class TestTimedeltaRound:
    @pytest.mark.parametrize(
        "freq,s1,s2",
        [
            # This first case has s1, s2 being the same as t1,t2 below
            (
                "ns",
                Timedelta("1 days 02:34:56.789123456"),
                Timedelta("-1 days 02:34:56.789123456"),
            ),
            (
                "us",
                Timedelta("1 days 02:34:56.789123000"),
                Timedelta("-1 days 02:34:56.789123000"),
            ),
            (
                "ms",
                Timedelta("1 days 02:34:56.789000000"),
                Timedelta("-1 days 02:34:56.789000000"),
            ),
            ("s", Timedelta("1 days 02:34:57"), Timedelta("-1 days 02:34:57")),
            ("2s", Timedelta("1 days 02:34:56"), Timedelta("-1 days 02:34:56")),
            ("5s", Timedelta("1 days 02:34:55"), Timedelta("-1 days 02:34:55")),
            ("min", Timedelta("1 days 02:35:00"), Timedelta("-1 days 02:35:00")),
            ("12min", Timedelta("1 days 02:36:00"), Timedelta("-1 days 02:36:00")),
            ("h", Timedelta("1 days 03:00:00"), Timedelta("-1 days 03:00:00")),
            ("d", Timedelta("1 days"), Timedelta("-1 days")),
        ],
    )
    def test_round(self, freq, s1, s2):
        t1 = Timedelta("1 days 02:34:56.789123456")
        t2 = Timedelta("-1 days 02:34:56.789123456")

        r1 = t1.round(freq)
        assert r1 == s1
        r2 = t2.round(freq)
        assert r2 == s2

    def test_round_invalid(self):
        t1 = Timedelta("1 days 02:34:56.789123456")

        for freq, msg in [
            ("YE", "<YearEnd: month=12> is a non-fixed frequency"),
            ("ME", "<MonthEnd> is a non-fixed frequency"),
            ("foobar", "Invalid frequency: foobar"),
        ]:
            with pytest.raises(ValueError, match=msg):
                t1.round(freq)

    @pytest.mark.skip_ubsan
    def test_round_implementation_bounds(self):
        # See also: analogous test for Timestamp
        # GH#38964
        result = Timedelta.min.ceil("s")
        expected = Timedelta.min + Timedelta(seconds=1) - Timedelta(145224193)
        assert result == expected

        result = Timedelta.max.floor("s")
        expected = Timedelta.max - Timedelta(854775807)
        assert result == expected

        msg = (
            r"Cannot round -106752 days \+00:12:43.145224193 to freq=s without overflow"
        )
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.min.floor("s")
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.min.round("s")

        msg = "Cannot round 106751 days 23:47:16.854775807 to freq=s without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.max.ceil("s")
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.max.round("s")

    @pytest.mark.skip_ubsan
    @given(val=st.integers(min_value=iNaT + 1, max_value=lib.i8max))
    @pytest.mark.parametrize(
        "method", [Timedelta.round, Timedelta.floor, Timedelta.ceil]
    )
    def test_round_sanity(self, val, method):
        cls = Timedelta
        err_cls = OutOfBoundsTimedelta

        val = np.int64(val)
        td = cls(val)

        def checker(ts, nanos, unit):
            # First check that we do raise in cases where we should
            if nanos == 1:
                pass
            else:
                div, mod = divmod(ts._value, nanos)
                diff = int(nanos - mod)
                lb = ts._value - mod
                assert lb <= ts._value  # i.e. no overflows with python ints
                ub = ts._value + diff
                assert ub > ts._value  # i.e. no overflows with python ints

                msg = "without overflow"
                if mod == 0:
                    # We should never be raising in this
                    pass
                elif method is cls.ceil:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif method is cls.floor:
                    if lb < cls.min._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif mod >= diff:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif lb < cls.min._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return

            res = method(ts, unit)

            td = res - ts
            diff = abs(td._value)
            assert diff < nanos
            assert res._value % nanos == 0

            if method is cls.round:
                assert diff <= nanos / 2
            elif method is cls.floor:
                assert res <= ts
            elif method is cls.ceil:
                assert res >= ts

        nanos = 1
        checker(td, nanos, "ns")

        nanos = 1000
        checker(td, nanos, "us")

        nanos = 1_000_000
        checker(td, nanos, "ms")

        nanos = 1_000_000_000
        checker(td, nanos, "s")

        nanos = 60 * 1_000_000_000
        checker(td, nanos, "min")

        nanos = 60 * 60 * 1_000_000_000
        checker(td, nanos, "h")

        nanos = 24 * 60 * 60 * 1_000_000_000
        checker(td, nanos, "D")

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_round_non_nano(self, unit):
        td = Timedelta("1 days 02:34:57").as_unit(unit)

        res = td.round("min")
        assert res == Timedelta("1 days 02:35:00")
        assert res._creso == td._creso

        res = td.floor("min")
        assert res == Timedelta("1 days 02:34:00")
        assert res._creso == td._creso

        res = td.ceil("min")
        assert res == Timedelta("1 days 02:35:00")
        assert res._creso == td._creso

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\merge\test_merge_index_as_string.py ===
import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm


@pytest.fixture
def df1():
    return DataFrame(
        {
            "outer": [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4],
            "inner": [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2],
            "v1": np.linspace(0, 1, 11),
        }
    )


@pytest.fixture
def df2():
    return DataFrame(
        {
            "outer": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3],
            "inner": [1, 2, 2, 3, 3, 4, 2, 3, 1, 1, 2, 3],
            "v2": np.linspace(10, 11, 12),
        }
    )


@pytest.fixture(params=[[], ["outer"], ["outer", "inner"]])
def left_df(request, df1):
    """Construct left test DataFrame with specified levels
    (any of 'outer', 'inner', and 'v1')
    """
    levels = request.param
    if levels:
        df1 = df1.set_index(levels)

    return df1


@pytest.fixture(params=[[], ["outer"], ["outer", "inner"]])
def right_df(request, df2):
    """Construct right test DataFrame with specified levels
    (any of 'outer', 'inner', and 'v2')
    """
    levels = request.param

    if levels:
        df2 = df2.set_index(levels)

    return df2


def compute_expected(df_left, df_right, on=None, left_on=None, right_on=None, how=None):
    """
    Compute the expected merge result for the test case.

    This method computes the expected result of merging two DataFrames on
    a combination of their columns and index levels. It does so by
    explicitly dropping/resetting their named index levels, performing a
    merge on their columns, and then finally restoring the appropriate
    index in the result.

    Parameters
    ----------
    df_left : DataFrame
        The left DataFrame (may have zero or more named index levels)
    df_right : DataFrame
        The right DataFrame (may have zero or more named index levels)
    on : list of str
        The on parameter to the merge operation
    left_on : list of str
        The left_on parameter to the merge operation
    right_on : list of str
        The right_on parameter to the merge operation
    how : str
        The how parameter to the merge operation

    Returns
    -------
    DataFrame
        The expected merge result
    """
    # Handle on param if specified
    if on is not None:
        left_on, right_on = on, on

    # Compute input named index levels
    left_levels = [n for n in df_left.index.names if n is not None]
    right_levels = [n for n in df_right.index.names if n is not None]

    # Compute output named index levels
    output_levels = [i for i in left_on if i in right_levels and i in left_levels]

    # Drop index levels that aren't involved in the merge
    drop_left = [n for n in left_levels if n not in left_on]
    if drop_left:
        df_left = df_left.reset_index(drop_left, drop=True)

    drop_right = [n for n in right_levels if n not in right_on]
    if drop_right:
        df_right = df_right.reset_index(drop_right, drop=True)

    # Convert remaining index levels to columns
    reset_left = [n for n in left_levels if n in left_on]
    if reset_left:
        df_left = df_left.reset_index(level=reset_left)

    reset_right = [n for n in right_levels if n in right_on]
    if reset_right:
        df_right = df_right.reset_index(level=reset_right)

    # Perform merge
    expected = df_left.merge(df_right, left_on=left_on, right_on=right_on, how=how)

    # Restore index levels
    if output_levels:
        expected = expected.set_index(output_levels)

    return expected


@pytest.mark.parametrize(
    "on,how",
    [
        (["outer"], "inner"),
        (["inner"], "left"),
        (["outer", "inner"], "right"),
        (["inner", "outer"], "outer"),
    ],
)
def test_merge_indexes_and_columns_on(left_df, right_df, on, how):
    # Construct expected result
    expected = compute_expected(left_df, right_df, on=on, how=how)

    # Perform merge
    result = left_df.merge(right_df, on=on, how=how)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "left_on,right_on,how",
    [
        (["outer"], ["outer"], "inner"),
        (["inner"], ["inner"], "right"),
        (["outer", "inner"], ["outer", "inner"], "left"),
        (["inner", "outer"], ["inner", "outer"], "outer"),
    ],
)
def test_merge_indexes_and_columns_lefton_righton(
    left_df, right_df, left_on, right_on, how
):
    # Construct expected result
    expected = compute_expected(
        left_df, right_df, left_on=left_on, right_on=right_on, how=how
    )

    # Perform merge
    result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize("left_index", ["inner", ["inner", "outer"]])
def test_join_indexes_and_columns_on(df1, df2, left_index, join_type):
    # Construct left_df
    left_df = df1.set_index(left_index)

    # Construct right_df
    right_df = df2.set_index(["outer", "inner"])

    # Result
    expected = (
        left_df.reset_index()
        .join(
            right_df, on=["outer", "inner"], how=join_type, lsuffix="_x", rsuffix="_y"
        )
        .set_index(left_index)
    )

    # Perform join
    result = left_df.join(
        right_df, on=["outer", "inner"], how=join_type, lsuffix="_x", rsuffix="_y"
    )

    tm.assert_frame_equal(result, expected, check_like=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_cfunctions.py ===
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.codegen.cfunctions import (
    expm1, log1p, exp2, log2, fma, log10, Sqrt, Cbrt, hypot, isnan, isinf
)
from sympy.core.function import expand_log


def test_expm1():
    # Eval
    assert expm1(0) == 0

    x = Symbol('x', real=True)

    # Expand and rewrite
    assert expm1(x).expand(func=True) - exp(x) == -1
    assert expm1(x).rewrite('tractable') - exp(x) == -1
    assert expm1(x).rewrite('exp') - exp(x) == -1

    # Precision
    assert not ((exp(1e-10).evalf() - 1) - 1e-10 - 5e-21) < 1e-22  # for comparison
    assert abs(expm1(1e-10).evalf() - 1e-10 - 5e-21) < 1e-22

    # Properties
    assert expm1(x).is_real
    assert expm1(x).is_finite

    # Diff
    assert expm1(42*x).diff(x) - 42*exp(42*x) == 0
    assert expm1(42*x).diff(x) - expm1(42*x).expand(func=True).diff(x) == 0


def test_log1p():
    # Eval
    assert log1p(0) == 0
    d = S(10)
    assert expand_log(log1p(d**-1000) - log(d**1000 + 1) + log(d**1000)) == 0

    x = Symbol('x', real=True)

    # Expand and rewrite
    assert log1p(x).expand(func=True) - log(x + 1) == 0
    assert log1p(x).rewrite('tractable') - log(x + 1) == 0
    assert log1p(x).rewrite('log') - log(x + 1) == 0

    # Precision
    assert not abs(log(1e-99 + 1).evalf() - 1e-99) < 1e-100  # for comparison
    assert abs(expand_log(log1p(1e-99)).evalf() - 1e-99) < 1e-100

    # Properties
    assert log1p(-2**Rational(-1, 2)).is_real

    assert not log1p(-1).is_finite
    assert log1p(pi).is_finite

    assert not log1p(x).is_positive
    assert log1p(Symbol('y', positive=True)).is_positive

    assert not log1p(x).is_zero
    assert log1p(Symbol('z', zero=True)).is_zero

    assert not log1p(x).is_nonnegative
    assert log1p(Symbol('o', nonnegative=True)).is_nonnegative

    # Diff
    assert log1p(42*x).diff(x) - 42/(42*x + 1) == 0
    assert log1p(42*x).diff(x) - log1p(42*x).expand(func=True).diff(x) == 0


def test_exp2():
    # Eval
    assert exp2(2) == 4

    x = Symbol('x', real=True)

    # Expand
    assert exp2(x).expand(func=True) - 2**x == 0

    # Diff
    assert exp2(42*x).diff(x) - 42*exp2(42*x)*log(2) == 0
    assert exp2(42*x).diff(x) - exp2(42*x).diff(x) == 0


def test_log2():
    # Eval
    assert log2(8) == 3
    assert log2(pi) != log(pi)/log(2)  # log2 should *save* (CPU) instructions

    x = Symbol('x', real=True)
    assert log2(x) != log(x)/log(2)
    assert log2(2**x) == x

    # Expand
    assert log2(x).expand(func=True) - log(x)/log(2) == 0

    # Diff
    assert log2(42*x).diff() - 1/(log(2)*x) == 0
    assert log2(42*x).diff() - log2(42*x).expand(func=True).diff(x) == 0


def test_fma():
    x, y, z = symbols('x y z')

    # Expand
    assert fma(x, y, z).expand(func=True) - x*y - z == 0

    expr = fma(17*x, 42*y, 101*z)

    # Diff
    assert expr.diff(x) - expr.expand(func=True).diff(x) == 0
    assert expr.diff(y) - expr.expand(func=True).diff(y) == 0
    assert expr.diff(z) - expr.expand(func=True).diff(z) == 0

    assert expr.diff(x) - 17*42*y == 0
    assert expr.diff(y) - 17*42*x == 0
    assert expr.diff(z) - 101 == 0


def test_log10():
    x = Symbol('x')

    # Expand
    assert log10(x).expand(func=True) - log(x)/log(10) == 0

    # Diff
    assert log10(42*x).diff(x) - 1/(log(10)*x) == 0
    assert log10(42*x).diff(x) - log10(42*x).expand(func=True).diff(x) == 0


def test_Cbrt():
    x = Symbol('x')

    # Expand
    assert Cbrt(x).expand(func=True) - x**Rational(1, 3) == 0

    # Diff
    assert Cbrt(42*x).diff(x) - 42*(42*x)**(Rational(1, 3) - 1)/3 == 0
    assert Cbrt(42*x).diff(x) - Cbrt(42*x).expand(func=True).diff(x) == 0


def test_Sqrt():
    x = Symbol('x')

    # Expand
    assert Sqrt(x).expand(func=True) - x**S.Half == 0

    # Diff
    assert Sqrt(42*x).diff(x) - 42*(42*x)**(S.Half - 1)/2 == 0
    assert Sqrt(42*x).diff(x) - Sqrt(42*x).expand(func=True).diff(x) == 0


def test_hypot():
    x, y = symbols('x y')

    # Expand
    assert hypot(x, y).expand(func=True) - (x**2 + y**2)**S.Half == 0

    # Diff
    assert hypot(17*x, 42*y).diff(x).expand(func=True) - hypot(17*x, 42*y).expand(func=True).diff(x) == 0
    assert hypot(17*x, 42*y).diff(y).expand(func=True) - hypot(17*x, 42*y).expand(func=True).diff(y) == 0

    assert hypot(17*x, 42*y).diff(x).expand(func=True) - 2*17*17*x*((17*x)**2 + (42*y)**2)**Rational(-1, 2)/2 == 0
    assert hypot(17*x, 42*y).diff(y).expand(func=True) - 2*42*42*y*((17*x)**2 + (42*y)**2)**Rational(-1, 2)/2 == 0


def test_isnan_isinf():
    x = Symbol('x')

    # isinf
    assert isinf(+S.Infinity) == True
    assert isinf(-S.Infinity) == True
    assert isinf(S.Pi) == False
    isinfx = isinf(x)
    assert isinfx not in (False, True)
    assert isinfx.func is isinf
    assert isinfx.args == (x,)

    # isnan
    assert isnan(S.NaN) == True
    assert isnan(S.Pi) == False
    isnanx = isnan(x)
    assert isnanx not in (False, True)
    assert isnanx.func is isnan
    assert isnanx.args == (x,)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_represent.py ===
from sympy.core.numbers import (Float, I, Integer)
from sympy.matrices.dense import Matrix
from sympy.external import import_module
from sympy.testing.pytest import skip

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.represent import (represent, rep_innerproduct,
                                             rep_expectation, enumerate_states)
from sympy.physics.quantum.state import Bra, Ket
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import matrix_tensor_product
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.matrixutils import (numpy_ndarray,
                                               scipy_sparse_matrix, to_numpy,
                                               to_scipy_sparse, to_sympy)
from sympy.physics.quantum.cartesian import XKet, XOp, XBra
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state
from sympy.testing.pytest import raises

Amat = Matrix([[1, I], [-I, 1]])
Bmat = Matrix([[1, 2], [3, 4]])
Avec = Matrix([[1], [I]])


class AKet(Ket):

    @classmethod
    def dual_class(self):
        return ABra

    def _represent_default_basis(self, **options):
        return self._represent_AOp(None, **options)

    def _represent_AOp(self, basis, **options):
        return Avec


class ABra(Bra):

    @classmethod
    def dual_class(self):
        return AKet


class AOp(Operator):

    def _represent_default_basis(self, **options):
        return self._represent_AOp(None, **options)

    def _represent_AOp(self, basis, **options):
        return Amat


class BOp(Operator):

    def _represent_default_basis(self, **options):
        return self._represent_AOp(None, **options)

    def _represent_AOp(self, basis, **options):
        return Bmat


k = AKet('a')
b = ABra('a')
A = AOp('A')
B = BOp('B')

_tests = [
    # Bra
    (b, Dagger(Avec)),
    (Dagger(b), Avec),
    # Ket
    (k, Avec),
    (Dagger(k), Dagger(Avec)),
    # Operator
    (A, Amat),
    (Dagger(A), Dagger(Amat)),
    # OuterProduct
    (OuterProduct(k, b), Avec*Avec.H),
    # TensorProduct
    (TensorProduct(A, B), matrix_tensor_product(Amat, Bmat)),
    # Pow
    (A**2, Amat**2),
    # Add/Mul
    (A*B + 2*A, Amat*Bmat + 2*Amat),
    # Commutator
    (Commutator(A, B), Amat*Bmat - Bmat*Amat),
    # AntiCommutator
    (AntiCommutator(A, B), Amat*Bmat + Bmat*Amat),
    # InnerProduct
    (InnerProduct(b, k), (Avec.H*Avec)[0])
]


def test_format_sympy():
    for test in _tests:
        lhs = represent(test[0], basis=A, format='sympy')
        rhs = to_sympy(test[1])
        assert lhs == rhs


def test_scalar_sympy():
    assert represent(Integer(1)) == Integer(1)
    assert represent(Float(1.0)) == Float(1.0)
    assert represent(1.0 + I) == 1.0 + I


np = import_module('numpy')


def test_format_numpy():
    if not np:
        skip("numpy not installed.")

    for test in _tests:
        lhs = represent(test[0], basis=A, format='numpy')
        rhs = to_numpy(test[1])
        if isinstance(lhs, numpy_ndarray):
            assert (lhs == rhs).all()
        else:
            assert lhs == rhs


def test_scalar_numpy():
    if not np:
        skip("numpy not installed.")

    assert represent(Integer(1), format='numpy') == 1
    assert represent(Float(1.0), format='numpy') == 1.0
    assert represent(1.0 + I, format='numpy') == 1.0 + 1.0j


scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def test_format_scipy_sparse():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")

    for test in _tests:
        lhs = represent(test[0], basis=A, format='scipy.sparse')
        rhs = to_scipy_sparse(test[1])
        if isinstance(lhs, scipy_sparse_matrix):
            assert np.linalg.norm((lhs - rhs).todense()) == 0.0
        else:
            assert lhs == rhs


def test_scalar_scipy_sparse():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")

    assert represent(Integer(1), format='scipy.sparse') == 1
    assert represent(Float(1.0), format='scipy.sparse') == 1.0
    assert represent(1.0 + I, format='scipy.sparse') == 1.0 + 1.0j

x_ket = XKet('x')
x_bra = XBra('x')
x_op = XOp('X')


def test_innerprod_represent():
    assert rep_innerproduct(x_ket) == InnerProduct(XBra("x_1"), x_ket).doit()
    assert rep_innerproduct(x_bra) == InnerProduct(x_bra, XKet("x_1")).doit()
    raises(TypeError, lambda: rep_innerproduct(x_op))


def test_operator_represent():
    basis_kets = enumerate_states(operators_to_state(x_op), 1, 2)
    assert rep_expectation(
        x_op) == qapply(basis_kets[1].dual*x_op*basis_kets[0])


def test_enumerate_states():
    test = XKet("foo")
    assert enumerate_states(test, 1, 1) == [XKet("foo_1")]
    assert enumerate_states(
        test, [1, 2, 4]) == [XKet("foo_1"), XKet("foo_2"), XKet("foo_4")]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_matrix_distributions.py ===
from sympy.concrete.products import Product
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices import Determinant, Matrix, Trace, MatrixSymbol, MatrixSet
from sympy.stats import density, sample
from sympy.stats.matrix_distributions import (MatrixGammaDistribution,
                MatrixGamma, MatrixPSpace, Wishart, MatrixNormal, MatrixStudentT)
from sympy.testing.pytest import raises, skip
from sympy.external import import_module


def test_MatrixPSpace():
    M = MatrixGammaDistribution(1, 2, [[2, 1], [1, 2]])
    MP = MatrixPSpace('M', M, 2, 2)
    assert MP.distribution == M
    raises(ValueError, lambda: MatrixPSpace('M', M, 1.2, 2))

def test_MatrixGamma():
    M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
    assert M.pspace.distribution.set == MatrixSet(2, 2, S.Reals)
    assert isinstance(density(M), MatrixGammaDistribution)
    X = MatrixSymbol('X', 2, 2)
    num = exp(Trace(Matrix([[-S(1)/2, 0], [0, -S(1)/2]])*X))
    assert density(M)(X).doit() == num/(4*pi*sqrt(Determinant(X)))
    assert density(M)([[2, 1], [1, 2]]).doit() == sqrt(3)*exp(-2)/(12*pi)
    X = MatrixSymbol('X', 1, 2)
    Y = MatrixSymbol('Y', 1, 2)
    assert density(M)([X, Y]).doit() == exp(-X[0, 0]/2 - Y[0, 1]/2)/(4*pi*sqrt(
                                X[0, 0]*Y[0, 1] - X[0, 1]*Y[0, 0]))
    # symbolic
    a, b = symbols('a b', positive=True)
    d = symbols('d', positive=True, integer=True)
    Y = MatrixSymbol('Y', d, d)
    Z = MatrixSymbol('Z', 2, 2)
    SM = MatrixSymbol('SM', d, d)
    M2 = MatrixGamma('M2', a, b, SM)
    M3 = MatrixGamma('M3', 2, 3, [[2, 1], [1, 2]])
    k = Dummy('k')
    exprd = pi**(-d*(d - 1)/4)*b**(-a*d)*exp(Trace((-1/b)*SM**(-1)*Y)
        )*Determinant(SM)**(-a)*Determinant(Y)**(a - d/2 - S(1)/2)/Product(
        gamma(-k/2 + a + S(1)/2), (k, 1, d))
    assert density(M2)(Y).dummy_eq(exprd)
    raises(NotImplementedError, lambda: density(M3 + M)(Z))
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, -2, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [0]]))

def test_Wishart():
    W = Wishart('W', 5, [[1, 0], [0, 1]])
    assert W.pspace.distribution.set == MatrixSet(2, 2, S.Reals)
    X = MatrixSymbol('X', 2, 2)
    term1 = exp(Trace(Matrix([[-S(1)/2, 0], [0, -S(1)/2]])*X))
    assert density(W)(X).doit() == term1 * Determinant(X)/(24*pi)
    assert density(W)([[2, 1], [1, 2]]).doit() == exp(-2)/(8*pi)
    n = symbols('n', positive=True)
    d = symbols('d', positive=True, integer=True)
    Y = MatrixSymbol('Y', d, d)
    SM = MatrixSymbol('SM', d, d)
    W = Wishart('W', n, SM)
    k = Dummy('k')
    exprd = 2**(-d*n/2)*pi**(-d*(d - 1)/4)*exp(Trace(-(S(1)/2)*SM**(-1)*Y)
    )*Determinant(SM)**(-n/2)*Determinant(Y)**(
    -d/2 + n/2 - S(1)/2)/Product(gamma(-k/2 + n/2 + S(1)/2), (k, 1, d))
    assert density(W)(Y).dummy_eq(exprd)
    raises(ValueError, lambda: density(W)(1))
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [2, 1]]))
    raises(ValueError, lambda: Wishart('W',  2, [[1, 0], [0]]))

def test_MatrixNormal():
    M = MatrixNormal('M', [[5, 6]], [4], [[2, 1], [1, 2]])
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    X = MatrixSymbol('X', 1, 2)
    term1 = exp(-Trace(Matrix([[ S(2)/3, -S(1)/3], [-S(1)/3, S(2)/3]])*(
            Matrix([[-5], [-6]]) + X.T)*Matrix([[S(1)/4]])*(Matrix([[-5, -6]]) + X))/2)
    assert density(M)(X).doit() == (sqrt(3)) * term1/(24*pi)
    assert density(M)([[7, 8]]).doit() == sqrt(3)*exp(-S(1)/3)/(24*pi)
    d, n = symbols('d n', positive=True, integer=True)
    SM2 = MatrixSymbol('SM2', d, d)
    SM1 = MatrixSymbol('SM1', n, n)
    LM = MatrixSymbol('LM', n, d)
    Y = MatrixSymbol('Y', n, d)
    M = MatrixNormal('M', LM, SM1, SM2)
    exprd = (2*pi)**(-d*n/2)*exp(-Trace(SM2**(-1)*(-LM.T + Y.T)*SM1**(-1)*(-LM + Y)
        )/2)*Determinant(SM1)**(-d/2)*Determinant(SM2)**(-n/2)
    assert density(M)(Y).doit() == exprd
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0]]))
    raises(ValueError, lambda: MatrixNormal('M', [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))
    raises(ValueError, lambda: MatrixNormal('M', [[1, 2]], [1], [[1, 0]]))

def test_MatrixStudentT():
    M = MatrixStudentT('M', 2, [[5, 6]], [[2, 1], [1, 2]], [4])
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    X = MatrixSymbol('X', 1, 2)
    D = pi ** (-1.0) * Determinant(Matrix([[4]])) ** (-1.0) * Determinant(Matrix([[2, 1], [1, 2]])) \
        ** (-0.5) / Determinant(Matrix([[S(1) / 4]]) * (Matrix([[-5, -6]]) + X)
                                * Matrix([[S(2) / 3, -S(1) / 3], [-S(1) / 3, S(2) / 3]]) * (
                                        Matrix([[-5], [-6]]) + X.T) + Matrix([[1]])) ** 2
    assert density(M)(X) == D

    v = symbols('v', positive=True)
    n, p = 1, 2
    Omega = MatrixSymbol('Omega', p, p)
    Sigma = MatrixSymbol('Sigma', n, n)
    Location = MatrixSymbol('Location', n, p)
    Y = MatrixSymbol('Y', n, p)
    M = MatrixStudentT('M', v, Location, Omega, Sigma)

    exprd = gamma(v/2 + 1)*Determinant(Matrix([[1]]) + Sigma**(-1)*(-Location + Y)*Omega**(-1)*(-Location.T + Y.T))**(-v/2 - 1) / \
            (pi*gamma(v/2)*sqrt(Determinant(Omega))*Determinant(Sigma))

    assert density(M)(Y) == exprd
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1], [2]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [1], [[1, 0]]))
    raises(ValueError, lambda: MatrixStudentT('M', -1, [1, 2], [[1, 0], [0, 1]], [4]))

def test_sample_scipy():
    distribs_scipy = [
        MatrixNormal('M', [[5, 6]], [4], [[2, 1], [1, 2]]),
        Wishart('W', 5, [[1, 0], [0, 1]])
    ]

    size = 5
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size)
            for sam in samps:
                assert Matrix(sam) in X.pspace.distribution.set
        M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
        raises(NotImplementedError, lambda: sample(M, size=3))

def test_sample_pymc():
    distribs_pymc = [
        MatrixNormal('M', [[5, 6], [3, 4]], [[1, 0], [0, 1]], [[2, 1], [1, 2]]),
        Wishart('W', 7, [[2, 1], [1, 2]])
    ]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert Matrix(sam) in X.pspace.distribution.set
        M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
        raises(NotImplementedError, lambda: sample(M, size=3))

def test_sample_seed():
    X = MatrixNormal('M', [[5, 6], [3, 4]], [[1, 0], [0, 1]], [[2, 1], [1, 2]])

    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0, s1, s2 = [], [], []
                s0 = sample(X, size=10, library=lib, seed=0)
                s1 = sample(X, size=10, library=lib, seed=0)
                s2 = sample(X, size=10, library=lib, seed=1)
                for i in range(10):
                    assert (s0[i] == s1[i]).all()
                    assert (s1[i] != s2[i]).all()

        except NotImplementedError:
            continue

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_cov_corr.py ===
import math

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
    isna,
)
import pandas._testing as tm


class TestSeriesCov:
    def test_cov(self, datetime_series):
        # full overlap
        tm.assert_almost_equal(
            datetime_series.cov(datetime_series), datetime_series.std() ** 2
        )

        # partial overlap
        tm.assert_almost_equal(
            datetime_series[:15].cov(datetime_series[5:]),
            datetime_series[5:15].std() ** 2,
        )

        # No overlap
        assert np.isnan(datetime_series[::2].cov(datetime_series[1::2]))

        # all NA
        cp = datetime_series[:10].copy()
        cp[:] = np.nan
        assert isna(cp.cov(cp))

        # min_periods
        assert isna(datetime_series[:15].cov(datetime_series[5:], min_periods=12))

        ts1 = datetime_series[:15].reindex(datetime_series.index)
        ts2 = datetime_series[5:].reindex(datetime_series.index)
        assert isna(ts1.cov(ts2, min_periods=12))

    @pytest.mark.parametrize("test_ddof", [None, 0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", ["float64", "Float64"])
    def test_cov_ddof(self, test_ddof, dtype):
        # GH#34611
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)

        s1 = Series(np_array1, dtype=dtype)
        s2 = Series(np_array2, dtype=dtype)

        result = s1.cov(s2, ddof=test_ddof)
        expected = np.cov(np_array1, np_array2, ddof=test_ddof)[0][1]
        assert math.isclose(expected, result)


class TestSeriesCorr:
    @pytest.mark.parametrize("dtype", ["float64", "Float64"])
    def test_corr(self, datetime_series, dtype):
        stats = pytest.importorskip("scipy.stats")

        datetime_series = datetime_series.astype(dtype)

        # full overlap
        tm.assert_almost_equal(datetime_series.corr(datetime_series), 1)

        # partial overlap
        tm.assert_almost_equal(datetime_series[:15].corr(datetime_series[5:]), 1)

        assert isna(datetime_series[:15].corr(datetime_series[5:], min_periods=12))

        ts1 = datetime_series[:15].reindex(datetime_series.index)
        ts2 = datetime_series[5:].reindex(datetime_series.index)
        assert isna(ts1.corr(ts2, min_periods=12))

        # No overlap
        assert np.isnan(datetime_series[::2].corr(datetime_series[1::2]))

        # all NA
        cp = datetime_series[:10].copy()
        cp[:] = np.nan
        assert isna(cp.corr(cp))

        A = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        B = A.copy()
        result = A.corr(B)
        expected, _ = stats.pearsonr(A, B)
        tm.assert_almost_equal(result, expected)

    def test_corr_rank(self):
        stats = pytest.importorskip("scipy.stats")

        # kendall and spearman
        A = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        B = A.copy()
        A[-5:] = A[:5].copy()
        result = A.corr(B, method="kendall")
        expected = stats.kendalltau(A, B)[0]
        tm.assert_almost_equal(result, expected)

        result = A.corr(B, method="spearman")
        expected = stats.spearmanr(A, B)[0]
        tm.assert_almost_equal(result, expected)

        # results from R
        A = Series(
            [
                -0.89926396,
                0.94209606,
                -1.03289164,
                -0.95445587,
                0.76910310,
                -0.06430576,
                -2.09704447,
                0.40660407,
                -0.89926396,
                0.94209606,
            ]
        )
        B = Series(
            [
                -1.01270225,
                -0.62210117,
                -1.56895827,
                0.59592943,
                -0.01680292,
                1.17258718,
                -1.06009347,
                -0.10222060,
                -0.89076239,
                0.89372375,
            ]
        )
        kexp = 0.4319297
        sexp = 0.5853767
        tm.assert_almost_equal(A.corr(B, method="kendall"), kexp)
        tm.assert_almost_equal(A.corr(B, method="spearman"), sexp)

    def test_corr_invalid_method(self):
        # GH PR #22298
        s1 = Series(np.random.default_rng(2).standard_normal(10))
        s2 = Series(np.random.default_rng(2).standard_normal(10))
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        with pytest.raises(ValueError, match=msg):
            s1.corr(s2, method="____")

    def test_corr_callable_method(self, datetime_series):
        # simple correlation example
        # returns 1 if exact equality, 0 otherwise
        my_corr = lambda a, b: 1.0 if (a == b).all() else 0.0

        # simple example
        s1 = Series([1, 2, 3, 4, 5])
        s2 = Series([5, 4, 3, 2, 1])
        expected = 0
        tm.assert_almost_equal(s1.corr(s2, method=my_corr), expected)

        # full overlap
        tm.assert_almost_equal(
            datetime_series.corr(datetime_series, method=my_corr), 1.0
        )

        # partial overlap
        tm.assert_almost_equal(
            datetime_series[:15].corr(datetime_series[5:], method=my_corr), 1.0
        )

        # No overlap
        assert np.isnan(
            datetime_series[::2].corr(datetime_series[1::2], method=my_corr)
        )

        # dataframe example
        df = pd.DataFrame([s1, s2])
        expected = pd.DataFrame([{0: 1.0, 1: 0}, {0: 0, 1: 1.0}])
        tm.assert_almost_equal(df.transpose().corr(method=my_corr), expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_polymatrix.py ===
from sympy.testing.pytest import raises

from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly

from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ

from sympy.abc import x, y


def _test_polymatrix():
    pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    v1 = PolyMatrix([[1, 0], [-1, 0]], ring='ZZ[x]')
    m1 = PolyMatrix([[1, 0], [-1, 0]], ring='ZZ[x]')
    A = PolyMatrix([[Poly(x**2 + x, x), Poly(0, x)], \
                    [Poly(x**3 - x + 1, x), Poly(0, x)]])
    B = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(-x**2, x), Poly(x, x)]])
    assert A.ring == ZZ[x]
    assert isinstance(pm1*v1, PolyMatrix)
    assert pm1*v1 == A
    assert pm1*m1 == A
    assert v1*pm1 == B

    pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**2, x, domain='QQ'), \
                    Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    assert pm2.ring == QQ[x]
    v2 = PolyMatrix([1, 0, 0, 0, 0, 0], ring='ZZ[x]')
    m2 = PolyMatrix([1, 0, 0, 0, 0, 0], ring='ZZ[x]')
    C = PolyMatrix([[Poly(x**2, x, domain='QQ')]])
    assert pm2*v2 == C
    assert pm2*m2 == C

    pm3 = PolyMatrix([[Poly(x**2, x), S.One]], ring='ZZ[x]')
    v3 = S.Half*pm3
    assert v3 == PolyMatrix([[Poly(S.Half*x**2, x, domain='QQ'), S.Half]], ring='QQ[x]')
    assert pm3*S.Half == v3
    assert v3.ring == QQ[x]

    pm4 = PolyMatrix([[Poly(x**2, x, domain='ZZ'), Poly(-x**2, x, domain='ZZ')]])
    v4 = PolyMatrix([1, -1], ring='ZZ[x]')
    assert pm4*v4 == PolyMatrix([[Poly(2*x**2, x, domain='ZZ')]])

    assert len(PolyMatrix(ring=ZZ[x])) == 0
    assert PolyMatrix([1, 0, 0, 1], x)/(-1) == PolyMatrix([-1, 0, 0, -1], x)


def test_polymatrix_constructor():
    M1 = PolyMatrix([[x, y]], ring=QQ[x,y])
    assert M1.ring == QQ[x,y]
    assert M1.domain == QQ
    assert M1.gens == (x, y)
    assert M1.shape == (1, 2)
    assert M1.rows == 1
    assert M1.cols == 2
    assert len(M1) == 2
    assert list(M1) == [Poly(x, (x, y), domain=QQ), Poly(y, (x, y), domain=QQ)]

    M2 = PolyMatrix([[x, y]], ring=QQ[x][y])
    assert M2.ring == QQ[x][y]
    assert M2.domain == QQ[x]
    assert M2.gens == (y,)
    assert M2.shape == (1, 2)
    assert M2.rows == 1
    assert M2.cols == 2
    assert len(M2) == 2
    assert list(M2) == [Poly(x, (y,), domain=QQ[x]), Poly(y, (y,), domain=QQ[x])]

    assert PolyMatrix([[x, y]], y) == PolyMatrix([[x, y]], ring=ZZ.frac_field(x)[y])
    assert PolyMatrix([[x, y]], ring='ZZ[x,y]') == PolyMatrix([[x, y]], ring=ZZ[x,y])

    assert PolyMatrix([[x, y]], (x, y)) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix([[x, y]], x, y) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix([x, y]) == PolyMatrix([[x], [y]], ring=QQ[x,y])
    assert PolyMatrix(1, 2, [x, y]) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix(1, 2, lambda i,j: [x,y][j]) == PolyMatrix([[x, y]], ring=QQ[x,y])
    assert PolyMatrix(0, 2, [], x, y).shape == (0, 2)
    assert PolyMatrix(2, 0, [], x, y).shape == (2, 0)
    assert PolyMatrix([[], []], x, y).shape == (2, 0)
    assert PolyMatrix(ring=QQ[x,y]) == PolyMatrix(0, 0, [], ring=QQ[x,y]) == PolyMatrix([], ring=QQ[x,y])
    raises(TypeError, lambda: PolyMatrix())
    raises(TypeError, lambda: PolyMatrix(1))

    assert PolyMatrix([Poly(x), Poly(y)]) == PolyMatrix([[x], [y]], ring=ZZ[x,y])

    # XXX: Maybe a bug in parallel_poly_from_expr (x lost from gens and domain):
    assert PolyMatrix([Poly(y, x), 1]) == PolyMatrix([[y], [1]], ring=QQ[y])


def test_polymatrix_eq():
    assert (PolyMatrix([x]) == PolyMatrix([x])) is True
    assert (PolyMatrix([y]) == PolyMatrix([x])) is False
    assert (PolyMatrix([x]) != PolyMatrix([x])) is False
    assert (PolyMatrix([y]) != PolyMatrix([x])) is True

    assert PolyMatrix([[x, y]]) != PolyMatrix([x, y]) == PolyMatrix([[x], [y]])

    assert PolyMatrix([x], ring=QQ[x]) != PolyMatrix([x], ring=ZZ[x])

    assert PolyMatrix([x]) != Matrix([x])
    assert PolyMatrix([x]).to_Matrix() == Matrix([x])

    assert PolyMatrix([1], x) == PolyMatrix([1], x)
    assert PolyMatrix([1], x) != PolyMatrix([1], y)


def test_polymatrix_from_Matrix():
    assert PolyMatrix.from_Matrix(Matrix([1, 2]), x) == PolyMatrix([1, 2], x, ring=QQ[x])
    assert PolyMatrix.from_Matrix(Matrix([1]), ring=QQ[x]) == PolyMatrix([1], x)
    pmx = PolyMatrix([1, 2], x)
    pmy = PolyMatrix([1, 2], y)
    assert pmx != pmy
    assert pmx.set_gens(y) == pmy


def test_polymatrix_repr():
    assert repr(PolyMatrix([[1, 2]], x)) == 'PolyMatrix([[1, 2]], ring=QQ[x])'
    assert repr(PolyMatrix(0, 2, [], x)) == 'PolyMatrix(0, 2, [], ring=QQ[x])'


def test_polymatrix_getitem():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M[:, :] == M
    assert M[0, :] == PolyMatrix([[1, 2]], x)
    assert M[:, 0] == PolyMatrix([1, 3], x)
    assert M[0, 0] == Poly(1, x, domain=QQ)
    assert M[0] == Poly(1, x, domain=QQ)
    assert M[:2] == [Poly(1, x, domain=QQ), Poly(2, x, domain=QQ)]


def test_polymatrix_arithmetic():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M + M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M - M == PolyMatrix([[0, 0], [0, 0]], x)
    assert -M == PolyMatrix([[-1, -2], [-3, -4]], x)
    raises(TypeError, lambda: M + 1)
    raises(TypeError, lambda: M - 1)
    raises(TypeError, lambda: 1 + M)
    raises(TypeError, lambda: 1 - M)

    assert M * M == PolyMatrix([[7, 10], [15, 22]], x)
    assert 2 * M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M * 2 == PolyMatrix([[2, 4], [6, 8]], x)
    assert S(2) * M == PolyMatrix([[2, 4], [6, 8]], x)
    assert M * S(2) == PolyMatrix([[2, 4], [6, 8]], x)
    raises(TypeError, lambda: [] * M)
    raises(TypeError, lambda: M * [])
    M2 = PolyMatrix([[1, 2]], ring=ZZ[x])
    assert S.Half * M2 == PolyMatrix([[S.Half, 1]], ring=QQ[x])
    assert M2 * S.Half == PolyMatrix([[S.Half, 1]], ring=QQ[x])

    assert M / 2 == PolyMatrix([[S(1)/2, 1], [S(3)/2, 2]], x)
    assert M / Poly(2, x) == PolyMatrix([[S(1)/2, 1], [S(3)/2, 2]], x)
    raises(TypeError, lambda: M / [])


def test_polymatrix_manipulations():
    M1 = PolyMatrix([[1, 2], [3, 4]], x)
    assert M1.transpose() == PolyMatrix([[1, 3], [2, 4]], x)
    M2 = PolyMatrix([[5, 6], [7, 8]], x)
    assert M1.row_join(M2) == PolyMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], x)
    assert M1.col_join(M2) == PolyMatrix([[1, 2], [3, 4], [5, 6], [7, 8]], x)
    assert M1.applyfunc(lambda e: 2*e) == PolyMatrix([[2, 4], [6, 8]], x)


def test_polymatrix_ones_zeros():
    assert PolyMatrix.zeros(1, 2, x) == PolyMatrix([[0, 0]], x)
    assert PolyMatrix.eye(2, x) == PolyMatrix([[1, 0], [0, 1]], x)


def test_polymatrix_rref():
    M = PolyMatrix([[1, 2], [3, 4]], x)
    assert M.rref() == (PolyMatrix.eye(2, x), (0, 1))
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).rref())
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).rref())


def test_polymatrix_nullspace():
    M = PolyMatrix([[1, 2], [3, 6]], x)
    assert M.nullspace() == [PolyMatrix([-2, 1], x)]
    raises(ValueError, lambda: PolyMatrix([1, 2], ring=ZZ[x]).nullspace())
    raises(ValueError, lambda: PolyMatrix([1, x], ring=QQ[x]).nullspace())
    assert M.rank() == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\test_period.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray

# ----------------------------------------------------------------------------
# Dtype


def test_registered():
    assert PeriodDtype in registry.dtypes
    result = registry.find("Period[D]")
    expected = PeriodDtype("D")
    assert result == expected


# ----------------------------------------------------------------------------
# period_array


def test_asi8():
    result = PeriodArray._from_sequence(["2000", "2001", None], dtype="period[D]").asi8
    expected = np.array([10957, 11323, iNaT])
    tm.assert_numpy_array_equal(result, expected)


def test_take_raises():
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr.take([0, -1], allow_fill=True, fill_value=pd.Period("2000", freq="W"))

    msg = "value should be a 'Period' or 'NaT'. Got 'str' instead"
    with pytest.raises(TypeError, match=msg):
        arr.take([0, -1], allow_fill=True, fill_value="foo")


def test_fillna_raises():
    arr = PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")
    with pytest.raises(ValueError, match="Length"):
        arr.fillna(arr[:2])


def test_fillna_copies():
    arr = PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")
    result = arr.fillna(pd.Period("2000", "D"))
    assert result is not arr


# ----------------------------------------------------------------------------
# setitem


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ([0], pd.Period("2000", "D"), [10957, 1, 2]),
        ([0], None, [iNaT, 1, 2]),
        ([0], np.nan, [iNaT, 1, 2]),
        ([0, 1, 2], pd.Period("2000", "D"), [10957] * 3),
        (
            [0, 1, 2],
            [pd.Period("2000", "D"), pd.Period("2001", "D"), pd.Period("2002", "D")],
            [10957, 11323, 11688],
        ),
    ],
)
def test_setitem(key, value, expected):
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    expected = PeriodArray(expected, dtype="period[D]")
    arr[key] = value
    tm.assert_period_array_equal(arr, expected)


def test_setitem_raises_incompatible_freq():
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr[0] = pd.Period("2000", freq="Y")

    other = PeriodArray._from_sequence(["2000", "2001"], dtype="period[Y]")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr[[0, 1]] = other


def test_setitem_raises_length():
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    with pytest.raises(ValueError, match="length"):
        arr[[0, 1]] = [pd.Period("2000", freq="D")]


def test_setitem_raises_type():
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    with pytest.raises(TypeError, match="int"):
        arr[0] = 1


# ----------------------------------------------------------------------------
# Ops


def test_sub_period():
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    other = pd.Period("2000", freq="M")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr - other


def test_sub_period_overflow():
    # GH#47538
    dti = pd.date_range("1677-09-22", periods=2, freq="D")
    pi = dti.to_period("ns")

    per = pd.Period._from_ordinal(10**14, pi.freq)

    with pytest.raises(OverflowError, match="Overflow in int64 addition"):
        pi - per

    with pytest.raises(OverflowError, match="Overflow in int64 addition"):
        per - pi


# ----------------------------------------------------------------------------
# Methods


@pytest.mark.parametrize(
    "other",
    [
        pd.Period("2000", freq="h"),
        PeriodArray._from_sequence(["2000", "2001", "2000"], dtype="period[h]"),
    ],
)
def test_where_different_freq_raises(other):
    # GH#45768 The PeriodArray method raises, the Series method coerces
    ser = pd.Series(
        PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")
    )
    cond = np.array([True, False, True])

    with pytest.raises(IncompatibleFrequency, match="freq"):
        ser.array._where(cond, other)

    res = ser.where(cond, other)
    expected = ser.astype(object).where(cond, other)
    tm.assert_series_equal(res, expected)


# ----------------------------------------------------------------------------
# Printing


def test_repr_small():
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    result = str(arr)
    expected = (
        "<PeriodArray>\n['2000-01-01', '2001-01-01']\nLength: 2, dtype: period[D]"
    )
    assert result == expected


def test_repr_large():
    arr = PeriodArray._from_sequence(["2000", "2001"] * 500, dtype="period[D]")
    result = str(arr)
    expected = (
        "<PeriodArray>\n"
        "['2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01',\n"
        " ...\n"
        " '2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01']\n"
        "Length: 1000, dtype: period[D]"
    )
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\index\test_index.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def index_view(index_data=[1, 2]):
    df = DataFrame({"a": index_data, "b": 1.5})
    view = df[:]
    df = df.set_index("a", drop=True)
    idx = df.index
    # df = None
    return idx, view


def test_set_index_update_column(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2], "b": 1})
    df = df.set_index("a", drop=False)
    expected = df.index.copy(deep=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(df.index, expected)
    else:
        tm.assert_index_equal(df.index, Index([100, 2], name="a"))


def test_set_index_drop_update_column(using_copy_on_write):
    df = DataFrame({"a": [1, 2], "b": 1.5})
    view = df[:]
    df = df.set_index("a", drop=True)
    expected = df.index.copy(deep=True)
    view.iloc[0, 0] = 100
    tm.assert_index_equal(df.index, expected)


def test_set_index_series(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2], "b": 1.5})
    ser = Series([10, 11])
    df = df.set_index(ser)
    expected = df.index.copy(deep=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(df.index, expected)
    else:
        tm.assert_index_equal(df.index, Index([100, 11]))


def test_assign_index_as_series(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2], "b": 1.5})
    ser = Series([10, 11])
    df.index = ser
    expected = df.index.copy(deep=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(df.index, expected)
    else:
        tm.assert_index_equal(df.index, Index([100, 11]))


def test_assign_index_as_index(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2], "b": 1.5})
    ser = Series([10, 11])
    rhs_index = Index(ser)
    df.index = rhs_index
    rhs_index = None  # overwrite to clear reference
    expected = df.index.copy(deep=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(df.index, expected)
    else:
        tm.assert_index_equal(df.index, Index([100, 11]))


def test_index_from_series(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2])
    idx = Index(ser)
    expected = idx.copy(deep=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected)
    else:
        tm.assert_index_equal(idx, Index([100, 2]))


def test_index_from_series_copy(using_copy_on_write):
    ser = Series([1, 2])
    idx = Index(ser, copy=True)  # noqa: F841
    arr = get_array(ser)
    ser.iloc[0] = 100
    assert np.shares_memory(get_array(ser), arr)


def test_index_from_index(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2])
    idx = Index(ser)
    idx = Index(idx)
    expected = idx.copy(deep=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected)
    else:
        tm.assert_index_equal(idx, Index([100, 2]))


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x._shallow_copy(x._values),
        lambda x: x.view(),
        lambda x: x.take([0, 1]),
        lambda x: x.repeat([1, 1]),
        lambda x: x[slice(0, 2)],
        lambda x: x[[0, 1]],
        lambda x: x._getitem_slice(slice(0, 2)),
        lambda x: x.delete([]),
        lambda x: x.rename("b"),
        lambda x: x.astype("Int64", copy=False),
    ],
    ids=[
        "_shallow_copy",
        "view",
        "take",
        "repeat",
        "getitem_slice",
        "getitem_list",
        "_getitem_slice",
        "delete",
        "rename",
        "astype",
    ],
)
def test_index_ops(using_copy_on_write, func, request):
    idx, view_ = index_view()
    expected = idx.copy(deep=True)
    if "astype" in request.node.callspec.id:
        expected = expected.astype("Int64")
    idx = func(idx)
    view_.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected, check_names=False)


def test_infer_objects(using_copy_on_write):
    idx, view_ = index_view(["a", "b"])
    expected = idx.copy(deep=True)
    idx = idx.infer_objects(copy=False)
    view_.iloc[0, 0] = "aaaa"
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected, check_names=False)


def test_index_to_frame(using_copy_on_write):
    idx = Index([1, 2, 3], name="a")
    expected = idx.copy(deep=True)
    df = idx.to_frame()
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "a"), idx._values)
        assert not df._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(df, "a"), idx._values)

    df.iloc[0, 0] = 100
    tm.assert_index_equal(idx, expected)


def test_index_values(using_copy_on_write):
    idx = Index([1, 2, 3])
    result = idx.values
    if using_copy_on_write:
        assert result.flags.writeable is False
    else:
        assert result.flags.writeable is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_rename.py ===
from datetime import datetime
import re

import numpy as np
import pytest

from pandas import (
    Index,
    MultiIndex,
    Series,
    array,
)
import pandas._testing as tm


class TestRename:
    def test_rename(self, datetime_series):
        ts = datetime_series
        renamer = lambda x: x.strftime("%Y%m%d")
        renamed = ts.rename(renamer)
        assert renamed.index[0] == renamer(ts.index[0])

        # dict
        rename_dict = dict(zip(ts.index, renamed.index))
        renamed2 = ts.rename(rename_dict)
        tm.assert_series_equal(renamed, renamed2)

    def test_rename_partial_dict(self):
        # partial dict
        ser = Series(np.arange(4), index=["a", "b", "c", "d"], dtype="int64")
        renamed = ser.rename({"b": "foo", "d": "bar"})
        tm.assert_index_equal(renamed.index, Index(["a", "foo", "c", "bar"]))

    def test_rename_retain_index_name(self):
        # index with name
        renamer = Series(
            np.arange(4), index=Index(["a", "b", "c", "d"], name="name"), dtype="int64"
        )
        renamed = renamer.rename({})
        assert renamed.index.name == renamer.index.name

    def test_rename_by_series(self):
        ser = Series(range(5), name="foo")
        renamer = Series({1: 10, 2: 20})
        result = ser.rename(renamer)
        expected = Series(range(5), index=[0, 10, 20, 3, 4], name="foo")
        tm.assert_series_equal(result, expected)

    def test_rename_set_name(self, using_infer_string):
        ser = Series(range(4), index=list("abcd"))
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:
            result = ser.rename(name)
            assert result.name == name
            if using_infer_string:
                tm.assert_extension_array_equal(result.index.values, ser.index.values)
            else:
                tm.assert_numpy_array_equal(result.index.values, ser.index.values)
            assert ser.name is None

    def test_rename_set_name_inplace(self, using_infer_string):
        ser = Series(range(3), index=list("abc"))
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:
            ser.rename(name, inplace=True)
            assert ser.name == name
            exp = np.array(["a", "b", "c"], dtype=np.object_)
            if using_infer_string:
                exp = array(exp, dtype="str")
                tm.assert_extension_array_equal(ser.index.values, exp)
            else:
                tm.assert_numpy_array_equal(ser.index.values, exp)

    def test_rename_axis_supported(self):
        # Supporting axis for compatibility, detailed in GH-18589
        ser = Series(range(5))
        ser.rename({}, axis=0)
        ser.rename({}, axis="index")

        with pytest.raises(ValueError, match="No axis named 5"):
            ser.rename({}, axis=5)

    def test_rename_inplace(self, datetime_series):
        renamer = lambda x: x.strftime("%Y%m%d")
        expected = renamer(datetime_series.index[0])

        datetime_series.rename(renamer, inplace=True)
        assert datetime_series.index[0] == expected

    def test_rename_with_custom_indexer(self):
        # GH 27814
        class MyIndexer:
            pass

        ix = MyIndexer()
        ser = Series([1, 2, 3]).rename(ix)
        assert ser.name is ix

    def test_rename_with_custom_indexer_inplace(self):
        # GH 27814
        class MyIndexer:
            pass

        ix = MyIndexer()
        ser = Series([1, 2, 3])
        ser.rename(ix, inplace=True)
        assert ser.name is ix

    def test_rename_callable(self):
        # GH 17407
        ser = Series(range(1, 6), index=Index(range(2, 7), name="IntIndex"))
        result = ser.rename(str)
        expected = ser.rename(lambda i: str(i))
        tm.assert_series_equal(result, expected)

        assert result.name == expected.name

    def test_rename_none(self):
        # GH 40977
        ser = Series([1, 2], name="foo")
        result = ser.rename(None)
        expected = Series([1, 2])
        tm.assert_series_equal(result, expected)

    def test_rename_series_with_multiindex(self):
        # issue #43659
        arrays = [
            ["bar", "baz", "baz", "foo", "qux"],
            ["one", "one", "two", "two", "one"],
        ]

        index = MultiIndex.from_arrays(arrays, names=["first", "second"])
        ser = Series(np.ones(5), index=index)
        result = ser.rename(index={"one": "yes"}, level="second", errors="raise")

        arrays_expected = [
            ["bar", "baz", "baz", "foo", "qux"],
            ["yes", "yes", "two", "two", "yes"],
        ]

        index_expected = MultiIndex.from_arrays(
            arrays_expected, names=["first", "second"]
        )
        series_expected = Series(np.ones(5), index=index_expected)

        tm.assert_series_equal(result, series_expected)

    def test_rename_series_with_multiindex_keeps_ea_dtypes(self):
        # GH21055
        arrays = [
            Index([1, 2, 3], dtype="Int64").astype("category"),
            Index([1, 2, 3], dtype="Int64"),
        ]
        mi = MultiIndex.from_arrays(arrays, names=["A", "B"])
        ser = Series(1, index=mi)
        result = ser.rename({1: 4}, level=1)

        arrays_expected = [
            Index([1, 2, 3], dtype="Int64").astype("category"),
            Index([4, 2, 3], dtype="Int64"),
        ]
        mi_expected = MultiIndex.from_arrays(arrays_expected, names=["A", "B"])
        expected = Series(1, index=mi_expected)

        tm.assert_series_equal(result, expected)

    def test_rename_error_arg(self):
        # GH 46889
        ser = Series(["foo", "bar"])
        match = re.escape("[2] not found in axis")
        with pytest.raises(KeyError, match=match):
            ser.rename({2: 9}, errors="raise")

    def test_rename_copy_false(self, using_copy_on_write, warn_copy_on_write):
        # GH 46889
        ser = Series(["foo", "bar"])
        ser_orig = ser.copy()
        shallow_copy = ser.rename({1: 9}, copy=False)
        with tm.assert_cow_warning(warn_copy_on_write):
            ser[0] = "foobar"
        if using_copy_on_write:
            assert ser_orig[0] == shallow_copy[0]
            assert ser_orig[1] == shallow_copy[9]
        else:
            assert ser[0] == shallow_copy[0]
            assert ser[1] == shallow_copy[9]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_rigidbody.py ===
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia, Inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy import expand, zeros, simplify, symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_rigidbody_default():
    # Test default
    b = RigidBody('B')
    I = inertia(b.frame, *symbols('B_ixx B_iyy B_izz B_ixy B_iyz B_izx'))
    assert b.name == 'B'
    assert b.mass == symbols('B_mass')
    assert b.masscenter.name == 'B_masscenter'
    assert b.inertia == (I, b.masscenter)
    assert b.central_inertia == I
    assert b.frame.name == 'B_frame'
    assert b.__str__() == 'B'
    assert b.__repr__() == (
        "RigidBody('B', masscenter=B_masscenter, frame=B_frame, mass=B_mass, "
        "inertia=Inertia(dyadic=B_ixx*(B_frame.x|B_frame.x) + "
        "B_ixy*(B_frame.x|B_frame.y) + B_izx*(B_frame.x|B_frame.z) + "
        "B_ixy*(B_frame.y|B_frame.x) + B_iyy*(B_frame.y|B_frame.y) + "
        "B_iyz*(B_frame.y|B_frame.z) + B_izx*(B_frame.z|B_frame.x) + "
        "B_iyz*(B_frame.z|B_frame.y) + B_izz*(B_frame.z|B_frame.z), "
        "point=B_masscenter))")


def test_rigidbody():
    m, m2, v1, v2, v3, omega = symbols('m m2 v1 v2 v3 omega')
    A = ReferenceFrame('A')
    A2 = ReferenceFrame('A2')
    P = Point('P')
    P2 = Point('P2')
    I = Dyadic(0)
    I2 = Dyadic(0)
    B = RigidBody('B', P, A, m, (I, P))
    assert B.mass == m
    assert B.frame == A
    assert B.masscenter == P
    assert B.inertia == (I, B.masscenter)

    B.mass = m2
    B.frame = A2
    B.masscenter = P2
    B.inertia = (I2, B.masscenter)
    raises(TypeError, lambda: RigidBody(P, P, A, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, P, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (P, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (I, I)))
    assert B.__str__() == 'B'
    assert B.mass == m2
    assert B.frame == A2
    assert B.masscenter == P2
    assert B.inertia == (I2, B.masscenter)
    assert isinstance(B.inertia, Inertia)

    # Testing linear momentum function assuming A2 is the inertial frame
    N = ReferenceFrame('N')
    P2.set_vel(N, v1 * N.x + v2 * N.y + v3 * N.z)
    assert B.linear_momentum(N) == m2 * (v1 * N.x + v2 * N.y + v3 * N.z)


def test_rigidbody2():
    M, v, r, omega, g, h = dynamicsymbols('M v r omega g h')
    N = ReferenceFrame('N')
    b = ReferenceFrame('b')
    b.set_ang_vel(N, omega * b.x)
    P = Point('P')
    I = outer(b.x, b.x)
    Inertia_tuple = (I, P)
    B = RigidBody('B', P, b, M, Inertia_tuple)
    P.set_vel(N, v * b.x)
    assert B.angular_momentum(P, N) == omega * b.x
    O = Point('O')
    O.set_vel(N, v * b.x)
    P.set_pos(O, r * b.y)
    assert B.angular_momentum(O, N) == omega * b.x - M*v*r*b.z
    B.potential_energy = M * g * h
    assert B.potential_energy == M * g * h
    assert expand(2 * B.kinetic_energy(N)) == omega**2 + M * v**2


def test_rigidbody3():
    q1, q2, q3, q4 = dynamicsymbols('q1:5')
    p1, p2, p3 = symbols('p1:4')
    m = symbols('m')

    A = ReferenceFrame('A')
    B = A.orientnew('B', 'axis', [q1, A.x])
    O = Point('O')
    O.set_vel(A, q2*A.x + q3*A.y + q4*A.z)
    P = O.locatenew('P', p1*B.x + p2*B.y + p3*B.z)
    P.v2pt_theory(O, A, B)
    I = outer(B.x, B.x)

    rb1 = RigidBody('rb1', P, B, m, (I, P))
    # I_S/O = I_S/S* + I_S*/O
    rb2 = RigidBody('rb2', P, B, m,
                    (I + inertia_of_point_mass(m, P.pos_from(O), B), O))

    assert rb1.central_inertia == rb2.central_inertia
    assert rb1.angular_momentum(O, A) == rb2.angular_momentum(O, A)


def test_pendulum_angular_momentum():
    """Consider a pendulum of length OA = 2a, of mass m as a rigid body of
    center of mass G (OG = a) which turn around (O,z). The angle between the
    reference frame R and the rod is q.  The inertia of the body is I =
    (G,0,ma^2/3,ma^2/3). """

    m, a = symbols('m, a')
    q = dynamicsymbols('q')

    R = ReferenceFrame('R')
    R1 = R.orientnew('R1', 'Axis', [q, R.z])
    R1.set_ang_vel(R, q.diff() * R.z)

    I = inertia(R1, 0, m * a**2 / 3, m * a**2 / 3)

    O = Point('O')

    A = O.locatenew('A', 2*a * R1.x)
    G = O.locatenew('G', a * R1.x)

    S = RigidBody('S', G, R1, m, (I, G))

    O.set_vel(R, 0)
    A.v2pt_theory(O, R, R1)
    G.v2pt_theory(O, R, R1)

    assert (4 * m * a**2 / 3 * q.diff() * R.z -
            S.angular_momentum(O, R).express(R)) == 0


def test_rigidbody_inertia():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    R = RigidBody('R', o, N, m, (Io, p))
    I_check = inertia(N, Ix - b ** 2 * m, Iy - a ** 2 * m,
                      Iz - m * (a ** 2 + b ** 2), m * a * b)
    assert isinstance(R.inertia, Inertia)
    assert R.inertia == (Io, p)
    assert R.central_inertia == I_check
    R.central_inertia = Io
    assert R.inertia == (Io, o)
    assert R.central_inertia == Io
    R.inertia = (Io, p)
    assert R.inertia == (Io, p)
    assert R.central_inertia == I_check
    # parse Inertia object
    R.inertia = Inertia(Io, o)
    assert R.inertia == (Io, o)


def test_parallel_axis():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    R = RigidBody('R', o, N, m, (Io, o))
    Ip = R.parallel_axis(p)
    Ip_expected = inertia(N, Ix + m * b**2, Iy + m * a**2,
                          Iz + m * (a**2 + b**2), ixy=-m * a * b)
    assert Ip == Ip_expected
    # Reference frame from which the parallel axis is viewed should not matter
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, 1)
    assert simplify(
        (R.parallel_axis(p, A) - Ip_expected).to_matrix(A)) == zeros(3, 3)


def test_deprecated_set_potential_energy():
    m, g, h = symbols('m g h')
    A = ReferenceFrame('A')
    P = Point('P')
    I = Dyadic(0)
    B = RigidBody('B', P, A, m, (I, P))
    with warns_deprecated_sympy():
        B.set_potential_energy(m*g*h)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_functions.py ===
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import express, matrix_to_vector, orthogonalize
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.testing.pytest import raises

N = CoordSys3D('N')
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
A = N.orient_new_axis('A', q1, N.k)  # type: ignore
B = A.orient_new_axis('B', q2, A.i)
C = B.orient_new_axis('C', q3, B.j)


def test_express():
    assert express(Vector.zero, N) == Vector.zero
    assert express(S.Zero, N) is S.Zero
    assert express(A.i, C) == cos(q3)*C.i + sin(q3)*C.k
    assert express(A.j, C) == sin(q2)*sin(q3)*C.i + cos(q2)*C.j - \
        sin(q2)*cos(q3)*C.k
    assert express(A.k, C) == -sin(q3)*cos(q2)*C.i + sin(q2)*C.j + \
        cos(q2)*cos(q3)*C.k
    assert express(A.i, N) == cos(q1)*N.i + sin(q1)*N.j
    assert express(A.j, N) == -sin(q1)*N.i + cos(q1)*N.j
    assert express(A.k, N) == N.k
    assert express(A.i, A) == A.i
    assert express(A.j, A) == A.j
    assert express(A.k, A) == A.k
    assert express(A.i, B) == B.i
    assert express(A.j, B) == cos(q2)*B.j - sin(q2)*B.k
    assert express(A.k, B) == sin(q2)*B.j + cos(q2)*B.k
    assert express(A.i, C) == cos(q3)*C.i + sin(q3)*C.k
    assert express(A.j, C) == sin(q2)*sin(q3)*C.i + cos(q2)*C.j - \
        sin(q2)*cos(q3)*C.k
    assert express(A.k, C) == -sin(q3)*cos(q2)*C.i + sin(q2)*C.j + \
        cos(q2)*cos(q3)*C.k
    # Check to make sure UnitVectors get converted properly
    assert express(N.i, N) == N.i
    assert express(N.j, N) == N.j
    assert express(N.k, N) == N.k
    assert express(N.i, A) == (cos(q1)*A.i - sin(q1)*A.j)
    assert express(N.j, A) == (sin(q1)*A.i + cos(q1)*A.j)
    assert express(N.k, A) == A.k
    assert express(N.i, B) == (cos(q1)*B.i - sin(q1)*cos(q2)*B.j +
            sin(q1)*sin(q2)*B.k)
    assert express(N.j, B) == (sin(q1)*B.i + cos(q1)*cos(q2)*B.j -
            sin(q2)*cos(q1)*B.k)
    assert express(N.k, B) == (sin(q2)*B.j + cos(q2)*B.k)
    assert express(N.i, C) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*C.i -
        sin(q1)*cos(q2)*C.j +
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*C.k)
    assert express(N.j, C) == (
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.i +
        cos(q1)*cos(q2)*C.j +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.k)
    assert express(N.k, C) == (-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k)

    assert express(A.i, N) == (cos(q1)*N.i + sin(q1)*N.j)
    assert express(A.j, N) == (-sin(q1)*N.i + cos(q1)*N.j)
    assert express(A.k, N) == N.k
    assert express(A.i, A) == A.i
    assert express(A.j, A) == A.j
    assert express(A.k, A) == A.k
    assert express(A.i, B) == B.i
    assert express(A.j, B) == (cos(q2)*B.j - sin(q2)*B.k)
    assert express(A.k, B) == (sin(q2)*B.j + cos(q2)*B.k)
    assert express(A.i, C) == (cos(q3)*C.i + sin(q3)*C.k)
    assert express(A.j, C) == (sin(q2)*sin(q3)*C.i + cos(q2)*C.j -
            sin(q2)*cos(q3)*C.k)
    assert express(A.k, C) == (-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k)

    assert express(B.i, N) == (cos(q1)*N.i + sin(q1)*N.j)
    assert express(B.j, N) == (-sin(q1)*cos(q2)*N.i +
            cos(q1)*cos(q2)*N.j + sin(q2)*N.k)
    assert express(B.k, N) == (sin(q1)*sin(q2)*N.i -
            sin(q2)*cos(q1)*N.j + cos(q2)*N.k)
    assert express(B.i, A) == A.i
    assert express(B.j, A) == (cos(q2)*A.j + sin(q2)*A.k)
    assert express(B.k, A) == (-sin(q2)*A.j + cos(q2)*A.k)
    assert express(B.i, B) == B.i
    assert express(B.j, B) == B.j
    assert express(B.k, B) == B.k
    assert express(B.i, C) == (cos(q3)*C.i + sin(q3)*C.k)
    assert express(B.j, C) == C.j
    assert express(B.k, C) == (-sin(q3)*C.i + cos(q3)*C.k)

    assert express(C.i, N) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*N.i +
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*N.j -
        sin(q3)*cos(q2)*N.k)
    assert express(C.j, N) == (
        -sin(q1)*cos(q2)*N.i + cos(q1)*cos(q2)*N.j + sin(q2)*N.k)
    assert express(C.k, N) == (
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*N.i +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*N.j +
        cos(q2)*cos(q3)*N.k)
    assert express(C.i, A) == (cos(q3)*A.i + sin(q2)*sin(q3)*A.j -
            sin(q3)*cos(q2)*A.k)
    assert express(C.j, A) == (cos(q2)*A.j + sin(q2)*A.k)
    assert express(C.k, A) == (sin(q3)*A.i - sin(q2)*cos(q3)*A.j +
            cos(q2)*cos(q3)*A.k)
    assert express(C.i, B) == (cos(q3)*B.i - sin(q3)*B.k)
    assert express(C.j, B) == B.j
    assert express(C.k, B) == (sin(q3)*B.i + cos(q3)*B.k)
    assert express(C.i, C) == C.i
    assert express(C.j, C) == C.j
    assert express(C.k, C) == C.k == (C.k)

    #  Check to make sure Vectors get converted back to UnitVectors
    assert N.i == express((cos(q1)*A.i - sin(q1)*A.j), N).simplify()
    assert N.j == express((sin(q1)*A.i + cos(q1)*A.j), N).simplify()
    assert N.i == express((cos(q1)*B.i - sin(q1)*cos(q2)*B.j +
            sin(q1)*sin(q2)*B.k), N).simplify()
    assert N.j == express((sin(q1)*B.i + cos(q1)*cos(q2)*B.j -
        sin(q2)*cos(q1)*B.k), N).simplify()
    assert N.k == express((sin(q2)*B.j + cos(q2)*B.k), N).simplify()


    assert A.i == express((cos(q1)*N.i + sin(q1)*N.j), A).simplify()
    assert A.j == express((-sin(q1)*N.i + cos(q1)*N.j), A).simplify()

    assert A.j == express((cos(q2)*B.j - sin(q2)*B.k), A).simplify()
    assert A.k == express((sin(q2)*B.j + cos(q2)*B.k), A).simplify()

    assert A.i == express((cos(q3)*C.i + sin(q3)*C.k), A).simplify()
    assert A.j == express((sin(q2)*sin(q3)*C.i + cos(q2)*C.j -
            sin(q2)*cos(q3)*C.k), A).simplify()

    assert A.k == express((-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k), A).simplify()
    assert B.i == express((cos(q1)*N.i + sin(q1)*N.j), B).simplify()
    assert B.j == express((-sin(q1)*cos(q2)*N.i +
            cos(q1)*cos(q2)*N.j + sin(q2)*N.k), B).simplify()

    assert B.k == express((sin(q1)*sin(q2)*N.i -
            sin(q2)*cos(q1)*N.j + cos(q2)*N.k), B).simplify()

    assert B.j == express((cos(q2)*A.j + sin(q2)*A.k), B).simplify()
    assert B.k == express((-sin(q2)*A.j + cos(q2)*A.k), B).simplify()
    assert B.i == express((cos(q3)*C.i + sin(q3)*C.k), B).simplify()
    assert B.k == express((-sin(q3)*C.i + cos(q3)*C.k), B).simplify()
    assert C.i == express((cos(q3)*A.i + sin(q2)*sin(q3)*A.j -
            sin(q3)*cos(q2)*A.k), C).simplify()
    assert C.j == express((cos(q2)*A.j + sin(q2)*A.k), C).simplify()
    assert C.k == express((sin(q3)*A.i - sin(q2)*cos(q3)*A.j +
            cos(q2)*cos(q3)*A.k), C).simplify()
    assert C.i == express((cos(q3)*B.i - sin(q3)*B.k), C).simplify()
    assert C.k == express((sin(q3)*B.i + cos(q3)*B.k), C).simplify()


def test_matrix_to_vector():
    m = Matrix([[1], [2], [3]])
    assert matrix_to_vector(m, C) == C.i + 2*C.j + 3*C.k
    m = Matrix([[0], [0], [0]])
    assert matrix_to_vector(m, N) == matrix_to_vector(m, C) == \
           Vector.zero
    m = Matrix([[q1], [q2], [q3]])
    assert matrix_to_vector(m, N) == q1*N.i + q2*N.j + q3*N.k


def test_orthogonalize():
    C = CoordSys3D('C')
    a, b = symbols('a b', integer=True)
    i, j, k = C.base_vectors()
    v1 = i + 2*j
    v2 = 2*i + 3*j
    v3 = 3*i + 5*j
    v4 = 3*i + j
    v5 = 2*i + 2*j
    v6 = a*i + b*j
    v7 = 4*a*i + 4*b*j
    assert orthogonalize(v1, v2) == [C.i + 2*C.j, C.i*Rational(2, 5) + -C.j/5]
    # from wikipedia
    assert orthogonalize(v4, v5, orthonormal=True) == \
        [(3*sqrt(10))*C.i/10 + (sqrt(10))*C.j/10, (-sqrt(10))*C.i/10 + (3*sqrt(10))*C.j/10]
    raises(ValueError, lambda: orthogonalize(v1, v2, v3))
    raises(ValueError, lambda: orthogonalize(v6, v7))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\datetimes\test_reductions.py ===
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import NaT
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestReductions:
    @pytest.fixture(params=["s", "ms", "us", "ns"])
    def unit(self, request):
        return request.param

    @pytest.fixture
    def arr1d(self, tz_naive_fixture):
        """Fixture returning DatetimeArray with parametrized timezones"""
        tz = tz_naive_fixture
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-03",
                "2000-01-03",
                "NaT",
                "2000-01-02",
                "2000-01-05",
                "2000-01-04",
            ],
            dtype=dtype,
        )
        return arr

    def test_min_max(self, arr1d, unit):
        arr = arr1d
        arr = arr.as_unit(unit)
        tz = arr.tz

        result = arr.min()
        expected = pd.Timestamp("2000-01-02", tz=tz).as_unit(unit)
        assert result == expected
        assert result.unit == expected.unit

        result = arr.max()
        expected = pd.Timestamp("2000-01-05", tz=tz).as_unit(unit)
        assert result == expected
        assert result.unit == expected.unit

        result = arr.min(skipna=False)
        assert result is NaT

        result = arr.max(skipna=False)
        assert result is NaT

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_min_max_empty(self, skipna, tz):
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        arr = DatetimeArray._from_sequence([], dtype=dtype)
        result = arr.min(skipna=skipna)
        assert result is NaT

        result = arr.max(skipna=skipna)
        assert result is NaT

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_median_empty(self, skipna, tz):
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        arr = DatetimeArray._from_sequence([], dtype=dtype)
        result = arr.median(skipna=skipna)
        assert result is NaT

        arr = arr.reshape(0, 3)
        result = arr.median(axis=0, skipna=skipna)
        expected = type(arr)._from_sequence([NaT, NaT, NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected)

        result = arr.median(axis=1, skipna=skipna)
        expected = type(arr)._from_sequence([], dtype=arr.dtype)
        tm.assert_equal(result, expected)

    def test_median(self, arr1d):
        arr = arr1d

        result = arr.median()
        assert result == arr[0]
        result = arr.median(skipna=False)
        assert result is NaT

        result = arr.dropna().median(skipna=False)
        assert result == arr[0]

        result = arr.median(axis=0)
        assert result == arr[0]

    def test_median_axis(self, arr1d):
        arr = arr1d
        assert arr.median(axis=0) == arr.median()
        assert arr.median(axis=0, skipna=False) is NaT

        msg = r"abs\(axis\) must be less than ndim"
        with pytest.raises(ValueError, match=msg):
            arr.median(axis=1)

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test_median_2d(self, arr1d):
        arr = arr1d.reshape(1, -1)

        # axis = None
        assert arr.median() == arr1d.median()
        assert arr.median(skipna=False) is NaT

        # axis = 0
        result = arr.median(axis=0)
        expected = arr1d
        tm.assert_equal(result, expected)

        # Since column 3 is all-NaT, we get NaT there with or without skipna
        result = arr.median(axis=0, skipna=False)
        expected = arr1d
        tm.assert_equal(result, expected)

        # axis = 1
        result = arr.median(axis=1)
        expected = type(arr)._from_sequence([arr1d.median()], dtype=arr.dtype)
        tm.assert_equal(result, expected)

        result = arr.median(axis=1, skipna=False)
        expected = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected)

    def test_mean(self, arr1d):
        arr = arr1d

        # manually verified result
        expected = arr[0] + 0.4 * pd.Timedelta(days=1)

        result = arr.mean()
        assert result == expected
        result = arr.mean(skipna=False)
        assert result is NaT

        result = arr.dropna().mean(skipna=False)
        assert result == expected

        result = arr.mean(axis=0)
        assert result == expected

    def test_mean_2d(self):
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        dta = dti._data.reshape(3, 2)

        result = dta.mean(axis=0)
        expected = dta[1]
        tm.assert_datetime_array_equal(result, expected)

        result = dta.mean(axis=1)
        expected = dta[:, 0] + pd.Timedelta(hours=12)
        tm.assert_datetime_array_equal(result, expected)

        result = dta.mean(axis=None)
        expected = dti.mean()
        assert result == expected

    @pytest.mark.parametrize("skipna", [True, False])
    def test_mean_empty(self, arr1d, skipna):
        arr = arr1d[:0]

        assert arr.mean(skipna=skipna) is NaT

        arr2d = arr.reshape(0, 3)
        result = arr2d.mean(axis=0, skipna=skipna)
        expected = DatetimeArray._from_sequence([NaT, NaT, NaT], dtype=arr.dtype)
        tm.assert_datetime_array_equal(result, expected)

        result = arr2d.mean(axis=1, skipna=skipna)
        expected = arr  # i.e. 1D, empty
        tm.assert_datetime_array_equal(result, expected)

        result = arr2d.mean(axis=None, skipna=skipna)
        assert result is NaT

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_quoting.py ===
"""
Tests that quoting specifications are properly handled
during parsing for all of the parsers defined in parsers.py
"""

import csv
from io import StringIO

import pytest

from pandas.compat import PY311
from pandas.errors import ParserError

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"quotechar": "foo"}, '"quotechar" must be a(n)? 1-character string'),
        (
            {"quotechar": None, "quoting": csv.QUOTE_MINIMAL},
            "quotechar must be set if quoting enabled",
        ),
        ({"quotechar": 2}, '"quotechar" must be string( or None)?, not int'),
    ],
)
@skip_pyarrow  # ParserError: CSV parse error: Empty CSV file or block
def test_bad_quote_char(all_parsers, kwargs, msg):
    data = "1,2,3"
    parser = all_parsers

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


@pytest.mark.parametrize(
    "quoting,msg",
    [
        ("foo", '"quoting" must be an integer|Argument'),
        (10, 'bad "quoting" value'),  # quoting must be in the range [0, 3]
    ],
)
@xfail_pyarrow  # ValueError: The 'quoting' option is not supported
def test_bad_quoting(all_parsers, quoting, msg):
    data = "1,2,3"
    parser = all_parsers

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), quoting=quoting)


def test_quote_char_basic(all_parsers):
    parser = all_parsers
    data = 'a,b,c\n1,2,"cat"'
    expected = DataFrame([[1, 2, "cat"]], columns=["a", "b", "c"])

    result = parser.read_csv(StringIO(data), quotechar='"')
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quote_char", ["~", "*", "%", "$", "@", "P"])
def test_quote_char_various(all_parsers, quote_char):
    parser = all_parsers
    expected = DataFrame([[1, 2, "cat"]], columns=["a", "b", "c"])

    data = 'a,b,c\n1,2,"cat"'
    new_data = data.replace('"', quote_char)

    result = parser.read_csv(StringIO(new_data), quotechar=quote_char)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: The 'quoting' option is not supported
@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
@pytest.mark.parametrize("quote_char", ["", None])
def test_null_quote_char(all_parsers, quoting, quote_char):
    kwargs = {"quotechar": quote_char, "quoting": quoting}
    data = "a,b,c\n1,2,3"
    parser = all_parsers

    if quoting != csv.QUOTE_NONE:
        # Sanity checking.
        msg = (
            '"quotechar" must be a 1-character string'
            if PY311 and all_parsers.engine == "python" and quote_char == ""
            else "quotechar must be set if quoting enabled"
        )

        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    elif not (PY311 and all_parsers.engine == "python"):
        # Python 3.11+ doesn't support null/blank quote chars in their csv parsers
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,exp_data",
    [
        ({}, [[1, 2, "foo"]]),  # Test default.
        # QUOTE_MINIMAL only applies to CSV writing, so no effect on reading.
        ({"quotechar": '"', "quoting": csv.QUOTE_MINIMAL}, [[1, 2, "foo"]]),
        # QUOTE_MINIMAL only applies to CSV writing, so no effect on reading.
        ({"quotechar": '"', "quoting": csv.QUOTE_ALL}, [[1, 2, "foo"]]),
        # QUOTE_NONE tells the reader to do no special handling
        # of quote characters and leave them alone.
        ({"quotechar": '"', "quoting": csv.QUOTE_NONE}, [[1, 2, '"foo"']]),
        # QUOTE_NONNUMERIC tells the reader to cast
        # all non-quoted fields to float
        ({"quotechar": '"', "quoting": csv.QUOTE_NONNUMERIC}, [[1.0, 2.0, "foo"]]),
    ],
)
@xfail_pyarrow  # ValueError: The 'quoting' option is not supported
def test_quoting_various(all_parsers, kwargs, exp_data):
    data = '1,2,"foo"'
    parser = all_parsers
    columns = ["a", "b", "c"]

    result = parser.read_csv(StringIO(data), names=columns, **kwargs)
    expected = DataFrame(exp_data, columns=columns)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "doublequote,exp_data", [(True, [[3, '4 " 5']]), (False, [[3, '4 " 5"']])]
)
def test_double_quote(all_parsers, doublequote, exp_data, request):
    parser = all_parsers
    data = 'a,b\n3,"4 "" 5"'

    if parser.engine == "pyarrow" and not doublequote:
        mark = pytest.mark.xfail(reason="Mismatched result")
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), quotechar='"', doublequote=doublequote)
    expected = DataFrame(exp_data, columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quotechar", ['"', "\u0001"])
def test_quotechar_unicode(all_parsers, quotechar):
    # see gh-14477
    data = "a\n1"
    parser = all_parsers
    expected = DataFrame({"a": [1]})

    result = parser.read_csv(StringIO(data), quotechar=quotechar)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("balanced", [True, False])
def test_unbalanced_quoting(all_parsers, balanced, request):
    # see gh-22789.
    parser = all_parsers
    data = 'a,b,c\n1,2,"3'

    if parser.engine == "pyarrow" and not balanced:
        mark = pytest.mark.xfail(reason="Mismatched result")
        request.applymarker(mark)

    if balanced:
        # Re-balance the quoting and read in without errors.
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        result = parser.read_csv(StringIO(data + '"'))
        tm.assert_frame_equal(result, expected)
    else:
        msg = (
            "EOF inside string starting at row 1"
            if parser.engine == "c"
            else "unexpected end of data"
        )

        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_rationaltools.py ===
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify

from sympy.integrals.rationaltools import ratint, ratint_logpart, log_to_atan

from sympy.abc import a, b, x, t

half = S.Half


def test_ratint():
    assert ratint(S.Zero, x) == 0
    assert ratint(S(7), x) == 7*x

    assert ratint(x, x) == x**2/2
    assert ratint(2*x, x) == x**2
    assert ratint(-2*x, x) == -x**2

    assert ratint(8*x**7 + 2*x + 1, x) == x**8 + x**2 + x

    f = S.One
    g = x + 1

    assert ratint(f / g, x) == log(x + 1)
    assert ratint((f, g), x) == log(x + 1)

    f = x**3 - x
    g = x - 1

    assert ratint(f/g, x) == x**3/3 + x**2/2

    f = x
    g = (x - a)*(x + a)

    assert ratint(f/g, x) == log(x**2 - a**2)/2

    f = S.One
    g = x**2 + 1

    assert ratint(f/g, x, real=None) == atan(x)
    assert ratint(f/g, x, real=True) == atan(x)

    assert ratint(f/g, x, real=False) == I*log(x + I)/2 - I*log(x - I)/2

    f = S(36)
    g = x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2

    assert ratint(f/g, x) == \
        -4*log(x + 1) + 4*log(x - 2) + (12*x + 6)/(x**2 - 1)

    f = x**4 - 3*x**2 + 6
    g = x**6 - 5*x**4 + 5*x**2 + 4

    assert ratint(f/g, x) == \
        atan(x) + atan(x**3) + atan(x/2 - Rational(3, 2)*x**3 + S.Half*x**5)

    f = x**7 - 24*x**4 - 4*x**2 + 8*x - 8
    g = x**8 + 6*x**6 + 12*x**4 + 8*x**2

    assert ratint(f/g, x) == \
        (4 + 6*x + 8*x**2 + 3*x**3)/(4*x + 4*x**3 + x**5) + log(x)

    assert ratint((x**3*f)/(x*g), x) == \
        -(12 - 16*x + 6*x**2 - 14*x**3)/(4 + 4*x**2 + x**4) - \
        5*sqrt(2)*atan(x*sqrt(2)/2) + S.Half*x**2 - 3*log(2 + x**2)

    f = x**5 - x**4 + 4*x**3 + x**2 - x + 5
    g = x**4 - 2*x**3 + 5*x**2 - 4*x + 4

    assert ratint(f/g, x) == \
        x + S.Half*x**2 + S.Half*log(2 - x + x**2) + (9 - 4*x)/(7*x**2 - 7*x + 14) + \
        13*sqrt(7)*atan(Rational(-1, 7)*sqrt(7) + 2*x*sqrt(7)/7)/49

    assert ratint(1/(x**2 + x + 1), x) == \
        2*sqrt(3)*atan(sqrt(3)/3 + 2*x*sqrt(3)/3)/3

    assert ratint(1/(x**3 + 1), x) == \
        -log(1 - x + x**2)/6 + log(1 + x)/3 + sqrt(3)*atan(-sqrt(3)
             /3 + 2*x*sqrt(3)/3)/3

    assert ratint(1/(x**2 + x + 1), x, real=False) == \
        -I*3**half*log(half + x - half*I*3**half)/3 + \
        I*3**half*log(half + x + half*I*3**half)/3

    assert ratint(1/(x**3 + 1), x, real=False) == log(1 + x)/3 + \
        (Rational(-1, 6) + I*3**half/6)*log(-half + x + I*3**half/2) + \
        (Rational(-1, 6) - I*3**half/6)*log(-half + x - I*3**half/2)

    # issue 4991
    assert ratint(1/(x*(a + b*x)**3), x) == \
        (3*a + 2*b*x)/(2*a**4 + 4*a**3*b*x + 2*a**2*b**2*x**2) + (
            log(x) - log(a/b + x))/a**3

    assert ratint(x/(1 - x**2), x) == -log(x**2 - 1)/2
    assert ratint(-x/(1 - x**2), x) == log(x**2 - 1)/2

    assert ratint((x/4 - 4/(1 - x)).diff(x), x) == x/4 + 4/(x - 1)

    ans = atan(x)
    assert ratint(1/(x**2 + 1), x, symbol=x) == ans
    assert ratint(1/(x**2 + 1), x, symbol='x') == ans
    assert ratint(1/(x**2 + 1), x, symbol=a) == ans
    # this asserts that as_dummy must return a unique symbol
    # even if the symbol is already a Dummy
    d = Dummy()
    assert ratint(1/(d**2 + 1), d, symbol=d) == atan(d)


def test_ratint_logpart():
    assert ratint_logpart(x, x**2 - 9, x, t) == \
        [(Poly(x**2 - 9, x), Poly(-2*t + 1, t))]
    assert ratint_logpart(x**2, x**3 - 5, x, t) == \
        [(Poly(x**3 - 5, x), Poly(-3*t + 1, t))]


def test_issue_5414():
    assert ratint(1/(x**2 + 16), x) == atan(x/4)/4


def test_issue_5249():
    assert ratint(
        1/(x**2 + a**2), x) == (-I*log(-I*a + x)/2 + I*log(I*a + x)/2)/a


def test_issue_5817():
    a, b, c = symbols('a,b,c', positive=True)

    assert simplify(ratint(a/(b*c*x**2 + a**2 + b*a), x)) == \
        sqrt(a)*atan(sqrt(
            b)*sqrt(c)*x/(sqrt(a)*sqrt(a + b)))/(sqrt(b)*sqrt(c)*sqrt(a + b))


def test_issue_5981():
    u = symbols('u')
    assert integrate(1/(u**2 + 1)) == atan(u)

def test_issue_10488():
    a,b,c,x = symbols('a b c x', positive=True)
    assert integrate(x/(a*x+b),x) == x/a - b*log(a*x + b)/a**2


def test_issues_8246_12050_13501_14080():
    a = symbols('a', nonzero=True)
    assert integrate(a/(x**2 + a**2), x) == atan(x/a)
    assert integrate(1/(x**2 + a**2), x) == atan(x/a)/a
    assert integrate(1/(1 + a**2*x**2), x) == atan(a*x)/a


def test_issue_6308():
    k, a0 = symbols('k a0', real=True)
    assert integrate((x**2 + 1 - k**2)/(x**2 + 1 + a0**2), x) == \
        x - (a0**2 + k**2)*atan(x/sqrt(a0**2 + 1))/sqrt(a0**2 + 1)


def test_issue_5907():
    a = symbols('a', nonzero=True)
    assert integrate(1/(x**2 + a**2)**2, x) == \
         x/(2*a**4 + 2*a**2*x**2) + atan(x/a)/(2*a**3)


def test_log_to_atan():
    f, g = (Poly(x + S.Half, x, domain='QQ'), Poly(sqrt(3)/2, x, domain='EX'))
    fg_ans = 2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)
    assert log_to_atan(f, g) == fg_ans
    assert log_to_atan(g, f) == -fg_ans


def test_issue_25896():
    # for both tests, C = 0 in log_to_real
    # but this only has a log result
    e = (2*x + 1)/(x**2 + x + 1) + 1/x
    assert ratint(e, x) == log(x**3 + x**2 + x)
    # while this has more
    assert ratint((4*x + 7)/(x**2 + 4*x + 6) + 2/x, x) == (
        2*log(x) + 2*log(x**2 + 4*x + 6) - sqrt(2)*atan(
        sqrt(2)*x/2 + sqrt(2))/2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_cg.py ===
from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.cg import Wigner3j, Wigner6j, Wigner9j, CG, cg_simp
from sympy.functions.special.tensor_functions import KroneckerDelta


def test_cg_simp_add():
    j, m1, m1p, m2, m2p = symbols('j m1 m1p m2 m2p')
    # Test Varshalovich 8.7.1 Eq 1
    a = CG(S.Half, S.Half, 0, 0, S.Half, S.Half)
    b = CG(S.Half, Rational(-1, 2), 0, 0, S.Half, Rational(-1, 2))
    c = CG(1, 1, 0, 0, 1, 1)
    d = CG(1, 0, 0, 0, 1, 0)
    e = CG(1, -1, 0, 0, 1, -1)
    assert cg_simp(a + b) == 2
    assert cg_simp(c + d + e) == 3
    assert cg_simp(a + b + c + d + e) == 5
    assert cg_simp(a + b + c) == 2 + c
    assert cg_simp(2*a + b) == 2 + a
    assert cg_simp(2*c + d + e) == 3 + c
    assert cg_simp(5*a + 5*b) == 10
    assert cg_simp(5*c + 5*d + 5*e) == 15
    assert cg_simp(-a - b) == -2
    assert cg_simp(-c - d - e) == -3
    assert cg_simp(-6*a - 6*b) == -12
    assert cg_simp(-4*c - 4*d - 4*e) == -12
    a = CG(S.Half, S.Half, j, 0, S.Half, S.Half)
    b = CG(S.Half, Rational(-1, 2), j, 0, S.Half, Rational(-1, 2))
    c = CG(1, 1, j, 0, 1, 1)
    d = CG(1, 0, j, 0, 1, 0)
    e = CG(1, -1, j, 0, 1, -1)
    assert cg_simp(a + b) == 2*KroneckerDelta(j, 0)
    assert cg_simp(c + d + e) == 3*KroneckerDelta(j, 0)
    assert cg_simp(a + b + c + d + e) == 5*KroneckerDelta(j, 0)
    assert cg_simp(a + b + c) == 2*KroneckerDelta(j, 0) + c
    assert cg_simp(2*a + b) == 2*KroneckerDelta(j, 0) + a
    assert cg_simp(2*c + d + e) == 3*KroneckerDelta(j, 0) + c
    assert cg_simp(5*a + 5*b) == 10*KroneckerDelta(j, 0)
    assert cg_simp(5*c + 5*d + 5*e) == 15*KroneckerDelta(j, 0)
    assert cg_simp(-a - b) == -2*KroneckerDelta(j, 0)
    assert cg_simp(-c - d - e) == -3*KroneckerDelta(j, 0)
    assert cg_simp(-6*a - 6*b) == -12*KroneckerDelta(j, 0)
    assert cg_simp(-4*c - 4*d - 4*e) == -12*KroneckerDelta(j, 0)
    # Test Varshalovich 8.7.1 Eq 2
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0)
    b = CG(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0)
    c = CG(1, 1, 1, -1, 0, 0)
    d = CG(1, 0, 1, 0, 0, 0)
    e = CG(1, -1, 1, 1, 0, 0)
    assert cg_simp(a - b) == sqrt(2)
    assert cg_simp(c - d + e) == sqrt(3)
    assert cg_simp(a - b + c - d + e) == sqrt(2) + sqrt(3)
    assert cg_simp(a - b + c) == sqrt(2) + c
    assert cg_simp(2*a - b) == sqrt(2) + a
    assert cg_simp(2*c - d + e) == sqrt(3) + c
    assert cg_simp(5*a - 5*b) == 5*sqrt(2)
    assert cg_simp(5*c - 5*d + 5*e) == 5*sqrt(3)
    assert cg_simp(-a + b) == -sqrt(2)
    assert cg_simp(-c + d - e) == -sqrt(3)
    assert cg_simp(-6*a + 6*b) == -6*sqrt(2)
    assert cg_simp(-4*c + 4*d - 4*e) == -4*sqrt(3)
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), j, 0)
    b = CG(S.Half, Rational(-1, 2), S.Half, S.Half, j, 0)
    c = CG(1, 1, 1, -1, j, 0)
    d = CG(1, 0, 1, 0, j, 0)
    e = CG(1, -1, 1, 1, j, 0)
    assert cg_simp(a - b) == sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(c - d + e) == sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(a - b + c - d + e) == sqrt(
        2)*KroneckerDelta(j, 0) + sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(a - b + c) == sqrt(2)*KroneckerDelta(j, 0) + c
    assert cg_simp(2*a - b) == sqrt(2)*KroneckerDelta(j, 0) + a
    assert cg_simp(2*c - d + e) == sqrt(3)*KroneckerDelta(j, 0) + c
    assert cg_simp(5*a - 5*b) == 5*sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(5*c - 5*d + 5*e) == 5*sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(-a + b) == -sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(-c + d - e) == -sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(-6*a + 6*b) == -6*sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(-4*c + 4*d - 4*e) == -4*sqrt(3)*KroneckerDelta(j, 0)
    # Test Varshalovich 8.7.2 Eq 9
    # alpha=alphap,beta=betap case
    # numerical
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 1, 0)**2
    b = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0)**2
    c = CG(1, 0, 1, 1, 1, 1)**2
    d = CG(1, 0, 1, 1, 2, 1)**2
    assert cg_simp(a + b) == 1
    assert cg_simp(c + d) == 1
    assert cg_simp(a + b + c + d) == 2
    assert cg_simp(4*a + 4*b) == 4
    assert cg_simp(4*c + 4*d) == 4
    assert cg_simp(5*a + 3*b) == 3 + 2*a
    assert cg_simp(5*c + 3*d) == 3 + 2*c
    assert cg_simp(-a - b) == -1
    assert cg_simp(-c - d) == -1
    # symbolic
    a = CG(S.Half, m1, S.Half, m2, 1, 1)**2
    b = CG(S.Half, m1, S.Half, m2, 1, 0)**2
    c = CG(S.Half, m1, S.Half, m2, 1, -1)**2
    d = CG(S.Half, m1, S.Half, m2, 0, 0)**2
    assert cg_simp(a + b + c + d) == 1
    assert cg_simp(4*a + 4*b + 4*c + 4*d) == 4
    assert cg_simp(3*a + 5*b + 3*c + 4*d) == 3 + 2*b + d
    assert cg_simp(-a - b - c - d) == -1
    a = CG(1, m1, 1, m2, 2, 2)**2
    b = CG(1, m1, 1, m2, 2, 1)**2
    c = CG(1, m1, 1, m2, 2, 0)**2
    d = CG(1, m1, 1, m2, 2, -1)**2
    e = CG(1, m1, 1, m2, 2, -2)**2
    f = CG(1, m1, 1, m2, 1, 1)**2
    g = CG(1, m1, 1, m2, 1, 0)**2
    h = CG(1, m1, 1, m2, 1, -1)**2
    i = CG(1, m1, 1, m2, 0, 0)**2
    assert cg_simp(a + b + c + d + e + f + g + h + i) == 1
    assert cg_simp(4*(a + b + c + d + e + f + g + h + i)) == 4
    assert cg_simp(a + b + 2*c + d + 4*e + f + g + h + i) == 1 + c + 3*e
    assert cg_simp(-a - b - c - d - e - f - g - h - i) == -1
    # alpha!=alphap or beta!=betap case
    # numerical
    a = CG(S.Half, S(
        1)/2, S.Half, Rational(-1, 2), 1, 0)*CG(S.Half, Rational(-1, 2), S.Half, S.Half, 1, 0)
    b = CG(S.Half, S(
        1)/2, S.Half, Rational(-1, 2), 0, 0)*CG(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0)
    c = CG(1, 1, 1, 0, 2, 1)*CG(1, 0, 1, 1, 2, 1)
    d = CG(1, 1, 1, 0, 1, 1)*CG(1, 0, 1, 1, 1, 1)
    assert cg_simp(a + b) == 0
    assert cg_simp(c + d) == 0
    # symbolic
    a = CG(S.Half, m1, S.Half, m2, 1, 1)*CG(S.Half, m1p, S.Half, m2p, 1, 1)
    b = CG(S.Half, m1, S.Half, m2, 1, 0)*CG(S.Half, m1p, S.Half, m2p, 1, 0)
    c = CG(S.Half, m1, S.Half, m2, 1, -1)*CG(S.Half, m1p, S.Half, m2p, 1, -1)
    d = CG(S.Half, m1, S.Half, m2, 0, 0)*CG(S.Half, m1p, S.Half, m2p, 0, 0)
    assert cg_simp(a + b + c + d) == KroneckerDelta(m1, m1p)*KroneckerDelta(m2, m2p)
    a = CG(1, m1, 1, m2, 2, 2)*CG(1, m1p, 1, m2p, 2, 2)
    b = CG(1, m1, 1, m2, 2, 1)*CG(1, m1p, 1, m2p, 2, 1)
    c = CG(1, m1, 1, m2, 2, 0)*CG(1, m1p, 1, m2p, 2, 0)
    d = CG(1, m1, 1, m2, 2, -1)*CG(1, m1p, 1, m2p, 2, -1)
    e = CG(1, m1, 1, m2, 2, -2)*CG(1, m1p, 1, m2p, 2, -2)
    f = CG(1, m1, 1, m2, 1, 1)*CG(1, m1p, 1, m2p, 1, 1)
    g = CG(1, m1, 1, m2, 1, 0)*CG(1, m1p, 1, m2p, 1, 0)
    h = CG(1, m1, 1, m2, 1, -1)*CG(1, m1p, 1, m2p, 1, -1)
    i = CG(1, m1, 1, m2, 0, 0)*CG(1, m1p, 1, m2p, 0, 0)
    assert cg_simp(
        a + b + c + d + e + f + g + h + i) == KroneckerDelta(m1, m1p)*KroneckerDelta(m2, m2p)


def test_cg_simp_sum():
    x, a, b, c, cp, alpha, beta, gamma, gammap = symbols(
        'x a b c cp alpha beta gamma gammap')
    # Varshalovich 8.7.1 Eq 1
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)
                   )) == x*(2*a + 1)*KroneckerDelta(b, 0)
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)) + CG(1, 0, 1, 0, 1, 0)) == x*(2*a + 1)*KroneckerDelta(b, 0) + CG(1, 0, 1, 0, 1, 0)
    assert cg_simp(2 * Sum(CG(1, alpha, 0, 0, 1, alpha), (alpha, -1, 1))) == 6
    # Varshalovich 8.7.1 Eq 2
    assert cg_simp(x*Sum((-1)**(a - alpha) * CG(a, alpha, a, -alpha, c,
                   0), (alpha, -a, a))) == x*sqrt(2*a + 1)*KroneckerDelta(c, 0)
    assert cg_simp(3*Sum((-1)**(2 - alpha) * CG(
        2, alpha, 2, -alpha, 0, 0), (alpha, -2, 2))) == 3*sqrt(5)
    # Varshalovich 8.7.2 Eq 4
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, cp, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, c, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(gamma, gammap)
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, cp, gamma), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp)
    assert cg_simp(Sum(CG(
        a, alpha, b, beta, c, gamma)**2, (alpha, -a, a), (beta, -b, b))) == 1
    assert cg_simp(Sum(CG(2, alpha, 1, beta, 2, gamma)*CG(2, alpha, 1, beta, 2, gammap), (alpha, -2, 2), (beta, -1, 1))) == KroneckerDelta(gamma, gammap)


def test_doit():
    assert Wigner3j(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0).doit() == -sqrt(2)/2
    assert Wigner3j(1/2,1/2,1/2,1/2,1/2,1/2).doit() == 0
    assert Wigner3j(9/2,9/2,9/2,9/2,9/2,9/2).doit() ==  0
    assert Wigner6j(1, 2, 3, 2, 1, 2).doit() == sqrt(21)/105
    assert Wigner6j(3, 1, 2, 2, 2, 1).doit() == sqrt(21) / 105
    assert Wigner9j(
        2, 1, 1, Rational(3, 2), S.Half, 1, S.Half, S.Half, 0).doit() == sqrt(2)/12
    assert CG(S.Half, S.Half, S.Half, Rational(-1, 2), 1, 0).doit() == sqrt(2)/2
    # J minus M is not integer
    assert Wigner3j(1, -1, S.Half, S.Half, 1, S.Half).doit() == 0
    assert CG(4, -1, S.Half, S.Half, 4, Rational(-1, 2)).doit() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_mangle_dupes.py ===
"""
Tests that duplicate columns are handled appropriately when parsed by the
CSV engine. In general, the expected result is that they are either thoroughly
de-duplicated (if mangling requested) or ignored otherwise.
"""
from io import StringIO

import pytest

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_basic(all_parsers):
    parser = all_parsers

    data = "a,a,b,b,b\n1,2,3,4,5"
    result = parser.read_csv(StringIO(data), sep=",")

    expected = DataFrame([[1, 2, 3, 4, 5]], columns=["a", "a.1", "b", "b.1", "b.2"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_basic_names(all_parsers):
    # See gh-7160
    parser = all_parsers

    data = "a,b,a\n0,1,2\n3,4,5"
    expected = DataFrame([[0, 1, 2], [3, 4, 5]], columns=["a", "b", "a.1"])

    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_basic_names_raise(all_parsers):
    # See gh-7160
    parser = all_parsers

    data = "0,1,2\n3,4,5"
    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=["a", "b", "a"])


@xfail_pyarrow  # ValueError: Found non-unique column index
@pytest.mark.parametrize(
    "data,expected",
    [
        ("a,a,a.1\n1,2,3", DataFrame([[1, 2, 3]], columns=["a", "a.2", "a.1"])),
        (
            "a,a,a.1,a.1.1,a.1.1.1,a.1.1.1.1\n1,2,3,4,5,6",
            DataFrame(
                [[1, 2, 3, 4, 5, 6]],
                columns=["a", "a.2", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1"],
            ),
        ),
        (
            "a,a,a.3,a.1,a.2,a,a\n1,2,3,4,5,6,7",
            DataFrame(
                [[1, 2, 3, 4, 5, 6, 7]],
                columns=["a", "a.4", "a.3", "a.1", "a.2", "a.5", "a.6"],
            ),
        ),
    ],
)
def test_thorough_mangle_columns(all_parsers, data, expected):
    # see gh-17060
    parser = all_parsers

    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,names,expected",
    [
        (
            "a,b,b\n1,2,3",
            ["a.1", "a.1", "a.1.1"],
            DataFrame(
                [["a", "b", "b"], ["1", "2", "3"]], columns=["a.1", "a.1.1", "a.1.1.1"]
            ),
        ),
        (
            "a,b,c,d,e,f\n1,2,3,4,5,6",
            ["a", "a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1"],
            DataFrame(
                [["a", "b", "c", "d", "e", "f"], ["1", "2", "3", "4", "5", "6"]],
                columns=["a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1", "a.1.1.1.1.1"],
            ),
        ),
        (
            "a,b,c,d,e,f,g\n1,2,3,4,5,6,7",
            ["a", "a", "a.3", "a.1", "a.2", "a", "a"],
            DataFrame(
                [
                    ["a", "b", "c", "d", "e", "f", "g"],
                    ["1", "2", "3", "4", "5", "6", "7"],
                ],
                columns=["a", "a.1", "a.3", "a.1.1", "a.2", "a.2.1", "a.3.1"],
            ),
        ),
    ],
)
def test_thorough_mangle_names(all_parsers, data, names, expected):
    # see gh-17095
    parser = all_parsers

    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=names)


@xfail_pyarrow  # AssertionError: DataFrame.columns are different
def test_mangled_unnamed_placeholders(all_parsers):
    # xref gh-13017
    orig_key = "0"
    parser = all_parsers

    orig_value = [1, 2, 3]
    df = DataFrame({orig_key: orig_value})

    # This test recursively updates `df`.
    for i in range(3):
        expected = DataFrame(columns=Index([], dtype="str"))

        for j in range(i + 1):
            col_name = "Unnamed: 0" + f".{1*j}" * min(j, 1)
            expected.insert(loc=0, column=col_name, value=[0, 1, 2])

        expected[orig_key] = orig_value
        df = parser.read_csv(StringIO(df.to_csv()))

        tm.assert_frame_equal(df, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_mangle_dupe_cols_already_exists(all_parsers):
    # GH#14704
    parser = all_parsers

    data = "a,a,a.1,a,a.3,a.1,a.1.1\n1,2,3,4,5,6,7"
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6, 7]],
        columns=["a", "a.2", "a.1", "a.4", "a.3", "a.1.2", "a.1.1"],
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_mangle_dupe_cols_already_exists_unnamed_col(all_parsers):
    # GH#14704
    parser = all_parsers

    data = ",Unnamed: 0,,Unnamed: 2\n1,2,3,4"
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [[1, 2, 3, 4]],
        columns=["Unnamed: 0.1", "Unnamed: 0", "Unnamed: 2.1", "Unnamed: 2"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("usecol, engine", [([0, 1, 1], "python"), ([0, 1, 1], "c")])
def test_mangle_cols_names(all_parsers, usecol, engine):
    # GH 11823
    parser = all_parsers
    data = "1,2,3"
    names = ["A", "A", "B"]
    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=names, usecols=usecol, engine=engine)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_rolling_quantile.py ===
from functools import partial

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    concat,
    isna,
    notna,
)
import pandas._testing as tm

from pandas.tseries import offsets


def scoreatpercentile(a, per):
    values = np.sort(a, axis=0)

    idx = int(per / 1.0 * (values.shape[0] - 1))

    if idx == values.shape[0] - 1:
        retval = values[-1]

    else:
        qlow = idx / (values.shape[0] - 1)
        qhig = (idx + 1) / (values.shape[0] - 1)
        vlow = values[idx]
        vhig = values[idx + 1]
        retval = vlow + (vhig - vlow) * (per - qlow) / (qhig - qlow)

    return retval


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_series(series, q, step):
    compare_func = partial(scoreatpercentile, per=q)
    result = series.rolling(50, step=step).quantile(q)
    assert isinstance(result, Series)
    end = range(0, len(series), step or 1)[-1] + 1
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[end - 50 : end]))


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_frame(raw, frame, q, step):
    compare_func = partial(scoreatpercentile, per=q)
    result = frame.rolling(50, step=step).quantile(q)
    assert isinstance(result, DataFrame)
    end = range(0, len(frame), step or 1)[-1] + 1
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[end - 50 : end, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_time_rule_series(series, q):
    compare_func = partial(scoreatpercentile, per=q)
    win = 25
    ser = series[::2].resample("B").mean()
    series_result = ser.rolling(window=win, min_periods=10).quantile(q)
    last_date = series_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_series = series[::2].truncate(prev_date, last_date)
    tm.assert_almost_equal(series_result.iloc[-1], compare_func(trunc_series))


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_time_rule_frame(raw, frame, q):
    compare_func = partial(scoreatpercentile, per=q)
    win = 25
    frm = frame[::2].resample("B").mean()
    frame_result = frm.rolling(window=win, min_periods=10).quantile(q)
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_nans(q):
    compare_func = partial(scoreatpercentile, per=q)
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = obj.rolling(50, min_periods=30).quantile(q)
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))

    # min_periods is working correctly
    result = obj.rolling(20, min_periods=15).quantile(q)
    assert isna(result.iloc[23])
    assert not isna(result.iloc[24])

    assert not isna(result.iloc[-6])
    assert isna(result.iloc[-5])

    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    result = obj2.rolling(10, min_periods=5).quantile(q)
    assert isna(result.iloc[3])
    assert notna(result.iloc[4])

    result0 = obj.rolling(20, min_periods=0).quantile(q)
    result1 = obj.rolling(20, min_periods=1).quantile(q)
    tm.assert_almost_equal(result0, result1)


@pytest.mark.parametrize("minp", [0, 99, 100])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_min_periods(series, minp, q, step):
    result = series.rolling(len(series) + 1, min_periods=minp, step=step).quantile(q)
    expected = series.rolling(len(series), min_periods=minp, step=step).quantile(q)
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_center(q):
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = obj.rolling(20, center=True).quantile(q)
    expected = (
        concat([obj, Series([np.nan] * 9)])
        .rolling(20)
        .quantile(q)
        .iloc[9:]
        .reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_center_reindex_series(series, q):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    series_xp = (
        series.reindex(list(series.index) + s)
        .rolling(window=25)
        .quantile(q)
        .shift(-12)
        .reindex(series.index)
    )

    series_rs = series.rolling(window=25, center=True).quantile(q)
    tm.assert_series_equal(series_xp, series_rs)


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_center_reindex_frame(frame, q):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    frame_xp = (
        frame.reindex(list(frame.index) + s)
        .rolling(window=25)
        .quantile(q)
        .shift(-12)
        .reindex(frame.index)
    )
    frame_rs = frame.rolling(window=25, center=True).quantile(q)
    tm.assert_frame_equal(frame_xp, frame_rs)


def test_keyword_quantile_deprecated():
    # GH #52550
    s = Series([1, 2, 3, 4])
    with tm.assert_produces_warning(FutureWarning):
        s.rolling(2).quantile(quantile=0.4)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_tableform.py ===
from sympy.core.singleton import S
from sympy.printing.tableform import TableForm
from sympy.printing.latex import latex
from sympy.abc import x
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises

from textwrap import dedent


def test_TableForm():
    s = str(TableForm([["a", "b"], ["c", "d"], ["e", 0]],
        headings="automatic"))
    assert s == (
        '  | 1 2\n'
        '-------\n'
        '1 | a b\n'
        '2 | c d\n'
        '3 | e  '
    )
    s = str(TableForm([["a", "b"], ["c", "d"], ["e", 0]],
        headings="automatic", wipe_zeros=False))
    assert s == dedent('''\
          | 1 2
        -------
        1 | a b
        2 | c d
        3 | e 0''')
    s = str(TableForm([[x**2, "b"], ["c", x**2], ["e", "f"]],
            headings=("automatic", None)))
    assert s == (
        '1 | x**2 b   \n'
        '2 | c    x**2\n'
        '3 | e    f   '
    )
    s = str(TableForm([["a", "b"], ["c", "d"], ["e", "f"]],
            headings=(None, "automatic")))
    assert s == dedent('''\
        1 2
        ---
        a b
        c d
        e f''')
    s = str(TableForm([[5, 7], [4, 2], [10, 3]],
            headings=[["Group A", "Group B", "Group C"], ["y1", "y2"]]))
    assert s == (
        '        | y1 y2\n'
        '---------------\n'
        'Group A | 5  7 \n'
        'Group B | 4  2 \n'
        'Group C | 10 3 '
    )
    raises(
        ValueError,
        lambda:
        TableForm(
            [[5, 7], [4, 2], [10, 3]],
            headings=[["Group A", "Group B", "Group C"], ["y1", "y2"]],
            alignments="middle")
    )
    s = str(TableForm([[5, 7], [4, 2], [10, 3]],
            headings=[["Group A", "Group B", "Group C"], ["y1", "y2"]],
            alignments="right"))
    assert s == dedent('''\
                | y1 y2
        ---------------
        Group A |  5  7
        Group B |  4  2
        Group C | 10  3''')

    # other alignment permutations
    d = [[1, 100], [100, 1]]
    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='l')
    assert str(s) == (
        'xxx | 1   100\n'
        '  x | 100 1  '
    )
    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='lr')
    assert str(s) == dedent('''\
    xxx | 1   100
      x | 100   1''')
    s = TableForm(d, headings=(('xxx', 'x'), None), alignments='clr')
    assert str(s) == dedent('''\
    xxx | 1   100
     x  | 100   1''')

    s = TableForm(d, headings=(('xxx', 'x'), None))
    assert str(s) == (
        'xxx | 1   100\n'
        '  x | 100 1  '
    )

    raises(ValueError, lambda: TableForm(d, alignments='clr'))

    #pad
    s = str(TableForm([[None, "-", 2], [1]], pad='?'))
    assert s == dedent('''\
        ? - 2
        1 ? ?''')


def test_TableForm_latex():
    s = latex(TableForm([[0, x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            wipe_zeros=True, headings=("automatic", "automatic")))
    assert s == (
        '\\begin{tabular}{r l l}\n'
        ' & 1 & 2 \\\\\n'
        '\\hline\n'
        '1 &   & $x^{3}$ \\\\\n'
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'
        '\\end{tabular}'
    )
    s = latex(TableForm([[0, x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            wipe_zeros=True, headings=("automatic", "automatic"), alignments='l'))
    assert s == (
        '\\begin{tabular}{r l l}\n'
        ' & 1 & 2 \\\\\n'
        '\\hline\n'
        '1 &   & $x^{3}$ \\\\\n'
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'
        '\\end{tabular}'
    )
    s = latex(TableForm([[0, x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            wipe_zeros=True, headings=("automatic", "automatic"), alignments='l'*3))
    assert s == (
        '\\begin{tabular}{l l l}\n'
        ' & 1 & 2 \\\\\n'
        '\\hline\n'
        '1 &   & $x^{3}$ \\\\\n'
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'
        '\\end{tabular}'
    )
    s = latex(TableForm([["a", x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            headings=("automatic", "automatic")))
    assert s == (
        '\\begin{tabular}{r l l}\n'
        ' & 1 & 2 \\\\\n'
        '\\hline\n'
        '1 & $a$ & $x^{3}$ \\\\\n'
        '2 & $c$ & $\\frac{1}{4}$ \\\\\n'
        '3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'
        '\\end{tabular}'
    )
    s = latex(TableForm([["a", x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]],
            formats=['(%s)', None], headings=("automatic", "automatic")))
    assert s == (
        '\\begin{tabular}{r l l}\n'
        ' & 1 & 2 \\\\\n'
        '\\hline\n'
        '1 & (a) & $x^{3}$ \\\\\n'
        '2 & (c) & $\\frac{1}{4}$ \\\\\n'
        '3 & (sqrt(x)) & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'
        '\\end{tabular}'
    )

    def neg_in_paren(x, i, j):
        if i % 2:
            return ('(%s)' if x < 0 else '%s') % x
        else:
            pass  # use default print
    s = latex(TableForm([[-1, 2], [-3, 4]],
            formats=[neg_in_paren]*2, headings=("automatic", "automatic")))
    assert s == (
        '\\begin{tabular}{r l l}\n'
        ' & 1 & 2 \\\\\n'
        '\\hline\n'
        '1 & -1 & 2 \\\\\n'
        '2 & (-3) & 4 \\\\\n'
        '\\end{tabular}'
    )
    s = latex(TableForm([["a", x**3], ["c", S.One/4], [sqrt(x), sin(x**2)]]))
    assert s == (
        '\\begin{tabular}{l l}\n'
        '$a$ & $x^{3}$ \\\\\n'
        '$c$ & $\\frac{1}{4}$ \\\\\n'
        '$\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n'
        '\\end{tabular}'
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_fuzz.py ===
"""This file contains test cases reported by third parties using
fuzzing tools, primarily from Google's oss-fuzz project. Some of these
represent real problems with Beautiful Soup, but many are problems in
libraries that Beautiful Soup depends on, and many of the test cases
represent different ways of triggering the same problem.

Grouping these test cases together makes it easy to see which test
cases represent the same problem, and puts the test cases in close
proximity to code that can trigger the problems.
"""

import os
import importlib
import pytest
from bs4 import (
    BeautifulSoup,
    ParserRejectedMarkup,
)

try:
    from soupsieve.util import SelectorSyntaxError
    has_lxml = importlib.util.find_spec("lxml")
    has_html5lib = importlib.util.find_spec("html5lib")
    fully_fuzzable = has_lxml != None and has_html5lib != None
except ImportError:
    fully_fuzzable = False


@pytest.mark.skipif(
    not fully_fuzzable, reason="Prerequisites for fuzz tests are not installed."
)
class TestFuzz(object):
    # Test case markup files from fuzzers are given this extension so
    # they can be included in builds.
    TESTCASE_SUFFIX = ".testcase"

    # Copied 20230512 from
    # https://github.com/google/oss-fuzz/blob/4ac6a645a197a695fe76532251feb5067076b3f3/projects/bs4/bs4_fuzzer.py
    #
    # Copying the code lets us precisely duplicate the behavior of
    # oss-fuzz.  The downside is that this code changes over time, so
    # multiple copies of the code must be kept around to run against
    # older tests. I'm not sure what to do about this, but I may
    # retire old tests after a time.
    def fuzz_test_with_css(self, filename: str) -> None:
        data = self.__markup(filename)
        parsers = ["lxml-xml", "html5lib", "html.parser", "lxml"]
        try:
            idx = int(data[0]) % len(parsers)
        except ValueError:
            return

        css_selector, data = data[1:10], data[10:]

        try:
            soup = BeautifulSoup(data[1:], features=parsers[idx])
        except ParserRejectedMarkup:
            return
        except ValueError:
            return

        list(soup.find_all(True))
        try:
            soup.css.select(css_selector.decode("utf-8", "replace"))
        except SelectorSyntaxError:
            return
        soup.prettify()

    # This class of error has been fixed by catching a less helpful
    # exception from html.parser and raising ParserRejectedMarkup
    # instead.
    @pytest.mark.parametrize(
        "filename",
        [
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5703933063462912",
            "crash-ffbdfa8a2b26f13537b68d3794b0478a4090ee4a",
        ],
    )
    def test_rejected_markup(self, filename):
        markup = self.__markup(filename)
        with pytest.raises(ParserRejectedMarkup):
            BeautifulSoup(markup, "html.parser")

    # This class of error has to do with very deeply nested documents
    # which overflow the Python call stack when the tree is converted
    # to a string. This is an issue with Beautiful Soup which was fixed
    # as part of [bug=1471755].
    #
    # These test cases are in the older format that doesn't specify
    # which parser to use or give a CSS selector.
    @pytest.mark.parametrize(
        "filename",
        [
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5984173902397440",
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5167584867909632",
            "clusterfuzz-testcase-minimized-bs4_fuzzer-6124268085182464",
            "clusterfuzz-testcase-minimized-bs4_fuzzer-6450958476902400",
        ],
    )
    def test_deeply_nested_document_without_css(self, filename):
        # Parsing the document and encoding it back to a string is
        # sufficient to demonstrate that the overflow problem has
        # been fixed.
        markup = self.__markup(filename)
        BeautifulSoup(markup, "html.parser").encode()

    # This class of error has to do with very deeply nested documents
    # which overflow the Python call stack when the tree is converted
    # to a string. This is an issue with Beautiful Soup which was fixed
    # as part of [bug=1471755].
    @pytest.mark.parametrize(
        "filename",
        [
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5000587759190016",
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5375146639360000",
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5492400320282624",
        ],
    )
    def test_deeply_nested_document(self, filename):
        self.fuzz_test_with_css(filename)

    @pytest.mark.parametrize(
        "filename",
        [
            "clusterfuzz-testcase-minimized-bs4_fuzzer-4670634698080256",
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5270998950477824",
        ],
    )
    def test_soupsieve_errors(self, filename):
        self.fuzz_test_with_css(filename)

    # This class of error represents problems with html5lib's parser,
    # not Beautiful Soup. I use
    # https://github.com/html5lib/html5lib-python/issues/568 to notify
    # the html5lib developers of these issues.
    #
    # These test cases are in the older format that doesn't specify
    # which parser to use or give a CSS selector.
    @pytest.mark.skip(reason="html5lib-specific problems")
    @pytest.mark.parametrize(
        "filename",
        [
            # b"""ÿ<!DOCTyPEV PUBLIC'''Ð'"""
            "clusterfuzz-testcase-minimized-bs4_fuzzer-4818336571064320",
            # b')<a><math><TR><a><mI><a><p><a>'
            "clusterfuzz-testcase-minimized-bs4_fuzzer-4999465949331456",
            # b'-<math><sElect><mi><sElect><sElect>'
            "clusterfuzz-testcase-minimized-bs4_fuzzer-5843991618256896",
            # b'ñ<table><svg><html>'
            "clusterfuzz-testcase-minimized-bs4_fuzzer-6241471367348224",
            # <TABLE>, some ^@ characters, some <math> tags.
            "clusterfuzz-testcase-minimized-bs4_fuzzer-6600557255327744",
            # Nested table
            "crash-0d306a50c8ed8bcd0785b67000fcd5dea1d33f08",
        ],
    )
    def test_html5lib_parse_errors_without_css(self, filename):
        markup = self.__markup(filename)
        print(BeautifulSoup(markup, "html5lib").encode())

    # This class of error represents problems with html5lib's parser,
    # not Beautiful Soup. I use
    # https://github.com/html5lib/html5lib-python/issues/568 to notify
    # the html5lib developers of these issues.
    @pytest.mark.skip(reason="html5lib-specific problems")
    @pytest.mark.parametrize(
        "filename",
        [
            # b'-      \xff\xff  <math>\x10<select><mi><select><select>t'
            "clusterfuzz-testcase-minimized-bs4_fuzzer-6306874195312640",
        ],
    )
    def test_html5lib_parse_errors(self, filename):
        self.fuzz_test_with_css(filename)

    def __markup(self, filename: str) -> bytes:
        if not filename.endswith(self.TESTCASE_SUFFIX):
            filename += self.TESTCASE_SUFFIX
        this_dir = os.path.split(__file__)[0]
        path = os.path.join(this_dir, "fuzz", filename)
        return open(path, "rb").read()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\linalg\tests\test_regression.py ===
""" Test functions for linalg module
"""

import pytest

import numpy as np
from numpy import arange, array, dot, float64, linalg, transpose
from numpy.testing import (
    assert_,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
)


class TestRegression:

    def test_eig_build(self):
        # Ticket #652
        rva = array([1.03221168e+02 + 0.j,
                     -1.91843603e+01 + 0.j,
                     -6.04004526e-01 + 15.84422474j,
                     -6.04004526e-01 - 15.84422474j,
                     -1.13692929e+01 + 0.j,
                     -6.57612485e-01 + 10.41755503j,
                     -6.57612485e-01 - 10.41755503j,
                     1.82126812e+01 + 0.j,
                     1.06011014e+01 + 0.j,
                     7.80732773e+00 + 0.j,
                     -7.65390898e-01 + 0.j,
                     1.51971555e-15 + 0.j,
                     -1.51308713e-15 + 0.j])
        a = arange(13 * 13, dtype=float64)
        a.shape = (13, 13)
        a = a % 17
        va, ve = linalg.eig(a)
        va.sort()
        rva.sort()
        assert_array_almost_equal(va, rva)

    def test_eigh_build(self):
        # Ticket 662.
        rvals = [68.60568999, 89.57756725, 106.67185574]

        cov = array([[77.70273908,  3.51489954, 15.64602427],
                     [ 3.51489954, 88.97013878, -1.07431931],
                     [15.64602427, -1.07431931, 98.18223512]])

        vals, vecs = linalg.eigh(cov)
        assert_array_almost_equal(vals, rvals)

    def test_svd_build(self):
        # Ticket 627.
        a = array([[0., 1.], [1., 1.], [2., 1.], [3., 1.]])
        m, n = a.shape
        u, s, vh = linalg.svd(a)

        b = dot(transpose(u[:, n:]), a)

        assert_array_almost_equal(b, np.zeros((2, 2)))

    def test_norm_vector_badarg(self):
        # Regression for #786: Frobenius norm for vectors raises
        # ValueError.
        assert_raises(ValueError, linalg.norm, array([1., 2., 3.]), 'fro')

    def test_lapack_endian(self):
        # For bug #1482
        a = array([[ 5.7998084, -2.1825367],
                   [-2.1825367,  9.85910595]], dtype='>f8')
        b = array(a, dtype='<f8')

        ap = linalg.cholesky(a)
        bp = linalg.cholesky(b)
        assert_array_equal(ap, bp)

    def test_large_svd_32bit(self):
        # See gh-4442, 64bit would require very large/slow matrices.
        x = np.eye(1000, 66)
        np.linalg.svd(x)

    def test_svd_no_uv(self):
        # gh-4733
        for shape in (3, 4), (4, 4), (4, 3):
            for t in float, complex:
                a = np.ones(shape, dtype=t)
                w = linalg.svd(a, compute_uv=False)
                c = np.count_nonzero(np.absolute(w) > 0.5)
                assert_equal(c, 1)
                assert_equal(np.linalg.matrix_rank(a), 1)
                assert_array_less(1, np.linalg.norm(a, ord=2))

                w_svdvals = linalg.svdvals(a)
                assert_array_almost_equal(w, w_svdvals)

    def test_norm_object_array(self):
        # gh-7575
        testvector = np.array([np.array([0, 1]), 0, 0], dtype=object)

        norm = linalg.norm(testvector)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        norm = linalg.norm(testvector, ord=1)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype != np.dtype('float64'))

        norm = linalg.norm(testvector, ord=2)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        assert_raises(ValueError, linalg.norm, testvector, ord='fro')
        assert_raises(ValueError, linalg.norm, testvector, ord='nuc')
        assert_raises(ValueError, linalg.norm, testvector, ord=np.inf)
        assert_raises(ValueError, linalg.norm, testvector, ord=-np.inf)
        assert_raises(ValueError, linalg.norm, testvector, ord=0)
        assert_raises(ValueError, linalg.norm, testvector, ord=-1)
        assert_raises(ValueError, linalg.norm, testvector, ord=-2)

        testmatrix = np.array([[np.array([0, 1]), 0, 0],
                               [0,                0, 0]], dtype=object)

        norm = linalg.norm(testmatrix)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        norm = linalg.norm(testmatrix, ord='fro')
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        assert_raises(TypeError, linalg.norm, testmatrix, ord='nuc')
        assert_raises(ValueError, linalg.norm, testmatrix, ord=np.inf)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=-np.inf)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=0)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=1)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=-1)
        assert_raises(TypeError, linalg.norm, testmatrix, ord=2)
        assert_raises(TypeError, linalg.norm, testmatrix, ord=-2)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=3)

    def test_lstsq_complex_larger_rhs(self):
        # gh-9891
        size = 20
        n_rhs = 70
        G = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        u = np.random.randn(size, n_rhs) + 1j * np.random.randn(size, n_rhs)
        b = G.dot(u)
        # This should work without segmentation fault.
        u_lstsq, res, rank, sv = linalg.lstsq(G, b, rcond=None)
        # check results just in case
        assert_array_almost_equal(u_lstsq, u)

    @pytest.mark.parametrize("upper", [True, False])
    def test_cholesky_empty_array(self, upper):
        # gh-25840 - upper=True hung before.
        res = np.linalg.cholesky(np.zeros((0, 0)), upper=upper)
        assert res.size == 0

    @pytest.mark.parametrize("rtol", [0.0, [0.0] * 4, np.zeros((4,))])
    def test_matrix_rank_rtol_argument(self, rtol):
        # gh-25877
        x = np.zeros((4, 3, 2))
        res = np.linalg.matrix_rank(x, rtol=rtol)
        assert res.shape == (4,)

    def test_openblas_threading(self):
        # gh-27036
        # Test whether matrix multiplication involving a large matrix always
        # gives the same (correct) answer
        x = np.arange(500000, dtype=np.float64)
        src = np.vstack((x, -10 * x)).T
        matrix = np.array([[0, 1], [1, 0]])
        expected = np.vstack((-10 * x, x)).T  # src @ matrix
        for i in range(200):
            result = src @ matrix
            mismatches = (~np.isclose(result, expected)).sum()
            if mismatches != 0:
                assert False, ("unexpected result from matmul, "
                    "probably due to OpenBLAS threading issues")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\test_equals.py ===
"""
Tests shared for DatetimeIndex/TimedeltaIndex/PeriodIndex
"""
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    PeriodIndex,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class EqualsTests:
    def test_not_equals_numeric(self, index):
        assert not index.equals(Index(index.asi8))
        assert not index.equals(Index(index.asi8.astype("u8")))
        assert not index.equals(Index(index.asi8).astype("f8"))

    def test_equals(self, index):
        assert index.equals(index)
        assert index.equals(index.astype(object))
        assert index.equals(CategoricalIndex(index))
        assert index.equals(CategoricalIndex(index.astype(object)))

    def test_not_equals_non_arraylike(self, index):
        assert not index.equals(list(index))

    def test_not_equals_strings(self, index):
        other = Index([str(x) for x in index], dtype=object)
        assert not index.equals(other)
        assert not index.equals(CategoricalIndex(other))

    def test_not_equals_misc_strs(self, index):
        other = Index(list("abc"))
        assert not index.equals(other)


class TestPeriodIndexEquals(EqualsTests):
    @pytest.fixture
    def index(self):
        return period_range("2013-01-01", periods=5, freq="D")

    # TODO: de-duplicate with other test_equals2 methods
    @pytest.mark.parametrize("freq", ["D", "M"])
    def test_equals2(self, freq):
        # GH#13107
        idx = PeriodIndex(["2011-01-01", "2011-01-02", "NaT"], freq=freq)
        assert idx.equals(idx)
        assert idx.equals(idx.copy())
        assert idx.equals(idx.astype(object))
        assert idx.astype(object).equals(idx)
        assert idx.astype(object).equals(idx.astype(object))
        assert not idx.equals(list(idx))
        assert not idx.equals(pd.Series(idx))

        idx2 = PeriodIndex(["2011-01-01", "2011-01-02", "NaT"], freq="h")
        assert not idx.equals(idx2)
        assert not idx.equals(idx2.copy())
        assert not idx.equals(idx2.astype(object))
        assert not idx.astype(object).equals(idx2)
        assert not idx.equals(list(idx2))
        assert not idx.equals(pd.Series(idx2))

        # same internal, different tz
        idx3 = PeriodIndex._simple_new(
            idx._values._simple_new(idx._values.asi8, dtype=pd.PeriodDtype("h"))
        )
        tm.assert_numpy_array_equal(idx.asi8, idx3.asi8)
        assert not idx.equals(idx3)
        assert not idx.equals(idx3.copy())
        assert not idx.equals(idx3.astype(object))
        assert not idx.astype(object).equals(idx3)
        assert not idx.equals(list(idx3))
        assert not idx.equals(pd.Series(idx3))


class TestDatetimeIndexEquals(EqualsTests):
    @pytest.fixture
    def index(self):
        return date_range("2013-01-01", periods=5)

    def test_equals2(self):
        # GH#13107
        idx = DatetimeIndex(["2011-01-01", "2011-01-02", "NaT"])
        assert idx.equals(idx)
        assert idx.equals(idx.copy())
        assert idx.equals(idx.astype(object))
        assert idx.astype(object).equals(idx)
        assert idx.astype(object).equals(idx.astype(object))
        assert not idx.equals(list(idx))
        assert not idx.equals(pd.Series(idx))

        idx2 = DatetimeIndex(["2011-01-01", "2011-01-02", "NaT"], tz="US/Pacific")
        assert not idx.equals(idx2)
        assert not idx.equals(idx2.copy())
        assert not idx.equals(idx2.astype(object))
        assert not idx.astype(object).equals(idx2)
        assert not idx.equals(list(idx2))
        assert not idx.equals(pd.Series(idx2))

        # same internal, different tz
        idx3 = DatetimeIndex(idx.asi8, tz="US/Pacific")
        tm.assert_numpy_array_equal(idx.asi8, idx3.asi8)
        assert not idx.equals(idx3)
        assert not idx.equals(idx3.copy())
        assert not idx.equals(idx3.astype(object))
        assert not idx.astype(object).equals(idx3)
        assert not idx.equals(list(idx3))
        assert not idx.equals(pd.Series(idx3))

        # check that we do not raise when comparing with OutOfBounds objects
        oob = Index([datetime(2500, 1, 1)] * 3, dtype=object)
        assert not idx.equals(oob)
        assert not idx2.equals(oob)
        assert not idx3.equals(oob)

        # check that we do not raise when comparing with OutOfBounds dt64
        oob2 = oob.map(np.datetime64)
        assert not idx.equals(oob2)
        assert not idx2.equals(oob2)
        assert not idx3.equals(oob2)

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_not_equals_bday(self, freq):
        rng = date_range("2009-01-01", "2010-01-01", freq=freq)
        assert not rng.equals(list(rng))


class TestTimedeltaIndexEquals(EqualsTests):
    @pytest.fixture
    def index(self):
        return timedelta_range("1 day", periods=10)

    def test_equals2(self):
        # GH#13107
        idx = TimedeltaIndex(["1 days", "2 days", "NaT"])
        assert idx.equals(idx)
        assert idx.equals(idx.copy())
        assert idx.equals(idx.astype(object))
        assert idx.astype(object).equals(idx)
        assert idx.astype(object).equals(idx.astype(object))
        assert not idx.equals(list(idx))
        assert not idx.equals(pd.Series(idx))

        idx2 = TimedeltaIndex(["2 days", "1 days", "NaT"])
        assert not idx.equals(idx2)
        assert not idx.equals(idx2.copy())
        assert not idx.equals(idx2.astype(object))
        assert not idx.astype(object).equals(idx2)
        assert not idx.astype(object).equals(idx2.astype(object))
        assert not idx.equals(list(idx2))
        assert not idx.equals(pd.Series(idx2))

        # Check that we dont raise OverflowError on comparisons outside the
        #  implementation range GH#28532
        oob = Index([timedelta(days=10**6)] * 3, dtype=object)
        assert not idx.equals(oob)
        assert not idx2.equals(oob)

        oob2 = Index([np.timedelta64(x) for x in oob], dtype=object)
        assert (oob == oob2).all()
        assert not idx.equals(oob2)
        assert not idx2.equals(oob2)

        oob3 = oob.map(np.timedelta64)
        assert (oob3 == oob).all()
        assert not idx.equals(oob3)
        assert not idx2.equals(oob3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\methods\test_astype.py ===
from datetime import timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    NaT,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestTimedeltaIndex:
    def test_astype_object(self):
        idx = timedelta_range(start="1 days", periods=4, freq="D", name="idx")
        expected_list = [
            Timedelta("1 days"),
            Timedelta("2 days"),
            Timedelta("3 days"),
            Timedelta("4 days"),
        ]
        result = idx.astype(object)
        expected = Index(expected_list, dtype=object, name="idx")
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    def test_astype_object_with_nat(self):
        idx = TimedeltaIndex(
            [timedelta(days=1), timedelta(days=2), NaT, timedelta(days=4)], name="idx"
        )
        expected_list = [
            Timedelta("1 days"),
            Timedelta("2 days"),
            NaT,
            Timedelta("4 days"),
        ]
        result = idx.astype(object)
        expected = Index(expected_list, dtype=object, name="idx")
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    def test_astype(self, using_infer_string):
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan], name="idx")

        result = idx.astype(object)
        expected = Index(
            [Timedelta("1 days 03:46:40")] + [NaT] * 3, dtype=object, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(np.int64)
        expected = Index(
            [100000000000000] + [-9223372036854775808] * 3, dtype=np.int64, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(str)
        if using_infer_string:
            expected = Index(
                [str(x) if x is not NaT else None for x in idx], name="idx", dtype="str"
            )
        else:
            expected = Index([str(x) for x in idx], name="idx", dtype=object)
        tm.assert_index_equal(result, expected)

        rng = timedelta_range("1 days", periods=10)
        result = rng.astype("i8")
        tm.assert_index_equal(result, Index(rng.asi8))
        tm.assert_numpy_array_equal(rng.asi8, result.values)

    def test_astype_uint(self):
        arr = timedelta_range("1h", periods=2)

        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint64")
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint32")

    def test_astype_timedelta64(self):
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan])

        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        with pytest.raises(ValueError, match=msg):
            idx.astype("timedelta64")

        result = idx.astype("timedelta64[ns]")
        tm.assert_index_equal(result, idx)
        assert result is not idx

        result = idx.astype("timedelta64[ns]", copy=False)
        tm.assert_index_equal(result, idx)
        assert result is idx

    def test_astype_to_td64d_raises(self, index_or_series):
        # We don't support "D" reso
        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )
        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64\[D\]. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        with pytest.raises(ValueError, match=msg):
            td.astype("timedelta64[D]")

    def test_astype_ms_to_s(self, index_or_series):
        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )

        exp_values = np.asarray(td).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(exp_values, dtype=exp_values.dtype)
        expected = index_or_series(exp_tda)
        assert expected.dtype == "m8[s]"
        result = td.astype("timedelta64[s]")
        tm.assert_equal(result, expected)

    def test_astype_freq_conversion(self):
        # pre-2.0 td64 astype converted to float64. now for supported units
        #  (s, ms, us, ns) this converts to the requested dtype.
        # This matches TDA and Series
        tdi = timedelta_range("1 Day", periods=30)

        res = tdi.astype("m8[s]")
        exp_values = np.asarray(tdi).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(
            exp_values, dtype=exp_values.dtype, freq=tdi.freq
        )
        expected = Index(exp_tda)
        assert expected.dtype == "m8[s]"
        tm.assert_index_equal(res, expected)

        # check this matches Series and TimedeltaArray
        res = tdi._data.astype("m8[s]")
        tm.assert_equal(res, expected._values)

        res = tdi.to_series().astype("m8[s]")
        tm.assert_equal(res._values, expected._values._with_freq(None))

    @pytest.mark.parametrize("dtype", [float, "datetime64", "datetime64[ns]"])
    def test_astype_raises(self, dtype):
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan])
        msg = "Cannot cast TimedeltaIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)

    def test_astype_category(self):
        obj = timedelta_range("1h", periods=2, freq="h")

        result = obj.astype("category")
        expected = pd.CategoricalIndex([Timedelta("1h"), Timedelta("2h")])
        tm.assert_index_equal(result, expected)

        result = obj._data.astype("category")
        expected = expected.values
        tm.assert_categorical_equal(result, expected)

    def test_astype_array_fallback(self):
        obj = timedelta_range("1h", periods=2)
        result = obj.astype(bool)
        expected = Index(np.array([True, True]))
        tm.assert_index_equal(result, expected)

        result = obj._data.astype(bool)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\dtypes\test_empty.py ===
"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm

skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_dtype_all_columns_empty(all_parsers):
    # see gh-12048
    parser = all_parsers
    result = parser.read_csv(StringIO("A,B"), dtype=str)

    expected = DataFrame({"A": [], "B": []}, dtype=str)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_pass_dtype(all_parsers):
    parser = all_parsers

    data = "one,two"
    result = parser.read_csv(StringIO(data), dtype={"one": "u1"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "two": np.empty(0, dtype=object)},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_index_pass_dtype(all_parsers):
    parser = all_parsers

    data = "one,two"
    result = parser.read_csv(
        StringIO(data), index_col=["one"], dtype={"one": "u1", 1: "f"}
    )

    expected = DataFrame(
        {"two": np.empty(0, dtype="f")}, index=Index([], dtype="u1", name="one")
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_multi_index_pass_dtype(all_parsers):
    parser = all_parsers

    data = "one,two,three"
    result = parser.read_csv(
        StringIO(data), index_col=["one", "two"], dtype={"one": "u1", 1: "f8"}
    )

    exp_idx = MultiIndex.from_arrays(
        [np.empty(0, dtype="u1"), np.empty(0, dtype=np.float64)],
        names=["one", "two"],
    )
    expected = DataFrame({"three": np.empty(0, dtype=object)}, index=exp_idx)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_mangled_column_pass_dtype_by_names(all_parsers):
    parser = all_parsers

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={"one": "u1", "one.1": "f"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "one.1": np.empty(0, dtype="f")},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_mangled_column_pass_dtype_by_indexes(all_parsers):
    parser = all_parsers

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={0: "u1", 1: "f"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "one.1": np.empty(0, dtype="f")},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_with_dup_column_pass_dtype_by_indexes(all_parsers):
    # see gh-9424
    parser = all_parsers
    expected = concat(
        [Series([], name="one", dtype="u1"), Series([], name="one.1", dtype="f")],
        axis=1,
    )

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={0: "u1", 1: "f"})
    tm.assert_frame_equal(result, expected)


def test_empty_with_dup_column_pass_dtype_by_indexes_raises(all_parsers):
    # see gh-9424
    parser = all_parsers
    expected = concat(
        [Series([], name="one", dtype="u1"), Series([], name="one.1", dtype="f")],
        axis=1,
    )
    expected.index = expected.index.astype(object)

    with pytest.raises(ValueError, match="Duplicate names"):
        data = ""
        parser.read_csv(StringIO(data), names=["one", "one"], dtype={0: "u1", 1: "f"})


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.float64, DataFrame(columns=["a", "b"], dtype=np.float64)),
        (
            "category",
            DataFrame({"a": Categorical([]), "b": Categorical([])}),
        ),
        (
            {"a": "category", "b": "category"},
            DataFrame({"a": Categorical([]), "b": Categorical([])}),
        ),
        ("datetime64[ns]", DataFrame(columns=["a", "b"], dtype="datetime64[ns]")),
        (
            "timedelta64[ns]",
            DataFrame(
                {
                    "a": Series([], dtype="timedelta64[ns]"),
                    "b": Series([], dtype="timedelta64[ns]"),
                },
            ),
        ),
        (
            {"a": np.int64, "b": np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),
        (
            {0: np.int64, 1: np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),
        (
            {"a": np.int64, 1: np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),
    ],
)
@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_empty_dtype(all_parsers, dtype, expected):
    # see gh-14712
    parser = all_parsers
    data = "a,b"

    result = parser.read_csv(StringIO(data), header=0, dtype=dtype)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_elliptic_integrals.py ===
from sympy.core.numbers import (I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (sin, tan)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.functions.special.elliptic_integrals import (elliptic_k as K,
    elliptic_f as F, elliptic_e as E, elliptic_pi as P)
from sympy.core.random import (test_derivative_numerically as td,
                                      random_complex_number as randcplx,
                                      verify_numerically as tn)
from sympy.abc import z, m, n

i = Symbol('i', integer=True)
j = Symbol('k', integer=True, positive=True)
t = Dummy('t')

def test_K():
    assert K(0) == pi/2
    assert K(S.Half) == 8*pi**Rational(3, 2)/gamma(Rational(-1, 4))**2
    assert K(1) is zoo
    assert K(-1) == gamma(Rational(1, 4))**2/(4*sqrt(2*pi))
    assert K(oo) == 0
    assert K(-oo) == 0
    assert K(I*oo) == 0
    assert K(-I*oo) == 0
    assert K(zoo) == 0

    assert K(z).diff(z) == (E(z) - (1 - z)*K(z))/(2*z*(1 - z))
    assert td(K(z), z)

    zi = Symbol('z', real=False)
    assert K(zi).conjugate() == K(zi.conjugate())
    zr = Symbol('z', negative=True)
    assert K(zr).conjugate() == K(zr)

    assert K(z).rewrite(hyper) == \
        (pi/2)*hyper((S.Half, S.Half), (S.One,), z)
    assert tn(K(z), (pi/2)*hyper((S.Half, S.Half), (S.One,), z))
    assert K(z).rewrite(meijerg) == \
        meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -z)/2
    assert tn(K(z), meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -z)/2)

    assert K(z).series(z) == pi/2 + pi*z/8 + 9*pi*z**2/128 + \
        25*pi*z**3/512 + 1225*pi*z**4/32768 + 3969*pi*z**5/131072 + O(z**6)

    assert K(m).rewrite(Integral).dummy_eq(
        Integral(1/sqrt(1 - m*sin(t)**2), (t, 0, pi/2)))

def test_F():
    assert F(z, 0) == z
    assert F(0, m) == 0
    assert F(pi*i/2, m) == i*K(m)
    assert F(z, oo) == 0
    assert F(z, -oo) == 0

    assert F(-z, m) == -F(z, m)

    assert F(z, m).diff(z) == 1/sqrt(1 - m*sin(z)**2)
    assert F(z, m).diff(m) == E(z, m)/(2*m*(1 - m)) - F(z, m)/(2*m) - \
        sin(2*z)/(4*(1 - m)*sqrt(1 - m*sin(z)**2))
    r = randcplx()
    assert td(F(z, r), z)
    assert td(F(r, m), m)

    mi = Symbol('m', real=False)
    assert F(z, mi).conjugate() == F(z.conjugate(), mi.conjugate())
    mr = Symbol('m', negative=True)
    assert F(z, mr).conjugate() == F(z.conjugate(), mr)

    assert F(z, m).series(z) == \
        z + z**5*(3*m**2/40 - m/30) + m*z**3/6 + O(z**6)

    assert F(z, m).rewrite(Integral).dummy_eq(
        Integral(1/sqrt(1 - m*sin(t)**2), (t, 0, z)))

def test_E():
    assert E(z, 0) == z
    assert E(0, m) == 0
    assert E(i*pi/2, m) == i*E(m)
    assert E(z, oo) is zoo
    assert E(z, -oo) is zoo
    assert E(0) == pi/2
    assert E(1) == 1
    assert E(oo) == I*oo
    assert E(-oo) is oo
    assert E(zoo) is zoo

    assert E(-z, m) == -E(z, m)

    assert E(z, m).diff(z) == sqrt(1 - m*sin(z)**2)
    assert E(z, m).diff(m) == (E(z, m) - F(z, m))/(2*m)
    assert E(z).diff(z) == (E(z) - K(z))/(2*z)
    r = randcplx()
    assert td(E(r, m), m)
    assert td(E(z, r), z)
    assert td(E(z), z)

    mi = Symbol('m', real=False)
    assert E(z, mi).conjugate() == E(z.conjugate(), mi.conjugate())
    assert E(mi).conjugate() == E(mi.conjugate())
    mr = Symbol('m', negative=True)
    assert E(z, mr).conjugate() == E(z.conjugate(), mr)
    assert E(mr).conjugate() == E(mr)

    assert E(z).rewrite(hyper) == (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), z)
    assert tn(E(z), (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), z))
    assert E(z).rewrite(meijerg) == \
        -meijerg(((S.Half, Rational(3, 2)), []), ((S.Zero,), (S.Zero,)), -z)/4
    assert tn(E(z), -meijerg(((S.Half, Rational(3, 2)), []), ((S.Zero,), (S.Zero,)), -z)/4)

    assert E(z, m).series(z) == \
        z + z**5*(-m**2/40 + m/30) - m*z**3/6 + O(z**6)
    assert E(z).series(z) == pi/2 - pi*z/8 - 3*pi*z**2/128 - \
        5*pi*z**3/512 - 175*pi*z**4/32768 - 441*pi*z**5/131072 + O(z**6)
    assert E(4*z/(z+1)).series(z) == \
        pi/2 - pi*z/2 + pi*z**2/8 - 3*pi*z**3/8 - 15*pi*z**4/128 - 93*pi*z**5/128 + O(z**6)

    assert E(z, m).rewrite(Integral).dummy_eq(
        Integral(sqrt(1 - m*sin(t)**2), (t, 0, z)))
    assert E(m).rewrite(Integral).dummy_eq(
        Integral(sqrt(1 - m*sin(t)**2), (t, 0, pi/2)))

def test_P():
    assert P(0, z, m) == F(z, m)
    assert P(1, z, m) == F(z, m) + \
        (sqrt(1 - m*sin(z)**2)*tan(z) - E(z, m))/(1 - m)
    assert P(n, i*pi/2, m) == i*P(n, m)
    assert P(n, z, 0) == atanh(sqrt(n - 1)*tan(z))/sqrt(n - 1)
    assert P(n, z, n) == F(z, n) - P(1, z, n) + tan(z)/sqrt(1 - n*sin(z)**2)
    assert P(oo, z, m) == 0
    assert P(-oo, z, m) == 0
    assert P(n, z, oo) == 0
    assert P(n, z, -oo) == 0
    assert P(0, m) == K(m)
    assert P(1, m) is zoo
    assert P(n, 0) == pi/(2*sqrt(1 - n))
    assert P(2, 1) is -oo
    assert P(-1, 1) is oo
    assert P(n, n) == E(n)/(1 - n)

    assert P(n, -z, m) == -P(n, z, m)

    ni, mi = Symbol('n', real=False), Symbol('m', real=False)
    assert P(ni, z, mi).conjugate() == \
        P(ni.conjugate(), z.conjugate(), mi.conjugate())
    nr, mr = Symbol('n', negative=True), \
        Symbol('m', negative=True)
    assert P(nr, z, mr).conjugate() == P(nr, z.conjugate(), mr)
    assert P(n, m).conjugate() == P(n.conjugate(), m.conjugate())

    assert P(n, z, m).diff(n) == (E(z, m) + (m - n)*F(z, m)/n +
        (n**2 - m)*P(n, z, m)/n - n*sqrt(1 -
            m*sin(z)**2)*sin(2*z)/(2*(1 - n*sin(z)**2)))/(2*(m - n)*(n - 1))
    assert P(n, z, m).diff(z) == 1/(sqrt(1 - m*sin(z)**2)*(1 - n*sin(z)**2))
    assert P(n, z, m).diff(m) == (E(z, m)/(m - 1) + P(n, z, m) -
        m*sin(2*z)/(2*(m - 1)*sqrt(1 - m*sin(z)**2)))/(2*(n - m))
    assert P(n, m).diff(n) == (E(m) + (m - n)*K(m)/n +
        (n**2 - m)*P(n, m)/n)/(2*(m - n)*(n - 1))
    assert P(n, m).diff(m) == (E(m)/(m - 1) + P(n, m))/(2*(n - m))

    # These tests fail due to
    # https://github.com/fredrik-johansson/mpmath/issues/571#issuecomment-777201962
    # https://github.com/sympy/sympy/issues/20933#issuecomment-777080385
    #
    # rx, ry = randcplx(), randcplx()
    # assert td(P(n, rx, ry), n)
    # assert td(P(rx, z, ry), z)
    # assert td(P(rx, ry, m), m)

    assert P(n, z, m).series(z) == z + z**3*(m/6 + n/3) + \
        z**5*(3*m**2/40 + m*n/10 - m/30 + n**2/5 - n/15) + O(z**6)

    assert P(n, z, m).rewrite(Integral).dummy_eq(
        Integral(1/((1 - n*sin(t)**2)*sqrt(1 - m*sin(t)**2)), (t, 0, z)))
    assert P(n, m).rewrite(Integral).dummy_eq(
        Integral(1/((1 - n*sin(t)**2)*sqrt(1 - m*sin(t)**2)), (t, 0, pi/2)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\sampling\tests\test_sample_continuous_rv.py ===
from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.sets.sets import Interval
from sympy.external import import_module
from sympy.stats import Beta, Chi, Normal, Gamma, Exponential, LogNormal, Pareto, ChiSquared, Uniform, sample, \
    BetaPrime, Cauchy, GammaInverse, GaussianInverse, StudentT, Weibull, density, ContinuousRV, FDistribution, \
    Gumbel, Laplace, Logistic, Rayleigh, Triangular
from sympy.testing.pytest import skip, raises


def test_sample_numpy():
    distribs_numpy = [
        Beta("B", 1, 1),
        Normal("N", 0, 1),
        Gamma("G", 2, 7),
        Exponential("E", 2),
        LogNormal("LN", 0, 1),
        Pareto("P", 1, 1),
        ChiSquared("CS", 2),
        Uniform("U", 0, 1),
        FDistribution("FD", 1, 2),
        Gumbel("GB", 1, 2),
        Laplace("L", 1, 2),
        Logistic("LO", 1, 2),
        Rayleigh("R", 1),
        Triangular("T", 1, 2, 2),
    ]
    size = 3
    numpy = import_module('numpy')
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        for X in distribs_numpy:
            samps = sample(X, size=size, library='numpy')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
               lambda: sample(Chi("C", 1), library='numpy'))
    raises(NotImplementedError,
           lambda: Chi("C", 1).pspace.distribution.sample(library='tensorflow'))


def test_sample_scipy():
    distribs_scipy = [
        Beta("B", 1, 1),
        BetaPrime("BP", 1, 1),
        Cauchy("C", 1, 1),
        Chi("C", 1),
        Normal("N", 0, 1),
        Gamma("G", 2, 7),
        GammaInverse("GI", 1, 1),
        GaussianInverse("GUI", 1, 1),
        Exponential("E", 2),
        LogNormal("LN", 0, 1),
        Pareto("P", 1, 1),
        StudentT("S", 2),
        ChiSquared("CS", 2),
        Uniform("U", 0, 1)
    ]
    size = 3
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size, library='scipy')
            samps2 = sample(X, size=(2, 2), library='scipy')
            for sam in samps:
                assert sam in X.pspace.domain.set
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


def test_sample_pymc():
    distribs_pymc = [
        Beta("B", 1, 1),
        Cauchy("C", 1, 1),
        Normal("N", 0, 1),
        Gamma("G", 2, 7),
        GaussianInverse("GI", 1, 1),
        Exponential("E", 2),
        LogNormal("LN", 0, 1),
        Pareto("P", 1, 1),
        ChiSquared("CS", 2),
        Uniform("U", 0, 1)
    ]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
               lambda: sample(Chi("C", 1), library='pymc'))


def test_sampling_gamma_inverse():
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for sampling of gamma inverse.')
    X = GammaInverse("x", 1, 1)
    assert sample(X) in X.pspace.domain.set


def test_lognormal_sampling():
    # Right now, only density function and sampling works
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    for i in range(3):
        X = LogNormal('x', i, 1)
        assert sample(X) in X.pspace.domain.set

    size = 5
    samps = sample(X, size=size)
    for samp in samps:
        assert samp in X.pspace.domain.set


def test_sampling_gaussian_inverse():
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for sampling of Gaussian inverse.')
    X = GaussianInverse("x", 1, 1)
    assert sample(X, library='scipy') in X.pspace.domain.set


def test_prefab_sampling():
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    N = Normal('X', 0, 1)
    L = LogNormal('L', 0, 1)
    E = Exponential('Ex', 1)
    P = Pareto('P', 1, 3)
    W = Weibull('W', 1, 1)
    U = Uniform('U', 0, 1)
    B = Beta('B', 2, 5)
    G = Gamma('G', 1, 3)

    variables = [N, L, E, P, W, U, B, G]
    niter = 10
    size = 5
    for var in variables:
        for _ in range(niter):
            assert sample(var) in var.pspace.domain.set
            samps = sample(var, size=size)
            for samp in samps:
                assert samp in var.pspace.domain.set


def test_sample_continuous():
    z = Symbol('z')
    Z = ContinuousRV(z, exp(-z), set=Interval(0, oo))
    assert density(Z)(-1) == 0

    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    assert sample(Z) in Z.pspace.domain.set
    sym, val = list(Z.pspace.sample().items())[0]
    assert sym == Z and val in Interval(0, oo)

    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0, s1, s2 = [], [], []
                s0 = sample(Z, size=10, library=lib, seed=0)
                s1 = sample(Z, size=10, library=lib, seed=0)
                s2 = sample(Z, size=10, library=lib, seed=1)
                assert all(s0 == s1)
                assert all(s1 != s2)
        except NotImplementedError:
            continue

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_pct_change.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFramePctChange:
    @pytest.mark.parametrize(
        "periods, fill_method, limit, exp",
        [
            (1, "ffill", None, [np.nan, np.nan, np.nan, 1, 1, 1.5, 0, 0]),
            (1, "ffill", 1, [np.nan, np.nan, np.nan, 1, 1, 1.5, 0, np.nan]),
            (1, "bfill", None, [np.nan, 0, 0, 1, 1, 1.5, np.nan, np.nan]),
            (1, "bfill", 1, [np.nan, np.nan, 0, 1, 1, 1.5, np.nan, np.nan]),
            (-1, "ffill", None, [np.nan, np.nan, -0.5, -0.5, -0.6, 0, 0, np.nan]),
            (-1, "ffill", 1, [np.nan, np.nan, -0.5, -0.5, -0.6, 0, np.nan, np.nan]),
            (-1, "bfill", None, [0, 0, -0.5, -0.5, -0.6, np.nan, np.nan, np.nan]),
            (-1, "bfill", 1, [np.nan, 0, -0.5, -0.5, -0.6, np.nan, np.nan, np.nan]),
        ],
    )
    def test_pct_change_with_nas(
        self, periods, fill_method, limit, exp, frame_or_series
    ):
        vals = [np.nan, np.nan, 1, 2, 4, 10, np.nan, np.nan]
        obj = frame_or_series(vals)

        msg = (
            "The 'fill_method' keyword being not None and the 'limit' keyword in "
            f"{type(obj).__name__}.pct_change are deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = obj.pct_change(periods=periods, fill_method=fill_method, limit=limit)
        tm.assert_equal(res, frame_or_series(exp))

    def test_pct_change_numeric(self):
        # GH#11150
        pnl = DataFrame(
            [np.arange(0, 40, 10), np.arange(0, 40, 10), np.arange(0, 40, 10)]
        ).astype(np.float64)
        pnl.iat[1, 0] = np.nan
        pnl.iat[1, 1] = np.nan
        pnl.iat[2, 3] = 60

        msg = (
            "The 'fill_method' keyword being not None and the 'limit' keyword in "
            "DataFrame.pct_change are deprecated"
        )

        for axis in range(2):
            expected = pnl.ffill(axis=axis) / pnl.ffill(axis=axis).shift(axis=axis) - 1

            with tm.assert_produces_warning(FutureWarning, match=msg):
                result = pnl.pct_change(axis=axis, fill_method="pad")
            tm.assert_frame_equal(result, expected)

    def test_pct_change(self, datetime_frame):
        msg = (
            "The 'fill_method' keyword being not None and the 'limit' keyword in "
            "DataFrame.pct_change are deprecated"
        )

        rs = datetime_frame.pct_change(fill_method=None)
        tm.assert_frame_equal(rs, datetime_frame / datetime_frame.shift(1) - 1)

        rs = datetime_frame.pct_change(2)
        filled = datetime_frame.ffill()
        tm.assert_frame_equal(rs, filled / filled.shift(2) - 1)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = datetime_frame.pct_change(fill_method="bfill", limit=1)
        filled = datetime_frame.bfill(limit=1)
        tm.assert_frame_equal(rs, filled / filled.shift(1) - 1)

        rs = datetime_frame.pct_change(freq="5D")
        filled = datetime_frame.ffill()
        tm.assert_frame_equal(
            rs, (filled / filled.shift(freq="5D") - 1).reindex_like(filled)
        )

    def test_pct_change_shift_over_nas(self):
        s = Series([1.0, 1.5, np.nan, 2.5, 3.0])

        df = DataFrame({"a": s, "b": s})

        msg = "The default fill_method='pad' in DataFrame.pct_change is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            chg = df.pct_change()

        expected = Series([np.nan, 0.5, 0.0, 2.5 / 1.5 - 1, 0.2])
        edf = DataFrame({"a": expected, "b": expected})
        tm.assert_frame_equal(chg, edf)

    @pytest.mark.parametrize(
        "freq, periods, fill_method, limit",
        [
            ("5B", 5, None, None),
            ("3B", 3, None, None),
            ("3B", 3, "bfill", None),
            ("7B", 7, "pad", 1),
            ("7B", 7, "bfill", 3),
            ("14B", 14, None, None),
        ],
    )
    def test_pct_change_periods_freq(
        self, datetime_frame, freq, periods, fill_method, limit
    ):
        msg = (
            "The 'fill_method' keyword being not None and the 'limit' keyword in "
            "DataFrame.pct_change are deprecated"
        )

        # GH#7292
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_freq = datetime_frame.pct_change(
                freq=freq, fill_method=fill_method, limit=limit
            )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_periods = datetime_frame.pct_change(
                periods, fill_method=fill_method, limit=limit
            )
        tm.assert_frame_equal(rs_freq, rs_periods)

        empty_ts = DataFrame(index=datetime_frame.index, columns=datetime_frame.columns)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_freq = empty_ts.pct_change(
                freq=freq, fill_method=fill_method, limit=limit
            )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_periods = empty_ts.pct_change(
                periods, fill_method=fill_method, limit=limit
            )
        tm.assert_frame_equal(rs_freq, rs_periods)


@pytest.mark.parametrize("fill_method", ["pad", "ffill", None])
def test_pct_change_with_duplicated_indices(fill_method):
    # GH30463
    data = DataFrame(
        {0: [np.nan, 1, 2, 3, 9, 18], 1: [0, 1, np.nan, 3, 9, 18]}, index=["a", "b"] * 3
    )

    warn = None if fill_method is None else FutureWarning
    msg = (
        "The 'fill_method' keyword being not None and the 'limit' keyword in "
        "DataFrame.pct_change are deprecated"
    )
    with tm.assert_produces_warning(warn, match=msg):
        result = data.pct_change(fill_method=fill_method)

    if fill_method is None:
        second_column = [np.nan, np.inf, np.nan, np.nan, 2.0, 1.0]
    else:
        second_column = [np.nan, np.inf, 0.0, 2.0, 2.0, 1.0]
    expected = DataFrame(
        {0: [np.nan, np.nan, 1.0, 0.5, 2.0, 1.0], 1: second_column},
        index=["a", "b"] * 3,
    )
    tm.assert_frame_equal(result, expected)


def test_pct_change_none_beginning_no_warning():
    # GH#54481
    df = DataFrame(
        [
            [1, None],
            [2, 1],
            [3, 2],
            [4, 3],
            [5, 4],
        ]
    )
    result = df.pct_change()
    expected = DataFrame(
        {0: [np.nan, 1, 0.5, 1 / 3, 0.25], 1: [np.nan, np.nan, 1, 0.5, 1 / 3]}
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_algorithms.py ===
import tempfile
from sympy import log, Min, Max, sqrt
from sympy.core.numbers import Float
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.trigonometric import cos
from sympy.codegen.ast import Assignment, Raise, RuntimeError_, QuotedString
from sympy.codegen.algorithms import newtons_method, newtons_method_function
from sympy.codegen.cfunctions import expm1
from sympy.codegen.fnodes import bind_C
from sympy.codegen.futils import render_as_module as f_module
from sympy.codegen.pyutils import render_as_module as py_module
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c, has_fortran
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, raises, skip_under_pyodide

cython = import_module('cython')
wurlitzer = import_module('wurlitzer')

def test_newtons_method():
    x, dx, atol = symbols('x dx atol')
    expr = cos(x) - x**3
    algo = newtons_method(expr, x, atol, dx)
    assert algo.has(Assignment(dx, -expr/expr.diff(x)))


@may_xfail
def test_newtons_method_function__ccode():
    x = Symbol('x', real=True)
    expr = cos(x) - x**3
    func = newtons_method_function(expr, x)

    if not cython:
        skip("cython not installed.")
    if not has_c():
        skip("No C compiler found.")

    compile_kw = {"std": 'c99'}
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('newton.c', ('#include <math.h>\n'
                          '#include <stdio.h>\n') + ccode(func)),
            ('_newton.pyx', ("#cython: language_level={}\n".format("3") +
                             "cdef extern double newton(double)\n"
                             "def py_newton(x):\n"
                             "    return newton(x)\n"))
        ], build_dir=folder, compile_kwargs=compile_kw)
        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12


@may_xfail
def test_newtons_method_function__fcode():
    x = Symbol('x', real=True)
    expr = cos(x) - x**3
    func = newtons_method_function(expr, x, attrs=[bind_C(name='newton')])

    if not cython:
        skip("cython not installed.")
    if not has_fortran():
        skip("No Fortran compiler found.")

    f_mod = f_module([func], 'mod_newton')
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('newton.f90', f_mod),
            ('_newton.pyx', ("#cython: language_level={}\n".format("3") +
                             "cdef extern double newton(double*)\n"
                             "def py_newton(double x):\n"
                             "    return newton(&x)\n"))
        ], build_dir=folder)
        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12


def test_newtons_method_function__pycode():
    x = Symbol('x', real=True)
    expr = cos(x) - x**3
    func = newtons_method_function(expr, x)
    py_mod = py_module(func)
    namespace = {}
    exec(py_mod, namespace, namespace)
    res = eval('newton(0.5)', namespace)
    assert abs(res - 0.865474033102) < 1e-12


@may_xfail
@skip_under_pyodide("Emscripten does not support process spawning")
def test_newtons_method_function__ccode_parameters():
    args = x, A, k, p = symbols('x A k p')
    expr = A*cos(k*x) - p*x**3
    raises(ValueError, lambda: newtons_method_function(expr, x))
    use_wurlitzer = wurlitzer

    func = newtons_method_function(expr, x, args, debug=use_wurlitzer)

    if not has_c():
        skip("No C compiler found.")
    if not cython:
        skip("cython not installed.")

    compile_kw = {"std": 'c99'}
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('newton_par.c', ('#include <math.h>\n'
                          '#include <stdio.h>\n') + ccode(func)),
            ('_newton_par.pyx', ("#cython: language_level={}\n".format("3") +
                                 "cdef extern double newton(double, double, double, double)\n"
                             "def py_newton(x, A=1, k=1, p=1):\n"
                             "    return newton(x, A, k, p)\n"))
        ], compile_kwargs=compile_kw, build_dir=folder)

        if use_wurlitzer:
            with wurlitzer.pipes() as (out, err):
                result = mod.py_newton(0.5)
        else:
            result = mod.py_newton(0.5)

        assert abs(result - 0.865474033102) < 1e-12

        if not use_wurlitzer:
            skip("C-level output only tested when package 'wurlitzer' is available.")

        out, err = out.read(), err.read()
        assert err == ''
        assert out == """\
x=         0.5
x=      1.1121 d_x=     0.61214
x=     0.90967 d_x=    -0.20247
x=     0.86726 d_x=   -0.042409
x=     0.86548 d_x=  -0.0017867
x=     0.86547 d_x= -3.1022e-06
x=     0.86547 d_x= -9.3421e-12
x=     0.86547 d_x=  3.6902e-17
"""  # try to run tests with LC_ALL=C if this assertion fails


def test_newtons_method_function__rtol_cse_nan():
    a, b, c, N_geo, N_tot = symbols('a b c N_geo N_tot', real=True, nonnegative=True)
    i = Symbol('i', integer=True, nonnegative=True)
    N_ari = N_tot - N_geo - 1
    delta_ari = (c-b)/N_ari
    ln_delta_geo = log(b) + log(-expm1((log(a)-log(b))/N_geo))
    eqb_log = ln_delta_geo - log(delta_ari)

    def _clamp(low, expr, high):
        return Min(Max(low, expr), high)

    meth_kw = {
        'clamped_newton': {'delta_fn': lambda e, x: _clamp(
            (sqrt(a*x)-x)*0.99,
            -e/e.diff(x),
            (sqrt(c*x)-x)*0.99
        )},
        'halley': {'delta_fn': lambda e, x: (-2*(e*e.diff(x))/(2*e.diff(x)**2 - e*e.diff(x, 2)))},
        'halley_alt': {'delta_fn': lambda e, x: (-e/e.diff(x)/(1-e/e.diff(x)*e.diff(x,2)/2/e.diff(x)))},
    }
    args = eqb_log, b
    for use_cse in [False, True]:
        kwargs = {
            'params': (b, a, c, N_geo, N_tot), 'itermax': 60, 'debug': True, 'cse': use_cse,
            'counter': i, 'atol': 1e-100, 'rtol': 2e-16, 'bounds': (a,c),
            'handle_nan': Raise(RuntimeError_(QuotedString("encountered NaN.")))
        }
        func = {k: newtons_method_function(*args, func_name=f"{k}_b", **dict(kwargs, **kw)) for k, kw in meth_kw.items()}
        py_mod = {k: py_module(v) for k, v in func.items()}
        namespace = {}
        root_find_b = {}
        for k, v in py_mod.items():
            ns = namespace[k] = {}
            exec(v, ns, ns)
            root_find_b[k] = ns[f'{k}_b']
        ref = Float('13.2261515064168768938151923226496')
        reftol = {'clamped_newton': 2e-16, 'halley': 2e-16, 'halley_alt': 3e-16}
        guess = 4.0
        for meth, func in root_find_b.items():
            result = func(guess, 1e-2, 1e2, 50, 100)
            req = ref*reftol[meth]
            if use_cse:
                req *= 2
            assert abs(result - ref) < req

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_eigen.py ===
#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpmath import mp
from mpmath import libmp

xrange = libmp.backend.xrange

def run_hessenberg(A, verbose = 0):
    if verbose > 1:
        print("original matrix (hessenberg):\n", A)

    n = A.rows

    Q, H = mp.hessenberg(A)

    if verbose > 1:
        print("Q:\n",Q)
        print("H:\n",H)

    B = Q * H * Q.transpose_conj()

    eps = mp.exp(0.8 * mp.log(mp.eps))

    err0 = 0
    for x in xrange(n):
        for y in xrange(n):
            err0 += abs(A[y,x] - B[y,x])
    err0 /= n * n

    err1 = 0
    for x in xrange(n):
        for y in xrange(x + 2, n):
            err1 += abs(H[y,x])

    if verbose > 0:
        print("difference (H):", err0, err1)

    if verbose > 1:
        print("B:\n", B)

    assert err0 < eps
    assert err1 == 0


def run_schur(A, verbose = 0):
    if verbose > 1:
        print("original matrix (schur):\n", A)

    n = A.rows

    Q, R = mp.schur(A)

    if verbose > 1:
        print("Q:\n", Q)
        print("R:\n", R)

    B = Q * R * Q.transpose_conj()
    C = Q * Q.transpose_conj()

    eps = mp.exp(0.8 * mp.log(mp.eps))

    err0 = 0
    for x in xrange(n):
        for y in xrange(n):
            err0 += abs(A[y,x] - B[y,x])
    err0 /= n * n

    err1 = 0
    for x in xrange(n):
        for y in xrange(n):
            if x == y:
                C[y,x] -= 1
            err1 += abs(C[y,x])
    err1 /= n * n

    err2 = 0
    for x in xrange(n):
        for y in xrange(x + 1, n):
            err2 += abs(R[y,x])

    if verbose > 0:
        print("difference (S):", err0, err1, err2)

    if verbose > 1:
        print("B:\n", B)

    assert err0 < eps
    assert err1 < eps
    assert err2 == 0

def run_eig(A, verbose = 0):
    if verbose > 1:
        print("original matrix (eig):\n", A)

    n = A.rows

    E, EL, ER = mp.eig(A, left = True, right = True)

    if verbose > 1:
        print("E:\n", E)
        print("EL:\n", EL)
        print("ER:\n", ER)

    eps = mp.exp(0.8 * mp.log(mp.eps))

    err0 = 0
    for i in xrange(n):
        B = A * ER[:,i] - E[i] * ER[:,i]
        err0 = max(err0, mp.mnorm(B))

        B = EL[i,:] * A - EL[i,:] * E[i]
        err0 = max(err0, mp.mnorm(B))

    err0 /= n * n

    if verbose > 0:
        print("difference (E):", err0)

    assert err0 < eps

#####################

def test_eig_dyn():
    v = 0
    for i in xrange(5):
        n = 1 + int(mp.rand() * 5)
        if mp.rand() > 0.5:
            # real
            A = 2 * mp.randmatrix(n, n) - 1
            if mp.rand() > 0.5:
                A *= 10
                for x in xrange(n):
                    for y in xrange(n):
                        A[x,y] = int(A[x,y])
        else:
            A = (2 * mp.randmatrix(n, n) - 1) + 1j * (2 * mp.randmatrix(n, n) - 1)
            if mp.rand() > 0.5:
                A *= 10
                for x in xrange(n):
                    for y in xrange(n):
                        A[x,y] = int(mp.re(A[x,y])) + 1j * int(mp.im(A[x,y]))

        run_hessenberg(A, verbose = v)
        run_schur(A, verbose = v)
        run_eig(A, verbose = v)

def test_eig():
    v = 0
    AS = []

    A = mp.matrix([[2, 1, 0],  # jordan block of size 3
                   [0, 2, 1],
                   [0, 0, 2]])
    AS.append(A)
    AS.append(A.transpose())

    A = mp.matrix([[2, 0, 0],  # jordan block of size 2
                   [0, 2, 1],
                   [0, 0, 2]])
    AS.append(A)
    AS.append(A.transpose())

    A = mp.matrix([[2, 0, 1],  # jordan block of size 2
                   [0, 2, 0],
                   [0, 0, 2]])
    AS.append(A)
    AS.append(A.transpose())

    A=  mp.matrix([[0, 0, 1],  # cyclic
                   [1, 0, 0],
                   [0, 1, 0]])
    AS.append(A)
    AS.append(A.transpose())

    for A in AS:
        run_hessenberg(A, verbose = v)
        run_schur(A, verbose = v)
        run_eig(A, verbose = v)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_to_csv.py ===
from datetime import datetime
from io import StringIO

import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm

from pandas.io.common import get_handle


class TestSeriesToCSV:
    def read_csv(self, path, **kwargs):
        params = {"index_col": 0, "header": None}
        params.update(**kwargs)

        header = params.get("header")
        out = pd.read_csv(path, **params).squeeze("columns")

        if header is None:
            out.name = out.index.name = None

        return out

    def test_from_csv(self, datetime_series, string_series):
        # freq doesn't round-trip
        datetime_series.index = datetime_series.index._with_freq(None)

        with tm.ensure_clean() as path:
            datetime_series.to_csv(path, header=False)
            ts = self.read_csv(path, parse_dates=True)
            tm.assert_series_equal(datetime_series, ts, check_names=False)

            assert ts.name is None
            assert ts.index.name is None

            # see gh-10483
            datetime_series.to_csv(path, header=True)
            ts_h = self.read_csv(path, header=0)
            assert ts_h.name == "ts"

            string_series.to_csv(path, header=False)
            series = self.read_csv(path)
            tm.assert_series_equal(string_series, series, check_names=False)

            assert series.name is None
            assert series.index.name is None

            string_series.to_csv(path, header=True)
            series_h = self.read_csv(path, header=0)
            assert series_h.name == "series"

            with open(path, "w", encoding="utf-8") as outfile:
                outfile.write("1998-01-01|1.0\n1999-01-01|2.0")

            series = self.read_csv(path, sep="|", parse_dates=True)
            check_series = Series(
                {datetime(1998, 1, 1): 1.0, datetime(1999, 1, 1): 2.0}
            )
            tm.assert_series_equal(check_series, series)

            series = self.read_csv(path, sep="|", parse_dates=False)
            check_series = Series({"1998-01-01": 1.0, "1999-01-01": 2.0})
            tm.assert_series_equal(check_series, series)

    def test_to_csv(self, datetime_series):
        with tm.ensure_clean() as path:
            datetime_series.to_csv(path, header=False)

            with open(path, newline=None, encoding="utf-8") as f:
                lines = f.readlines()
            assert lines[1] != "\n"

            datetime_series.to_csv(path, index=False, header=False)
            arr = np.loadtxt(path)
            tm.assert_almost_equal(arr, datetime_series.values)

    def test_to_csv_unicode_index(self):
        buf = StringIO()
        s = Series(["\u05d0", "d2"], index=["\u05d0", "\u05d1"])

        s.to_csv(buf, encoding="UTF-8", header=False)
        buf.seek(0)

        s2 = self.read_csv(buf, index_col=0, encoding="UTF-8")
        tm.assert_series_equal(s, s2)

    def test_to_csv_float_format(self):
        with tm.ensure_clean() as filename:
            ser = Series([0.123456, 0.234567, 0.567567])
            ser.to_csv(filename, float_format="%.2f", header=False)

            rs = self.read_csv(filename)
            xp = Series([0.12, 0.23, 0.57])
            tm.assert_series_equal(rs, xp)

    def test_to_csv_list_entries(self):
        s = Series(["jack and jill", "jesse and frank"])

        split = s.str.split(r"\s+and\s+")

        buf = StringIO()
        split.to_csv(buf, header=False)

    def test_to_csv_path_is_none(self):
        # GH 8215
        # Series.to_csv() was returning None, inconsistent with
        # DataFrame.to_csv() which returned string
        s = Series([1, 2, 3])
        csv_str = s.to_csv(path_or_buf=None, header=False)
        assert isinstance(csv_str, str)

    @pytest.mark.parametrize(
        "s,encoding",
        [
            (
                Series([0.123456, 0.234567, 0.567567], index=["A", "B", "C"], name="X"),
                None,
            ),
            # GH 21241, 21118
            (Series(["abc", "def", "ghi"], name="X"), "ascii"),
            (Series(["123", "你好", "世界"], name="中文"), "gb2312"),
            (
                Series(["123", "Γειά σου", "Κόσμε"], name="Ελληνικά"),  # noqa: RUF001
                "cp737",
            ),
        ],
    )
    def test_to_csv_compression(self, s, encoding, compression):
        with tm.ensure_clean() as filename:
            s.to_csv(filename, compression=compression, encoding=encoding, header=True)
            # test the round trip - to_csv -> read_csv
            result = pd.read_csv(
                filename,
                compression=compression,
                encoding=encoding,
                index_col=0,
            ).squeeze("columns")
            tm.assert_series_equal(s, result)

            # test the round trip using file handle - to_csv -> read_csv
            with get_handle(
                filename, "w", compression=compression, encoding=encoding
            ) as handles:
                s.to_csv(handles.handle, encoding=encoding, header=True)

            result = pd.read_csv(
                filename,
                compression=compression,
                encoding=encoding,
                index_col=0,
            ).squeeze("columns")
            tm.assert_series_equal(s, result)

            # explicitly ensure file was compressed
            with tm.decompress_file(filename, compression) as fh:
                text = fh.read().decode(encoding or "utf8")
                assert s.name in text

            with tm.decompress_file(filename, compression) as fh:
                tm.assert_series_equal(
                    s,
                    pd.read_csv(fh, index_col=0, encoding=encoding).squeeze("columns"),
                )

    def test_to_csv_interval_index(self, using_infer_string):
        # GH 28210
        s = Series(["foo", "bar", "baz"], index=pd.interval_range(0, 3))

        with tm.ensure_clean("__tmp_to_csv_interval_index__.csv") as path:
            s.to_csv(path, header=False)
            result = self.read_csv(path, index_col=0)

            # can't roundtrip intervalindex via read_csv so check string repr (GH 23595)
            expected = s
            expected.index = expected.index.astype("str")
            tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_constantsimp.py ===
"""
If the arbitrary constant class from issue 4435 is ever implemented, this
should serve as a set of test cases.
"""

from sympy.core.function import Function
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.integrals.integrals import Integral
from sympy.solvers.ode.ode import constantsimp, constant_renumber
from sympy.testing.pytest import XFAIL


x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
u2 = Symbol('u2')
_a = Symbol('_a')
C1 = Symbol('C1')
C2 = Symbol('C2')
C3 = Symbol('C3')
f = Function('f')


def test_constant_mul():
    # We want C1 (Constant) below to absorb the y's, but not the x's
    assert constant_renumber(constantsimp(y*C1, [C1])) == C1*y
    assert constant_renumber(constantsimp(C1*y, [C1])) == C1*y
    assert constant_renumber(constantsimp(x*C1, [C1])) == x*C1
    assert constant_renumber(constantsimp(C1*x, [C1])) == x*C1
    assert constant_renumber(constantsimp(2*C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1*2, [C1])) == C1
    assert constant_renumber(constantsimp(y*C1*x, [C1, y])) == C1*x
    assert constant_renumber(constantsimp(x*y*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(y*x*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(C1*x*y, [C1, y])) == C1*x
    assert constant_renumber(constantsimp(x*C1*y, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(C1*y*(y + 1), [C1])) == C1*y*(y+1)
    assert constant_renumber(constantsimp(y*C1*(y + 1), [C1])) == C1*y*(y+1)
    assert constant_renumber(constantsimp(x*(y*C1), [C1])) == x*y*C1
    assert constant_renumber(constantsimp(x*(C1*y), [C1])) == x*y*C1
    assert constant_renumber(constantsimp(C1*(x*y), [C1, y])) == C1*x
    assert constant_renumber(constantsimp((x*y)*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp((y*x)*C1, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(y*(y + 1)*C1, [C1, y])) == C1
    assert constant_renumber(constantsimp((C1*x)*y, [C1, y])) == C1*x
    assert constant_renumber(constantsimp(y*(x*C1), [C1, y])) == x*C1
    assert constant_renumber(constantsimp((x*C1)*y, [C1, y])) == x*C1
    assert constant_renumber(constantsimp(C1*x*y*x*y*2, [C1, y])) == C1*x**2
    assert constant_renumber(constantsimp(C1*x*y*z, [C1, y, z])) == C1*x
    assert constant_renumber(constantsimp(C1*x*y**2*sin(z), [C1, y, z])) == C1*x
    assert constant_renumber(constantsimp(C1*C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1*C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2*C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1*C1*C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1*x*2**x, [C1])) == C1*x*2**x

def test_constant_add():
    assert constant_renumber(constantsimp(C1 + C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 + 2, [C1])) == C1
    assert constant_renumber(constantsimp(2 + C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 + y, [C1, y])) == C1
    assert constant_renumber(constantsimp(C1 + x, [C1])) == C1 + x
    assert constant_renumber(constantsimp(C1 + C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 + C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2 + C1, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1 + C2 + C1, [C1, C2])) == C1


def test_constant_power_as_base():
    assert constant_renumber(constantsimp(C1**C1, [C1])) == C1
    assert constant_renumber(constantsimp(Pow(C1, C1), [C1])) == C1
    assert constant_renumber(constantsimp(C1**C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1**C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2**C1, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2**C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1**y, [C1, y])) == C1
    assert constant_renumber(constantsimp(C1**x, [C1])) == C1**x
    assert constant_renumber(constantsimp(C1**2, [C1])) == C1
    assert constant_renumber(
        constantsimp(C1**(x*y), [C1])) == C1**(x*y)


def test_constant_power_as_exp():
    assert constant_renumber(constantsimp(x**C1, [C1])) == x**C1
    assert constant_renumber(constantsimp(y**C1, [C1, y])) == C1
    assert constant_renumber(constantsimp(x**y**C1, [C1, y])) == x**C1
    assert constant_renumber(
        constantsimp((x**y)**C1, [C1])) == (x**y)**C1
    assert constant_renumber(
        constantsimp(x**(y**C1), [C1, y])) == x**C1
    assert constant_renumber(constantsimp(x**C1**y, [C1, y])) == x**C1
    assert constant_renumber(
        constantsimp(x**(C1**y), [C1, y])) == x**C1
    assert constant_renumber(
        constantsimp((x**C1)**y, [C1])) == (x**C1)**y
    assert constant_renumber(constantsimp(2**C1, [C1])) == C1
    assert constant_renumber(constantsimp(S(2)**C1, [C1])) == C1
    assert constant_renumber(constantsimp(exp(C1), [C1])) == C1
    assert constant_renumber(
        constantsimp(exp(C1 + x), [C1])) == C1*exp(x)
    assert constant_renumber(constantsimp(Pow(2, C1), [C1])) == C1


def test_constant_function():
    assert constant_renumber(constantsimp(sin(C1), [C1])) == C1
    assert constant_renumber(constantsimp(f(C1), [C1])) == C1
    assert constant_renumber(constantsimp(f(C1, C1), [C1])) == C1
    assert constant_renumber(constantsimp(f(C1, C2), [C1, C2])) == C1
    assert constant_renumber(constantsimp(f(C2, C1), [C1, C2])) == C1
    assert constant_renumber(constantsimp(f(C2, C2), [C1, C2])) == C1
    assert constant_renumber(
        constantsimp(f(C1, x), [C1])) == f(C1, x)
    assert constant_renumber(constantsimp(f(C1, y), [C1, y])) == C1
    assert constant_renumber(constantsimp(f(y, C1), [C1, y])) == C1
    assert constant_renumber(constantsimp(f(C1, y, C2), [C1, C2, y])) == C1


def test_constant_function_multiple():
    # The rules to not renumber in this case would be too complicated, and
    # dsolve is not likely to ever encounter anything remotely like this.
    assert constant_renumber(
        constantsimp(f(C1, C1, x), [C1])) == f(C1, C1, x)


def test_constant_multiple():
    assert constant_renumber(constantsimp(C1*2 + 2, [C1])) == C1
    assert constant_renumber(constantsimp(x*2/C1, [C1])) == C1*x
    assert constant_renumber(constantsimp(C1**2*2 + 2, [C1])) == C1
    assert constant_renumber(
        constantsimp(sin(2*C1) + x + sqrt(2), [C1])) == C1 + x
    assert constant_renumber(constantsimp(2*C1 + C2, [C1, C2])) == C1

def test_constant_repeated():
    assert C1 + C1*x == constant_renumber( C1 + C1*x)

def test_ode_solutions():
    # only a few examples here, the rest will be tested in the actual dsolve tests
    assert constant_renumber(constantsimp(C1*exp(2*x) + exp(x)*(C2 + C3), [C1, C2, C3])) == \
        constant_renumber(C1*exp(x) + C2*exp(2*x))
    assert constant_renumber(
        constantsimp(Eq(f(x), I*C1*sinh(x/3) + C2*cosh(x/3)), [C1, C2])
        ) == constant_renumber(Eq(f(x), C1*sinh(x/3) + C2*cosh(x/3)))
    assert constant_renumber(constantsimp(Eq(f(x), acos((-C1)/cos(x))), [C1])) == \
        Eq(f(x), acos(C1/cos(x)))
    assert constant_renumber(
        constantsimp(Eq(log(f(x)/C1) + 2*exp(x/f(x)), 0), [C1])
        ) == Eq(log(C1*f(x)) + 2*exp(x/f(x)), 0)
    assert constant_renumber(constantsimp(Eq(log(x*sqrt(2)*sqrt(1/x)*sqrt(f(x))
        /C1) + x**2/(2*f(x)**2), 0), [C1])) == \
        Eq(log(C1*sqrt(x)*sqrt(f(x))) + x**2/(2*f(x)**2), 0)
    assert constant_renumber(constantsimp(Eq(-exp(-f(x)/x)*sin(f(x)/x)/2 + log(x/C1) -
        cos(f(x)/x)*exp(-f(x)/x)/2, 0), [C1])) == \
        Eq(-exp(-f(x)/x)*sin(f(x)/x)/2 + log(C1*x) - cos(f(x)/x)*
           exp(-f(x)/x)/2, 0)
    assert constant_renumber(constantsimp(Eq(-Integral(-1/(sqrt(1 - u2**2)*u2),
        (u2, _a, x/f(x))) + log(f(x)/C1), 0), [C1])) == \
        Eq(-Integral(-1/(u2*sqrt(1 - u2**2)), (u2, _a, x/f(x))) +
        log(C1*f(x)), 0)
    assert [constantsimp(i, [C1]) for i in [Eq(f(x), sqrt(-C1*x + x**2)), Eq(f(x), -sqrt(-C1*x + x**2))]] == \
        [Eq(f(x), sqrt(x*(C1 + x))), Eq(f(x), -sqrt(x*(C1 + x)))]


@XFAIL
def test_nonlocal_simplification():
    assert constantsimp(C1 + C2+x*C2, [C1, C2]) == C1 + C2*x


def test_constant_Eq():
    # C1 on the rhs is well-tested, but the lhs is only tested here
    assert constantsimp(Eq(C1, 3 + f(x)*x), [C1]) == Eq(x*f(x), C1)
    assert constantsimp(Eq(C1, 3 * f(x)*x), [C1]) == Eq(f(x)*x, C1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_enumerative.py ===
import string
from itertools import zip_longest

from sympy.utilities.enumerative import (
    list_visitor,
    MultisetPartitionTraverser,
    multiset_partitions_taocp
    )
from sympy.utilities.iterables import _set_partitions

# first some functions only useful as test scaffolding - these provide
# straightforward, but slow reference implementations against which to
# compare the real versions, and also a comparison to verify that
# different versions are giving identical results.

def part_range_filter(partition_iterator, lb, ub):
    """
    Filters (on the number of parts) a multiset partition enumeration

    Arguments
    =========

    lb, and ub are a range (in the Python slice sense) on the lpart
    variable returned from a multiset partition enumeration.  Recall
    that lpart is 0-based (it points to the topmost part on the part
    stack), so if you want to return parts of sizes 2,3,4,5 you would
    use lb=1 and ub=5.
    """
    for state in partition_iterator:
        f, lpart, pstack = state
        if lpart >= lb and lpart < ub:
            yield state

def multiset_partitions_baseline(multiplicities, components):
    """Enumerates partitions of a multiset

    Parameters
    ==========

    multiplicities
         list of integer multiplicities of the components of the multiset.

    components
         the components (elements) themselves

    Returns
    =======

    Set of partitions.  Each partition is tuple of parts, and each
    part is a tuple of components (with repeats to indicate
    multiplicity)

    Notes
    =====

    Multiset partitions can be created as equivalence classes of set
    partitions, and this function does just that.  This approach is
    slow and memory intensive compared to the more advanced algorithms
    available, but the code is simple and easy to understand.  Hence
    this routine is strictly for testing -- to provide a
    straightforward baseline against which to regress the production
    versions.  (This code is a simplified version of an earlier
    production implementation.)
    """

    canon = []                  # list of components with repeats
    for ct, elem in zip(multiplicities, components):
        canon.extend([elem]*ct)

    # accumulate the multiset partitions in a set to eliminate dups
    cache = set()
    n = len(canon)
    for nc, q in _set_partitions(n):
        rv = [[] for i in range(nc)]
        for i in range(n):
            rv[q[i]].append(canon[i])
        canonical = tuple(
            sorted([tuple(p) for p in rv]))
        cache.add(canonical)
    return cache


def compare_multiset_w_baseline(multiplicities):
    """
    Enumerates the partitions of multiset with AOCP algorithm and
    baseline implementation, and compare the results.

    """
    letters = string.ascii_lowercase
    bl_partitions = multiset_partitions_baseline(multiplicities, letters)

    # The partitions returned by the different algorithms may have
    # their parts in different orders.  Also, they generate partitions
    # in different orders.  Hence the sorting, and set comparison.

    aocp_partitions = set()
    for state in multiset_partitions_taocp(multiplicities):
        p1 = tuple(sorted(
                [tuple(p) for p in list_visitor(state, letters)]))
        aocp_partitions.add(p1)

    assert bl_partitions == aocp_partitions

def compare_multiset_states(s1, s2):
    """compare for equality two instances of multiset partition states

    This is useful for comparing different versions of the algorithm
    to verify correctness."""
    # Comparison is physical, the only use of semantics is to ignore
    # trash off the top of the stack.
    f1, lpart1, pstack1 = s1
    f2, lpart2, pstack2 = s2

    if (lpart1 == lpart2) and (f1[0:lpart1+1] == f2[0:lpart2+1]):
        if pstack1[0:f1[lpart1+1]] == pstack2[0:f2[lpart2+1]]:
            return True
    return False

def test_multiset_partitions_taocp():
    """Compares the output of multiset_partitions_taocp with a baseline
    (set partition based) implementation."""

    # Test cases should not be too large, since the baseline
    # implementation is fairly slow.
    multiplicities = [2,2]
    compare_multiset_w_baseline(multiplicities)

    multiplicities = [4,3,1]
    compare_multiset_w_baseline(multiplicities)

def test_multiset_partitions_versions():
    """Compares Knuth-based versions of multiset_partitions"""
    multiplicities = [5,2,2,1]
    m = MultisetPartitionTraverser()
    for s1, s2 in zip_longest(m.enum_all(multiplicities),
                              multiset_partitions_taocp(multiplicities)):
        assert compare_multiset_states(s1, s2)

def subrange_exercise(mult, lb, ub):
    """Compare filter-based and more optimized subrange implementations

    Helper for tests, called with both small and larger multisets.
    """
    m = MultisetPartitionTraverser()
    assert m.count_partitions(mult) == \
        m.count_partitions_slow(mult)

    # Note - multiple traversals from the same
    # MultisetPartitionTraverser object cannot execute at the same
    # time, hence make several instances here.
    ma = MultisetPartitionTraverser()
    mc = MultisetPartitionTraverser()
    md = MultisetPartitionTraverser()

    #  Several paths to compute just the size two partitions
    a_it = ma.enum_range(mult, lb, ub)
    b_it = part_range_filter(multiset_partitions_taocp(mult), lb, ub)
    c_it = part_range_filter(mc.enum_small(mult, ub), lb, sum(mult))
    d_it = part_range_filter(md.enum_large(mult, lb), 0, ub)

    for sa, sb, sc, sd in zip_longest(a_it, b_it, c_it, d_it):
        assert compare_multiset_states(sa, sb)
        assert compare_multiset_states(sa, sc)
        assert compare_multiset_states(sa, sd)

def test_subrange():
    # Quick, but doesn't hit some of the corner cases
    mult = [4,4,2,1] # mississippi
    lb = 1
    ub = 2
    subrange_exercise(mult, lb, ub)


def test_subrange_large():
    # takes a second or so, depending on cpu, Python version, etc.
    mult = [6,3,2,1]
    lb = 4
    ub = 7
    subrange_exercise(mult, lb, ub)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_autolev.py ===
import os

from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.parsing.autolev import parse_autolev

antlr4 = import_module("antlr4")

if not antlr4:
    disabled = True

FILE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.realpath(__file__))))


def _test_examples(in_filename, out_filename, test_name=""):

    in_file_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                                in_filename)
    correct_file_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                                     out_filename)
    with open(in_file_path) as f:
        generated_code = parse_autolev(f, include_numeric=True)

    with open(correct_file_path) as f:
        for idx, line1 in enumerate(f):
            if line1.startswith("#"):
                break
            try:
                line2 = generated_code.split('\n')[idx]
                assert line1.rstrip() == line2.rstrip()
            except Exception:
                msg = 'mismatch in ' + test_name + ' in line no: {0}'
                raise AssertionError(msg.format(idx+1))


def test_rule_tests():

    l = ["ruletest1", "ruletest2", "ruletest3", "ruletest4", "ruletest5",
         "ruletest6", "ruletest7", "ruletest8", "ruletest9", "ruletest10",
         "ruletest11", "ruletest12"]

    for i in l:
        in_filepath = i + ".al"
        out_filepath = i + ".py"
        _test_examples(in_filepath, out_filepath, i)


def test_pydy_examples():

    l = ["mass_spring_damper", "chaos_pendulum", "double_pendulum",
         "non_min_pendulum"]

    for i in l:
        in_filepath = os.path.join("pydy-example-repo", i + ".al")
        out_filepath = os.path.join("pydy-example-repo", i + ".py")
        _test_examples(in_filepath, out_filepath, i)


def test_autolev_tutorial():

    dir_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                            'autolev-tutorial')

    if os.path.isdir(dir_path):
        l = ["tutor1", "tutor2", "tutor3", "tutor4", "tutor5", "tutor6",
             "tutor7"]
        for i in l:
            in_filepath = os.path.join("autolev-tutorial", i + ".al")
            out_filepath = os.path.join("autolev-tutorial", i + ".py")
            _test_examples(in_filepath, out_filepath, i)


def test_dynamics_online():

    dir_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                            'dynamics-online')

    if os.path.isdir(dir_path):
        ch1 = ["1-4", "1-5", "1-6", "1-7", "1-8", "1-9_1", "1-9_2", "1-9_3"]
        ch2 = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9",
               "circular"]
        ch3 = ["3-1_1", "3-1_2", "3-2_1", "3-2_2", "3-2_3", "3-2_4", "3-2_5",
               "3-3"]
        ch4 = ["4-1_1", "4-2_1", "4-4_1", "4-4_2", "4-5_1", "4-5_2"]
        chapters = [(ch1, "ch1"), (ch2, "ch2"), (ch3, "ch3"), (ch4, "ch4")]
        for ch, name in chapters:
            for i in ch:
                in_filepath = os.path.join("dynamics-online", name, i + ".al")
                out_filepath = os.path.join("dynamics-online", name, i + ".py")
                _test_examples(in_filepath, out_filepath, i)


def test_output_01():
    """Autolev example calculates the position, velocity, and acceleration of a
    point and expresses in a single reference frame::

          (1) FRAMES C,D,F
          (2) VARIABLES FD'',DC''
          (3) CONSTANTS R,L
          (4) POINTS O,E
          (5) SIMPROT(F,D,1,FD)
       -> (6) F_D = [1, 0, 0; 0, COS(FD), -SIN(FD); 0, SIN(FD), COS(FD)]
          (7) SIMPROT(D,C,2,DC)
       -> (8) D_C = [COS(DC), 0, SIN(DC); 0, 1, 0; -SIN(DC), 0, COS(DC)]
          (9) W_C_F> = EXPRESS(W_C_F>, F)
       -> (10) W_C_F> = FD'*F1> + COS(FD)*DC'*F2> + SIN(FD)*DC'*F3>
          (11) P_O_E>=R*D2>-L*C1>
          (12) P_O_E>=EXPRESS(P_O_E>, D)
       -> (13) P_O_E> = -L*COS(DC)*D1> + R*D2> + L*SIN(DC)*D3>
          (14) V_E_F>=EXPRESS(DT(P_O_E>,F),D)
       -> (15) V_E_F> = L*SIN(DC)*DC'*D1> - L*SIN(DC)*FD'*D2> + (R*FD'+L*COS(DC)*DC')*D3>
          (16) A_E_F>=EXPRESS(DT(V_E_F>,F),D)
       -> (17) A_E_F> = L*(COS(DC)*DC'^2+SIN(DC)*DC'')*D1> + (-R*FD'^2-2*L*COS(DC)*DC'*FD'-L*SIN(DC)*FD'')*D2> + (R*FD''+L*COS(DC)*DC''-L*SIN(DC)*DC'^2-L*SIN(DC)*FD'^2)*D3>

    """

    if not antlr4:
        skip('Test skipped: antlr4 is not installed.')

    autolev_input = """\
FRAMES C,D,F
VARIABLES FD'',DC''
CONSTANTS R,L
POINTS O,E
SIMPROT(F,D,1,FD)
SIMPROT(D,C,2,DC)
W_C_F>=EXPRESS(W_C_F>,F)
P_O_E>=R*D2>-L*C1>
P_O_E>=EXPRESS(P_O_E>,D)
V_E_F>=EXPRESS(DT(P_O_E>,F),D)
A_E_F>=EXPRESS(DT(V_E_F>,F),D)\
"""

    sympy_input = parse_autolev(autolev_input)

    g = {}
    l = {}
    exec(sympy_input, g, l)

    w_c_f = l['frame_c'].ang_vel_in(l['frame_f'])
    # P_O_E> means "the position of point E wrt to point O"
    p_o_e = l['point_e'].pos_from(l['point_o'])
    v_e_f = l['point_e'].vel(l['frame_f'])
    a_e_f = l['point_e'].acc(l['frame_f'])

    # NOTE : The Autolev outputs above were manually transformed into
    # equivalent SymPy physics vector expressions. Would be nice to automate
    # this transformation.
    expected_w_c_f = (l['fd'].diff()*l['frame_f'].x +
                      cos(l['fd'])*l['dc'].diff()*l['frame_f'].y +
                      sin(l['fd'])*l['dc'].diff()*l['frame_f'].z)

    assert (w_c_f - expected_w_c_f).simplify() == 0

    expected_p_o_e = (-l['l']*cos(l['dc'])*l['frame_d'].x +
                      l['r']*l['frame_d'].y +
                      l['l']*sin(l['dc'])*l['frame_d'].z)

    assert (p_o_e - expected_p_o_e).simplify() == 0

    expected_v_e_f = (l['l']*sin(l['dc'])*l['dc'].diff()*l['frame_d'].x -
                      l['l']*sin(l['dc'])*l['fd'].diff()*l['frame_d'].y +
                      (l['r']*l['fd'].diff() +
                       l['l']*cos(l['dc'])*l['dc'].diff())*l['frame_d'].z)
    assert (v_e_f - expected_v_e_f).simplify() == 0

    expected_a_e_f = (l['l']*(cos(l['dc'])*l['dc'].diff()**2 +
                              sin(l['dc'])*l['dc'].diff().diff())*l['frame_d'].x +
                      (-l['r']*l['fd'].diff()**2 -
                       2*l['l']*cos(l['dc'])*l['dc'].diff()*l['fd'].diff() -
                       l['l']*sin(l['dc'])*l['fd'].diff().diff())*l['frame_d'].y +
                      (l['r']*l['fd'].diff().diff() +
                       l['l']*cos(l['dc'])*l['dc'].diff().diff() -
                       l['l']*sin(l['dc'])*l['dc'].diff()**2 -
                       l['l']*sin(l['dc'])*l['fd'].diff()**2)*l['frame_d'].z)
    assert (a_e_f - expected_a_e_f).simplify() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\test_util.py ===
from sympy.core.containers import Tuple
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.printing.str import sstr
from sympy.physics.units import (
    G, centimeter, coulomb, day, degree, gram, hbar, hour, inch, joule, kelvin,
    kilogram, kilometer, length, meter, mile, minute, newton, planck,
    planck_length, planck_mass, planck_temperature, planck_time, radians,
    second, speed_of_light, steradian, time, km)
from sympy.physics.units.util import convert_to, check_dimensions
from sympy.testing.pytest import raises
from sympy.functions.elementary.miscellaneous import sqrt


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


L = length
T = time


def test_dim_simplify_add():
    # assert Add(L, L) == L
    assert L + L == L


def test_dim_simplify_mul():
    # assert Mul(L, T) == L*T
    assert L*T == L*T


def test_dim_simplify_pow():
    assert Pow(L, 2) == L**2


def test_dim_simplify_rec():
    # assert Mul(Add(L, L), T) == L*T
    assert (L + L) * T == L*T


def test_convert_to_quantities():
    assert convert_to(3, meter) == 3

    assert convert_to(mile, kilometer) == 25146*kilometer/15625
    assert convert_to(meter/second, speed_of_light) == speed_of_light/299792458
    assert convert_to(299792458*meter/second, speed_of_light) == speed_of_light
    assert convert_to(2*299792458*meter/second, speed_of_light) == 2*speed_of_light
    assert convert_to(speed_of_light, meter/second) == 299792458*meter/second
    assert convert_to(2*speed_of_light, meter/second) == 599584916*meter/second
    assert convert_to(day, second) == 86400*second
    assert convert_to(2*hour, minute) == 120*minute
    assert convert_to(mile, meter) == 201168*meter/125
    assert convert_to(mile/hour, kilometer/hour) == 25146*kilometer/(15625*hour)
    assert convert_to(3*newton, meter/second) == 3*newton
    assert convert_to(3*newton, kilogram*meter/second**2) == 3*meter*kilogram/second**2
    assert convert_to(kilometer + mile, meter) == 326168*meter/125
    assert convert_to(2*kilometer + 3*mile, meter) == 853504*meter/125
    assert convert_to(inch**2, meter**2) == 16129*meter**2/25000000
    assert convert_to(3*inch**2, meter) == 48387*meter**2/25000000
    assert convert_to(2*kilometer/hour + 3*mile/hour, meter/second) == 53344*meter/(28125*second)
    assert convert_to(2*kilometer/hour + 3*mile/hour, centimeter/second) == 213376*centimeter/(1125*second)
    assert convert_to(kilometer * (mile + kilometer), meter) == 2609344 * meter ** 2

    assert convert_to(steradian, coulomb) == steradian
    assert convert_to(radians, degree) == 180*degree/pi
    assert convert_to(radians, [meter, degree]) == 180*degree/pi
    assert convert_to(pi*radians, degree) == 180*degree
    assert convert_to(pi, degree) == 180*degree

    # https://github.com/sympy/sympy/issues/26263
    assert convert_to(sqrt(meter**2 + meter**2.0), meter) == sqrt(meter**2 + meter**2.0)
    assert convert_to((meter**2 + meter**2.0)**2, meter) == (meter**2 + meter**2.0)**2


def test_convert_to_tuples_of_quantities():
    from sympy.core.symbol import symbols

    alpha, beta = symbols('alpha beta')

    assert convert_to(speed_of_light, [meter, second]) == 299792458 * meter / second
    assert convert_to(speed_of_light, (meter, second)) == 299792458 * meter / second
    assert convert_to(speed_of_light, Tuple(meter, second)) == 299792458 * meter / second
    assert convert_to(joule, [meter, kilogram, second]) == kilogram*meter**2/second**2
    assert convert_to(joule, [centimeter, gram, second]) == 10000000*centimeter**2*gram/second**2
    assert convert_to(299792458*meter/second, [speed_of_light]) == speed_of_light
    assert convert_to(speed_of_light / 2, [meter, second, kilogram]) == meter/second*299792458 / 2
    # This doesn't make physically sense, but let's keep it as a conversion test:
    assert convert_to(2 * speed_of_light, [meter, second, kilogram]) == 2 * 299792458 * meter / second
    assert convert_to(G, [G, speed_of_light, planck]) == 1.0*G

    assert NS(convert_to(meter, [G, speed_of_light, hbar]), n=7) == '6.187142e+34*gravitational_constant**0.5000000*hbar**0.5000000/speed_of_light**1.500000'
    assert NS(convert_to(planck_mass, kilogram), n=7) == '2.176434e-8*kilogram'
    assert NS(convert_to(planck_length, meter), n=7) == '1.616255e-35*meter'
    assert NS(convert_to(planck_time, second), n=6) == '5.39125e-44*second'
    assert NS(convert_to(planck_temperature, kelvin), n=7) == '1.416784e+32*kelvin'
    assert NS(convert_to(convert_to(meter, [G, speed_of_light, planck]), meter), n=10) == '1.000000000*meter'

    # similar to https://github.com/sympy/sympy/issues/26263
    assert convert_to(sqrt(meter**2 + second**2.0), [meter, second]) == sqrt(meter**2 + second**2.0)
    assert convert_to((meter**2 + second**2.0)**2, [meter, second]) == (meter**2 + second**2.0)**2

    # similar to https://github.com/sympy/sympy/issues/21463
    assert convert_to(1/(beta*meter + meter), 1/meter) == 1/(beta*meter + meter)
    assert convert_to(1/(beta*meter + alpha*meter), 1/kilometer) == (1/(kilometer*beta/1000 + alpha*kilometer/1000))

def test_eval_simplify():
    from sympy.physics.units import cm, mm, km, m, K, kilo
    from sympy.core.symbol import symbols

    x, y = symbols('x y')

    assert (cm/mm).simplify() == 10
    assert (km/m).simplify() == 1000
    assert (km/cm).simplify() == 100000
    assert (10*x*K*km**2/m/cm).simplify() == 1000000000*x*kelvin
    assert (cm/km/m).simplify() == 1/(10000000*centimeter)

    assert (3*kilo*meter).simplify() == 3000*meter
    assert (4*kilo*meter/(2*kilometer)).simplify() == 2
    assert (4*kilometer**2/(kilo*meter)**2).simplify() == 4


def test_quantity_simplify():
    from sympy.physics.units.util import quantity_simplify
    from sympy.physics.units import kilo, foot
    from sympy.core.symbol import symbols

    x, y = symbols('x y')

    assert quantity_simplify(x*(8*kilo*newton*meter + y)) == x*(8000*meter*newton + y)
    assert quantity_simplify(foot*inch*(foot + inch)) == foot**2*(foot + foot/12)/12
    assert quantity_simplify(foot*inch*(foot*foot + inch*(foot + inch))) == foot**2*(foot**2 + foot/12*(foot + foot/12))/12
    assert quantity_simplify(2**(foot/inch*kilo/1000)*inch) == 4096*foot/12
    assert quantity_simplify(foot**2*inch + inch**2*foot) == 13*foot**3/144

def test_quantity_simplify_across_dimensions():
    from sympy.physics.units.util import quantity_simplify
    from sympy.physics.units import ampere, ohm, volt, joule, pascal, farad, second, watt, siemens, henry, tesla, weber, hour, newton

    assert quantity_simplify(ampere*ohm, across_dimensions=True, unit_system="SI") == volt
    assert quantity_simplify(6*ampere*ohm, across_dimensions=True, unit_system="SI") == 6*volt
    assert quantity_simplify(volt/ampere, across_dimensions=True, unit_system="SI") == ohm
    assert quantity_simplify(volt/ohm, across_dimensions=True, unit_system="SI") == ampere
    assert quantity_simplify(joule/meter**3, across_dimensions=True, unit_system="SI") == pascal
    assert quantity_simplify(farad*ohm, across_dimensions=True, unit_system="SI") == second
    assert quantity_simplify(joule/second, across_dimensions=True, unit_system="SI") == watt
    assert quantity_simplify(meter**3/second, across_dimensions=True, unit_system="SI") == meter**3/second
    assert quantity_simplify(joule/second, across_dimensions=True, unit_system="SI") == watt

    assert quantity_simplify(joule/coulomb, across_dimensions=True, unit_system="SI") == volt
    assert quantity_simplify(volt/ampere, across_dimensions=True, unit_system="SI") == ohm
    assert quantity_simplify(ampere/volt, across_dimensions=True, unit_system="SI") == siemens
    assert quantity_simplify(coulomb/volt, across_dimensions=True, unit_system="SI") == farad
    assert quantity_simplify(volt*second/ampere, across_dimensions=True, unit_system="SI") == henry
    assert quantity_simplify(volt*second/meter**2, across_dimensions=True, unit_system="SI") == tesla
    assert quantity_simplify(joule/ampere, across_dimensions=True, unit_system="SI") == weber

    assert quantity_simplify(5*kilometer/hour, across_dimensions=True, unit_system="SI") == 25*meter/(18*second)
    assert quantity_simplify(5*kilogram*meter/second**2, across_dimensions=True, unit_system="SI") == 5*newton

def test_check_dimensions():
    x = symbols('x')
    assert check_dimensions(inch + x) == inch + x
    assert check_dimensions(length + x) == length + x
    # after subs we get 2*length; check will clear the constant
    assert check_dimensions((length + x).subs(x, length)) == length
    assert check_dimensions(newton*meter + joule) == joule + meter*newton
    raises(ValueError, lambda: check_dimensions(inch + 1))
    raises(ValueError, lambda: check_dimensions(length + 1))
    raises(ValueError, lambda: check_dimensions(length + time))
    raises(ValueError, lambda: check_dimensions(meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + 3 * second))
    raises(ValueError, lambda: check_dimensions(1 / second + 1 / meter))
    raises(ValueError, lambda: check_dimensions(2 * meter*(mile + centimeter) + km))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\ranges\test_join.py ===
import numpy as np

from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm


class TestJoin:
    def test_join_outer(self):
        # join with Index[int64]
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        res, lidx, ridx = index.join(other, how="outer", return_indexers=True)
        noidx_res = index.join(other, how="outer")
        tm.assert_index_equal(res, noidx_res)

        eres = Index(
            [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        )
        elidx = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, -1, 8, -1, 9, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.intp,
        )
        eridx = np.array(
            [-1, -1, -1, -1, -1, -1, -1, -1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            dtype=np.intp,
        )

        assert isinstance(res, Index) and res.dtype == np.dtype(np.int64)
        assert not isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres, exact=True)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # join with RangeIndex
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = index.join(other, how="outer", return_indexers=True)
        noidx_res = index.join(other, how="outer")
        tm.assert_index_equal(res, noidx_res)

        assert isinstance(res, Index) and res.dtype == np.int64
        assert not isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_inner(self):
        # Join with non-RangeIndex
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        res, lidx, ridx = index.join(other, how="inner", return_indexers=True)

        # no guarantee of sortedness, so sort for comparison purposes
        ind = res.argsort()
        res = res.take(ind)
        lidx = lidx.take(ind)
        ridx = ridx.take(ind)

        eres = Index([16, 18])
        elidx = np.array([8, 9], dtype=np.intp)
        eridx = np.array([9, 7], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # Join two RangeIndex
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = index.join(other, how="inner", return_indexers=True)

        assert isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres, exact="equiv")
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_left(self):
        # Join with Index[int64]
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        res, lidx, ridx = index.join(other, how="left", return_indexers=True)
        eres = index
        eridx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 9, 7], dtype=np.intp)

        assert isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

        # Join withRangeIndex
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        res, lidx, ridx = index.join(other, how="left", return_indexers=True)

        assert isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_right(self):
        # Join with Index[int64]
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index(np.arange(25, 14, -1, dtype=np.int64))

        res, lidx, ridx = index.join(other, how="right", return_indexers=True)
        eres = other
        elidx = np.array([-1, -1, -1, -1, -1, -1, -1, 9, -1, 8, -1], dtype=np.intp)

        assert isinstance(other, Index) and other.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None

        # Join withRangeIndex
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = index.join(other, how="right", return_indexers=True)
        eres = other

        assert isinstance(other, RangeIndex)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None

    def test_join_non_int_index(self):
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index([3, 6, 7, 8, 10], dtype=object)

        outer = index.join(other, how="outer")
        outer2 = other.join(index, how="outer")
        expected = Index([0, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18])
        tm.assert_index_equal(outer, outer2)
        tm.assert_index_equal(outer, expected)

        inner = index.join(other, how="inner")
        inner2 = other.join(index, how="inner")
        expected = Index([6, 8, 10])
        tm.assert_index_equal(inner, inner2)
        tm.assert_index_equal(inner, expected)

        left = index.join(other, how="left")
        tm.assert_index_equal(left, index.astype(object))

        left2 = other.join(index, how="left")
        tm.assert_index_equal(left2, other)

        right = index.join(other, how="right")
        tm.assert_index_equal(right, other)

        right2 = other.join(index, how="right")
        tm.assert_index_equal(right2, index.astype(object))

    def test_join_non_unique(self):
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index([4, 4, 3, 3])

        res, lidx, ridx = index.join(other, return_indexers=True)

        eres = Index([0, 2, 4, 4, 6, 8, 10, 12, 14, 16, 18])
        elidx = np.array([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp)
        eridx = np.array([-1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1], dtype=np.intp)

        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_self(self, join_type):
        index = RangeIndex(start=0, stop=20, step=2)
        joined = index.join(index, how=join_type)
        assert index is joined

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_limitseq.py ===
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.combinatorial.numbers import (fibonacci, harmonic)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.series.limitseq import limit_seq
from sympy.series.limitseq import difference_delta as dd
from sympy.testing.pytest import raises, XFAIL
from sympy.calculus.accumulationbounds import AccumulationBounds

n, m, k = symbols('n m k', integer=True)


def test_difference_delta():
    e = n*(n + 1)
    e2 = e * k

    assert dd(e) == 2*n + 2
    assert dd(e2, n, 2) == k*(4*n + 6)

    raises(ValueError, lambda: dd(e2))
    raises(ValueError, lambda: dd(e2, n, oo))


def test_difference_delta__Sum():
    e = Sum(1/k, (k, 1, n))
    assert dd(e, n) == 1/(n + 1)
    assert dd(e, n, 5) == Add(*[1/(i + n + 1) for i in range(5)])

    e = Sum(1/k, (k, 1, 3*n))
    assert dd(e, n) == Add(*[1/(i + 3*n + 1) for i in range(3)])

    e = n * Sum(1/k, (k, 1, n))
    assert dd(e, n) == 1 + Sum(1/k, (k, 1, n))

    e = Sum(1/k, (k, 1, n), (m, 1, n))
    assert dd(e, n) == harmonic(n)


def test_difference_delta__Add():
    e = n + n*(n + 1)
    assert dd(e, n) == 2*n + 3
    assert dd(e, n, 2) == 4*n + 8

    e = n + Sum(1/k, (k, 1, n))
    assert dd(e, n) == 1 + 1/(n + 1)
    assert dd(e, n, 5) == 5 + Add(*[1/(i + n + 1) for i in range(5)])


def test_difference_delta__Pow():
    e = 4**n
    assert dd(e, n) == 3*4**n
    assert dd(e, n, 2) == 15*4**n

    e = 4**(2*n)
    assert dd(e, n) == 15*4**(2*n)
    assert dd(e, n, 2) == 255*4**(2*n)

    e = n**4
    assert dd(e, n) == (n + 1)**4 - n**4

    e = n**n
    assert dd(e, n) == (n + 1)**(n + 1) - n**n


def test_limit_seq():
    e = binomial(2*n, n) / Sum(binomial(2*k, k), (k, 1, n))
    assert limit_seq(e) == S(3) / 4
    assert limit_seq(e, m) == e

    e = (5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5)
    assert limit_seq(e, n) == S(5) / 3

    e = (harmonic(n) * Sum(harmonic(k), (k, 1, n))) / (n * harmonic(2*n)**2)
    assert limit_seq(e, n) == 1

    e = Sum(k**2 * Sum(2**m/m, (m, 1, k)), (k, 1, n)) / (2**n*n)
    assert limit_seq(e, n) == 4

    e = (Sum(binomial(3*k, k) * binomial(5*k, k), (k, 1, n)) /
         (binomial(3*n, n) * binomial(5*n, n)))
    assert limit_seq(e, n) == S(84375) / 83351

    e = Sum(harmonic(k)**2/k, (k, 1, 2*n)) / harmonic(n)**3
    assert limit_seq(e, n) == S.One / 3

    raises(ValueError, lambda: limit_seq(e * m))


def test_alternating_sign():
    assert limit_seq((-1)**n/n**2, n) == 0
    assert limit_seq((-2)**(n+1)/(n + 3**n), n) == 0
    assert limit_seq((2*n + (-1)**n)/(n + 1), n) == 2
    assert limit_seq(sin(pi*n), n) == 0
    assert limit_seq(cos(2*pi*n), n) == 1
    assert limit_seq((S.NegativeOne/5)**n, n) == 0
    assert limit_seq((Rational(-1, 5))**n, n) == 0
    assert limit_seq((I/3)**n, n) == 0
    assert limit_seq(sqrt(n)*(I/2)**n, n) == 0
    assert limit_seq(n**7*(I/3)**n, n) == 0
    assert limit_seq(n/(n + 1) + (I/2)**n, n) == 1


def test_accum_bounds():
    assert limit_seq((-1)**n, n) == AccumulationBounds(-1, 1)
    assert limit_seq(cos(pi*n), n) == AccumulationBounds(-1, 1)
    assert limit_seq(sin(pi*n/2)**2, n) == AccumulationBounds(0, 1)
    assert limit_seq(2*(-3)**n/(n + 3**n), n) == AccumulationBounds(-2, 2)
    assert limit_seq(3*n/(n + 1) + 2*(-1)**n, n) == AccumulationBounds(1, 5)


def test_limitseq_sum():
    from sympy.abc import x, y, z
    assert limit_seq(Sum(1/x, (x, 1, y)) - log(y), y) == S.EulerGamma
    assert limit_seq(Sum(1/x, (x, 1, y)) - 1/y, y) is S.Infinity
    assert (limit_seq(binomial(2*x, x) / Sum(binomial(2*y, y), (y, 1, x)), x) ==
            S(3) / 4)
    assert (limit_seq(Sum(y**2 * Sum(2**z/z, (z, 1, y)), (y, 1, x)) /
                  (2**x*x), x) == 4)


def test_issue_9308():
    assert limit_seq(subfactorial(n)/factorial(n), n) == exp(-1)


def test_issue_10382():
    n = Symbol('n', integer=True)
    assert limit_seq(fibonacci(n+1)/fibonacci(n), n).together() == S.GoldenRatio


def test_issue_11672():
    assert limit_seq(Rational(-1, 2)**n, n) == 0


def test_issue_14196():
    k, n  = symbols('k, n', positive=True)
    m = Symbol('m')
    assert limit_seq(Sum(m**k, (m, 1, n)).doit()/(n**(k + 1)), n) == 1/(k + 1)


def test_issue_16735():
    assert limit_seq(5**n/factorial(n), n) == 0


def test_issue_19868():
    assert limit_seq(1/gamma(n + S.One/2), n) == 0


@XFAIL
def test_limit_seq_fail():
    # improve Summation algorithm or add ad-hoc criteria
    e = (harmonic(n)**3 * Sum(1/harmonic(k), (k, 1, n)) /
         (n * Sum(harmonic(k)/k, (k, 1, n))))
    assert limit_seq(e, n) == 2

    # No unique dominant term
    e = (Sum(2**k * binomial(2*k, k) / k**2, (k, 1, n)) /
         (Sum(2**k/k*2, (k, 1, n)) * Sum(binomial(2*k, k), (k, 1, n))))
    assert limit_seq(e, n) == S(3) / 7

    # Simplifications of summations needs to be improved.
    e = n**3*Sum(2**k/k**2, (k, 1, n))**2 / (2**n * Sum(2**k/k, (k, 1, n)))
    assert limit_seq(e, n) == 2

    e = (harmonic(n) * Sum(2**k/k, (k, 1, n)) /
         (n * Sum(2**k*harmonic(k)/k**2, (k, 1, n))))
    assert limit_seq(e, n) == 1

    e = (Sum(2**k*factorial(k) / k**2, (k, 1, 2*n)) /
         (Sum(4**k/k**2, (k, 1, n)) * Sum(factorial(k), (k, 1, 2*n))))
    assert limit_seq(e, n) == S(3) / 16

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_pythonmpq.py ===
"""
test_pythonmpq.py

Test the PythonMPQ class for consistency with gmpy2's mpq type. If gmpy2 is
installed run the same tests for both.
"""
from fractions import Fraction
from decimal import Decimal
import pickle
from typing import Callable, List, Tuple, Type

from sympy.testing.pytest import raises

from sympy.external.pythonmpq import PythonMPQ

#
# If gmpy2 is installed then run the tests for both mpq and PythonMPQ.
# That should ensure consistency between the implementation here and mpq.
#
rational_types: List[Tuple[Callable, Type, Callable, Type]]
rational_types = [(PythonMPQ, PythonMPQ, int, int)]
try:
    from gmpy2 import mpq, mpz
    rational_types.append((mpq, type(mpq(1)), mpz, type(mpz(1))))
except ImportError:
    pass


def test_PythonMPQ():
    #
    # Test PythonMPQ and also mpq if gmpy/gmpy2 is installed.
    #
    for Q, TQ, Z, TZ in rational_types:

        def check_Q(q):
            assert isinstance(q, TQ)
            assert isinstance(q.numerator, TZ)
            assert isinstance(q.denominator, TZ)
            return q.numerator, q.denominator

        # Check construction from different types
        assert check_Q(Q(3)) == (3, 1)
        assert check_Q(Q(3, 5)) == (3, 5)
        assert check_Q(Q(Q(3, 5))) == (3, 5)
        assert check_Q(Q(0.5)) == (1, 2)
        assert check_Q(Q('0.5')) == (1, 2)
        assert check_Q(Q(Fraction(3, 5))) == (3, 5)

        # https://github.com/aleaxit/gmpy/issues/327
        if Q is PythonMPQ:
            assert check_Q(Q(Decimal('0.6'))) == (3, 5)

        # Invalid types
        raises(TypeError, lambda: Q([]))
        raises(TypeError, lambda: Q([], []))

        # Check normalisation of signs
        assert check_Q(Q(2, 3)) == (2, 3)
        assert check_Q(Q(-2, 3)) == (-2, 3)
        assert check_Q(Q(2, -3)) == (-2, 3)
        assert check_Q(Q(-2, -3)) == (2, 3)

        # Check gcd calculation
        assert check_Q(Q(12, 8)) == (3, 2)

        # __int__/__float__
        assert int(Q(5, 3)) == 1
        assert int(Q(-5, 3)) == -1
        assert float(Q(5, 2)) == 2.5
        assert float(Q(-5, 2)) == -2.5

        # __str__/__repr__
        assert str(Q(2, 1)) == "2"
        assert str(Q(1, 2)) == "1/2"
        if Q is PythonMPQ:
            assert repr(Q(2, 1)) == "MPQ(2,1)"
            assert repr(Q(1, 2)) == "MPQ(1,2)"
        else:
            assert repr(Q(2, 1)) == "mpq(2,1)"
            assert repr(Q(1, 2)) == "mpq(1,2)"

        # __bool__
        assert bool(Q(1, 2)) is True
        assert bool(Q(0)) is False

        # __eq__/__ne__
        assert (Q(2, 3) == Q(2, 3)) is True
        assert (Q(2, 3) == Q(2, 5)) is False
        assert (Q(2, 3) != Q(2, 3)) is False
        assert (Q(2, 3) != Q(2, 5)) is True

        # __hash__
        assert hash(Q(3, 5)) == hash(Fraction(3, 5))

        # __reduce__
        q = Q(2, 3)
        assert pickle.loads(pickle.dumps(q)) == q

        # __ge__/__gt__/__le__/__lt__
        assert (Q(1, 3) < Q(2, 3)) is True
        assert (Q(2, 3) < Q(2, 3)) is False
        assert (Q(2, 3) < Q(1, 3)) is False
        assert (Q(-2, 3) < Q(1, 3)) is True
        assert (Q(1, 3) < Q(-2, 3)) is False

        assert (Q(1, 3) <= Q(2, 3)) is True
        assert (Q(2, 3) <= Q(2, 3)) is True
        assert (Q(2, 3) <= Q(1, 3)) is False
        assert (Q(-2, 3) <= Q(1, 3)) is True
        assert (Q(1, 3) <= Q(-2, 3)) is False

        assert (Q(1, 3) > Q(2, 3)) is False
        assert (Q(2, 3) > Q(2, 3)) is False
        assert (Q(2, 3) > Q(1, 3)) is True
        assert (Q(-2, 3) > Q(1, 3)) is False
        assert (Q(1, 3) > Q(-2, 3)) is True

        assert (Q(1, 3) >= Q(2, 3)) is False
        assert (Q(2, 3) >= Q(2, 3)) is True
        assert (Q(2, 3) >= Q(1, 3)) is True
        assert (Q(-2, 3) >= Q(1, 3)) is False
        assert (Q(1, 3) >= Q(-2, 3)) is True

        # __abs__/__pos__/__neg__
        assert abs(Q(2, 3)) == abs(Q(-2, 3)) == Q(2, 3)
        assert +Q(2, 3) == Q(2, 3)
        assert -Q(2, 3) == Q(-2, 3)

        # __add__/__radd__
        assert Q(2, 3) + Q(5, 7) == Q(29, 21)
        assert Q(2, 3) + 1 == Q(5, 3)
        assert 1 + Q(2, 3) == Q(5, 3)
        raises(TypeError, lambda: [] + Q(1))
        raises(TypeError, lambda: Q(1) + [])

        # __sub__/__rsub__
        assert Q(2, 3) - Q(5, 7) == Q(-1, 21)
        assert Q(2, 3) - 1 == Q(-1, 3)
        assert 1 - Q(2, 3) == Q(1, 3)
        raises(TypeError, lambda: [] - Q(1))
        raises(TypeError, lambda: Q(1) - [])

        # __mul__/__rmul__
        assert Q(2, 3) * Q(5, 7) == Q(10, 21)
        assert Q(2, 3) * 1 == Q(2, 3)
        assert 1 * Q(2, 3) == Q(2, 3)
        raises(TypeError, lambda: [] * Q(1))
        raises(TypeError, lambda: Q(1) * [])

        # __pow__/__rpow__
        assert Q(2, 3) ** 2 == Q(4, 9)
        assert Q(2, 3) ** 1 == Q(2, 3)
        assert Q(-2, 3) ** 2 == Q(4, 9)
        assert Q(-2, 3) ** -1 == Q(-3, 2)
        if Q is PythonMPQ:
            raises(TypeError, lambda: 1 ** Q(2, 3))
            raises(TypeError, lambda: Q(1, 4) ** Q(1, 2))
        raises(TypeError, lambda: [] ** Q(1))
        raises(TypeError, lambda: Q(1) ** [])

        # __div__/__rdiv__
        assert Q(2, 3) / Q(5, 7) == Q(14, 15)
        assert Q(2, 3) / 1 == Q(2, 3)
        assert 1 / Q(2, 3) == Q(3, 2)
        raises(TypeError, lambda: [] / Q(1))
        raises(TypeError, lambda: Q(1) / [])
        raises(ZeroDivisionError, lambda: Q(1, 2) / Q(0))

        # __divmod__
        if Q is PythonMPQ:
            raises(TypeError, lambda: Q(2, 3) // Q(1, 3))
            raises(TypeError, lambda: Q(2, 3) % Q(1, 3))
            raises(TypeError, lambda: 1 // Q(1, 3))
            raises(TypeError, lambda: 1 % Q(1, 3))
            raises(TypeError, lambda: Q(2, 3) // 1)
            raises(TypeError, lambda: Q(2, 3) % 1)